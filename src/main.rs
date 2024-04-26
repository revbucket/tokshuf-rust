
use std::io::Read;
use std::time::Instant;
use anyhow::{anyhow, bail, Result, Error};
use clap::Parser;
use std::path::PathBuf;
use std::convert::TryFrom;
use glob::glob;
use std::io::{BufReader, BufRead, BufWriter, Cursor, Write};
use std::fs::{OpenOptions, File};
use std::fs;
use std::thread::available_parallelism;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::hash::{Hash, Hasher, DefaultHasher};
use threadpool::ThreadPool;
use crate::s3::{is_s3, expand_s3_dir, get_reader_from_s3, write_cursor_to_s3};
use serde_json::Value;
use tar::Builder;
use serde_json;
use uuid::Uuid;
use indicatif::{ProgressBar,ProgressStyle};
use tokenizers::tokenizer::{
    Tokenizer
};
use bincode::{serialize_into};
use rand::seq::SliceRandom;
use rand::thread_rng;
use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use zstd::stream::read::Decoder as ZstdDecoder;
use serde::de::DeserializeOwned;

pub mod s3;

const EOT_TOKEN: i32 = 0;
const PAD_TOKEN: i32 = -1; 
/*======================================================
=                    Helpers/utilities                 =
======================================================*/
// Args 

#[derive(Parser, Debug)]
struct Args {
    /// (List of) directories/files (on s3 or local) that are jsonl.gz or jsonl.zstd files
    #[arg(required=true, long)]
    input: Vec<PathBuf>,

    /// Local directory (need not exist yet) where local cells will exist/grow
    #[arg(required=true, long)]
    local_cell_dir: PathBuf,

    /// Output location (may be an s3 uri)
    #[arg(required=true, long)]
    output: PathBuf,

    /// Which tokenizer we want to use by default 
    #[arg(long, default_value_t=String::from("EleutherAI/gpt-neox-20b"))]
    tokenizer: String,

    /// How long each context is (in tokens)
    #[arg(long, default_value_t=2049)]
    seqlen: usize,

    /// Size of the webdataset chunk size (in contexts)
    #[arg(long, default_value_t=8192)]
    wds_chunk_size: usize,

    /// How many threads to use (default is max available)
    #[arg(long, default_value_t=0)]
    threads: usize,

    /// How many local cells we have. 
    /// IMPORTANT: This should probably be 
    #[arg(long, default_value_t=128)]
    num_local_cells: usize,

    /// Seed to use for the hashing of documents
    #[arg(long, default_value_t=1234)]
    hash_seed: usize,

    // How many times to retry s3 operations
    #[arg(long, default_value_t=3)]
    s3_retries: usize,

}


pub(crate) fn expand_dirs(paths: Vec<PathBuf>) -> Result<Vec<PathBuf>> {
    // For local directories -> does a glob over each directory to get all json*.gz files
    // For s3 directories -> does an aws s3 ls to search for files
    let mut files: Vec<PathBuf> = Vec::new();
    let runtime = tokio::runtime::Runtime::new().unwrap();

    for path in paths {
        if is_s3(path.clone()) {
            // Use async_std to block until we scour the s3 directory for files
            runtime.block_on(async {
                let s3_paths = expand_s3_dir(&path).await.unwrap();
                files.extend(s3_paths);                
            });                
        }
        else if path.is_dir() {
            let path_str = path
                .to_str()
                .ok_or_else(|| anyhow!("invalid path '{}'", path.to_string_lossy()))?;
            let mut num_hits = 0;
            for entry in glob(&format!("{}/**/*.json*.gz", path_str))? {
                files.push(entry?.to_path_buf());
                num_hits += 1;
            }
            if num_hits == 0 {
                bail!("No JSON Gz files found in '{}'", path_str);
            }
        } else {
            files.push(path.clone());
        }
    }
    Ok(files)
}


fn local_cell_id(local_cell_dir: &PathBuf, fid: usize) -> PathBuf {
    // Standardized method to name "local cell" files
    local_cell_dir.clone().join(format!("local_cell_{:08}.ubin", fid))
}


fn setup_writers(filenames: Vec<&PathBuf>) -> HashMap<usize, Arc<Mutex<BufWriter<File>>>> {
    let mut mapper = HashMap::new();
    for (idx, filename) in filenames.into_iter().enumerate() {
        let writer = Arc::new(
            Mutex::new(
            BufWriter::new(
            OpenOptions::new()
            .append(true)
            .create(true)
            .open(filename)
            .unwrap()
        )));
        mapper.insert(idx, writer);
    }
    mapper
}

fn setup_local_cell_mapper(local_cell_dir: &PathBuf, num_local_cells:usize) ->  HashMap<usize, Arc<Mutex<BufWriter<File>>>>{
    // Creates num_local_cells files in the local_cell_dir and returns a map from id-> threadsafe writer
    if !local_cell_dir.exists() {
        fs::create_dir_all(local_cell_dir).unwrap()
    }
    let filenames: Vec<PathBuf> = (0..num_local_cells)
        .map(|fid| local_cell_id(local_cell_dir, fid))
        .collect();
    let filenames_ref = filenames.iter().collect::<Vec<_>>();
    setup_writers(filenames_ref)
}
    

fn hash_vec<T>(vec: &Vec<T>, seed: usize) -> u64 where T: Hash {
    // Hashes a vector of u16s into a u64 hash value
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    vec.hash(&mut hasher);
    hasher.finish()
}

fn cast_vector<T, U>(vec: Vec<T>) -> Result<Vec<U>, String>
where
    U: TryFrom<T>,
    <U as TryFrom<T>>::Error: std::fmt::Debug,
{
    vec.into_iter()
        .map(|item| {
            U::try_from(item).map_err(|e| format!("Cannot cast element: {:?}", e))
        })
        .collect()
}


fn read_file_into_memory(input_file: &PathBuf) ->Result<Cursor<Vec<u8>>, Error>{
    let mut file = File::open(input_file).expect("Failed to open file");

    let mut contents = Vec::new();
    let ext = input_file.extension().unwrap().to_string_lossy().to_lowercase();
    if ext == "gz" {
        // Gzip case        
        let mut decoder = MultiGzDecoder::new(file);
        decoder.read_to_end(&mut contents).expect("Failed to read loca gzip file");
    } else if ext == "zstd" || ext == "zst" {
        // Zstd case
        let mut decoder = ZstdDecoder::new(file).unwrap();
        decoder.read_to_end(&mut contents).expect("Failed to read local zstd file");
    } else {
        file.read_to_end(&mut contents).expect("Failed to read local file");

        // No compression case 
    }
    Ok(Cursor::new(contents))
}

fn load_tokenizer(tokenizer_name: &String) -> Result<Tokenizer> {
    let tokenizer = Tokenizer::from_pretrained(tokenizer_name, None).unwrap();
    /*
    tokenizer.add_special_tokens(&[
        AddedToken {
            content: String::from("<EOT>"),
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized:false,
            special: true,
        },
        AddedToken {
            content: String::from("<PAD>"),
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized:false,
            special: true
        },
    ]);
    */
    Ok(tokenizer)
}



/*======================================================
=              Tokenize/semishuffle code               =
======================================================*/

fn process_input_file(input_file: &PathBuf, local_cell_mapper: &HashMap<usize, Arc<Mutex<BufWriter<File>>>>, 
                      seqlen: usize, tokenizer_name: String, num_local_cells: usize, hash_seed: usize,
                      pbar: Arc<Mutex<ProgressBar>>) -> Result<(), Error> {
    // Gather file into reader 

    let reader = if is_s3(input_file) {
        let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();   
        match rt.block_on(get_reader_from_s3(input_file, Some(5))) {
            Ok(result) => result,
            Err(err) => {
                eprintln!("Error! {:?}", err);
                return Err(err.into());
            }
        }
    } else {
        let contents = read_file_into_memory(input_file).expect("Failed to read contents into memory");
        BufReader::new(contents)
    };
    match tokenize_semishuffle_file(reader, local_cell_mapper, seqlen, tokenizer_name, num_local_cells, hash_seed) {
        Ok(_) => {
            pbar.lock().unwrap().inc(1);
            return Ok(());
        }
        Err(err) => {
            eprintln!("Errored tok/shuffling {:?} | {:?}", input_file, err);
            return Err(err.into());
        }
    }
}


fn tokenize_semishuffle_file(reader: BufReader<Cursor<Vec<u8>>>, local_cell_mapper: &HashMap<usize, Arc<Mutex<BufWriter<File>>>>, 
                            seqlen: usize, tokenizer_name: String, num_local_cells: usize, 
                            hash_seed: usize) -> Result<(), Error> {
    // For a reader, will tokenize each line, build contexts, and put each context into the appropriate local cell
    // Load tokenizer
    let tokenizer = load_tokenizer(&tokenizer_name).unwrap();

    // Tokenize all lines in the file
    let mut all_tokens = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let json: Value = serde_json::from_str(&line)?;
        let text = json["text"].as_str().unwrap();

        let encoded = tokenizer.encode(text, false).unwrap();
        let mut tokens = cast_vector::<u32, i32>(encoded.get_ids().to_vec()).unwrap();
        tokens.push(EOT_TOKEN);
        //tokens.push(tokenizer.token_to_id("<EOT>").unwrap());
        all_tokens.extend(tokens);       
    }
    // Group tokens into contexts of length seqlen
    // And then figure out where each group should live and append it to that file
    for chunk in all_tokens.chunks(seqlen) {
        let mut context = chunk.to_vec();
        if context.len() < seqlen {
            let padding_size = seqlen - context.len();
            context.extend(vec![PAD_TOKEN; padding_size]);
        }
        let local_cell_fid = (hash_vec(&context, hash_seed) % num_local_cells as u64) as usize;
        let mut writer = local_cell_mapper.get(&local_cell_fid).unwrap().lock().unwrap();
        serialize_into(&mut *writer, &context).unwrap();        
    }

    Ok(())
}


/*======================================================
=                 Final shuffle code                   =
======================================================*/

fn read_serialized_file<T>(filename: &PathBuf) -> Result<Vec<Vec<i32>>, Box<dyn std::error::Error>>
where
    T: DeserializeOwned + TryFrom<i32>,
    <T as TryFrom<i32>>::Error: std::error::Error + 'static, i32: From<T>, i32: From<T>
{
    let file = File::open(filename)?;
    let mut reader = BufReader::new(file);
    let mut output = Vec::new();

    while let Ok(element) = bincode::deserialize_from::<_, Vec<T>>(&mut reader) {
        let element_i32: Result<Vec<i32>, _> = element.into_iter().map(|x| i32::try_from(x)).collect();
        output.push(element_i32?);
    }

    Ok(output)
}

fn get_chunk_filename(output_dir: &PathBuf, chunk_id: usize) -> PathBuf {
    // Standardized method to name each output chunk tarfile
    output_dir.join(format!("chunk_{:08}.tar", chunk_id))
}

fn get_overflow_filename(local_cell_dir: &PathBuf, overflow_round: usize, overflow_id: usize) -> PathBuf {
    local_cell_dir.join(format!("overflow_{:04}_{:08}.ubin", overflow_round, overflow_id))
}

fn build_overflow_writers(local_cell_dir: &PathBuf, overflow_round: usize, num_overflows: usize) -> 
    (Option<HashMap<usize, Arc<Mutex<BufWriter<File>>>>>, Vec<PathBuf>) {    
    let mut overflow_filenames: Vec<PathBuf> = Vec::new();
    if num_overflows == 0 {
        return (None, Vec::new());
    }

    for idx in 0..num_overflows {
        let overflow_filename = get_overflow_filename(local_cell_dir, overflow_round, idx);
        overflow_filenames.push(overflow_filename);
    }

    let overflow_filenames_ref = overflow_filenames.iter().collect::<Vec<_>>();
    (Some(setup_writers(overflow_filenames_ref)), overflow_filenames)
}


fn finalize_chunk(chunk: &[Vec<i32>], output_dir: &PathBuf, 
                  wds_chunk_id: &AtomicUsize, total_token_count: &AtomicUsize, 
                ) -> Result<()> {
    // Given a COMPLETE chunk, output directory, and atomic id/namer, and atomic token-counter
    // Wraps the chunk in a tarfile and saves it in the output dir

    // Computes the filename for the chunk
    let chunk_id = wds_chunk_id.fetch_add(1, Ordering::SeqCst);
    let chunk_filename = get_chunk_filename(output_dir, chunk_id);

    // And then wraps the chunk in a tarfile 
    let tokens_here = chunk.len() * chunk[0].len();
    total_token_count.fetch_add(tokens_here, Ordering::SeqCst);

    let mut bio = Cursor::new(Vec::new());
    {
        let mut builder = Builder::new(&mut bio);

        for context in chunk {
            let json_string = serde_json::to_string(&context).unwrap();
            let mut header = tar::Header::new_gnu();
            let mut uid = Uuid::new_v4().to_string();
            uid.push_str(".jsonl.gz");

            let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(json_string.as_bytes()).unwrap();
            let compressed_data = encoder.finish().unwrap();
            let compressed_data = compressed_data.as_slice();            
            header.set_size(compressed_data.len() as u64);
            header.set_cksum();
            builder.append_data(&mut header, uid, compressed_data).unwrap();        
        }
        builder.finish().unwrap();
    }
    bio.set_position(0);


    // And finally saves the chunk on disk/s3 
    if is_s3(&chunk_filename) {
        let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();   
        rt.block_on(write_cursor_to_s3(&chunk_filename.clone(), bio)).unwrap();
    } else {
        let mut file = File::create(chunk_filename).expect("Failed to create file");
        std::io::copy(&mut bio, &mut file).expect("Failed to write to file");
    }

    Ok(())   

}


fn process_local_cell(filename: &PathBuf, overflow_writer: &Option<HashMap<usize, Arc<Mutex<BufWriter<File>>>>>, output_dir: &PathBuf, 
                      wds_chunk_size: usize, wds_chunk_id: &AtomicUsize, total_token_count: &AtomicUsize,
                      hash_seed: usize, pbar: Arc<Mutex<ProgressBar>>) -> Result<()> {

    // Given a "local cell" which has a bunch of contexts in it, shuffles it and groups into chunks of wds_chunk_size
    // For complete chunks, finalizes these and pushes to output directory
    // For incomplete chunks, if overflow writer exists -> write incomplete chunks to overflow file
    // Also does some branching: if no overflow writer, then this is final step and we can write chunks in parallel
    
    let mut rng = thread_rng();
    let mut contexts = read_serialized_file::<i32>(filename).unwrap();
    

    println!("FILENAME {:?} HAS LEN {:?}", filename, contexts.len());
    contexts.shuffle(&mut rng);
    for chunk in contexts.chunks(wds_chunk_size) {
        if chunk.len() != wds_chunk_size && !overflow_writer.is_none() { // short chunk, send to one of the overflows
            //let mut writer = local_cell_mapper.get(&local_cell_fid).unwrap().lock().unwrap();
            let num_overflows = overflow_writer.as_ref().unwrap().len() as u64;
            let overflow_writer = overflow_writer.as_ref().unwrap();
            for context in chunk {
                let context_hash = hash_vec::<i32>(&context, hash_seed);
                let mut writer = overflow_writer.get(&((context_hash % num_overflows) as usize)).unwrap().lock().unwrap();
                serialize_into(&mut *writer, &context).unwrap();            
            }
        } else { // regular length, finalize chunk
            finalize_chunk(chunk, output_dir, wds_chunk_id, total_token_count).unwrap();                
        }
    }

    //fs::remove_file(filename).unwrap();
    pbar.lock().unwrap().inc(1);
    Ok(())
}

/*======================================================
=                     Main block                       =
======================================================*/


fn main() {

    // Step 1: Setup phase: parse args and set up the:
    //    - files to tokshuf
    //    - local cells
    //    - threadpool
    println!("Setting up Tok/Shuffle run");
    let start_time = Instant::now();    
    let args = Args::parse();
    let threads = if args.threads == 0 {
        available_parallelism().unwrap().get()
    } else {
        args.threads
    };
    let mut local_cell_mapper = setup_local_cell_mapper(&args.local_cell_dir, args.num_local_cells);
    let input_files = expand_dirs(args.input).unwrap();

    let pbar = ProgressBar::new(input_files.len() as u64)
        .with_style(
            ProgressStyle::with_template(
                "Files {human_pos}/{human_len} [{elapsed_precise}/{duration_precise}] [{wide_bar:.cyan/blue}]",
            ).unwrap()
        );
    let pbar = Arc::new(Mutex::new(pbar));
    pbar.lock().unwrap().inc(0); // Makes pbar show up with 0/N files complete
    let threadpool = ThreadPool::new(threads);


    // Step 2: Tokenize each document, make contexts, and put contexts into local cells
    println!("Starting tokenize/coarseSort loop...");
    for input_file in input_files {
        let input_file = input_file.clone();
        let tokenizer_name = args.tokenizer.clone();
        let local_cell_mapper = local_cell_mapper.clone();
        let pbar = pbar.clone();
        threadpool.execute(move || {
            process_input_file(&input_file, &local_cell_mapper, args.seqlen, tokenizer_name,
                               args.num_local_cells, args.hash_seed, pbar).unwrap()
        });
    }
    threadpool.join();
    for (_, writer) in local_cell_mapper.iter_mut() {
        writer.lock().unwrap().flush().unwrap();
    }


    // Step 3: For each local cell, group into outputs of wds chunk size
    // Create cascade of overflow writers 
    let overflow_reduction = 16;
    let mut total_calls = 0;
    let mut remaining_files = 128;
    while remaining_files  > 0 {
        total_calls += remaining_files;
        remaining_files = remaining_files / overflow_reduction;
    }

    println!("Starting fineSort/upload loop...");
    let pbar = ProgressBar::new(total_calls as u64)
        .with_style(
            ProgressStyle::with_template(
                "Files {human_pos}/{human_len} [{elapsed_precise}/{duration_precise}] [{wide_bar:.cyan/blue}]",
            ).unwrap()
        );
    let pbar = Arc::new(Mutex::new(pbar));
    pbar.lock().unwrap().inc(0); // Makes pbar show up with 0/N files complete



    let wds_chunk_id = Arc::new(AtomicUsize::new(0));
    let total_token_count = Arc::new(AtomicUsize::new(0));
    let mut overflow_round = 0;
    let mut num_overflows = args.num_local_cells / overflow_reduction;
    let mut src_filenames: Vec<PathBuf> = (0..args.num_local_cells)
        .map(|idx| local_cell_id(&args.local_cell_dir, idx)).collect();

    while src_filenames.len() > 0 {

        let threadpool = ThreadPool::new(threads);    
        let (overflow_writers, overflow_filenames) = build_overflow_writers(&args.local_cell_dir, overflow_round, num_overflows);
        println!("STARTING ROUND {:?} | {:?} SRC FILES | {:?} WRITERS",
                 overflow_round, src_filenames.len(), overflow_filenames.len());        
        let src_pointer = &src_filenames;
        for filename in src_pointer {     
            let filename = filename.clone();       
            let output_dir = args.output.clone();
            let wds_chunk_id = wds_chunk_id.clone();
            let total_token_count = total_token_count.clone();
            let overflow_writers = overflow_writers.clone();
            let pbar = pbar.clone();
            threadpool.execute(move || {
                process_local_cell(&filename, &overflow_writers, &output_dir, args.wds_chunk_size, &wds_chunk_id,
                                   &total_token_count, args.hash_seed, pbar).unwrap()
            });                        
        } 
        threadpool.join();
        overflow_round += 1;
        if num_overflows == 1 {
            num_overflows = 0;
        } else {
            num_overflows = num_overflows / overflow_reduction;
            if num_overflows == 0 {
                num_overflows = 1;
            }
        }
        src_filenames = overflow_filenames.clone();
    };
    


    // Step 4: Finalize by finishing the overflow writer, and writing some stats
    println!("Finishing tokenize shuffle run!");
    println!("-------------------------------");
    println!("Ran in {:?} (s)", start_time.elapsed().as_secs());
    println!("Processed {:?} tokens", total_token_count.fetch_add(0, Ordering::SeqCst));


}
