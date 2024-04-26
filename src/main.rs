
use std::io::Read;
use std::time::Instant;
use std::num::TryFromIntError;
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
    AddedToken, 
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
use tiktoken_rs::p50k_base;
pub mod s3;

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
    #[arg(long)]
    local_cell_dir: PathBuf,

    /// Output location (may be an s3 uri)
    #[arg(long)]
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

    /// How many local cells we have 
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

fn vecu32_to_vecu16(vec_u32: &Vec<u32>) -> Result<Vec<u16>, TryFromIntError> {
    // Casts a vector of u32s to u16s, and if any element explodes, the whole fxn should err
    let x = vec_u32
        .into_iter()
        .map(|x| u16::try_from(*x))
        .collect::<Result<Vec<u16>, TryFromIntError>>();
    x
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
    let mut tokenizer = Tokenizer::from_pretrained(tokenizer_name, None).unwrap();
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
    Ok(tokenizer)
}



/*======================================================
=              Tokenize/semishuffle code               =
======================================================*/

fn process_input_file(input_file: &PathBuf, local_cell_mapper: &HashMap<usize, Arc<Mutex<BufWriter<File>>>>, 
                      seqlen: usize, tokenizer_name: String, num_local_cells: usize, hash_seed: usize,
                      use_u16: bool, pbar: Arc<Mutex<ProgressBar>>) -> Result<(), Error> {
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
    match tokenize_semishuffle_file(reader, local_cell_mapper, seqlen, tokenizer_name, num_local_cells, hash_seed, use_u16) {
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
                            hash_seed: usize, use_u16: bool) -> Result<(), Error> {
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
        let mut tokens = encoded.get_ids().to_vec();
        tokens.push(tokenizer.token_to_id("<EOT>").unwrap());
        all_tokens.extend(tokens);       
    }
    // Group tokens into contexts of length seqlen
    // And then figure out where each group should live and append it to that file
    let padding_token_id = tokenizer.token_to_id("<PAD>").unwrap();
    for chunk in all_tokens.chunks(seqlen) {
        let mut context = chunk.to_vec();
        if context.len() < seqlen {
            let padding_size = seqlen - context.len();
            context.extend(vec![padding_token_id; padding_size]);
        }
        let local_cell_fid = (hash_vec(&context, hash_seed) % num_local_cells as u64) as usize;
        let mut writer = local_cell_mapper.get(&local_cell_fid).unwrap().lock().unwrap();

        if use_u16 {
            let context = vecu32_to_vecu16(&context).unwrap();
            serialize_into(&mut *writer, &context).unwrap();
        }
        else {
            serialize_into(&mut *writer, &context).unwrap();
        }
    }
    Ok(())
}



fn count_tokens(input_file: &PathBuf, tokenizer_name: String, token_count: &AtomicUsize) -> Result<(), Error> {
    // For a reader, will tokenize each line, build contexts, and put each context into the appropriate local cell
    // Load tokenizer

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


    let tokenizer = load_tokenizer(&tokenizer_name).unwrap();

    // Tokenize all lines in the file
    let mut all_tokens = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let json: Value = serde_json::from_str(&line)?;
        let text = json["text"].as_str().unwrap();

        let encoded = tokenizer.encode(text, false).unwrap();
        let mut tokens = encoded.get_ids().to_vec();
        tokens.push(tokenizer.token_to_id("<EOT>").unwrap());
        all_tokens.extend(tokens);       
    }

    token_count.fetch_add(all_tokens.len(), Ordering::SeqCst);
    Ok(())
}


fn count_tokens_tiktoken(input_file: &PathBuf, tokenizer_name: String, token_counter: &AtomicUsize) -> Result<(), Error> {
    // For a reader, will tokenize each line, build contexts, and put each context into the appropriate local cell
    // Load tokenizer
    //let tokenizer = load_tokenizer(&tokenizer_name).unwrap();

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



    let bpe = p50k_base().unwrap();

    // Tokenize all lines in the file
    let mut all_tokens = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let json: Value = serde_json::from_str(&line)?;
        let text = json["text"].as_str().unwrap();

        //let encoded = tokenizer.encode(text, false).unwrap();
        //let mut tokens = encoded.get_ids().to_vec();
        let tokens = bpe.encode_with_special_tokens(text);
        //tokens.push(tokenizer.token_to_id("<EOT>").unwrap());
        all_tokens.extend(tokens);       
    }

    token_counter.fetch_add(all_tokens.len(), Ordering::SeqCst);
    Ok(())
}



/*======================================================
=                 Final shuffle code                   =
======================================================*/

fn read_serialized_file<T>(filename: &PathBuf) -> Result<Vec<Vec<u32>>, Box<dyn std::error::Error>>
where
    T: DeserializeOwned + TryFrom<u16>,
    <T as TryFrom<u16>>::Error: std::error::Error + 'static, u32: From<T>, u32: From<T>
{
    let file = File::open(filename)?;
    let mut reader = BufReader::new(file);
    let mut output = Vec::new();

    while let Ok(element) = bincode::deserialize_from::<_, Vec<T>>(&mut reader) {
        let element_u32: Result<Vec<u32>, _> = element.into_iter().map(|x| u32::try_from(x)).collect();
        output.push(element_u32?);
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


fn finalize_chunk(chunk: &[Vec<u32>], output_dir: &PathBuf, 
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
                      use_u16:bool, hash_seed: usize, pbar: Arc<Mutex<ProgressBar>>) -> Result<()> {

    // Given a "local cell" which has a bunch of contexts in it, shuffles it and groups into chunks of wds_chunk_size
    // For complete chunks, finalizes these and pushes to output directory
    // For incomplete chunks, if overflow writer exists -> write incomplete chunks to overflow file
    // Also does some branching: if no overflow writer, then this is final step and we can write chunks in parallel
    
    let mut rng = thread_rng();
    let mut contexts = if use_u16 {
        read_serialized_file::<u16>(filename).unwrap()
    } else {
        read_serialized_file::<u32>(filename).unwrap()
    };

    println!("FILENAME {:?} HAS LEN {:?}", filename, contexts.len());
    contexts.shuffle(&mut rng);
    for chunk in contexts.chunks(wds_chunk_size) {
        if chunk.len() != wds_chunk_size && !overflow_writer.is_none() { // short chunk, send to one of the overflows
            //let mut writer = local_cell_mapper.get(&local_cell_fid).unwrap().lock().unwrap();
            let num_overflows = overflow_writer.as_ref().unwrap().len() as u64;
            let overflow_writer = overflow_writer.as_ref().unwrap();
            for context in chunk {
                let context_hash = hash_vec::<u32>(&context, hash_seed);
                let mut writer = overflow_writer.get(&((context_hash % num_overflows) as usize)).unwrap().lock().unwrap();
                if use_u16 {
                    let context = vecu32_to_vecu16(context).unwrap();
                    serialize_into(&mut *writer, &context).unwrap();
                }
                else {
                    serialize_into(&mut *writer, &context).unwrap();
                }

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
    //let mut local_cell_mapper = setup_local_cell_mapper(&args.local_cell_dir, args.num_local_cells);
    let mut input_files = expand_dirs(args.input).unwrap();
    input_files.truncate(16);

    let vocab_size = load_tokenizer(&args.tokenizer).unwrap().get_vocab_size(true);
    let use_u16 = vocab_size < 65536;



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
    let total_token_count = Arc::new(AtomicUsize::new(0));
    for input_file in input_files {
        let input_file = input_file.clone();
        let tokenizer_name = args.tokenizer.clone();
        let total_token_count = total_token_count.clone();
        let pbar = pbar.clone();
        threadpool.execute(move || {
            count_tokens_tiktoken(&input_file, tokenizer_name, &total_token_count).unwrap();

            pbar.lock().unwrap().inc(1);            
        });
    }
    threadpool.join();


    // Step 4: Finalize by finishing the overflow writer, and writing some stats
    println!("Finishing tokenize shuffle run!");
    println!("-------------------------------");
    println!("Ran in {:?} (s)", start_time.elapsed().as_secs());
    println!("Processed {:?} tokens", total_token_count.fetch_add(0, Ordering::SeqCst));
    println!("Tok/s/cpu: {:?}", (total_token_count.fetch_add(0, Ordering::SeqCst) as u64) / (threads as u64 * start_time.elapsed().as_secs()) as u64);

}
