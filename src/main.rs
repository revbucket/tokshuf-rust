
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
use bincode::{deserialize_from, serialize_into};
use rand::seq::SliceRandom;
use rand::thread_rng;

use flate2::write::GzEncoder;
use flate2::Compression;


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

    /// How many local cells we have 
    #[arg(long, default_value_t=128)]
    num_local_cells: usize,

    /// Seed to use for the hashing of documents
    #[arg(long, default_value_t=1234)]
    hash_seed: usize

}


pub(crate) fn expand_dirs(paths: Vec<PathBuf>) -> Result<Vec<PathBuf>> {
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
    local_cell_dir.clone().join(format!("local_cell_{:08}.u32", fid))
}


fn setup_local_cell_mapper(local_cell_dir: &PathBuf, num_local_cells:usize) ->  HashMap<usize, Arc<Mutex<BufWriter<File>>>>{
    // Creates num_local_cells files in the local_cell_dir and 
    if !local_cell_dir.exists() {
        fs::create_dir_all(local_cell_dir).unwrap()
    }

    let mut local_cell_mapper = HashMap::new();
    for fid in 0..num_local_cells {
        let filename = local_cell_id(local_cell_dir, fid);
        let writer = Arc::new(
            Mutex::new(
            BufWriter::new(
            OpenOptions::new()
            .append(true)
            .create(true)
            .open(filename)
            .unwrap()
        )));
        local_cell_mapper.insert(fid, writer);
    }
    local_cell_mapper
}
    

fn hash_vec(vec: Vec<u16>, seed: usize) -> u64 {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    vec.hash(&mut hasher);
    hasher.finish()
}

fn vecu32_to_vecu16(vec_u32: Vec<u32>) -> Result<Vec<u16>, TryFromIntError> {
    let x = vec_u32
        .into_iter()
        .map(|x| u16::try_from(x))
        .collect::<Result<Vec<_>, _>>();
    x

}

/*======================================================
=              Tokenize/semishuffle code               =
======================================================*/

fn process_input_file(input_file: &PathBuf, local_cell_mapper: &HashMap<usize, Arc<Mutex<BufWriter<File>>>>, 
                      seqlen: usize, tokenizer_name: String, num_local_cells: usize, hash_seed: usize,
                      pbar: Arc<Mutex<ProgressBar>>) -> Result<()> {
    // Gather file into reader 

    let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();   
    let reader = rt.block_on(get_reader_from_s3(input_file, Some(5))).unwrap();        


    let _result = tokenize_semishuffle_file(reader, local_cell_mapper, seqlen, tokenizer_name, num_local_cells, hash_seed);
    pbar.lock().unwrap().inc(0);
    Ok(())
}


fn tokenize_semishuffle_file(reader: BufReader<Cursor<Vec<u8>>>, local_cell_mapper: &HashMap<usize, Arc<Mutex<BufWriter<File>>>>, 
                            seqlen: usize, tokenizer_name: String, num_local_cells: usize, 
                            hash_seed: usize) -> Result<()> {

    // Load tokenizer
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
    // Convert all tokens to u16 


    // Group tokens into contexts of length seqlen
    // And then figure out where each group should live and append it to that file
    let padding_token_id = tokenizer.token_to_id("<PAD>").unwrap();
    for chunk in all_tokens.chunks(seqlen) {
        let mut context = chunk.to_vec();
        if context.len() < seqlen {
            let padding_size = seqlen - context.len();
            context.extend(vec![padding_token_id; padding_size]);
        }
        let context = vecu32_to_vecu16(context).unwrap();
        let local_cell_fid = (hash_vec(context.clone(), hash_seed) % num_local_cells as u64) as usize;
        let mut writer = local_cell_mapper.get(&local_cell_fid).unwrap().lock().unwrap();
        serialize_into(&mut *writer, &context).unwrap();
    }
    Ok(())
}


/*======================================================
=                 Final shuffle code                   =
======================================================*/

fn read_u32_file(filename: &PathBuf) -> Result<Vec<Vec<u32>>> {
    let file = File::open(filename).unwrap();
    let mut reader = BufReader::new(file);
    let mut output = Vec::new();

    while let Ok(element) = deserialize_from::<_, Vec<u32>>(&mut reader) {
        output.push(element)
    }

    Ok(output)
}

fn get_chunk_filename(output_dir: &PathBuf, chunk_id: usize) -> PathBuf {
    output_dir.join(format!("chunk_{:08}.tar", chunk_id))
}

fn write_chunk_to_file(chunk: &[Vec<u32>], chunk_filename: PathBuf, total_token_count: &AtomicUsize) -> Result<()> {
    // First collect chunk into tarfile and wrap in a bufWriter and then either send it to s3 or 

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

    if is_s3(&chunk_filename) {
        let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();   
        rt.block_on(write_cursor_to_s3(&chunk_filename.clone(), bio)).unwrap()
    } else {
        let mut file = File::create(chunk_filename).expect("Failed to create file");
        std::io::copy(&mut bio, &mut file).expect("Failed to write to file");
    }

    // Use the `bio` buffer as needed
    Ok(())
}


fn process_local_cell(filename: &PathBuf, overflow_writer: Option<Arc<Mutex<BufWriter<File>>>>, output_dir: &PathBuf, 
                      wds_chunk_size: usize, wds_chunk_id: &AtomicUsize, total_token_count: &AtomicUsize,
                      pbar: Arc<Mutex<ProgressBar>>) -> Result<()> {
    // open and read files into vec of vecs
    let mut rng = thread_rng();
    let mut contexts = read_u32_file(filename).unwrap();
    contexts.shuffle(&mut rng);
    for chunk in contexts.chunks(wds_chunk_size) {
        if chunk.len() == wds_chunk_size || overflow_writer.is_none() {
            let chunk_id = wds_chunk_id.fetch_add(1, Ordering::SeqCst);
            let chunk_filename = get_chunk_filename(output_dir, chunk_id);
            write_chunk_to_file(chunk, chunk_filename, total_token_count).unwrap();
        } else {
            match overflow_writer {
                Some(ref writer) => {
                    let mut writer = writer.lock().unwrap();
                    for context in chunk {
                        serialize_into(&mut *writer, &context).unwrap();                    
                    }
                },
                _ => {}
            };
        } 
    }
    fs::remove_file(filename).unwrap();
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

    println!("Starting fineSort/upload loop...");
    let pbar = ProgressBar::new(1 + args.num_local_cells as u64)
        .with_style(
            ProgressStyle::with_template(
                "Files {human_pos}/{human_len} [{elapsed_precise}/{duration_precise}] [{wide_bar:.cyan/blue}]",
            ).unwrap()
        );
    let pbar = Arc::new(Mutex::new(pbar));
    pbar.lock().unwrap().inc(0); // Makes pbar show up with 0/N files complete
    let threadpool = ThreadPool::new(threads);    
    let overflow_filename  = args.local_cell_dir.clone().join("overflow.u32");
    let wds_chunk_id = Arc::new(AtomicUsize::new(0));
    let total_token_count = Arc::new(AtomicUsize::new(0));
    let overflow_writer = Arc::new(
            Mutex::new(
            BufWriter::new(
            OpenOptions::new()
            .append(true)
            .create(true)
            .open(overflow_filename.clone())
            .unwrap()
        )));
    for fid in 0..args.num_local_cells {
        let filename = local_cell_id(&args.local_cell_dir, fid);
        let overflow = Arc::clone(&overflow_writer);
        let output_dir = args.output.clone();
        let wds_chunk_id = wds_chunk_id.clone();
        let total_token_count = total_token_count.clone();
        let pbar = pbar.clone();
        threadpool.execute(move || {
            process_local_cell(&filename, Some(overflow), &output_dir, args.wds_chunk_size, &wds_chunk_id, &total_token_count, pbar).unwrap()
        });
    }
    threadpool.join();
    overflow_writer.lock().unwrap().flush().unwrap();

    // Step 4: Finalize by finishing the overflow writer, and writing some stats
    process_local_cell(&overflow_filename, None, &args.output, args.wds_chunk_size, &wds_chunk_id, &total_token_count, pbar).unwrap();
    println!("Finishing tokenize shuffle run!");
    println!("-------------------------------");
    println!("Ran in {:?} (s)", start_time.elapsed().as_secs());
    println!("Processed {:?} tokens", total_token_count.fetch_add(0, Ordering::SeqCst));


}
