
use std::io::Read;
use std::time::Instant;
use anyhow::{anyhow, bail, Result, Error};
use clap::Parser;
use std::path::{PathBuf};
use std::convert::TryFrom;
use glob::glob;
use std::io::{BufReader, BufRead, BufWriter, Cursor, Write};
use std::fs::{OpenOptions, File};
use std::fs;


use std::thread::available_parallelism;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::hash::{Hash, Hasher, DefaultHasher};
use threadpool::ThreadPool;
use crate::s3::{is_s3, expand_s3_dir, get_reader_from_s3, write_cursor_to_s3};
use serde_json::{Value, json};
use tar::{Builder, Archive};
use serde_json;

use uuid::Uuid;
use indicatif::{ProgressBar,ProgressStyle};
use tokenizers::tokenizer::{
    Tokenizer
};



use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use zstd::stream::read::Decoder as ZstdDecoder;

use base64::{engine::general_purpose, Engine as _};

use rustc_hash::FxHashMap;
use rand::seq::SliceRandom;
use rand::thread_rng;

use tiktoken_rs::CoreBPE;
pub mod s3;


const OVERFLOW_REDUCTION: usize = 16; // maybe make this an arg?
const LLAMA3_BOS_TOKEN: i32 = 128000;
const EOT_TOKEN: i32 = 0;
const PAD_TOKEN: i32 = -1; 

type CellMap = HashMap<usize, Arc<Mutex<Builder<BufWriter<File>>>>>;

/*
Rough description of how this works:
Necessary args: {input, output, local_cell_dir}
    - input: s3 or local directories containing .jsonl.gz or .jsonl.zstd files to tok/shuffle
    - output: s3 or local directory where the output tars + manifest should go
    - local_cell_dir: local directory where temporary files get stored


Flow:
    1. Setup everything: threadpools, list of files, logging info

    2. For each file in the input will process (in parallel). To process each file:
        a. iterate over lines, tokenizing each line["text"] and adding an EOT token after each document;
           then concatenate all tokens into a big vector
        b. chunk the vector into contexts of len seqlen (default 2049), 
           padding out the final context with PAD_TOKENs until it is seqlen tokens long
        c. Each context gets hashed and APPENDED into a "local_cell", which serves as a rough sorting process
           ^Note: we refer to this process as "semishuffling"

    3. Once all files are processed and all "local_cell"s built, in a series of rounds we process each local cell
        a. To process a local cell, we load all contexts into memory and shuffle them. 
        b. Then we group into chunks of wds_chunk_size (default 8192) and upload each chunk to `output` 
        c. Any leftovers (%wd_chunk_size) get APPENDED into a smaller set of "overflows local cells"
        d. Once all "local cells" are processed, we proceed to the next round, 
           where the "overflow local cells" become the "local cells" and we repeat until no local cells remain
        e. The final chunk will probably have fewer than wds_chunk_size contexts in it
    4. Finishing up: build/save a manifest.jsonl, and print out all logs
*/

/*======================================================
=                              ARGS                    =
======================================================*/

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

    /// Which tokenizer we want to use by default // other option is "meta-llama/Meta-Llama-3-8B"
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

    /// How many times to retry s3 operations
    #[arg(long, default_value_t=3)]
    s3_retries: usize,

    /// If present, we use tiktoken to encode (only works with "EleutherAI/gpt-neox-20b"!)
    #[arg(long, default_value_t=false)]
    use_tiktoken: bool,

    /// Shuffle only, if set to true we should accept .tar files only that have
    /// contexts pre-built, and we just want to shuffle them 
    #[arg(long, default_value_t=false)]
    shuffle_only: bool,

    /// Extension for which files to look for:
    /// In shuffle only case this should be tar,
    /// In regular case this should be either jsonl.zstd or jsonl.gz
    #[arg(long)]
    ext: String,

}


/*=========================================================
=                        I/O Utilities                    =
=========================================================*/



pub(crate) fn expand_dirs(paths: Vec<PathBuf>, ext: Option<&str>) -> Result<Vec<PathBuf>> {
    // For local directories -> does a glob over each directory to get all files with given extension
    // For s3 directories -> does an aws s3 ls to search for files
    let ext = ext.unwrap_or(".jsonl.gz"); // Defaults to jsonl.gz, json.gz

    let mut files: Vec<PathBuf> = Vec::new();
    let runtime = tokio::runtime::Runtime::new().unwrap();


    for path in paths {
        if is_s3(path.clone()) {
            // Use async_std to block until we scour the s3 directory for files
            runtime.block_on(async {
                let s3_paths = expand_s3_dir(&path, Some(ext)).await.unwrap();
                files.extend(s3_paths);                
            });                
        }
        else if path.is_dir() {
            let path_str = path
                .to_str()
                .ok_or_else(|| anyhow!("invalid path '{}'", path.to_string_lossy()))?;
            let mut num_hits = 0;
            //for entry in glob(&format!("{}/**/*.json*.gz", path_str))? {
            for entry in glob(&format!("{}/**/*{}", path_str, ext))? {

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

fn read_pathbuf_to_mem(input_file: &PathBuf) -> Result<BufReader<Cursor<Vec<u8>>>, Error> {
    // Generic method to read local or s3 file into memory
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
        let contents = read_local_file_into_memory(input_file).expect("Failed to read contents into memory");
        BufReader::new(contents)
    };
    Ok(reader)
} 


fn read_local_file_into_memory(input_file: &PathBuf) ->Result<Cursor<Vec<u8>>, Error>{
    // Takes a local file (must be local!) and reads it into a Cursor of bytes
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




// NAMER FUNCTIONS: NO DIRECTORY INFORMATION HERE!
fn get_local_cell_basename(fid: usize) -> PathBuf {
    // Standardized method to name "local cell" files
    PathBuf::from(format!("local_cell_{:08}.tar", fid))
}


fn get_overflow_basename(overflow_round: usize, overflow_id: usize) -> PathBuf {
    // Standardized method to name each overflow filename
    PathBuf::from(format!("overflow_{:04}_{:08}.tar", overflow_round, overflow_id))
}


fn get_chunk_basename(chunk_id: usize) -> PathBuf {
    // Standardized method to name each output chunk tarfile
    PathBuf::from(format!("chunk_{:08}.tar", chunk_id))
}



fn build_cellmap(filenames: &Vec<PathBuf>) -> Option<CellMap> {
    // Note: filenames SHOULD have full path info
    if filenames.len() == 0 {
        return None;
    }

    let mut mapper = HashMap::new();
    for (idx, filename) in filenames.into_iter().enumerate() {
        if let Some(parent_dir) = filename.parent() {
            fs::create_dir_all(parent_dir).unwrap();
        }
        let writer = Arc::new(
            Mutex::new(
            Builder::new(
            BufWriter::new(
            OpenOptions::new()
            .append(true)
            .create(true)
            .open(filename)
            .unwrap()
        ))));
        mapper.insert(idx, writer);
    }
    Some(mapper)
}


fn write_contexts(mut tokens: VecDeque<i32>, local_cell_mapper: &CellMap,
                  seqlen: usize, hash_seed: usize) -> Result<VecDeque<i32>> {
    // Given a vector of tokens, will try to take the first seqlen and write them to their appropriate file
    // If the tokens have length less than seqlen, they will do nothing
    let num_local_cells = local_cell_mapper.len(); 
    while tokens.len() >= seqlen {
        let context: Vec<i32> = tokens.drain(..seqlen).collect();
        let local_cell_fid = (hash_vec(&context, hash_seed) % num_local_cells as u64) as usize;
        let json_string = serde_json::to_string(&context).unwrap();
        let mut header = tar::Header::new_gnu();
        let mut uid = Uuid::new_v4().to_string();
        uid.push_str(".json.gz");

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(json_string.as_bytes()).unwrap();
        let compressed_data = encoder.finish().unwrap();
        let compressed_data = compressed_data.as_slice();
        header.set_size(compressed_data.len() as u64);
        header.set_cksum();
        let mut builder = local_cell_mapper.get(&local_cell_fid).unwrap().lock().unwrap();
        builder.append_data(&mut header, uid, compressed_data).unwrap();
    }
    Ok(tokens)
}

fn get_manifest_line(chunk_filename: &PathBuf, num_sequences: usize) -> Result<String, Error> {
    // Given a chunk filename (like "s3://bucket/chunk_0000.tar") and num sequences, gets the line for the manifest
    // like '{"shard": "chunk_0000", "num_sequences": 1234}'

    let shard = chunk_filename.file_stem().and_then(|s| s.to_str()).unwrap().to_string();
    let data = json!({
        "shard": shard,
        "num_sequences": num_sequences
    }).to_string();

    Ok(data)
}


fn save_manifest(manifest_lines: Arc<Mutex<Vec<String>>>, output_dir: &PathBuf) -> Result<(), Error> {
    // Saves the manifest to local or s3 location

    let mut output_loc = output_dir.clone();
    output_loc.push("manifest.jsonl");
    let manifest_contents = manifest_lines.lock().unwrap().join("\n");
    let mut manifest_contents = Cursor::new(manifest_contents.into_bytes());
    if is_s3(&output_loc) {
        let rt = tokio::runtime::Builder::new_current_thread()
                 .enable_all()
                 .build()
                 .unwrap();
        rt.block_on(write_cursor_to_s3(&output_loc, manifest_contents)).unwrap();
    } else {
        let mut file = File::create(output_loc).expect("Failed to create manifest file");
        std::io::copy(&mut manifest_contents, &mut file).expect("Failed to write manifest");
    }

    Ok(())
}




/*================================================================
=                    Other assorted utilities                    =
================================================================*/


fn hash_vec<T>(vec: &Vec<T>, seed: usize) -> u64 where T: Hash {
    // Hashes a vector of type T into a u64 hash value
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
    // Casts vector of type T to type U (e.g. for usize -> i32)
    vec.into_iter()
        .map(|item| {
            U::try_from(item).map_err(|e| format!("Cannot cast element: {:?}", e))
        })
        .collect()
}




/*===================================================================
=                     Tokenization utilities                        =
===================================================================*/


fn load_tokenizer(tokenizer_name: &String) -> Result<Tokenizer> {
    // Loads a huggingface tokenizer from pretrained name
    // Note this uses an OLDER version of huggingface tokenizers (may be a deprecated method)
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


fn load_tiktoken_tokenizer(tokenizer_name: &String) -> Result<CoreBPE> {
    // Loats the tiktoken tokenizer. Some magic strings here, but don't worry about it ;)
    let (pattern, tiktoken_data) = match (*tokenizer_name).as_str() {
        "EleutherAI/gpt-neox-20b" => {
            (r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
             include_str!("../EleutherAI_gpt-neox-20b.tiktoken"))
        }  
        "meta-llama/Meta-Llama-3-8B" => {
            (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
             include_str!("../meta-llama-3-8B.tiktoken"))
        }
        _ => {
            return Err(anyhow!("Unknown tokenizer name: {}", tokenizer_name));
        }
    };

    let mut encoder = FxHashMap::default();

    for line in tiktoken_data.lines() {
        let mut parts = line.split(' ');
        let raw = parts.next().unwrap();
        let token = &general_purpose::STANDARD.decode(raw)?;
        let rank: usize = parts.next().unwrap().parse().unwrap();
        encoder.insert(token.clone(), rank);        
    }
    let special_tokens = FxHashMap::default();
    let bpe = CoreBPE::new(
        encoder, 
        special_tokens,
        pattern,
        )?;

    Ok(bpe)
}


/*======================================================
=              Tokenize/Coarse Shuffle code            =
======================================================*/
/*
Code here does the tokenization and coarse shuffling.
Two cases:
    - inputs are already tokenized and split into contexts, in which case
      each context just needs to be "coarsely" sorted into the local cells
    - inputs are NOT tokenized, and need to be tokenized/broken into contexts and 
      then "coarsely" sorted into the local cells

    The output of this whole process will be the number of TOKENS handled. 
    (but we don't count from tarfiles)
*/ 


fn coarse_shuffle(input_files: &Vec<PathBuf>, local_cell_dir: &PathBuf, 
                    threads: usize, num_local_cells: usize, seqlen: usize, hash_seed: usize, shuffle_only: bool,
                    tokenizer_name: &String, use_tiktoken: bool) -> Result<Arc<AtomicUsize>, Error> {
    // Takes a list of tars and spawns a buncha threads to process each one individually
    // by taking the pre-created contexts and putting them into tar-buckets based on their hashes (or randomly?)
    println!("Starting coarseSort loop...");
    println!("NUM LOCAL CELLS {:?}", num_local_cells);
    let local_cell_filenames: Vec<PathBuf> = (0..num_local_cells)
        .map(|i| local_cell_dir.clone().join(get_local_cell_basename(i)))
        .collect();
    println!("LOCAL CELL NAMES {:?}", local_cell_filenames);
    let mut local_cell_mapper = build_cellmap(&local_cell_filenames).unwrap();
    let pbar = ProgressBar::new(input_files.len() as u64)
        .with_style(
            ProgressStyle::with_template(
                "Files {human_pos}/{human_len} [{elapsed_precise}/{duration_precise}] [{wide_bar:.cyan/blue}]",
            ).unwrap()
        );
    let pbar = Arc::new(Mutex::new(pbar));
    pbar.lock().unwrap().inc(0); // Makes pbar show up with 0/N files complete
    let threadpool = ThreadPool::new(threads);
    let token_count = Arc::new(AtomicUsize::new(0));

    for input_file in input_files {
        let input_file = input_file.clone();
        let local_cell_mapper = local_cell_mapper.clone();
        let hash_seed = hash_seed.clone();
        let pbar = pbar.clone();
        let seqlen = seqlen.clone();
        let token_count = token_count.clone();
        let shuffle_only = shuffle_only.clone();
        let tokenizer_name = tokenizer_name.clone();
        threadpool.execute(move || {
            if shuffle_only {
                coarse_shuffle_single_tarfile(&input_file, local_cell_mapper, hash_seed).unwrap();
            } else {
                tokenize_coarse_shuffle_single(&input_file, local_cell_mapper, hash_seed, seqlen, 
                                               token_count, &tokenizer_name, use_tiktoken).unwrap();
            }
            pbar.lock().unwrap().inc(1);
        });
    }
    threadpool.join();
    for (_, builder) in local_cell_mapper.iter_mut() {
        builder.lock().unwrap().finish().unwrap(); 
        // Note: this ^ doesn't always work, so it's important that the tar::Builders go out of scope!
    }

    Ok(token_count)
}

fn tokenize_coarse_shuffle_single(input_file: &PathBuf, local_cell_mapper: CellMap, hash_seed: usize, seqlen: usize, token_count: Arc<AtomicUsize>,
                               tokenizer_name: &String, use_tiktoken: bool) -> Result<()> {

    let reader = read_pathbuf_to_mem(input_file).unwrap();
    // For a reader, will tokenize each line, build contexts, and put each context into the appropriate local cell
    let mut all_tokens = VecDeque::new();
    if use_tiktoken {
        // There's probably a better/drier way to do this tiktoken/HF branch, but I'm bad at rust =P
        let tokenizer = load_tiktoken_tokenizer(&tokenizer_name)?;
        for line in reader.lines() {
            let line = line?;
            let json: Value = serde_json::from_str(&line)?;
            let text = json["text"].as_str().unwrap();
            let encoded = tokenizer.encode_with_special_tokens(text);

            let mut tokens = cast_vector::<usize, i32>(encoded).unwrap();
            tokens.push(EOT_TOKEN);
            if tokenizer_name.as_str() == "meta-llama/Meta-Llama-3-8B" {
                all_tokens.push_back(LLAMA3_BOS_TOKEN);
            }
            token_count.fetch_add(tokens.len(), Ordering::SeqCst);
            all_tokens.append(&mut VecDeque::from(tokens));
            all_tokens = write_contexts(all_tokens, &local_cell_mapper, seqlen, hash_seed).unwrap();
        }
    } else {
        let tokenizer = load_tokenizer(&tokenizer_name).unwrap();
        for line in reader.lines() {
            let line = line?;
            let json: Value = serde_json::from_str(&line)?;
            let text = json["text"].as_str().unwrap();

            let encoded = tokenizer.encode(text, false).unwrap();
            let mut tokens = cast_vector::<u32, i32>(encoded.get_ids().to_vec()).unwrap();
            tokens.push(EOT_TOKEN);
            //tokens.push(tokenizer.token_to_id("<EOT>").unwrap());
            token_count.fetch_add(tokens.len(), Ordering::SeqCst);
            all_tokens.append(&mut VecDeque::from(tokens));
            all_tokens = write_contexts(all_tokens, &local_cell_mapper, seqlen, hash_seed).unwrap();
        }        
    }

    if all_tokens.len() < seqlen {
        token_count.fetch_add(seqlen - all_tokens.len(), Ordering::SeqCst);
    }
    while all_tokens.len() < seqlen {

        all_tokens.push_back(PAD_TOKEN);
    }
    let _ = write_contexts(all_tokens, &local_cell_mapper, seqlen, hash_seed).unwrap();

    Ok(())
}


fn coarse_shuffle_single_tarfile(input_file: &PathBuf, 
                                 local_cell_mapper: CellMap,
                                 hash_seed: usize,
                                ) -> Result<(), Error> {
    // Loads the entries from a single tarfile and pushes them into their 
    let reader = read_pathbuf_to_mem(input_file).unwrap();
    let num_local_cells = local_cell_mapper.len() as u64;


    let mut tar = Archive::new(reader);
    for entry in tar.entries()? {
        let mut entry = entry?;
        let mut data : Vec<u8> = Vec::new();
        entry.read_to_end(&mut data).unwrap();
        let mut header = tar::Header::new_gnu();
        header.set_size(data.len() as u64);
        header.set_cksum();
        let local_cell_fid = (hash_vec(&data, hash_seed) % num_local_cells) as usize;
        let path = entry.path().unwrap();
        let mut builder = local_cell_mapper.get(&local_cell_fid).unwrap().lock().unwrap();
        builder.append_data(&mut header, path, data.as_slice()).unwrap();
    }
    Ok(())
}


/*======================================================
=                 Fine shuffle and upload code         =
======================================================*/


fn process_local_cell(filename: &PathBuf, overflow_writer: &Option<CellMap>, output_dir: &PathBuf, 
                      wds_chunk_size: usize, wds_chunk_id: &AtomicUsize, total_token_count: &Arc<AtomicUsize>,
                      hash_seed: usize, manifest_vec: Arc<Mutex<Vec<String>>>, pbar: Arc<Mutex<ProgressBar>>) -> Result<()> {

    // Given a "local cell" which has a bunch of contexts in it, shuffles it and groups into chunks of wds_chunk_size
    // For complete chunks, finalizes these and pushes to output directory
    // For incomplete chunks, if overflow writer exists -> write incomplete chunks to overflow file
    // Also does some branching: if no overflow writer, then this is final step and we can write chunks in parallel
    let mut rng = thread_rng(); // TODO: reproducible shuffling (maybe seed the filename?)
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);
    let mut archive = Archive::new(reader);


    // Load all tar file into vec of (path, contents) [idk, weird errors if I try to skip this step]
    let mut loaded_entries: Vec<(PathBuf, Vec<u8>)> = Vec::new();    
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.to_path_buf();
        let mut contents: Vec<u8> = Vec::new();
        entry.read_to_end(&mut contents).unwrap();
        loaded_entries.push((path, contents));
    }
    loaded_entries.shuffle(&mut rng);

    //println!("FILENAME {:?} | ENTRIES {:?}", filename, loaded_entries.len());
    for chunk in loaded_entries.chunks(wds_chunk_size) {
        if chunk.len() != wds_chunk_size && !overflow_writer.is_none() {
            // If chunk needs to be kicked back to a local overflow cell
            let overflow_writer = overflow_writer.as_ref().unwrap();
            let num_overflows = overflow_writer.len() as u64;
            // println!("FILENAME {:?} | HAS LEFTOVERS {:?} | {:?}", filename, chunk.len(), chunk[0].0);
            for (path, contents) in chunk {
                let content_hash = hash_vec::<u8>(contents, hash_seed);
                let mut header = tar::Header::new_gnu();
                let mut contents = contents.as_slice();
                header.set_size(contents.len() as u64);
                header.set_cksum();
                let mut builder = overflow_writer
                    .get(&((content_hash % num_overflows) as usize))
                    .unwrap()
                    .lock()
                    .unwrap();
                builder.append_data(&mut header, path, &mut contents).expect("FAILED TO APPEND DATA???");
            }
        } else {
            // If chunk is complete (either full size chunk, or final chunk)
            finalize_chunk(chunk, output_dir, wds_chunk_id, total_token_count, &manifest_vec).unwrap();
        }
    }

    fs::remove_file(filename).unwrap();
    pbar.lock().unwrap().inc(1);
    Ok(())
}



fn finalize_chunk(chunk: &[(PathBuf, Vec<u8>)], output_dir: &PathBuf, 
                  wds_chunk_id: &AtomicUsize, total_context_count: &Arc<AtomicUsize>, 
                  manifest_vec: &Arc<Mutex<Vec<String>>>
                ) -> Result<()> {
    // Given a chunk, output directory, and atomic id/namer, and atomic token-counter
    // Wraps the chunk in a tarfile and saves it in the output dir

    // Computes the filename for the chunk
    let chunk_id = wds_chunk_id.fetch_add(1, Ordering::SeqCst);
    let chunk_filename = output_dir.clone().join(get_chunk_basename(chunk_id));

    // And then wraps the chunk in a tarfile 
    let mut bio = Cursor::new(Vec::new());
    {
        let mut builder = Builder::new(&mut bio);
        for (path, contents) in chunk {
            let mut header = tar::Header::new_gnu();
            let mut contents = contents.as_slice();
            header.set_size(contents.len() as u64);
            header.set_cksum();
            builder.append_data(&mut header, path, &mut contents).unwrap();
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
        if let Some(parent_dir) = chunk_filename.parent() {
            fs::create_dir_all(parent_dir).unwrap();
        }        
        let mut file = File::create(chunk_filename.clone()).expect("Failed to create file");
        std::io::copy(&mut bio, &mut file).expect("Failed to write to file");
    }

    // And also saves this chunk into the manifest
    total_context_count.fetch_add(chunk.len(), Ordering::SeqCst);
    let manifest_line = get_manifest_line(&chunk_filename, chunk.len()).unwrap();
    manifest_vec.lock().unwrap().push(manifest_line);
    Ok(())   

}




fn fine_sort_and_save(local_cell_dir: &PathBuf, output: &PathBuf, num_local_cells: usize,
                      threads: usize, wds_chunk_size: usize, hash_seed: usize) -> Result<Arc<AtomicUsize>, Error> {
    println!("Starting fineSort/upload loop...");

    // First count how many TOTAL files we need to process (and build a pbar from this)
    let mut total_calls = 0;
    let mut remaining_files = num_local_cells;
    while remaining_files > 0 {
        total_calls += remaining_files;
        remaining_files = remaining_files / OVERFLOW_REDUCTION;
    }
    let pbar = ProgressBar::new(total_calls as u64)
        .with_style(
            ProgressStyle::with_template(
                "Files {human_pos}/{human_len} [{elapsed_precise}/{duration_precise}] [{wide_bar:.cyan/blue}]",
            ).unwrap()
        );
    let pbar = Arc::new(Mutex::new(pbar));
    pbar.lock().unwrap().inc(0); // Makes pbar show up with 0/N files complete


    // And then set up the loop
    let manifest_vec: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let wds_chunk_id = Arc::new(AtomicUsize::new(0));
    let total_context_count = Arc::new(AtomicUsize::new(0));
    let mut overflow_round = 0;
    let mut num_overflows = num_local_cells / OVERFLOW_REDUCTION;
    let mut src_filenames: Vec<PathBuf> = (0..num_local_cells)
        .map(|idx| local_cell_dir.clone().join(get_local_cell_basename(idx))).collect();

    while src_filenames.len() > 0 {
        let threadpool = ThreadPool::new(threads);    
        let overflow_filenames: Vec<PathBuf> = (0..num_overflows)
            .map(|idx| local_cell_dir.clone().join(get_overflow_basename(overflow_round, idx)))
            .collect();
        let overflow_writers = build_cellmap(&overflow_filenames);
        println!("STARTING ROUND {:?} | {:?} SRC FILES | {:?} WRITERS",
                 overflow_round, src_filenames.len(), overflow_filenames.len());        
        let src_pointer = &src_filenames;
        for filename in src_pointer {     
            let filename = filename.clone();       
            let output_dir = output.clone();
            let wds_chunk_id = wds_chunk_id.clone();
            let overflow_writers = overflow_writers.clone();
            let manifest_vec = manifest_vec.clone();
            let total_context_count = total_context_count.clone();
            let pbar = pbar.clone();
            threadpool.execute(move || {
                process_local_cell(&filename, &overflow_writers, &output_dir, wds_chunk_size, &wds_chunk_id,
                                   &total_context_count, hash_seed, manifest_vec, pbar).unwrap()
            });                        
        } 
        threadpool.join();
        overflow_round += 1;
        if num_overflows == 1 {
            num_overflows = 0;
        } else {
            num_overflows = num_overflows / OVERFLOW_REDUCTION;
            if num_overflows == 0 {
                num_overflows = 1;
            }
        }
        if let Some(mut writers) = overflow_writers {
            for (_, builder) in writers.iter_mut() {
                builder.lock().unwrap().finish().unwrap();
                // Need to check that these finish appropriately ^^
            }                    
        }

        src_filenames = overflow_filenames.clone();
    };
    save_manifest(manifest_vec, &output).unwrap();

    Ok(total_context_count)
}



/*==============================================================
=                         MAIN BLOCK                           =
==============================================================*/

fn main() -> Result<()> {
    /*
    Side method that does ONLY the shuffling. 
    - Takes in a directory/directories of .tar's, each of which contain some .json.gz contexts 
    - Shuffles all of them in two phases:
    1. Setup 
    2. Coarse shuffle into files
    3. Load each file and actually shuffle -> do the overflow loop
    */

    // Step 1: setup 
    println!("Setting up Shuffle run");
    let start_time = Instant::now();    
    let args = Args::parse();
    let threads = if args.threads == 0 {
        available_parallelism().unwrap().get()
    } else {
        args.threads
    };
    let ext = if args.ext.len() > 0 {
        Some(args.ext.as_str())
    } else if args.shuffle_only {
        Some("tar")
    } else {
        Some("jsonl.gz")
    };

    let input_files = expand_dirs(args.input, ext).unwrap();
    // Step 2: Do the coarse shuffle
    let total_token_count = coarse_shuffle(&input_files, &args.local_cell_dir, threads, args.num_local_cells,
                                           args.seqlen, args.hash_seed, args.shuffle_only,
                                           &args.tokenizer, args.use_tiktoken).unwrap();
    
    // Step 3: Do the "fine-grained" sorting and upload/save in wds format
    let total_context_count = fine_sort_and_save(&args.local_cell_dir, &args.output, args.num_local_cells, 
                                                 threads, args.wds_chunk_size, args.hash_seed).unwrap();



    // Step 4: Finalize by finishing the overflow writer, and writing some stats
    println!("Finishing tokenize shuffle run!");
    println!("-------------------------------");
    println!("Ran in {:?} (s)", start_time.elapsed().as_secs());
    println!("Processed {:?} tokens", total_token_count.fetch_add(0, Ordering::SeqCst));
    println!("Processed {:?} contexts", total_context_count.fetch_add(0, Ordering::SeqCst));
    Ok(())
}

