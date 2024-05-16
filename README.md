# tokshuf-rust

Quick'n'diirty rust tool to tokenize and shuffle a collection of .jsonl.gz or .jsonl.zstd files into a file type consumable by webdatasets. Still very much a work in progress, anyone using this right now is probably better off forking and running things off of their forked version. Check back periodically to see updates to API/features/etc.

# Install
I usually spin up a c6a or c6g machine off of AWS with the AMI that has the AWSCLI already installed. THen I run this script: 
```
sudo yum install git -y 
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup.sh
bash rustup.sh -y
source ~/.bashrc
git clone https://github.com/revbucket/tokshuf-rust.git # CHANGE IF USING FORKED REPO
cd tokshuf-rust
sudo yum install gcc -y
sudo yum install cmake -y
sudo yum install openssl-devel -y
sudo yum install g++ -y
aws configure set aws_access_key_id [REDACTED: FILL IN WITH YOUR DATA]
aws configure set aws_secret_access_key [REDACTED: FILL IN WITH YOUR DATA]
aws configure set default.region [REDACTED: FILL IN WITH YOUR DATA]
cargo build --release 
```

# Usage/Internals: 
Just copying the docstring in `src/main.rs`:
    
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

# Example:
Suppose you have a bunch of .jsonl.gz files on s3 at `s3://my-bucket/path/to/input/data` and want to put the tokenized/shuffled data at `s3://my-bucket/path/to/output/data`, and you have plenty of hard drive storage at `/tmp`, then run something like:

`cargo run --release -- --input s3://my-bucket/path/to/input/data --outut s3://my/bucket/path/to/output/data --local-cell-dir /tmp/ --ext jsonl.gz`

System reqs for storage are kinda high, so probably not good for TB-scale datasets yet.

Hit me up with any questions/problems @ mattjordan.mail@gmail.com 
