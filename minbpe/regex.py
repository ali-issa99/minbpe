"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

from .base import Tokenizer, get_stats, merge
from tqdm import tqdm
import regex as re
import os
import tempfile
import pyarrow.parquet as pq
import pyarrow as pa
import glob
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens =  {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def train_from_iterator(self, text_iterator, vocab_size, max_batches=None, max_workers=None):
        """
        Train the tokenizer using batches of text from an iterator.
        
        Args:
            text_iterator: An iterator that yields batches of text strings
            vocab_size: The final vocabulary size (including the 256 base tokens)
            max_batches: Maximum number of batches to process (default: None, process all batches)
            max_workers: Maximum number of threads to use for parallel processing (default: None)
            
        Example:
            ```python
            from datasets import load_dataset
            from minbpe import RegexTokenizer
            
            # Define a batched text iterator
            def get_training_corpus():
                ds = load_dataset(...)
                for i in range(0, len(ds['train']), 10000):
                    batch = ds['train'][i : i + 10000]['text']
                    yield batch
            
            # Create tokenizer and train with all available batches
            tokenizer = RegexTokenizer()
            tokenizer.train_from_iterator(get_training_corpus(), vocab_size=32768)
            
            # Or limit to processing only the first 50 batches for faster training
            tokenizer = RegexTokenizer()
            tokenizer.train_from_iterator(get_training_corpus(), vocab_size=32768, max_batches=50)
            ```
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        
        # Initialize merges and vocab
        merges = {}  # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}  # idx -> bytes
        
        # Process all batches and collect token IDs
        all_ids = []
        batch_count = 0
        
        print("Processing text batches...")
        
        # Define a function to process a single text and return its IDs
        def process_text(text):
            text_chunks = self.compiled_pattern.findall(text)
            return [list(ch.encode("utf-8")) for ch in text_chunks]
        
        # Define a function to process an entire batch
        def process_batch(batch_texts):
            batch_ids = []
            for text in batch_texts:
                batch_ids.extend(process_text(text))
            return batch_ids
        
        # First pass: collect all token IDs from the iterator
        # Use parallel processing if max_workers is specified
        if max_workers is not None and max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for batch_texts in tqdm(text_iterator, desc="Processing text batches"):
                    batch_count += 1
                    
                    # Process the batch in parallel
                    all_ids.extend(process_batch(batch_texts))
                    
                    # Stop if we've reached the maximum number of batches
                    if max_batches is not None and batch_count >= max_batches:
                        print(f"Reached maximum number of batches ({max_batches})")
                        break
        else:
            # Sequential processing
            for batch_texts in tqdm(text_iterator, desc="Processing text batches"):
                batch_count += 1
                
                # Process each text in the batch
                all_ids.extend(process_batch(batch_texts))
                
                # Stop if we've reached the maximum number of batches
                if max_batches is not None and batch_count >= max_batches:
                    print(f"Reached maximum number of batches ({max_batches})")
                    break
        
        print(f"Processed {batch_count} batches, found {len(all_ids)} text chunks")
        
        # Iteratively find and apply merges
        for i in range(num_merges):
            print(f"\nStarting merge {i+1}/{num_merges}")
            
            # Count the number of times every consecutive pair appears
            stats = {}                
            
            # Define a function to update stats for a subset of all_ids
            def update_stats_for_subset(subset_ids):
                subset_stats = {}
                for ids in subset_ids:
                    get_stats(ids, subset_stats)
                return subset_stats
            
            # Use parallel processing for statistics computation if max_workers is specified
            if max_workers is not None and max_workers > 1:
                # Split all_ids into chunks for parallel processing
                chunk_size = max(1, len(all_ids) // max_workers)
                id_chunks = [all_ids[j:j+chunk_size] for j in range(0, len(all_ids), chunk_size)]
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    chunk_stats_list = list(tqdm(
                        executor.map(update_stats_for_subset, id_chunks),
                        total=len(id_chunks),
                        desc="Computing statistics"
                    ))
                    
                    # Merge all stats dictionaries
                    for chunk_stats in chunk_stats_list:
                        for pair, count in chunk_stats.items():
                            stats[pair] = stats.get(pair, 0) + count
            else:
                # Sequential processing
                for ids in tqdm(all_ids, desc="Computing statistics"):
                    get_stats(ids, stats)
            
            # If no pairs found, we're done
            if not stats:
                print(f"No more pairs found after {i} merges. Stopping early.")
                break
            
            # Find the pair with the highest count
            pair = max(stats, key=stats.get)
            # Mint a new token: assign it the next available id
            idx = 256 + i
            
            # Define a function to apply merge to a subset of all_ids
            def apply_merge_to_subset(subset_ids, merge_pair, merge_idx):
                return [merge(ids, merge_pair, merge_idx) for ids in subset_ids]
            
            # Use parallel processing for merge application if max_workers is specified
            if max_workers is not None and max_workers > 1:
                # Split all_ids into chunks for parallel processing
                chunk_size = max(1, len(all_ids) // max_workers)
                id_chunks = [all_ids[j:j+chunk_size] for j in range(0, len(all_ids), chunk_size)]
                
                # Create a partial function with the pair and idx already set
                apply_merge_partial = partial(apply_merge_to_subset, merge_pair=pair, merge_idx=idx)
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    merged_chunks = list(tqdm(
                        executor.map(apply_merge_partial, id_chunks),
                        total=len(id_chunks),
                        desc="Applying merge"
                    ))
                    
                    # Concatenate all merged chunks
                    all_ids = [ids for chunk in merged_chunks for ids in chunk]
            else:
                # Sequential processing
                all_ids = [merge(ids, pair, idx) for ids in all_ids]
            
            # Save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            
            # Print progress
            print(f"Merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
        
        # Save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab    # used in decode()

    def train_from_parquet(self, parquet_files, text_column, vocab_size, temp_dir=None, verbose=False, chunk_size=10000):
        """
        Train the tokenizer using parquet files as the data source with optimized performance.
        
        Args:
            parquet_files: List of paths to parquet files
            text_column: Name of the column containing text data in the parquet files
            vocab_size: The final vocabulary size (including the 256 base tokens)
            temp_dir: Directory to store temporary processed files (default: uses system temp dir)
            verbose: Whether to print progress information
            chunk_size: Number of rows to process at once (for memory efficiency)
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        
        # Create temporary directory if not provided
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="minbpe_")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Initialize merges and vocab
        merges = {}  # (int, int) -> int
        merge_list = []  # Store merges in order for efficient application
        vocab = {idx: bytes([idx]) for idx in range(256)}
        
        # Initialize parquet files for processed data
        processed_files = []
        for i, parquet_file in tqdm(enumerate(parquet_files), total=len(parquet_files), desc="Processing parquet files"):
            processed_file = os.path.join(temp_dir, f"processed_{i}.parquet")
            processed_files.append(processed_file)
            
            if verbose:
                print(f"Initial processing of {parquet_file}")
            
            # Read the parquet file using PyArrow (only the text column)
            parquet_dataset = pq.ParquetFile(parquet_file)
            num_rows = parquet_dataset.metadata.num_rows
            
            # Process in chunks to save memory
            all_token_ids = []
            
            # Read the file in batches using iter_batches instead of trying to access row groups directly
            for batch in tqdm(parquet_dataset.iter_batches(batch_size=chunk_size, columns=[text_column]), 
                            desc=f"Processing file in chunks", 
                            total=(num_rows + chunk_size - 1) // chunk_size):
                
                # Convert to pandas for easier processing
                batch_df = batch.to_pandas()
                
                if verbose and len(all_token_ids) == 0:
                    print(f"  Processing chunks of {chunk_size} rows, total rows: {num_rows}")
                
                # Process text column
                for text in batch_df[text_column]:
                    # Split using regex pattern and convert to byte IDs
                    chunks = self.compiled_pattern.findall(text)
                    token_ids = [list(ch.encode("utf-8")) for ch in chunks]
                    all_token_ids.append(token_ids)
            
            # Save processed data using PyArrow
            token_ids_arr = pa.array(all_token_ids)
            table = pa.table([token_ids_arr], names=['token_ids'])
            pq.write_table(table, processed_file)
            
            if verbose:
                print(f"Saved processed data to {processed_file}")
        
        # Perform merges
        for merge_idx in range(num_merges):
            if verbose:
                print(f"\nStarting merge {merge_idx+1}/{num_merges}")
            
            # Compute statistics across all processed files
            stats = {}
            for i, processed_file in enumerate(processed_files):
                if verbose:
                    print(f"Computing statistics for file {i+1}/{len(processed_files)}")
                
                # Read processed data with PyArrow
                file_reader = pq.ParquetFile(processed_file)
                num_rows = file_reader.metadata.num_rows
                
                # Process in chunks using iter_batches
                for batch in tqdm(file_reader.iter_batches(batch_size=chunk_size),
                                desc=f"Computing stats for file {i+1}",
                                total=(num_rows + chunk_size - 1) // chunk_size):
                    
                    # Convert to pandas for processing
                    batch_df = batch.to_pandas()
                    
                    # Compute statistics
                    for token_ids_list in batch_df['token_ids']:
                        for ids in token_ids_list:
                            if len(ids) >= 2:
                                get_stats(ids, stats)
            
            # If no pairs found, we're done
            if not stats:
                if verbose:
                    print(f"No more pairs found after {merge_idx} merges. Stopping early.")
                break
            
            # Find the best pair to merge
            best_pair = max(stats, key=stats.get)
            
            # Mint a new token ID
            new_idx = 256 + merge_idx
            
            # Save this merge
            merges[best_pair] = new_idx
            merge_list.append((best_pair, new_idx))
            vocab[new_idx] = vocab[best_pair[0]] + vocab[best_pair[1]]
            
            if verbose:
                print(f"Merge {merge_idx+1}: {best_pair} -> {new_idx} ({vocab[new_idx]}) with {stats[best_pair]} occurrences")
            
            # Apply the merge to all files
            for i, processed_file in enumerate(processed_files):
                if verbose:
                    print(f"Applying merge to file {i+1}/{len(processed_files)}")
                
                # Read processed data with PyArrow
                file_reader = pq.ParquetFile(processed_file)
                num_rows = file_reader.metadata.num_rows
                
                all_updated_token_ids = []
                
                # Process in chunks using iter_batches
                for batch in tqdm(file_reader.iter_batches(batch_size=chunk_size),
                                desc=f"Applying merge to file {i+1}",
                                total=(num_rows + chunk_size - 1) // chunk_size):
                    
                    # Convert to pandas for processing
                    batch_df = batch.to_pandas()
                    
                    # Apply the latest merge
                    for token_ids_list in batch_df['token_ids']:
                        # Apply only the latest merge
                        updated_token_ids = [merge(ids, best_pair, new_idx) for ids in token_ids_list]
                        all_updated_token_ids.append(updated_token_ids)
                
                # Save updated data with PyArrow
                token_ids_arr = pa.array(all_updated_token_ids)
                table = pa.table([token_ids_arr], names=['token_ids'])
                pq.write_table(table, processed_file)
                
                if verbose:
                    print(f"Saved updated data to {processed_file}")
        
        # Save final results to the tokenizer
        self.merges = merges
        self.vocab = vocab
        
        # Clean up temporary files if needed
        if temp_dir != None and verbose:
            print(f"Temporary files are stored in {temp_dir}")
            print("You may want to delete them manually after verifying the results.")

    def train_from_parquet_dir(self, data_dir, text_column, vocab_size, temp_dir=None, verbose=False, pattern="*.parquet", chunk_size=10000):
        """
        Train the tokenizer using parquet files found in a directory. This method implements
        a memory-efficient approach by:
        1. Finding all parquet files in the specified directory
        2. Processing each parquet file (applying regex, converting to bytes)
        3. Computing statistics across all files 
        4. Finding the most frequent pair and applying the merge
        5. Repeating the process for each merge iteration
        
        Args:
            data_dir: Directory containing parquet files
            text_column: Name of the column containing text data in the parquet files
            vocab_size: The final vocabulary size (including the 256 base tokens)
            temp_dir: Directory to store temporary processed files (default: uses system temp dir)
            verbose: Whether to print progress information
            pattern: Glob pattern to match parquet files (default: "*.parquet")
            chunk_size: Number of rows to process at once (for memory efficiency)
            
        Example:
            ```python
            from minbpe.regex import RegexTokenizer
            
            # Create tokenizer and train
            tokenizer = RegexTokenizer()
            tokenizer.train_from_parquet_dir(
                data_dir='data/corpus/',
                text_column='text',
                vocab_size=5000,
                verbose=True
            )
            
            # Save the trained tokenizer
            tokenizer.save('my_tokenizer')
            ```
        """
        # Find all parquet files in the directory
        parquet_path = os.path.join(data_dir, pattern)
        parquet_files = glob.glob(parquet_path)
        
        if not parquet_files:
            raise ValueError(f"No parquet files found in {data_dir} matching pattern {pattern}")
        
        if verbose:
            print(f"Found {len(parquet_files)} parquet files in {data_dir}")
            for file in parquet_files:
                print(f"  - {file}")
                
        # Train using the found parquet files
        self.train_from_parquet(
            parquet_files=parquet_files,
            text_column=text_column,
            vocab_size=vocab_size,
            temp_dir=temp_dir,
            verbose=verbose,
            chunk_size=chunk_size
        )

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids

 