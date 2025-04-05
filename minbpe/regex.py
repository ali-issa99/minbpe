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
import pandas as pd
import numpy as np
import os
import tempfile

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
import glob

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
    
    def train_from_parquet(self, parquet_files, text_column, vocab_size, temp_dir=None, verbose=False):
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
        for i, parquet_file in enumerate(parquet_files):
            processed_file = os.path.join(temp_dir, f"processed_{i}.parquet")
            processed_files.append(processed_file)
            
            if verbose:
                print(f"Initial processing of {parquet_file}")
                
            # Load the parquet file
            df = pd.read_parquet(parquet_file)
            
            # Apply regex pattern and convert to bytes
            def process_text(text):
                if not isinstance(text, str):
                    return []
                chunks = self.compiled_pattern.findall(text)
                return [list(ch.encode("utf-8")) for ch in chunks]
            
            # Process text column
            df['token_ids'] = df[text_column].apply(process_text)
            
            # Save processed data
            df[['token_ids']].to_parquet(processed_file)
            
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
                
                # Load processed data
                df = pd.read_parquet(processed_file)
                
                # Compute statistics
                for token_ids_list in tqdm(df['token_ids'], desc=f"Computing stats for file {i+1}"):
                    for ids in token_ids_list:
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
                
                # Load processed data
                df = pd.read_parquet(processed_file)
                # Apply the latest merge
                def apply_merge_to_list(token_ids_list):
                    return [merge(ids, best_pair, new_idx) for ids in token_ids_list]
                
                df['token_ids'] = df['token_ids'].apply(apply_merge_to_list)
                
                # Save updated data
                df[['token_ids']].to_parquet(processed_file)
                
                if verbose:
                    print(f"Saved updated data to {processed_file}")
        
        # Save final results to the tokenizer
        self.merges = merges
        self.vocab = vocab
        
        # Clean up temporary files if needed
        if temp_dir != None and verbose:
            print(f"Temporary files are stored in {temp_dir}")
            print("You may want to delete them manually after verifying the results.")
            
    def train_from_parquet_dir(self, data_dir, text_column, vocab_size, temp_dir=None, verbose=False, pattern="*.parquet"): 
        
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
            verbose=verbose
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