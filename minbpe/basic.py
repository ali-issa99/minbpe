"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def train_from_iterator(self, text_iterator, vocab_size, batch_size, max_batches):
        assert vocab_size >= 1
        num_merges = vocab_size - 256
        # Initialize merges and vocab
        merges = {}  # (int, int) -> int
        # Process text batches
        batch_count = 0
        
        print("Collecting text batches...")
        # Sort unique characters for consistent ordering

        # Initialize vocab with unique characters
        vocab = {idx: bytes([idx]) for idx in range(256)}
        # Convert all text to character IDs
        all_ids = []
        print("Converting to character IDs...")
        from tqdm import tqdm
        for batch_texts in tqdm(text_iterator, desc="Converting to character IDs"):
            # Process each text in the batch
            for text in batch_texts:
                # Apply regex pattern to split text
                text_bytes = text.encode("utf-8") # raw bytes
                ids = list(text_bytes) # list of integers in range 0..255
                # input text preprocessing
                all_ids.extend(ids)

            # Stop if we've reached the maximum number of batches
            batch_count += 1
            if max_batches is not None and batch_count >= max_batches:
                print(f"Reached maximum number of batches ({max_batches})")
                break
                        
        print(f"Processed {batch_count} batches, found {len(all_ids)} text chunks")
        
        # Iteratively find and apply merges
        for i in range(num_merges):
            print(f"\nrtingrting merge {i+1}/{num_merges}")
            # Count the number of times every consecutive pair appears
            stats = {}                
            for chunk_ids in tqdm(all_ids, desc="Computing statistics"):  
                # Update statistics in-place
                get_stats(chunk_ids, stats)
            
            # If no pairs found, we're done
            if not stats:
                print(f"No more pairs found after {i} merges. Stopping early.")
                break
            
            # Find the pair with the highest count
            pair = max(stats, key=stats.get)
            # Mint a new token: assign it the next available id
            idx = 256 + i
            # Replace all occurrences of pair in ids with idx
            all_ids = [merge(chunk_ids, pair, idx) for chunk_ids in all_ids]
            # save the merge
            merges[pair] = idx
            
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            
            # Print progress
            print(f"Merge {i+1}/{num_merges}: {pair} -> {idx} {vocab[idx]} had {stats[pair]} occurrences")
        
        # Save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab    # used in decode()

    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
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
