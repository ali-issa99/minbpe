"""
Memory-efficient example of training a RegexTokenizer with a Hugging Face dataset.
This script loads a local dataset from './cache' and trains a tokenizer with a vocabulary size of 32768
using the built-in train_from_iterator method which processes batches efficiently.
"""

import os
import time
from datasets import load_dataset
from minbpe import RegexTokenizer

# Create a directory for models if it doesn't exist
os.makedirs("models", exist_ok=True)

def get_training_corpus():
    """Iterator that yields batches of text from the dataset"""
    for i in range(0, len(ds['train']), 10000):
        batch = ds['train'][i : i + 10000][text_column]
        # batch = [text for text in batch if text is not None]  # Remove None values
        if batch:  # Yield only if batch is not empty
            yield batch

# Load the dataset
print("Loading dataset...")

ds= load_dataset('ali-issa/arb_diacritized_tokenized_filtered_dataset_with_arb-bpe-tokenizer-32768')
dataset_size=len(ds['train'])
max_batch_size=int(dataset_size//2)
# ds = load_dataset('dataset', cache_dir='./cache')
print(f"Dataset loaded with {dataset_size} examples")

# Print available columns to help choose the text column
print(f"Available columns: {ds['train'].column_names}")

# Choose the text column
text_column = "diacritized_text"

# Parameters for training
vocab_size = 32768
# Create the tokenizer
print(f"Training tokenizer with vocab size {vocab_size}...")
tokenizer = RegexTokenizer()

# Train using the memory-efficient iterator-based approach
t0 = time.time()
tokenizer.train_from_iterator(get_training_corpus(), vocab_size,max_batches=max_batch_size)
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")

# Save the tokenizer
prefix = os.path.join("models", "my_dataset_tokenizer")
tokenizer.save(prefix)
print(f"Tokenizer saved to {prefix}.model and {prefix}.vocab")

# # Test the tokenizer on a sample text
# sample_batch = next(get_training_corpus())
# if sample_batch:
#     sample_text = sample_batch[0][:100]  # Take first 100 chars of first example
#     print("\nTesting tokenizer on sample text:")
#     print(f"Sample: {sample_text}")
#     tokens = tokenizer.encode(sample_text)
#     print(f"Encoded to {len(tokens)} tokens: {tokens}")
#     decoded = tokenizer.decode(tokens)
#     print(f"Decoded: {decoded}")
#     print(f"Roundtrip successful: {sample_text == decoded}") 