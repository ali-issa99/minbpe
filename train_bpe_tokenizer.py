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
# Load the dataset
print("Loading dataset...")
# ds= load_dataset('ali-issa/arb_diacritized_tokenized_filtered_dataset_with_arb-bpe-tokenizer-32768',data_dir='test')
split='train'
ds= load_dataset('parquet',data_files='dataset/test/*.parquet')
dataset_size=len(ds[split])
max_batches=10000
batch_size=10000

def get_training_corpus():
    """Iterator that yields batches of text from the dataset"""
    for i in range(0, len(ds[split]), batch_size):
        batch = ds[split][i : i + batch_size][text_column]
        if batch:  # Yield only if batch is not empty
            yield batch

# ds = load_dataset('dataset', cache_dir='./cache')
print(f"Dataset loaded with {dataset_size} examples")

# Print available columns to help choose the text column
print(f"Available columns: {ds[split].column_names}")

# Choose the text column
text_column = "diacritized_text"

# Parameters for training
vocab_size = 32768
# Create the tokenizer
print(f"Training tokenizer with vocab size {vocab_size}...")
tokenizer = RegexTokenizer()

# Train using the memory-efficient iterator-based approach
t0 = time.time()
tokenizer.train_from_iterator(get_training_corpus(), vocab_size,batch_size, max_batches)
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")

# Save the tokenizer
prefix = os.path.join("models", "my_dataset_tokenizer")
tokenizer.save(prefix)
print(f"Tokenizer saved to {prefix}.model and {prefix}.vocab")