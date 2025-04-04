"""
Train the RegexTokenizer with GPT4 pattern on a local parquet files and convert
to a Huggingface-compatible tokenizer.

Example usage:
python train_hf.py
"""

import os
import time
import logging
from typing import Iterator, List
from minbpe import RegexTokenizer, GPT4_SPLIT_PATTERN
from datasets import load_dataset, disable_caching
import gc

# Disable HF caching to prevent disk space issues
disable_caching()
os.environ["HF_DATASETS_CACHE"] = "NO_CACHE"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default configuration - edit these values as needed
DEFAULT_CONFIG = {
    "dataset_path": "./dataset/train",    # Directory containing parquet files
    "text_column": "diacritized_text",    # Column in the parquet files containing text
    "vocab_size": 32768,                  # Vocabulary size (must be >= 256)
    "batch_size": 500,                   # Batch size for processing
    "batch_limit": None,                  # Optional limit on number of batches to process
    "special_tokens": ["<|endoftext|>"],  # Special tokens to include in the vocabulary
    "output_dir": "models",               # Directory to save the trained tokenizer
    "model_name": "regex_hf",             # Name for the saved tokenizer files
    "streaming": True,                    # Whether to use streaming mode
}

def read_parquet_files(directory: str, text_column: str, batch_size: int, streaming: bool = True) -> Iterator[List[str]]:
    """
    Read parquet files using Hugging Face's load_dataset with efficient streaming.
    
    Args:
        directory: Directory containing parquet files
        text_column: Name of the column containing text data
        batch_size: Number of texts to yield at once
        streaming: Whether to use streaming mode
    """
    # Get all parquet files in directory
    data_files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
    if not data_files:
        raise ValueError(f"No parquet files found in {directory}")
    
    # Create data_files dict for load_dataset
    data_files = {
        "train": [os.path.join(directory, f) for f in data_files]
    }
    
    # Load dataset with streaming for memory efficiency
    dataset = load_dataset(
        "parquet",
        data_files=data_files,
        streaming=streaming,
        split="train"
    )
    
    current_batch = []
    
    # Process in batches
    for example in dataset:
        text = example.get(text_column)
        
        if not isinstance(text, str):
            continue
            
        current_batch.append(text)
        if len(current_batch) >= batch_size:
            yield current_batch
            current_batch = []
            gc.collect()  # Help manage memory
    
    # Yield remaining texts
    if current_batch:
        yield current_batch

def train_tokenizer(config):
    """
    Train the tokenizer with the given configuration.
    
    Args:
        config: Dictionary containing configuration parameters
    """
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Create training corpus iterator
    logging.info(f"Creating training corpus from parquet files in {config['dataset_path']}")
    training_corpus = read_parquet_files(
        directory=config["dataset_path"],
        text_column=config["text_column"],
        batch_size=config["batch_size"],
        streaming=config["streaming"]
    )
    
    # Create tokenizer with GPT4 pattern
    tokenizer = RegexTokenizer(pattern=GPT4_SPLIT_PATTERN)
    
    # Start training timer
    t0 = time.time()
    
    # Train the tokenizer
    logging.info(f"Training tokenizer with vocabulary size {config['vocab_size']}")
    tokenizer.train_from_iterator(
        iterator=training_corpus,
        vocab_size=config["vocab_size"],
        verbose=True,
        batch_processing_limit=config["batch_limit"]
    )
    
    # Register special tokens
    if config["special_tokens"]:
        special_tokens_dict = {token: config["vocab_size"] + i for i, token in enumerate(config["special_tokens"])}
        tokenizer.register_special_tokens(special_tokens_dict)
        logging.info(f"Registered special tokens: {special_tokens_dict}")
    
    # End training timer
    t1 = time.time()
    logging.info(f"Training completed in {t1 - t0:.2f} seconds")
    
    # Save the tokenizer in minbpe format
    prefix = os.path.join(config["output_dir"], config["model_name"])
    tokenizer.save(prefix)
    logging.info(f"Saved tokenizer to {prefix}.model and {prefix}.vocab")
    
    return tokenizer

def main():
    # Use default configuration
    config = DEFAULT_CONFIG
    logging.info("Using default configuration")
    
    # Train the tokenizer
    train_tokenizer(config)

if __name__ == "__main__":
    main() 