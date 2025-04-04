"""
Train the RegexTokenizer with GPT4 pattern on a Huggingface dataset and convert
to a Huggingface-compatible tokenizer.

Example usage:
1. Run with default values:
   python train_hf.py

2. Or run with command-line arguments:
   python train_hf.py --dataset_path "wikitext" --dataset_name "wikitext-103-raw-v1" --vocab_size 32768 --output_dir "hf-tokenizer"
"""

import os
import time
import argparse
import logging
from minbpe import RegexTokenizer, GPT4_SPLIT_PATTERN, get_training_corpus_from_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default configuration - edit these values as needed
DEFAULT_CONFIG = {
    "dataset_path": "ali-issa/arb_diacritized_tokenized_filtered_dataset_with_arb-bpe-tokenizer-32768",       # Huggingface dataset path or name
    "dataset_name": '',  # Specific dataset configuration name
    "text_column": "diacritized_text",            # Column in the dataset containing text
    "split": "train",                 # Dataset split to use
    "vocab_size": 32768,              # Vocabulary size (must be >= 256)
    "batch_size": 10000,              # Batch size for processing
    "batch_limit": None,              # Optional limit on number of batches to process
    "special_tokens": ["<|endoftext|>"],  # Special tokens to include in the vocabulary
    "output_dir": "models",           # Directory to save the trained tokenizer
    "model_name": "regex_hf",         # Name for the saved tokenizer files
    "convert_to_hf": True,            # Convert to Huggingface tokenizer format
    "hf_output_dir": None,            # Directory to save the Huggingface tokenizer
}

def train_tokenizer(config):
    """
    Train the tokenizer with the given configuration.
    
    Args:
        config: Dictionary containing configuration parameters
    """
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Load dataset and create training corpus
    dataset_kwargs = {"split": config["split"]}
    if config["dataset_name"]:
        dataset_kwargs["name"] = config["dataset_name"]
    
    logging.info(f"Creating training corpus from dataset {config['dataset_path']}")
    training_corpus = get_training_corpus_from_dataset(
        dataset_path=config["dataset_path"],
        text_column=config["text_column"],
        batch_size=config["batch_size"],
        **dataset_kwargs
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
    config = DEFAULT_CONFIG
    logging.info("Using default configuration")
    # Train the tokenizer
    train_tokenizer(config)

if __name__ == "__main__":
    main() 