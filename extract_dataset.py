"""
Script to extract content from a dataset and save it to a text file for tokenizer training.
This simplified version uses hard-coded parameters instead of command-line arguments.
"""

import os
from datasets import load_dataset
from tqdm import tqdm

# Configuration (modify these variables as needed)
DATASET_NAME = "dataset"  # Change to your dataset name
SPLIT = "train"
OUTPUT_FILE = "dataset_text.txt"
COLUMN = "diacritized_text"  # Specific column to extract
BATCH_SIZE = 1000
CACHE_DIR = "./cache"
MAX_SAMPLES = None  # Set to None to process all samples

def extract_text_from_dataset():
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_FILE)) if os.path.dirname(OUTPUT_FILE) else ".", exist_ok=True)
    
    # Load the dataset
    print(f"Loading dataset {DATASET_NAME}...")
    try:
        ds = load_dataset(DATASET_NAME, split=SPLIT, cache_dir=CACHE_DIR)
        print(f"Dataset loaded with {len(ds)} examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Check if the column exists in the dataset
    if COLUMN not in ds.column_names:
        print(f"Column '{COLUMN}' not found in dataset. Available columns: {ds.column_names}")
        return
    
    print(f"Extracting text from column: {COLUMN}")
    
    # Prepare to write to the output file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # Process the dataset in batches
        num_samples = min(len(ds), MAX_SAMPLES) if MAX_SAMPLES else len(ds)
        for i in tqdm(range(0, num_samples, BATCH_SIZE), desc="Processing batches"):
            end_idx = min(i + BATCH_SIZE, num_samples)
            batch = ds[i:end_idx]
            
            # Get the column data directly
            try:
                # If batch is a dictionary with column keys
                if isinstance(batch, dict) and COLUMN in batch:
                    texts = batch[COLUMN]
                    if isinstance(texts, list):
                        for text in texts:
                            if isinstance(text, str):
                                f.write(text + "\n")
                # If batch is a list of examples
                elif isinstance(batch, list):
                    # Try getting column directly if Dataset returns column data
                    try:
                        texts = batch[COLUMN]
                        if isinstance(texts, list):
                            for text in texts:
                                if isinstance(text, str):
                                    f.write(text + "\n")
                    except (KeyError, TypeError):
                        # If that fails, try accessing each example individually
                        for example in batch:
                            if isinstance(example, dict) and COLUMN in example:
                                text = example[COLUMN]
                                if isinstance(text, str):
                                    f.write(text + "\n")
                            elif isinstance(example, str):
                                # If examples are directly strings, write them
                                f.write(example + "\n")
            except Exception as e:
                print(f"Error processing batch: {e}")
                print(f"Batch type: {type(batch)}")
                print(f"First item type: {type(batch[0]) if len(batch) > 0 else 'empty'}")
                
                # Last resort: just print the batch and try to extract manually
                print("Batch content sample:", batch[:2] if isinstance(batch, list) else batch)
                
                # Try a direct access by column if this is a Dataset object
                try:
                    column_data = ds[i:end_idx][COLUMN]
                    print(f"Direct column access type: {type(column_data)}")
                    if isinstance(column_data, list):
                        for text in column_data:
                            if isinstance(text, str):
                                f.write(text + "\n")
                except Exception as inner_e:
                    print(f"Direct column access failed: {inner_e}")
    
    print(f"Extraction complete. Text saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_text_from_dataset() 