"""
Utilities for loading and processing datasets from Huggingface for tokenizer training.
"""

from datasets import load_dataset
from typing import List, Iterator, Dict, Any, Optional
import logging

def get_training_corpus_from_dataset(
    dataset_path: str,
    text_column: str = "text",
    batch_size: int = 10000,
    split: str = "train",
    data_dir: Optional[str] = None
) -> Iterator[List[str]]:
    """
    Load a dataset from Huggingface and yield batches of text for tokenizer training.
    
    Args:
        dataset_path: Name of the dataset on Huggingface or path to local dataset
        text_column: Column name containing the text to tokenize
        batch_size: Number of examples to include in each batch
        split: Dataset split to use (train, validation, test)
        data_dir: Optional path to local dataset files
        
    Yields:
        Batches of text for tokenizer training
    """
    # Load the dataset
    logging.info(f"Loading dataset: {dataset_path}")
    if data_dir:
        dataset = load_dataset(dataset_path, data_dir=data_dir)
    else:
        dataset = load_dataset(dataset_path)
    
    # Check if the specified column exists
    if text_column not in dataset[split].column_names:
        available_columns = ", ".join(dataset[split].column_names)
        raise ValueError(
            f"Column '{text_column}' not found in dataset. "
            f"Available columns: {available_columns}"
        )
    
    # Yield batches of text
    logging.info(f"Processing dataset with {len(dataset[split])} examples")
    for i in range(0, len(dataset[split]), batch_size):
        batch = dataset[split][i:i + batch_size][text_column]
        # Remove None values and empty strings
        batch = [text for text in batch if text and isinstance(text, str)]
        if batch:  # Yield only if batch is not empty
            yield batch 