"""
Utilities for converting minbpe tokenizers to Huggingface-compatible tokenizers.
"""

import json
import os
from typing import Dict, List, Optional, Union, Tuple
from .regex import RegexTokenizer
import logging

try:
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
    from tokenizers.processors import ByteLevel as ByteLevelProcessor
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from transformers import PreTrainedTokenizerFast
    _has_tokenizers = True
except ImportError:
    _has_tokenizers = False
    logging.warning("Huggingface tokenizers package not found. To use huggingface conversion, install with: pip install tokenizers transformers")

def convert_to_huggingface(
    tokenizer: RegexTokenizer,
    bos_token: str = "<|endoftext|>",
    eos_token: str = "<|endoftext|>",
    output_dir: Optional[str] = None,
    name: str = "minbpe-tokenizer"
) -> Optional["PreTrainedTokenizerFast"]:
    """
    Convert a minbpe RegexTokenizer to a Huggingface PreTrainedTokenizerFast.
    
    Args:
        tokenizer: A trained RegexTokenizer
        bos_token: The beginning of sequence token
        eos_token: The end of sequence token
        output_dir: Directory to save the tokenizer files to
        name: Name of the tokenizer
        
    Returns:
        A PreTrainedTokenizerFast instance if tokenizers package is available
    """
    if not _has_tokenizers:
        raise ImportError(
            "Huggingface tokenizers package not found. Please install with: "
            "pip install tokenizers transformers"
        )
    
    # Generate the vocabulary and merges for Huggingface BPE format
    # Vocabulary is a mapping of token to id
    vocab = {}
    for i, token_bytes in tokenizer.vocab.items():
        # For Huggingface, we need to encode bytes as strings
        token_str = "".join([f"\\u{b:04x}" for b in token_bytes])
        vocab[token_str] = i
    
    # Add special tokens to vocabulary if they're not already there
    for token, idx in tokenizer.special_tokens.items():
        token_bytes = token.encode("utf-8")
        token_str = "".join([f"\\u{b:04x}" for b in token_bytes])
        vocab[token_str] = idx
    
    # Convert merges to the Huggingface format (pair of strings)
    merges = []
    for (p0, p1), _ in sorted(tokenizer.merges.items(), key=lambda x: x[1]):
        # Get the byte representations of each token
        token0_bytes = tokenizer.vocab[p0]
        token1_bytes = tokenizer.vocab[p1]
        
        # Convert to string representation
        token0_str = "".join([f"\\u{b:04x}" for b in token0_bytes])
        token1_str = "".join([f"\\u{b:04x}" for b in token1_bytes])
        
        # Add to merges
        merges.append(f"{token0_str} {token1_str}")
    
    # Create Huggingface BPE model
    hf_tokenizer = HFTokenizer(BPE(
        vocab=vocab,
        merges=merges,
        dropout=None,
        continuing_subword_prefix="",
        end_of_word_suffix="",
        fuse_unk=False
    ))
    
    # Add pre-tokenizer that matches our RegexTokenizer
    # First split by regex pattern
    hf_tokenizer.pre_tokenizer = Sequence([
        Split(pattern=tokenizer.compiled_pattern, behavior="isolated"),
        ByteLevel(add_prefix_space=False, use_regex=False)
    ])
    
    # Add post-processor and decoder
    hf_tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    hf_tokenizer.decoder = ByteLevelDecoder()
    
    # Wrap in a PreTrainedTokenizerFast
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=hf_tokenizer,
        bos_token=bos_token,
        eos_token=eos_token,
    )
    
    # Save if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        wrapped_tokenizer.save_pretrained(output_dir)
        print(f"Tokenizer saved to {output_dir}")
    
    return wrapped_tokenizer 