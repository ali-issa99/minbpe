# MinBPE with Huggingface Integration

This extension to the MinBPE package allows training tokenizers on Huggingface datasets and converting them to Huggingface's tokenizer format.

## Installation

```bash
pip install -r requirements.txt
```

For Huggingface conversion support, make sure to install the optional dependencies:

```bash
pip install tokenizers transformers
```

## Training on Huggingface Datasets

The package now supports training on datasets from Huggingface's datasets library:

```python
from minbpe import RegexTokenizer, GPT4_SPLIT_PATTERN, get_training_corpus_from_dataset

# Create a training corpus from a Huggingface dataset
training_corpus = get_training_corpus_from_dataset(
    dataset_path="wikitext",
    text_column="text",
    batch_size=10000,
    name="wikitext-103-raw-v1"
)

# Create and train the tokenizer
tokenizer = RegexTokenizer(pattern=GPT4_SPLIT_PATTERN)
tokenizer.train_from_iterator(
    iterator=training_corpus,
    vocab_size=32768,
    verbose=True
)

# Register special tokens
special_tokens = {"<|endoftext|>": 32768}
tokenizer.register_special_tokens(special_tokens)

# Save the tokenizer
tokenizer.save("models/my_tokenizer")
```

## Converting to Huggingface Format

You can convert a trained MinBPE tokenizer to Huggingface format:

```python
from minbpe import RegexTokenizer
from minbpe.huggingface_utils import convert_to_huggingface

# Load a trained tokenizer
tokenizer = RegexTokenizer()
tokenizer.load("models/my_tokenizer.model")

# Convert to Huggingface format
hf_tokenizer = convert_to_huggingface(
    tokenizer=tokenizer,
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
    output_dir="models/hf_tokenizer"
)

# The tokenizer can now be used with Huggingface's transformers library
text = "Hello, world!"
tokens = hf_tokenizer.encode(text)
print(tokens)
print(hf_tokenizer.decode(tokens))
```

## Command-line Usage

The package includes a command-line script for training tokenizers on Huggingface datasets:

```bash
python train_hf.py --dataset_path "wikitext" --dataset_name "wikitext-103-raw-v1" --vocab_size 32768 --output_dir "models" --convert_to_hf
```

### Command-line Options

- `--dataset_path`: Huggingface dataset path or name
- `--dataset_name`: Specific dataset configuration name
- `--text_column`: Column in the dataset containing text
- `--split`: Dataset split to use (default: train)
- `--vocab_size`: Vocabulary size (must be >= 256)
- `--batch_size`: Batch size for processing
- `--batch_limit`: Optional limit on number of batches to process
- `--special_tokens`: Special tokens to include in the vocabulary
- `--output_dir`: Directory to save the trained tokenizer
- `--model_name`: Name for the saved tokenizer files
- `--convert_to_hf`: Convert to Huggingface tokenizer format
- `--hf_output_dir`: Directory to save the Huggingface tokenizer 