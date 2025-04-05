# Create tokenizer
from minbpe.regex import RegexTokenizer


tokenizer = RegexTokenizer()

# Train from a directory containing parquet files
tokenizer.train_from_parquet_dir(
    data_dir='arb_diacritized_tokenized_filtered_dataset_with_arb-bpe-tokenizer-32768/train',
    text_column='diacritized_text',
    vocab_size=32768,
    temp_dir='tokenizer_temp',
    verbose=True,
    chunk_size=10000  # Process 5000 rows at a time (smaller for less memory usage)
)

# Save the trained tokenizer
tokenizer.save('my_tokenizer')