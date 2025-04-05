from minbpe.regex import RegexTokenizer

# Create tokenizer
tokenizer = RegexTokenizer()

# Train from a directory containing parquet files
tokenizer.train_from_parquet_dir(
    data_dir='data/corpus/',     # Directory containing your parquet files
    text_column='text',          # Column in parquet files that contains the text
    vocab_size=32768,             # Desired vocabulary size
    verbose=True                 # Show progress information
)

# Save the trained tokenizer
tokenizer.save('my_tokenizer')