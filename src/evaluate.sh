#!/bin/bash

INPUT_FILE="/home/vuhl/nlp/transformer-mt/sampled_data/src/preprocessed_test.txt"  # Path to the input text file
CKPT_NAME="best_ckpt.tar"  # Specify the checkpoint name
DECODE_STRATEGY="beam"  # Specify the decoding strategy
OUTPUT_FILE="output.txt"  # Specify the output file path
# Remove all contents of the output file
> "$OUTPUT_FILE"
# Read each line of the input text file
while IFS= read -r line; do
  # Call the Python script for each line and redirect the output to the output file
  python src/main.py --mode='inference' --ckpt_name="$CKPT_NAME" --input="$line" --decode="$DECODE_STRATEGY" | tail -n 1 >> "$OUTPUT_FILE"
done < "$INPUT_FILE"

REFERENCE_FILE="/home/vuhl/nlp/transformer-mt/sampled_data/trg/test.txt"  # Specify the reference file path

# Run the evaluation Python script
result=$(python3 src/evaluate_bleu.py --output_file="$OUTPUT_FILE" --ref_file="$REFERENCE_FILE")

# Print the result to the screen
echo "$result"
