#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <data directory> <test data output filename>"
    exit 1
fi

DATA_DIR=$1
OUTPUT_FILE=$2

echo "Running seq2seq model..."
python3 testing.py "$DATA_DIR" "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "Model run successful. Output saved to $OUTPUT_FILE"
else
    echo "Model run failed."
    exit 1
fi
