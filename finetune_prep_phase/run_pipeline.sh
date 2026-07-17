#!/bin/bash

# ==========================================
# Configuration and parameters for Pipeline
# ==========================================

# Input JSON file (format: list of {topic, nl, fol})
INPUT_JSON="data/sample.json"

# Test file can also be used
# INPUT_JSON="data/test.json"

# Directory to save output files (.npz, .txt, .jsonl errors)
OUTPUT_DIR="output/"

# Prefix for output files (e.g. val -> val_cpp_paths.npz)
PREFIX="sample"

# T5 tokenizer model name (can be a local path or huggingface model name)
TOKENIZER_NAME="t5-base"

# Model parameters
MAX_LENGTH=256
MAX_DEPTH=10

# Enable this flag to generate fol_trees.txt for human readability (leave empty if not needed)
DUMP_TEXT_TREE="--dump_text_tree"

# ==========================================
# Run Pipeline Script
# ==========================================

echo "Starting Logic-JEPA Parsing Pipeline..."
echo "Input: $INPUT_JSON"
echo "Output: $OUTPUT_DIR"
echo "Tokenizer: $TOKENIZER_NAME"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

/home/server/miniconda3/envs/logic-jepa/bin/python process_pipeline.py \
    --input "$INPUT_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --prefix "$PREFIX" \
    --tokenizer "$TOKENIZER_NAME" \
    --max_length $MAX_LENGTH \
    --max_depth $MAX_DEPTH \
    $DUMP_TEXT_TREE

echo "Completed!"
