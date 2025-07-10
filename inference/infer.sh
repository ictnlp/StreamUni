#!/bin/bash

MODEL_DIR="model_dir"
CHUNK_LENGTH=640
QUEUE_SIZE=3
WAIT_K=5
INSTRUCTION='Transcribe the audio to text, and then translate the audio to German. Use <sep> as a separator between the original transcript and the translation.'
JSON_DIR="json_dir"
OUTPUT_DIR="output_dir"
LANG_PAIR="en_de"

python stream_st_infer.py \
    --model_path "$MODEL_DIR" \
    --chunk_length $CHUNK_LENGTH \
    --queue_size $QUEUE_SIZE \
    --wait_k $WAIT_K \
    --cot_instruction "$INSTRUCTION" \
    --infer_json "$JSON_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --lang_pair "$LANG_PAIR"