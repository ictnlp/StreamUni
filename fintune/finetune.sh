#!/bin/bash

MASTER_PORT=$(shuf -i 25000-30000 -n 1)
WORLD_SIZE=8

MODEL_NAME=model_dir
VOICE_DIR=train_json_dir
OUTPUT_DIR=StreamUni_model

BATCH_SIZE=32
BATCH_SIZE_PER_GPU=2
NUM_EPOCHS=1
LEARNING_RATE=4e-5
WEIGHT_DECAY=0.01

deepspeed \
    --include localhost:0,1,2,3,4,5,6,7 \
    --master_port $MASTER_PORT \
    speech_finetune.py \
    --deepspeed zero2.json \
    --model_name_or_path $MODEL_NAME \
    --voice_dir $VOICE_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --batch_size_per_gpu $BATCH_SIZE_PER_GPU \
    --learning_rate $LEARNING_RATE \
    --wd $WEIGHT_DECAY \
    --use_flash_attention