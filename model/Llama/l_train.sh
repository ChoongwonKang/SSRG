#!/usr/bin/env bash
set -euo pipefail

############################################
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct" 
TRAIN_PATH=""
VAL_PATH=""
OUTPUT_DIR=""

MAX_SEQ_LENGTH=512
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=8
GRAD_ACC_STEPS=1
EPOCHS=5
LR=1e-5

# SAVE_STEPS=50
# LOGGING_STEPS=1
# EVAL_STEPS=50

BF16_FLAG="--bf16"
############################################

echo "▶ Starting fine-tuning LLaMA 3.1 Instruct"
deepspeed --num_gpus 4 llama_fullfinetuning.py \
  --base_model                   "${BASE_MODEL}" \
  --train_path                   "${TRAIN_PATH}" \
  --val_path                     "${VAL_PATH}" \
  --output_dir                   "${OUTPUT_DIR}" \
  --hf_token hf_ \
  # --max_seq_length               ${MAX_SEQ_LENGTH} \
  # --per_device_train_batch_size  ${TRAIN_BATCH_SIZE} \
  # --per_device_eval_batch_size   ${EVAL_BATCH_SIZE} \
  # --gradient_accumulation_steps  ${GRAD_ACC_STEPS} \
  # --num_train_epochs             ${EPOCHS} \
  # --learning_rate                ${LR} \
  # --save_steps                   ${SAVE_STEPS} \
  # --logging_steps                ${LOGGING_STEPS} \
  # --eval_steps                   ${EVAL_STEPS} \
  #${BF16_FLAG}

echo "✅ Fine-tuning complete. Model saved to ${OUTPUT_DIR}"
