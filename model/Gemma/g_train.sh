#!/usr/bin/env bash
set -euo pipefail

############################################
BASE_MODEL="google/gemma-2-9b-it"
TRAIN_PATH=""
VAL_PATH=""
OUTPUT_DIR=""

MAX_SEQ_LENGTH=512
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=8
GRAD_ACC_STEPS=1
EPOCHS=5

LR=3e-5

SAVE_STEPS=50
LOGGING_STEPS=1
EVAL_STEPS=50

BF16_FLAG="--bf16"
############################################

echo "▶ Starting fine-tuning Gemma-2-9b-it"
deepspeed --num_gpus 4 ft_gemma_full.py \
  --base_model    "${BASE_MODEL}" \
  --train_path    "${TRAIN_PATH}" \
  --val_path      "${VAL_PATH}" \
  --output_dir    "${OUTPUT_DIR}" \
  #--max_seq_length                 ${MAX_SEQ_LENGTH} \
  #--per_device_train_batch_size    ${TRAIN_BATCH_SIZE} \
  #--per_device_eval_batch_size     ${EVAL_BATCH_SIZE} \
  #--gradient_accumulation_steps    ${GRAD_ACC_STEPS} \
  #--num_train_epochs               ${EPOCHS} \
  #--learning_rate                  ${LR} \
 # --save_steps                     ${SAVE_STEPS} \
  #--logging_steps                  ${LOGGING_STEPS} \
  #--eval_steps                     ${EVAL_STEPS} \
 # ${BF16_FLAG}

echo "✅ Fine-tuning complete. Model saved to ${OUTPUT_DIR}"
