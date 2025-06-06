#!/usr/bin/env bash
export HF_HUB_OFFLINE=1

set -euo pipefail

############################################
MODEL_DIR=""
TEST_PATH=""
HF_TOKEN="hf_"
MAX_NEW_TOKENS=1024
TEMPERATURE=0.0
PREDICTIONS_OUT=""
METRICS_OUT=""
############################################

echo "▶ Running inference and evaluation on LLaMA-3.1"
python inference_llama3_evaluate.py \
  --model_dir       "${MODEL_DIR}" \
  --test_path       "${TEST_PATH}" \
  --hf_token        "${HF_TOKEN}" \
  --max_new_tokens  ${MAX_NEW_TOKENS} \
  --temperature     ${TEMPERATURE} \
  --predictions_out "${PREDICTIONS_OUT}" \
  --metrics_out     "${METRICS_OUT}"

echo "✅ Inference complete"
echo "• Predictions: ${PREDICTIONS_OUT}"
echo "• Metrics:     ${METRICS_OUT}"
