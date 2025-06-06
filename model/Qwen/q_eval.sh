#!/usr/bin/env bash
export HF_HUB_OFFLINE=1

set -euo pipefail

############################################
MODEL_DIR=""
TEST_PATH=""
MAX_NEW_TOKENS=1024
TEMPERATURE=0.0
PREDICTIONS_OUT=""
METRICS_OUT=""
############################################

echo "▶ Running inference and evaluation"
python inference_and_evaluate.py \
  --model_dir       "${MODEL_DIR}" \
  --test_path       "${TEST_PATH}" \
  --max_new_tokens  ${MAX_NEW_TOKENS} \
  --temperature     ${TEMPERATURE} \
  --predictions_out "${PREDICTIONS_OUT}" \
  --metrics_out     "${METRICS_OUT}"

echo "✅ Inference complete"
echo "• Predictions: ${PREDICTIONS_OUT}"
echo "• Metrics:     ${METRICS_OUT}"
