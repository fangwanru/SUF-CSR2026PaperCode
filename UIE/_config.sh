#!/usr/bin/env bash
# SUF-CSR UIE experiment configuration

set -euo pipefail

UIE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUF_ROOT="$(cd "${UIE_DIR}/.." && pwd)"

export PYTHONPATH="${UIE_DIR}/UIE:${UIE_DIR}:${PYTHONPATH:-}"

CITRUS_NER_DIR="${SUF_ROOT}/CitrusNER"
CITRUS_RE_DIR="${SUF_ROOT}/CitrusRE"
OUTPUT_ROOT="${UIE_DIR}/outputs"
CITRUS_OUT="${OUTPUT_ROOT}/citrus"
CHECKPOINT_ROOT="${UIE_DIR}/checkpoints"

UIE_BASE_MODEL="${UIE_BASE_MODEL:-${CHECKPOINT_ROOT}/uie-base-en}"
SUF_NER_MODEL="${SUF_NER_MODEL:-${CHECKPOINT_ROOT}/SUF-NER/model_best}"
SUF_RE_MODEL="${SUF_RE_MODEL:-${CHECKPOINT_ROOT}/SUF-RE/model_best}"

UIE_ROOT="${UIE_DIR}/UIE"
EVAL_SCRIPT="${UIE_ROOT}/scripts/eval_extraction.py"
CONVERT_SCRIPT="${UIE_DIR}/convert_for_uie_eval.py"

NER_TEST="${CITRUS_NER_DIR}/test.txt"
RE_TEST="${CITRUS_RE_DIR}/test.txt"
RE_SENTENCES="${CITRUS_RE_DIR}/test_sentences.txt"
GOLD_UIE="${CITRUS_OUT}/uie_official/gold/test.json"

check_model_dir() {
  local label="$1"
  local path="$2"
  if [[ -d "${path}" && -f "${path}/pytorch_model.bin" ]]; then
    return 0
  fi
  echo "错误: ${label} 模型不存在: ${path}" >&2
  echo "请下载 UIE-base 或将微调权重放入 checkpoints/，见 UIE/checkpoints/README.md" >&2
  return 1
}
