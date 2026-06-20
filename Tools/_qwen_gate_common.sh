# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${EDGE_QWEN_TOPK_MODEL:-}"
PYTHON_ENV="${EDGE_QWEN_TOPK_PYTHON_ENV:-}"
TOP_K="${EDGE_QWEN_TOPK_K:-5}"
MAX_ABS_DELTA="${EDGE_QWEN_TOPK_MAX_ABS_DELTA:-0.5}"
DECODE_STEPS="${EDGE_QWEN_TOPK_DECODE_STEPS:-${DEFAULT_DECODE_STEPS:-0}}"
MIN_STEP_TOKEN_OVERLAP="${EDGE_QWEN_TOPK_MIN_STEP_TOKEN_OVERLAP:-}"
PYTHON_BIN="${EDGE_QWEN_TOPK_PYTHON:-python3}"

if [[ -z "$MODEL_PATH" ]]; then
  printf 'Set EDGE_QWEN_TOPK_MODEL to a local Qwen model bundle.\n' >&2
  exit 2
fi

if [[ -n "$PYTHON_ENV" && -f "$PYTHON_ENV" ]]; then
  source "$PYTHON_ENV"
  PYTHON_BIN="${EDGE_QWEN_TOPK_PYTHON:-python}"
fi

qwen_append_optional_step_overlap() {
  if [[ -n "$MIN_STEP_TOKEN_OVERLAP" ]]; then
    COMMAND+=(--min-step-token-overlap "$MIN_STEP_TOKEN_OVERLAP")
  fi
}
