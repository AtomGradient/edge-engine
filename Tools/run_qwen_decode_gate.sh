#!/usr/bin/env bash
# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

set -euo pipefail

DEFAULT_DECODE_STEPS=2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_qwen_gate_common.sh"

COMMAND=(
  "$PYTHON_BIN" "$ROOT_DIR/Tools/qwen_prefill_topk_compare.py"
  --model "$MODEL_PATH"
  --case single:0
  --case seq4:0,1,2,3
  --top-k "$TOP_K"
  --max-abs-delta "$MAX_ABS_DELTA"
  --decode-steps "$DECODE_STEPS"
)

qwen_append_optional_step_overlap

exec "${COMMAND[@]}"
