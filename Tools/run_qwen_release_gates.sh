#!/usr/bin/env bash
# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GATES=(
  run_qwen_topk_gate.sh
  run_qwen_decode_gate.sh
  run_qwen_long_context_gate.sh
)

for gate in "${GATES[@]}"; do
  printf '==> %s\n' "$gate" >&2
  "$ROOT_DIR/Tools/$gate"
done
