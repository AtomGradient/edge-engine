#!/usr/bin/env bash
# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

set -euo pipefail

PYTHON_ENV="${EDGE_PYTHON_ORACLE_ENV:-}"
PYTHON_BIN="${EDGE_PYTHON_ORACLE_PYTHON:-python3}"

if [[ -n "$PYTHON_ENV" ]]; then
  source "$PYTHON_ENV"
  PYTHON_BIN="${EDGE_PYTHON_ORACLE_PYTHON:-python}"
fi

"$PYTHON_BIN" "$(dirname "$0")/python_oracle_smoke.py"
