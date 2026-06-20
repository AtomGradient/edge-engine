#!/usr/bin/env bash
# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWIFT_BIN="${EDGE_PREFLIGHT_SWIFT:-swift}"
QWEN_MODEL="${EDGE_PREFLIGHT_QWEN_MODEL:-}"
ASR_MODEL="${EDGE_PREFLIGHT_ASR_MODEL:-}"
TTS_MODEL="${EDGE_PREFLIGHT_TTS_MODEL:-}"
QWEN_READ_SHARD_HEADERS="${EDGE_PREFLIGHT_QWEN_READ_SHARD_HEADERS:-0}"

require_path() {
  local variable_name="$1"
  local value="$2"
  if [[ -z "$value" ]]; then
    printf 'Set %s to a local model bundle.\n' "$variable_name" >&2
    exit 2
  fi
}

require_path EDGE_PREFLIGHT_QWEN_MODEL "$QWEN_MODEL"
require_path EDGE_PREFLIGHT_ASR_MODEL "$ASR_MODEL"
require_path EDGE_PREFLIGHT_TTS_MODEL "$TTS_MODEL"

is_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes) return 0 ;;
    *) return 1 ;;
  esac
}

qwen_header_args=()
if is_truthy "$QWEN_READ_SHARD_HEADERS"; then
  qwen_header_args=(--read-shard-headers true)
else
  qwen_header_args=(--no-read-shard-headers)
fi

printf '==> Qwen preflight: %s\n' "$QWEN_MODEL" >&2
(
  cd "$ROOT_DIR"
  "$SWIFT_BIN" run EdgeRuntimeQwenSmoke \
    --model "$QWEN_MODEL" \
    --preflight \
    "${qwen_header_args[@]}" \
    --require-pass
)

printf '==> Qwen3-ASR preflight: %s\n' "$ASR_MODEL" >&2
(
  cd "$ROOT_DIR"
  "$SWIFT_BIN" run EdgeRuntimeSpeechPreflight \
    --model "$ASR_MODEL" \
    --family qwen3-asr \
    --require-pass
)

printf '==> Qwen3-TTS preflight: %s\n' "$TTS_MODEL" >&2
(
  cd "$ROOT_DIR"
  "$SWIFT_BIN" run EdgeRuntimeSpeechPreflight \
    --model "$TTS_MODEL" \
    --family qwen3-tts \
    --require-pass
)
