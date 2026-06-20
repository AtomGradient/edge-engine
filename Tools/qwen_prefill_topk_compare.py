#!/usr/bin/env python3
# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Compare native Qwen smoke top-k logits against mlx-lm.

This tool compares the final prefill row and, optionally, greedy decode-step
top-k entries. It is a narrow regression gate for native kernel/layout changes
without dumping the full vocabulary logits.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TopLogit:
    token_id: int
    logit: float


@dataclass(frozen=True)
class ComparisonCase:
    label: str
    token_ids: list[int]


def parse_token_ids(value: str) -> list[int]:
    try:
        token_ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid token list: {value}") from error
    if not token_ids:
        raise argparse.ArgumentTypeError("token list must not be empty")
    return token_ids


def parse_case(value: str) -> ComparisonCase:
    if ":" in value:
        label, raw_token_ids = value.split(":", maxsplit=1)
        label = label.strip()
        if not label:
            raise argparse.ArgumentTypeError("case label must not be empty")
    else:
        raw_token_ids = value
        label = value
    token_ids = parse_token_ids(raw_token_ids)
    return ComparisonCase(label=label, token_ids=token_ids)


def run_native(
    args: argparse.Namespace,
    package_root: Path,
    token_ids: list[int],
) -> dict[str, Any]:
    command = [
        "swift",
        "run",
        "EdgeRuntimeQwenSmoke",
        "--model",
        args.model,
        "--tokens",
        ",".join(str(token_id) for token_id in token_ids),
        "--max-new-tokens",
        str(args.decode_steps),
        "--prefill-top-logits",
        str(args.top_k),
        "--step-top-logits",
        str(args.top_k),
        "--max-ops-per-buffer",
        str(args.max_ops_per_buffer),
        "--max-mb-per-buffer",
        str(args.max_mb_per_buffer),
        "--release-quantized-host-storage",
        "true" if args.release_quantized_host_storage else "false",
    ]
    if args.kv_capacity is not None:
        command.extend(["--kv-capacity", str(args.kv_capacity)])
    if args.quantized_cache_mb is not None:
        command.extend(["--quantized-cache-mb", str(args.quantized_cache_mb)])

    completed = subprocess.run(
        command,
        cwd=package_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if args.show_native_stderr and completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    return json.loads(completed.stdout)


def load_mlx_model(model_path: str) -> tuple[Any, Any, Any]:
    try:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
    except ImportError as error:
        raise RuntimeError(
            "mlx-lm is not importable. Activate the EdgeStudio Python env first."
        ) from error

    model, _ = load(model_path, lazy=False)
    return mx, model, make_prompt_cache


def run_mlx(
    mx: Any,
    model: Any,
    make_prompt_cache: Any,
    token_ids: list[int],
    top_k: int,
    decode_steps: int,
) -> tuple[list[TopLogit], list[list[TopLogit]], list[int]]:
    cache = make_prompt_cache(model)
    inputs = mx.array([token_ids], dtype=mx.int32)
    logits = model(inputs, cache=cache)
    eval_logits_and_cache(mx=mx, logits=logits, cache=cache)
    prefill_top_logits = top_logits_for_row(mx=mx, logits_row=logits[0, -1], top_k=top_k)
    step_top_logits: list[list[TopLogit]] = []
    generated_token_ids: list[int] = []
    for _ in range(decode_steps):
        step_top_logits.append(top_logits_for_row(mx=mx, logits_row=logits[0, -1], top_k=top_k))
        token_id = step_top_logits[-1][0].token_id
        generated_token_ids.append(token_id)
        inputs = mx.array([[token_id]], dtype=mx.int32)
        logits = model(inputs, cache=cache)
        eval_logits_and_cache(mx=mx, logits=logits, cache=cache)
    return prefill_top_logits, step_top_logits, generated_token_ids


def eval_logits_and_cache(mx: Any, logits: Any, cache: list[Any]) -> None:
    mx.eval(logits, [layer_cache.state for layer_cache in cache])


def top_logits_for_row(mx: Any, logits_row: Any, top_k: int) -> list[TopLogit]:
    token_indices = mx.argsort(-logits_row)[:top_k]
    values = logits_row[token_indices]
    mx.eval(values, token_indices)
    return [
        TopLogit(token_id=int(token_id), logit=float(logit))
        for token_id, logit in zip(token_indices.tolist(), values.tolist())
    ]


def native_top_logits(result: dict[str, Any]) -> list[TopLogit]:
    return [
        TopLogit(token_id=int(entry["tokenId"]), logit=float(entry["logit"]))
        for entry in result["prefillTopLogits"]
    ]


def native_step_top_logits(result: dict[str, Any]) -> list[list[TopLogit]]:
    return [
        [
            TopLogit(token_id=int(entry["tokenId"]), logit=float(entry["logit"]))
            for entry in step["topLogits"]
        ]
        for step in result["steps"]
    ]


def compare(
    native: list[TopLogit],
    mlx: list[TopLogit],
    max_abs_delta: float,
    strict_rank: bool,
    min_token_overlap: int | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    if strict_rank:
        return compare_by_rank(native=native, mlx=mlx, max_abs_delta=max_abs_delta)
    return compare_by_token(
        native=native,
        mlx=mlx,
        max_abs_delta=max_abs_delta,
        min_token_overlap=min_token_overlap,
    )


def compare_by_rank(
    native: list[TopLogit],
    mlx: list[TopLogit],
    max_abs_delta: float,
) -> tuple[bool, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    ok = len(native) == len(mlx)
    for rank, (native_entry, mlx_entry) in enumerate(zip(native, mlx), start=1):
        token_matches = native_entry.token_id == mlx_entry.token_id
        delta = abs(native_entry.logit - mlx_entry.logit)
        row_ok = token_matches and delta <= max_abs_delta
        ok = ok and row_ok
        rows.append(
            {
                "rank": rank,
                "nativeTokenId": native_entry.token_id,
                "mlxTokenId": mlx_entry.token_id,
                "nativeLogit": native_entry.logit,
                "mlxLogit": mlx_entry.logit,
                "absDelta": delta,
                "ok": row_ok,
            }
        )
    if len(native) != len(mlx):
        rows.append(
            {
                "rank": None,
                "nativeCount": len(native),
                "mlxCount": len(mlx),
                "ok": False,
            }
        )
    return ok, rows


def compare_by_token(
    native: list[TopLogit],
    mlx: list[TopLogit],
    max_abs_delta: float,
    min_token_overlap: int | None,
) -> tuple[bool, list[dict[str, Any]]]:
    native_by_token = {entry.token_id: entry for entry in native}
    mlx_by_token = {entry.token_id: entry for entry in mlx}
    native_ranks = {entry.token_id: rank for rank, entry in enumerate(native, start=1)}
    mlx_ranks = {entry.token_id: rank for rank, entry in enumerate(mlx, start=1)}
    token_ids = sorted(
        set(native_by_token) | set(mlx_by_token),
        key=lambda token_id: (
            mlx_ranks.get(token_id, sys.maxsize),
            native_ranks.get(token_id, sys.maxsize),
            token_id,
        ),
    )

    rows: list[dict[str, Any]] = []
    required_overlap = min_token_overlap if min_token_overlap is not None else len(native)
    required_overlap = max(0, min(required_overlap, len(native), len(mlx)))
    matching_token_count = 0
    has_bad_shared_token = False
    for token_id in token_ids:
        native_entry = native_by_token.get(token_id)
        mlx_entry = mlx_by_token.get(token_id)
        token_present = native_entry is not None and mlx_entry is not None
        delta = (
            abs(native_entry.logit - mlx_entry.logit)
            if native_entry is not None and mlx_entry is not None
            else None
        )
        row_ok = token_present and delta is not None and delta <= max_abs_delta
        if row_ok:
            matching_token_count += 1
        elif token_present:
            has_bad_shared_token = True
        rows.append(
            {
                "tokenId": token_id,
                "nativeRank": native_ranks.get(token_id),
                "mlxRank": mlx_ranks.get(token_id),
                "nativeLogit": native_entry.logit if native_entry is not None else None,
                "mlxLogit": mlx_entry.logit if mlx_entry is not None else None,
                "absDelta": delta,
                "ok": row_ok,
            }
        )
    ok = (
        len(native) == len(mlx)
        and not has_bad_shared_token
        and matching_token_count >= required_overlap
    )
    return ok, rows


def compare_case(
    args: argparse.Namespace,
    package_root: Path,
    mx: Any,
    mlx_model: Any,
    make_prompt_cache: Any,
    case: ComparisonCase,
) -> dict[str, Any]:
    native_result = run_native(
        args=args,
        package_root=package_root,
        token_ids=case.token_ids,
    )
    native = native_top_logits(native_result)
    native_steps = native_step_top_logits(native_result)
    mlx, mlx_steps, mlx_generated_token_ids = run_mlx(
        mx=mx,
        model=mlx_model,
        make_prompt_cache=make_prompt_cache,
        token_ids=case.token_ids,
        top_k=args.top_k,
        decode_steps=args.decode_steps,
    )
    ok, rows = compare(
        native=native,
        mlx=mlx,
        max_abs_delta=args.max_abs_delta,
        strict_rank=args.strict_rank,
    )
    required_step_overlap = args.min_step_token_overlap
    if required_step_overlap is None:
        required_step_overlap = max(1, args.top_k - 1)
    required_step_overlap = min(required_step_overlap, args.top_k)
    step_summaries: list[dict[str, Any]] = []
    for index, (native_step, mlx_step) in enumerate(zip(native_steps, mlx_steps)):
        step_ok, step_rows = compare(
            native=native_step,
            mlx=mlx_step,
            max_abs_delta=args.max_abs_delta,
            strict_rank=args.strict_rank,
            min_token_overlap=required_step_overlap,
        )
        matched_token_count = sum(1 for row in step_rows if row["ok"])
        step_summaries.append(
            {
                "ok": step_ok,
                "index": index,
                "nativeTokenId": native_result["steps"][index]["tokenId"],
                "mlxTokenId": mlx_generated_token_ids[index],
                "tokenMatch": native_result["steps"][index]["tokenId"]
                == mlx_generated_token_ids[index],
                "matchedTokenCount": matched_token_count,
                "requiredTokenOverlap": required_step_overlap,
                "comparison": step_rows,
            }
        )
        step_summaries[-1]["ok"] = step_summaries[-1]["ok"] and step_summaries[-1]["tokenMatch"]
    if len(native_steps) != len(mlx_steps):
        step_summaries.append(
            {
                "ok": False,
                "index": None,
                "nativeStepCount": len(native_steps),
                "mlxStepCount": len(mlx_steps),
                "comparison": [],
            }
        )
    ok = ok and all(step["ok"] for step in step_summaries)
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()

    return {
        "ok": ok,
        "label": case.label,
        "tokens": case.token_ids,
        "topK": args.top_k,
        "decodeSteps": args.decode_steps,
        "minStepTokenOverlap": required_step_overlap,
        "strictRank": args.strict_rank,
        "maxAbsDelta": args.max_abs_delta,
        "prefillLogitsShape": native_result.get("prefillLogitsShape"),
        "generatedTokenIds": native_result.get("generatedTokenIds"),
        "mlxGeneratedTokenIds": mlx_generated_token_ids,
        "runtime": native_result.get("runtime"),
        "comparison": rows,
        "prefillComparison": rows,
        "stepComparisons": step_summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a local Qwen MLX bundle.")
    parser.add_argument(
        "--tokens",
        type=parse_token_ids,
        default=[0],
        help="Comma-separated prompt token ids for single-case mode. Default: 0",
    )
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        dest="cases",
        help=(
            "Add a comparison case. Use '<label>:<id,id,...>' or just '<id,id,...>'. "
            "May be repeated. When present, --tokens is ignored."
        ),
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top logits to compare. Default: 5")
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=0,
        help="Greedy decode steps to compare after prefill. Default: 0",
    )
    parser.add_argument(
        "--min-step-token-overlap",
        type=int,
        help=(
            "Minimum matching top-k tokens required per decode step. "
            "Default: top-k - 1, clamped to at least 1."
        ),
    )
    parser.add_argument(
        "--max-abs-delta",
        type=float,
        default=0.5,
        help="Allowed per-logit absolute delta. Default: 0.5",
    )
    parser.add_argument("--max-ops-per-buffer", type=int, default=3)
    parser.add_argument("--max-mb-per-buffer", type=int, default=64)
    parser.add_argument("--kv-capacity", type=int)
    parser.add_argument("--quantized-cache-mb", type=int)
    parser.add_argument(
        "--strict-rank",
        action="store_true",
        help="Require exact rank-by-rank token matches instead of top-k set matches.",
    )
    parser.add_argument(
        "--no-release-quantized-host-storage",
        dest="release_quantized_host_storage",
        action="store_false",
        help="Keep native quantized host arrays after upload.",
    )
    parser.add_argument("--show-native-stderr", action="store_true")
    parser.set_defaults(release_quantized_host_storage=True)
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.max_abs_delta < 0:
        parser.error("--max-abs-delta must be non-negative")
    if args.decode_steps < 0:
        parser.error("--decode-steps must be non-negative")
    if args.min_step_token_overlap is not None and args.min_step_token_overlap < 0:
        parser.error("--min-step-token-overlap must be non-negative")
    min_step_token_overlap = args.min_step_token_overlap
    if min_step_token_overlap is None:
        min_step_token_overlap = max(1, args.top_k - 1)
    min_step_token_overlap = min(min_step_token_overlap, args.top_k)

    package_root = Path(__file__).resolve().parents[1]
    cases = args.cases or [
        ComparisonCase(
            label="tokens",
            token_ids=args.tokens,
        )
    ]
    mx, mlx_model, make_prompt_cache = load_mlx_model(args.model)
    case_summaries = [
        compare_case(
            args=args,
            package_root=package_root,
            mx=mx,
            mlx_model=mlx_model,
            make_prompt_cache=make_prompt_cache,
            case=case,
        )
        for case in cases
    ]
    ok = all(case["ok"] for case in case_summaries)

    if args.cases:
        summary = {
            "ok": ok,
            "model": args.model,
            "caseCount": len(case_summaries),
            "topK": args.top_k,
            "decodeSteps": args.decode_steps,
            "minStepTokenOverlap": min_step_token_overlap,
            "strictRank": args.strict_rank,
            "maxAbsDelta": args.max_abs_delta,
            "cases": case_summaries,
        }
    else:
        summary = dict(case_summaries[0])
        summary["model"] = args.model
        summary.pop("label", None)

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
