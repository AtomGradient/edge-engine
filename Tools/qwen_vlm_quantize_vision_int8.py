#!/usr/bin/env python3
"""Build a validation Qwen VLM bundle with an INT8 vision tower.

The Qwen3.5 9B 4-bit bundle keeps the decoder quantized but stores the vision
tower as BF16. This tool creates a separate validation bundle where every
vision tensor is served from one new safetensors shard:

- linear vision weights become MLX affine INT8 packed weights with scales/biases
- non-linear vision tensors stay float32
- decoder shards and config/tokenizer files are hard-linked from the source

The source model directory is never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import torch
from safetensors import safe_open


VISION_SHARD_NAME = "model-vision-int8-g64.safetensors"
VISION_PREFIX = "vision_tower."
DTYPE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass
class TensorHeader:
    shape: list[int]
    dtype: str
    file_name: str

    @property
    def byte_count(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count * DTYPE_SIZES[self.dtype]


@dataclass
class QuantizationPlan:
    vision_count: int
    quantized_weight_count: int
    float_tensor_count: int
    original_vision_bytes: int
    estimated_vision_bytes: int
    padded_columns: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Qwen3.5 VLM validation bundle with INT8 vision weights."
    )
    parser.add_argument("--model", required=True, type=Path, help="Source model directory.")
    parser.add_argument("--output", required=True, type=Path, help="Validation output directory.")
    parser.add_argument("--group-size", type=int, default=64, help="MLX affine quantization group size.")
    parser.add_argument("--bits", type=int, default=8, help="MLX affine quantization bits.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the validation bundle. Without this flag, only print the plan.",
    )
    return parser.parse_args()


def load_index(model_dir: Path) -> dict[str, Any]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing model index: {index_path}")
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    if not isinstance(index.get("weight_map"), dict):
        raise ValueError("model index has no weight_map")
    return index


def load_headers(model_dir: Path, weight_map: dict[str, str]) -> dict[str, TensorHeader]:
    headers: dict[str, TensorHeader] = {}
    for file_name in sorted(set(weight_map.values())):
        path = model_dir / file_name
        with safe_open(path, framework="pt", device="cpu") as shard:
            for name in shard.keys():
                slice_view = shard.get_slice(name)
                headers[name] = TensorHeader(
                    shape=list(slice_view.get_shape()),
                    dtype=slice_view.get_dtype(),
                    file_name=file_name,
                )
    missing = sorted(set(weight_map).difference(headers))
    if missing:
        raise ValueError(f"{len(missing)} tensors are listed in index but missing from shards")
    return headers


def is_vision_tensor(name: str) -> bool:
    return name.startswith(VISION_PREFIX)


def is_quantizable_vision_weight(name: str, shape: list[int]) -> bool:
    if name == "vision_tower.patch_embed.proj.weight":
        return len(shape) == 5
    suffixes = (
        ".attn.qkv.weight",
        ".attn.proj.weight",
        ".mlp.linear_fc1.weight",
        ".mlp.linear_fc2.weight",
        ".merger.linear_fc1.weight",
        ".merger.linear_fc2.weight",
    )
    return len(shape) == 2 and name.endswith(suffixes)


def logical_linear_shape(name: str, shape: list[int]) -> tuple[int, int]:
    if name == "vision_tower.patch_embed.proj.weight":
        rows = shape[0]
        columns = 1
        for dim in shape[1:]:
            columns *= dim
        return rows, columns
    return shape[0], shape[1]


def padded_columns(columns: int, group_size: int) -> int:
    remainder = columns % group_size
    return columns if remainder == 0 else columns + group_size - remainder


def make_plan(headers: dict[str, TensorHeader], group_size: int, bits: int) -> QuantizationPlan:
    if group_size <= 0 or bits <= 0:
        raise ValueError("group-size and bits must be positive")
    vision_headers = {
        name: header
        for name, header in headers.items()
        if is_vision_tensor(name)
    }
    padded: dict[str, int] = {}
    quantized_count = 0
    estimated_bytes = 0
    original_bytes = 0
    for name, header in sorted(vision_headers.items()):
        original_bytes += header.byte_count
        if is_quantizable_vision_weight(name, header.shape):
            rows, columns = logical_linear_shape(name, header.shape)
            padded_cols = padded_columns(columns, group_size)
            if padded_cols != columns:
                padded[name] = padded_cols - columns
            packed_cols = padded_cols * bits // 32
            scale_cols = padded_cols // group_size
            estimated_bytes += rows * packed_cols * DTYPE_SIZES["U32"]
            estimated_bytes += rows * scale_cols * DTYPE_SIZES["F32"] * 2
            quantized_count += 1
        else:
            count = 1
            for dim in header.shape:
                count *= dim
            estimated_bytes += count * DTYPE_SIZES["F32"]
    return QuantizationPlan(
        vision_count=len(vision_headers),
        quantized_weight_count=quantized_count,
        float_tensor_count=len(vision_headers) - quantized_count,
        original_vision_bytes=original_bytes,
        estimated_vision_bytes=estimated_bytes,
        padded_columns=padded,
    )


def tensor_from_shard(model_dir: Path, header: TensorHeader, name: str) -> torch.Tensor:
    with safe_open(model_dir / header.file_name, framework="pt", device="cpu") as shard:
        return shard.get_tensor(name)


def to_mlx_float32(tensor: torch.Tensor) -> mx.array:
    return mx.array(tensor.to(dtype=torch.float32).numpy(), dtype=mx.float32)


def quantize_weight(
    name: str,
    tensor: torch.Tensor,
    group_size: int,
    bits: int,
) -> dict[str, mx.array]:
    rows, columns = logical_linear_shape(name, list(tensor.shape))
    weight = to_mlx_float32(tensor).reshape((rows, columns))
    target_columns = padded_columns(columns, group_size)
    if target_columns != columns:
        padding = mx.zeros((rows, target_columns - columns), dtype=weight.dtype)
        weight = mx.concatenate([weight, padding], axis=1)
    packed, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)
    mx.eval(packed, scales, biases)
    base = name[:-len(".weight")]
    return {
        name: packed,
        f"{base}.scales": scales,
        f"{base}.biases": biases,
    }


def hardlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_output_dir(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    for item in source.iterdir():
        if not item.is_file():
            continue
        if item.name == "model.safetensors.index.json":
            continue
        hardlink_or_copy(item, output / item.name)


def write_bundle(
    model_dir: Path,
    output_dir: Path,
    index: dict[str, Any],
    headers: dict[str, TensorHeader],
    group_size: int,
    bits: int,
) -> None:
    prepare_output_dir(model_dir, output_dir)
    arrays: dict[str, mx.array] = {}
    new_weight_map = dict(index["weight_map"])
    for name, header in sorted(headers.items()):
        if not is_vision_tensor(name):
            continue
        tensor = tensor_from_shard(model_dir, header, name)
        if is_quantizable_vision_weight(name, header.shape):
            entries = quantize_weight(name, tensor, group_size=group_size, bits=bits)
            arrays.update(entries)
            for entry_name in entries:
                new_weight_map[entry_name] = VISION_SHARD_NAME
        else:
            value = to_mlx_float32(tensor)
            mx.eval(value)
            arrays[name] = value
            new_weight_map[name] = VISION_SHARD_NAME
        if len(arrays) % 48 == 0:
            print(f"prepared {len(arrays)} vision tensors/companions", flush=True)

    mx.save_safetensors(
        output_dir / VISION_SHARD_NAME,
        arrays,
        metadata={
            "format": "mlx",
            "edge_vision_quantization": f"int{bits}_group{group_size}",
        },
    )
    new_index = dict(index)
    new_index["weight_map"] = dict(sorted(new_weight_map.items()))
    original_total = int(index.get("metadata", {}).get("total_size", 0) or 0)
    if original_total > 0:
        original_vision = sum(
            header.byte_count
            for name, header in headers.items()
            if is_vision_tensor(name)
        )
        new_vision = sum(array.nbytes for array in arrays.values())
        new_index["metadata"] = dict(index.get("metadata", {}))
        new_index["metadata"]["total_size"] = original_total - original_vision + new_vision
    with (output_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as handle:
        json.dump(new_index, handle, indent=2, sort_keys=True)
        handle.write("\n")


def print_plan(plan: QuantizationPlan, output: Path, write: bool) -> None:
    mb = 1024 * 1024
    print("Qwen VLM vision INT8 validation plan")
    print(f"  output: {output}")
    print(f"  vision tensors: {plan.vision_count}")
    print(f"  quantized weights: {plan.quantized_weight_count}")
    print(f"  float tensors: {plan.float_tensor_count}")
    print(f"  original vision bytes: {plan.original_vision_bytes / mb:.1f} MB")
    print(f"  estimated vision bytes: {plan.estimated_vision_bytes / mb:.1f} MB")
    print(f"  estimated saving: {(plan.original_vision_bytes - plan.estimated_vision_bytes) / mb:.1f} MB")
    print(f"  padded weights: {len(plan.padded_columns)}")
    for name, added in list(plan.padded_columns.items())[:8]:
        print(f"    {name}: +{added} cols")
    if len(plan.padded_columns) > 8:
        print(f"    ... {len(plan.padded_columns) - 8} more")
    print(f"  mode: {'write' if write else 'dry-run'}")


def main() -> int:
    args = parse_args()
    model_dir = args.model.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    index = load_index(model_dir)
    headers = load_headers(model_dir, index["weight_map"])
    plan = make_plan(headers, group_size=args.group_size, bits=args.bits)
    print_plan(plan, output_dir, args.write)
    if not args.write:
        return 0
    write_bundle(
        model_dir,
        output_dir,
        index,
        headers,
        group_size=args.group_size,
        bits=args.bits,
    )
    print(f"wrote validation bundle: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
