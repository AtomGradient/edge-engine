#!/usr/bin/env python3
# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Small numeric oracle for EdgeRuntimeSwift smoke tests.

This script intentionally uses NumPy only. MLX can be used by later oracle
scripts, but this one stays lightweight so it can validate the configured
Python environment quickly.
"""

from __future__ import annotations

import json

import numpy as np


def main() -> None:
    lhs = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    rhs = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)
    add_lhs = np.array([[1, 2], [3, 4]], dtype=np.float32)
    add_rhs = np.array([[10, 20], [30, 40]], dtype=np.float32)

    result = {
        "matmul": np.matmul(lhs, rhs).reshape(-1).tolist(),
        "addition": (add_lhs + add_rhs).reshape(-1).tolist(),
    }

    assert result["matmul"] == [58.0, 64.0, 139.0, 154.0]
    assert result["addition"] == [11.0, 22.0, 33.0, 44.0]
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
