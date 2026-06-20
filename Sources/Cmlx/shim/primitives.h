// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

#pragma once

/// primitives — Stateless atomic compute operations.
///
/// These functions have ZERO model/session dependency. They accept raw arrays
/// and return results. Model-specific pipelines use these as building blocks.
///
/// Design rules:
/// - No session parameter. No tensor ID lookup. No state mutation.
/// - Small wrappers are inline. Heavy ops (conv, quantized matmul) go in .cpp.
/// - All functions live in namespace edge_cmlx::primitives.

#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/fast.h"

#include <optional>

namespace edge_cmlx::primitives {

using mlx::core::array;
using mlx::core::StreamOrDevice;
using mlx::core::Dtype;
using mlx::core::float16;
using mlx::core::float32;

inline array silu(const array& x, StreamOrDevice s) {
    return mlx::core::multiply(
        x,
        mlx::core::sigmoid(x, s),
        s);
}

inline array gelu_tanh(const array& x, StreamOrDevice s) {
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coeff = 0.044715f;
    auto x3 = mlx::core::multiply(
        mlx::core::multiply(x, x, s), x, s);
    auto inner = mlx::core::multiply(
        mlx::core::add(x, mlx::core::multiply(array(coeff, x.dtype()), x3, s), s),
        array(sqrt_2_over_pi, x.dtype()),
        s);
    auto tanh_val = mlx::core::tanh(inner, s);
    return mlx::core::multiply(
        mlx::core::multiply(array(0.5f, x.dtype()), x, s),
        mlx::core::add(array(1.0f, x.dtype()), tanh_val, s),
        s);
}

inline array astype_like(const array& x, const array& ref, StreamOrDevice s) {
    return x.dtype() == ref.dtype() ? x : mlx::core::astype(x, ref.dtype(), s);
}

inline array astype_like(const array& x, Dtype target, StreamOrDevice s) {
    return x.dtype() == target ? x : mlx::core::astype(x, target, s);
}

/// RMS Norm: x * weight / sqrt(mean(x^2) + eps)
array rms_norm(
    const array& input,
    const array& weight,
    float eps,
    StreamOrDevice s);

/// Layer Norm: (x - mean) / sqrt(var + eps) * weight + bias
array layer_norm(
    const array& input,
    const array& weight,
    const array* bias,
    float eps,
    StreamOrDevice s);

/// Float linear: input @ weight^T + optional bias
array linear(
    const array& input,
    const array& weight,
    const array* bias,
    StreamOrDevice s);

struct QuantizedWeight {
    const array& packed;
    const array& scales;
    const array& biases;
    int group_size;
    int bits;
};

/// Affine quantized linear: input @ dequant(weight)^T.
array quantized_linear(
    const array& input,
    const QuantizedWeight& weight,
    StreamOrDevice s);

/// Float embedding lookup: table[indices]
array embedding(
    const array& table,
    const array& indices,
    StreamOrDevice s);

/// Snake activation: x + (1/alpha) * sin^2(alpha * x)
/// Used by TTS speech decoder.
array snake(
    const array& x,
    const array& alpha,
    StreamOrDevice s);

/// Snake activation with separate alpha and beta:
/// x + sin^2(alpha * x) / (beta + epsilon).
array snake(
    const array& x,
    const array& alpha,
    const array& beta,
    StreamOrDevice s,
    float epsilon = 1e-9f);

}
