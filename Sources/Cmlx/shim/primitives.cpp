// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

#include "primitives.h"

#include "mlx/fast.h"
#include "mlx/ops.h"

#include <stdexcept>

namespace edge_cmlx::primitives {

using namespace mlx::core;


array rms_norm(
    const array& input,
    const array& weight,
    float eps,
    StreamOrDevice s) {
    return fast::rms_norm(input, weight, eps, s);
}

array layer_norm(
    const array& input,
    const array& weight,
    const array* bias,
    float eps,
    StreamOrDevice s) {
    if (bias != nullptr) {
        return fast::layer_norm(input, weight, *bias, eps, s);
    }
    return fast::layer_norm(
        input, weight, zeros_like(weight, s), eps, s);
}


array linear(
    const array& input,
    const array& weight,
    const array* bias,
    StreamOrDevice s) {
    const int input_rank = input.ndim();
    if (input_rank < 2) {
        throw std::runtime_error(
            "edge_cmlx::primitives::linear input must have rank >= 2");
    }

    auto typed_weight = astype_like(weight, input, s);
    auto output = matmul(
        input,
        transpose(typed_weight, {1, 0}, s),
        s);

    if (bias != nullptr) {
        auto typed_bias = astype_like(*bias, output, s);
        output = add(output, typed_bias, s);
    }
    return output;
}

array quantized_linear(
    const array& input,
    const QuantizedWeight& weight,
    StreamOrDevice s) {
    return quantized_matmul(
        input,
        weight.packed,
        weight.scales,
        weight.biases,
        true,
        weight.group_size,
        weight.bits,
        "affine",
        s);
}


array embedding(
    const array& table,
    const array& indices,
    StreamOrDevice s) {
    auto int_indices = indices.dtype() == int32
        ? indices
        : astype(indices, int32, s);
    return take(table, int_indices, 0, s);
}


array snake(
    const array& x,
    const array& alpha,
    StreamOrDevice s) {
    return snake(x, alpha, alpha, s, 0.0f);
}

array snake(
    const array& x,
    const array& alpha,
    const array& beta,
    StreamOrDevice s,
    float epsilon) {
    auto typed_alpha = astype_like(alpha, x, s);
    auto typed_beta = astype_like(beta, x, s);
    auto ax = multiply(typed_alpha, x, s);
    auto sin_ax = sin(ax, s);
    auto sin_sq = multiply(sin_ax, sin_ax, s);
    auto denominator = epsilon == 0.0f
        ? typed_beta
        : add(typed_beta, array(epsilon, x.dtype()), s);
    return add(
        x,
        divide(
            sin_sq,
            denominator,
            s),
        s);
}

}
