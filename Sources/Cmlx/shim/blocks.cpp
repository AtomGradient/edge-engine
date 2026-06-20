// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

#include "blocks.h"

#include "mlx/compile.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

#include <cmath>
#include <stdexcept>

namespace edge_cmlx::blocks {

using namespace mlx::core;

namespace {

std::vector<array> swiglu_activation_body(const std::vector<array>& inputs) {
    if (inputs.size() != 2) {
        throw std::runtime_error(
            "edge_cmlx::blocks::swiglu_activation expected two inputs");
    }
    const auto& gate = inputs[0];
    const auto& up = inputs[1];
    auto silu = multiply(gate, sigmoid(gate), {});
    return {multiply(silu, up, {})};
}

}

array swiglu_activation(
    const array& gate,
    const array& up,
    StreamOrDevice s) {
    StreamContext context(s);
    static const auto compiled_activation =
        compile(swiglu_activation_body, true);
    return compiled_activation({gate, up}).front();
}

array swiglu_mlp(
    const array& input,
    const primitives::QuantizedWeight& gate,
    const primitives::QuantizedWeight& up,
    const primitives::QuantizedWeight& down,
    const MLPConfig& config,
    StreamOrDevice s) {
    auto gate_output = primitives::quantized_linear(input, gate, s);
    auto up_output = primitives::quantized_linear(input, up, s);
    if (config.use_f32_activation) {
        gate_output = astype(gate_output, float32, s);
        up_output = astype(up_output, float32, s);
    }
    auto activation = swiglu_activation(gate_output, up_output, s);
    return primitives::quantized_linear(activation, down, s);
}

array projected_encoder_attention(
    const array& query,
    const array& key,
    const array& value,
    const array& output_projection,
    const array* output_bias,
    int heads,
    int head_dim,
    const std::optional<array>& attention_mask,
    StreamOrDevice s) {
    if (heads <= 0 || head_dim <= 0) {
        throw std::runtime_error(
            "edge_cmlx::blocks::projected_encoder_attention received invalid heads");
    }
    if (query.ndim() != key.ndim() || query.ndim() != value.ndim()) {
        throw std::runtime_error(
            "edge_cmlx::blocks::projected_encoder_attention rank mismatch");
    }
    const int hidden = heads * head_dim;
    const int rank = query.ndim();
    if (rank == 3) {
        const int token_count = static_cast<int>(query.shape(0));
        auto q = transpose(
            reshape(query, Shape{1, token_count, heads, head_dim}, s),
            {0, 2, 1, 3},
            s);
        auto k = transpose(
            reshape(key, Shape{1, token_count, heads, head_dim}, s),
            {0, 2, 1, 3},
            s);
        auto v = transpose(
            reshape(value, Shape{1, token_count, heads, head_dim}, s),
            {0, 2, 1, 3},
            s);
        auto attended = fast::scaled_dot_product_attention(
            q,
            k,
            v,
            std::pow(static_cast<float>(head_dim), -0.5f),
            "",
            attention_mask,
            std::nullopt,
            s);
        attended = reshape(
            transpose(attended, {0, 2, 1, 3}, s),
            Shape{token_count, hidden},
            s);
        if (output_bias != nullptr) {
            auto typed_weight = primitives::astype_like(
                output_projection,
                attended,
                s);
            auto typed_bias = primitives::astype_like(
                *output_bias,
                attended,
                s);
            return addmm(
                typed_bias,
                attended,
                transpose(typed_weight, {1, 0}, s),
                1.0f,
                1.0f,
                s);
        }
        return primitives::linear(
            attended,
            output_projection,
            output_bias,
            s);
    }
    if (rank == 4) {
        const int batch = static_cast<int>(query.shape(0));
        const int token_count = static_cast<int>(query.shape(1));
        auto q = transpose(query, {0, 2, 1, 3}, s);
        auto k = transpose(key, {0, 2, 1, 3}, s);
        auto v = transpose(value, {0, 2, 1, 3}, s);
        auto attended = fast::scaled_dot_product_attention(
            q,
            k,
            v,
            std::pow(static_cast<float>(head_dim), -0.5f),
            "",
            attention_mask,
            std::nullopt,
            s);
        attended = reshape(
            transpose(attended, {0, 2, 1, 3}, s),
            Shape{batch, token_count, hidden},
            s);
        return primitives::linear(
            attended,
            output_projection,
            output_bias,
            s);
    }
    throw std::runtime_error(
        "edge_cmlx::blocks::projected_encoder_attention expects rank 3 or 4 projections");
}

array encoder_attention(
    const array& input,
    const array& query_projection,
    const array& key_projection,
    const array& value_projection,
    const array& output_projection,
    const array* query_bias,
    const array* key_bias,
    const array* value_bias,
    const array* output_bias,
    int heads,
    int head_dim,
    StreamOrDevice s) {
    const int rank = input.ndim();
    if (rank != 2 && rank != 3) {
        throw std::runtime_error(
            "edge_cmlx::blocks::encoder_attention expects rank 2 or 3 input");
    }
    auto query = primitives::linear(
        input, query_projection, query_bias, s);
    auto key = primitives::linear(
        input, key_projection, key_bias, s);
    auto value = primitives::linear(
        input, value_projection, value_bias, s);
    if (rank == 2) {
        const int token_count = static_cast<int>(input.shape(0));
        query = reshape(query, Shape{token_count, heads, head_dim}, s);
        key = reshape(key, Shape{token_count, heads, head_dim}, s);
        value = reshape(value, Shape{token_count, heads, head_dim}, s);
    } else {
        const int batch = static_cast<int>(input.shape(0));
        const int token_count = static_cast<int>(input.shape(1));
        query = reshape(query, Shape{batch, token_count, heads, head_dim}, s);
        key = reshape(key, Shape{batch, token_count, heads, head_dim}, s);
        value = reshape(value, Shape{batch, token_count, heads, head_dim}, s);
    }
    return projected_encoder_attention(
        query,
        key,
        value,
        output_projection,
        output_bias,
        heads,
        head_dim,
        std::nullopt,
        s);
}

}
