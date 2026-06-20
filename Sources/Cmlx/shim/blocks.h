// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

#pragma once

/// blocks — Stateless reusable compute blocks.
///
/// Blocks compose primitives into model-family level patterns, but still avoid
/// session ownership, tensor lookup, and cache mutation.

#include "primitives.h"

#include <optional>
#include <vector>

namespace edge_cmlx::blocks {

using mlx::core::array;
using mlx::core::StreamOrDevice;

struct MLPConfig {
    bool use_f32_activation = false;
};

struct LinearWeight {
    std::optional<primitives::QuantizedWeight> quantized;
    const array* dense = nullptr;
};

array linear(
    const array& input,
    const LinearWeight& weight,
    StreamOrDevice s);

array swiglu_activation(
    const array& gate,
    const array& up,
    StreamOrDevice s);

array swiglu_mlp(
    const array& input,
    const primitives::QuantizedWeight& gate,
    const primitives::QuantizedWeight& up,
    const primitives::QuantizedWeight& down,
    const MLPConfig& config,
    StreamOrDevice s);

array projected_encoder_attention(
    const array& query,
    const array& key,
    const array& value,
    const array& output_projection,
    const array* output_bias,
    int heads,
    int head_dim,
    const std::optional<array>& attention_mask,
    StreamOrDevice s);

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
    StreamOrDevice s);

struct GDNConfig {
    int key_head_count;
    int value_head_count;
    int key_head_dimension;
    int value_head_dimension;
    int conv_kernel_size;
    float rms_norm_epsilon;
};

struct GDNWeights {
    primitives::QuantizedWeight qkv;
    primitives::QuantizedWeight z;
    primitives::QuantizedWeight a;
    primitives::QuantizedWeight b;
    const array& conv;
    const array& a_log;
    const array* neg_exp_a_log;
    const array& dt_bias;
    const array& norm;
    primitives::QuantizedWeight output;
};

struct GDNDecodeResult {
    array output;
    array next_conv_state;
    array next_recurrent_state;
};

GDNDecodeResult gdn_attention(
    const array& input,
    const array& conv_state,
    const array& recurrent_state,
    const GDNWeights& weights,
    const GDNConfig& config,
    bool log_dtype_diagnostic,
    StreamOrDevice s);

struct TransformerAttentionConfig {
    int attention_heads;
    int key_value_heads;
    int head_dim;
    int rotary_dimension;
    int query_projection_hidden;
    float rope_theta;
    float rms_norm_epsilon;
};

struct TransformerAttentionWeights {
    LinearWeight query;
    LinearWeight key;
    LinearWeight value;
    LinearWeight output;
    const array* query_norm;
    const array* key_norm;
};

struct TransformerAttentionProjection {
    array queries;
    array keys;
    array values;
    std::optional<array> gate;
    int token_count;
    int attention_hidden;
};

TransformerAttentionProjection transformer_attention_projection(
    const array& input,
    const TransformerAttentionWeights& weights,
    const TransformerAttentionConfig& config,
    int position_offset,
    StreamOrDevice s);

struct TransformerAttentionCacheView {
    const array* dense_keys = nullptr;
    const array* dense_values = nullptr;
    const array* quantized_key_packed = nullptr;
    const array* quantized_key_scales = nullptr;
    const array* quantized_key_biases = nullptr;
    const array* quantized_value_packed = nullptr;
    const array* quantized_value_scales = nullptr;
    const array* quantized_value_biases = nullptr;
    int quantized_group_size = 0;
    int quantized_bits = 0;
};

struct TransformerAttentionRuntime {
    bool causal = false;
    bool compute_scores = false;
    bool log_fused_check = false;
    int layer_index = 0;
};

struct TransformerAttentionResult {
    array output;
    std::optional<array> dsr_scores;
    bool fused_check_logged = false;
    bool fused_attention_used = false;
};

TransformerAttentionResult transformer_attention(
    const TransformerAttentionProjection& projection,
    const TransformerAttentionCacheView& cache,
    const TransformerAttentionWeights& weights,
    const TransformerAttentionConfig& config,
    const TransformerAttentionRuntime& runtime,
    StreamOrDevice s);

}
