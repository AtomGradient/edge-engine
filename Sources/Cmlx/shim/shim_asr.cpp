// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

#include "blocks.h"
#include "shim_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mlx/fast.h"
#include "mlx/io.h"
#include "mlx/memory.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace edge_cmlx::detail {

int qwen35_audio_conv2d1_weight_id() {
  return 300000;
}

int qwen35_audio_conv2d1_bias_id() {
  return 300001;
}

int qwen35_audio_conv2d2_weight_id() {
  return 300002;
}

int qwen35_audio_conv2d2_bias_id() {
  return 300003;
}

int qwen35_audio_conv2d3_weight_id() {
  return 300004;
}

int qwen35_audio_conv2d3_bias_id() {
  return 300005;
}

int qwen35_audio_conv_out_weight_id() {
  return 300006;
}

int qwen35_audio_ln_post_weight_id() {
  return 300010;
}

int qwen35_audio_ln_post_bias_id() {
  return 300011;
}

int qwen35_audio_proj1_weight_id() {
  return 300012;
}

int qwen35_audio_proj1_bias_id() {
  return 300013;
}

int qwen35_audio_proj2_weight_id() {
  return 300014;
}

int qwen35_audio_proj2_bias_id() {
  return 300015;
}

int qwen35_audio_layer_tensor_id(int layer_index, int offset) {
  return 310000 + layer_index * 100 + offset;
}

int qwen35_audio_layer_self_attn_ln_weight_id(int layer_index) {
  return qwen35_audio_layer_tensor_id(layer_index, 1);
}

int qwen35_audio_layer_self_attn_ln_bias_id(int layer_index) {
  return qwen35_audio_layer_tensor_id(layer_index, 2);
}

int qwen35_audio_layer_final_ln_weight_id(int layer_index) {
  return qwen35_audio_layer_tensor_id(layer_index, 3);
}

int qwen35_audio_layer_final_ln_bias_id(int layer_index) {
  return qwen35_audio_layer_tensor_id(layer_index, 4);
}

int qwen35_audio_layer_q_proj_weight_id(int layer_index) {
  return qwen35_audio_layer_tensor_id(layer_index, 10);
}

int qwen35_audio_layer_q_proj_bias_id(int layer_index) {
  return qwen35_audio_layer_tensor_id(layer_index, 11);
}

int qwen35_audio_layer_k_proj_weight_id(int layer_index) {
  return qwen35_audio_layer_tensor_id(layer_index, 12);
}

int qwen35_audio_layer_k_proj_bias_id(int layer_index) {
  return qwen35_audio_layer_tensor_id(layer_index, 13);
}

int qwen35_audio_layer_v_proj_weight_id(int layer_index) {
  return qwen35_audio_layer_tensor_id(layer_index, 14);
}

int qwen35_audio_layer_v_proj_bias_id(int layer_index) {
  return qwen35_audio_layer_tensor_id(layer_index, 15);
}

int qwen35_audio_layer_out_proj_weight_id(int layer_index) {
  return qwen35_audio_layer_tensor_id(layer_index, 16);
}

int qwen35_audio_layer_out_proj_bias_id(int layer_index) {
  return qwen35_audio_layer_tensor_id(layer_index, 17);
}

int qwen35_audio_layer_fc1_weight_id(int layer_index) {
  return qwen35_audio_layer_tensor_id(layer_index, 30);
}

int qwen35_audio_layer_fc1_bias_id(int layer_index) {
  return qwen35_audio_layer_tensor_id(layer_index, 31);
}

int qwen35_audio_layer_fc2_weight_id(int layer_index) {
  return qwen35_audio_layer_tensor_id(layer_index, 32);
}

int qwen35_audio_layer_fc2_bias_id(int layer_index) {
  return qwen35_audio_layer_tensor_id(layer_index, 33);
}

void register_qwen35_audio_tensors(
    EdgeCmlxQwen35Session& session,
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    const std::string& prefix) {
  const auto& config = checked_qwen35_audio_config(session);
  register_loaded_vision_float_tensor(
      session,
      qwen35_audio_conv2d1_weight_id(),
      tensors,
      prefix + ".conv2d1.weight");
  register_loaded_vision_float_tensor(
      session,
      qwen35_audio_conv2d1_bias_id(),
      tensors,
      prefix + ".conv2d1.bias");
  register_loaded_vision_float_tensor(
      session,
      qwen35_audio_conv2d2_weight_id(),
      tensors,
      prefix + ".conv2d2.weight");
  register_loaded_vision_float_tensor(
      session,
      qwen35_audio_conv2d2_bias_id(),
      tensors,
      prefix + ".conv2d2.bias");
  register_loaded_vision_float_tensor(
      session,
      qwen35_audio_conv2d3_weight_id(),
      tensors,
      prefix + ".conv2d3.weight");
  register_loaded_vision_float_tensor(
      session,
      qwen35_audio_conv2d3_bias_id(),
      tensors,
      prefix + ".conv2d3.bias");
  register_loaded_vision_float_tensor(
      session,
      qwen35_audio_conv_out_weight_id(),
      tensors,
      prefix + ".conv_out.weight");
  for (int layer = 0; layer < config.encoder_layers; ++layer) {
    const auto layer_prefix =
        prefix + ".layers." + std::to_string(layer);
    register_loaded_vision_float_tensor(
        session,
        qwen35_audio_layer_self_attn_ln_weight_id(layer),
        tensors,
        layer_prefix + ".self_attn_layer_norm.weight");
    register_loaded_vision_float_tensor(
        session,
        qwen35_audio_layer_self_attn_ln_bias_id(layer),
        tensors,
        layer_prefix + ".self_attn_layer_norm.bias");
    register_loaded_vision_float_tensor(
        session,
        qwen35_audio_layer_final_ln_weight_id(layer),
        tensors,
        layer_prefix + ".final_layer_norm.weight");
    register_loaded_vision_float_tensor(
        session,
        qwen35_audio_layer_final_ln_bias_id(layer),
        tensors,
        layer_prefix + ".final_layer_norm.bias");
    register_loaded_vision_float_tensor(
        session,
        qwen35_audio_layer_q_proj_weight_id(layer),
        tensors,
        layer_prefix + ".self_attn.q_proj.weight");
    register_loaded_vision_float_tensor(
        session,
        qwen35_audio_layer_q_proj_bias_id(layer),
        tensors,
        layer_prefix + ".self_attn.q_proj.bias");
    register_loaded_vision_float_tensor(
        session,
        qwen35_audio_layer_k_proj_weight_id(layer),
        tensors,
        layer_prefix + ".self_attn.k_proj.weight");
    register_loaded_vision_float_tensor(
        session,
        qwen35_audio_layer_k_proj_bias_id(layer),
        tensors,
        layer_prefix + ".self_attn.k_proj.bias");
    register_loaded_vision_float_tensor(
        session,
        qwen35_audio_layer_v_proj_weight_id(layer),
        tensors,
        layer_prefix + ".self_attn.v_proj.weight");
    register_loaded_vision_float_tensor(
        session,
        qwen35_audio_layer_v_proj_bias_id(layer),
        tensors,
        layer_prefix + ".self_attn.v_proj.bias");
    register_loaded_vision_float_tensor(
        session,
        qwen35_audio_layer_out_proj_weight_id(layer),
        tensors,
        layer_prefix + ".self_attn.out_proj.weight");
    register_loaded_vision_float_tensor(
        session,
        qwen35_audio_layer_out_proj_bias_id(layer),
        tensors,
        layer_prefix + ".self_attn.out_proj.bias");
    register_loaded_vision_float_tensor(
        session,
        qwen35_audio_layer_fc1_weight_id(layer),
        tensors,
        layer_prefix + ".fc1.weight");
    register_loaded_vision_float_tensor(
        session,
        qwen35_audio_layer_fc1_bias_id(layer),
        tensors,
        layer_prefix + ".fc1.bias");
    register_loaded_vision_float_tensor(
        session,
        qwen35_audio_layer_fc2_weight_id(layer),
        tensors,
        layer_prefix + ".fc2.weight");
    register_loaded_vision_float_tensor(
        session,
        qwen35_audio_layer_fc2_bias_id(layer),
        tensors,
        layer_prefix + ".fc2.bias");
  }
  register_loaded_vision_float_tensor(
      session,
      qwen35_audio_ln_post_weight_id(),
      tensors,
      prefix + ".ln_post.weight");
  register_loaded_vision_float_tensor(
      session,
      qwen35_audio_ln_post_bias_id(),
      tensors,
      prefix + ".ln_post.bias");
  register_loaded_vision_float_tensor(
      session,
      qwen35_audio_proj1_weight_id(),
      tensors,
      prefix + ".proj1.weight");
  register_loaded_vision_float_tensor(
      session,
      qwen35_audio_proj1_bias_id(),
      tensors,
      prefix + ".proj1.bias");
  register_loaded_vision_float_tensor(
      session,
      qwen35_audio_proj2_weight_id(),
      tensors,
      prefix + ".proj2.weight");
  register_loaded_vision_float_tensor(
      session,
      qwen35_audio_proj2_bias_id(),
      tensors,
      prefix + ".proj2.bias");
}

void validate_qwen35_audio_config(const EdgeCmlxQwen35AudioConfig& config) {
  if (config.num_mel_bins <= 0 ||
      config.encoder_layers <= 0 ||
      config.encoder_attention_heads <= 0 ||
      config.encoder_ffn_dim <= 0 ||
      config.d_model <= 0 ||
      config.max_source_positions <= 0 ||
      config.n_window <= 0 ||
      config.output_dim <= 0 ||
      config.n_window_infer <= 0 ||
      config.downsample_hidden_size <= 0 ||
      config.layer_norm_epsilon < 0.0f ||
      config.d_model % config.encoder_attention_heads != 0) {
    throw std::runtime_error("Qwen3 ASR Cmlx audio config is invalid");
  }
}

const EdgeCmlxQwen35AudioConfig& checked_qwen35_audio_config(
    const EdgeCmlxQwen35Session& session) {
  if (!session.audio_config.has_value()) {
    throw std::runtime_error("Qwen3 ASR Cmlx audio config is not set");
  }
  return *session.audio_config;
}

}

using namespace edge_cmlx::detail;

int qwen35_audio_conv_output_length(int input_length) {
  int length = input_length;
  for (int i = 0; i < 3; ++i) {
    length = (length + 1) / 2;
  }
  return std::max(1, length);
}

std::vector<int> qwen35_audio_chunk_lengths(
    int frame_count,
    int chunk_size) {
  std::vector<int> lengths;
  int offset = 0;
  while (offset < frame_count) {
    const int length = std::min(chunk_size, frame_count - offset);
    lengths.push_back(length);
    offset += length;
  }
  return lengths;
}

std::vector<int> qwen35_audio_window_lengths(
    const std::vector<int>& chunk_output_lengths,
    int chunks_per_window) {
  std::vector<int> lengths;
  for (size_t offset = 0; offset < chunk_output_lengths.size();) {
    int length = 0;
    const size_t end = std::min(
        offset + static_cast<size_t>(chunks_per_window),
        chunk_output_lengths.size());
    for (size_t i = offset; i < end; ++i) {
      length += chunk_output_lengths[i];
    }
    if (length > 0) {
      lengths.push_back(length);
    }
    offset = end;
  }
  return lengths;
}

std::vector<float> qwen35_audio_sinusoidal_positional_embedding_values(
    int seq_len,
    int channels) {
  if (channels <= 0 || channels % 2 != 0) {
    throw std::runtime_error("Qwen3 ASR audio positional embedding shape mismatch");
  }
  std::vector<float> values(
      static_cast<size_t>(seq_len) * static_cast<size_t>(channels),
      0.0f);
  const int half = channels / 2;
  const float log_increment = std::log(10000.0f) / static_cast<float>(half - 1);
  for (int position = 0; position < seq_len; ++position) {
    for (int index = 0; index < half; ++index) {
      const float inv_timescale =
          std::exp(-log_increment * static_cast<float>(index));
      const float scaled = static_cast<float>(position) * inv_timescale;
      const size_t base =
          static_cast<size_t>(position) * static_cast<size_t>(channels);
      values[base + static_cast<size_t>(index)] = std::sin(scaled);
      values[base + static_cast<size_t>(half + index)] = std::cos(scaled);
    }
  }
  return values;
}

mlx::core::array qwen35_audio_linear(
    const mlx::core::array& input,
    const mlx::core::array& weight,
    const mlx::core::array* bias,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  const int input_rank = input.ndim();
  if (input_rank < 2) {
    throw std::runtime_error("Qwen3 ASR audio linear input rank mismatch");
  }
  const int input_dim = static_cast<int>(input.shape(input_rank - 1));
  if (weight.ndim() != 2 || weight.shape(1) != input_dim) {
    throw std::runtime_error("Qwen3 ASR audio linear weight shape mismatch");
  }
  const int output_dim = static_cast<int>(weight.shape(0));
  const size_t input_count = array_element_count(input.shape());
  const int rows = static_cast<int>(input_count / static_cast<size_t>(input_dim));
  auto typed_weight = weight.dtype() == input.dtype()
      ? weight
      : astype(weight, input.dtype(), stream);
  auto flat = reshape(input, Shape{rows, input_dim}, stream);
  auto output = matmul(flat, transpose(typed_weight, {1, 0}, stream), stream);
  if (bias != nullptr) {
    auto typed_bias = bias->dtype() == input.dtype()
        ? *bias
        : astype(*bias, input.dtype(), stream);
    output = add(output, typed_bias, stream);
  }
  auto output_shape = input.shape();
  output_shape[static_cast<size_t>(input_rank - 1)] = output_dim;
  return reshape(output, std::move(output_shape), stream);
}

mlx::core::array qwen35_audio_conv2d_gelu(
    const EdgeCmlxQwen35Session& session,
    const mlx::core::array& input,
    int weight_id,
    int bias_id,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  auto output = conv2d(
      input,
      checked_qwen35_float_tensor(session, weight_id),
      {2, 2},
      {1, 1},
      {1, 1},
      1,
      stream);
  output = add(
      output,
      checked_qwen35_float_tensor(session, bias_id),
      stream);
  return qwen35_vision_gelu_tanh(output, stream);
}

mlx::core::array qwen35_audio_attention(
    const EdgeCmlxQwen35Session& session,
    int layer,
    const mlx::core::array& input,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  const auto& config = checked_qwen35_audio_config(session);
  const int heads = config.encoder_attention_heads;
  const int head_dim = config.d_model / heads;
  return edge_cmlx::blocks::encoder_attention(
      input,
      checked_qwen35_float_tensor(session, qwen35_audio_layer_q_proj_weight_id(layer)),
      checked_qwen35_float_tensor(session, qwen35_audio_layer_k_proj_weight_id(layer)),
      checked_qwen35_float_tensor(session, qwen35_audio_layer_v_proj_weight_id(layer)),
      checked_qwen35_float_tensor(session, qwen35_audio_layer_out_proj_weight_id(layer)),
      &checked_qwen35_float_tensor(session, qwen35_audio_layer_q_proj_bias_id(layer)),
      &checked_qwen35_float_tensor(session, qwen35_audio_layer_k_proj_bias_id(layer)),
      &checked_qwen35_float_tensor(session, qwen35_audio_layer_v_proj_bias_id(layer)),
      &checked_qwen35_float_tensor(session, qwen35_audio_layer_out_proj_bias_id(layer)),
      heads,
      head_dim,
      stream);
}

mlx::core::array qwen35_audio_encoder_layer(
    const EdgeCmlxQwen35Session& session,
    int layer,
    const mlx::core::array& input,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  const auto& config = checked_qwen35_audio_config(session);
  auto residual = input;
  auto hidden = qwen35_vision_layer_norm(
      input,
      checked_qwen35_float_tensor(session, qwen35_audio_layer_self_attn_ln_weight_id(layer)),
      checked_qwen35_float_tensor(session, qwen35_audio_layer_self_attn_ln_bias_id(layer)),
      config.layer_norm_epsilon,
      stream);
  hidden = add(
      residual,
      qwen35_audio_attention(session, layer, hidden, stream),
      stream);
  residual = hidden;
  hidden = qwen35_vision_layer_norm(
      hidden,
      checked_qwen35_float_tensor(session, qwen35_audio_layer_final_ln_weight_id(layer)),
      checked_qwen35_float_tensor(session, qwen35_audio_layer_final_ln_bias_id(layer)),
      config.layer_norm_epsilon,
      stream);
  hidden = qwen35_audio_linear(
      hidden,
      checked_qwen35_float_tensor(session, qwen35_audio_layer_fc1_weight_id(layer)),
      &checked_qwen35_float_tensor(session, qwen35_audio_layer_fc1_bias_id(layer)),
      stream);
  hidden = qwen35_vision_gelu_tanh(hidden, stream);
  hidden = qwen35_audio_linear(
      hidden,
      checked_qwen35_float_tensor(session, qwen35_audio_layer_fc2_weight_id(layer)),
      &checked_qwen35_float_tensor(session, qwen35_audio_layer_fc2_bias_id(layer)),
      stream);
  return add(residual, hidden, stream);
}

mlx::core::array qwen35_audio_encode_array(
    const EdgeCmlxQwen35Session& session,
    const float* log_mel_features,
    int frame_count,
    int mel_bin_count,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  const auto& config = checked_qwen35_audio_config(session);
  if (mel_bin_count != config.num_mel_bins) {
    throw std::runtime_error("Qwen3 ASR audio mel bin count mismatch");
  }
  const int chunk_size = config.n_window * 2;
  const auto chunk_lengths = qwen35_audio_chunk_lengths(frame_count, chunk_size);
  const int chunk_count = static_cast<int>(chunk_lengths.size());
  const int max_chunk_length =
      *std::max_element(chunk_lengths.begin(), chunk_lengths.end());
  std::vector<float> chunk_values(
      static_cast<size_t>(chunk_count) *
          static_cast<size_t>(mel_bin_count) *
          static_cast<size_t>(max_chunk_length),
      0.0f);
  int source_frame_offset = 0;
  for (int chunk = 0; chunk < chunk_count; ++chunk) {
    for (int frame = 0; frame < chunk_lengths[static_cast<size_t>(chunk)]; ++frame) {
      for (int mel = 0; mel < mel_bin_count; ++mel) {
        const size_t destination =
            (static_cast<size_t>(chunk) * static_cast<size_t>(mel_bin_count) *
                 static_cast<size_t>(max_chunk_length)) +
            (static_cast<size_t>(mel) * static_cast<size_t>(max_chunk_length)) +
            static_cast<size_t>(frame);
        const size_t source =
            (static_cast<size_t>(source_frame_offset + frame) *
                 static_cast<size_t>(mel_bin_count)) +
            static_cast<size_t>(mel);
        chunk_values[destination] = log_mel_features[source];
      }
    }
    source_frame_offset += chunk_lengths[static_cast<size_t>(chunk)];
  }

  std::vector<int> chunk_output_lengths;
  chunk_output_lengths.reserve(chunk_lengths.size());
  for (const int length : chunk_lengths) {
    chunk_output_lengths.push_back(qwen35_audio_conv_output_length(length));
  }
  auto hidden = array(
      chunk_values.data(),
      Shape{chunk_count, mel_bin_count, max_chunk_length, 1},
      float32);
  hidden = qwen35_audio_conv2d_gelu(
      session,
      hidden,
      qwen35_audio_conv2d1_weight_id(),
      qwen35_audio_conv2d1_bias_id(),
      stream);
  hidden = qwen35_audio_conv2d_gelu(
      session,
      hidden,
      qwen35_audio_conv2d2_weight_id(),
      qwen35_audio_conv2d2_bias_id(),
      stream);
  hidden = qwen35_audio_conv2d_gelu(
      session,
      hidden,
      qwen35_audio_conv2d3_weight_id(),
      qwen35_audio_conv2d3_bias_id(),
      stream);
  const int frequency = static_cast<int>(hidden.shape(1));
  const int time = static_cast<int>(hidden.shape(2));
  const int channels = static_cast<int>(hidden.shape(3));
  hidden = reshape(
      transpose(hidden, {0, 2, 3, 1}, stream),
      Shape{chunk_count, time, channels * frequency},
      stream);
  hidden = qwen35_audio_linear(
      hidden,
      checked_qwen35_float_tensor(session, qwen35_audio_conv_out_weight_id()),
      nullptr,
      stream);
  if (time > config.max_source_positions) {
    throw std::runtime_error("Qwen3 ASR audio positional length exceeds config");
  }
  auto pos_values = qwen35_audio_sinusoidal_positional_embedding_values(
      time,
      config.d_model);
  auto pos = reshape(
      array(pos_values.data(), Shape{time, config.d_model}, float32),
      Shape{1, time, config.d_model},
      stream);
  hidden = add(hidden, pos, stream);
  eval(hidden);

  std::vector<array> valid_chunks;
  valid_chunks.reserve(chunk_output_lengths.size());
  for (int chunk = 0; chunk < chunk_count; ++chunk) {
    const int valid = chunk_output_lengths[static_cast<size_t>(chunk)];
    valid_chunks.push_back(
        squeeze(
            slice(
                hidden,
                Shape{chunk, 0, 0},
                Shape{chunk + 1, valid, config.d_model},
                stream),
            0,
            stream));
  }
  hidden = concatenate(valid_chunks, 0, stream);

  const int chunks_per_window = std::max(1, config.n_window_infer / chunk_size);
  const auto window_lengths = qwen35_audio_window_lengths(
      chunk_output_lengths,
      chunks_per_window);
  std::vector<array> processed_windows;
  processed_windows.reserve(window_lengths.size());
  int token_offset = 0;
  for (const int window_length : window_lengths) {
    auto window = slice(
        hidden,
        Shape{token_offset, 0},
        Shape{token_offset + window_length, config.d_model},
        stream);
    window = expand_dims(window, 0, stream);
    for (int layer = 0; layer < config.encoder_layers; ++layer) {
      window = qwen35_audio_encoder_layer(session, layer, window, stream);
    }
    eval(window);
    processed_windows.push_back(squeeze(window, 0, stream));
    token_offset += window_length;
  }
  hidden = concatenate(processed_windows, 0, stream);
  hidden = qwen35_vision_layer_norm(
      hidden,
      checked_qwen35_float_tensor(session, qwen35_audio_ln_post_weight_id()),
      checked_qwen35_float_tensor(session, qwen35_audio_ln_post_bias_id()),
      config.layer_norm_epsilon,
      stream);
  hidden = qwen35_audio_linear(
      hidden,
      checked_qwen35_float_tensor(session, qwen35_audio_proj1_weight_id()),
      &checked_qwen35_float_tensor(session, qwen35_audio_proj1_bias_id()),
      stream);
  hidden = qwen35_vision_gelu_tanh(hidden, stream);
  return qwen35_audio_linear(
      hidden,
      checked_qwen35_float_tensor(session, qwen35_audio_proj2_weight_id()),
      &checked_qwen35_float_tensor(session, qwen35_audio_proj2_bias_id()),
      stream);
}

int edge_cmlx_qwen35_session_set_audio_config(
    void* session,
    const EdgeCmlxQwen35AudioConfig* config) {
  edge_cmlx_error.clear();
  if (config == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_set_audio_config received a null config");
  }
  try {
    auto* qwen_session = checked_qwen35_session(session);
    validate_qwen35_audio_config(*config);
    qwen_session->audio_config = *config;
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_set_audio_config failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_load_audio_safetensors(
    void* session,
    const char* const* shard_paths,
    int shard_count,
    const char* audio_prefix) {
  edge_cmlx_error.clear();
  if (shard_paths == nullptr || shard_count <= 0 || audio_prefix == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_load_audio_safetensors received invalid arguments");
  }

  try {
    auto* qwen_session = checked_qwen35_session(session);
    checked_qwen35_audio_config(*qwen_session);
    std::unordered_map<std::string, mlx::core::array> tensors;
    const std::string prefix(audio_prefix);
    const std::string prefix_with_dot = prefix + ".";
    for (int i = 0; i < shard_count; ++i) {
      if (shard_paths[i] == nullptr) {
        return set_error(
            "edge_cmlx_qwen35_session_load_audio_safetensors received a null shard path");
      }
      auto loaded = mlx::core::load_safetensors(std::string(shard_paths[i]));
      for (auto& item : loaded.first) {
        if (item.first.rfind(prefix_with_dot, 0) == 0) {
          tensors.insert_or_assign(item.first, std::move(item.second));
        }
      }
    }
    register_qwen35_audio_tensors(*qwen_session, tensors, prefix);
    mlx::core::clear_cache();
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_load_audio_safetensors failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_audio_encode(
    void* session,
    const float* log_mel_features,
    int frame_count,
    int mel_bin_count,
    float* output,
    size_t output_count,
    int* output_frames,
    int* output_hidden_size) {
  edge_cmlx_error.clear();
  if (log_mel_features == nullptr || output == nullptr ||
      output_frames == nullptr || output_hidden_size == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_audio_encode received a null pointer");
  }
  if (frame_count <= 0 || mel_bin_count <= 0 || output_count == 0) {
    return set_error(
        "edge_cmlx_qwen35_session_audio_encode received an invalid shape");
  }

  try {
    auto* qwen_session = checked_qwen35_session(session);
    const auto& config = checked_qwen35_audio_config(*qwen_session);
    auto gpu_device = mlx::core::Device{mlx::core::Device::gpu};
    auto encoded = qwen35_audio_encode_array(
        *qwen_session,
        log_mel_features,
        frame_count,
        mel_bin_count,
        gpu_device);
    mlx::core::eval(encoded);
    const int rows = static_cast<int>(encoded.shape(0));
    const int columns = static_cast<int>(encoded.shape(1));
    if (columns != config.output_dim) {
      return set_error(
          "edge_cmlx_qwen35_session_audio_encode output hidden mismatch");
    }
    const size_t count =
        static_cast<size_t>(rows) * static_cast<size_t>(columns);
    if (count > output_count) {
      return set_error(
          "edge_cmlx_qwen35_session_audio_encode output buffer is too small");
    }
    const float* data = encoded.data<float>();
    std::copy(data, data + count, output);
    *output_frames = rows;
    *output_hidden_size = columns;
    mlx::core::clear_cache();
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_audio_encode failed with an unknown error");
  }
}
