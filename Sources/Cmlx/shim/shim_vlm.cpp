// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

#include "blocks.h"
#include "shim_internal.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
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

const EdgeCmlxQwen35VisionConfig& checked_qwen35_vision_config(
    const EdgeCmlxQwen35Session& session);

int qwen35_vision_patch_embed_weight_id() {
  return 200000;
}

int qwen35_vision_patch_embed_bias_id() {
  return 200001;
}

int qwen35_vision_pos_embed_id() {
  return 200002;
}

int qwen35_vision_merger_norm_weight_id() {
  return 200010;
}

int qwen35_vision_merger_norm_bias_id() {
  return 200011;
}

int qwen35_vision_merger_fc1_weight_id() {
  return 200012;
}

int qwen35_vision_merger_fc1_bias_id() {
  return 200013;
}

int qwen35_vision_merger_fc2_weight_id() {
  return 200014;
}

int qwen35_vision_merger_fc2_bias_id() {
  return 200015;
}

int qwen35_vision_layer_tensor_id(int layer_index, int offset) {
  return 210000 + layer_index * 100 + offset;
}

int qwen35_vision_layer_norm1_weight_id(int layer_index) {
  return qwen35_vision_layer_tensor_id(layer_index, 1);
}

int qwen35_vision_layer_norm1_bias_id(int layer_index) {
  return qwen35_vision_layer_tensor_id(layer_index, 2);
}

int qwen35_vision_layer_qkv_weight_id(int layer_index) {
  return qwen35_vision_layer_tensor_id(layer_index, 10);
}

int qwen35_vision_layer_qkv_bias_id(int layer_index) {
  return qwen35_vision_layer_tensor_id(layer_index, 11);
}

int qwen35_vision_layer_proj_weight_id(int layer_index) {
  return qwen35_vision_layer_tensor_id(layer_index, 12);
}

int qwen35_vision_layer_proj_bias_id(int layer_index) {
  return qwen35_vision_layer_tensor_id(layer_index, 13);
}

int qwen35_vision_layer_norm2_weight_id(int layer_index) {
  return qwen35_vision_layer_tensor_id(layer_index, 20);
}

int qwen35_vision_layer_norm2_bias_id(int layer_index) {
  return qwen35_vision_layer_tensor_id(layer_index, 21);
}

int qwen35_vision_layer_fc1_weight_id(int layer_index) {
  return qwen35_vision_layer_tensor_id(layer_index, 30);
}

int qwen35_vision_layer_fc1_bias_id(int layer_index) {
  return qwen35_vision_layer_tensor_id(layer_index, 31);
}

int qwen35_vision_layer_fc2_weight_id(int layer_index) {
  return qwen35_vision_layer_tensor_id(layer_index, 32);
}

int qwen35_vision_layer_fc2_bias_id(int layer_index) {
  return qwen35_vision_layer_tensor_id(layer_index, 33);
}

void register_qwen35_vision_tensors(
    EdgeCmlxQwen35Session& session,
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    const std::string& prefix) {
  const auto& config = checked_qwen35_vision_config(session);
  register_loaded_vision_float_tensor(
      session,
      qwen35_vision_patch_embed_weight_id(),
      tensors,
      prefix + ".patch_embed.proj.weight");
  register_loaded_vision_float_tensor(
      session,
      qwen35_vision_patch_embed_bias_id(),
      tensors,
      prefix + ".patch_embed.proj.bias");
  register_loaded_vision_float_tensor(
      session,
      qwen35_vision_pos_embed_id(),
      tensors,
      prefix + ".pos_embed.weight");
  for (int layer = 0; layer < config.layer_count; ++layer) {
    const auto layer_prefix =
        prefix + ".blocks." + std::to_string(layer);
    register_loaded_vision_float_tensor(
        session,
        qwen35_vision_layer_norm1_weight_id(layer),
        tensors,
        layer_prefix + ".norm1.weight");
    register_loaded_vision_float_tensor(
        session,
        qwen35_vision_layer_norm1_bias_id(layer),
        tensors,
        layer_prefix + ".norm1.bias");
    register_loaded_vision_float_tensor(
        session,
        qwen35_vision_layer_qkv_weight_id(layer),
        tensors,
        layer_prefix + ".attn.qkv.weight");
    register_loaded_vision_float_tensor(
        session,
        qwen35_vision_layer_qkv_bias_id(layer),
        tensors,
        layer_prefix + ".attn.qkv.bias");
    register_loaded_vision_float_tensor(
        session,
        qwen35_vision_layer_proj_weight_id(layer),
        tensors,
        layer_prefix + ".attn.proj.weight");
    register_loaded_vision_float_tensor(
        session,
        qwen35_vision_layer_proj_bias_id(layer),
        tensors,
        layer_prefix + ".attn.proj.bias");
    register_loaded_vision_float_tensor(
        session,
        qwen35_vision_layer_norm2_weight_id(layer),
        tensors,
        layer_prefix + ".norm2.weight");
    register_loaded_vision_float_tensor(
        session,
        qwen35_vision_layer_norm2_bias_id(layer),
        tensors,
        layer_prefix + ".norm2.bias");
    register_loaded_vision_float_tensor(
        session,
        qwen35_vision_layer_fc1_weight_id(layer),
        tensors,
        layer_prefix + ".mlp.linear_fc1.weight");
    register_loaded_vision_float_tensor(
        session,
        qwen35_vision_layer_fc1_bias_id(layer),
        tensors,
        layer_prefix + ".mlp.linear_fc1.bias");
    register_loaded_vision_float_tensor(
        session,
        qwen35_vision_layer_fc2_weight_id(layer),
        tensors,
        layer_prefix + ".mlp.linear_fc2.weight");
    register_loaded_vision_float_tensor(
        session,
        qwen35_vision_layer_fc2_bias_id(layer),
        tensors,
        layer_prefix + ".mlp.linear_fc2.bias");
  }
  register_loaded_vision_float_tensor(
      session,
      qwen35_vision_merger_norm_weight_id(),
      tensors,
      prefix + ".merger.norm.weight");
  register_loaded_vision_float_tensor(
      session,
      qwen35_vision_merger_norm_bias_id(),
      tensors,
      prefix + ".merger.norm.bias");
  register_loaded_vision_float_tensor(
      session,
      qwen35_vision_merger_fc1_weight_id(),
      tensors,
      prefix + ".merger.linear_fc1.weight");
  register_loaded_vision_float_tensor(
      session,
      qwen35_vision_merger_fc1_bias_id(),
      tensors,
      prefix + ".merger.linear_fc1.bias");
  register_loaded_vision_float_tensor(
      session,
      qwen35_vision_merger_fc2_weight_id(),
      tensors,
      prefix + ".merger.linear_fc2.weight");
  register_loaded_vision_float_tensor(
      session,
      qwen35_vision_merger_fc2_bias_id(),
      tensors,
      prefix + ".merger.linear_fc2.bias");
}

void validate_qwen35_vision_config(const EdgeCmlxQwen35VisionConfig& config) {
  if (config.hidden_size <= 0 ||
      config.intermediate_size <= 0 ||
      config.layer_count <= 0 ||
      config.head_count <= 0 ||
      config.patch_size <= 0 ||
      config.spatial_merge_size <= 0 ||
      config.temporal_patch_size <= 0 ||
      config.output_hidden_size <= 0 ||
      config.layer_norm_epsilon < 0.0f ||
      config.hidden_size % config.head_count != 0) {
    throw std::runtime_error("Qwen3.5 Cmlx vision config is invalid");
  }
}

const EdgeCmlxQwen35VisionConfig& checked_qwen35_vision_config(
    const EdgeCmlxQwen35Session& session) {
  if (!session.vision_config.has_value()) {
    throw std::runtime_error("Qwen3.5 Cmlx vision config is not set");
  }
  return *session.vision_config;
}

}

using namespace edge_cmlx::detail;

struct Qwen35VisionGrid {
  int temporal;
  int height;
  int width;
};

std::vector<Qwen35VisionGrid> qwen35_vision_grids_from_raw(
    const int* grid_thw,
    int grid_count,
    int merge) {
  if (grid_thw == nullptr || grid_count <= 0) {
    throw std::runtime_error("Qwen3.5 vision grid is empty");
  }
  std::vector<Qwen35VisionGrid> grids;
  grids.reserve(static_cast<size_t>(grid_count));
  for (int i = 0; i < grid_count; ++i) {
    Qwen35VisionGrid grid{
        grid_thw[i * 3],
        grid_thw[i * 3 + 1],
        grid_thw[i * 3 + 2]};
    if (grid.temporal <= 0 ||
        grid.height <= 0 ||
        grid.width <= 0 ||
        grid.height % merge != 0 ||
        grid.width % merge != 0) {
      throw std::runtime_error("Qwen3.5 vision grid is invalid");
    }
    grids.push_back(grid);
  }
  return grids;
}

int qwen35_vision_token_count(const std::vector<Qwen35VisionGrid>& grids) {
  int total = 0;
  for (const auto& grid : grids) {
    total += grid.temporal * grid.height * grid.width;
  }
  return total;
}

std::vector<int> qwen35_vision_cumulative_lengths(
    const std::vector<Qwen35VisionGrid>& grids) {
  std::vector<int> lengths;
  lengths.reserve(grids.size() + 1);
  lengths.push_back(0);
  int total = 0;
  for (const auto& grid : grids) {
    total += grid.temporal * grid.height * grid.width;
    lengths.push_back(total);
  }
  return lengths;
}

mlx::core::array qwen35_vision_linear(
    const mlx::core::array& input,
    const mlx::core::array& weight,
    const mlx::core::array& bias,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  auto typed_weight = weight.dtype() == input.dtype()
      ? weight
      : astype(weight, input.dtype(), stream);
  auto typed_bias = bias.dtype() == input.dtype()
      ? bias
      : astype(bias, input.dtype(), stream);
  return addmm(
      typed_bias,
      input,
      transpose(typed_weight, {1, 0}, stream),
      1.0f,
      1.0f,
      stream);
}

mlx::core::array qwen35_vision_layer_norm(
    const mlx::core::array& input,
    const mlx::core::array& weight,
    const mlx::core::array& bias,
    float epsilon,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  auto typed_weight = weight.dtype() == input.dtype()
      ? weight
      : astype(weight, input.dtype(), stream);
  auto typed_bias = bias.dtype() == input.dtype()
      ? bias
      : astype(bias, input.dtype(), stream);
  return fast::layer_norm(
      input,
      std::optional<array>(typed_weight),
      std::optional<array>(typed_bias),
      epsilon,
      stream);
}

mlx::core::array qwen35_vision_gelu_tanh(
    const mlx::core::array& input,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  auto squared = multiply(input, input, stream);
  auto cubed = multiply(squared, input, stream);
  auto inner = add(
      input,
      multiply(array(0.044715f, input.dtype()), cubed, stream),
      stream);
  auto scaled = multiply(
      array(0.7978845608028654f, input.dtype()),
      inner,
      stream);
  return multiply(
      multiply(array(0.5f, input.dtype()), input, stream),
      add(array(1.0f, input.dtype()), tanh(scaled, stream), stream),
      stream);
}

mlx::core::array qwen35_vision_patch_embed(
    const EdgeCmlxQwen35Session& session,
    const mlx::core::array& pixel_values,
    int num_patches,
    int patch_dim,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  const auto& config = checked_qwen35_vision_config(session);
  const int expected_patch_dim =
      3 * config.temporal_patch_size * config.patch_size * config.patch_size;
  if (patch_dim != expected_patch_dim) {
    throw std::runtime_error("Qwen3.5 vision patch dimension mismatch");
  }
  auto pixels = reshape(
      pixel_values,
      Shape{
          num_patches,
          3,
          config.temporal_patch_size,
          config.patch_size,
          config.patch_size},
      stream);
  pixels = reshape(
      transpose(pixels, {0, 2, 3, 4, 1}, stream),
      Shape{num_patches, patch_dim},
      stream);
  auto weight = reshape(
      checked_qwen35_float_tensor(session, qwen35_vision_patch_embed_weight_id()),
      Shape{config.hidden_size, patch_dim},
      stream);
  return qwen35_vision_linear(
      pixels,
      weight,
      checked_qwen35_float_tensor(session, qwen35_vision_patch_embed_bias_id()),
      stream);
}

std::vector<float> qwen35_vision_positional_embedding_values(
    const EdgeCmlxQwen35Session& session,
    const std::vector<Qwen35VisionGrid>& grids) {
  const auto& config = checked_qwen35_vision_config(session);
  const auto& pos = checked_qwen35_float_tensor(
      session, qwen35_vision_pos_embed_id());
  if (pos.ndim() != 2 || pos.shape(1) != config.hidden_size) {
    throw std::runtime_error("Qwen3.5 vision pos_embed shape mismatch");
  }
  const int embedding_count = static_cast<int>(pos.shape(0));
  const int grid_side =
      static_cast<int>(std::sqrt(static_cast<double>(embedding_count)));
  if (grid_side * grid_side != embedding_count) {
    throw std::runtime_error("Qwen3.5 vision pos_embed is not square");
  }
  const int max_index = grid_side - 1;
  const int hidden = config.hidden_size;
  const int merge = config.spatial_merge_size;
  const float* pos_data = pos.data<float>();
  std::vector<float> output;
  output.reserve(static_cast<size_t>(qwen35_vision_token_count(grids)) *
                 static_cast<size_t>(hidden));

  for (const auto& grid : grids) {
    std::vector<float> base(
        static_cast<size_t>(grid.height) *
            static_cast<size_t>(grid.width) *
            static_cast<size_t>(hidden),
        0.0f);
    for (int y = 0; y < grid.height; ++y) {
      const float y_position = grid.height == 1
          ? 0.0f
          : static_cast<float>(y) * static_cast<float>(max_index) /
                static_cast<float>(grid.height - 1);
      const int y_floor =
          std::min(max_index, static_cast<int>(std::floor(y_position)));
      const int y_ceil = std::min(max_index, y_floor + 1);
      const float dy = y_position - static_cast<float>(y_floor);
      for (int x = 0; x < grid.width; ++x) {
        const float x_position = grid.width == 1
            ? 0.0f
            : static_cast<float>(x) * static_cast<float>(max_index) /
                  static_cast<float>(grid.width - 1);
        const int x_floor =
            std::min(max_index, static_cast<int>(std::floor(x_position)));
        const int x_ceil = std::min(max_index, x_floor + 1);
        const float dx = x_position - static_cast<float>(x_floor);
        const int corners[4] = {
            y_floor * grid_side + x_floor,
            y_floor * grid_side + x_ceil,
            y_ceil * grid_side + x_floor,
            y_ceil * grid_side + x_ceil};
        const float scales[4] = {
            (1.0f - dy) * (1.0f - dx),
            (1.0f - dy) * dx,
            dy * (1.0f - dx),
            dy * dx};
        const size_t destination =
            (static_cast<size_t>(y) * static_cast<size_t>(grid.width) +
             static_cast<size_t>(x)) *
            static_cast<size_t>(hidden);
        for (int corner = 0; corner < 4; ++corner) {
          const size_t source =
              static_cast<size_t>(corners[corner]) *
              static_cast<size_t>(hidden);
          for (int h = 0; h < hidden; ++h) {
            base[destination + static_cast<size_t>(h)] +=
                pos_data[source + static_cast<size_t>(h)] * scales[corner];
          }
        }
      }
    }
    for (int temporal = 0; temporal < grid.temporal; ++temporal) {
      (void)temporal;
      for (int block_h = 0; block_h < grid.height / merge; ++block_h) {
        for (int block_w = 0; block_w < grid.width / merge; ++block_w) {
          for (int intra_h = 0; intra_h < merge; ++intra_h) {
            for (int intra_w = 0; intra_w < merge; ++intra_w) {
              const int y = block_h * merge + intra_h;
              const int x = block_w * merge + intra_w;
              const size_t source =
                  (static_cast<size_t>(y) * static_cast<size_t>(grid.width) +
                   static_cast<size_t>(x)) *
                  static_cast<size_t>(hidden);
              output.insert(
                  output.end(),
                  base.begin() + static_cast<std::ptrdiff_t>(source),
                  base.begin() + static_cast<std::ptrdiff_t>(source + hidden));
            }
          }
        }
      }
    }
  }
  return output;
}

std::vector<float> qwen35_vision_rotary_frequency_values(
    const EdgeCmlxQwen35Session& session,
    const std::vector<Qwen35VisionGrid>& grids) {
  const auto& config = checked_qwen35_vision_config(session);
  const int head_dim = config.hidden_size / config.head_count;
  const int rotary_width = head_dim / 2;
  const int frequency_width = rotary_width / 2;
  if (head_dim <= 0 || head_dim % 2 != 0 || frequency_width <= 0) {
    throw std::runtime_error("Qwen3.5 vision rotary shape mismatch");
  }
  int max_hw = 0;
  for (const auto& grid : grids) {
    max_hw = std::max(max_hw, std::max(grid.height, grid.width));
  }
  std::vector<float> table(
      static_cast<size_t>(max_hw) * static_cast<size_t>(frequency_width),
      0.0f);
  for (int position = 0; position < max_hw; ++position) {
    for (int index = 0; index < frequency_width; ++index) {
      const float exponent =
          static_cast<float>(index * 2) / static_cast<float>(rotary_width);
      table[static_cast<size_t>(position) * frequency_width + index] =
          static_cast<float>(position) / std::pow(10000.0f, exponent);
    }
  }

  const int merge = config.spatial_merge_size;
  std::vector<float> output;
  output.reserve(static_cast<size_t>(qwen35_vision_token_count(grids)) *
                 static_cast<size_t>(rotary_width));
  for (const auto& grid : grids) {
    for (int temporal = 0; temporal < grid.temporal; ++temporal) {
      (void)temporal;
      for (int block_h = 0; block_h < grid.height / merge; ++block_h) {
        for (int block_w = 0; block_w < grid.width / merge; ++block_w) {
          for (int intra_h = 0; intra_h < merge; ++intra_h) {
            for (int intra_w = 0; intra_w < merge; ++intra_w) {
              const int y = block_h * merge + intra_h;
              const int x = block_w * merge + intra_w;
              output.insert(
                  output.end(),
                  table.begin() + y * frequency_width,
                  table.begin() + (y + 1) * frequency_width);
              output.insert(
                  output.end(),
                  table.begin() + x * frequency_width,
                  table.begin() + (x + 1) * frequency_width);
            }
          }
        }
      }
    }
  }
  return output;
}

mlx::core::array qwen35_vision_apply_rotary(
    const mlx::core::array& value,
    const mlx::core::array& frequencies,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  const int token_count = static_cast<int>(value.shape(0));
  const int head_count = static_cast<int>(value.shape(1));
  const int head_dim = static_cast<int>(value.shape(2));
  const int rotary_width = head_dim / 2;
  auto first = slice(
      value,
      Shape{0, 0, 0},
      Shape{token_count, head_count, rotary_width},
      stream);
  auto second = slice(
      value,
      Shape{0, 0, rotary_width},
      Shape{token_count, head_count, head_dim},
      stream);
  auto cosine = expand_dims(cos(frequencies, stream), 1, stream);
  auto sine = expand_dims(sin(frequencies, stream), 1, stream);
  auto rotated_first = subtract(
      multiply(first, cosine, stream),
      multiply(second, sine, stream),
      stream);
  auto rotated_second = add(
      multiply(second, cosine, stream),
      multiply(first, sine, stream),
      stream);
  return concatenate({rotated_first, rotated_second}, 2, stream);
}

std::optional<mlx::core::array> qwen35_vision_attention_mask(
    const std::vector<int>& cumulative_lengths,
    int token_count) {
  using namespace mlx::core;
  if (cumulative_lengths.size() <= 2) {
    return std::nullopt;
  }
  std::vector<float> values(
      static_cast<size_t>(token_count) * static_cast<size_t>(token_count),
      -1.0e9f);
  for (size_t i = 1; i < cumulative_lengths.size(); ++i) {
    const int start = cumulative_lengths[i - 1];
    const int end = cumulative_lengths[i];
    for (int row = start; row < end; ++row) {
      const size_t row_offset =
          static_cast<size_t>(row) * static_cast<size_t>(token_count);
      for (int column = start; column < end; ++column) {
        values[row_offset + static_cast<size_t>(column)] = 0.0f;
      }
    }
  }
  return array(values.data(), Shape{1, token_count, token_count}, float32);
}

mlx::core::array qwen35_vision_attention(
    const EdgeCmlxQwen35Session& session,
    int layer,
    const mlx::core::array& input,
    const mlx::core::array& rotary_frequencies,
    const std::optional<mlx::core::array>& attention_mask,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  const auto& config = checked_qwen35_vision_config(session);
  const int token_count = static_cast<int>(input.shape(0));
  const int hidden = config.hidden_size;
  const int heads = config.head_count;
  const int head_dim = hidden / heads;
  auto qkv = qwen35_vision_linear(
      input,
      checked_qwen35_float_tensor(session, qwen35_vision_layer_qkv_weight_id(layer)),
      checked_qwen35_float_tensor(session, qwen35_vision_layer_qkv_bias_id(layer)),
      stream);
  qkv = reshape(qkv, Shape{token_count, 3, heads, head_dim}, stream);
  auto query = squeeze(
      slice(qkv, Shape{0, 0, 0, 0}, Shape{token_count, 1, heads, head_dim}, stream),
      1,
      stream);
  auto key = squeeze(
      slice(qkv, Shape{0, 1, 0, 0}, Shape{token_count, 2, heads, head_dim}, stream),
      1,
      stream);
  auto value = squeeze(
      slice(qkv, Shape{0, 2, 0, 0}, Shape{token_count, 3, heads, head_dim}, stream),
      1,
      stream);
  query = qwen35_vision_apply_rotary(query, rotary_frequencies, stream);
  key = qwen35_vision_apply_rotary(key, rotary_frequencies, stream);
  return edge_cmlx::blocks::projected_encoder_attention(
      query,
      key,
      value,
      checked_qwen35_float_tensor(session, qwen35_vision_layer_proj_weight_id(layer)),
      &checked_qwen35_float_tensor(session, qwen35_vision_layer_proj_bias_id(layer)),
      heads,
      head_dim,
      attention_mask,
      stream);
}

mlx::core::array qwen35_vision_patch_merger(
    const EdgeCmlxQwen35Session& session,
    const mlx::core::array& hidden,
    int token_count,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  const auto& config = checked_qwen35_vision_config(session);
  const int merge = config.spatial_merge_size;
  const int patches_per_token = merge * merge;
  if (token_count % patches_per_token != 0) {
    throw std::runtime_error("Qwen3.5 vision merger token count mismatch");
  }
  const int merged_rows = token_count / patches_per_token;
  const auto& norm_weight = checked_qwen35_float_tensor(
      session, qwen35_vision_merger_norm_weight_id());
  const auto& norm_bias = checked_qwen35_float_tensor(
      session, qwen35_vision_merger_norm_bias_id());
  auto merged_input = [&]() -> array {
    if (norm_weight.shape(0) == config.hidden_size) {
      auto normalized = qwen35_vision_layer_norm(
          hidden,
          norm_weight,
          norm_bias,
          config.layer_norm_epsilon,
          stream);
      return reshape(
          normalized,
          Shape{merged_rows, patches_per_token * config.hidden_size},
          stream);
    }
    auto reshaped = reshape(
        hidden,
        Shape{merged_rows, patches_per_token * config.hidden_size},
        stream);
    return qwen35_vision_layer_norm(
        reshaped,
        norm_weight,
        norm_bias,
        config.layer_norm_epsilon,
        stream);
  }();
  auto output = qwen35_vision_linear(
      merged_input,
      checked_qwen35_float_tensor(session, qwen35_vision_merger_fc1_weight_id()),
      checked_qwen35_float_tensor(session, qwen35_vision_merger_fc1_bias_id()),
      stream);
  output = qwen35_vision_gelu_tanh(output, stream);
  return qwen35_vision_linear(
      output,
      checked_qwen35_float_tensor(session, qwen35_vision_merger_fc2_weight_id()),
      checked_qwen35_float_tensor(session, qwen35_vision_merger_fc2_bias_id()),
      stream);
}

mlx::core::array qwen35_vision_encode_array(
    const EdgeCmlxQwen35Session& session,
    const float* pixel_values,
    int num_patches,
    int patch_dim,
    const int* grid_thw,
    int grid_count,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  const auto& config = checked_qwen35_vision_config(session);
  const auto grids = qwen35_vision_grids_from_raw(
      grid_thw, grid_count, config.spatial_merge_size);
  const int token_count = qwen35_vision_token_count(grids);
  if (num_patches != token_count) {
    throw std::runtime_error("Qwen3.5 vision pixel/grid token mismatch");
  }
  auto pixels = array(
      pixel_values,
      Shape{num_patches, patch_dim},
      float32);
  auto hidden = qwen35_vision_patch_embed(
      session, pixels, num_patches, patch_dim, stream);
  auto pos_values = qwen35_vision_positional_embedding_values(session, grids);
  auto pos = array(
      pos_values.data(),
      Shape{token_count, config.hidden_size},
      float32);
  hidden = add(hidden, pos, stream);
  auto rotary_values = qwen35_vision_rotary_frequency_values(session, grids);
  const int rotary_width = (config.hidden_size / config.head_count) / 2;
  auto rotary = array(
      rotary_values.data(),
      Shape{token_count, rotary_width},
      float32);
  const auto cumulative_lengths = qwen35_vision_cumulative_lengths(grids);
  const auto attention_mask = qwen35_vision_attention_mask(
      cumulative_lengths,
      token_count);

  for (int layer = 0; layer < config.layer_count; ++layer) {
    auto residual = hidden;
    auto normed = qwen35_vision_layer_norm(
        hidden,
        checked_qwen35_float_tensor(session, qwen35_vision_layer_norm1_weight_id(layer)),
        checked_qwen35_float_tensor(session, qwen35_vision_layer_norm1_bias_id(layer)),
        config.layer_norm_epsilon,
        stream);
    hidden = add(
        residual,
        qwen35_vision_attention(
            session,
            layer,
            normed,
            rotary,
            attention_mask,
            stream),
        stream);
    residual = hidden;
    normed = qwen35_vision_layer_norm(
        hidden,
        checked_qwen35_float_tensor(session, qwen35_vision_layer_norm2_weight_id(layer)),
        checked_qwen35_float_tensor(session, qwen35_vision_layer_norm2_bias_id(layer)),
        config.layer_norm_epsilon,
        stream);
    auto mlp = qwen35_vision_linear(
        normed,
        checked_qwen35_float_tensor(session, qwen35_vision_layer_fc1_weight_id(layer)),
        checked_qwen35_float_tensor(session, qwen35_vision_layer_fc1_bias_id(layer)),
        stream);
    mlp = qwen35_vision_gelu_tanh(mlp, stream);
    mlp = qwen35_vision_linear(
        mlp,
        checked_qwen35_float_tensor(session, qwen35_vision_layer_fc2_weight_id(layer)),
        checked_qwen35_float_tensor(session, qwen35_vision_layer_fc2_bias_id(layer)),
        stream);
    hidden = add(residual, mlp, stream);
  }
  return qwen35_vision_patch_merger(
      session,
      hidden,
      token_count,
      stream);
}

int edge_cmlx_qwen35_session_set_vision_config(
    void* session,
    const EdgeCmlxQwen35VisionConfig* config) {
  edge_cmlx_error.clear();
  if (config == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_set_vision_config received a null config");
  }
  try {
    auto* qwen_session = checked_qwen35_session(session);
    validate_qwen35_vision_config(*config);
    qwen_session->vision_config = *config;
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_set_vision_config failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_load_vision_safetensors(
    void* session,
    const char* const* shard_paths,
    int shard_count,
    const char* vision_prefix) {
  edge_cmlx_error.clear();
  if (shard_paths == nullptr || shard_count <= 0 || vision_prefix == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_load_vision_safetensors received invalid arguments");
  }

  try {
    auto* qwen_session = checked_qwen35_session(session);
    checked_qwen35_vision_config(*qwen_session);
    std::unordered_map<std::string, mlx::core::array> tensors;
    const std::string prefix(vision_prefix);
    const std::string prefix_with_dot = prefix + ".";
    for (int i = 0; i < shard_count; ++i) {
      if (shard_paths[i] == nullptr) {
        return set_error(
            "edge_cmlx_qwen35_session_load_vision_safetensors received a null shard path");
      }
      auto loaded = mlx::core::load_safetensors(std::string(shard_paths[i]));
      for (auto& item : loaded.first) {
        if (item.first.rfind(prefix_with_dot, 0) == 0) {
          tensors.insert_or_assign(item.first, std::move(item.second));
        }
      }
    }
    register_qwen35_vision_tensors(*qwen_session, tensors, prefix);
    mlx::core::clear_cache();
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_load_vision_safetensors failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_vision_encode(
    void* session,
    const float* pixel_values,
    int num_patches,
    int patch_dim,
    const int* grid_thw,
    int grid_count,
    float* output,
    int* output_patches,
    int* output_hidden_size) {
  edge_cmlx_error.clear();
  if (pixel_values == nullptr || grid_thw == nullptr || output == nullptr ||
      output_patches == nullptr || output_hidden_size == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_vision_encode received a null pointer");
  }
  if (num_patches <= 0 || patch_dim <= 0 || grid_count <= 0) {
    return set_error(
        "edge_cmlx_qwen35_session_vision_encode received an invalid shape");
  }

  try {
    auto* qwen_session = checked_qwen35_session(session);
    const auto& config = checked_qwen35_vision_config(*qwen_session);
    auto gpu_device = mlx::core::Device{mlx::core::Device::gpu};
    auto encoded = qwen35_vision_encode_array(
        *qwen_session,
        pixel_values,
        num_patches,
        patch_dim,
        grid_thw,
        grid_count,
        gpu_device);
    mlx::core::eval(encoded);
    const int rows = static_cast<int>(encoded.shape(0));
    const int columns = static_cast<int>(encoded.shape(1));
    if (columns != config.output_hidden_size) {
      return set_error(
          "edge_cmlx_qwen35_session_vision_encode output hidden mismatch");
    }
    const size_t count =
        static_cast<size_t>(rows) * static_cast<size_t>(columns);
    const float* data = encoded.data<float>();
    std::copy(data, data + count, output);
    *output_patches = rows;
    *output_hidden_size = columns;
    mlx::core::clear_cache();
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_vision_encode failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_prefill_image_features(
    void* session,
    const int* token_ids,
    int token_count,
    const float* image_features,
    int image_feature_count,
    int hidden_size,
    int image_token_id,
    int* output_token_id) {
  return edge_cmlx_qwen35_session_prefill_media_features(
      session,
      token_ids,
      token_count,
      image_features,
      image_feature_count,
      hidden_size,
      image_token_id,
      output_token_id);
}
