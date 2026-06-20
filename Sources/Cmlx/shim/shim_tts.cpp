// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

#include "primitives.h"
#include "shim_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
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
#include "mlx/random.h"
#include "mlx/transforms.h"

namespace edge_cmlx::detail {

int qwen35_tts_text_embedding_id() {
  return 200000;
}

int qwen35_tts_text_projection_fc1_id() {
  return 200001;
}

int qwen35_tts_text_projection_fc1_bias_id() {
  return 200002;
}

int qwen35_tts_text_projection_fc2_id() {
  return 200003;
}

int qwen35_tts_text_projection_fc2_bias_id() {
  return 200004;
}

int qwen35_tts_code_predictor_projection_id() {
  return 200100;
}

int qwen35_tts_code_predictor_projection_bias_id() {
  return 200101;
}

int qwen35_tts_code_predictor_embedding_id(int codebook_index) {
  return 201000 + codebook_index;
}

int qwen35_tts_code_predictor_lm_head_id(int codebook_index) {
  return 201100 + codebook_index;
}

void register_qwen35_tts_talker_tensors(
    EdgeCmlxQwen35Session& session,
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    int group_size,
    int bits) {
  register_loaded_weight_tensor(
      session,
      qwen35_embedding_id(),
      tensors,
      "talker.model.codec_embedding.weight",
      group_size,
      bits);
  register_loaded_weight_tensor(
      session,
      qwen35_tts_text_embedding_id(),
      tensors,
      "talker.model.text_embedding.weight",
      group_size,
      bits);
  register_loaded_float_tensor(
      session,
      qwen35_final_norm_id(),
      tensors,
      "talker.model.norm.weight");
  register_loaded_weight_tensor(
      session,
      qwen35_lm_head_id(),
      tensors,
      "talker.codec_head.weight",
      group_size,
      bits);
  register_loaded_weight_tensor(
      session,
      qwen35_tts_text_projection_fc1_id(),
      tensors,
      "talker.text_projection.linear_fc1.weight",
      group_size,
      bits);
  register_loaded_float_tensor(
      session,
      qwen35_tts_text_projection_fc1_bias_id(),
      tensors,
      "talker.text_projection.linear_fc1.bias");
  register_loaded_weight_tensor(
      session,
      qwen35_tts_text_projection_fc2_id(),
      tensors,
      "talker.text_projection.linear_fc2.weight",
      group_size,
      bits);
  register_loaded_float_tensor(
      session,
      qwen35_tts_text_projection_fc2_bias_id(),
      tensors,
      "talker.text_projection.linear_fc2.bias");

  for (int layer = 0; layer < session.config.layer_count; ++layer) {
    session.layer_kinds[static_cast<size_t>(layer)] =
        EdgeCmlxQwen35LayerKindFullAttention;
    register_qwen35_full_attention_layer_tensors(
        session,
        tensors,
        "talker.model.layers." + std::to_string(layer),
        layer,
        group_size,
        bits);
  }
}

void register_qwen35_tts_code_predictor_tensors(
    EdgeCmlxQwen35Session& session,
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    int group_size,
    int bits) {
  register_loaded_float_tensor(
      session,
      qwen35_final_norm_id(),
      tensors,
      "talker.code_predictor.model.norm.weight");
  register_loaded_weight_tensor_if_exists(
      session,
      qwen35_tts_code_predictor_projection_id(),
      tensors,
      "talker.code_predictor.small_to_mtp_projection.weight",
      group_size,
      bits);
  register_loaded_float_tensor_if_exists(
      session,
      qwen35_tts_code_predictor_projection_bias_id(),
      tensors,
      "talker.code_predictor.small_to_mtp_projection.bias");
  for (int layer = 0; layer < session.config.layer_count; ++layer) {
    session.layer_kinds[static_cast<size_t>(layer)] =
        EdgeCmlxQwen35LayerKindFullAttention;
    register_qwen35_full_attention_layer_tensors(
        session,
        tensors,
        "talker.code_predictor.model.layers." + std::to_string(layer),
        layer,
        group_size,
        bits);
  }
  for (int codebook = 0; codebook < 15; ++codebook) {
    register_loaded_weight_tensor(
        session,
        qwen35_tts_code_predictor_embedding_id(codebook),
        tensors,
        "talker.code_predictor.model.codec_embedding." +
            std::to_string(codebook) + ".weight",
        group_size,
        bits);
    register_loaded_weight_tensor(
        session,
        qwen35_tts_code_predictor_lm_head_id(codebook),
        tensors,
        "talker.code_predictor.lm_head." +
            std::to_string(codebook) + ".weight",
        group_size,
        bits);
  }
}

}

using namespace edge_cmlx::detail;

mlx::core::array qwen35_tts_text_projection(
    const EdgeCmlxQwen35Session& qwen_session,
    const mlx::core::array& text_embedding,
    mlx::core::StreamOrDevice stream) {
  auto hidden = qwen35_linear_array(
      text_embedding,
      qwen_session,
      qwen35_tts_text_projection_fc1_id(),
      stream);
  hidden = qwen35_add_optional_bias(
      hidden,
      qwen_session,
      qwen35_tts_text_projection_fc1_bias_id(),
      stream);
  hidden = silu_array(hidden, stream);
  hidden = qwen35_linear_array(
      hidden,
      qwen_session,
      qwen35_tts_text_projection_fc2_id(),
      stream);
  return qwen35_add_optional_bias(
      hidden,
      qwen_session,
      qwen35_tts_text_projection_fc2_bias_id(),
      stream);
}

mlx::core::array qwen35_tts_projected_text_embedding(
    const EdgeCmlxQwen35Session& qwen_session,
    const mlx::core::array& token_ids,
    mlx::core::StreamOrDevice stream) {
  return qwen35_tts_text_projection(
      qwen_session,
      qwen35_embedding_for_indices(
          qwen_session,
          qwen35_tts_text_embedding_id(),
          token_ids,
          stream),
      stream);
}

mlx::core::array qwen35_tts_token_embedding(
    const EdgeCmlxQwen35Session& qwen_session,
    int token_id,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  return qwen35_embedding_for_indices(
      qwen_session,
      qwen35_tts_text_embedding_id(),
      array(&token_id, Shape{1}, int32),
      stream);
}

mlx::core::array qwen35_tts_projected_special_text_token(
    const EdgeCmlxQwen35Session& qwen_session,
    int token_id,
    mlx::core::StreamOrDevice stream) {
  return qwen35_tts_text_projection(
      qwen_session,
      qwen35_tts_token_embedding(qwen_session, token_id, stream),
      stream);
}

mlx::core::array qwen35_tts_codec_embedding(
    const EdgeCmlxQwen35Session& qwen_session,
    int token_id,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  return qwen35_embedding_for_indices(
      qwen_session,
      qwen35_embedding_id(),
      array(&token_id, Shape{1}, int32),
      stream);
}

mlx::core::array qwen35_tts_code_predictor_embedding(
    const EdgeCmlxQwen35Session& code_session,
    int codebook_index,
    int token_id,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  return qwen35_embedding_for_indices(
      code_session,
      qwen35_tts_code_predictor_embedding_id(codebook_index),
      array(&token_id, Shape{1}, int32),
      stream);
}

mlx::core::array qwen35_tts_code_predictor_project_input_if_needed(
    const EdgeCmlxQwen35Session& code_session,
    const mlx::core::array& input,
    mlx::core::StreamOrDevice stream) {
  if (optional_qwen35_float_tensor(
          code_session,
          qwen35_tts_code_predictor_projection_id()) == nullptr &&
      code_session.quantized_tensors.find(
          qwen35_tts_code_predictor_projection_id()) ==
          code_session.quantized_tensors.end()) {
    return input;
  }
  auto projected = qwen35_linear_array(
      input,
      code_session,
      qwen35_tts_code_predictor_projection_id(),
      stream);
  return qwen35_add_optional_bias(
      projected,
      code_session,
      qwen35_tts_code_predictor_projection_bias_id(),
      stream);
}

mlx::core::array qwen35_tts_suppressed_argmax(
    const mlx::core::array& logits,
    int vocabulary_size,
    int eos_token_id,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  auto indices = arange(0, vocabulary_size, int32, stream);
  auto suppress = logical_and(
      greater_equal(
          indices,
          array(vocabulary_size - 1024, int32),
          stream),
      not_equal(indices, array(eos_token_id, int32), stream),
      stream);
  auto masked = where(
      suppress,
      array(std::numeric_limits<float>::lowest(), logits.dtype()),
      logits,
      stream);
  return argmax(masked, -1, false, stream);
}

mlx::core::array qwen35_tts_sample_token(
    const mlx::core::array& logits,
    float temperature,
    int top_k,
    uint64_t seed,
    bool suppress_special_tail,
    int vocabulary_size,
    int eos_token_id,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  array processed = logits;
  if (processed.ndim() == 1) {
    processed = expand_dims(processed, 0, stream);
  }
  const int vocab = static_cast<int>(processed.shape(-1));
  if (suppress_special_tail) {
    auto indices = arange(0, vocabulary_size, int32, stream);
    auto suppress = logical_and(
        greater_equal(
            indices,
            array(vocabulary_size - 1024, int32),
            stream),
        not_equal(indices, array(eos_token_id, int32), stream),
        stream);
    processed = where(
        suppress,
        array(std::numeric_limits<float>::lowest(), processed.dtype()),
        processed,
        stream);
  }
  if (temperature <= 0.0f) {
    return argmax(processed, -1, false, stream);
  }

  const auto key = random::key(seed);
  if (top_k > 0 && top_k < vocab) {
    auto sorted_indices = argsort(processed, -1, stream);
    auto top_indices = slice(
        sorted_indices,
        Shape{0, vocab - top_k},
        Shape{1, vocab},
        stream);
    auto top_logits = take_along_axis(processed, top_indices, -1, stream);
    auto sampled_top_index = random::categorical(
        divide(
            top_logits,
            array(temperature, top_logits.dtype()),
            stream),
        -1,
        std::optional<array>(key),
        stream);
    auto flattened_top_indices = reshape(top_indices, Shape{top_k}, stream);
    return take(flattened_top_indices, sampled_top_index, stream);
  }
  return random::categorical(
      divide(
          processed,
          array(temperature, processed.dtype()),
          stream),
      -1,
      std::optional<array>(key),
      stream);
}

struct Qwen35TTSPreparedInputs {
  mlx::core::array input_embeddings;
  mlx::core::array trailing_text_hidden;
  mlx::core::array tts_pad_embedding;
};

Qwen35TTSPreparedInputs qwen35_tts_prepare_generation_inputs(
    EdgeCmlxQwen35Session& qwen_session,
    const int* target_token_ids,
    int target_token_count,
    int tts_bos_token_id,
    int tts_eos_token_id,
    int tts_pad_token_id,
    const int* codec_prefix_ids,
    int codec_prefix_count,
    int codec_pad_id,
    int codec_bos_id,
    int speaker_id,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  if (target_token_count < 9) {
    throw std::runtime_error("Qwen3 TTS target prompt is too short");
  }

  auto target_ids = array(target_token_ids, Shape{target_token_count}, int32);
  auto text_embed = qwen35_tts_projected_text_embedding(
      qwen_session,
      target_ids,
      stream);
  const int hidden_size = qwen_session.config.hidden_size;
  auto tts_bos = qwen35_tts_projected_special_text_token(
      qwen_session,
      tts_bos_token_id,
      stream);
  auto tts_eos = qwen35_tts_projected_special_text_token(
      qwen_session,
      tts_eos_token_id,
      stream);
  auto tts_pad = qwen35_tts_projected_special_text_token(
      qwen_session,
      tts_pad_token_id,
      stream);

  std::vector<array> codec_parts;
  codec_parts.push_back(qwen35_embedding_for_indices(
      qwen_session,
      qwen35_embedding_id(),
      array(codec_prefix_ids, Shape{codec_prefix_count}, int32),
      stream));
  if (speaker_id >= 0) {
    codec_parts.push_back(qwen35_tts_codec_embedding(
        qwen_session,
        speaker_id,
        stream));
  }
  int suffix_ids[2] = {codec_pad_id, codec_bos_id};
  codec_parts.push_back(qwen35_embedding_for_indices(
      qwen_session,
      qwen35_embedding_id(),
      array(suffix_ids, Shape{2}, int32),
      stream));
  auto codec_embed = concatenate(codec_parts, 0, stream);
  const int codec_len = static_cast<int>(codec_embed.shape(0));
  if (codec_len < 2) {
    throw std::runtime_error("Qwen3 TTS codec prefix is invalid");
  }

  auto role_embed = slice(
      text_embed,
      Shape{0, 0},
      Shape{3, hidden_size},
      stream);
  const int pad_count = codec_len - 2;
  auto pad_embeds = broadcast_to(
      tts_pad,
      Shape{pad_count, hidden_size},
      stream);
  auto combined = concatenate(
      std::vector<array>{pad_embeds, tts_bos},
      0,
      stream);
  combined = add(
      combined,
      slice(
          codec_embed,
          Shape{0, 0},
          Shape{codec_len - 1, hidden_size},
          stream),
      stream);
  auto first_text = add(
      slice(
          text_embed,
          Shape{3, 0},
          Shape{4, hidden_size},
          stream),
      slice(
          codec_embed,
          Shape{codec_len - 1, 0},
          Shape{codec_len, hidden_size},
          stream),
      stream);
  auto input_embeddings = concatenate(
      std::vector<array>{role_embed, combined, first_text},
      0,
      stream);

  const int trailing_start = 4;
  const int trailing_end = target_token_count - 5;
  array trailing = tts_eos;
  if (trailing_end > trailing_start) {
    trailing = concatenate(
        std::vector<array>{
            slice(
                text_embed,
                Shape{trailing_start, 0},
                Shape{trailing_end, hidden_size},
                stream),
            tts_eos},
        0,
        stream);
  }

  return Qwen35TTSPreparedInputs{input_embeddings, trailing, tts_pad};
}

std::vector<int> qwen35_tts_predict_codebooks(
    EdgeCmlxQwen35Session& talker_session,
    EdgeCmlxQwen35Session& code_session,
    const mlx::core::array& code_hidden,
    int first_code,
    float temperature,
    int top_k,
    uint64_t seed,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  std::vector<int> code_tokens;
  code_tokens.reserve(16);
  code_tokens.push_back(first_code);

  qwen35_reset_decode_cache(code_session);
  for (int codebook = 0; codebook < 15; ++codebook) {
    array code_input = [&]() -> array {
      if (codebook == 0) {
        auto first_code_embed = qwen35_tts_codec_embedding(
            talker_session,
            first_code,
            stream);
        return concatenate(
            std::vector<array>{code_hidden, first_code_embed},
            0,
            stream);
      }
      return qwen35_tts_code_predictor_embedding(
          code_session,
          codebook - 1,
          code_tokens[static_cast<size_t>(codebook)],
          stream);
    }();
    code_input = qwen35_tts_code_predictor_project_input_if_needed(
        code_session,
        code_input,
        stream);
    const int token_count = static_cast<int>(code_input.shape(0));
    auto result = qwen35_session_advance_hidden_with_state(
        code_session,
        code_input,
        token_count,
        false,
        0.0f,
        0,
        1.0f,
        0.0f,
        0,
        "edge_cmlx_qwen35_session_tts_generate_codes(code_predictor)",
        qwen35_tts_code_predictor_lm_head_id(codebook));
    auto sampled = qwen35_tts_sample_token(
        result.logits,
        temperature,
        top_k,
        seed + static_cast<uint64_t>(codebook + 1),
        false,
        code_session.config.vocabulary_size,
        -1,
        stream);
    eval(sampled);
    code_tokens.push_back(static_cast<int>(sampled.data<uint32_t>()[0]));
  }
  return code_tokens;
}

const mlx::core::array& checked_qwen35_tts_speech_tensor(
    const EdgeCmlxQwen35Session& session,
    const std::string& name) {
  const auto item = session.tts_speech_tensors.find(name);
  if (item == session.tts_speech_tensors.end()) {
    throw std::runtime_error(
        "Qwen3 TTS speech tokenizer tensor is not registered: " + name);
  }
  return item->second;
}

const mlx::core::array* optional_qwen35_tts_speech_tensor(
    const EdgeCmlxQwen35Session& session,
    const std::string& name) {
  const auto item = session.tts_speech_tensors.find(name);
  if (item == session.tts_speech_tensors.end()) {
    return nullptr;
  }
  return &item->second;
}

mlx::core::array qwen35_tts_speech_cast(
    const mlx::core::array& value,
    mlx::core::Dtype dtype,
    mlx::core::StreamOrDevice stream) {
  return value.dtype() == dtype
      ? value
      : mlx::core::astype(value, dtype, stream);
}

mlx::core::array qwen35_tts_speech_tensor(
    const EdgeCmlxQwen35Session& session,
    const std::string& name,
    mlx::core::Dtype dtype,
    mlx::core::StreamOrDevice stream) {
  return qwen35_tts_speech_cast(
      checked_qwen35_tts_speech_tensor(session, name),
      dtype,
      stream);
}

mlx::core::array qwen35_tts_speech_conv_weight(
    const EdgeCmlxQwen35Session& session,
    const std::string& name,
    mlx::core::Dtype dtype,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  auto weight = qwen35_tts_speech_tensor(session, name, dtype, stream);
  if (weight.ndim() != 3) {
    throw std::runtime_error("Qwen3 TTS conv weight must be rank 3: " + name);
  }
  return transpose(weight, {0, 2, 1}, stream);
}

mlx::core::array qwen35_tts_speech_transpose_conv_weight(
    const EdgeCmlxQwen35Session& session,
    const std::string& name,
    mlx::core::Dtype dtype,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  auto weight = qwen35_tts_speech_tensor(session, name, dtype, stream);
  if (weight.ndim() != 3) {
    throw std::runtime_error(
        "Qwen3 TTS transpose conv weight must be rank 3: " + name);
  }
  return transpose(weight, {1, 2, 0}, stream);
}

mlx::core::array qwen35_tts_speech_add_bias_nct(
    const EdgeCmlxQwen35Session& session,
    const mlx::core::array& input,
    const std::string& bias_name,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  const auto* bias = optional_qwen35_tts_speech_tensor(session, bias_name);
  if (bias == nullptr) {
    return input;
  }
  auto typed_bias = qwen35_tts_speech_cast(*bias, input.dtype(), stream);
  return add(
      input,
      reshape(typed_bias, Shape{1, typed_bias.shape(0), 1}, stream),
      stream);
}

mlx::core::array qwen35_tts_speech_conv1d_nct(
    const EdgeCmlxQwen35Session& session,
    const mlx::core::array& input,
    const std::string& weight_name,
    const std::string& bias_name,
    int kernel_size,
    int dilation,
    int groups,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  auto x = transpose(input, {0, 2, 1}, stream);
  const int left_padding = (kernel_size - 1) * dilation;
  if (left_padding > 0) {
    x = pad(
        x,
        std::vector<std::pair<int, int>>{{0, 0}, {left_padding, 0}, {0, 0}},
        array(0.0f, x.dtype()),
        "constant",
        stream);
  }
  auto weight = qwen35_tts_speech_conv_weight(
      session,
      weight_name,
      x.dtype(),
      stream);
  auto output = conv1d(x, weight, 1, 0, dilation, groups, stream);
  output = transpose(output, {0, 2, 1}, stream);
  return qwen35_tts_speech_add_bias_nct(
      session,
      output,
      bias_name,
      stream);
}

mlx::core::array qwen35_tts_speech_conv_transpose1d_nct(
    const EdgeCmlxQwen35Session& session,
    const mlx::core::array& input,
    const std::string& weight_name,
    const std::string& bias_name,
    int kernel_size,
    int stride,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  auto x = transpose(input, {0, 2, 1}, stream);
  auto weight = qwen35_tts_speech_transpose_conv_weight(
      session,
      weight_name,
      x.dtype(),
      stream);
  auto output = conv_transpose1d(x, weight, stride, 0, 1, 0, 1, stream);
  const auto* bias = optional_qwen35_tts_speech_tensor(session, bias_name);
  if (bias != nullptr) {
    output = add(
        output,
        qwen35_tts_speech_cast(*bias, output.dtype(), stream),
        stream);
  }
  const int trim = kernel_size - stride;
  if (trim > 0) {
    const int batch = static_cast<int>(output.shape(0));
    const int time = static_cast<int>(output.shape(1));
    const int channels = static_cast<int>(output.shape(2));
    output = slice(
        output,
        Shape{0, 0, 0},
        Shape{batch, time - trim, channels},
        stream);
  }
  return transpose(output, {0, 2, 1}, stream);
}

mlx::core::array qwen35_tts_speech_linear(
    const EdgeCmlxQwen35Session& session,
    const mlx::core::array& input,
    const std::string& weight_name,
    const std::string& bias_name,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  auto weight = qwen35_tts_speech_tensor(
      session,
      weight_name,
      input.dtype(),
      stream);
  const auto* bias = optional_qwen35_tts_speech_tensor(session, bias_name);
  return edge_cmlx::primitives::linear(
      input,
      weight,
      bias,
      stream);
}

mlx::core::array qwen35_tts_speech_linear_no_bias(
    const EdgeCmlxQwen35Session& session,
    const mlx::core::array& input,
    const std::string& weight_name,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  auto weight = qwen35_tts_speech_tensor(
      session,
      weight_name,
      input.dtype(),
      stream);
  return edge_cmlx::primitives::linear(input, weight, nullptr, stream);
}

mlx::core::array qwen35_tts_speech_rms_norm(
    const EdgeCmlxQwen35Session& session,
    const mlx::core::array& input,
    const std::string& weight_name,
    float epsilon,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  auto weight = qwen35_tts_speech_tensor(
      session,
      weight_name,
      input.dtype(),
      stream);
  return edge_cmlx::primitives::rms_norm(
      input,
      weight,
      epsilon,
      stream);
}

mlx::core::array qwen35_tts_speech_layer_norm(
    const EdgeCmlxQwen35Session& session,
    const mlx::core::array& input,
    const std::string& weight_name,
    const std::string& bias_name,
    float epsilon,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  auto weight = qwen35_tts_speech_tensor(
      session,
      weight_name,
      input.dtype(),
      stream);
  auto bias = qwen35_tts_speech_tensor(
      session,
      bias_name,
      input.dtype(),
      stream);
  return edge_cmlx::primitives::layer_norm(
      input,
      weight,
      &bias,
      epsilon,
      stream);
}

mlx::core::array qwen35_tts_speech_snake(
    const EdgeCmlxQwen35Session& session,
    const mlx::core::array& input,
    const std::string& prefix,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  const int channels = static_cast<int>(input.shape(1));
  auto alpha = exp(
      reshape(
          qwen35_tts_speech_tensor(
              session,
              prefix + ".alpha",
              input.dtype(),
              stream),
          Shape{1, channels, 1},
          stream),
      stream);
  auto beta = exp(
      reshape(
          qwen35_tts_speech_tensor(
              session,
              prefix + ".beta",
              input.dtype(),
              stream),
          Shape{1, channels, 1},
          stream),
      stream);
  return edge_cmlx::primitives::snake(input, alpha, beta, stream);
}

mlx::core::array qwen35_tts_speech_codebook_embedding(
    const EdgeCmlxQwen35Session& session,
    const std::string& prefix,
    const mlx::core::array& codes,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  auto embedding_sum = checked_qwen35_tts_speech_tensor(
      session,
      prefix + "._codebook.embedding_sum");
  auto cluster_usage = checked_qwen35_tts_speech_tensor(
      session,
      prefix + "._codebook.cluster_usage");
  auto usage = maximum(
      expand_dims(cluster_usage, 1, stream),
      array(1e-5f, cluster_usage.dtype()),
      stream);
  auto embedding = divide(embedding_sum, usage, stream);
  return take(embedding, codes, 0, stream);
}

mlx::core::array qwen35_tts_speech_quantizer_decode(
    const EdgeCmlxQwen35Session& session,
    const mlx::core::array& codes,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  const int batch = static_cast<int>(codes.shape(0));
  const int time = static_cast<int>(codes.shape(2));
  auto first_codes = reshape(
      slice(codes, Shape{0, 0, 0}, Shape{batch, 1, time}, stream),
      Shape{batch, time},
      stream);
  auto first = transpose(
      qwen35_tts_speech_codebook_embedding(
          session,
          "decoder.quantizer.rvq_first.vq.layers.0",
          first_codes,
          stream),
      {0, 2, 1},
      stream);
  first = qwen35_tts_speech_conv1d_nct(
      session,
      first,
      "decoder.quantizer.rvq_first.output_proj.weight",
      "decoder.quantizer.rvq_first.output_proj.bias",
      1,
      1,
      1,
      stream);

  mlx::core::array rest = zeros_like(first, stream);
  for (int index = 0; index < 15; ++index) {
    auto layer_codes = reshape(
        slice(
            codes,
            Shape{0, index + 1, 0},
            Shape{batch, index + 2, time},
            stream),
        Shape{batch, time},
        stream);
    auto embedded = transpose(
        qwen35_tts_speech_codebook_embedding(
            session,
            "decoder.quantizer.rvq_rest.vq.layers." + std::to_string(index),
            layer_codes,
            stream),
        {0, 2, 1},
        stream);
    rest = index == 0 ? embedded : add(rest, embedded, stream);
  }
  rest = qwen35_tts_speech_conv1d_nct(
      session,
      rest,
      "decoder.quantizer.rvq_rest.output_proj.weight",
      "decoder.quantizer.rvq_rest.output_proj.bias",
      1,
      1,
      1,
      stream);
  return add(first, rest, stream);
}

mlx::core::array qwen35_tts_speech_transformer(
    const EdgeCmlxQwen35Session& session,
    const mlx::core::array& input,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  constexpr int kLayerCount = 8;
  constexpr int kHeads = 16;
  constexpr int kHeadDim = 64;
  constexpr float kEpsilon = 1e-5f;

  auto hidden = qwen35_tts_speech_linear(
      session,
      input,
      "decoder.pre_transformer.input_proj.weight",
      "decoder.pre_transformer.input_proj.bias",
      stream);
  const int batch = static_cast<int>(hidden.shape(0));
  const int time = static_cast<int>(hidden.shape(1));

  for (int layer = 0; layer < kLayerCount; ++layer) {
    const std::string prefix =
        "decoder.pre_transformer.layers." + std::to_string(layer);
    auto residual = hidden;
    auto normed = qwen35_tts_speech_rms_norm(
        session,
        hidden,
        prefix + ".input_layernorm.weight",
        kEpsilon,
        stream);
    auto query = qwen35_tts_speech_linear_no_bias(
        session,
        normed,
        prefix + ".self_attn.q_proj.weight",
        stream);
    auto key = qwen35_tts_speech_linear_no_bias(
        session,
        normed,
        prefix + ".self_attn.k_proj.weight",
        stream);
    auto value = qwen35_tts_speech_linear_no_bias(
        session,
        normed,
        prefix + ".self_attn.v_proj.weight",
        stream);
    query = transpose(
        reshape(query, Shape{batch, time, kHeads, kHeadDim}, stream),
        {0, 2, 1, 3},
        stream);
    key = transpose(
        reshape(key, Shape{batch, time, kHeads, kHeadDim}, stream),
        {0, 2, 1, 3},
        stream);
    value = transpose(
        reshape(value, Shape{batch, time, kHeads, kHeadDim}, stream),
        {0, 2, 1, 3},
        stream);
    auto attention = fast::scaled_dot_product_attention(
        query,
        key,
        value,
        std::pow(static_cast<float>(kHeadDim), -0.5f),
        "",
        std::nullopt,
        std::nullopt,
        stream);
    attention = reshape(
        transpose(attention, {0, 2, 1, 3}, stream),
        Shape{batch, time, kHeads * kHeadDim},
        stream);
    attention = qwen35_tts_speech_linear_no_bias(
        session,
        attention,
        prefix + ".self_attn.o_proj.weight",
        stream);
    attention = multiply(
        attention,
        qwen35_tts_speech_tensor(
            session,
            prefix + ".self_attn_layer_scale.scale",
            attention.dtype(),
            stream),
        stream);
    hidden = add(residual, attention, stream);

    residual = hidden;
    normed = qwen35_tts_speech_rms_norm(
        session,
        hidden,
        prefix + ".post_attention_layernorm.weight",
        kEpsilon,
        stream);
    auto gate = qwen35_tts_speech_linear_no_bias(
        session,
        normed,
        prefix + ".mlp.gate_proj.weight",
        stream);
    auto up = qwen35_tts_speech_linear_no_bias(
        session,
        normed,
        prefix + ".mlp.up_proj.weight",
        stream);
    auto mlp = qwen35_tts_speech_linear_no_bias(
        session,
        multiply(silu_array(gate, stream), up, stream),
        prefix + ".mlp.down_proj.weight",
        stream);
    mlp = multiply(
        mlp,
        qwen35_tts_speech_tensor(
            session,
            prefix + ".mlp_layer_scale.scale",
            mlp.dtype(),
            stream),
        stream);
    hidden = add(residual, mlp, stream);
  }

  hidden = qwen35_tts_speech_rms_norm(
      session,
      hidden,
      "decoder.pre_transformer.norm.weight",
      kEpsilon,
      stream);
  return qwen35_tts_speech_linear(
      session,
      hidden,
      "decoder.pre_transformer.output_proj.weight",
      "decoder.pre_transformer.output_proj.bias",
      stream);
}

mlx::core::array qwen35_tts_speech_convnext_block(
    const EdgeCmlxQwen35Session& session,
    const mlx::core::array& input,
    const std::string& prefix,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  const int channels = static_cast<int>(input.shape(1));
  auto residual = input;
  auto hidden = qwen35_tts_speech_conv1d_nct(
      session,
      input,
      prefix + ".dwconv.conv.weight",
      prefix + ".dwconv.conv.bias",
      7,
      1,
      channels,
      stream);
  hidden = transpose(hidden, {0, 2, 1}, stream);
  hidden = qwen35_tts_speech_layer_norm(
      session,
      hidden,
      prefix + ".norm.weight",
      prefix + ".norm.bias",
      1e-6f,
      stream);
  hidden = qwen35_tts_speech_linear(
      session,
      hidden,
      prefix + ".pwconv1.weight",
      prefix + ".pwconv1.bias",
      stream);
  hidden = qwen35_vision_gelu_tanh(hidden, stream);
  hidden = qwen35_tts_speech_linear(
      session,
      hidden,
      prefix + ".pwconv2.weight",
      prefix + ".pwconv2.bias",
      stream);
  hidden = multiply(
      hidden,
      qwen35_tts_speech_tensor(
          session,
          prefix + ".gamma",
          hidden.dtype(),
          stream),
      stream);
  hidden = transpose(hidden, {0, 2, 1}, stream);
  return add(residual, hidden, stream);
}

mlx::core::array qwen35_tts_speech_residual_unit(
    const EdgeCmlxQwen35Session& session,
    const mlx::core::array& input,
    const std::string& prefix,
    int dilation,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  auto residual = input;
  auto hidden = qwen35_tts_speech_snake(
      session,
      input,
      prefix + ".act1",
      stream);
  hidden = qwen35_tts_speech_conv1d_nct(
      session,
      hidden,
      prefix + ".conv1.conv.weight",
      prefix + ".conv1.conv.bias",
      7,
      dilation,
      1,
      stream);
  hidden = qwen35_tts_speech_snake(
      session,
      hidden,
      prefix + ".act2",
      stream);
  hidden = qwen35_tts_speech_conv1d_nct(
      session,
      hidden,
      prefix + ".conv2.conv.weight",
      prefix + ".conv2.conv.bias",
      1,
      1,
      1,
      stream);
  return add(residual, hidden, stream);
}

mlx::core::array qwen35_tts_speech_decoder_block(
    const EdgeCmlxQwen35Session& session,
    const mlx::core::array& input,
    int block_index,
    int upsample_rate,
    mlx::core::StreamOrDevice stream) {
  const std::string prefix =
      "decoder.decoder." + std::to_string(block_index) + ".block";
  auto hidden = qwen35_tts_speech_snake(
      session,
      input,
      prefix + ".0",
      stream);
  hidden = qwen35_tts_speech_conv_transpose1d_nct(
      session,
      hidden,
      prefix + ".1.conv.weight",
      prefix + ".1.conv.bias",
      upsample_rate * 2,
      upsample_rate,
      stream);
  hidden = qwen35_tts_speech_residual_unit(
      session,
      hidden,
      prefix + ".2",
      1,
      stream);
  hidden = qwen35_tts_speech_residual_unit(
      session,
      hidden,
      prefix + ".3",
      3,
      stream);
  return qwen35_tts_speech_residual_unit(
      session,
      hidden,
      prefix + ".4",
      9,
      stream);
}

mlx::core::array qwen35_tts_speech_decode_codes(
    const EdgeCmlxQwen35Session& session,
    const mlx::core::array& code_steps,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  if (session.tts_speech_tensors.empty()) {
    throw std::runtime_error("Qwen3 TTS speech tokenizer weights are not loaded");
  }
  if (code_steps.ndim() != 2 || code_steps.shape(1) != 16) {
    throw std::runtime_error("Qwen3 TTS speech tokenizer codes must be [steps, 16]");
  }
  const int steps = static_cast<int>(code_steps.shape(0));
  auto codes = transpose(
      reshape(code_steps, Shape{1, steps, 16}, stream),
      {0, 2, 1},
      stream);

  auto hidden = qwen35_tts_speech_quantizer_decode(session, codes, stream);
  hidden = qwen35_tts_speech_conv1d_nct(
      session,
      hidden,
      "decoder.pre_conv.conv.weight",
      "decoder.pre_conv.conv.bias",
      3,
      1,
      1,
      stream);
  hidden = transpose(hidden, {0, 2, 1}, stream);
  hidden = qwen35_tts_speech_transformer(session, hidden, stream);
  hidden = transpose(hidden, {0, 2, 1}, stream);

  for (int upsample = 0; upsample < 2; ++upsample) {
    const std::string prefix =
        "decoder.upsample." + std::to_string(upsample);
    hidden = qwen35_tts_speech_conv_transpose1d_nct(
        session,
        hidden,
        prefix + ".0.conv.weight",
        prefix + ".0.conv.bias",
        2,
        2,
        stream);
    hidden = qwen35_tts_speech_convnext_block(
        session,
        hidden,
        prefix + ".1",
        stream);
  }

  auto wav = qwen35_tts_speech_conv1d_nct(
      session,
      hidden,
      "decoder.decoder.0.conv.weight",
      "decoder.decoder.0.conv.bias",
      7,
      1,
      1,
      stream);
  const int rates[4] = {8, 5, 4, 3};
  for (int block = 0; block < 4; ++block) {
    wav = qwen35_tts_speech_decoder_block(
        session,
        wav,
        block + 1,
        rates[block],
        stream);
  }
  wav = qwen35_tts_speech_snake(
      session,
      wav,
      "decoder.decoder.5",
      stream);
  wav = qwen35_tts_speech_conv1d_nct(
      session,
      wav,
      "decoder.decoder.6.conv.weight",
      "decoder.decoder.6.conv.bias",
      7,
      1,
      1,
      stream);
  wav = clip(
      wav,
      std::optional<array>(array(-1.0f, wav.dtype())),
      std::optional<array>(array(1.0f, wav.dtype())),
      stream);
  return reshape(wav, Shape{static_cast<ShapeElem>(wav.size())}, stream);
}

int edge_cmlx_qwen35_session_load_tts_safetensors(
    void* session,
    const char* const* shard_paths,
    int shard_count,
    int group_size,
    int bits,
    const EdgeCmlxQwen35Config* code_predictor_config) {
  edge_cmlx_error.clear();
  if (shard_paths == nullptr || shard_count <= 0 ||
      code_predictor_config == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_load_tts_safetensors received invalid arguments");
  }
  if (group_size <= 0 || bits <= 0) {
    return set_error(
        "edge_cmlx_qwen35_session_load_tts_safetensors received invalid quantization");
  }
  try {
    validate_qwen35_config(*code_predictor_config);
    auto* qwen_session = checked_qwen35_session(session);
    std::unordered_map<std::string, mlx::core::array> tensors;
    for (int i = 0; i < shard_count; ++i) {
      if (shard_paths[i] == nullptr) {
        return set_error(
            "edge_cmlx_qwen35_session_load_tts_safetensors received a null shard path");
      }
      auto loaded = mlx::core::load_safetensors(std::string(shard_paths[i]));
      for (auto& item : loaded.first) {
        tensors.insert_or_assign(item.first, std::move(item.second));
      }
    }

    register_qwen35_tts_talker_tensors(
        *qwen_session,
        tensors,
        group_size,
        bits);
    qwen_session->tts_code_predictor_session =
        std::make_unique<EdgeCmlxQwen35Session>(*code_predictor_config);
    register_qwen35_tts_code_predictor_tensors(
        *qwen_session->tts_code_predictor_session,
        tensors,
        group_size,
        bits);
    mlx::core::clear_cache();
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_load_tts_safetensors failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_load_tts_speech_tokenizer_safetensors(
    void* session,
    const char* model_path) {
  edge_cmlx_error.clear();
  if (model_path == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_load_tts_speech_tokenizer_safetensors received a null model path");
  }
  try {
    auto* qwen_session = checked_qwen35_session(session);
    auto loaded = mlx::core::load_safetensors(std::string(model_path));
    qwen_session->tts_speech_tensors.clear();
    for (auto& item : loaded.first) {
      if (item.first.rfind("decoder.", 0) == 0) {
        qwen_session->tts_speech_tensors.insert_or_assign(
            item.first,
            std::move(item.second));
      }
    }
    if (qwen_session->tts_speech_tensors.empty()) {
      return set_error(
          "edge_cmlx_qwen35_session_load_tts_speech_tokenizer_safetensors found no decoder tensors");
    }
    mlx::core::clear_cache();
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_load_tts_speech_tokenizer_safetensors failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_tts_generate_codes(
    void* session,
    const int* target_token_ids,
    int target_token_count,
    int tts_bos_token_id,
    int tts_eos_token_id,
    int tts_pad_token_id,
    const int* codec_prefix_ids,
    int codec_prefix_count,
    int codec_pad_id,
    int codec_bos_id,
    int codec_eos_token_id,
    int speaker_id,
    int max_tokens,
    float temperature,
    int top_k,
    uint64_t seed,
    int* output_codes,
    int output_code_capacity,
    int* output_steps,
    int* output_codebooks) {
  edge_cmlx_error.clear();
  if (target_token_ids == nullptr || codec_prefix_ids == nullptr ||
      output_codes == nullptr || output_steps == nullptr ||
      output_codebooks == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_tts_generate_codes received a null pointer");
  }
  if (target_token_count <= 0 || codec_prefix_count <= 0 ||
      max_tokens <= 0 || output_code_capacity <= 0) {
    return set_error(
        "edge_cmlx_qwen35_session_tts_generate_codes received an invalid shape");
  }
  if (temperature < 0.0f || !std::isfinite(temperature) || top_k < 0) {
    return set_error(
        "edge_cmlx_qwen35_session_tts_generate_codes received invalid sampling parameters");
  }
  try {
    using namespace mlx::core;
    auto* qwen_session = checked_qwen35_session(session);
    auto* code_session = qwen_session->tts_code_predictor_session.get();
    if (code_session == nullptr) {
      return set_error(
          "edge_cmlx_qwen35_session_tts_generate_codes missing code predictor");
    }
    validate_qwen35_decode_layer_kinds(
        *qwen_session,
        "edge_cmlx_qwen35_session_tts_generate_codes(talker)");
    validate_qwen35_decode_layer_kinds(
        *code_session,
        "edge_cmlx_qwen35_session_tts_generate_codes(code_predictor)");
    const int max_output_steps = output_code_capacity / 16;
    if (max_output_steps <= 0) {
      return set_error(
          "edge_cmlx_qwen35_session_tts_generate_codes output buffer is too small");
    }
    const int effective_max_tokens = std::min(max_tokens, max_output_steps);
    auto gpu_device = Device{Device::gpu};
    qwen35_reset_decode_cache(*qwen_session);
    qwen35_reset_decode_cache(*code_session);

    auto prepared = qwen35_tts_prepare_generation_inputs(
        *qwen_session,
        target_token_ids,
        target_token_count,
        tts_bos_token_id,
        tts_eos_token_id,
        tts_pad_token_id,
        codec_prefix_ids,
        codec_prefix_count,
        codec_pad_id,
        codec_bos_id,
        speaker_id,
        gpu_device);
    auto current_input = prepared.input_embeddings;
    int trailing_index = 0;
    int generated_steps = 0;

    for (int step = 0; step < effective_max_tokens; ++step) {
      const int token_count = static_cast<int>(current_input.shape(0));
      auto talker_result = qwen35_session_advance_hidden_with_state(
          *qwen_session,
          current_input,
          token_count,
          false,
          0.0f,
          0,
          1.0f,
          0.0f,
          0,
          "edge_cmlx_qwen35_session_tts_generate_codes(talker)",
          qwen35_lm_head_id());
      auto first_code_token = qwen35_tts_sample_token(
          talker_result.logits,
          temperature,
          top_k,
          seed + static_cast<uint64_t>(step * 17),
          true,
          qwen_session->config.vocabulary_size,
          codec_eos_token_id,
          gpu_device);
      eval(first_code_token);
      const int first_code =
          static_cast<int>(first_code_token.data<uint32_t>()[0]);
      if (first_code == codec_eos_token_id) {
        break;
      }

      auto code_tokens = qwen35_tts_predict_codebooks(
          *qwen_session,
          *code_session,
          talker_result.last_hidden,
          first_code,
          temperature,
          top_k,
          seed + static_cast<uint64_t>(step * 17 + 1),
          gpu_device);
      if (code_tokens.size() != 16) {
        return set_error(
            "edge_cmlx_qwen35_session_tts_generate_codes codebook count mismatch");
      }
      for (int codebook = 0; codebook < 16; ++codebook) {
        output_codes[generated_steps * 16 + codebook] =
            code_tokens[static_cast<size_t>(codebook)];
      }
      generated_steps += 1;

      array text_embed = [&]() -> array {
        if (trailing_index <
            static_cast<int>(prepared.trailing_text_hidden.shape(0))) {
          auto value = slice(
              prepared.trailing_text_hidden,
              Shape{trailing_index, 0},
              Shape{trailing_index + 1, qwen_session->config.hidden_size},
              gpu_device);
          trailing_index += 1;
          return value;
        }
        return prepared.tts_pad_embedding;
      }();

      auto codec_embed = qwen35_tts_codec_embedding(
          *qwen_session,
          first_code,
          gpu_device);
      for (int codebook = 1; codebook < 16; ++codebook) {
        codec_embed = add(
            codec_embed,
            qwen35_tts_code_predictor_embedding(
                *code_session,
                codebook - 1,
                code_tokens[static_cast<size_t>(codebook)],
                gpu_device),
            gpu_device);
      }
      current_input = add(text_embed, codec_embed, gpu_device);
      mlx::core::clear_cache();
    }

    if (generated_steps <= 0) {
      return set_error(
          "edge_cmlx_qwen35_session_tts_generate_codes generated no codec tokens");
    }
    *output_steps = generated_steps;
    *output_codebooks = 16;
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_tts_generate_codes failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_tts_decode_audio_codes(
    void* session,
    const int* codes,
    int step_count,
    int codebook_count,
    float* output_samples,
    int output_sample_capacity,
    int* output_sample_count) {
  edge_cmlx_error.clear();
  if (codes == nullptr || output_samples == nullptr ||
      output_sample_count == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_tts_decode_audio_codes received a null pointer");
  }
  if (step_count <= 0 || codebook_count != 16 ||
      output_sample_capacity <= 0) {
    return set_error(
        "edge_cmlx_qwen35_session_tts_decode_audio_codes received an invalid shape");
  }
  try {
    using namespace mlx::core;
    auto* qwen_session = checked_qwen35_session(session);
    auto gpu_device = Device{Device::gpu};
    auto code_array = array(
        codes,
        Shape{step_count, codebook_count},
        int32);
    auto audio = qwen35_tts_speech_decode_codes(
        *qwen_session,
        code_array,
        gpu_device);
    audio = astype(audio, float32, gpu_device);
    eval(audio);
    const int sample_count = static_cast<int>(audio.size());
    if (sample_count > output_sample_capacity) {
      return set_error(
          "edge_cmlx_qwen35_session_tts_decode_audio_codes output buffer is too small");
    }
    std::memcpy(
        output_samples,
        audio.data<float>(),
        static_cast<size_t>(sample_count) * sizeof(float));
    *output_sample_count = sample_count;
    mlx::core::clear_cache();
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_tts_decode_audio_codes failed with an unknown error");
  }
}
