// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

#pragma once

#include "CmlxShim.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mlx/array.h"
#include "mlx/ops.h"

namespace edge_cmlx::detail {

struct EdgeCmlxQuantizedArray {
  mlx::core::array packed;
  mlx::core::array scales;
  mlx::core::array biases;
  int group_size;
  int bits;
};

struct EdgeCmlxQwen35EvalProfileCounters {
  uint64_t eval_barriers = 0;
  uint64_t sync_eval_barriers = 0;
  uint64_t async_eval_barriers = 0;
  uint64_t token_read_barriers = 0;
  uint64_t eval_outputs = 0;
  uint64_t eval_bytes = 0;
  uint64_t token_read_bytes = 0;
  double eval_elapsed_ms = 0.0;
  double token_read_elapsed_ms = 0.0;
};

struct EdgeCmlxQwen35EvalProfileInventoryBucket {
  uint64_t count = 0;
  uint64_t bytes = 0;
};

struct EdgeCmlxQwen35Session {
  explicit EdgeCmlxQwen35Session(EdgeCmlxQwen35Config config)
      : config(std::move(config)),
        layer_kinds(static_cast<size_t>(this->config.layer_count), 0) {}

  EdgeCmlxQwen35Config config;
  std::vector<int> layer_kinds;
  std::unordered_map<int, mlx::core::array> float_tensors;
  std::unordered_map<int, EdgeCmlxQuantizedArray> quantized_tensors;
  std::unordered_map<int, mlx::core::array> gdn_neg_exp_a_log_tensors;
  std::unordered_map<int, mlx::core::array> gdn_conv_states;
  std::unordered_map<int, mlx::core::array> gdn_recurrent_states;
  std::unordered_map<int, mlx::core::array> attention_key_states;
  std::unordered_map<int, mlx::core::array> attention_value_states;
  std::unordered_map<int, EdgeCmlxQuantizedArray> attention_quantized_key_states;
  std::unordered_map<int, EdgeCmlxQuantizedArray> attention_quantized_value_states;
  std::unordered_map<int, EdgeCmlxQwen35DSRPolicy> attention_dsr_policies;
  std::unordered_map<int, mlx::core::array> attention_score_states;
  std::unordered_map<int, int> attention_active_lengths;
  std::unordered_map<int, int> attention_dsr_tokens_since_eviction;
  std::optional<EdgeCmlxQwen35VisionConfig> vision_config;
  std::optional<EdgeCmlxQwen35AudioConfig> audio_config;
  std::unordered_map<std::string, mlx::core::array> tts_speech_tensors;
  std::optional<mlx::core::array> pending_token;
  std::string pending_sample_diagnostics;
  std::string emitted_sample_diagnostics;
  EdgeCmlxQwen35EvalProfileCounters eval_profile_total;
  EdgeCmlxQwen35EvalProfileCounters eval_profile_prefill;
  EdgeCmlxQwen35EvalProfileCounters eval_profile_decode;
  EdgeCmlxQwen35EvalProfileCounters eval_profile_token_read;
  std::unordered_map<std::string, EdgeCmlxQwen35EvalProfileInventoryBucket>
      eval_profile_inventory_total;
  std::unordered_map<std::string, EdgeCmlxQwen35EvalProfileInventoryBucket>
      eval_profile_inventory_prefill;
  std::unordered_map<std::string, EdgeCmlxQwen35EvalProfileInventoryBucket>
      eval_profile_inventory_decode;
  std::string eval_profile_last_caller;
  std::string eval_profile_last_mode;
  std::string eval_profile_last_inventory;
  std::string eval_profile_last_graph_summary;
  std::string eval_profile_last_metal_summary;
  int eval_profile_last_token_count = 0;
  int eval_profile_last_decoded_before = 0;
  uint64_t eval_profile_fused_rms_scale_hits = 0;
  uint64_t eval_profile_fused_rms_scale_tokens = 0;
  int eval_profile_fused_rms_scale_last_layer = -1;
  int eval_profile_fused_rms_scale_last_token_count = 0;
  int attention_cache_limit = 0;
  int attention_cache_quantization_group_size = 0;
  int attention_cache_quantization_bits = 0;
  int attention_cache_base_position = 0;
  int attention_cache_base_index = 0;
  int decoded_token_count = 0;
  uint64_t frog_jump_layer_mask = 0;
  bool fused_attention_check_logged = false;
  bool prefill_fp16_attention_logged = false;
  bool prefill_fp16_attention_materialized_pending_clear = false;
  bool dtype_chain_logged = false;
  bool frog_jump_logged = false;
  float repetition_penalty = 1.0f;
  std::unordered_set<int> repetition_context_tokens;
  float presence_penalty = 0.0f;
  std::unordered_set<int> presence_context_tokens;
  float frequency_penalty = 0.0f;
  std::vector<int> frequency_context_tokens;
  bool eos_sampling_suppressed = false;
  float eos_sampling_logit_penalty = 0.0f;
  std::unordered_set<int> eos_sampling_token_ids;
  std::unique_ptr<EdgeCmlxQwen35Session> tts_code_predictor_session;
};

extern thread_local std::string edge_cmlx_error;

int set_error(const std::string& message);
EdgeCmlxQwen35Session* checked_qwen35_session(void* opaque_session);
const EdgeCmlxQwen35Session* checked_qwen35_session(const void* opaque_session);

const mlx::core::array& checked_qwen35_float_tensor(
    const EdgeCmlxQwen35Session& session,
    int tensor_id);
const mlx::core::array* optional_qwen35_float_tensor(
    const EdgeCmlxQwen35Session& session,
    int tensor_id);
const EdgeCmlxQuantizedArray& checked_qwen35_quantized_tensor(
    const EdgeCmlxQwen35Session& session,
    int tensor_id);
const EdgeCmlxQwen35AudioConfig& checked_qwen35_audio_config(
    const EdgeCmlxQwen35Session& session);

size_t array_element_count(const mlx::core::Shape& shape);

int qwen35_embedding_id();
int qwen35_final_norm_id();
int qwen35_lm_head_id();
int qwen35_layer_tensor_id(int layer_index, int offset);
int qwen35_layer_attention_query_id(int layer_index);
int qwen35_layer_attention_key_id(int layer_index);
int qwen35_layer_attention_value_id(int layer_index);
int qwen35_layer_attention_output_id(int layer_index);
int qwen35_layer_gdn_qkv_id(int layer_index);
int qwen35_layer_gdn_z_id(int layer_index);
int qwen35_layer_gdn_a_id(int layer_index);
int qwen35_layer_gdn_b_id(int layer_index);
int qwen35_layer_gdn_output_id(int layer_index);
int qwen35_layer_mlp_gate_id(int layer_index);
int qwen35_layer_mlp_up_id(int layer_index);
int qwen35_layer_mlp_down_id(int layer_index);
int quantized_output_columns(
    const EdgeCmlxQuantizedArray& weights,
    bool transpose);
int quantized_inner_columns(
    const EdgeCmlxQuantizedArray& weights,
    bool transpose);

void validate_qwen35_config(const EdgeCmlxQwen35Config& config);
void qwen35_reset_decode_cache(EdgeCmlxQwen35Session& qwen_session);

void register_loaded_float_tensor(
    EdgeCmlxQwen35Session& session,
    int tensor_id,
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    const std::string& tensor_name);
void register_loaded_vision_float_tensor(
    EdgeCmlxQwen35Session& session,
    int tensor_id,
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    const std::string& tensor_name);
void register_loaded_float_tensor_if_exists(
    EdgeCmlxQwen35Session& session,
    int tensor_id,
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    const std::string& tensor_name);
void register_loaded_weight_tensor(
    EdgeCmlxQwen35Session& session,
    int tensor_id,
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    const std::string& weight_name,
    int group_size,
    int bits);
void register_loaded_weight_tensor_if_exists(
    EdgeCmlxQwen35Session& session,
    int tensor_id,
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    const std::string& weight_name,
    int group_size,
    int bits);
void register_qwen35_full_attention_layer_tensors(
    EdgeCmlxQwen35Session& session,
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    const std::string& layer_prefix,
    int layer_index,
    int group_size,
    int bits);

mlx::core::array silu_array(
    const mlx::core::array& input,
    mlx::core::StreamOrDevice stream);
mlx::core::array qwen35_embedding_for_indices(
    const EdgeCmlxQwen35Session& session,
    int embedding_id,
    const mlx::core::array& token_indices,
    mlx::core::StreamOrDevice stream);
mlx::core::array qwen35_linear_array(
    const mlx::core::array& input,
    const EdgeCmlxQwen35Session& session,
    int tensor_id,
    mlx::core::StreamOrDevice stream);
mlx::core::array qwen35_add_optional_bias(
    const mlx::core::array& input,
    const EdgeCmlxQwen35Session& session,
    int tensor_id,
    mlx::core::StreamOrDevice stream);

}

struct Qwen35AdvanceResult {
  mlx::core::array token;
  mlx::core::array last_hidden;
  mlx::core::array logits;
  std::optional<mlx::core::array> captured_hidden;
};

void validate_qwen35_decode_layer_kinds(
    const edge_cmlx::detail::EdgeCmlxQwen35Session& qwen_session,
    const char* caller);

mlx::core::array qwen35_vision_gelu_tanh(
    const mlx::core::array& input,
    mlx::core::StreamOrDevice stream);
mlx::core::array qwen35_vision_layer_norm(
    const mlx::core::array& input,
    const mlx::core::array& weight,
    const mlx::core::array& bias,
    float epsilon,
    mlx::core::StreamOrDevice stream);

Qwen35AdvanceResult qwen35_session_advance_hidden_with_state(
    edge_cmlx::detail::EdgeCmlxQwen35Session& qwen_session,
    mlx::core::array hidden,
    int token_count,
    bool async_schedule,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    uint64_t seed,
    const char* caller,
    int lm_head_id,
    int capture_layer = -1,
    bool stop_after_capture = false);
