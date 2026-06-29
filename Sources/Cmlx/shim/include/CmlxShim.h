// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

const char* edge_cmlx_vendor_version(void);
int edge_cmlx_vendor_version_numeric(void);
int edge_cmlx_vendor_present(void);
const char* edge_cmlx_last_error(void);
int edge_cmlx_default_metallib_available(void);

typedef struct EdgeCmlxStateProbeSnapshot {
    int status;
    int is_available;
    int has_primitive;
    int sibling_count;
} EdgeCmlxStateProbeSnapshot;

typedef struct EdgeCmlxStateBoundaryProbeResult {
    int sync_steps;
    EdgeCmlxStateProbeSnapshot sync_recurrent_after_token_eval;
    EdgeCmlxStateProbeSnapshot sync_conv_after_token_eval;
    EdgeCmlxStateProbeSnapshot sync_recurrent_stop_gradient_after_token_eval;
    EdgeCmlxStateProbeSnapshot sync_conv_stop_gradient_after_token_eval;
    EdgeCmlxStateProbeSnapshot sync_custom_recurrent_after_token_eval;
    EdgeCmlxStateProbeSnapshot sync_custom_recurrent_stop_gradient_after_token_eval;
    int async_steps;
    EdgeCmlxStateProbeSnapshot async_recurrent_after_schedule;
    EdgeCmlxStateProbeSnapshot async_conv_after_schedule;
    EdgeCmlxStateProbeSnapshot async_recurrent_stop_gradient_after_schedule;
    EdgeCmlxStateProbeSnapshot async_conv_stop_gradient_after_schedule;
    EdgeCmlxStateProbeSnapshot async_custom_recurrent_after_schedule;
    EdgeCmlxStateProbeSnapshot async_custom_recurrent_stop_gradient_after_schedule;
    EdgeCmlxStateProbeSnapshot async_recurrent_after_final_eval;
    EdgeCmlxStateProbeSnapshot async_conv_after_final_eval;
    EdgeCmlxStateProbeSnapshot async_custom_recurrent_after_final_eval;
} EdgeCmlxStateBoundaryProbeResult;

int edge_cmlx_run_state_boundary_probe(
    int steps,
    EdgeCmlxStateBoundaryProbeResult* result);
int edge_cmlx_run_cross_thread_stream_probe(void);

int edge_cmlx_set_command_buffer_limits(int max_ops, int max_mb);
int edge_cmlx_get_command_buffer_limits(int* max_ops, int* max_mb);
int edge_cmlx_set_memory_limit(size_t bytes);
int edge_cmlx_get_memory_limit(size_t* bytes);

int edge_cmlx_matmul_f32_cpu(
    const float* lhs,
    int lhs_rows,
    int lhs_cols,
    const float* rhs,
    int rhs_rows,
    int rhs_cols,
    float* output,
    size_t output_count);

int edge_cmlx_matmul_f32_gpu(
    const float* lhs,
    int lhs_rows,
    int lhs_cols,
    const float* rhs,
    int rhs_rows,
    int rhs_cols,
    float* output,
    size_t output_count);

int edge_cmlx_softmax_f32_gpu(
    const float* input,
    int rows,
    int columns,
    float* output,
    size_t output_count);

int edge_cmlx_sample_token_f32_gpu(
    const float* logits,
    int vocabulary_size,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    uint64_t seed,
    int* output_token_id);

int edge_cmlx_fast_rms_norm_f32_gpu(
    const float* input,
    int rows,
    int columns,
    const float* weight,
    float epsilon,
    float* output,
    size_t output_count);

int edge_cmlx_rms_norm_scale_f32_gpu(
    const float* input,
    int rows,
    int columns,
    float epsilon,
    float scale,
    float* output,
    size_t output_count);

int edge_cmlx_encode_fast_rms_norm_f32_mtl(
    void* command_buffer,
    const void* input_buffer,
    size_t input_offset,
    int rows,
    int columns,
    const void* weight_buffer,
    size_t weight_offset,
    float epsilon,
    void* output_buffer,
    size_t output_offset,
    size_t output_count);

int edge_cmlx_affine_quantized_matmul_f32_gpu(
    const float* lhs,
    int lhs_rows,
    int lhs_cols,
    const uint32_t* packed_weights,
    int packed_rows,
    int packed_cols,
    const float* scales,
    int scale_rows,
    int scale_cols,
    const float* biases,
    int group_size,
    int bits,
    int transpose,
    float* output,
    size_t output_count);

int edge_cmlx_affine_quantized_matmul_f32_mtl(
    const void* lhs_buffer,
    int lhs_rows,
    int lhs_cols,
    const void* packed_weights_buffer,
    int packed_rows,
    int packed_cols,
    const void* scales_buffer,
    int scale_rows,
    int scale_cols,
    const void* biases_buffer,
    int group_size,
    int bits,
    int transpose,
    void* output_buffer,
    size_t output_count);

int edge_cmlx_rms_norm_affine_quantized_matmul_f32_mtl(
    const void* lhs_buffer,
    int lhs_rows,
    int lhs_cols,
    const void* norm_weight_buffer,
    float epsilon,
    const void* packed_weights_buffer,
    int packed_rows,
    int packed_cols,
    const void* scales_buffer,
    int scale_rows,
    int scale_cols,
    const void* biases_buffer,
    int group_size,
    int bits,
    int transpose,
    void* output_buffer,
    size_t output_count);

int edge_cmlx_encode_affine_qmm_t_f32_mtl(
    void* command_buffer,
    const void* lhs_buffer,
    size_t lhs_offset,
    int lhs_rows,
    int lhs_cols,
    const void* packed_weights_buffer,
    size_t packed_weights_offset,
    int packed_rows,
    int packed_cols,
    const void* scales_buffer,
    size_t scales_offset,
    int scale_rows,
    int scale_cols,
    const void* biases_buffer,
    size_t biases_offset,
    int group_size,
    int bits,
    void* output_buffer,
    size_t output_offset,
    size_t output_count);

typedef struct EdgeCmlxFloatTensorDescriptor {
    const void* buffer;
    int rank;
    int dim0;
    int dim1;
    int dim2;
    int dim3;
} EdgeCmlxFloatTensorDescriptor;

typedef struct EdgeCmlxQuantizedTensorDescriptor {
    const void* packed_buffer;
    size_t packed_offset;
    int packed_rows;
    int packed_cols;
    const void* scales_buffer;
    size_t scales_offset;
    int scale_rows;
    int scale_cols;
    const void* biases_buffer;
    size_t biases_offset;
    int group_size;
    int bits;
} EdgeCmlxQuantizedTensorDescriptor;

typedef struct EdgeCmlxQwen35Config {
    int layer_count;
    int hidden_size;
    int vocabulary_size;
    int intermediate_size;
    int attention_head_count;
    int key_value_head_count;
    int attention_head_dimension;
    int linear_key_head_count;
    int linear_value_head_count;
    int linear_key_head_dimension;
    int linear_value_head_dimension;
    int linear_conv_kernel_size;
    int rotary_dimension;
    float rope_theta;
    float rms_norm_epsilon;
    int uses_sparse_moe;
    int moe_num_experts;
    int moe_experts_per_token;
    int moe_intermediate_size;
    int moe_shared_expert_intermediate_size;
    int moe_normalize_topk_probabilities;
} EdgeCmlxQwen35Config;

typedef struct EdgeCmlxQwen35VisionConfig {
    int hidden_size;
    int intermediate_size;
    int layer_count;
    int head_count;
    int patch_size;
    int spatial_merge_size;
    int temporal_patch_size;
    int output_hidden_size;
    float layer_norm_epsilon;
} EdgeCmlxQwen35VisionConfig;

typedef struct EdgeCmlxQwen35AudioConfig {
    int num_mel_bins;
    int encoder_layers;
    int encoder_attention_heads;
    int encoder_ffn_dim;
    int d_model;
    int max_source_positions;
    int n_window;
    int output_dim;
    int n_window_infer;
    int downsample_hidden_size;
    float layer_norm_epsilon;
} EdgeCmlxQwen35AudioConfig;

typedef enum EdgeCmlxQwen35LayerKind {
    EdgeCmlxQwen35LayerKindUnknown = 0,
    EdgeCmlxQwen35LayerKindFullAttention = 1,
    EdgeCmlxQwen35LayerKindGDN = 2,
} EdgeCmlxQwen35LayerKind;

typedef struct EdgeCmlxQwen35DSRPolicy {
    int max_size;
    int heavy_budget;
    int recent_budget;
    int sink_size;
    int eviction_interval;
    float score_activation_ratio;
    float score_decay;
} EdgeCmlxQwen35DSRPolicy;

void* edge_cmlx_qwen35_session_create(const EdgeCmlxQwen35Config* config);
void edge_cmlx_qwen35_session_destroy(void* session);

int edge_cmlx_qwen35_session_set_layer_kind(
    void* session,
    int layer_index,
    int layer_kind);

int edge_cmlx_qwen35_session_set_vision_config(
    void* session,
    const EdgeCmlxQwen35VisionConfig* config);

int edge_cmlx_qwen35_session_set_audio_config(
    void* session,
    const EdgeCmlxQwen35AudioConfig* config);

int edge_cmlx_qwen35_session_clear_dsr_policies(void* session);

int edge_cmlx_qwen35_session_set_dsr_policy(
    void* session,
    int layer_index,
    const EdgeCmlxQwen35DSRPolicy* policy);

int edge_cmlx_qwen35_session_update_dsr_policy_fields(
    void* session,
    int layer_index,
    const EdgeCmlxQwen35DSRPolicy* policy);

int edge_cmlx_qwen35_session_set_attention_cache_quantization(
    void* session,
    int group_size,
    int bits);

int edge_cmlx_qwen35_session_set_frog_jump_mask(
    void* session,
    uint64_t layer_mask);

int edge_cmlx_qwen35_session_set_float_tensor(
    void* session,
    int tensor_id,
    const EdgeCmlxFloatTensorDescriptor* descriptor);

int edge_cmlx_qwen35_session_set_quantized_tensor(
    void* session,
    int tensor_id,
    const EdgeCmlxQuantizedTensorDescriptor* descriptor);

int edge_cmlx_qwen35_session_load_safetensors(
    void* session,
    const char* const* shard_paths,
    int shard_count,
    const char* model_prefix,
    int group_size,
    int bits);

int edge_cmlx_qwen35_session_materialize_decoder_weights(void* session);

int edge_cmlx_qwen35_session_has_decoder_weights(const void* session);

int edge_cmlx_qwen35_session_unload_decoder_weights_preserving_state(void* session);

int edge_cmlx_qwen35_session_restore_neural_imprint_cache(
    void* session,
    const char* artifact_path,
    int prefix_token_count);

int edge_cmlx_qwen35_session_save_neural_imprint_cache(
    void* session,
    const char* artifact_path,
    const char* const* metadata_keys,
    const char* const* metadata_values,
    int metadata_count);

int edge_cmlx_qwen35_session_decoded_token_count(const void* session);

int edge_cmlx_qwen35_session_load_tts_safetensors(
    void* session,
    const char* const* shard_paths,
    int shard_count,
    int group_size,
    int bits,
    const EdgeCmlxQwen35Config* code_predictor_config);

int edge_cmlx_qwen35_session_load_tts_speech_tokenizer_safetensors(
    void* session,
    const char* model_path);

int edge_cmlx_qwen35_session_load_vision_safetensors(
    void* session,
    const char* const* shard_paths,
    int shard_count,
    const char* vision_prefix);

int edge_cmlx_qwen35_session_load_audio_safetensors(
    void* session,
    const char* const* shard_paths,
    int shard_count,
    const char* audio_prefix);

int edge_cmlx_qwen35_session_registered_float_tensor_count(const void* session);
int edge_cmlx_qwen35_session_registered_quantized_tensor_count(const void* session);

int edge_cmlx_qwen35_session_vision_encode(
    void* session,
    const float* pixel_values,
    int num_patches,
    int patch_dim,
    const int* grid_thw,
    int grid_count,
    float* output,
    int* output_patches,
    int* output_hidden_size);

int edge_cmlx_qwen35_session_audio_encode(
    void* session,
    const float* log_mel_features,
    int frame_count,
    int mel_bin_count,
    float* output,
    size_t output_count,
    int* output_frames,
    int* output_hidden_size);

int edge_cmlx_qwen35_session_eval_quantized_mlp_f32_mtl(
    void* session,
    const void* input_buffer,
    int input_rows,
    int input_cols,
    int gate_tensor_id,
    int up_tensor_id,
    int down_tensor_id,
    void* output_buffer,
    int output_cols,
    size_t output_count);

int edge_cmlx_qwen35_session_eval_quantized_gdn_decode_layer_f32_mtl(
    void* session,
    int layer_index,
    const void* input_buffer,
    const void* conv_state_buffer,
    const void* recurrent_state_buffer,
    void* output_buffer,
    void* next_conv_state_buffer,
    void* next_recurrent_state_buffer,
    size_t output_count);

int edge_cmlx_qwen35_session_eval_quantized_full_attention_decode_layer_f32_mtl(
    void* session,
    int layer_index,
    const void* input_buffer,
    int position_offset,
    void* output_buffer,
    size_t output_count);

int edge_cmlx_qwen35_session_decode_step(
    void* session,
    const int* token_ids,
    int token_count,
    int* output_token_id);

int edge_cmlx_qwen35_session_prefill(
    void* session,
    const int* token_ids,
    int token_count,
    int* output_token_id);

int edge_cmlx_qwen35_session_capture_last_hidden(
    void* session,
    const int* token_ids,
    int token_count,
    int target_layer,
    float* output,
    int output_count);

int edge_cmlx_qwen35_session_prefill_async(
    void* session,
    const int* token_ids,
    int token_count);

int edge_cmlx_qwen35_session_synchronize(void* session);

int edge_cmlx_qwen35_session_prefill_sampled_async(
    void* session,
    const int* token_ids,
    int token_count,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    uint64_t seed);

int edge_cmlx_qwen35_session_prefill_embeddings(
    void* session,
    const float* embeddings,
    int token_count,
    int hidden_size,
    int* output_token_id);

int edge_cmlx_qwen35_session_prefill_media_features(
    void* session,
    const int* token_ids,
    int token_count,
    const float* media_features,
    int media_feature_count,
    int hidden_size,
    int media_token_id,
    int* output_token_id);

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
    int* output_codebooks);

int edge_cmlx_qwen35_session_tts_decode_audio_codes(
    void* session,
    const int* codes,
    int step_count,
    int codebook_count,
    float* output_samples,
    int output_sample_capacity,
    int* output_sample_count);

int edge_cmlx_qwen35_session_prefill_image_features(
    void* session,
    const int* token_ids,
    int token_count,
    const float* image_features,
    int image_feature_count,
    int hidden_size,
    int image_token_id,
    int* output_token_id);

int edge_cmlx_qwen35_session_next_token(
    void* session,
    int* output_token_id);

int edge_cmlx_qwen35_session_next_sampled_token(
    void* session,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    uint64_t seed,
    int* output_token_id);

int edge_cmlx_qwen35_session_set_sampling_penalties(
    void* session,
    float repetition_penalty,
    const int* repetition_context_token_ids,
    int repetition_context_token_count,
    float presence_penalty,
    const int* presence_context_token_ids,
    int presence_context_token_count,
    float frequency_penalty,
    const int* frequency_context_token_ids,
    int frequency_context_token_count);

int edge_cmlx_qwen35_session_set_eos_sampling_bias(
    void* session,
    const int* token_ids,
    int token_count,
    int suppress,
    float logit_penalty);

int edge_cmlx_qwen35_session_clear_eos_sampling_bias(
    void* session);

int edge_cmlx_qwen35_session_set_repetition_penalty(
    void* session,
    float penalty,
    const int* context_token_ids,
    int context_token_count);

int edge_cmlx_qwen35_session_clear_repetition_penalty(
    void* session);

int edge_cmlx_qwen35_session_copy_last_sample_diagnostics(
    void* session,
    char* output,
    int output_capacity);

int edge_cmlx_qwen35_session_copy_memory_summary(
    void* session,
    char* output,
    int output_capacity);

int edge_cmlx_qwen35_session_reset_eval_profile(void* session);

int edge_cmlx_qwen35_session_copy_eval_profile(
    void* session,
    char* output,
    int output_capacity);

int edge_cmlx_qwen35_session_set_attention_cache_limit(
    void* session,
    int max_tokens);

int edge_cmlx_qwen35_session_reset_decode_cache(void* session);

uint64_t edge_cmlx_mtl_buffer_retain_count(const void* buffer);

#ifdef __cplusplus
}
#endif
