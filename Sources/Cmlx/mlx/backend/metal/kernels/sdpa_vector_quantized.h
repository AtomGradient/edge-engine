// Copyright © 2024 Apple Inc.
// Quantized SDPA vector kernel — fused dequantize + attention
// Eliminates the separate dequantize step for quantized KV cache
//
// This header must be included AFTER sdpa_vector.h which declares
// the function constants (has_mask, query_transposed, do_causal,
// bool_mask, float_mask, has_sinks, blocks, output_scores).

// Thread-local dequantize: extracts `N` float values from packed quantized bytes
template <typename U, int N, int bits>
inline void dequantize_to_thread(
    const device uint8_t* w, U scale, U bias, thread U* dst) {
  static_assert(
      bits == 4 || bits == 8,
      "Only 4-bit and 8-bit quantization supported");

  if (bits == 8) {
    for (int i = 0; i < N; i++) {
      dst[i] = scale * static_cast<U>(w[i]) + bias;
    }
  } else if (bits == 4) {
    for (int i = 0; i < N / 2; i++) {
      dst[2 * i] = scale * static_cast<U>(w[i] & 0x0f) + bias;
      dst[2 * i + 1] = scale * static_cast<U>((w[i] >> 4) & 0x0f) + bias;
    }
  }
}

template <typename T, int D, int V = D, int bits = 8, int group_size = 64>
[[kernel]] void sdpa_vector_quantized(
    const device T* queries [[buffer(0)]],
    const device uint32_t* key_weights [[buffer(1)]],
    const device T* key_scales [[buffer(2)]],
    const device T* key_biases [[buffer(3)]],
    const device uint32_t* val_weights [[buffer(4)]],
    const device T* val_scales [[buffer(5)]],
    const device T* val_biases [[buffer(6)]],
    device T* out [[buffer(7)]],
    const constant int& gqa_factor [[buffer(8)]],
    const constant int& N [[buffer(9)]],
    const constant size_t& k_head_stride [[buffer(10)]],
    const constant size_t& v_head_stride [[buffer(11)]],
    const constant size_t& ks_head_stride [[buffer(12)]],
    const constant size_t& vs_head_stride [[buffer(13)]],
    const constant float& scale [[buffer(14)]],
    const device T* sinks [[buffer(15), function_constant(has_sinks)]],
    const constant int& num_q_heads
    [[buffer(16), function_constant(has_sinks)]],
    device float* scores_out [[buffer(17), function_constant(output_scores)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;

  // Quantization layout constants
  constexpr int packed_per_seq = D * bits / 32;      // uint32s per seq pos for keys
  constexpr int v_packed_per_seq = V * bits / 32;     // uint32s per seq pos for values
  constexpr int scales_per_seq = D / group_size;       // scale entries per seq pos for keys
  constexpr int v_scales_per_seq = V / group_size;     // scale entries per seq pos for values

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Thread-specific offsets
  int thread_byte_off = simd_lid * qk_per_thread * bits / 8;
  int thread_group_idx = (simd_lid * qk_per_thread) / group_size;
  int v_thread_byte_off = simd_lid * v_per_thread * bits / 8;
  int v_thread_group_idx = (simd_lid * v_per_thread) / group_size;

  // Adjust positions
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset =
      query_transposed ? tpg.x * q_seq_idx + q_batch_head_idx : o_offset;

  // Query pointer
  queries += q_offset * D + simd_lid * qk_per_thread;

  // Quantized key pointers — start at simd_gid-th position
  const device uint32_t* kw = key_weights + kv_head_idx * k_head_stride +
      simd_gid * packed_per_seq;
  const device T* ks = key_scales + kv_head_idx * ks_head_stride +
      simd_gid * scales_per_seq;
  const device T* kb = key_biases + kv_head_idx * ks_head_stride +
      simd_gid * scales_per_seq;

  // Quantized value pointers
  const device uint32_t* vw = val_weights + kv_head_idx * v_head_stride +
      simd_gid * v_packed_per_seq;
  const device T* vs = val_scales + kv_head_idx * vs_head_stride +
      simd_gid * v_scales_per_seq;
  const device T* vb = val_biases + kv_head_idx * vs_head_stride +
      simd_gid * v_scales_per_seq;

  out += o_offset * V + simd_gid * v_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && simd_gid == 0) {
    max_score = static_cast<U>(sinks[q_batch_head_idx % num_q_heads]);
    sum_exp_score = 1;
  }

  // Inner strides (advance by BN positions)
  constexpr int inner_kw_stride = BN * packed_per_seq;
  constexpr int inner_ks_stride = BN * scales_per_seq;
  constexpr int inner_vw_stride = BN * v_packed_per_seq;
  constexpr int inner_vs_stride = BN * v_scales_per_seq;

  // For each key
  for (int i = simd_gid; i < N; i += BN) {
    // Dequantize key for this thread
    const device uint8_t* kw_bytes =
        (const device uint8_t*)kw + thread_byte_off;
    U k_scale = static_cast<U>(ks[thread_group_idx]);
    U k_bias = static_cast<U>(kb[thread_group_idx]);
    dequantize_to_thread<U, qk_per_thread, bits>(
        kw_bytes, k_scale, k_bias, k);

    // Compute the i-th score
    U score = 0;
    for (int j = 0; j < qk_per_thread; j++) {
      score += q[j] * k[j];
    }
    score = simd_sum(score);

    // Output pre-softmax score for H₂O attention
    if (output_scores && simd_lid == 0) {
      scores_out[o_offset * N + i] = score;
    }

    // Update the accumulators
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    // Dequantize value for this thread and accumulate
    const device uint8_t* vw_bytes =
        (const device uint8_t*)vw + v_thread_byte_off;
    U v_scale = static_cast<U>(vs[v_thread_group_idx]);
    U v_bias = static_cast<U>(vb[v_thread_group_idx]);
    thread U v_local[v_per_thread];
    dequantize_to_thread<U, v_per_thread, bits>(
        vw_bytes, v_scale, v_bias, v_local);

    for (int j = 0; j < v_per_thread; j++) {
      o[j] = o[j] * factor + exp_score * v_local[j];
    }

    // Advance pointers
    kw += inner_kw_stride;
    ks += inner_ks_stride;
    kb += inner_ks_stride;
    vw += inner_vw_stride;
    vs += inner_vs_stride;
    vb += inner_vs_stride;
  }

  // Reduce across simdgroups — same as original sdpa_vector

  // First communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Aggregate all the outputs
  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write the output
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

template <typename T, int D, int V = D, int bits = 8, int group_size = 64>
[[kernel]] void sdpa_vector_2pass_1_quantized(
    const device T* queries [[buffer(0)]],
    const device uint32_t* key_weights [[buffer(1)]],
    const device T* key_scales [[buffer(2)]],
    const device T* key_biases [[buffer(3)]],
    const device uint32_t* val_weights [[buffer(4)]],
    const device T* val_scales [[buffer(5)]],
    const device T* val_biases [[buffer(6)]],
    device T* out [[buffer(7)]],
    device float* sums [[buffer(8)]],
    device float* maxs [[buffer(9)]],
    const constant int& N [[buffer(10)]],
    const constant size_t& k_head_stride [[buffer(11)]],
    const constant size_t& v_head_stride [[buffer(12)]],
    const constant size_t& ks_head_stride [[buffer(13)]],
    const constant size_t& vs_head_stride [[buffer(14)]],
    const constant float& scale [[buffer(15)]],
    const device T* sinks [[buffer(16), function_constant(has_sinks)]],
    device float* scores_out [[buffer(17), function_constant(output_scores)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;

  // Quantization layout constants
  constexpr int packed_per_seq = D * bits / 32;
  constexpr int v_packed_per_seq = V * bits / 32;
  constexpr int scales_per_seq = D / group_size;
  constexpr int v_scales_per_seq = V / group_size;

  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};

  // Thread-specific offsets
  int thread_byte_off = simd_lid * qk_per_thread * bits / 8;
  int thread_group_idx = (simd_lid * qk_per_thread) / group_size;
  int v_thread_byte_off = simd_lid * v_per_thread * bits / 8;
  int v_thread_group_idx = (simd_lid * v_per_thread) / group_size;

  // Adjust positions
  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_seq_len = tptg.z;
  const int q_seq_idx = tidtg.z;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int q_batch_head_idx = (batch_idx * num_q_heads + q_head_idx);
  const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;
  const int q_offset =
      query_transposed ? num_q_heads * q_seq_idx + q_batch_head_idx : o_offset;

  queries += q_offset * D + simd_lid * qk_per_thread;

  const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;

  // Quantized key pointers — start at block_idx-th position
  const device uint32_t* kw =
      key_weights + kv_batch_head_idx * k_head_stride +
      block_idx * packed_per_seq;
  const device T* ks =
      key_scales + kv_batch_head_idx * ks_head_stride +
      block_idx * scales_per_seq;
  const device T* kbi =
      key_biases + kv_batch_head_idx * ks_head_stride +
      block_idx * scales_per_seq;

  // Quantized value pointers
  const device uint32_t* vw =
      val_weights + kv_batch_head_idx * v_head_stride +
      block_idx * v_packed_per_seq;
  const device T* vsc =
      val_scales + kv_batch_head_idx * vs_head_stride +
      block_idx * v_scales_per_seq;
  const device T* vbi =
      val_biases + kv_batch_head_idx * vs_head_stride +
      block_idx * v_scales_per_seq;

  out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;

  // Read the query
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && block_idx == 0) {
    max_score = static_cast<U>(sinks[q_head_idx]);
    sum_exp_score = 1;
  }

  // Inner strides (advance by blocks positions)
  int inner_kw_stride = blocks * packed_per_seq;
  int inner_ks_stride = blocks * scales_per_seq;
  int inner_vw_stride = blocks * v_packed_per_seq;
  int inner_vs_stride = blocks * v_scales_per_seq;

  // For each key
  for (int i = block_idx; i < N; i += blocks) {
    // Dequantize key for this thread
    const device uint8_t* kw_bytes =
        (const device uint8_t*)kw + thread_byte_off;
    U k_scale = static_cast<U>(ks[thread_group_idx]);
    U k_bias = static_cast<U>(kbi[thread_group_idx]);
    thread U k_local[qk_per_thread];
    dequantize_to_thread<U, qk_per_thread, bits>(
        kw_bytes, k_scale, k_bias, k_local);

    // Compute the i-th score
    U score = 0;
    for (int j = 0; j < qk_per_thread; j++) {
      score += q[j] * k_local[j];
    }
    score = simd_sum(score);

    // Output pre-softmax score for H₂O attention
    if (output_scores && simd_lid == 0) {
      scores_out[o_offset * N + i] = score;
    }

    // Update the accumulators
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    // Dequantize value for this thread and accumulate
    const device uint8_t* vw_bytes =
        (const device uint8_t*)vw + v_thread_byte_off;
    U v_scale = static_cast<U>(vsc[v_thread_group_idx]);
    U v_bias = static_cast<U>(vbi[v_thread_group_idx]);
    thread U v_local[v_per_thread];
    dequantize_to_thread<U, v_per_thread, bits>(
        vw_bytes, v_scale, v_bias, v_local);

    for (int j = 0; j < v_per_thread; j++) {
      o[j] = o[j] * factor + exp_score * v_local[j];
    }

    // Advance pointers
    kw += inner_kw_stride;
    ks += inner_ks_stride;
    kbi += inner_ks_stride;
    vw += inner_vw_stride;
    vsc += inner_vs_stride;
    vbi += inner_vs_stride;
  }

  // Write the sum and max and outputs
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }

  for (int i = 0; i < v_per_thread; i++) {
    out[i] = static_cast<T>(o[i]);
  }
}
