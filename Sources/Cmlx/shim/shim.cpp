// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

#include "CmlxShim.h"
#include "blocks.h"
#include "primitives.h"
#include "shim_internal.h"

#include <json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mlx/array.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/edge_profile.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/device.h"
#include "mlx/fast.h"
#include "mlx/io.h"
#include "mlx/memory.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/random.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"
#include "mlx/utils.h"
#include "mlx/version.h"

namespace edge_cmlx::detail {

thread_local std::string edge_cmlx_error;

int set_error(const std::string& message) {
  edge_cmlx_error = message;
  return -1;
}

EdgeCmlxStateProbeSnapshot state_probe_snapshot(
    const mlx::core::array& value) {
  const bool is_available = value.is_available();
  return EdgeCmlxStateProbeSnapshot{
      static_cast<int>(value.status()),
      is_available ? 1 : 0,
      value.has_primitive() ? 1 : 0,
      static_cast<int>(value.siblings().size())};
}

size_t array_byte_count(const mlx::core::Shape& shape, mlx::core::Dtype dtype) {
  size_t count = 1;
  for (const auto dimension : shape) {
    count *= static_cast<size_t>(dimension);
  }
  return count * mlx::core::size_of(dtype);
}

size_t array_element_count(const mlx::core::Shape& shape) {
  size_t count = 1;
  for (const auto dimension : shape) {
    count *= static_cast<size_t>(dimension);
  }
  return count;
}

void validate_array_element_count(
    const mlx::core::array& value,
    size_t expected_count,
    const std::string& context) {
  const size_t actual_count = array_element_count(value.shape());
  if (actual_count != expected_count) {
    std::ostringstream message;
    message << context << " expected " << expected_count << " elements, got "
            << actual_count << " with shape " << value.shape();
    throw std::runtime_error(message.str());
  }
}

const char* dtype_label(mlx::core::Dtype dtype) {
  if (dtype == mlx::core::float16) {
    return "f16";
  }
  if (dtype == mlx::core::float32) {
    return "f32";
  }
  return "other";
}

mlx::core::Strides row_contiguous_strides(const mlx::core::Shape& shape) {
  mlx::core::Strides strides(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        strides[static_cast<size_t>(i + 1)] *
        static_cast<int64_t>(shape[static_cast<size_t>(i + 1)]);
  }
  return strides;
}

mlx::core::array::Flags row_contiguous_flags(const mlx::core::Shape& shape) {
  mlx::core::array::Flags flags{true, true, true};
  const auto size = array_element_count(shape);
  const auto max_dim = std::max_element(shape.begin(), shape.end());
  flags.col_contiguous =
      size <= 1 || (max_dim != shape.end() && size == static_cast<size_t>(*max_dim));
  return flags;
}

MTL::Buffer* checked_mtl_buffer(
    const void* opaque_buffer,
    size_t byte_offset,
    size_t minimum_byte_count,
    const char* name) {
  if (opaque_buffer == nullptr) {
    throw std::runtime_error(std::string(name) + " buffer is null");
  }
  auto* buffer =
      const_cast<MTL::Buffer*>(static_cast<const MTL::Buffer*>(opaque_buffer));
  if (byte_offset > buffer->length() ||
      buffer->length() - byte_offset < minimum_byte_count) {
    throw std::runtime_error(std::string(name) + " buffer is too small");
  }
  return buffer;
}

MTL::Buffer* checked_mtl_buffer(
    const void* opaque_buffer,
    size_t minimum_byte_count,
    const char* name) {
  return checked_mtl_buffer(
      opaque_buffer, 0, minimum_byte_count, name);
}

template <typename... Args>
MTL::ComputePipelineState* get_quantized_kernel_wrapped(
    mlx::core::metal::Device& d,
    const std::string& name,
    const std::string& func,
    const std::string& mode,
    const std::string& type,
    int group_size,
    int bits,
    Args... args) {
  std::string fname = ((mode == "affine") ? "affine_" : "fp_") + func;
  std::string template_def = mlx::core::get_template_definition(
      name, fname, type, group_size, bits, std::forward<Args>(args)...);
  return mlx::core::get_quantized_kernel(d, name, template_def, mode);
}

mlx::core::array array_from_mtl_buffer(
    const void* opaque_buffer,
    size_t byte_offset,
    mlx::core::Shape shape,
    mlx::core::Dtype dtype,
    const char* name) {
  const auto minimum_byte_count = array_byte_count(shape, dtype);
  auto* buffer = checked_mtl_buffer(
      opaque_buffer, byte_offset, minimum_byte_count, name);
  const auto item_size = mlx::core::size_of(dtype);
  if (byte_offset % item_size != 0) {
    throw std::runtime_error(std::string(name) + " buffer offset is not item-aligned");
  }
  const auto buffer_element_count = buffer->length() / item_size;
  if (buffer_element_count > static_cast<size_t>(std::numeric_limits<mlx::core::ShapeElem>::max())) {
    throw std::runtime_error(std::string(name) + " buffer is too large");
  }
  buffer->retain();
  mlx::core::metal::device(mlx::core::Device{mlx::core::Device::gpu})
      .residency_set()
      .insert(buffer);
  auto retained_buffer_deleter = [](mlx::core::allocator::Buffer data) {
    auto* retained_buffer = static_cast<MTL::Buffer*>(data.ptr());
    if (retained_buffer == nullptr) {
      return;
    }
    mlx::core::metal::device(mlx::core::Device{mlx::core::Device::gpu})
        .residency_set()
        .erase(retained_buffer);
    retained_buffer->release();
  };
  if (byte_offset == 0 && buffer->length() == minimum_byte_count) {
    return mlx::core::array(
        mlx::core::allocator::Buffer{static_cast<void*>(buffer)},
        std::move(shape),
        dtype,
        retained_buffer_deleter);
  }
  auto base = mlx::core::array(
      mlx::core::allocator::Buffer{static_cast<void*>(buffer)},
      mlx::core::Shape{static_cast<mlx::core::ShapeElem>(buffer_element_count)},
      dtype,
      retained_buffer_deleter);
  auto view = mlx::core::array(
      shape,
      dtype,
      nullptr,
      {});
  view.copy_shared_buffer(
      base,
      row_contiguous_strides(shape),
      row_contiguous_flags(shape),
      array_element_count(shape),
      static_cast<int64_t>(byte_offset / item_size));
  view.set_status(mlx::core::array::Status::available);
  return view;
}

mlx::core::array array_from_mtl_buffer(
    const void* opaque_buffer,
    mlx::core::Shape shape,
    mlx::core::Dtype dtype,
    const char* name) {
  return array_from_mtl_buffer(
      opaque_buffer, 0, std::move(shape), dtype, name);
}

mlx::core::Shape shape_from_descriptor(
    const EdgeCmlxFloatTensorDescriptor& descriptor) {
  if (descriptor.rank <= 0 || descriptor.rank > 4) {
    throw std::runtime_error("float tensor descriptor rank is invalid");
  }
  const int dimensions[4] = {
      descriptor.dim0, descriptor.dim1, descriptor.dim2, descriptor.dim3};
  mlx::core::Shape shape;
  shape.reserve(static_cast<size_t>(descriptor.rank));
  for (int i = 0; i < descriptor.rank; ++i) {
    if (dimensions[i] <= 0) {
      throw std::runtime_error("float tensor descriptor has non-positive dimension");
    }
    shape.push_back(static_cast<mlx::core::ShapeElem>(dimensions[i]));
  }
  return shape;
}

constexpr int kQwen35DecodeClearCacheInterval = 1024;
constexpr int kQwen35AttentionCacheStep = 256;

bool env_truthy(const char* value) {
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  return std::strcmp(value, "1") == 0 ||
      std::strcmp(value, "true") == 0 ||
      std::strcmp(value, "TRUE") == 0 ||
      std::strcmp(value, "yes") == 0 ||
      std::strcmp(value, "YES") == 0 ||
      std::strcmp(value, "on") == 0 ||
      std::strcmp(value, "ON") == 0;
}

bool env_falsey(const char* value) {
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  return std::strcmp(value, "0") == 0 ||
      std::strcmp(value, "false") == 0 ||
      std::strcmp(value, "FALSE") == 0 ||
      std::strcmp(value, "no") == 0 ||
      std::strcmp(value, "NO") == 0 ||
      std::strcmp(value, "off") == 0 ||
      std::strcmp(value, "OFF") == 0;
}

int qwen35_decode_clear_cache_interval() {
  const char* value = std::getenv("EDGE_CMLX_DECODE_CLEAR_CACHE_INTERVAL");
  if (value == nullptr || value[0] == '\0') {
    value = std::getenv("CMLX_DECODE_CLEAR_CACHE_INTERVAL");
  }
  if (value == nullptr || value[0] == '\0') {
    return kQwen35DecodeClearCacheInterval;
  }
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value ||
      parsed < std::numeric_limits<int>::min() ||
      parsed > std::numeric_limits<int>::max()) {
    return kQwen35DecodeClearCacheInterval;
  }
  return static_cast<int>(parsed);
}

bool qwen35_sample_diagnostics_enabled() {
  const char* value = std::getenv("EDGE_CMLX_SAMPLE_DIAGNOSTICS");
  if (value == nullptr || value[0] == '\0') {
    value = std::getenv("CMLX_SAMPLE_DIAGNOSTICS");
  }
  return env_truthy(value);
}

bool qwen35_fast_topp_topk_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("EDGE_CMLX_FAST_TOPP_TOPK");
    if (value == nullptr || value[0] == '\0') {
      value = std::getenv("CMLX_FAST_TOPP_TOPK");
    }
    return !env_falsey(value);
  }();
  return enabled;
}

bool qwen35_eval_profile_enabled() {
  const char* value = std::getenv("EDGE_CMLX_EVAL_PROFILE");
  if (value == nullptr || value[0] == '\0') {
    value = std::getenv("CMLX_EVAL_PROFILE");
  }
  return env_truthy(value);
}

bool qwen35_metal_profile_enabled() {
  return mlx::core::metal::edge_metal_profile_enabled();
}

bool qwen35_any_eval_profile_enabled() {
  return qwen35_eval_profile_enabled() || qwen35_metal_profile_enabled();
}

bool qwen35_graph_profile_enabled() {
  const char* value = std::getenv("EDGE_CMLX_GRAPH_PROFILE");
  if (value == nullptr || value[0] == '\0') {
    value = std::getenv("CMLX_GRAPH_PROFILE");
  }
  return env_truthy(value);
}

bool qwen35_should_record_graph_profile(
    const EdgeCmlxQwen35Session& qwen_session,
    const char* caller,
    int token_count) {
  if (!qwen35_graph_profile_enabled() ||
      !qwen_session.eval_profile_last_graph_summary.empty() ||
      token_count != 1) {
    return false;
  }
  const std::string caller_value = caller == nullptr ? "" : caller;
  return caller_value.find("prefill") == std::string::npos;
}

bool qwen35_lazy_eval_outputs_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("EDGE_CMLX_LAZY_EVAL_OUTPUTS");
    if (value == nullptr || value[0] == '\0') {
      value = std::getenv("CMLX_LAZY_EVAL_OUTPUTS");
    }
    return env_truthy(value);
  }();
  return enabled;
}

bool qwen35_lazy_quantized_eval_outputs_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("EDGE_CMLX_LAZY_QUANTIZED_EVAL_OUTPUTS");
    if (value == nullptr || value[0] == '\0') {
      value = std::getenv("CMLX_LAZY_QUANTIZED_EVAL_OUTPUTS");
    }
    return env_truthy(value);
  }();
  return enabled;
}

bool qwen35_lazy_gdn_eval_outputs_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("EDGE_CMLX_LAZY_GDN_EVAL_OUTPUTS");
    if (value == nullptr || value[0] == '\0') {
      value = std::getenv("CMLX_LAZY_GDN_EVAL_OUTPUTS");
    }
    return env_truthy(value);
  }();
  return enabled;
}

bool qwen35_skip_gdn_stop_gradient_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("EDGE_CMLX_SKIP_GDN_STOP_GRADIENT");
    if (value == nullptr || value[0] == '\0') {
      value = std::getenv("CMLX_SKIP_GDN_STOP_GRADIENT");
    }
    return env_truthy(value);
  }();
  return enabled;
}

bool qwen35_skip_fa_stop_gradient_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("EDGE_CMLX_SKIP_FA_STOP_GRADIENT");
    if (value == nullptr || value[0] == '\0') {
      value = std::getenv("CMLX_SKIP_FA_STOP_GRADIENT");
    }
    return env_truthy(value);
  }();
  return enabled;
}

bool qwen35_dsr_prefill_eviction_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("EDGE_CMLX_DSR_PREFILL_EVICTION");
    if (value == nullptr || value[0] == '\0') {
      value = std::getenv("CMLX_DSR_PREFILL_EVICTION");
    }
    return env_truthy(value);
  }();
  return enabled;
}

bool qwen35_fused_rms_scale_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("EDGE_CMLX_FUSED_RMS_SCALE");
    if (value == nullptr || value[0] == '\0') {
      value = std::getenv("CMLX_FUSED_RMS_SCALE");
    }
    return !env_falsey(value);
  }();
  return enabled;
}

bool qwen35_fused_gdn_gate_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("EDGE_CMLX_FUSED_GDN_GATE");
    if (value == nullptr || value[0] == '\0') {
      value = std::getenv("CMLX_FUSED_GDN_GATE");
    }
    return !env_falsey(value);
  }();
  return enabled;
}

bool qwen35_gdn_precompute_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("EDGE_CMLX_GDN_PRECOMPUTE");
    if (value == nullptr || value[0] == '\0') {
      value = std::getenv("CMLX_GDN_PRECOMPUTE");
    }
    return !env_falsey(value);
  }();
  return enabled;
}

bool qwen35_can_lazy_dense_eval_output_kind(const char* kind) {
  return std::strcmp(kind, "fa_dense_key_state") == 0 ||
      std::strcmp(kind, "fa_dense_value_state") == 0 ||
      std::strcmp(kind, "dsr_score_state") == 0;
}

bool qwen35_can_lazy_quantized_eval_output_kind(const char* kind) {
  return std::strcmp(kind, "quant_key_packed") == 0 ||
      std::strcmp(kind, "quant_key_scales") == 0 ||
      std::strcmp(kind, "quant_key_biases") == 0 ||
      std::strcmp(kind, "quant_value_packed") == 0 ||
      std::strcmp(kind, "quant_value_scales") == 0 ||
      std::strcmp(kind, "quant_value_biases") == 0;
}

bool qwen35_can_lazy_gdn_eval_output_kind(const char* kind) {
  return std::strcmp(kind, "gdn_conv_state") == 0 ||
      std::strcmp(kind, "gdn_recurrent_state") == 0;
}

bool qwen35_should_lazy_eval_output_kind(const char* kind) {
  return (qwen35_lazy_eval_outputs_enabled() &&
             qwen35_can_lazy_dense_eval_output_kind(kind)) ||
      (qwen35_lazy_quantized_eval_outputs_enabled() &&
          qwen35_can_lazy_quantized_eval_output_kind(kind)) ||
      (qwen35_lazy_gdn_eval_outputs_enabled() &&
          qwen35_can_lazy_gdn_eval_output_kind(kind));
}

bool qwen35_immutable_fa_cache_enabled() {
  return env_truthy(std::getenv("EDGE_CMLX_IMMUTABLE_FA_CACHE"));
}

int qwen35_immutable_fa_cache_layer() {
  const char* value = std::getenv("EDGE_CMLX_IMMUTABLE_FA_LAYER");
  if (value == nullptr || value[0] == '\0') {
    return -1;
  }
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
    return -1;
  }
  return static_cast<int>(parsed);
}

std::chrono::steady_clock::time_point qwen35_profile_now() {
  return std::chrono::steady_clock::now();
}

double qwen35_profile_elapsed_ms(
    std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start)
      .count();
}

size_t qwen35_eval_outputs_bytes(const std::vector<mlx::core::array>& arrays) {
  size_t bytes = 0;
  for (const auto& array : arrays) {
    bytes += array.nbytes();
  }
  return bytes;
}

struct Qwen35EvalOutputInventory {
  using Bucket = EdgeCmlxQwen35EvalProfileInventoryBucket;

  std::unordered_map<std::string, Bucket> buckets;

  void add(const char* kind, const mlx::core::array& array) {
    auto& bucket = buckets[std::string(kind)];
    bucket.count += 1;
    bucket.bytes += static_cast<uint64_t>(array.nbytes());
  }

  std::string summary() const {
    if (buckets.empty()) {
      return "empty";
    }
    std::vector<std::string> keys;
    keys.reserve(buckets.size());
    for (const auto& entry : buckets) {
      keys.push_back(entry.first);
    }
    std::sort(keys.begin(), keys.end());

    std::ostringstream out;
    bool first = true;
    for (const auto& key : keys) {
      const auto& bucket = buckets.at(key);
      if (!first) {
        out << ",";
      }
      first = false;
      out << key << "=" << bucket.count << "/" << bucket.bytes;
    }
    return out.str();
  }
};

struct Qwen35GraphProfile {
  uint64_t root_count = 0;
  uint64_t array_count = 0;
  uint64_t input_count = 0;
  uint64_t primitive_count = 0;
  uint64_t primitive_output_count = 0;
  uint64_t edge_count = 0;
  int max_depth = 0;
  std::unordered_map<std::string, uint64_t> primitive_counts;
};

int qwen35_collect_graph_profile(
    const mlx::core::array& value,
    Qwen35GraphProfile& profile,
    std::unordered_map<std::uintptr_t, int>& depth_by_array,
    std::unordered_set<std::uintptr_t>& primitive_ids) {
  const auto array_id = value.id();
  if (const auto found = depth_by_array.find(array_id);
      found != depth_by_array.end()) {
    return found->second;
  }

  profile.array_count += 1;
  int parent_depth = 0;
  if (value.has_primitive()) {
    const auto primitive_id = value.primitive_id();
    if (primitive_ids.insert(primitive_id).second) {
      profile.primitive_count += 1;
      profile.primitive_output_count +=
          static_cast<uint64_t>(value.outputs().size());
      profile.primitive_counts[value.primitive().name()] += 1;
    }
    for (const auto& input : value.inputs()) {
      profile.edge_count += 1;
      parent_depth = std::max(
          parent_depth,
          qwen35_collect_graph_profile(
              input, profile, depth_by_array, primitive_ids));
    }
  } else {
    profile.input_count += 1;
  }

  const int depth = parent_depth + 1;
  profile.max_depth = std::max(profile.max_depth, depth);
  depth_by_array.emplace(array_id, depth);
  return depth;
}

std::string qwen35_graph_profile_summary(
    const std::vector<mlx::core::array>& outputs) {
  Qwen35GraphProfile profile;
  profile.root_count = static_cast<uint64_t>(outputs.size());
  std::unordered_map<std::uintptr_t, int> depth_by_array;
  std::unordered_set<std::uintptr_t> primitive_ids;
  for (const auto& output : outputs) {
    qwen35_collect_graph_profile(
        output, profile, depth_by_array, primitive_ids);
  }

  std::vector<std::pair<std::string, uint64_t>> primitive_counts(
      profile.primitive_counts.begin(), profile.primitive_counts.end());
  std::sort(
      primitive_counts.begin(),
      primitive_counts.end(),
      [](const auto& lhs, const auto& rhs) {
        if (lhs.second != rhs.second) {
          return lhs.second > rhs.second;
        }
        return lhs.first < rhs.first;
      });

  std::ostringstream out;
  out << "roots=" << profile.root_count
      << ",arrays=" << profile.array_count
      << ",inputs=" << profile.input_count
      << ",prims=" << profile.primitive_count
      << ",primOutputs=" << profile.primitive_output_count
      << ",edges=" << profile.edge_count
      << ",depth=" << profile.max_depth
      << ",ops=";
  if (primitive_counts.empty()) {
    out << "empty";
  } else {
    bool first = true;
    for (const auto& entry : primitive_counts) {
      if (!first) {
        out << "|";
      }
      first = false;
      out << entry.first << ":" << entry.second;
    }
  }
  return out.str();
}

void qwen35_reset_eval_profile(EdgeCmlxQwen35Session& qwen_session) {
  qwen_session.eval_profile_total = {};
  qwen_session.eval_profile_prefill = {};
  qwen_session.eval_profile_decode = {};
  qwen_session.eval_profile_token_read = {};
  qwen_session.eval_profile_inventory_total.clear();
  qwen_session.eval_profile_inventory_prefill.clear();
  qwen_session.eval_profile_inventory_decode.clear();
  qwen_session.eval_profile_last_caller.clear();
  qwen_session.eval_profile_last_mode.clear();
  qwen_session.eval_profile_last_inventory.clear();
  qwen_session.eval_profile_last_graph_summary.clear();
  qwen_session.eval_profile_last_metal_summary.clear();
  qwen_session.eval_profile_last_token_count = 0;
  qwen_session.eval_profile_last_decoded_before = 0;
  qwen_session.eval_profile_fused_rms_scale_hits = 0;
  qwen_session.eval_profile_fused_rms_scale_tokens = 0;
  qwen_session.eval_profile_fused_rms_scale_last_layer = -1;
  qwen_session.eval_profile_fused_rms_scale_last_token_count = 0;
  mlx::core::metal::edge_metal_profile_reset();
}

EdgeCmlxQwen35EvalProfileCounters& qwen35_eval_profile_phase_bucket(
    EdgeCmlxQwen35Session& qwen_session,
    const char* caller,
    int token_count) {
  const std::string caller_value = caller == nullptr ? "" : caller;
  if (caller_value.find("prefill") != std::string::npos ||
      token_count > 1) {
    return qwen_session.eval_profile_prefill;
  }
  return qwen_session.eval_profile_decode;
}

std::unordered_map<std::string, EdgeCmlxQwen35EvalProfileInventoryBucket>&
qwen35_eval_profile_inventory_phase_bucket(
    EdgeCmlxQwen35Session& qwen_session,
    const char* caller,
    int token_count) {
  const std::string caller_value = caller == nullptr ? "" : caller;
  if (caller_value.find("prefill") != std::string::npos ||
      token_count > 1) {
    return qwen_session.eval_profile_inventory_prefill;
  }
  return qwen_session.eval_profile_inventory_decode;
}

void qwen35_merge_eval_profile_inventory(
    std::unordered_map<std::string, EdgeCmlxQwen35EvalProfileInventoryBucket>&
        target,
    const Qwen35EvalOutputInventory& inventory) {
  for (const auto& entry : inventory.buckets) {
    auto& bucket = target[entry.first];
    bucket.count += entry.second.count;
    bucket.bytes += entry.second.bytes;
  }
}

void qwen35_record_eval_profile_barrier(
    EdgeCmlxQwen35Session& qwen_session,
    const char* caller,
    const char* mode,
    int token_count,
    int decoded_before,
    size_t output_count,
    size_t output_bytes,
    double elapsed_ms,
    const Qwen35EvalOutputInventory& inventory,
    const mlx::core::metal::EdgeMetalProfileSnapshot& metal_profile) {
  auto record = [&](EdgeCmlxQwen35EvalProfileCounters& counters) {
    counters.eval_barriers += 1;
    counters.eval_outputs += static_cast<uint64_t>(output_count);
    counters.eval_bytes += static_cast<uint64_t>(output_bytes);
    counters.eval_elapsed_ms += elapsed_ms;
    if (std::strcmp(mode, "async_eval") == 0) {
      counters.async_eval_barriers += 1;
    } else {
      counters.sync_eval_barriers += 1;
    }
  };

  record(qwen_session.eval_profile_total);
  record(qwen35_eval_profile_phase_bucket(qwen_session, caller, token_count));
  qwen35_merge_eval_profile_inventory(
      qwen_session.eval_profile_inventory_total, inventory);
  qwen35_merge_eval_profile_inventory(
      qwen35_eval_profile_inventory_phase_bucket(qwen_session, caller, token_count),
      inventory);
  qwen_session.eval_profile_last_caller = caller == nullptr ? "" : caller;
  qwen_session.eval_profile_last_mode = mode == nullptr ? "" : mode;
  qwen_session.eval_profile_last_inventory = inventory.summary();
  qwen_session.eval_profile_last_metal_summary =
      mlx::core::metal::edge_metal_profile_summary(metal_profile);
  qwen_session.eval_profile_last_token_count = token_count;
  qwen_session.eval_profile_last_decoded_before = decoded_before;
}

void qwen35_record_token_read_profile(
    EdgeCmlxQwen35Session& qwen_session,
    const char* caller,
    int decoded_before,
    size_t bytes,
    double elapsed_ms) {
  auto record = [&](EdgeCmlxQwen35EvalProfileCounters& counters) {
    counters.token_read_barriers += 1;
    counters.token_read_bytes += static_cast<uint64_t>(bytes);
    counters.token_read_elapsed_ms += elapsed_ms;
  };

  record(qwen_session.eval_profile_total);
  record(qwen_session.eval_profile_token_read);
}

std::string qwen35_eval_profile_counters_summary(
    const EdgeCmlxQwen35EvalProfileCounters& counters) {
  std::ostringstream out;
  out << "barriers=" << counters.eval_barriers
      << ",sync=" << counters.sync_eval_barriers
      << ",async=" << counters.async_eval_barriers
      << ",evalMs=" << counters.eval_elapsed_ms
      << ",outputs=" << counters.eval_outputs
      << ",bytes=" << counters.eval_bytes
      << ",tokenReads=" << counters.token_read_barriers
      << ",tokenReadMs=" << counters.token_read_elapsed_ms
      << ",tokenReadBytes=" << counters.token_read_bytes;
  return out.str();
}

std::string qwen35_eval_profile_inventory_summary(
    const std::unordered_map<
        std::string,
        EdgeCmlxQwen35EvalProfileInventoryBucket>& buckets) {
  if (buckets.empty()) {
    return "empty";
  }
  std::vector<std::string> keys;
  keys.reserve(buckets.size());
  for (const auto& entry : buckets) {
    keys.push_back(entry.first);
  }
  std::sort(keys.begin(), keys.end());

  std::ostringstream out;
  bool first = true;
  for (const auto& key : keys) {
    const auto& bucket = buckets.at(key);
    if (!first) {
      out << ",";
    }
    first = false;
    out << key << "=" << bucket.count << "/" << bucket.bytes;
  }
  return out.str();
}

std::string qwen35_eval_profile_summary(
    const EdgeCmlxQwen35Session& qwen_session) {
  std::ostringstream out;
  out << "total={" << qwen35_eval_profile_counters_summary(
             qwen_session.eval_profile_total)
      << "} prefill={" << qwen35_eval_profile_counters_summary(
             qwen_session.eval_profile_prefill)
      << "} decode={" << qwen35_eval_profile_counters_summary(
             qwen_session.eval_profile_decode)
      << "} tokenRead={" << qwen35_eval_profile_counters_summary(
             qwen_session.eval_profile_token_read)
      << "} inventoryTotal={"
      << qwen35_eval_profile_inventory_summary(
             qwen_session.eval_profile_inventory_total)
      << "} inventoryPrefill={"
      << qwen35_eval_profile_inventory_summary(
             qwen_session.eval_profile_inventory_prefill)
      << "} inventoryDecode={"
      << qwen35_eval_profile_inventory_summary(
             qwen_session.eval_profile_inventory_decode)
      << "} last={caller=" << qwen_session.eval_profile_last_caller
      << ",mode=" << qwen_session.eval_profile_last_mode
      << ",tokenCount=" << qwen_session.eval_profile_last_token_count
      << ",decodedBefore=" << qwen_session.eval_profile_last_decoded_before
      << ",inventory=" << (qwen_session.eval_profile_last_inventory.empty()
             ? "empty"
             : qwen_session.eval_profile_last_inventory)
      << ",graph=" << (qwen_session.eval_profile_last_graph_summary.empty()
             ? "disabled"
             : qwen_session.eval_profile_last_graph_summary)
      << ",metal=" << (qwen_session.eval_profile_last_metal_summary.empty()
             ? "disabled"
             : qwen_session.eval_profile_last_metal_summary)
      << "} fusedRMSScale={hits="
      << qwen_session.eval_profile_fused_rms_scale_hits
      << ",tokens=" << qwen_session.eval_profile_fused_rms_scale_tokens
      << ",lastLayer="
      << qwen_session.eval_profile_fused_rms_scale_last_layer
      << ",lastTokenCount="
      << qwen_session.eval_profile_fused_rms_scale_last_token_count
      << "} metalTotal={"
      << mlx::core::metal::edge_metal_profile_summary(
             mlx::core::metal::edge_metal_profile_snapshot())
      << "}";
  return out.str();
}

void qwen35_eval_output_push(
    std::vector<mlx::core::array>& outputs,
    const mlx::core::array& array,
    Qwen35EvalOutputInventory* inventory,
    const char* kind) {
  if (qwen35_should_lazy_eval_output_kind(kind)) {
    if (inventory != nullptr) {
      const std::string lazy_kind = std::string("lazy_skip_") + kind;
      inventory->add(lazy_kind.c_str(), array);
    }
    return;
  }

  outputs.push_back(array);
  if (inventory != nullptr) {
    inventory->add(kind, array);
  }
}

int qwen35_attention_cache_capacity(int limit) {
  if (limit <= 0) {
    return 0;
  }
  return ((limit + kQwen35AttentionCacheStep - 1) /
          kQwen35AttentionCacheStep) *
      kQwen35AttentionCacheStep;
}

int qwen35_dsr_transient_capacity(const EdgeCmlxQwen35DSRPolicy& policy) {
  return qwen35_attention_cache_capacity(
      policy.max_size + std::max(1, policy.eviction_interval) + 1);
}

bool qwen35_attention_cache_quantization_enabled(
    const EdgeCmlxQwen35Session& qwen_session) {
  return qwen_session.attention_cache_quantization_group_size > 0 &&
      qwen_session.attention_cache_quantization_bits > 0;
}

bool qwen35_uses_sparse_moe_mlp(
    const EdgeCmlxQwen35Session& qwen_session) {
  const auto& config = qwen_session.config;
  return config.uses_sparse_moe != 0 &&
      config.moe_num_experts > 0 &&
      config.moe_experts_per_token > 0 &&
      config.moe_intermediate_size > 0 &&
      config.moe_shared_expert_intermediate_size > 0;
}

bool qwen35_should_skip_frog_jump_layer(
    const EdgeCmlxQwen35Session& qwen_session,
    int layer) {
  return layer >= 0 &&
      layer < 64 &&
      (qwen_session.frog_jump_layer_mask & (uint64_t{1} << layer)) != 0;
}

int qwen35_attention_capacity_for_layer(
    const EdgeCmlxQwen35Session& qwen_session,
    int layer,
    int requested_length) {
  int capacity = qwen35_attention_cache_capacity(requested_length);
  const auto policy = qwen_session.attention_dsr_policies.find(layer);
  if (policy != qwen_session.attention_dsr_policies.end()) {
    capacity = std::max(capacity, qwen35_dsr_transient_capacity(policy->second));
  }
  return capacity;
}

void validate_qwen35_dsr_policy(const EdgeCmlxQwen35DSRPolicy& policy) {
  if (policy.max_size <= 0 ||
      policy.heavy_budget <= 0 ||
      policy.recent_budget <= 0 ||
      policy.sink_size < 0 ||
      policy.eviction_interval <= 0 ||
      policy.score_activation_ratio < 0.0f ||
      policy.score_activation_ratio > 1.0f ||
      policy.score_decay < 0.0f ||
      policy.score_decay > 1.0f ||
      policy.sink_size + policy.heavy_budget + policy.recent_budget >
          policy.max_size) {
    throw std::runtime_error("invalid Qwen3.5 Cmlx DSR policy");
  }
}

EdgeCmlxQwen35Session* checked_qwen35_session(void* opaque_session) {
  if (opaque_session == nullptr) {
    throw std::runtime_error("Qwen3.5 Cmlx session is null");
  }
  return static_cast<EdgeCmlxQwen35Session*>(opaque_session);
}

const EdgeCmlxQwen35Session* checked_qwen35_session(const void* opaque_session) {
  if (opaque_session == nullptr) {
    throw std::runtime_error("Qwen3.5 Cmlx session is null");
  }
  return static_cast<const EdgeCmlxQwen35Session*>(opaque_session);
}

struct Qwen35DecodeStateGuard {
  explicit Qwen35DecodeStateGuard(EdgeCmlxQwen35Session& session)
      : session(session),
        gdn_conv_states(session.gdn_conv_states),
        gdn_recurrent_states(session.gdn_recurrent_states),
        attention_key_states(session.attention_key_states),
        attention_value_states(session.attention_value_states),
        attention_quantized_key_states(session.attention_quantized_key_states),
        attention_quantized_value_states(session.attention_quantized_value_states),
        attention_score_states(session.attention_score_states),
        attention_active_lengths(session.attention_active_lengths),
        attention_dsr_tokens_since_eviction(session.attention_dsr_tokens_since_eviction),
        pending_token(session.pending_token),
        pending_sample_diagnostics(session.pending_sample_diagnostics),
        emitted_sample_diagnostics(session.emitted_sample_diagnostics),
        attention_cache_base_position(session.attention_cache_base_position),
        attention_cache_base_index(session.attention_cache_base_index),
        decoded_token_count(session.decoded_token_count),
        frog_jump_layer_mask(session.frog_jump_layer_mask),
        prefill_fp16_attention_materialized_pending_clear(
            session.prefill_fp16_attention_materialized_pending_clear) {}

  ~Qwen35DecodeStateGuard() {
    if (active) {
      restore();
    }
  }

  void restore() {
    session.gdn_conv_states = std::move(gdn_conv_states);
    session.gdn_recurrent_states = std::move(gdn_recurrent_states);
    session.attention_key_states = std::move(attention_key_states);
    session.attention_value_states = std::move(attention_value_states);
    session.attention_quantized_key_states =
        std::move(attention_quantized_key_states);
    session.attention_quantized_value_states =
        std::move(attention_quantized_value_states);
    session.attention_score_states = std::move(attention_score_states);
    session.attention_active_lengths = std::move(attention_active_lengths);
    session.attention_dsr_tokens_since_eviction =
        std::move(attention_dsr_tokens_since_eviction);
    session.pending_token = std::move(pending_token);
    session.pending_sample_diagnostics = std::move(pending_sample_diagnostics);
    session.emitted_sample_diagnostics = std::move(emitted_sample_diagnostics);
    session.attention_cache_base_position = attention_cache_base_position;
    session.attention_cache_base_index = attention_cache_base_index;
    session.decoded_token_count = decoded_token_count;
    session.frog_jump_layer_mask = frog_jump_layer_mask;
    session.prefill_fp16_attention_materialized_pending_clear =
        prefill_fp16_attention_materialized_pending_clear;
    active = false;
  }

  EdgeCmlxQwen35Session& session;
  std::unordered_map<int, mlx::core::array> gdn_conv_states;
  std::unordered_map<int, mlx::core::array> gdn_recurrent_states;
  std::unordered_map<int, mlx::core::array> attention_key_states;
  std::unordered_map<int, mlx::core::array> attention_value_states;
  std::unordered_map<int, EdgeCmlxQuantizedArray> attention_quantized_key_states;
  std::unordered_map<int, EdgeCmlxQuantizedArray> attention_quantized_value_states;
  std::unordered_map<int, mlx::core::array> attention_score_states;
  std::unordered_map<int, int> attention_active_lengths;
  std::unordered_map<int, int> attention_dsr_tokens_since_eviction;
  std::optional<mlx::core::array> pending_token;
  std::string pending_sample_diagnostics;
  std::string emitted_sample_diagnostics;
  int attention_cache_base_position;
  int attention_cache_base_index;
  int decoded_token_count;
  uint64_t frog_jump_layer_mask;
  bool prefill_fp16_attention_materialized_pending_clear;
  bool active = true;
};

const EdgeCmlxQwen35AudioConfig& checked_qwen35_audio_config(
    const EdgeCmlxQwen35Session& session);

const mlx::core::array& checked_qwen35_float_tensor(
    const EdgeCmlxQwen35Session& session,
    int tensor_id) {
  const auto item = session.float_tensors.find(tensor_id);
  if (item == session.float_tensors.end()) {
    throw std::runtime_error("Qwen3.5 Cmlx float tensor is not registered");
  }
  return item->second;
}

const mlx::core::array* optional_qwen35_float_tensor(
    const EdgeCmlxQwen35Session& session,
    int tensor_id) {
  const auto item = session.float_tensors.find(tensor_id);
  if (item == session.float_tensors.end()) {
    return nullptr;
  }
  return &item->second;
}

const EdgeCmlxQuantizedArray& checked_qwen35_quantized_tensor(
    const EdgeCmlxQwen35Session& session,
    int tensor_id) {
  const auto item = session.quantized_tensors.find(tensor_id);
  if (item == session.quantized_tensors.end()) {
    throw std::runtime_error("Qwen3.5 Cmlx quantized tensor is not registered");
  }
  return item->second;
}

int qwen35_layer_tensor_id(int layer_index, int offset) {
  return 1000 + layer_index * 100 + offset;
}

int qwen35_embedding_id() {
  return 1;
}

int qwen35_final_norm_id() {
  return 2;
}

int qwen35_lm_head_id() {
  return 3;
}
int qwen35_layer_attention_query_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 10);
}

int qwen35_layer_attention_key_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 11);
}

int qwen35_layer_attention_value_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 12);
}

int qwen35_layer_attention_query_norm_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 13);
}

int qwen35_layer_attention_key_norm_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 14);
}

int qwen35_layer_attention_output_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 15);
}

int qwen35_layer_input_norm_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 1);
}

int qwen35_layer_post_attention_norm_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 2);
}

int qwen35_layer_gdn_qkv_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 20);
}

int qwen35_layer_gdn_z_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 21);
}

int qwen35_layer_gdn_a_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 22);
}

int qwen35_layer_gdn_b_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 23);
}

int qwen35_layer_gdn_conv1d_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 24);
}

int qwen35_layer_gdn_a_log_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 25);
}

int qwen35_layer_gdn_dt_bias_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 26);
}

int qwen35_layer_gdn_norm_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 27);
}

int qwen35_layer_gdn_output_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 28);
}

int qwen35_layer_mlp_gate_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 40);
}

int qwen35_layer_mlp_up_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 41);
}

int qwen35_layer_mlp_down_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 42);
}

int qwen35_layer_moe_gate_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 50);
}

int qwen35_layer_moe_switch_gate_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 51);
}

int qwen35_layer_moe_switch_up_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 52);
}

int qwen35_layer_moe_switch_down_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 53);
}

int qwen35_layer_moe_shared_gate_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 54);
}

int qwen35_layer_moe_shared_up_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 55);
}

int qwen35_layer_moe_shared_down_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 56);
}

int qwen35_layer_moe_shared_expert_gate_id(int layer_index) {
  return qwen35_layer_tensor_id(layer_index, 57);
}

const mlx::core::array& checked_loaded_tensor(
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    const std::string& name) {
  const auto item = tensors.find(name);
  if (item == tensors.end()) {
    throw std::runtime_error("Cmlx safetensors missing tensor: " + name);
  }
  return item->second;
}

bool loaded_tensor_exists(
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    const std::string& name) {
  return tensors.find(name) != tensors.end();
}

std::string qwen35_neural_imprint_state_name(int layer, int state) {
  std::ostringstream name;
  name << "layer_" << std::setfill('0') << std::setw(2) << layer
       << ".state_" << state;
  return name.str();
}

std::string qwen35_neural_imprint_safetensor_dtype(mlx::core::Dtype dtype) {
  if (dtype == mlx::core::float32) {
    return "F32";
  }
  if (dtype == mlx::core::bfloat16) {
    return "BF16";
  }
  if (dtype == mlx::core::float16) {
    return "F16";
  }
  if (dtype == mlx::core::int64) {
    return "I64";
  }
  if (dtype == mlx::core::int32) {
    return "I32";
  }
  if (dtype == mlx::core::int16) {
    return "I16";
  }
  if (dtype == mlx::core::int8) {
    return "I8";
  }
  if (dtype == mlx::core::uint64) {
    return "U64";
  }
  if (dtype == mlx::core::uint32) {
    return "U32";
  }
  if (dtype == mlx::core::uint16) {
    return "U16";
  }
  if (dtype == mlx::core::uint8) {
    return "U8";
  }
  if (dtype == mlx::core::bool_) {
    return "BOOL";
  }
  if (dtype == mlx::core::complex64) {
    return "C64";
  }
  throw std::runtime_error(
      "edge_cmlx_qwen35_session_save_neural_imprint_cache received an unsupported tensor dtype");
}

void qwen35_validate_neural_imprint_shape(
    const mlx::core::array& tensor,
    const mlx::core::Shape& expected,
    const std::string& name) {
  if (tensor.shape() != expected) {
    std::ostringstream message;
    message << "neural_imprint tensor " << name << " shape mismatch: expected "
            << expected << ", got " << tensor.shape();
    throw std::runtime_error(message.str());
  }
}

mlx::core::array qwen35_restore_neural_imprint_tensor(
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    const std::string& name,
    const mlx::core::Shape& expected_shape,
    mlx::core::StreamOrDevice stream) {
  const auto& source = checked_loaded_tensor(tensors, name);
  qwen35_validate_neural_imprint_shape(source, expected_shape, name);
  return mlx::core::stop_gradient(source, stream);
}

mlx::core::array qwen35_restore_neural_imprint_batched_tensor(
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    const std::string& name,
    const mlx::core::Shape& batched_shape,
    const mlx::core::Shape& session_shape,
    mlx::core::StreamOrDevice stream) {
  const auto& source = checked_loaded_tensor(tensors, name);
  if (source.shape() == session_shape) {
    return mlx::core::stop_gradient(source, stream);
  }
  qwen35_validate_neural_imprint_shape(source, batched_shape, name);
  return mlx::core::stop_gradient(
      mlx::core::reshape(source, session_shape, stream),
      stream);
}

mlx::core::array qwen35_export_neural_imprint_batched_tensor(
    const mlx::core::array& tensor,
    const std::string& name,
    const mlx::core::Shape& batched_shape,
    const mlx::core::Shape& session_shape,
    mlx::core::StreamOrDevice stream) {
  if (tensor.shape() == batched_shape) {
    return tensor;
  }
  qwen35_validate_neural_imprint_shape(tensor, session_shape, name);
  return mlx::core::reshape(tensor, batched_shape, stream);
}

mlx::core::array qwen35_export_neural_imprint_attention_tensor(
    const mlx::core::array& tensor,
    const std::string& name,
    const mlx::core::Shape& expected_shape,
    mlx::core::StreamOrDevice stream) {
  if (tensor.ndim() != 4 ||
      tensor.shape(0) != expected_shape[0] ||
      tensor.shape(1) != expected_shape[1] ||
      tensor.shape(3) != expected_shape[3] ||
      tensor.shape(2) < expected_shape[2]) {
    qwen35_validate_neural_imprint_shape(tensor, expected_shape, name);
  }
  if (tensor.shape() == expected_shape) {
    return tensor;
  }
  return mlx::core::slice(
      tensor,
      mlx::core::Shape{0, 0, 0, 0},
      expected_shape,
      stream);
}

enum class Qwen35NeuralImprintExportKind {
  batched,
  attention,
};

struct Qwen35NeuralImprintExportEntry {
  Qwen35NeuralImprintExportKind kind;
  int layer;
  int state;
  std::string name;
  mlx::core::Shape batched_shape;
  mlx::core::Shape session_shape;
  mlx::core::Dtype dtype;
  size_t data_offset{0};
  size_t byte_count{0};
};

const mlx::core::array& qwen35_required_neural_imprint_source(
    const EdgeCmlxQwen35Session& session,
    const Qwen35NeuralImprintExportEntry& entry) {
  if (entry.kind == Qwen35NeuralImprintExportKind::batched) {
    const auto& states =
        entry.state == 0 ? session.gdn_conv_states : session.gdn_recurrent_states;
    const auto item = states.find(entry.layer);
    if (item == states.end()) {
      throw std::runtime_error(
          "edge_cmlx_qwen35_session_save_neural_imprint_cache missing GDN cache state");
    }
    return item->second;
  }

  const auto& states =
      entry.state == 0 ? session.attention_key_states : session.attention_value_states;
  const auto item = states.find(entry.layer);
  if (item == states.end()) {
    throw std::runtime_error(
        "edge_cmlx_qwen35_session_save_neural_imprint_cache missing full-attention cache state");
  }
  return item->second;
}

mlx::core::array qwen35_export_neural_imprint_entry_tensor(
    const EdgeCmlxQwen35Session& session,
    const Qwen35NeuralImprintExportEntry& entry,
    mlx::core::StreamOrDevice stream) {
  const auto& source = qwen35_required_neural_imprint_source(session, entry);
  if (entry.kind == Qwen35NeuralImprintExportKind::batched) {
    return qwen35_export_neural_imprint_batched_tensor(
        source,
        entry.name,
        entry.batched_shape,
        entry.session_shape,
        stream);
  }
  return qwen35_export_neural_imprint_attention_tensor(
      source,
      entry.name,
      entry.batched_shape,
      stream);
}

void qwen35_finalize_neural_imprint_export_entry(
    const EdgeCmlxQwen35Session& session,
    Qwen35NeuralImprintExportEntry& entry,
    mlx::core::StreamOrDevice stream,
    size_t& offset) {
  auto exported = qwen35_export_neural_imprint_entry_tensor(session, entry, stream);
  if (exported.nbytes() == 0) {
    throw std::runtime_error(
        "edge_cmlx_qwen35_session_save_neural_imprint_cache cannot serialize an empty tensor");
  }
  entry.batched_shape = exported.shape();
  entry.dtype = exported.dtype();
  entry.data_offset = offset;
  entry.byte_count = exported.nbytes();
  offset += entry.byte_count;
}

std::string qwen35_neural_imprint_artifact_path(std::string file) {
  constexpr const char* suffix = ".safetensors";
  constexpr size_t suffix_length = 12;
  if (file.length() < suffix_length ||
      file.substr(file.length() - suffix_length, suffix_length) != suffix) {
    file += suffix;
  }
  return file;
}

void qwen35_write_neural_imprint_safetensor_entry(
    mlx::core::io::Writer& out_stream,
    const EdgeCmlxQwen35Session& session,
    const Qwen35NeuralImprintExportEntry& entry,
    mlx::core::StreamOrDevice stream) {
  {
    auto exported = qwen35_export_neural_imprint_entry_tensor(session, entry, stream);
    if (exported.shape() != entry.batched_shape) {
      qwen35_validate_neural_imprint_shape(exported, entry.batched_shape, entry.name);
    }
    if (exported.dtype() != entry.dtype || exported.nbytes() != entry.byte_count) {
      throw std::runtime_error(
          "edge_cmlx_qwen35_session_save_neural_imprint_cache tensor metadata changed while exporting");
    }

    auto contiguous_tensor = mlx::core::contiguous(exported, false, stream);
    mlx::core::eval(contiguous_tensor);
    out_stream.write(contiguous_tensor.data<char>(), contiguous_tensor.nbytes());
  }
  mlx::core::clear_cache();
}

void qwen35_save_neural_imprint_safetensors_low_peak(
    const EdgeCmlxQwen35Session& session,
    std::string artifact_path,
    const std::vector<Qwen35NeuralImprintExportEntry>& entries,
    const std::unordered_map<std::string, std::string>& metadata,
    mlx::core::StreamOrDevice stream) {
  nlohmann::json parent;
  nlohmann::json metadata_json;
  for (const auto& item : metadata) {
    metadata_json[item.first] = item.second;
  }
  parent["__metadata__"] = std::move(metadata_json);

  for (const auto& entry : entries) {
    nlohmann::json child;
    child["dtype"] = qwen35_neural_imprint_safetensor_dtype(entry.dtype);
    child["shape"] = entry.batched_shape;
    child["data_offsets"] =
        std::vector<size_t>{entry.data_offset, entry.data_offset + entry.byte_count};
    parent[entry.name] = std::move(child);
  }

  auto out_stream = std::make_shared<mlx::core::io::FileWriter>(
      qwen35_neural_imprint_artifact_path(std::move(artifact_path)));
  if (!out_stream->good() || !out_stream->is_open()) {
    std::ostringstream message;
    message << "edge_cmlx_qwen35_session_save_neural_imprint_cache failed to open "
            << out_stream->label();
    throw std::runtime_error(message.str());
  }

  const auto header = parent.dump();
  uint64_t header_length = header.length();
  out_stream->write(reinterpret_cast<const char*>(&header_length), 8);
  out_stream->write(header.c_str(), header_length);
  for (const auto& entry : entries) {
    qwen35_write_neural_imprint_safetensor_entry(
        *out_stream,
        session,
        entry,
        stream);
  }
}

bool qwen35_loaded_tensor_name_has_prefix(
    const std::string& name,
    const char* prefix) {
  return name.rfind(prefix, 0) == 0;
}

bool qwen35_should_skip_text_decode_tensor(const std::string& name) {
  constexpr const char* vision_prefixes[] = {
      "audio_tower.",
      "visual.",
      "vision_tower.",
      "vision_model.",
      "model.visual.",
      "language_model.visual.",
  };
  for (const char* prefix : vision_prefixes) {
    if (qwen35_loaded_tensor_name_has_prefix(name, prefix)) {
      return true;
    }
  }
  return name.find(".vision_tower.") != std::string::npos ||
      name.find(".audio_tower.") != std::string::npos ||
      name.find(".vision_model.") != std::string::npos ||
      name.find(".model.visual.") != std::string::npos;
}

std::string quantized_base_name(const std::string& weight_name) {
  constexpr const char* suffix = ".weight";
  constexpr size_t suffix_length = 7;
  if (weight_name.size() <= suffix_length ||
      weight_name.compare(
          weight_name.size() - suffix_length,
          suffix_length,
          suffix) != 0) {
    throw std::runtime_error(
        "Cmlx quantized safetensor name does not end with .weight: " +
        weight_name);
  }
  return weight_name.substr(0, weight_name.size() - suffix_length);
}

bool loaded_quantized_companions_exist(
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    const std::string& weight_name) {
  return loaded_tensor_exists(tensors, quantized_base_name(weight_name) + ".scales");
}

mlx::core::array qwen35_quantized_aux_tensor_for_decode(
    const mlx::core::array& tensor,
    mlx::core::StreamOrDevice stream) {
  return tensor.dtype() == mlx::core::float16
      ? tensor
      : mlx::core::astype(tensor, mlx::core::float16, stream);
}

bool qwen35_float_tensor_prefers_decode_float16(int tensor_id) {
  if (tensor_id == qwen35_final_norm_id()) {
    return true;
  }
  if (tensor_id < 1000) {
    return false;
  }
  const int offset = (tensor_id - 1000) % 100;
  switch (offset) {
    case 1:
    case 2:
    case 13:
    case 14:
    case 24:
    case 27:
      return true;
    default:
      return false;
  }
}

bool qwen35_float_tensor_prefers_decode_float32(int tensor_id) {
  if (tensor_id < 1000) {
    return false;
  }
  const int offset = (tensor_id - 1000) % 100;
  switch (offset) {
    case 25:
    case 26:
      return true;
    default:
      return false;
  }
}

mlx::core::array qwen35_float_tensor_for_decode(
    int tensor_id,
    const mlx::core::array& tensor,
    mlx::core::StreamOrDevice stream) {
  if (qwen35_float_tensor_prefers_decode_float32(tensor_id)) {
    return tensor.dtype() == mlx::core::float32
        ? tensor
        : mlx::core::astype(tensor, mlx::core::float32, stream);
  }
  return qwen35_float_tensor_prefers_decode_float16(tensor_id)
      ? qwen35_quantized_aux_tensor_for_decode(tensor, stream)
      : tensor;
}

void register_qwen35_gdn_neg_exp_a_log(
    EdgeCmlxQwen35Session& session,
    int layer_index) {
  if (!qwen35_gdn_precompute_enabled()) {
    return;
  }
  auto stream = mlx::core::Device{mlx::core::Device::gpu};
  const auto& a_log = checked_qwen35_float_tensor(
      session, qwen35_layer_gdn_a_log_id(layer_index));
  auto typed_a_log = a_log.dtype() == mlx::core::float32
      ? a_log
      : mlx::core::astype(a_log, mlx::core::float32, stream);
  auto neg_exp_a_log = mlx::core::multiply(
      mlx::core::exp(typed_a_log, stream),
      mlx::core::array(-1.0f, mlx::core::float32),
      stream);
  mlx::core::eval(neg_exp_a_log);
  session.gdn_neg_exp_a_log_tensors.insert_or_assign(
      layer_index, std::move(neg_exp_a_log));
}

void register_loaded_float_tensor(
    EdgeCmlxQwen35Session& session,
    int tensor_id,
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    const std::string& tensor_name) {
  const auto& source = checked_loaded_tensor(tensors, tensor_name);
  auto decode_tensor = qwen35_float_tensor_for_decode(
      tensor_id, source, mlx::core::Device{mlx::core::Device::gpu});
  if (decode_tensor.dtype() != source.dtype()) {
    mlx::core::eval(decode_tensor);
  }
  session.float_tensors.insert_or_assign(tensor_id, std::move(decode_tensor));
}

void register_loaded_vision_float_tensor(
    EdgeCmlxQwen35Session& session,
    int tensor_id,
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    const std::string& tensor_name) {
  const auto& source = checked_loaded_tensor(tensors, tensor_name);
  auto tensor = source.dtype() == mlx::core::float32
      ? source
      : mlx::core::astype(
            source,
            mlx::core::float32,
            mlx::core::Device{mlx::core::Device::gpu});
  mlx::core::eval(tensor);
  session.float_tensors.insert_or_assign(tensor_id, std::move(tensor));
}

void register_loaded_quantized_tensor(
    EdgeCmlxQwen35Session& session,
    int tensor_id,
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    const std::string& weight_name,
    int group_size,
    int bits) {
  const auto base_name = quantized_base_name(weight_name);
  const auto scales_name = base_name + ".scales";
  const auto biases_name = base_name + ".biases";
  const auto& scales = checked_loaded_tensor(tensors, scales_name);
  auto decode_scales = qwen35_quantized_aux_tensor_for_decode(
      scales, mlx::core::Device{mlx::core::Device::gpu});
  auto decode_biases = qwen35_quantized_aux_tensor_for_decode(
      loaded_tensor_exists(tensors, biases_name)
          ? checked_loaded_tensor(tensors, biases_name)
          : mlx::core::zeros_like(scales),
      mlx::core::Device{mlx::core::Device::gpu});
  mlx::core::eval(decode_scales, decode_biases);
  EdgeCmlxQuantizedArray quantized{
      checked_loaded_tensor(tensors, weight_name),
      std::move(decode_scales),
      std::move(decode_biases),
      group_size,
      bits};
  session.quantized_tensors.insert_or_assign(tensor_id, std::move(quantized));
}

void register_loaded_weight_tensor(
    EdgeCmlxQwen35Session& session,
    int tensor_id,
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    const std::string& weight_name,
    int group_size,
    int bits) {
  if (loaded_quantized_companions_exist(tensors, weight_name)) {
    register_loaded_quantized_tensor(
        session, tensor_id, tensors, weight_name, group_size, bits);
  } else {
    register_loaded_float_tensor(session, tensor_id, tensors, weight_name);
  }
}

void register_loaded_weight_tensor_if_exists(
    EdgeCmlxQwen35Session& session,
    int tensor_id,
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    const std::string& weight_name,
    int group_size,
    int bits) {
  if (!loaded_tensor_exists(tensors, weight_name)) {
    return;
  }
  register_loaded_weight_tensor(
      session, tensor_id, tensors, weight_name, group_size, bits);
}

void register_loaded_float_tensor_if_exists(
    EdgeCmlxQwen35Session& session,
    int tensor_id,
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    const std::string& tensor_name) {
  if (loaded_tensor_exists(tensors, tensor_name)) {
    register_loaded_float_tensor(session, tensor_id, tensors, tensor_name);
  }
}

std::string qwen35_lm_head_weight_name(
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    const std::string& model_prefix) {
  std::vector<std::string> candidates;
  constexpr const char* model_suffix = ".model";
  constexpr size_t model_suffix_length = 6;
  if (model_prefix.size() > model_suffix_length &&
      model_prefix.compare(
          model_prefix.size() - model_suffix_length,
          model_suffix_length,
          model_suffix) == 0) {
    candidates.push_back(
        model_prefix.substr(0, model_prefix.size() - model_suffix_length) +
        ".lm_head.weight");
  }
  candidates.push_back(model_prefix + ".lm_head.weight");
  candidates.push_back("lm_head.weight");
  for (const auto& candidate : candidates) {
    if (loaded_tensor_exists(tensors, candidate)) {
      return candidate;
    }
  }
  return model_prefix + ".embed_tokens.weight";
}

void register_qwen35_common_layer_tensors(
    EdgeCmlxQwen35Session& session,
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    const std::string& layer_prefix,
    int layer_index,
    int group_size,
    int bits) {
  register_loaded_float_tensor(
      session,
      qwen35_layer_input_norm_id(layer_index),
      tensors,
      layer_prefix + ".input_layernorm.weight");
  register_loaded_float_tensor(
      session,
      qwen35_layer_post_attention_norm_id(layer_index),
      tensors,
      layer_prefix + ".post_attention_layernorm.weight");
  if (qwen35_uses_sparse_moe_mlp(session)) {
    register_loaded_quantized_tensor(
        session,
        qwen35_layer_moe_gate_id(layer_index),
        tensors,
        layer_prefix + ".mlp.gate.weight",
        group_size,
        bits);
    register_loaded_quantized_tensor(
        session,
        qwen35_layer_moe_switch_gate_id(layer_index),
        tensors,
        layer_prefix + ".mlp.switch_mlp.gate_proj.weight",
        group_size,
        bits);
    register_loaded_quantized_tensor(
        session,
        qwen35_layer_moe_switch_up_id(layer_index),
        tensors,
        layer_prefix + ".mlp.switch_mlp.up_proj.weight",
        group_size,
        bits);
    register_loaded_quantized_tensor(
        session,
        qwen35_layer_moe_switch_down_id(layer_index),
        tensors,
        layer_prefix + ".mlp.switch_mlp.down_proj.weight",
        group_size,
        bits);
    register_loaded_quantized_tensor(
        session,
        qwen35_layer_moe_shared_gate_id(layer_index),
        tensors,
        layer_prefix + ".mlp.shared_expert.gate_proj.weight",
        group_size,
        bits);
    register_loaded_quantized_tensor(
        session,
        qwen35_layer_moe_shared_up_id(layer_index),
        tensors,
        layer_prefix + ".mlp.shared_expert.up_proj.weight",
        group_size,
        bits);
    register_loaded_quantized_tensor(
        session,
        qwen35_layer_moe_shared_down_id(layer_index),
        tensors,
        layer_prefix + ".mlp.shared_expert.down_proj.weight",
        group_size,
        bits);
    register_loaded_quantized_tensor(
        session,
        qwen35_layer_moe_shared_expert_gate_id(layer_index),
        tensors,
        layer_prefix + ".mlp.shared_expert_gate.weight",
        group_size,
        bits);
  } else {
    register_loaded_weight_tensor(
        session,
        qwen35_layer_mlp_gate_id(layer_index),
        tensors,
        layer_prefix + ".mlp.gate_proj.weight",
        group_size,
        bits);
    register_loaded_weight_tensor(
        session,
        qwen35_layer_mlp_up_id(layer_index),
        tensors,
        layer_prefix + ".mlp.up_proj.weight",
        group_size,
        bits);
    register_loaded_weight_tensor(
        session,
        qwen35_layer_mlp_down_id(layer_index),
        tensors,
        layer_prefix + ".mlp.down_proj.weight",
        group_size,
        bits);
  }
}

void register_qwen35_full_attention_layer_tensors(
    EdgeCmlxQwen35Session& session,
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    const std::string& layer_prefix,
    int layer_index,
    int group_size,
    int bits) {
  register_qwen35_common_layer_tensors(
      session, tensors, layer_prefix, layer_index, group_size, bits);
  register_loaded_weight_tensor(
      session,
      qwen35_layer_attention_query_id(layer_index),
      tensors,
      layer_prefix + ".self_attn.q_proj.weight",
      group_size,
      bits);
  register_loaded_weight_tensor(
      session,
      qwen35_layer_attention_key_id(layer_index),
      tensors,
      layer_prefix + ".self_attn.k_proj.weight",
      group_size,
      bits);
  register_loaded_weight_tensor(
      session,
      qwen35_layer_attention_value_id(layer_index),
      tensors,
      layer_prefix + ".self_attn.v_proj.weight",
      group_size,
      bits);
  if (loaded_tensor_exists(tensors, layer_prefix + ".self_attn.q_norm.weight")) {
    register_loaded_float_tensor(
        session,
        qwen35_layer_attention_query_norm_id(layer_index),
        tensors,
        layer_prefix + ".self_attn.q_norm.weight");
  }
  if (loaded_tensor_exists(tensors, layer_prefix + ".self_attn.k_norm.weight")) {
    register_loaded_float_tensor(
        session,
        qwen35_layer_attention_key_norm_id(layer_index),
        tensors,
        layer_prefix + ".self_attn.k_norm.weight");
  }
  register_loaded_weight_tensor(
      session,
      qwen35_layer_attention_output_id(layer_index),
      tensors,
      layer_prefix + ".self_attn.o_proj.weight",
      group_size,
      bits);
}

void register_qwen35_gdn_layer_tensors(
    EdgeCmlxQwen35Session& session,
    const std::unordered_map<std::string, mlx::core::array>& tensors,
    const std::string& layer_prefix,
    int layer_index,
    int group_size,
    int bits) {
  register_qwen35_common_layer_tensors(
      session, tensors, layer_prefix, layer_index, group_size, bits);
  register_loaded_quantized_tensor(
      session,
      qwen35_layer_gdn_qkv_id(layer_index),
      tensors,
      layer_prefix + ".linear_attn.in_proj_qkv.weight",
      group_size,
      bits);
  register_loaded_quantized_tensor(
      session,
      qwen35_layer_gdn_z_id(layer_index),
      tensors,
      layer_prefix + ".linear_attn.in_proj_z.weight",
      group_size,
      bits);
  register_loaded_quantized_tensor(
      session,
      qwen35_layer_gdn_a_id(layer_index),
      tensors,
      layer_prefix + ".linear_attn.in_proj_a.weight",
      group_size,
      bits);
  register_loaded_quantized_tensor(
      session,
      qwen35_layer_gdn_b_id(layer_index),
      tensors,
      layer_prefix + ".linear_attn.in_proj_b.weight",
      group_size,
      bits);
  register_loaded_float_tensor(
      session,
      qwen35_layer_gdn_conv1d_id(layer_index),
      tensors,
      layer_prefix + ".linear_attn.conv1d.weight");
  register_loaded_float_tensor(
      session,
      qwen35_layer_gdn_a_log_id(layer_index),
      tensors,
      layer_prefix + ".linear_attn.A_log");
  register_qwen35_gdn_neg_exp_a_log(session, layer_index);
  register_loaded_float_tensor(
      session,
      qwen35_layer_gdn_dt_bias_id(layer_index),
      tensors,
      layer_prefix + ".linear_attn.dt_bias");
  register_loaded_float_tensor(
      session,
      qwen35_layer_gdn_norm_id(layer_index),
      tensors,
      layer_prefix + ".linear_attn.norm.weight");
  register_loaded_quantized_tensor(
      session,
      qwen35_layer_gdn_output_id(layer_index),
      tensors,
      layer_prefix + ".linear_attn.out_proj.weight",
      group_size,
      bits);
}

void validate_qwen35_config(const EdgeCmlxQwen35Config& config) {
  if (config.layer_count <= 0 || config.hidden_size <= 0 ||
      config.vocabulary_size <= 0 || config.intermediate_size <= 0 ||
      config.attention_head_count <= 0 || config.key_value_head_count <= 0 ||
      config.attention_head_dimension <= 0 ||
      config.linear_key_head_count <= 0 ||
      config.linear_value_head_count <= 0 ||
      config.linear_key_head_dimension <= 0 ||
      config.linear_value_head_dimension <= 0 ||
      config.linear_conv_kernel_size <= 0 || config.rotary_dimension <= 0 ||
      config.rope_theta <= 0.0f || config.rms_norm_epsilon < 0.0f) {
    throw std::runtime_error("Qwen3.5 Cmlx session config is invalid");
  }
}

int quantized_output_columns(
    const EdgeCmlxQuantizedArray& weights,
    bool transpose) {
  const int packed_rows = static_cast<int>(weights.packed.shape(0));
  const int packed_cols = static_cast<int>(weights.packed.shape(1));
  const int expanded_packed_cols = packed_cols * 32 / weights.bits;
  return transpose ? packed_rows : expanded_packed_cols;
}

int quantized_inner_columns(
    const EdgeCmlxQuantizedArray& weights,
    bool transpose) {
  const int packed_rows = static_cast<int>(weights.packed.shape(0));
  const int packed_cols = static_cast<int>(weights.packed.shape(1));
  const int expanded_packed_cols = packed_cols * 32 / weights.bits;
  return transpose ? expanded_packed_cols : packed_rows;
}

edge_cmlx::primitives::QuantizedWeight qwen35_quantized_weight_ref(
    const EdgeCmlxQuantizedArray& weights) {
  return edge_cmlx::primitives::QuantizedWeight{
      weights.packed,
      weights.scales,
      weights.biases,
      weights.group_size,
      weights.bits};
}

edge_cmlx::blocks::LinearWeight qwen35_linear_weight_ref(
    const EdgeCmlxQwen35Session& session,
    int tensor_id) {
  if (const auto* dense = optional_qwen35_float_tensor(session, tensor_id)) {
    return edge_cmlx::blocks::LinearWeight{std::nullopt, dense};
  }
  return edge_cmlx::blocks::LinearWeight{
      qwen35_quantized_weight_ref(checked_qwen35_quantized_tensor(session, tensor_id)),
      nullptr};
}

int qwen35_linear_output_columns(
    const edge_cmlx::blocks::LinearWeight& weight,
    int input_hidden,
    bool quantized_transpose) {
  if (weight.dense != nullptr) {
    (void)input_hidden;
    return static_cast<int>(weight.dense->shape(0));
  }
  if (weight.quantized.has_value()) {
    const int packed_rows = static_cast<int>(weight.quantized->packed.shape(0));
    const int packed_cols = static_cast<int>(weight.quantized->packed.shape(1));
    const int expanded_packed_cols = packed_cols * 32 / weight.quantized->bits;
    return quantized_transpose ? packed_rows : expanded_packed_cols;
  }
  throw std::runtime_error("Qwen3.5 linear weight is missing");
}

mlx::core::array affine_quantized_matmul_array(
    const mlx::core::array& input,
    const EdgeCmlxQuantizedArray& weights,
    bool transpose,
    mlx::core::StreamOrDevice stream) {
  if (transpose) {
    return edge_cmlx::primitives::quantized_linear(
        input,
        qwen35_quantized_weight_ref(weights),
        stream);
  }
  return mlx::core::quantized_matmul(
      input,
      weights.packed,
      weights.scales,
      weights.biases,
      transpose,
      weights.group_size,
      weights.bits,
      "affine",
      stream);
}

EdgeCmlxQuantizedArray qwen35_quantize_attention_cache_update(
    const mlx::core::array& update,
    int group_size,
    int bits,
    mlx::core::StreamOrDevice stream) {
  auto quantized = mlx::core::quantize(
      update,
      group_size,
      bits,
      "affine",
      std::nullopt,
      stream);
  if (quantized.size() < 3) {
    throw std::runtime_error("Qwen3.5 quantized attention cache update missing affine components");
  }
  return EdgeCmlxQuantizedArray{
      quantized[0],
      quantized[1],
      quantized[2],
      group_size,
      bits};
}

mlx::core::array qwen35_grow_quantized_cache_component(
    const mlx::core::array& update_component,
    const std::optional<mlx::core::array>& existing_component,
    int copy_length,
    int cache_capacity,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;

  auto shape = update_component.shape();
  if (shape.size() < 3) {
    throw std::runtime_error("Qwen3.5 quantized cache component rank is invalid");
  }
  shape[2] = cache_capacity;
  auto grown = zeros(shape, update_component.dtype(), stream);
  if (existing_component.has_value() && copy_length > 0) {
    const auto& existing = *existing_component;
    grown = slice_update(
        grown,
        slice(
            existing,
            Shape{0, 0, 0, 0},
            Shape{
                existing.shape(0),
                existing.shape(1),
                copy_length,
                existing.shape(3)},
            stream),
        Shape{0, 0, 0, 0},
        Shape{
            existing.shape(0),
            existing.shape(1),
            copy_length,
            existing.shape(3)},
        stream);
  }
  return grown;
}

EdgeCmlxQuantizedArray qwen35_update_quantized_attention_cache(
    std::unordered_map<int, EdgeCmlxQuantizedArray>& states,
    int layer_index,
    const mlx::core::array& update,
    int local_position_offset,
    int local_new_offset,
    int cache_capacity,
    int group_size,
    int bits,
    mlx::core::StreamOrDevice stream,
    std::vector<mlx::core::array>& eval_outputs,
    Qwen35EvalOutputInventory* eval_inventory,
    const char* cache_role) {
  using namespace mlx::core;

  auto quantized_update = qwen35_quantize_attention_cache_update(
      update,
      group_size,
      bits,
      stream);
  auto state_item = states.find(layer_index);
  EdgeCmlxQuantizedArray state = [&]() -> EdgeCmlxQuantizedArray {
    if (state_item == states.end() ||
        state_item->second.packed.shape(2) < local_new_offset) {
      std::optional<array> existing_packed;
      std::optional<array> existing_scales;
      std::optional<array> existing_biases;
      int copy_length = 0;
      if (state_item != states.end()) {
        existing_packed = state_item->second.packed;
        existing_scales = state_item->second.scales;
        existing_biases = state_item->second.biases;
        copy_length = std::min(
            static_cast<int>(state_item->second.packed.shape(2)),
            local_position_offset);
      }
      return EdgeCmlxQuantizedArray{
          qwen35_grow_quantized_cache_component(
              quantized_update.packed,
              existing_packed,
              copy_length,
              cache_capacity,
              stream),
          qwen35_grow_quantized_cache_component(
              quantized_update.scales,
              existing_scales,
              copy_length,
              cache_capacity,
              stream),
          qwen35_grow_quantized_cache_component(
              quantized_update.biases,
              existing_biases,
              copy_length,
              cache_capacity,
              stream),
          group_size,
          bits};
    }
    return state_item->second;
  }();

  const int update_end = local_position_offset + static_cast<int>(update.shape(2));
  if (update_end > static_cast<int>(state.packed.shape(2))) {
    throw std::runtime_error("Qwen3.5 quantized attention cache update crossed capacity");
  }
  state.packed = slice_update(
      state.packed,
      quantized_update.packed,
      Shape{0, 0, local_position_offset, 0},
      Shape{
          quantized_update.packed.shape(0),
          quantized_update.packed.shape(1),
          update_end,
          quantized_update.packed.shape(3)},
      stream);
  state.scales = slice_update(
      state.scales,
      quantized_update.scales,
      Shape{0, 0, local_position_offset, 0},
      Shape{
          quantized_update.scales.shape(0),
          quantized_update.scales.shape(1),
          update_end,
          quantized_update.scales.shape(3)},
      stream);
  state.biases = slice_update(
      state.biases,
      quantized_update.biases,
      Shape{0, 0, local_position_offset, 0},
      Shape{
          quantized_update.biases.shape(0),
          quantized_update.biases.shape(1),
          update_end,
          quantized_update.biases.shape(3)},
      stream);

  auto stored_packed = stop_gradient(state.packed, stream);
  auto stored_scales = stop_gradient(state.scales, stream);
  auto stored_biases = stop_gradient(state.biases, stream);
  states.insert_or_assign(
      layer_index,
      EdgeCmlxQuantizedArray{
          stored_packed,
          stored_scales,
          stored_biases,
          group_size,
          bits});
  const bool is_key_cache = std::strcmp(cache_role, "key") == 0;
  qwen35_eval_output_push(
      eval_outputs,
      stored_packed,
      eval_inventory,
      is_key_cache ? "quant_key_packed" : "quant_value_packed");
  qwen35_eval_output_push(
      eval_outputs,
      stored_scales,
      eval_inventory,
      is_key_cache ? "quant_key_scales" : "quant_value_scales");
  qwen35_eval_output_push(
      eval_outputs,
      stored_biases,
      eval_inventory,
      is_key_cache ? "quant_key_biases" : "quant_value_biases");

  return EdgeCmlxQuantizedArray{
      slice(
          state.packed,
          Shape{0, 0, 0, 0},
          Shape{state.packed.shape(0), state.packed.shape(1), local_new_offset, state.packed.shape(3)},
          stream),
      slice(
          state.scales,
          Shape{0, 0, 0, 0},
          Shape{state.scales.shape(0), state.scales.shape(1), local_new_offset, state.scales.shape(3)},
          stream),
      slice(
          state.biases,
          Shape{0, 0, 0, 0},
          Shape{state.biases.shape(0), state.biases.shape(1), local_new_offset, state.biases.shape(3)},
          stream),
      group_size,
      bits};
}

std::optional<EdgeCmlxQuantizedArray> qwen35_materialize_prefill_attention_cache(
    std::unordered_map<int, mlx::core::array>& dense_states,
    std::unordered_map<int, EdgeCmlxQuantizedArray>& quantized_states,
    int layer_index,
    int active_length,
    int cache_capacity,
    int group_size,
    int bits,
    mlx::core::StreamOrDevice stream,
    std::vector<mlx::core::array>& eval_outputs,
    Qwen35EvalOutputInventory* eval_inventory,
    const char* cache_role) {
  using namespace mlx::core;

  if (active_length <= 0 ||
      quantized_states.find(layer_index) != quantized_states.end()) {
    return std::nullopt;
  }
  auto dense_item = dense_states.find(layer_index);
  if (dense_item == dense_states.end()) {
    return std::nullopt;
  }
  const auto& dense_state = dense_item->second;
  if (dense_state.ndim() != 4 || dense_state.shape(2) < active_length) {
    throw std::runtime_error(
        "Qwen3.5 prefill FP16 attention cache has invalid shape");
  }
  auto active_state = slice(
      dense_state,
      Shape{0, 0, 0, 0},
      Shape{
          dense_state.shape(0),
          dense_state.shape(1),
          active_length,
          dense_state.shape(3)},
      stream);
  auto quantized_state = qwen35_update_quantized_attention_cache(
      quantized_states,
      layer_index,
      active_state,
      0,
      active_length,
      cache_capacity,
      group_size,
      bits,
      stream,
      eval_outputs,
      eval_inventory,
      cache_role);
  dense_states.erase(layer_index);
  return quantized_state;
}

mlx::core::array qwen35_dequantize_attention_cache(
    const EdgeCmlxQuantizedArray& cache,
    mlx::core::Dtype dtype,
    mlx::core::StreamOrDevice stream) {
  return mlx::core::dequantize(
      cache.packed,
      cache.scales,
      cache.biases,
      cache.group_size,
      cache.bits,
      "affine",
      std::nullopt,
      dtype,
      stream);
}

mlx::core::array qwen35_quantized_attention_scores(
    const mlx::core::array& queries,
    const EdgeCmlxQuantizedArray& key_cache,
    float scale,
    int kv_heads,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;

  auto scaled_queries = multiply(array(scale, queries.dtype()), queries, stream);
  auto q_keys = key_cache;
  const int q_heads = static_cast<int>(queries.shape(1));
  const int n_repeats = q_heads / kv_heads;
  if (n_repeats > 1) {
    scaled_queries = unflatten(scaled_queries, 1, {kv_heads, n_repeats}, stream);
    q_keys.packed = expand_dims(q_keys.packed, 2, stream);
    q_keys.scales = expand_dims(q_keys.scales, 2, stream);
    q_keys.biases = expand_dims(q_keys.biases, 2, stream);
  }
  auto raw_scores = quantized_matmul(
      scaled_queries,
      q_keys.packed,
      q_keys.scales,
      q_keys.biases,
      true,
      q_keys.group_size,
      q_keys.bits,
      "affine",
      stream);
  if (n_repeats > 1) {
    raw_scores = mean(raw_scores, 2, false, stream);
  }
  return raw_scores;
}

mlx::core::array qwen35_scores_by_kv_head(
    const mlx::core::array& scores,
    int q_heads,
    int kv_heads,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;

  if (q_heads == kv_heads) {
    return scores;
  }
  if (q_heads % kv_heads != 0) {
    throw std::runtime_error("qwen35 DSR scores received incompatible head counts");
  }
  auto grouped_scores =
      unflatten(scores, 1, {kv_heads, q_heads / kv_heads}, stream);
  return mean(grouped_scores, 2, false, stream);
}

EdgeCmlxQuantizedArray qwen35_compact_quantized_attention_cache(
    const EdgeCmlxQuantizedArray& cache,
    const mlx::core::array& keep_indices,
    int keep_count,
    int cache_capacity,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;

  const auto compact_component = [&](
      const array& component) -> array {
    auto compacted = take(component, keep_indices, 2, stream);
    return slice_update(
        zeros(
            Shape{
                component.shape(0),
                component.shape(1),
                cache_capacity,
                component.shape(3)},
            component.dtype(),
            stream),
        compacted,
        Shape{0, 0, 0, 0},
        Shape{
            component.shape(0),
            component.shape(1),
            keep_count,
            component.shape(3)},
        stream);
  };
  return EdgeCmlxQuantizedArray{
      compact_component(cache.packed),
      compact_component(cache.scales),
      compact_component(cache.biases),
      cache.group_size,
      cache.bits};
}

mlx::core::array qwen35_quantized_attention_output(
    const mlx::core::array& queries,
    const EdgeCmlxQuantizedArray& key_cache,
    const EdgeCmlxQuantizedArray& value_cache,
    float scale,
    int kv_heads,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;

  auto scaled_queries = multiply(array(scale, queries.dtype()), queries, stream);
  auto q_keys = key_cache;
  auto q_values = value_cache;
  const int batch_size = static_cast<int>(queries.shape(0));
  const int q_heads = static_cast<int>(queries.shape(1));
  const int token_count = static_cast<int>(queries.shape(2));
  const int head_dim = static_cast<int>(queries.shape(3));
  const int n_repeats = q_heads / kv_heads;
  if (n_repeats > 1) {
    scaled_queries = unflatten(scaled_queries, 1, {kv_heads, n_repeats}, stream);
    q_keys.packed = expand_dims(q_keys.packed, 2, stream);
    q_keys.scales = expand_dims(q_keys.scales, 2, stream);
    q_keys.biases = expand_dims(q_keys.biases, 2, stream);
    q_values.packed = expand_dims(q_values.packed, 2, stream);
    q_values.scales = expand_dims(q_values.scales, 2, stream);
    q_values.biases = expand_dims(q_values.biases, 2, stream);
  }

  auto scores = quantized_matmul(
      scaled_queries,
      q_keys.packed,
      q_keys.scales,
      q_keys.biases,
      true,
      q_keys.group_size,
      q_keys.bits,
      "affine",
      stream);
  if (token_count > 1) {
    const int q_len = static_cast<int>(scores.shape(scores.ndim() - 2));
    const int k_len = static_cast<int>(scores.shape(scores.ndim() - 1));
    auto q_indices = add(
        arange(0, q_len, int32, stream),
        array(k_len - q_len, int32),
        stream);
    auto k_indices = arange(0, k_len, int32, stream);
    auto causal_mask = greater_equal(
        expand_dims(q_indices, 1, stream),
        expand_dims(k_indices, 0, stream),
        stream);
    scores = where(
        causal_mask,
        scores,
        array(std::numeric_limits<float>::lowest(), scores.dtype()),
        stream);
  }

  auto weights = softmax(scores, -1, false, stream);
  auto output = quantized_matmul(
      weights,
      q_values.packed,
      q_values.scales,
      q_values.biases,
      false,
      q_values.group_size,
      q_values.bits,
      "affine",
      stream);
  if (n_repeats > 1) {
    output = reshape(
        output,
        Shape{batch_size, q_heads, token_count, head_dim},
        stream);
  }
  return output;
}

mlx::core::array silu_array(
    const mlx::core::array& input,
    mlx::core::StreamOrDevice stream) {
  return mlx::core::multiply(
      input, mlx::core::sigmoid(input, stream), stream);
}

mlx::core::array softplus_array(
    const mlx::core::array& input,
    mlx::core::StreamOrDevice stream) {
  return mlx::core::logaddexp(
      input, mlx::core::array(0.0f, input.dtype()), stream);
}

mlx::core::array qwen35_rms_norm_optional(
    const mlx::core::array& input,
    const mlx::core::array* weight,
    float epsilon,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;

  if (weight == nullptr) {
    return fast::rms_norm(
        input, std::optional<array>{}, epsilon, stream);
  }
  auto typed_weight = weight->dtype() == input.dtype()
      ? *weight
      : astype(*weight, input.dtype(), stream);
  return fast::rms_norm(
      input, std::optional<array>(typed_weight), epsilon, stream);
}

mlx::core::array qwen35_quantized_mlp_array(
    const mlx::core::array& input,
    const EdgeCmlxQuantizedArray& gate,
    const EdgeCmlxQuantizedArray& up,
    const EdgeCmlxQuantizedArray& down,
    mlx::core::StreamOrDevice stream,
    bool activation_float32 = false) {
  return edge_cmlx::blocks::swiglu_mlp(
      input,
      qwen35_quantized_weight_ref(gate),
      qwen35_quantized_weight_ref(up),
      qwen35_quantized_weight_ref(down),
      edge_cmlx::blocks::MLPConfig{activation_float32},
      stream);
}

mlx::core::array qwen35_quantized_switch_linear_array(
    const mlx::core::array& input,
    const EdgeCmlxQuantizedArray& weights,
    const mlx::core::array& expert_indices,
    bool expand_input,
    bool squeeze_output,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;

  auto switch_input = input;
  if (expand_input) {
    if (input.ndim() != 2) {
      throw std::runtime_error("Qwen3.5 MoE switch input must be rank-2");
    }
    switch_input = reshape(
        input,
        Shape{input.shape(0), 1, 1, input.shape(1)},
        stream);
  }
  auto output = gather_qmm(
      switch_input,
      weights.packed,
      weights.scales,
      std::optional<array>(weights.biases),
      std::nullopt,
      expert_indices,
      true,
      weights.group_size,
      weights.bits,
      "affine",
      false,
      stream);
  return squeeze_output ? squeeze(output, -2, stream) : output;
}

mlx::core::array qwen35_quantized_moe_mlp_array(
    const mlx::core::array& input,
    const EdgeCmlxQwen35Session& qwen_session,
    int layer_index,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;

  const auto& config = qwen_session.config;
  const int top_k = config.moe_experts_per_token;
  const int expert_count = config.moe_num_experts;
  if (top_k <= 0 || expert_count <= 0 || top_k > expert_count) {
    throw std::runtime_error("Qwen3.5 MoE configuration is invalid");
  }

  auto logits = affine_quantized_matmul_array(
      input,
      checked_qwen35_quantized_tensor(
          qwen_session, qwen35_layer_moe_gate_id(layer_index)),
      true,
      stream);
  auto probabilities = softmax(logits, -1, true, stream);
  const int kth = expert_count - top_k;
  auto partitioned = argpartition(probabilities, kth, -1, stream);
  auto expert_indices = slice(
      partitioned,
      Shape{0, kth},
      Shape{partitioned.shape(0), expert_count},
      stream);
  auto expert_scores = take_along_axis(
      probabilities,
      expert_indices,
      -1,
      stream);
  if (config.moe_normalize_topk_probabilities != 0) {
    expert_scores = divide(
        expert_scores,
        sum(expert_scores, -1, true, stream),
        stream);
  }

  auto up = qwen35_quantized_switch_linear_array(
      input,
      checked_qwen35_quantized_tensor(
          qwen_session, qwen35_layer_moe_switch_up_id(layer_index)),
      expert_indices,
      true,
      false,
      stream);
  auto gate = qwen35_quantized_switch_linear_array(
      input,
      checked_qwen35_quantized_tensor(
          qwen_session, qwen35_layer_moe_switch_gate_id(layer_index)),
      expert_indices,
      true,
      false,
      stream);
  auto activated = multiply(silu_array(gate, stream), up, stream);
  auto switched = qwen35_quantized_switch_linear_array(
      activated,
      checked_qwen35_quantized_tensor(
          qwen_session, qwen35_layer_moe_switch_down_id(layer_index)),
      expert_indices,
      false,
      true,
      stream);
  auto combined = sum(
      multiply(switched, expand_dims(expert_scores, -1, stream), stream),
      -2,
      false,
      stream);

  auto shared = qwen35_quantized_mlp_array(
      input,
      checked_qwen35_quantized_tensor(
          qwen_session, qwen35_layer_moe_shared_gate_id(layer_index)),
      checked_qwen35_quantized_tensor(
          qwen_session, qwen35_layer_moe_shared_up_id(layer_index)),
      checked_qwen35_quantized_tensor(
          qwen_session, qwen35_layer_moe_shared_down_id(layer_index)),
      stream);
  auto shared_gate = affine_quantized_matmul_array(
      input,
      checked_qwen35_quantized_tensor(
          qwen_session, qwen35_layer_moe_shared_expert_gate_id(layer_index)),
      true,
      stream);
  shared = multiply(sigmoid(shared_gate, stream), shared, stream);
  return add(combined, shared, stream);
}

mlx::core::array qwen35_mlp_array(
    const mlx::core::array& input,
    const EdgeCmlxQwen35Session& qwen_session,
    int layer_index,
    mlx::core::StreamOrDevice stream,
    bool activation_float32 = false) {
  using namespace mlx::core;

  if (qwen35_uses_sparse_moe_mlp(qwen_session)) {
    return qwen35_quantized_moe_mlp_array(
        input,
        qwen_session,
        layer_index,
        stream);
  }
  const int gate_id = qwen35_layer_mlp_gate_id(layer_index);
  const int up_id = qwen35_layer_mlp_up_id(layer_index);
  const int down_id = qwen35_layer_mlp_down_id(layer_index);
  if (optional_qwen35_float_tensor(qwen_session, gate_id) != nullptr ||
      optional_qwen35_float_tensor(qwen_session, up_id) != nullptr ||
      optional_qwen35_float_tensor(qwen_session, down_id) != nullptr) {
    auto gate_output = qwen35_linear_array(input, qwen_session, gate_id, stream);
    auto up_output = qwen35_linear_array(input, qwen_session, up_id, stream);
    if (activation_float32) {
      gate_output = astype(gate_output, float32, stream);
      up_output = astype(up_output, float32, stream);
    }
    auto activation = multiply(silu_array(gate_output, stream), up_output, stream);
    return qwen35_linear_array(activation, qwen_session, down_id, stream);
  }
  return qwen35_quantized_mlp_array(
      input,
      checked_qwen35_quantized_tensor(
          qwen_session, gate_id),
      checked_qwen35_quantized_tensor(
          qwen_session, up_id),
      checked_qwen35_quantized_tensor(
          qwen_session, down_id),
      stream,
      activation_float32);
}

mlx::core::array qwen35_rms_norm(
    const mlx::core::array& input,
    const mlx::core::array& weight,
    float epsilon,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;

  auto typed_weight = weight.dtype() == input.dtype()
      ? weight
      : astype(weight, input.dtype(), stream);
  return fast::rms_norm(
      input, std::optional<array>(typed_weight), epsilon, stream);
}

mlx::core::array qwen35_embedding_for_indices(
    const EdgeCmlxQwen35Session& session,
    const mlx::core::array& token_indices,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  auto indices = token_indices.dtype() == int32
      ? token_indices
      : astype(token_indices, int32, stream);
  if (indices.ndim() == 0) {
    indices = reshape(indices, Shape{1}, stream);
  }
  const int embedding_id = qwen35_embedding_id();
  if (const auto* embedding =
          optional_qwen35_float_tensor(session, embedding_id)) {
    return take(*embedding, indices, 0, stream);
  }

  const auto& quantized_embedding =
      checked_qwen35_quantized_tensor(session, embedding_id);
  auto packed = take(quantized_embedding.packed, indices, 0, stream);
  auto scales = take(quantized_embedding.scales, indices, 0, stream);
  auto biases = take(quantized_embedding.biases, indices, 0, stream);
  return dequantize(
      packed,
      scales,
      std::optional<array>(biases),
      quantized_embedding.group_size,
      quantized_embedding.bits,
      "affine",
      std::nullopt,
      float16,
      stream);
}

mlx::core::array qwen35_embedding_for_indices(
    const EdgeCmlxQwen35Session& session,
    int embedding_id,
    const mlx::core::array& token_indices,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  auto indices = token_indices.dtype() == int32
      ? token_indices
      : astype(token_indices, int32, stream);
  if (indices.ndim() == 0) {
    indices = reshape(indices, Shape{1}, stream);
  }
  if (const auto* embedding =
          optional_qwen35_float_tensor(session, embedding_id)) {
    return take(*embedding, indices, 0, stream);
  }

  const auto& quantized_embedding =
      checked_qwen35_quantized_tensor(session, embedding_id);
  auto packed = take(quantized_embedding.packed, indices, 0, stream);
  auto scales = take(quantized_embedding.scales, indices, 0, stream);
  auto biases = take(quantized_embedding.biases, indices, 0, stream);
  return dequantize(
      packed,
      scales,
      std::optional<array>(biases),
      quantized_embedding.group_size,
      quantized_embedding.bits,
      "affine",
      std::nullopt,
      float16,
      stream);
}

mlx::core::array qwen35_embedding_array(
    const EdgeCmlxQwen35Session& session,
    const int* token_ids,
    int token_count,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  return qwen35_embedding_for_indices(
      session,
      array(token_ids, Shape{token_count}, int32),
      stream);
}

mlx::core::array qwen35_linear_array(
    const mlx::core::array& input,
    const EdgeCmlxQwen35Session& session,
    int tensor_id,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  if (const auto* weight = optional_qwen35_float_tensor(session, tensor_id)) {
    return edge_cmlx::primitives::linear(input, *weight, nullptr, stream);
  }
  return affine_quantized_matmul_array(
      input,
      checked_qwen35_quantized_tensor(session, tensor_id),
      true,
      stream);
}

mlx::core::array qwen35_add_optional_bias(
    const mlx::core::array& input,
    const EdgeCmlxQwen35Session& session,
    int tensor_id,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  const auto* bias = optional_qwen35_float_tensor(session, tensor_id);
  if (bias == nullptr) {
    return input;
  }
  auto typed_bias = bias->dtype() == input.dtype()
      ? *bias
      : astype(*bias, input.dtype(), stream);
  return add(input, typed_bias, stream);
}

struct Qwen35GDNDecodeAttentionArrays {
  mlx::core::array output;
  mlx::core::array next_conv_state;
  mlx::core::array next_recurrent_state;
};

const mlx::core::fast::CustomKernelFunction& qwen35_gdn_conv_silu_kernel() {
  static const auto kernel = mlx::core::fast::metal_kernel(
      "edge_qwen35_gdn_conv_silu",
      {"conv_input", "conv_weights"},
      {"conv_output"},
      R"(
            auto token = thread_position_in_grid.x;
            auto channel = thread_position_in_grid.y;
            float acc = 0.0f;
            for (int kernel_idx = 0; kernel_idx < K; ++kernel_idx) {
              auto input_idx = (token + kernel_idx) * C + channel;
              auto weight_idx = channel * K + kernel_idx;
              acc += static_cast<float>(conv_input[input_idx]) *
                  static_cast<float>(conv_weights[weight_idx]);
            }
            conv_output[token * C + channel] =
                static_cast<InT>(acc / (1.0f + exp(-acc)));
        )");
  return kernel;
}

mlx::core::array qwen35_gdn_conv_silu_array(
    const mlx::core::array& conv_input,
    const mlx::core::array& conv_weights,
    int token_count,
    int conv_hidden,
    int conv_kernel,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;

  auto outputs = qwen35_gdn_conv_silu_kernel()(
      {conv_input, conv_weights},
      {Shape{token_count, conv_hidden}},
      {conv_input.dtype()},
      {token_count, conv_hidden, 1},
      {1, 1, 1},
      {
          {"InT", conv_input.dtype()},
          {"C", conv_hidden},
          {"K", conv_kernel},
      },
      std::nullopt,
      false,
      stream);
  return outputs[0];
}

struct Qwen35TopKLastRowArrays {
  mlx::core::array logits;
  mlx::core::array indices;
};

const mlx::core::fast::CustomKernelFunction& qwen35_topk_last_row_kernel() {
  static const auto kernel = mlx::core::fast::metal_kernel(
      "edge_qwen35_topk_last_row",
      {"logits"},
      {"top_logits", "top_indices"},
      R"(
            auto tid = thread_position_in_threadgroup.x;
            constexpr uint invalid_index = 0xffffffffu;
            const int columns = logits_shape[logits_ndim - 1];
            const int rows = logits_ndim >= 2 ? logits_shape[logits_ndim - 2] : 1;
            const int row_offset = (rows - 1) * columns;

            float local_values[K];
            uint local_indices[K];
            for (int i = 0; i < K; ++i) {
              local_values[i] = -INFINITY;
              local_indices[i] = invalid_index;
            }

            for (int column = static_cast<int>(tid);
                 column < columns;
                 column += ThreadCount) {
              const float value = static_cast<float>(logits[row_offset + column]);
              const uint token = static_cast<uint>(column);
              const bool beats_worst =
                  value > local_values[K - 1] ||
                  (value == local_values[K - 1] && token < local_indices[K - 1]);
              if (!beats_worst) {
                continue;
              }

              int insert = K - 1;
              while (insert > 0 &&
                     (value > local_values[insert - 1] ||
                      (value == local_values[insert - 1] &&
                       token < local_indices[insert - 1]))) {
                local_values[insert] = local_values[insert - 1];
                local_indices[insert] = local_indices[insert - 1];
                --insert;
              }
              local_values[insert] = value;
              local_indices[insert] = token;
            }

            threadgroup float shared_values[ThreadCount * K];
            threadgroup uint shared_indices[ThreadCount * K];
            const int base = static_cast<int>(tid) * K;
            for (int i = 0; i < K; ++i) {
              shared_values[base + i] = local_values[i];
              shared_indices[base + i] = local_indices[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0) {
              float final_values[K];
              uint final_indices[K];
              for (int i = 0; i < K; ++i) {
                final_values[i] = -INFINITY;
                final_indices[i] = invalid_index;
              }

              for (int candidate = 0; candidate < ThreadCount * K; ++candidate) {
                const float value = shared_values[candidate];
                const uint token = shared_indices[candidate];
                if (token == invalid_index) {
                  continue;
                }
                const bool beats_worst =
                    value > final_values[K - 1] ||
                    (value == final_values[K - 1] && token < final_indices[K - 1]);
                if (!beats_worst) {
                  continue;
                }

                int insert = K - 1;
                while (insert > 0 &&
                       (value > final_values[insert - 1] ||
                        (value == final_values[insert - 1] &&
                         token < final_indices[insert - 1]))) {
                  final_values[insert] = final_values[insert - 1];
                  final_indices[insert] = final_indices[insert - 1];
                  --insert;
                }
                final_values[insert] = value;
                final_indices[insert] = token;
              }

              for (int i = 0; i < K; ++i) {
                top_logits[i] = final_values[i];
                top_indices[i] = static_cast<uint32_t>(final_indices[i]);
              }
            }
        )");
  return kernel;
}

std::optional<Qwen35TopKLastRowArrays> qwen35_topk_last_row_arrays(
    const mlx::core::array& logits,
    int top_k,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;
  constexpr int thread_count = 64;
  constexpr int max_top_k = 64;
  if (top_k <= 0 || top_k > max_top_k || logits.ndim() < 1 ||
      logits.shape(-1) <= top_k) {
    return std::nullopt;
  }

  auto outputs = qwen35_topk_last_row_kernel()(
      {logits},
      {Shape{1, top_k}, Shape{1, top_k}},
      {float32, uint32},
      {thread_count, 1, 1},
      {thread_count, 1, 1},
      {
          {"K", top_k},
          {"ThreadCount", thread_count},
      },
      std::nullopt,
      false,
      stream);
  return Qwen35TopKLastRowArrays{outputs[0], outputs[1]};
}

void qwen35_record_sample_diagnostics(
    EdgeCmlxQwen35Session& qwen_session,
    const mlx::core::array& logits,
    const mlx::core::array& sampled_token,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;

  qwen_session.pending_sample_diagnostics.clear();
  if (!qwen35_sample_diagnostics_enabled()) {
    return;
  }

  const int vocab = static_cast<int>(logits.shape(-1));
  const int diagnostic_top_k = std::min(
      std::max(top_k > 0 ? top_k : 40, 1),
      std::min(vocab - 1, 64));
  auto top = qwen35_topk_last_row_arrays(logits, diagnostic_top_k, stream);
  if (!top.has_value()) {
    return;
  }

  eval({sampled_token, top->logits, top->indices});
  const int sampled_id = static_cast<int>(sampled_token.data<uint32_t>()[0]);
  const float* top_logits = top->logits.data<float>();
  const uint32_t* top_indices = top->indices.data<uint32_t>();

  std::vector<float> base_prob(static_cast<size_t>(diagnostic_top_k), 0.0f);
  std::vector<bool> keep(static_cast<size_t>(diagnostic_top_k), true);
  float max_logit = top_logits[0];
  for (int i = 1; i < diagnostic_top_k; ++i) {
    max_logit = std::max(max_logit, top_logits[i]);
  }
  double base_sum = 0.0;
  for (int i = 0; i < diagnostic_top_k; ++i) {
    const double value = std::exp(static_cast<double>(top_logits[i] - max_logit));
    base_prob[static_cast<size_t>(i)] = static_cast<float>(value);
    base_sum += value;
  }
  if (base_sum > 0.0) {
    for (float& probability : base_prob) {
      probability = static_cast<float>(static_cast<double>(probability) / base_sum);
    }
  }

  if (top_p > 0.0f && top_p < 1.0f) {
    std::vector<int> order(static_cast<size_t>(diagnostic_top_k));
    for (int i = 0; i < diagnostic_top_k; ++i) {
      order[static_cast<size_t>(i)] = i;
    }
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
      return base_prob[static_cast<size_t>(lhs)] < base_prob[static_cast<size_t>(rhs)];
    });
    double cumulative = 0.0;
    const double drop_mass = 1.0 - static_cast<double>(top_p);
    for (int index : order) {
      cumulative += base_prob[static_cast<size_t>(index)];
      keep[static_cast<size_t>(index)] = cumulative > drop_mass;
    }
  }
  if (min_p > 0.0f) {
    float max_probability = 0.0f;
    for (float probability : base_prob) {
      max_probability = std::max(max_probability, probability);
    }
    const float threshold = max_probability * min_p;
    for (int i = 0; i < diagnostic_top_k; ++i) {
      keep[static_cast<size_t>(i)] =
          keep[static_cast<size_t>(i)] &&
          base_prob[static_cast<size_t>(i)] >= threshold;
    }
    keep[0] = true;
  }

  std::vector<float> final_prob = base_prob;
  if (temperature > 0.0f) {
    float kept_max_logit = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < diagnostic_top_k; ++i) {
      if (keep[static_cast<size_t>(i)]) {
        kept_max_logit = std::max(kept_max_logit, top_logits[i]);
      }
    }
    final_prob.assign(static_cast<size_t>(diagnostic_top_k), 0.0f);
    double final_sum = 0.0;
    if (std::isfinite(kept_max_logit)) {
      for (int i = 0; i < diagnostic_top_k; ++i) {
        if (!keep[static_cast<size_t>(i)]) {
          continue;
        }
        const double value = std::exp(
            static_cast<double>(top_logits[i] - kept_max_logit) /
            static_cast<double>(temperature));
        final_prob[static_cast<size_t>(i)] = static_cast<float>(value);
        final_sum += value;
      }
    }
    if (final_sum > 0.0) {
      for (float& probability : final_prob) {
        probability = static_cast<float>(static_cast<double>(probability) / final_sum);
      }
    }
  }

  int sampled_rank = -1;
  float sampled_probability = 0.0f;
  float sampled_logit = std::numeric_limits<float>::quiet_NaN();
  float sampled_base_probability = 0.0f;
  for (int i = 0; i < diagnostic_top_k; ++i) {
    if (static_cast<int>(top_indices[i]) == sampled_id) {
      sampled_rank = i + 1;
      sampled_probability = temperature <= 0.0f
          ? base_prob[static_cast<size_t>(i)]
          : final_prob[static_cast<size_t>(i)];
      sampled_base_probability = base_prob[static_cast<size_t>(i)];
      sampled_logit = top_logits[i];
      break;
    }
  }
  const float argmax_probability = temperature <= 0.0f
      ? base_prob[0]
      : final_prob[0];
  const float second_logit = diagnostic_top_k > 1
      ? top_logits[1]
      : std::numeric_limits<float>::quiet_NaN();
  const float margin = diagnostic_top_k > 1
      ? top_logits[0] - second_logit
      : std::numeric_limits<float>::quiet_NaN();

  std::ostringstream output;
  output << std::fixed << std::setprecision(6)
         << "sample token=" << sampled_id
         << " rank=" << sampled_rank
         << " prob=" << sampled_probability
         << " baseProb=" << sampled_base_probability
         << " logit=" << sampled_logit
         << " argmax=" << static_cast<int>(top_indices[0])
         << " argmaxProb=" << argmax_probability
         << " argmaxBaseProb=" << base_prob[0]
         << " second=" << (diagnostic_top_k > 1 ? static_cast<int>(top_indices[1]) : -1)
         << " secondBaseProb=" << (diagnostic_top_k > 1 ? base_prob[1] : 0.0f)
         << " margin=" << margin
         << " mode=" << (temperature <= 0.0f ? "greedy" : "sampled")
         << " temp=" << temperature
         << " topK=" << top_k
         << " topP=" << top_p
         << " minP=" << min_p
         << " candidates=";
  const int candidate_count = std::min(diagnostic_top_k, 8);
  for (int i = 0; i < candidate_count; ++i) {
    if (i > 0) {
      output << ",";
    }
    output << static_cast<int>(top_indices[i])
           << ":" << final_prob[static_cast<size_t>(i)]
           << ":" << top_logits[i];
  }
  qwen_session.pending_sample_diagnostics = output.str();
}

const mlx::core::fast::CustomKernelFunction& qwen35_gated_delta_kernel() {
  static const auto kernel = mlx::core::fast::metal_kernel(
      "edge_qwen35_gated_delta_step",
      {"q", "k", "v", "g", "beta", "state_in", "T"},
      {"y", "state_out"},
      R"(
            auto hv_idx = thread_position_in_grid.z;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            auto q_ = q + hk_idx * Dk;
            auto k_ = k + hk_idx * Dk;

            auto v_ = v + hv_idx * Dv;
            y += hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            auto g_ = g + hv_idx;
            auto beta_ = beta + hv_idx;

            auto i_state = state_in + (hv_idx * Dv + dv_idx) * Dk;
            auto o_state = state_out + (hv_idx * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            for (int t = 0; t < T; ++t) {
              float kv_mem = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] * g_[0];
                kv_mem += state[i] * static_cast<float>(k_[s_idx]);
              }
              kv_mem = simd_sum(kv_mem);

              auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem) *
                  static_cast<float>(beta_[0]);

              float out = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
                out += state[i] * static_cast<float>(q_[s_idx]);
              }
              out = simd_sum(out);
              if (thread_index_in_simdgroup == 0) {
                y[dv_idx] = static_cast<InT>(out);
              }

              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv;
              beta_ += Hv;
            }

            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<StT>(state[i]);
            }
        )");
  return kernel;
}

std::pair<mlx::core::array, mlx::core::array> qwen35_gated_delta_update_array(
    const mlx::core::array& query,
    const mlx::core::array& key,
    const mlx::core::array& value,
    const mlx::core::array& decay,
    const mlx::core::array& beta,
    const mlx::core::array& recurrent_state,
    int key_head_count,
    int value_head_count,
    int key_head_dimension,
    int value_head_dimension,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;

  const int token_count = static_cast<int>(query.shape(1));
  const auto dtype = query.dtype();
  if (key_head_dimension % 32 != 0) {
    throw std::runtime_error(
        "edge_qwen35_gated_delta_step requires key head dimension divisible by 32");
  }
  if (value_head_count % key_head_count != 0) {
    throw std::runtime_error(
        "edge_qwen35_gated_delta_step requires value heads divisible by key heads");
  }

  auto token_count_scalar = array(token_count, int32);
  auto outputs = qwen35_gated_delta_kernel()(
      {query, key, value, decay, beta, recurrent_state, token_count_scalar},
      {
          Shape{token_count, value_head_count, value_head_dimension},
          recurrent_state.shape(),
      },
      {dtype, recurrent_state.dtype()},
      {32, value_head_dimension, value_head_count},
      {32, 4, 1},
      {
          {"InT", dtype},
          {"StT", recurrent_state.dtype()},
          {"Dk", key_head_dimension},
          {"Dv", value_head_dimension},
          {"Hk", key_head_count},
          {"Hv", value_head_count},
      },
      std::nullopt,
      false,
      stream);
  return {outputs[0], outputs[1]};
}

const mlx::core::fast::CustomKernelFunction& qwen35_gdn_gate_kernel() {
  static const auto kernel = mlx::core::fast::metal_kernel(
      "edge_qwen35_gdn_gate",
      {"a", "b", "dt_bias", "neg_exp_a_log"},
      {"decay", "beta"},
      R"(
            auto idx = thread_position_in_grid.x;
            const int columns = a_shape[a_ndim - 1];
            const int rows = a_ndim >= 2 ? a_shape[a_ndim - 2] : 1;
            if (idx >= rows * columns) {
              return;
            }
            const int column = idx % columns;

            const float gate =
                static_cast<float>(a[idx]) +
                static_cast<float>(dt_bias[column]);
            const float softplus = gate > 0.0f
                ? gate + log(1.0f + exp(-gate))
                : log(1.0f + exp(gate));
            decay[idx] = exp(static_cast<float>(neg_exp_a_log[column]) * softplus);

            const float b_value = static_cast<float>(b[idx]);
            beta[idx] = static_cast<BT>(1.0f / (1.0f + exp(-b_value)));
        )");
  return kernel;
}

std::pair<mlx::core::array, mlx::core::array> qwen35_gdn_gate_array(
    const mlx::core::array& a,
    const mlx::core::array& b,
    const mlx::core::array& dt_bias,
    const mlx::core::array& neg_exp_a_log,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;

  constexpr int thread_count = 256;
  const auto outputs = qwen35_gdn_gate_kernel()(
      {a, b, dt_bias, neg_exp_a_log},
      {a.shape(), b.shape()},
      {float32, b.dtype()},
      {static_cast<int>(a.size()), 1, 1},
      {thread_count, 1, 1},
      {
          {"BT", b.dtype()},
      },
      std::nullopt,
      false,
      stream);
  return {outputs[0], outputs[1]};
}

std::pair<mlx::core::array, mlx::core::array> qwen35_gated_delta_ops_array(
    const mlx::core::array& query,
    const mlx::core::array& key,
    const mlx::core::array& value,
    const mlx::core::array& decay,
    const mlx::core::array& beta,
    const mlx::core::array& recurrent_state,
    int key_head_count,
    int value_head_count,
    int key_head_dimension,
    int value_head_dimension,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;

  const int token_count = static_cast<int>(query.shape(1));
  auto repeat_factor = value_head_count / key_head_count;
  auto query_repeated = repeat(query, repeat_factor, 2, stream);
  auto key_repeated = repeat(key, repeat_factor, 2, stream);

  auto next_recurrent_state = recurrent_state;
  std::vector<array> recurrent_outputs;
  recurrent_outputs.reserve(static_cast<size_t>(token_count));
  for (int token = 0; token < token_count; ++token) {
    auto query_token = reshape(
        slice(
            query_repeated,
            Shape{0, token, 0, 0},
            Shape{1, token + 1, value_head_count, key_head_dimension},
            stream),
        Shape{value_head_count, key_head_dimension},
        stream);
    auto key_token = reshape(
        slice(
            key_repeated,
            Shape{0, token, 0, 0},
            Shape{1, token + 1, value_head_count, key_head_dimension},
            stream),
        Shape{value_head_count, key_head_dimension},
        stream);
    auto value_token = reshape(
        slice(
            value,
            Shape{0, token, 0, 0},
            Shape{1, token + 1, value_head_count, value_head_dimension},
            stream),
        Shape{value_head_count, value_head_dimension},
        stream);
    auto decay_token = reshape(
        slice(
            decay,
            Shape{token, 0},
            Shape{token + 1, value_head_count},
            stream),
        Shape{value_head_count},
        stream);
    auto beta_token = reshape(
        slice(
            beta,
            Shape{token, 0},
            Shape{token + 1, value_head_count},
            stream),
        Shape{value_head_count},
        stream);
    auto decayed_state = multiply(
        next_recurrent_state,
        reshape(decay_token, Shape{value_head_count, 1, 1}, stream),
        stream);
    auto kv_memory = sum(
        multiply(
            decayed_state,
            expand_dims(key_token, 1, stream),
            stream),
        -1,
        false,
        stream);
    auto delta = multiply(
        subtract(value_token, kv_memory, stream),
        reshape(beta_token, Shape{value_head_count, 1}, stream),
        stream);
    next_recurrent_state = add(
        decayed_state,
        multiply(
            expand_dims(key_token, 1, stream),
            expand_dims(delta, 2, stream),
            stream),
        stream);
    recurrent_outputs.push_back(
        expand_dims(
            sum(
                multiply(
                    next_recurrent_state,
                    expand_dims(query_token, 1, stream),
                    stream),
                -1,
                false,
                stream),
            0,
            stream));
  }
  return {concatenate(recurrent_outputs, 0, stream), next_recurrent_state};
}

}

namespace edge_cmlx::blocks {

array linear(
    const array& input,
    const LinearWeight& weight,
    StreamOrDevice stream) {
  if (weight.dense != nullptr) {
    return primitives::linear(input, *weight.dense, nullptr, stream);
  }
  if (weight.quantized.has_value()) {
    return primitives::quantized_linear(input, *weight.quantized, stream);
  }
  throw std::runtime_error("blocks::linear received no weight");
}

GDNDecodeResult gdn_attention(
    const array& input,
    const array& conv_state,
    const array& recurrent_state,
    const GDNWeights& weights,
    const GDNConfig& config,
    bool log_dtype_diagnostic,
    StreamOrDevice stream) {
  using namespace mlx::core;

  const int key_hidden = config.key_head_count * config.key_head_dimension;
  const int value_hidden = config.value_head_count * config.value_head_dimension;
  const int conv_hidden = key_hidden * 2 + value_hidden;
  const int token_count = static_cast<int>(input.shape(0));

  auto mixed_qkv = primitives::quantized_linear(input, weights.qkv, stream);
  auto z = primitives::quantized_linear(input, weights.z, stream);
  auto a = primitives::quantized_linear(input, weights.a, stream);
  auto b = primitives::quantized_linear(input, weights.b, stream);

  const auto gdn_dtype = input.dtype();
  auto typed_conv_state = conv_state.dtype() == gdn_dtype
      ? conv_state
      : astype(conv_state, gdn_dtype, stream);
  auto typed_conv_weights = weights.conv.dtype() == gdn_dtype
      ? weights.conv
      : astype(weights.conv, gdn_dtype, stream);

  auto conv_input = concatenate({typed_conv_state, mixed_qkv}, 0, stream);
  auto next_conv_state = slice(
      conv_input,
      Shape{token_count, 0},
      Shape{token_count + config.conv_kernel_size - 1, conv_hidden},
      stream);

  auto conv_activated = ::edge_cmlx::detail::qwen35_gdn_conv_silu_array(
      conv_input,
      typed_conv_weights,
      token_count,
      conv_hidden,
      config.conv_kernel_size,
      stream);

  auto query = slice(
      conv_activated,
      Shape{0, 0},
      Shape{token_count, key_hidden},
      stream);
  auto key = slice(
      conv_activated,
      Shape{0, key_hidden},
      Shape{token_count, key_hidden * 2},
      stream);
  auto value = slice(
      conv_activated,
      Shape{0, key_hidden * 2},
      Shape{token_count, conv_hidden},
      stream);
  ::edge_cmlx::detail::validate_array_element_count(
      query,
      static_cast<size_t>(token_count * key_hidden),
      "blocks::gdn_attention query");
  ::edge_cmlx::detail::validate_array_element_count(
      key,
      static_cast<size_t>(token_count * key_hidden),
      "blocks::gdn_attention key");
  ::edge_cmlx::detail::validate_array_element_count(
      value,
      static_cast<size_t>(token_count * value_hidden),
      "blocks::gdn_attention value");

  auto query_heads = reshape(
      query,
      Shape{1, token_count, config.key_head_count, config.key_head_dimension},
      stream);
  auto key_heads = reshape(
      key,
      Shape{1, token_count, config.key_head_count, config.key_head_dimension},
      stream);
  const float inv_key_dim =
      1.0f / std::sqrt(static_cast<float>(config.key_head_dimension));
  if (::edge_cmlx::detail::qwen35_fused_rms_scale_enabled()) {
    query_heads = fast::rms_norm_scale(
        query_heads,
        config.rms_norm_epsilon,
        inv_key_dim * inv_key_dim,
        stream);
    key_heads = fast::rms_norm_scale(
        key_heads,
        config.rms_norm_epsilon,
        inv_key_dim,
        stream);
  } else {
    auto query_norm = fast::rms_norm(
        query_heads,
        std::optional<array>{},
        config.rms_norm_epsilon,
        stream);
    query_heads = multiply(
        query_norm,
        array(inv_key_dim * inv_key_dim, query_norm.dtype()),
        stream);
    auto key_norm = fast::rms_norm(
        key_heads,
        std::optional<array>{},
        config.rms_norm_epsilon,
        stream);
    key_heads = multiply(
        key_norm,
        array(inv_key_dim, key_norm.dtype()),
        stream);
  }

  auto value_heads = reshape(
      value,
      Shape{
          1,
          token_count,
          config.value_head_count,
          config.value_head_dimension},
      stream);

  auto a_f32 = a.dtype() == float32 ? a : astype(a, float32, stream);
  auto typed_dt_bias = weights.dt_bias.dtype() == float32
      ? weights.dt_bias
      : astype(weights.dt_bias, float32, stream);
  auto neg_exp_a_log = [&]() -> array {
    if (weights.neg_exp_a_log != nullptr) {
      return weights.neg_exp_a_log->dtype() == float32
          ? *weights.neg_exp_a_log
          : astype(*weights.neg_exp_a_log, float32, stream);
    }
    auto typed_a_log = weights.a_log.dtype() == float32
        ? weights.a_log
        : astype(weights.a_log, float32, stream);
    return multiply(exp(typed_a_log, stream), array(-1.0f, float32), stream);
  }();
  auto [decay, beta] = [&]() -> std::pair<array, array> {
    if (::edge_cmlx::detail::qwen35_fused_gdn_gate_enabled()) {
      return ::edge_cmlx::detail::qwen35_gdn_gate_array(
          a_f32,
          b,
          typed_dt_bias,
          neg_exp_a_log,
          stream);
    }
    auto gate_a = add(a_f32, typed_dt_bias, stream);
    auto decay = exp(
        multiply(
            neg_exp_a_log,
            ::edge_cmlx::detail::softplus_array(gate_a, stream),
            stream),
        stream);
    return {decay, sigmoid(b, stream)};
  }();
  if (log_dtype_diagnostic) {
    fprintf(stderr,
        "[CmlxShim] GDN DTYPE layer0 conv=%s q=%s k=%s decay=%s beta=%s\n",
        ::edge_cmlx::detail::dtype_label(conv_activated.dtype()),
        ::edge_cmlx::detail::dtype_label(query_heads.dtype()),
        ::edge_cmlx::detail::dtype_label(key_heads.dtype()),
        ::edge_cmlx::detail::dtype_label(decay.dtype()),
        ::edge_cmlx::detail::dtype_label(beta.dtype()));
  }

  auto recurrent_update = config.key_head_dimension % 32 == 0
      ? ::edge_cmlx::detail::qwen35_gated_delta_update_array(
            query_heads,
            key_heads,
            value_heads,
            decay,
            beta,
            recurrent_state,
            config.key_head_count,
            config.value_head_count,
            config.key_head_dimension,
            config.value_head_dimension,
            stream)
      : ::edge_cmlx::detail::qwen35_gated_delta_ops_array(
            query_heads,
            key_heads,
            value_heads,
            decay,
            beta,
            recurrent_state,
            config.key_head_count,
            config.value_head_count,
            config.key_head_dimension,
            config.value_head_dimension,
            stream);
  auto recurrent_output = recurrent_update.first;
  auto next_recurrent_state = recurrent_update.second;

  auto recurrent_norm_input = reshape(
      recurrent_output,
      Shape{
          token_count,
          config.value_head_count,
          config.value_head_dimension},
      stream);
  auto normalized = ::edge_cmlx::detail::qwen35_rms_norm(
      recurrent_norm_input,
      weights.norm,
      config.rms_norm_epsilon,
      stream);
  auto gated = multiply(
      reshape(primitives::silu(z, stream), Shape{token_count, value_hidden}, stream),
      reshape(normalized, Shape{token_count, value_hidden}, stream),
      stream);
  auto output = primitives::quantized_linear(gated, weights.output, stream);
  return GDNDecodeResult{output, next_conv_state, next_recurrent_state};
}

TransformerAttentionProjection transformer_attention_projection(
    const array& input,
    const TransformerAttentionWeights& weights,
    const TransformerAttentionConfig& config,
    int position_offset,
    StreamOrDevice stream) {
  using namespace mlx::core;

  const int attention_hidden = config.attention_heads * config.head_dim;
  const int kv_hidden = config.key_value_heads * config.head_dim;
  const int token_count = static_cast<int>(input.shape(0));
  const bool uses_query_gate =
      config.query_projection_hidden == attention_hidden * 2;
  if (!uses_query_gate &&
      config.query_projection_hidden != attention_hidden) {
    throw std::runtime_error(
        "blocks::transformer_attention_projection query projection shape mismatch");
  }

  auto projection_input = input.dtype() == float16
      ? input
      : astype(input, float16, stream);
  auto query_projection = linear(projection_input, weights.query, stream);
  auto query = query_projection;
  std::optional<array> gate;
  if (uses_query_gate) {
    auto query_projection_heads = reshape(
        query_projection,
        Shape{1, token_count, config.attention_heads, config.head_dim * 2},
        stream);
    auto query_projection_split = split(query_projection_heads, 2, 3, stream);
    query = reshape(
        query_projection_split[0],
        Shape{token_count, attention_hidden},
        stream);
    gate = reshape(
        query_projection_split[1],
        Shape{token_count, attention_hidden},
        stream);
  }
  auto key = linear(projection_input, weights.key, stream);
  auto value = linear(projection_input, weights.value, stream);
  ::edge_cmlx::detail::validate_array_element_count(
      query,
      static_cast<size_t>(token_count * attention_hidden),
      "blocks::transformer_attention_projection query");
  ::edge_cmlx::detail::validate_array_element_count(
      key,
      static_cast<size_t>(token_count * kv_hidden),
      "blocks::transformer_attention_projection key");
  ::edge_cmlx::detail::validate_array_element_count(
      value,
      static_cast<size_t>(token_count * kv_hidden),
      "blocks::transformer_attention_projection value");

  auto queries = transpose(
      ::edge_cmlx::detail::qwen35_rms_norm_optional(
          reshape(
              query,
              Shape{1, token_count, config.attention_heads, config.head_dim},
              stream),
          weights.query_norm,
          config.rms_norm_epsilon,
          stream),
      {0, 2, 1, 3},
      stream);
  auto keys = transpose(
      ::edge_cmlx::detail::qwen35_rms_norm_optional(
          reshape(
              key,
              Shape{1, token_count, config.key_value_heads, config.head_dim},
              stream),
          weights.key_norm,
          config.rms_norm_epsilon,
          stream),
      {0, 2, 1, 3},
      stream);
  auto values = transpose(
      reshape(
          value,
          Shape{1, token_count, config.key_value_heads, config.head_dim},
          stream),
      {0, 2, 1, 3},
      stream);

  queries = fast::rope(
      queries,
      config.rotary_dimension,
      false,
      std::optional<float>(config.rope_theta),
      1.0f,
      position_offset,
      std::nullopt,
      stream);
  keys = fast::rope(
      keys,
      config.rotary_dimension,
      false,
      std::optional<float>(config.rope_theta),
      1.0f,
      position_offset,
      std::nullopt,
      stream);

  return TransformerAttentionProjection{
      queries,
      keys,
      values,
      gate,
      token_count,
      attention_hidden};
}

TransformerAttentionResult transformer_attention(
    const TransformerAttentionProjection& projection,
    const TransformerAttentionCacheView& cache,
    const TransformerAttentionWeights& weights,
    const TransformerAttentionConfig& config,
    const TransformerAttentionRuntime& runtime,
    StreamOrDevice stream) {
  using namespace mlx::core;

  const float scale = 1.0f / std::sqrt(static_cast<float>(config.head_dim));
  const bool has_quantized_cache =
      cache.quantized_key_packed != nullptr ||
      cache.quantized_key_scales != nullptr ||
      cache.quantized_key_biases != nullptr ||
      cache.quantized_value_packed != nullptr ||
      cache.quantized_value_scales != nullptr ||
      cache.quantized_value_biases != nullptr;
  const bool has_dense_cache =
      cache.dense_keys != nullptr || cache.dense_values != nullptr;
  if (has_quantized_cache && has_dense_cache) {
    throw std::runtime_error(
        "blocks::transformer_attention received both dense and quantized cache");
  }

  std::optional<array> dsr_scores;
  bool fused_check_logged = false;
  bool fused_attention_used = false;
  std::optional<array> attention;
  if (has_quantized_cache) {
    if (cache.quantized_key_packed == nullptr ||
        cache.quantized_key_scales == nullptr ||
        cache.quantized_key_biases == nullptr ||
        cache.quantized_value_packed == nullptr ||
        cache.quantized_value_scales == nullptr ||
        cache.quantized_value_biases == nullptr) {
      throw std::runtime_error(
          "blocks::transformer_attention received partial quantized cache");
    }
    ::edge_cmlx::detail::EdgeCmlxQuantizedArray quantized_key_cache{
        *cache.quantized_key_packed,
        *cache.quantized_key_scales,
        *cache.quantized_key_biases,
        cache.quantized_group_size,
        cache.quantized_bits};
    ::edge_cmlx::detail::EdgeCmlxQuantizedArray quantized_value_cache{
        *cache.quantized_value_packed,
        *cache.quantized_value_scales,
        *cache.quantized_value_biases,
        cache.quantized_group_size,
        cache.quantized_bits};
    const int token_count = projection.token_count;
    const auto fused_stream = to_stream(stream);
    const bool can_use_fused =
        token_count == 1 &&
        fused_stream.device == Device::gpu &&
        (projection.queries.dtype() == float16 ||
         projection.queries.dtype() == bfloat16) &&
        (config.head_dim == 128 || config.head_dim == 256) &&
        projection.queries.shape(2) <= 8 &&
        (cache.quantized_bits == 4 || cache.quantized_bits == 8) &&
        (cache.quantized_group_size == 64 ||
         cache.quantized_group_size == 128);
    if (token_count == 1 && runtime.log_fused_check) {
      fused_check_logged = true;
      fprintf(stderr,
          "[CmlxShim] FUSED CHECK layer%d: queries_dtype=%s can_use_fused=%s "
          "quant_bits=%d quant_gs=%d head_dim=%d\n",
          runtime.layer_index,
          projection.queries.dtype() == float16 ? "f16" :
              (projection.queries.dtype() == float32 ? "f32" : "other"),
          can_use_fused ? "YES" : "NO",
          cache.quantized_bits,
          cache.quantized_group_size,
          config.head_dim);
    }
    if (can_use_fused) {
      auto [attn_out, attn_scores] =
          fast::scaled_dot_product_attention_quantized_with_scores(
              projection.queries,
              quantized_key_cache.packed,
              quantized_key_cache.scales,
              quantized_key_cache.biases,
              quantized_value_cache.packed,
              quantized_value_cache.scales,
              quantized_value_cache.biases,
              scale,
              cache.quantized_bits,
              cache.quantized_group_size,
              std::nullopt,
              stream);
      attention = attn_out;
      fused_attention_used = true;
      if (runtime.compute_scores) {
        dsr_scores = ::edge_cmlx::detail::qwen35_scores_by_kv_head(
            attn_scores,
            config.attention_heads,
            config.key_value_heads,
            stream);
      }
    } else {
      attention = ::edge_cmlx::detail::qwen35_quantized_attention_output(
          projection.queries,
          quantized_key_cache,
          quantized_value_cache,
          scale,
          config.key_value_heads,
          stream);
      if (runtime.compute_scores) {
        dsr_scores = ::edge_cmlx::detail::qwen35_quantized_attention_scores(
            projection.queries,
            quantized_key_cache,
            scale,
            config.key_value_heads,
            stream);
      }
    }
  } else {
    if (cache.dense_keys == nullptr || cache.dense_values == nullptr) {
      throw std::runtime_error(
          "blocks::transformer_attention received missing dense cache");
    }
    attention = fast::scaled_dot_product_attention(
        projection.queries,
        *cache.dense_keys,
        *cache.dense_values,
        scale,
        runtime.causal ? "causal" : "",
        std::nullopt,
        std::nullopt,
        stream);
    if (runtime.compute_scores) {
      auto score_query =
          multiply(array(scale, projection.queries.dtype()), projection.queries, stream);
      auto score_keys = *cache.dense_keys;
      const int q_heads = static_cast<int>(projection.queries.shape(1));
      const int n_repeats = q_heads / config.key_value_heads;
      if (n_repeats > 1) {
        score_query =
            unflatten(score_query, 1, {config.key_value_heads, n_repeats}, stream);
        score_keys = expand_dims(score_keys, 2, stream);
      }
      auto scores =
          matmul(score_query, swapaxes(score_keys, -1, -2, stream), stream);
      if (n_repeats > 1) {
        scores = mean(scores, 2, false, stream);
      }
      dsr_scores = scores;
    }
  }

  if (!attention.has_value()) {
    throw std::runtime_error("blocks::transformer_attention produced no output");
  }
  auto attention_output = reshape(
      transpose(*attention, {0, 2, 1, 3}, stream),
      Shape{projection.token_count, projection.attention_hidden},
      stream);
  auto gated = projection.gate.has_value()
      ? multiply(attention_output, sigmoid(*projection.gate, stream), stream)
      : attention_output;
  auto output = linear(gated, weights.output, stream);
  return TransformerAttentionResult{
      output,
      dsr_scores,
      fused_check_logged,
      fused_attention_used};
}

}

namespace edge_cmlx::detail {

Qwen35GDNDecodeAttentionArrays qwen35_gdn_decode_attention_array(
    const EdgeCmlxQwen35Session& session,
    int layer_index,
    const mlx::core::array& input,
    const mlx::core::array& conv_state,
    const mlx::core::array& recurrent_state,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;

  const auto& qkv_weights = checked_qwen35_quantized_tensor(
      session, qwen35_layer_gdn_qkv_id(layer_index));
  const auto& z_weights = checked_qwen35_quantized_tensor(
      session, qwen35_layer_gdn_z_id(layer_index));
  const auto& a_weights = checked_qwen35_quantized_tensor(
      session, qwen35_layer_gdn_a_id(layer_index));
  const auto& b_weights = checked_qwen35_quantized_tensor(
      session, qwen35_layer_gdn_b_id(layer_index));
  const auto& conv_weights = checked_qwen35_float_tensor(
      session, qwen35_layer_gdn_conv1d_id(layer_index));
  const auto& a_log = checked_qwen35_float_tensor(
      session, qwen35_layer_gdn_a_log_id(layer_index));
  const auto neg_exp_a_log_item =
      session.gdn_neg_exp_a_log_tensors.find(layer_index);
  const auto* neg_exp_a_log =
      neg_exp_a_log_item == session.gdn_neg_exp_a_log_tensors.end()
      ? nullptr
      : &neg_exp_a_log_item->second;
  const auto& dt_bias = checked_qwen35_float_tensor(
      session, qwen35_layer_gdn_dt_bias_id(layer_index));
  const auto& norm = checked_qwen35_float_tensor(
      session, qwen35_layer_gdn_norm_id(layer_index));
  const auto& out_weights = checked_qwen35_quantized_tensor(
      session, qwen35_layer_gdn_output_id(layer_index));

  const auto& config = session.config;
  auto attention = edge_cmlx::blocks::gdn_attention(
      input,
      conv_state,
      recurrent_state,
      edge_cmlx::blocks::GDNWeights{
          qwen35_quantized_weight_ref(qkv_weights),
          qwen35_quantized_weight_ref(z_weights),
          qwen35_quantized_weight_ref(a_weights),
          qwen35_quantized_weight_ref(b_weights),
          conv_weights,
          a_log,
          neg_exp_a_log,
          dt_bias,
          norm,
          qwen35_quantized_weight_ref(out_weights)},
      edge_cmlx::blocks::GDNConfig{
          config.linear_key_head_count,
          config.linear_value_head_count,
          config.linear_key_head_dimension,
          config.linear_value_head_dimension,
          config.linear_conv_kernel_size,
          config.rms_norm_epsilon},
      session.decoded_token_count == 0 && layer_index == 0,
      stream);
  return Qwen35GDNDecodeAttentionArrays{
      attention.output,
      attention.next_conv_state,
      attention.next_recurrent_state};
}

mlx::core::array qwen35_full_attention_decode_array(
    EdgeCmlxQwen35Session& session,
    int layer_index,
    const mlx::core::array& input,
    int position_offset,
    std::vector<mlx::core::array>& eval_outputs,
    Qwen35EvalOutputInventory* eval_inventory,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;

  const auto query_weights = qwen35_linear_weight_ref(
      session, qwen35_layer_tensor_id(layer_index, 10));
  const auto key_weights = qwen35_linear_weight_ref(
      session, qwen35_layer_tensor_id(layer_index, 11));
  const auto value_weights = qwen35_linear_weight_ref(
      session, qwen35_layer_tensor_id(layer_index, 12));
  const auto* query_norm = optional_qwen35_float_tensor(
      session, qwen35_layer_tensor_id(layer_index, 13));
  const auto* key_norm = optional_qwen35_float_tensor(
      session, qwen35_layer_tensor_id(layer_index, 14));
  const auto output_weights = qwen35_linear_weight_ref(
      session, qwen35_layer_tensor_id(layer_index, 15));

  const auto& config = session.config;
  const int attention_heads = config.attention_head_count;
  const int kv_heads = config.key_value_head_count;
  const int head_dim = config.attention_head_dimension;
  const edge_cmlx::blocks::TransformerAttentionConfig attention_config{
      attention_heads,
      kv_heads,
      head_dim,
      config.rotary_dimension,
      qwen35_linear_output_columns(query_weights, config.hidden_size, true),
      config.rope_theta,
      config.rms_norm_epsilon};
  const edge_cmlx::blocks::TransformerAttentionWeights attention_weights{
      query_weights,
      key_weights,
      value_weights,
      output_weights,
      query_norm,
      key_norm};
  auto projection = edge_cmlx::blocks::transformer_attention_projection(
      input,
      attention_weights,
      attention_config,
      position_offset,
      stream);
  const int token_count = projection.token_count;

  const auto dsr_policy_item = session.attention_dsr_policies.find(layer_index);
  const EdgeCmlxQwen35DSRPolicy* dsr_policy =
      dsr_policy_item == session.attention_dsr_policies.end()
      ? nullptr
      : &dsr_policy_item->second;
  const int cache_base_position = session.attention_cache_base_position;
  if (position_offset < cache_base_position) {
    throw std::runtime_error(
        "qwen35_full_attention_decode_array received a position before the attention cache base");
  }
  const int local_position_offset = dsr_policy != nullptr
      ? session.attention_active_lengths[layer_index]
      : position_offset - cache_base_position;
  const int local_new_offset = local_position_offset + token_count;
  const int cache_limit = session.attention_cache_limit;
  const bool ring_decode =
      dsr_policy == nullptr &&
      cache_limit > 0 &&
      token_count == 1 &&
      local_new_offset >= cache_limit;
  const int cache_capacity = dsr_policy != nullptr
      ? std::max(
            qwen35_dsr_transient_capacity(*dsr_policy),
            qwen35_attention_cache_capacity(local_new_offset))
      : qwen35_attention_cache_capacity(local_new_offset);
  const bool use_quantized_attention_cache =
      qwen35_attention_cache_quantization_enabled(session);
  const bool immutable_fa_cache_requested =
      qwen35_immutable_fa_cache_enabled() ||
      qwen35_immutable_fa_cache_layer() == layer_index;
  const bool use_immutable_fa_cache =
      immutable_fa_cache_requested &&
      dsr_policy == nullptr &&
      !use_quantized_attention_cache &&
      !ring_decode;
  const auto update_attention_cache =
      [&](std::unordered_map<int, array>& states,
          const array& update,
          int head_count,
          int dimension,
          const char* inventory_kind) -> array {
    if (use_immutable_fa_cache) {
      auto state_item = states.find(layer_index);
      array state = [&]() -> array {
        if (state_item == states.end()) {
          if (local_position_offset != 0) {
            throw std::runtime_error(
                "Qwen3.5 immutable FA cache missing previous state");
          }
          return update;
        }
        const auto& existing = state_item->second;
        if (existing.shape(2) < local_position_offset) {
          throw std::runtime_error(
              "Qwen3.5 immutable FA cache previous state is too short");
        }
        auto visible_existing = existing.shape(2) == local_position_offset
            ? existing
            : slice(
                  existing,
                  Shape{0, 0, 0, 0},
                  Shape{1, head_count, local_position_offset, dimension},
                  stream);
        return concatenate({visible_existing, update}, 2, stream);
      }();
      auto stored = qwen35_skip_fa_stop_gradient_enabled()
          ? state
          : stop_gradient(state, stream);
      states.insert_or_assign(layer_index, stored);
      return stored;
    }

    auto state_item = states.find(layer_index);
    array state = [&]() -> array {
	      if (state_item == states.end() ||
	          state_item->second.shape(2) < local_new_offset) {
	        auto grown = zeros(
	            Shape{1, head_count, cache_capacity, dimension},
            update.dtype(),
            stream);
        if (state_item != states.end()) {
          const auto& existing = state_item->second;
          grown = slice_update(
              grown,
              existing,
              Shape{0, 0, 0, 0},
              Shape{1, head_count, existing.shape(2), dimension},
              stream);
        }
        return grown;
      }
	      return state_item->second;
	    }();

    const int state_capacity = static_cast<int>(state.shape(2));
    const int update_start =
        ring_decode && state_capacity >= cache_limit
        ? (session.attention_cache_base_index + local_position_offset) %
            state_capacity
        : local_position_offset;
    const int update_end = update_start + token_count;
    if (update_end > state_capacity) {
      throw std::runtime_error(
          "qwen35_full_attention_decode_array ring cache update crossed capacity");
    }
	    state = slice_update(
	        state,
	        update,
	        Shape{0, 0, update_start, 0},
	        Shape{1, head_count, update_end, dimension},
	        stream);
    auto visible = ring_decode && state_capacity >= cache_limit
        ? state
        : slice(
              state,
              Shape{0, 0, 0, 0},
              Shape{1, head_count, local_new_offset, dimension},
              stream);
    auto stored = qwen35_skip_fa_stop_gradient_enabled()
        ? state
        : stop_gradient(state, stream);
    states.insert_or_assign(layer_index, stored);
    qwen35_eval_output_push(
        eval_outputs,
        stored,
        eval_inventory,
        inventory_kind);
    return visible;
  };

  const bool has_quantized_key_cache =
      session.attention_quantized_key_states.find(layer_index) !=
      session.attention_quantized_key_states.end();
  const bool has_quantized_value_cache =
      session.attention_quantized_value_states.find(layer_index) !=
      session.attention_quantized_value_states.end();
  if (has_quantized_key_cache != has_quantized_value_cache) {
    throw std::runtime_error(
        "Qwen3.5 quantized attention cache key/value state mismatch");
  }
  const bool use_prefill_fp16_attention_cache =
      use_quantized_attention_cache &&
      token_count > 1 &&
      !has_quantized_key_cache;
  std::optional<array> key_cache;
  std::optional<array> value_cache;
  std::optional<EdgeCmlxQuantizedArray> quantized_key_cache;
  std::optional<EdgeCmlxQuantizedArray> quantized_value_cache;
  if (use_quantized_attention_cache && token_count == 1) {
    quantized_key_cache = qwen35_materialize_prefill_attention_cache(
        session.attention_key_states,
        session.attention_quantized_key_states,
        layer_index,
        local_position_offset,
        cache_capacity,
        session.attention_cache_quantization_group_size,
        session.attention_cache_quantization_bits,
        stream,
        eval_outputs,
        eval_inventory,
        "key");
    quantized_value_cache = qwen35_materialize_prefill_attention_cache(
        session.attention_value_states,
        session.attention_quantized_value_states,
        layer_index,
        local_position_offset,
        cache_capacity,
        session.attention_cache_quantization_group_size,
        session.attention_cache_quantization_bits,
        stream,
        eval_outputs,
        eval_inventory,
        "value");
    if (quantized_key_cache.has_value() != quantized_value_cache.has_value()) {
      throw std::runtime_error(
          "Qwen3.5 prefill FP16 attention cache key/value materialization mismatch");
    }
    if (quantized_key_cache.has_value() &&
        !session.prefill_fp16_attention_logged) {
      session.prefill_fp16_attention_logged = true;
      fprintf(stderr,
          "[CmlxShim] PREFILL FP16 ATTENTION: quantized dense KV at decode "
          "layer=%d tokens=%d bits=%d group=%d\n",
          layer_index,
          local_position_offset,
          session.attention_cache_quantization_bits,
          session.attention_cache_quantization_group_size);
    }
    if (quantized_key_cache.has_value()) {
      session.prefill_fp16_attention_materialized_pending_clear = true;
    }
  }
  if (use_quantized_attention_cache && !use_prefill_fp16_attention_cache) {
    quantized_key_cache = qwen35_update_quantized_attention_cache(
        session.attention_quantized_key_states,
        layer_index,
        projection.keys,
        local_position_offset,
        local_new_offset,
        cache_capacity,
        session.attention_cache_quantization_group_size,
        session.attention_cache_quantization_bits,
        stream,
        eval_outputs,
        eval_inventory,
        "key");
    quantized_value_cache = qwen35_update_quantized_attention_cache(
        session.attention_quantized_value_states,
        layer_index,
        projection.values,
        local_position_offset,
        local_new_offset,
        cache_capacity,
        session.attention_cache_quantization_group_size,
        session.attention_cache_quantization_bits,
        stream,
        eval_outputs,
        eval_inventory,
        "value");
  } else {
    if (use_prefill_fp16_attention_cache &&
        !session.prefill_fp16_attention_logged) {
      session.prefill_fp16_attention_logged = true;
      fprintf(stderr,
          "[CmlxShim] PREFILL FP16 ATTENTION: using dense batch SDPA "
          "layer=%d tokens=%d bits=%d group=%d\n",
          layer_index,
          token_count,
          session.attention_cache_quantization_bits,
          session.attention_cache_quantization_group_size);
    }
    key_cache = update_attention_cache(
        session.attention_key_states,
        projection.keys,
        kv_heads,
        head_dim,
        "fa_dense_key_state");
    value_cache = update_attention_cache(
        session.attention_value_states,
        projection.values,
        kv_heads,
        head_dim,
        "fa_dense_value_state");
  }
  const bool compute_dsr_scores =
      dsr_policy != nullptr &&
      token_count == 1 &&
      local_new_offset >=
          static_cast<int>(
              static_cast<float>(dsr_policy->max_size) *
              dsr_policy->score_activation_ratio);
  edge_cmlx::blocks::TransformerAttentionCacheView attention_cache;
  if (quantized_key_cache.has_value() != quantized_value_cache.has_value()) {
    throw std::runtime_error(
        "Qwen3.5 quantized attention cache key/value state mismatch");
  }
  if (quantized_key_cache.has_value()) {
    attention_cache.quantized_key_packed = &quantized_key_cache->packed;
    attention_cache.quantized_key_scales = &quantized_key_cache->scales;
    attention_cache.quantized_key_biases = &quantized_key_cache->biases;
    attention_cache.quantized_value_packed = &quantized_value_cache->packed;
    attention_cache.quantized_value_scales = &quantized_value_cache->scales;
    attention_cache.quantized_value_biases = &quantized_value_cache->biases;
    attention_cache.quantized_group_size =
        session.attention_cache_quantization_group_size;
    attention_cache.quantized_bits =
        session.attention_cache_quantization_bits;
  } else {
    if (!key_cache.has_value() || !value_cache.has_value()) {
      throw std::runtime_error("Qwen3.5 dense attention cache missing");
    }
    attention_cache.dense_keys = &*key_cache;
    attention_cache.dense_values = &*value_cache;
  }
  auto attention = edge_cmlx::blocks::transformer_attention(
      projection,
      attention_cache,
      attention_weights,
      attention_config,
      edge_cmlx::blocks::TransformerAttentionRuntime{
          token_count > 1,
          compute_dsr_scores,
          !session.fused_attention_check_logged,
          layer_index},
      stream);
  if (attention.fused_check_logged) {
    session.fused_attention_check_logged = true;
  }
  if (dsr_policy != nullptr) {
    session.attention_active_lengths[layer_index] = local_new_offset;
    if (compute_dsr_scores) {
      if (!attention.dsr_scores.has_value()) {
        throw std::runtime_error("qwen35 DSR attention scores missing");
      }
      const array& raw_scores = *attention.dsr_scores;
      auto score_importance = abs(
          reshape(raw_scores, Shape{1, kv_heads, local_new_offset}, stream),
          stream);

      auto score_state_item = session.attention_score_states.find(layer_index);
      const bool had_score_state =
          score_state_item != session.attention_score_states.end();
      const int score_capacity = std::max(
          cache_capacity,
          had_score_state
              ? static_cast<int>(score_state_item->second.shape(2))
              : 0);
      array score_state = [&]() -> array {
        if (!had_score_state ||
            score_state_item->second.shape(2) < local_new_offset) {
          auto grown = zeros(
              Shape{1, kv_heads, score_capacity},
              float32,
              stream);
          if (had_score_state) {
            const auto& existing = score_state_item->second;
            const int copy_length = std::min(
                static_cast<int>(existing.shape(2)),
                local_position_offset);
            if (copy_length > 0) {
              grown = slice_update(
                  grown,
                  slice(
                      existing,
                      Shape{0, 0, 0},
                      Shape{1, kv_heads, copy_length},
                      stream),
                  Shape{0, 0, 0},
                  Shape{1, kv_heads, copy_length},
                  stream);
            }
          }
          return grown;
        }
        return score_state_item->second;
      }();

      auto previous_scores = slice(
          score_state,
          Shape{0, 0, 0},
          Shape{1, kv_heads, local_new_offset},
          stream);
      auto updated_scores = had_score_state
          ? add(
                multiply(previous_scores, array(dsr_policy->score_decay, float32), stream),
                multiply(
                    score_importance,
                    array(1.0f - dsr_policy->score_decay, float32),
                    stream),
                stream)
          : score_importance;
      score_state = slice_update(
          score_state,
          updated_scores,
          Shape{0, 0, 0},
          Shape{1, kv_heads, local_new_offset},
          stream);
      auto stored_scores = stop_gradient(score_state, stream);
      session.attention_score_states.insert_or_assign(layer_index, stored_scores);
      qwen35_eval_output_push(
          eval_outputs,
          stored_scores,
          eval_inventory,
          "dsr_score_state");

      int& tokens_since_eviction =
          session.attention_dsr_tokens_since_eviction[layer_index];
      tokens_since_eviction += 1;
      if (tokens_since_eviction >= dsr_policy->eviction_interval) {
        tokens_since_eviction = 0;
        if (local_new_offset > dsr_policy->max_size) {
          const int sink_count =
              std::min(dsr_policy->sink_size, local_new_offset);
          const int recent_start = std::max(
              sink_count,
              local_new_offset - dsr_policy->recent_budget);
          const int middle_start = sink_count;
          const int middle_end = recent_start;
          const int middle_len = middle_end - middle_start;
          const int keep_k =
              std::min(dsr_policy->heavy_budget, middle_len);
          if (middle_len > 0 && keep_k > 0 && keep_k < middle_len) {
            auto middle_scores = slice(
                updated_scores,
                Shape{0, 0, middle_start},
                Shape{1, kv_heads, middle_end},
                stream);
            auto avg_middle_scores = mean(middle_scores, 1, false, stream);
            auto partitioned =
                argpartition(avg_middle_scores, -keep_k, -1, stream);
            auto top_indices = slice(
                partitioned,
                Shape{0, middle_len - keep_k},
                Shape{1, middle_len},
                stream);
            top_indices = reshape(
                astype(
                    add(
                        sort(top_indices, -1, stream),
                        array(middle_start, top_indices.dtype()),
                        stream),
                    int32,
                    stream),
                Shape{keep_k},
                stream);
            std::vector<array> keep_parts;
            if (sink_count > 0) {
              keep_parts.push_back(arange(0, sink_count, int32, stream));
            }
            keep_parts.push_back(top_indices);
            if (recent_start < local_new_offset) {
              keep_parts.push_back(
                  arange(recent_start, local_new_offset, int32, stream));
            }
            auto keep_indices = concatenate(keep_parts, 0, stream);
            const int keep_count =
                sink_count + keep_k + (local_new_offset - recent_start);

            auto compacted_scores = take(
                slice(
                    score_state,
                    Shape{0, 0, 0},
                    Shape{1, kv_heads, local_new_offset},
                    stream),
                keep_indices,
                2,
                stream);

            auto next_scores = slice_update(
                zeros(Shape{1, kv_heads, cache_capacity}, float32, stream),
                compacted_scores,
                Shape{0, 0, 0},
                Shape{1, kv_heads, keep_count},
                stream);
            auto stored_next_scores = stop_gradient(next_scores, stream);
            if (use_quantized_attention_cache) {
              auto next_quantized_keys = qwen35_compact_quantized_attention_cache(
                  *quantized_key_cache,
                  keep_indices,
                  keep_count,
                  cache_capacity,
                  stream);
              auto next_quantized_values = qwen35_compact_quantized_attention_cache(
                  *quantized_value_cache,
                  keep_indices,
                  keep_count,
                  cache_capacity,
                  stream);
              auto stored_key_packed =
                  stop_gradient(next_quantized_keys.packed, stream);
              auto stored_key_scales =
                  stop_gradient(next_quantized_keys.scales, stream);
              auto stored_key_biases =
                  stop_gradient(next_quantized_keys.biases, stream);
              auto stored_value_packed =
                  stop_gradient(next_quantized_values.packed, stream);
              auto stored_value_scales =
                  stop_gradient(next_quantized_values.scales, stream);
              auto stored_value_biases =
                  stop_gradient(next_quantized_values.biases, stream);
              session.attention_quantized_key_states.insert_or_assign(
                  layer_index,
                  EdgeCmlxQuantizedArray{
                      stored_key_packed,
                      stored_key_scales,
                      stored_key_biases,
                      next_quantized_keys.group_size,
                      next_quantized_keys.bits});
              session.attention_quantized_value_states.insert_or_assign(
                  layer_index,
                  EdgeCmlxQuantizedArray{
                      stored_value_packed,
                      stored_value_scales,
                      stored_value_biases,
                      next_quantized_values.group_size,
                      next_quantized_values.bits});
              qwen35_eval_output_push(
                  eval_outputs,
                  stored_key_packed,
                  eval_inventory,
                  "dsr_compacted_quant_key_packed");
              qwen35_eval_output_push(
                  eval_outputs,
                  stored_key_scales,
                  eval_inventory,
                  "dsr_compacted_quant_key_scales");
              qwen35_eval_output_push(
                  eval_outputs,
                  stored_key_biases,
                  eval_inventory,
                  "dsr_compacted_quant_key_biases");
              qwen35_eval_output_push(
                  eval_outputs,
                  stored_value_packed,
                  eval_inventory,
                  "dsr_compacted_quant_value_packed");
              qwen35_eval_output_push(
                  eval_outputs,
                  stored_value_scales,
                  eval_inventory,
                  "dsr_compacted_quant_value_scales");
              qwen35_eval_output_push(
                  eval_outputs,
                  stored_value_biases,
                  eval_inventory,
                  "dsr_compacted_quant_value_biases");
            } else {
              auto compacted_keys = take(*key_cache, keep_indices, 2, stream);
              auto compacted_values =
                  take(*value_cache, keep_indices, 2, stream);
              auto next_keys = slice_update(
                  zeros(
                      Shape{1, kv_heads, cache_capacity, head_dim},
                      key_cache->dtype(),
                      stream),
                  compacted_keys,
                  Shape{0, 0, 0, 0},
                  Shape{1, kv_heads, keep_count, head_dim},
                  stream);
              auto next_values = slice_update(
                  zeros(
                      Shape{1, kv_heads, cache_capacity, head_dim},
                      value_cache->dtype(),
                      stream),
                  compacted_values,
                  Shape{0, 0, 0, 0},
                  Shape{1, kv_heads, keep_count, head_dim},
                  stream);
              const bool skip_fa_stop_gradient =
                  qwen35_skip_fa_stop_gradient_enabled();
              auto stored_keys = skip_fa_stop_gradient
                  ? next_keys
                  : stop_gradient(next_keys, stream);
              auto stored_values = skip_fa_stop_gradient
                  ? next_values
                  : stop_gradient(next_values, stream);
              session.attention_key_states.insert_or_assign(
                  layer_index, stored_keys);
              session.attention_value_states.insert_or_assign(
                  layer_index, stored_values);
              qwen35_eval_output_push(
                  eval_outputs,
                  stored_keys,
                  eval_inventory,
                  "dsr_compacted_dense_key_state");
              qwen35_eval_output_push(
                  eval_outputs,
                  stored_values,
                  eval_inventory,
                  "dsr_compacted_dense_value_state");
            }
            session.attention_score_states.insert_or_assign(
                layer_index, stored_next_scores);
            session.attention_active_lengths[layer_index] = keep_count;
            qwen35_eval_output_push(
                eval_outputs,
                stored_next_scores,
                eval_inventory,
                "dsr_compacted_score_state");
          }
        }
      }
    }

    if (!compute_dsr_scores &&
        qwen35_dsr_prefill_eviction_enabled() &&
        token_count > 1 &&
        local_new_offset > dsr_policy->max_size &&
        key_cache.has_value() &&
        value_cache.has_value()) {
      const int sink_count =
          std::min(dsr_policy->sink_size, local_new_offset);
      const int recent_start = std::max(
          sink_count,
          local_new_offset - dsr_policy->recent_budget);
      const int recent_count = local_new_offset - recent_start;
      const int middle_start = sink_count;
      const int middle_end = recent_start;
      const int fallback_budget =
          std::max(0, dsr_policy->max_size - sink_count - recent_count);
      const int middle_keep_start =
          std::max(middle_start, middle_end - fallback_budget);
      const int middle_keep_count =
          std::max(0, middle_end - middle_keep_start);
      const int keep_count = sink_count + middle_keep_count + recent_count;
      if (keep_count > 0 && keep_count < local_new_offset) {
        std::vector<array> keep_parts;
        if (sink_count > 0) {
          keep_parts.push_back(arange(0, sink_count, int32, stream));
        }
        if (middle_keep_count > 0) {
          keep_parts.push_back(
              arange(middle_keep_start, middle_end, int32, stream));
        }
        if (recent_count > 0) {
          keep_parts.push_back(
              arange(recent_start, local_new_offset, int32, stream));
        }
        auto keep_indices = concatenate(keep_parts, 0, stream);
        auto compacted_keys = take(*key_cache, keep_indices, 2, stream);
        auto compacted_values =
            take(*value_cache, keep_indices, 2, stream);
        auto next_keys = slice_update(
            zeros(
                Shape{1, kv_heads, cache_capacity, head_dim},
                key_cache->dtype(),
                stream),
            compacted_keys,
            Shape{0, 0, 0, 0},
            Shape{1, kv_heads, keep_count, head_dim},
            stream);
        auto next_values = slice_update(
            zeros(
                Shape{1, kv_heads, cache_capacity, head_dim},
                value_cache->dtype(),
                stream),
            compacted_values,
            Shape{0, 0, 0, 0},
            Shape{1, kv_heads, keep_count, head_dim},
            stream);
        const bool skip_fa_stop_gradient =
            qwen35_skip_fa_stop_gradient_enabled();
        auto stored_keys = skip_fa_stop_gradient
            ? next_keys
            : stop_gradient(next_keys, stream);
        auto stored_values = skip_fa_stop_gradient
            ? next_values
            : stop_gradient(next_values, stream);
        session.attention_key_states.insert_or_assign(
            layer_index,
            stored_keys);
        session.attention_value_states.insert_or_assign(
            layer_index,
            stored_values);
        qwen35_eval_output_push(
            eval_outputs,
            stored_keys,
            eval_inventory,
            "dsr_prefill_compacted_dense_key_state");
        qwen35_eval_output_push(
            eval_outputs,
            stored_values,
            eval_inventory,
            "dsr_prefill_compacted_dense_value_state");

        auto score_state_item =
            session.attention_score_states.find(layer_index);
        if (score_state_item != session.attention_score_states.end()) {
          auto compacted_scores = take(
              slice(
                  score_state_item->second,
                  Shape{0, 0, 0},
                  Shape{1, kv_heads, local_new_offset},
                  stream),
              keep_indices,
              2,
              stream);
          auto next_scores = slice_update(
              zeros(Shape{1, kv_heads, cache_capacity}, float32, stream),
              compacted_scores,
              Shape{0, 0, 0},
              Shape{1, kv_heads, keep_count},
              stream);
          auto stored_scores = stop_gradient(next_scores, stream);
          session.attention_score_states.insert_or_assign(
              layer_index,
              stored_scores);
          qwen35_eval_output_push(
              eval_outputs,
              stored_scores,
              eval_inventory,
              "dsr_prefill_compacted_score_state");
        }
        session.attention_active_lengths[layer_index] = keep_count;
        session.attention_dsr_tokens_since_eviction[layer_index] = 0;
      }
    }
  }
  return attention.output;
}

int qwen35_linearize_attention_cache_map(
    std::unordered_map<int, mlx::core::array>& states,
    int cached_length,
    int base_index,
    mlx::core::StreamOrDevice stream,
    std::vector<mlx::core::array>* materialized_states = nullptr) {
  using namespace mlx::core;

  if (base_index <= 0 || cached_length <= 0) {
    return 0;
  }
  int linearized_state_count = 0;
  for (auto& item : states) {
    auto& state = item.second;
    if (state.ndim() != 4) {
      continue;
    }
    const int active_length =
        std::min(static_cast<int>(state.shape(2)), cached_length);
    if (active_length <= 0 || base_index >= active_length) {
      continue;
    }
    auto tail = slice(
        state,
        Shape{0, 0, base_index, 0},
        Shape{state.shape(0), state.shape(1), active_length, state.shape(3)},
        stream);
    auto head = slice(
        state,
        Shape{0, 0, 0, 0},
        Shape{state.shape(0), state.shape(1), base_index, state.shape(3)},
        stream);
    state = stop_gradient(concatenate({tail, head}, 2, stream), stream);
    if (materialized_states != nullptr) {
      materialized_states->push_back(state);
    }
    linearized_state_count += 1;
  }
  return linearized_state_count;
}

int qwen35_trim_attention_cache_map(
    std::unordered_map<int, mlx::core::array>& states,
    int cached_length,
    int trim_count,
    mlx::core::StreamOrDevice stream,
    std::vector<mlx::core::array>* materialized_states = nullptr) {
  using namespace mlx::core;

  if (trim_count <= 0) {
    return 0;
  }
  int trimmed_state_count = 0;
  for (auto& item : states) {
    auto& state = item.second;
    if (state.ndim() != 4 || state.shape(2) <= trim_count) {
      continue;
    }
    const int active_end =
        std::min(static_cast<int>(state.shape(2)), cached_length);
    if (active_end <= trim_count) {
      continue;
    }
    state = stop_gradient(
        slice(
            state,
            Shape{0, 0, trim_count, 0},
            Shape{state.shape(0), state.shape(1), active_end, state.shape(3)},
            stream),
        stream);
    if (materialized_states != nullptr) {
      materialized_states->push_back(state);
    }
    trimmed_state_count += 1;
  }
  return trimmed_state_count;
}

void qwen35_trim_attention_cache_if_needed(
    EdgeCmlxQwen35Session& qwen_session,
    mlx::core::StreamOrDevice stream) {
  if (!qwen_session.attention_dsr_policies.empty()) {
    return;
  }
  const int limit = qwen_session.attention_cache_limit;
  if (limit <= 0) {
    return;
  }
  const int cached_length =
      qwen_session.decoded_token_count -
      qwen_session.attention_cache_base_position;
  if (cached_length <= limit) {
    return;
  }
  const int trim_count = cached_length - limit;
  if (qwen_session.attention_cache_base_index > 0) {
    std::vector<mlx::core::array> materialized_states;
    qwen35_linearize_attention_cache_map(
        qwen_session.attention_key_states,
        cached_length,
        qwen_session.attention_cache_base_index,
        stream,
        &materialized_states);
    qwen35_linearize_attention_cache_map(
        qwen_session.attention_value_states,
        cached_length,
        qwen_session.attention_cache_base_index,
        stream,
        &materialized_states);
    qwen_session.attention_cache_base_index = 0;
    if (!materialized_states.empty()) {
      mlx::core::eval(std::move(materialized_states));
      mlx::core::clear_cache();
    }
  }
  qwen35_trim_attention_cache_map(
      qwen_session.attention_key_states,
      cached_length,
      trim_count,
      stream);
  qwen35_trim_attention_cache_map(
      qwen_session.attention_value_states,
      cached_length,
      trim_count,
      stream);
  qwen_session.attention_cache_base_position += trim_count;
  qwen_session.attention_cache_base_index = 0;
}

void qwen35_prepare_attention_cache_for_append(
    EdgeCmlxQwen35Session& qwen_session,
    int incoming_token_count,
    mlx::core::StreamOrDevice stream) {
  if (!qwen_session.attention_dsr_policies.empty()) {
    return;
  }
  const int limit = qwen_session.attention_cache_limit;
  if (limit <= 0 || incoming_token_count <= 0) {
    return;
  }
  const int cached_length =
      qwen_session.decoded_token_count -
      qwen_session.attention_cache_base_position;
  if (cached_length <= 0) {
    return;
  }
  if (incoming_token_count == 1 && cached_length >= limit) {
    const int trim_count = cached_length - (limit - 1);
    qwen_session.attention_cache_base_position += trim_count;
    qwen_session.attention_cache_base_index =
        (qwen_session.attention_cache_base_index + trim_count) % limit;
    return;
  }
  std::vector<mlx::core::array> materialized_states;
  if (incoming_token_count > 1 && qwen_session.attention_cache_base_index > 0) {
    qwen35_linearize_attention_cache_map(
        qwen_session.attention_key_states,
        cached_length,
        qwen_session.attention_cache_base_index,
        stream,
        &materialized_states);
    qwen35_linearize_attention_cache_map(
        qwen_session.attention_value_states,
        cached_length,
        qwen_session.attention_cache_base_index,
        stream,
        &materialized_states);
    qwen_session.attention_cache_base_index = 0;
  }
  if (incoming_token_count >= limit) {
    qwen_session.attention_key_states.clear();
    qwen_session.attention_value_states.clear();
    qwen_session.attention_cache_base_position =
        qwen_session.decoded_token_count;
    qwen_session.attention_cache_base_index = 0;
    mlx::core::clear_cache();
    return;
  }
  const int projected_length = cached_length + incoming_token_count;
  if (projected_length <= limit) {
    if (!materialized_states.empty()) {
      mlx::core::eval(std::move(materialized_states));
      mlx::core::clear_cache();
    }
    return;
  }
  const int overflow = projected_length - limit;
  const int trim_count =
      std::min(cached_length, std::max(overflow, kQwen35AttentionCacheStep));
  qwen35_trim_attention_cache_map(
      qwen_session.attention_key_states,
      cached_length,
      trim_count,
      stream,
      &materialized_states);
  qwen35_trim_attention_cache_map(
      qwen_session.attention_value_states,
      cached_length,
      trim_count,
      stream,
      &materialized_states);
  qwen_session.attention_cache_base_position += trim_count;
  qwen_session.attention_cache_base_index = 0;
  if (!materialized_states.empty()) {
    mlx::core::eval(std::move(materialized_states));
    mlx::core::clear_cache();
  }
}

void qwen35_preallocate_attention_cache(
    EdgeCmlxQwen35Session& qwen_session,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;

  const int limit = qwen_session.attention_cache_limit;
  if (limit <= 0 || qwen35_attention_cache_quantization_enabled(qwen_session)) {
    return;
  }
  std::vector<array> materialized_states;
  for (int layer = 0; layer < qwen_session.config.layer_count; ++layer) {
    if (qwen_session.layer_kinds[static_cast<size_t>(layer)] !=
        EdgeCmlxQwen35LayerKindFullAttention) {
      continue;
    }
    const int capacity = qwen35_attention_capacity_for_layer(
        qwen_session,
        layer,
        limit);
    auto key_state = stop_gradient(
        zeros(
            Shape{
                1,
                qwen_session.config.key_value_head_count,
                capacity,
                qwen_session.config.attention_head_dimension},
            float32,
            stream),
        stream);
    auto value_state = stop_gradient(
        zeros(
            Shape{
                1,
                qwen_session.config.key_value_head_count,
                capacity,
                qwen_session.config.attention_head_dimension},
            float32,
            stream),
        stream);
    qwen_session.attention_key_states.insert_or_assign(layer, key_state);
    qwen_session.attention_value_states.insert_or_assign(layer, value_state);
    materialized_states.push_back(std::move(key_state));
    materialized_states.push_back(std::move(value_state));
  }
  if (!materialized_states.empty()) {
    eval(std::move(materialized_states));
    mlx::core::clear_cache();
  }
}

void qwen35_reset_decode_cache(EdgeCmlxQwen35Session& qwen_session) {
  qwen_session.gdn_conv_states.clear();
  qwen_session.gdn_recurrent_states.clear();
  qwen_session.attention_key_states.clear();
  qwen_session.attention_value_states.clear();
  qwen_session.attention_quantized_key_states.clear();
  qwen_session.attention_quantized_value_states.clear();
  qwen_session.attention_score_states.clear();
  qwen_session.attention_active_lengths.clear();
  qwen_session.attention_dsr_tokens_since_eviction.clear();
  qwen_session.pending_token.reset();
  qwen_session.pending_sample_diagnostics.clear();
  qwen_session.emitted_sample_diagnostics.clear();
  qwen_session.repetition_context_tokens.clear();
  qwen_session.presence_context_tokens.clear();
  qwen_session.frequency_context_tokens.clear();
  qwen_session.eos_sampling_token_ids.clear();
  qwen_session.eos_sampling_suppressed = false;
  qwen_session.eos_sampling_logit_penalty = 0.0f;
  qwen_session.prefill_fp16_attention_materialized_pending_clear = false;
  qwen_session.attention_cache_base_position = 0;
  qwen_session.attention_cache_base_index = 0;
  qwen_session.decoded_token_count = 0;
  qwen35_preallocate_attention_cache(
      qwen_session,
      mlx::core::Device{mlx::core::Device::gpu});
}

int matmul_f32(
    const float* lhs,
    int lhs_rows,
    int lhs_cols,
    const float* rhs,
    int rhs_rows,
    int rhs_cols,
    float* output,
    size_t output_count,
    mlx::core::Device device) {
  edge_cmlx_error.clear();

  if (lhs == nullptr || rhs == nullptr || output == nullptr) {
    return set_error("edge_cmlx_matmul_f32 received a null buffer");
  }
  if (lhs_rows <= 0 || lhs_cols <= 0 || rhs_rows <= 0 || rhs_cols <= 0) {
    return set_error("edge_cmlx_matmul_f32 received a non-positive shape");
  }
  if (lhs_cols != rhs_rows) {
    return set_error("edge_cmlx_matmul_f32 shape mismatch");
  }

  const size_t expected_count =
      static_cast<size_t>(lhs_rows) * static_cast<size_t>(rhs_cols);
  if (output_count < expected_count) {
    return set_error("edge_cmlx_matmul_f32 output buffer is too small");
  }

  try {
    using namespace mlx::core;

    const Shape lhs_shape{
        static_cast<ShapeElem>(lhs_rows), static_cast<ShapeElem>(lhs_cols)};
    const Shape rhs_shape{
        static_cast<ShapeElem>(rhs_rows), static_cast<ShapeElem>(rhs_cols)};
    auto lhs_array = array(lhs, lhs_shape, float32);
    auto rhs_array = array(rhs, rhs_shape, float32);
    auto result = matmul(lhs_array, rhs_array, device);
    eval(result);

    const float* result_data = result.data<float>();
    std::copy(result_data, result_data + expected_count, output);
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error("edge_cmlx_matmul_f32 failed with an unknown error");
  }
}

int softmax_f32_gpu(
    const float* input,
    int rows,
    int columns,
    float* output,
    size_t output_count) {
  edge_cmlx_error.clear();

  if (input == nullptr || output == nullptr) {
    return set_error("edge_cmlx_softmax_f32_gpu received a null buffer");
  }
  if (rows <= 0 || columns <= 0) {
    return set_error("edge_cmlx_softmax_f32_gpu received a non-positive shape");
  }

  const size_t expected_count =
      static_cast<size_t>(rows) * static_cast<size_t>(columns);
  if (output_count < expected_count) {
    return set_error("edge_cmlx_softmax_f32_gpu output buffer is too small");
  }

  try {
    using namespace mlx::core;

    const Shape shape{
        static_cast<ShapeElem>(rows), static_cast<ShapeElem>(columns)};
    auto input_array = array(input, shape, float32);
    auto result = softmax(
        input_array, -1, true, Device{Device::gpu});
    eval(result);

    const float* result_data = result.data<float>();
    std::copy(result_data, result_data + expected_count, output);
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error("edge_cmlx_softmax_f32_gpu failed with an unknown error");
  }
}

mlx::core::array qwen35_sample_token_from_logits(
    const mlx::core::array& input_logits,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    uint64_t seed,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;

  auto logits = input_logits.ndim() == 1
      ? expand_dims(input_logits, 0, stream)
      : input_logits;
  const int vocab = static_cast<int>(logits.shape(-1));
  if (temperature <= 0.0f) {
    return argmax(logits, -1, false, stream);
  }

  const auto key = random::key(seed);
  if ((top_p > 0.0f && top_p < 1.0f) || min_p > 0.0f) {
    if (top_k > 0 && top_k < vocab && top_k <= 64 &&
        qwen35_fast_topp_topk_enabled()) {
      auto fused_top_k = qwen35_topk_last_row_arrays(logits, top_k, stream);
      if (fused_top_k.has_value()) {
        auto top_logits = fused_top_k->logits;
        auto top_indices = fused_top_k->indices;
        auto max_token_mask = equal(
            reshape(
                arange(0, top_k, int32, stream),
                Shape{1, static_cast<ShapeElem>(top_k)},
                stream),
            array(0, int32),
            stream);
        auto keep_mask = max_token_mask;
        if (top_p > 0.0f && top_p < 1.0f) {
          auto top_log_probs = subtract(
              top_logits,
              logsumexp(logits, -1, true, stream),
              stream);
          auto top_probs = exp(top_log_probs, stream);
          auto cumulative_probs = cumsum(top_probs, -1, false, true, stream);
          auto previous_cumulative = subtract(cumulative_probs, top_probs, stream);
          keep_mask = logical_or(
              max_token_mask,
              less(previous_cumulative, array(top_p, float32), stream),
              stream);
        }
        if (min_p > 0.0f) {
          auto max_logit = slice(
              top_logits,
              Shape{0, 0},
              Shape{top_logits.shape(0), 1},
              stream);
          auto min_p_mask = greater_equal(
              top_logits,
              add(
                  max_logit,
                  array(static_cast<float>(std::log(static_cast<double>(min_p))), float32),
                  stream),
              stream);
          keep_mask = logical_or(
              max_token_mask,
              logical_and(keep_mask, min_p_mask, stream),
              stream);
        }
        auto filtered_logits = where(
            keep_mask,
            top_logits,
            array(-std::numeric_limits<float>::infinity(), top_logits.dtype()),
            stream);
        auto sampled_top_index = random::categorical(
            divide(
                filtered_logits,
                array(temperature, top_logits.dtype()),
                stream),
            -1,
            std::optional<array>(key),
            stream);
        auto flattened_top_indices = reshape(
            top_indices,
            Shape{static_cast<ShapeElem>(top_indices.size())},
            stream);
        return take(flattened_top_indices, sampled_top_index, stream);
      }
    }
    auto probs = softmax(logits, -1, false, stream);
    auto sorted_indices = argsort(probs, -1, stream);
    auto sorted_probs = take_along_axis(probs, sorted_indices, -1, stream);
    auto filtered_probs = sorted_probs;
    auto zero_prob = array(0.0f, filtered_probs.dtype());
    if (top_p > 0.0f && top_p < 1.0f) {
      auto cumulative_probs = cumsum(sorted_probs, -1, false, true, stream);
      auto keep_mask = greater(
          cumulative_probs,
          subtract(array(1.0f, float32), array(top_p, float32), stream),
          stream);
      filtered_probs = where(
          keep_mask,
          filtered_probs,
          zero_prob,
          stream);
    }
    if (min_p > 0.0f) {
      auto max_probs = slice(
          sorted_probs,
          Shape{0, vocab - 1},
          Shape{sorted_probs.shape(0), vocab},
          stream);
      auto keep_mask = greater_equal(
          sorted_probs,
          multiply(max_probs, array(min_p, float32), stream),
          stream);
      filtered_probs = where(
          keep_mask,
          filtered_probs,
          zero_prob,
          stream);
    }

    auto sorted_positions = reshape(
        arange(0, vocab, int32, stream),
        Shape{1, static_cast<ShapeElem>(vocab)},
        stream);
    if (top_k > 0 && top_k < vocab) {
      auto top_k_mask = greater_equal(
          sorted_positions,
          array(vocab - top_k, int32),
          stream);
      filtered_probs = where(
          top_k_mask,
          filtered_probs,
          zero_prob,
          stream);
    }

    auto max_token_mask = equal(
        sorted_positions,
        array(vocab - 1, int32),
        stream);
    filtered_probs = where(
        max_token_mask,
        sorted_probs,
        filtered_probs,
        stream);

    auto sampled_sorted_index = random::categorical(
        divide(
            log(filtered_probs, stream),
            array(temperature, float32),
            stream),
        -1,
        std::optional<array>(key),
        stream);
    auto flattened_sorted_indices = reshape(
        sorted_indices,
        Shape{static_cast<ShapeElem>(sorted_indices.size())},
        stream);
    return take(flattened_sorted_indices, sampled_sorted_index, stream);
  }

  if (top_k > 0 && top_k < vocab) {
    auto fused_top_k = qwen35_topk_last_row_arrays(logits, top_k, stream);
    auto top_indices = [&]() -> array {
      if (fused_top_k.has_value()) {
        return fused_top_k->indices;
      }
      const int kth = vocab - top_k;
      return slice(
          argpartition(logits, kth, -1, stream),
          Shape{0, kth},
          Shape{1, vocab},
          stream);
    }();
    auto top_logits = fused_top_k.has_value()
        ? fused_top_k->logits
        : take_along_axis(logits, top_indices, -1, stream);
    auto sampled_top_index = random::categorical(
        divide(
            top_logits,
            array(temperature, top_logits.dtype()),
            stream),
        -1,
        std::optional<array>(key),
        stream);
    auto flattened_top_indices = reshape(
        top_indices,
        Shape{static_cast<ShapeElem>(top_indices.size())},
        stream);
    return take(flattened_top_indices, sampled_top_index, stream);
  }

  return random::categorical(
      divide(logits, array(temperature, float32), stream),
      -1,
      std::optional<array>(key),
      stream);
}

int sample_token_f32_gpu(
    const float* logits,
    int vocabulary_size,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    uint64_t seed,
    int* output_token_id) {
  edge_cmlx_error.clear();
  if (logits == nullptr || output_token_id == nullptr) {
    return set_error("edge_cmlx_sample_token_f32_gpu received a null pointer");
  }
  if (vocabulary_size <= 0) {
    return set_error(
        "edge_cmlx_sample_token_f32_gpu received a non-positive vocabulary size");
  }
  if (!std::isfinite(temperature) || temperature < 0.0f) {
    return set_error(
        "edge_cmlx_sample_token_f32_gpu received an invalid temperature");
  }
  if (top_k < 0) {
    return set_error("edge_cmlx_sample_token_f32_gpu received an invalid top_k");
  }
  if (!(top_p > 0.0f && top_p <= 1.0f) || !std::isfinite(top_p)) {
    return set_error("edge_cmlx_sample_token_f32_gpu received an invalid top_p");
  }
  if (min_p < 0.0f || !std::isfinite(min_p)) {
    return set_error("edge_cmlx_sample_token_f32_gpu received an invalid min_p");
  }

  try {
    using namespace mlx::core;
    auto logits_array = array(
        logits,
        Shape{1, static_cast<ShapeElem>(vocabulary_size)},
        float32);
    auto token = qwen35_sample_token_from_logits(
        logits_array,
        temperature,
        top_k,
        top_p,
        min_p,
        seed,
        Device{Device::gpu});
    eval(token);
    *output_token_id = static_cast<int>(token.data<uint32_t>()[0]);
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_sample_token_f32_gpu failed with an unknown error");
  }
}

int default_metallib_available() {
  edge_cmlx_error.clear();

  try {
    auto& device =
        mlx::core::metal::device(mlx::core::Device{mlx::core::Device::gpu});
    device.get_kernel("rmsfloat32");
    return 1;
  } catch (const std::exception& error) {
    set_error(error.what());
    return 0;
  } catch (...) {
    set_error("edge_cmlx_default_metallib_available failed with an unknown error");
    return 0;
  }
}

int fast_rms_norm_f32_gpu(
    const float* input,
    int rows,
    int columns,
    const float* weight,
    float epsilon,
    float* output,
    size_t output_count) {
  edge_cmlx_error.clear();

  if (input == nullptr || weight == nullptr || output == nullptr) {
    return set_error("edge_cmlx_fast_rms_norm_f32_gpu received a null buffer");
  }
  if (rows <= 0 || columns <= 0 || epsilon < 0) {
    return set_error(
        "edge_cmlx_fast_rms_norm_f32_gpu received an invalid shape");
  }

  const size_t expected_count =
      static_cast<size_t>(rows) * static_cast<size_t>(columns);
  if (output_count < expected_count) {
    return set_error("edge_cmlx_fast_rms_norm_f32_gpu output buffer is too small");
  }

  try {
    using namespace mlx::core;

    const Shape input_shape{
        static_cast<ShapeElem>(rows), static_cast<ShapeElem>(columns)};
    const Shape weight_shape{static_cast<ShapeElem>(columns)};
    auto input_array = array(input, input_shape, float32);
    auto weight_array = array(weight, weight_shape, float32);
    auto result = fast::rms_norm(
        input_array,
        std::optional<array>(weight_array),
        epsilon,
        Device{Device::gpu});
    eval(result);

    const float* result_data = result.data<float>();
    std::copy(result_data, result_data + expected_count, output);
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_fast_rms_norm_f32_gpu failed with an unknown error");
  }
}

int rms_norm_scale_f32_gpu(
    const float* input,
    int rows,
    int columns,
    float epsilon,
    float scale,
    float* output,
    size_t output_count) {
  edge_cmlx_error.clear();

  if (input == nullptr || output == nullptr) {
    return set_error("edge_cmlx_rms_norm_scale_f32_gpu received a null buffer");
  }
  if (rows <= 0 || columns <= 0 || epsilon < 0 || !std::isfinite(scale)) {
    return set_error(
        "edge_cmlx_rms_norm_scale_f32_gpu received an invalid shape or scale");
  }

  const size_t expected_count =
      static_cast<size_t>(rows) * static_cast<size_t>(columns);
  if (output_count < expected_count) {
    return set_error(
        "edge_cmlx_rms_norm_scale_f32_gpu output buffer is too small");
  }

  try {
    using namespace mlx::core;

    const Shape input_shape{
        static_cast<ShapeElem>(rows), static_cast<ShapeElem>(columns)};
    auto input_array = array(input, input_shape, float32);
    auto result = fast::rms_norm_scale(
        input_array,
        epsilon,
        scale,
        Device{Device::gpu});
    eval(result);

    const float* result_data = result.data<float>();
    std::copy(result_data, result_data + expected_count, output);
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_rms_norm_scale_f32_gpu failed with an unknown error");
  }
}

int encode_fast_rms_norm_f32_mtl(
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
    size_t output_count) {
  edge_cmlx_error.clear();

  if (command_buffer == nullptr) {
    return set_error(
        "edge_cmlx_encode_fast_rms_norm_f32_mtl received a null command buffer");
  }
  if (rows <= 0 || columns <= 0 || epsilon < 0) {
    return set_error(
        "edge_cmlx_encode_fast_rms_norm_f32_mtl received an invalid shape");
  }

  const size_t expected_count =
      static_cast<size_t>(rows) * static_cast<size_t>(columns);
  if (output_count < expected_count) {
    return set_error(
        "edge_cmlx_encode_fast_rms_norm_f32_mtl output buffer is too small");
  }

  try {
    const size_t input_byte_count = expected_count * sizeof(float);
    const size_t weight_byte_count = static_cast<size_t>(columns) * sizeof(float);
    auto* cb = static_cast<MTL::CommandBuffer*>(command_buffer);
    auto* input = checked_mtl_buffer(
        input_buffer, input_offset, input_byte_count, "input");
    auto* weight = checked_mtl_buffer(
        weight_buffer, weight_offset, weight_byte_count, "weight");
    auto* output = checked_mtl_buffer(
        output_buffer, output_offset, input_byte_count, "output");

    auto& d = mlx::core::metal::device(
        mlx::core::Device{mlx::core::Device::gpu});
    constexpr uint32_t simd_size = 32;
    constexpr uint32_t n_reads = RMS_N_READS;
    constexpr uint32_t looped_limit = RMS_LOOPED_LIMIT;
    uint32_t axis_size = static_cast<uint32_t>(columns);
    uint32_t w_stride = 1;
    bool use_looped = axis_size > looped_limit;
    std::string kernel_name = use_looped ? "rms_loopedfloat32" : "rmsfloat32";
    auto* kernel = d.get_kernel(kernel_name);

    MTL::Size grid_dims;
    MTL::Size group_dims;
    if (!use_looped) {
      size_t threadgroup_needed = (axis_size + n_reads - 1) / n_reads;
      size_t simds_needed = (threadgroup_needed + simd_size - 1) / simd_size;
      size_t threadgroup_size = simd_size * simds_needed;
      if (threadgroup_size > kernel->maxTotalThreadsPerThreadgroup()) {
        use_looped = true;
        kernel_name = "rms_loopedfloat32";
        kernel = d.get_kernel(kernel_name);
      } else {
        grid_dims = MTL::Size(
            static_cast<size_t>(rows) * threadgroup_size, 1, 1);
        group_dims = MTL::Size(threadgroup_size, 1, 1);
      }
    }
    if (use_looped) {
      size_t threadgroup_size = kernel->maxTotalThreadsPerThreadgroup();
      grid_dims =
          MTL::Size(static_cast<size_t>(rows) * threadgroup_size, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    }

    auto* encoder = cb->computeCommandEncoder();
    if (encoder == nullptr) {
      return set_error(
          "edge_cmlx_encode_fast_rms_norm_f32_mtl could not create a compute encoder");
    }
    encoder->setComputePipelineState(kernel);
    encoder->setBuffer(input, input_offset, 0);
    encoder->setBuffer(weight, weight_offset, 1);
    encoder->setBuffer(output, output_offset, 2);
    encoder->setBytes(&epsilon, sizeof(float), 3);
    encoder->setBytes(&axis_size, sizeof(uint32_t), 4);
    encoder->setBytes(&w_stride, sizeof(uint32_t), 5);
    encoder->dispatchThreads(grid_dims, group_dims);
    encoder->endEncoding();
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_encode_fast_rms_norm_f32_mtl failed with an unknown error");
  }
}

int affine_quantized_matmul_f32_gpu(
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
    size_t output_count) {
  edge_cmlx_error.clear();

  if (lhs == nullptr || packed_weights == nullptr || scales == nullptr ||
      biases == nullptr || output == nullptr) {
    return set_error(
        "edge_cmlx_affine_quantized_matmul_f32_gpu received a null buffer");
  }
  if (lhs_rows <= 0 || lhs_cols <= 0 || packed_rows <= 0 ||
      packed_cols <= 0 || scale_rows <= 0 || scale_cols <= 0 ||
      group_size <= 0 || bits <= 0) {
    return set_error(
        "edge_cmlx_affine_quantized_matmul_f32_gpu received an invalid shape");
  }
  if (scale_rows != packed_rows) {
    return set_error(
        "edge_cmlx_affine_quantized_matmul_f32_gpu scale row mismatch");
  }

  const int expanded_packed_cols = packed_cols * 32 / bits;
  const int inner_dims = transpose ? expanded_packed_cols : packed_rows;
  const int output_cols = transpose ? packed_rows : expanded_packed_cols;
  if (lhs_cols != inner_dims) {
    return set_error(
        "edge_cmlx_affine_quantized_matmul_f32_gpu lhs/weight mismatch");
  }
  if (scale_cols * group_size != expanded_packed_cols) {
    return set_error(
        "edge_cmlx_affine_quantized_matmul_f32_gpu scale shape mismatch");
  }

  const size_t expected_count =
      static_cast<size_t>(lhs_rows) * static_cast<size_t>(output_cols);
  if (output_count < expected_count) {
    return set_error(
        "edge_cmlx_affine_quantized_matmul_f32_gpu output buffer is too small");
  }

  try {
    using namespace mlx::core;

    auto lhs_array = array(
        lhs,
        Shape{
            static_cast<ShapeElem>(lhs_rows),
            static_cast<ShapeElem>(lhs_cols)},
        float32);
    auto weight_array = array(
        packed_weights,
        Shape{
            static_cast<ShapeElem>(packed_rows),
            static_cast<ShapeElem>(packed_cols)},
        uint32);
    auto scale_array = array(
        scales,
        Shape{
            static_cast<ShapeElem>(scale_rows),
            static_cast<ShapeElem>(scale_cols)},
        float32);
    auto bias_array = array(
        biases,
        Shape{
            static_cast<ShapeElem>(scale_rows),
            static_cast<ShapeElem>(scale_cols)},
        float32);

    auto result = quantized_matmul(
        lhs_array,
        weight_array,
        scale_array,
        bias_array,
        transpose != 0,
        group_size,
        bits,
        "affine",
        Device{Device::gpu});
    eval(result);

    const float* result_data = result.data<float>();
    std::copy(result_data, result_data + expected_count, output);
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_affine_quantized_matmul_f32_gpu failed with an unknown error");
  }
}

int affine_quantized_matmul_f32_mtl(
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
    size_t output_count) {
  edge_cmlx_error.clear();

  if (lhs_rows <= 0 || lhs_cols <= 0 || packed_rows <= 0 ||
      packed_cols <= 0 || scale_rows <= 0 || scale_cols <= 0 ||
      group_size <= 0 || bits <= 0) {
    return set_error(
        "edge_cmlx_affine_quantized_matmul_f32_mtl received an invalid shape");
  }
  if (scale_rows != packed_rows) {
    return set_error(
        "edge_cmlx_affine_quantized_matmul_f32_mtl scale row mismatch");
  }

  const int expanded_packed_cols = packed_cols * 32 / bits;
  const int inner_dims = transpose ? expanded_packed_cols : packed_rows;
  const int output_cols = transpose ? packed_rows : expanded_packed_cols;
  if (lhs_cols != inner_dims) {
    return set_error(
        "edge_cmlx_affine_quantized_matmul_f32_mtl lhs/weight mismatch");
  }
  if (scale_cols * group_size != expanded_packed_cols) {
    return set_error(
        "edge_cmlx_affine_quantized_matmul_f32_mtl scale shape mismatch");
  }

  const size_t expected_count =
      static_cast<size_t>(lhs_rows) * static_cast<size_t>(output_cols);
  if (output_count < expected_count) {
    return set_error(
        "edge_cmlx_affine_quantized_matmul_f32_mtl output buffer is too small");
  }

  try {
    using namespace mlx::core;

    {
      auto lhs_array = array_from_mtl_buffer(
          lhs_buffer,
          Shape{
              static_cast<ShapeElem>(lhs_rows),
              static_cast<ShapeElem>(lhs_cols)},
          float32,
          "lhs");
      auto weight_array = array_from_mtl_buffer(
          packed_weights_buffer,
          Shape{
              static_cast<ShapeElem>(packed_rows),
              static_cast<ShapeElem>(packed_cols)},
          uint32,
          "packed weights");
      auto scale_array = array_from_mtl_buffer(
          scales_buffer,
          Shape{
              static_cast<ShapeElem>(scale_rows),
              static_cast<ShapeElem>(scale_cols)},
          float32,
          "scales");
      auto bias_array = array_from_mtl_buffer(
          biases_buffer,
          Shape{
              static_cast<ShapeElem>(scale_rows),
              static_cast<ShapeElem>(scale_cols)},
          float32,
          "biases");
      auto output_array = array_from_mtl_buffer(
          output_buffer,
          Shape{
              static_cast<ShapeElem>(lhs_rows),
              static_cast<ShapeElem>(output_cols)},
          float32,
          "output");

      auto result = quantized_matmul(
          lhs_array,
          weight_array,
          scale_array,
          bias_array,
          transpose != 0,
          group_size,
          bits,
          "affine",
          Device{Device::gpu});
      result.copy_shared_buffer(output_array);
      eval(result);
    }
    mlx::core::clear_cache();
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_affine_quantized_matmul_f32_mtl failed with an unknown error");
  }
}

int rms_norm_affine_quantized_matmul_f32_mtl(
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
    size_t output_count) {
  edge_cmlx_error.clear();

  if (lhs_rows <= 0 || lhs_cols <= 0 || packed_rows <= 0 ||
      packed_cols <= 0 || scale_rows <= 0 || scale_cols <= 0 ||
      group_size <= 0 || bits <= 0 || epsilon < 0) {
    return set_error(
        "edge_cmlx_rms_norm_affine_quantized_matmul_f32_mtl received an invalid shape");
  }
  if (scale_rows != packed_rows) {
    return set_error(
        "edge_cmlx_rms_norm_affine_quantized_matmul_f32_mtl scale row mismatch");
  }

  const int expanded_packed_cols = packed_cols * 32 / bits;
  const int inner_dims = transpose ? expanded_packed_cols : packed_rows;
  const int output_cols = transpose ? packed_rows : expanded_packed_cols;
  if (lhs_cols != inner_dims) {
    return set_error(
        "edge_cmlx_rms_norm_affine_quantized_matmul_f32_mtl lhs/weight mismatch");
  }
  if (scale_cols * group_size != expanded_packed_cols) {
    return set_error(
        "edge_cmlx_rms_norm_affine_quantized_matmul_f32_mtl scale shape mismatch");
  }

  const size_t expected_count =
      static_cast<size_t>(lhs_rows) * static_cast<size_t>(output_cols);
  if (output_count < expected_count) {
    return set_error(
        "edge_cmlx_rms_norm_affine_quantized_matmul_f32_mtl output buffer is too small");
  }

  try {
    using namespace mlx::core;

    {
      auto lhs_array = array_from_mtl_buffer(
          lhs_buffer,
          Shape{
              static_cast<ShapeElem>(lhs_rows),
              static_cast<ShapeElem>(lhs_cols)},
          float32,
          "lhs");
      auto norm_weight_array = array_from_mtl_buffer(
          norm_weight_buffer,
          Shape{static_cast<ShapeElem>(lhs_cols)},
          float32,
          "norm weight");
      auto weight_array = array_from_mtl_buffer(
          packed_weights_buffer,
          Shape{
              static_cast<ShapeElem>(packed_rows),
              static_cast<ShapeElem>(packed_cols)},
          uint32,
          "packed weights");
      auto scale_array = array_from_mtl_buffer(
          scales_buffer,
          Shape{
              static_cast<ShapeElem>(scale_rows),
              static_cast<ShapeElem>(scale_cols)},
          float32,
          "scales");
      auto bias_array = array_from_mtl_buffer(
          biases_buffer,
          Shape{
              static_cast<ShapeElem>(scale_rows),
              static_cast<ShapeElem>(scale_cols)},
          float32,
          "biases");
      auto output_array = array_from_mtl_buffer(
          output_buffer,
          Shape{
              static_cast<ShapeElem>(lhs_rows),
              static_cast<ShapeElem>(output_cols)},
          float32,
          "output");

      auto gpu_device = Device{Device::gpu};
      auto mean_square = mean(
          square(lhs_array, gpu_device),
          -1,
          true,
          gpu_device);
      auto inv_rms = rsqrt(
          add(mean_square, array(epsilon, float32), gpu_device),
          gpu_device);
      auto normalized = multiply(
          multiply(lhs_array, inv_rms, gpu_device),
          norm_weight_array,
          gpu_device);
      auto result = quantized_matmul(
          normalized,
          weight_array,
          scale_array,
          bias_array,
          transpose != 0,
          group_size,
          bits,
          "affine",
          gpu_device);
      result.copy_shared_buffer(output_array);
      eval(result);
    }
    mlx::core::clear_cache();
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_rms_norm_affine_quantized_matmul_f32_mtl failed with an unknown error");
  }
}

int encode_affine_qmm_t_f32_mtl(
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
    size_t output_count) {
  edge_cmlx_error.clear();

  if (command_buffer == nullptr) {
    return set_error(
        "edge_cmlx_encode_affine_qmm_t_f32_mtl received a null command buffer");
  }
  if (lhs_rows <= 0 || lhs_cols <= 0 || packed_rows <= 0 ||
      packed_cols <= 0 || scale_rows <= 0 || scale_cols <= 0 ||
      group_size <= 0 || bits <= 0) {
    return set_error(
        "edge_cmlx_encode_affine_qmm_t_f32_mtl received an invalid shape");
  }
  if (scale_rows != packed_rows) {
    return set_error(
        "edge_cmlx_encode_affine_qmm_t_f32_mtl scale row mismatch");
  }
  if (!(bits == 4 || bits == 6)) {
    return set_error(
        "edge_cmlx_encode_affine_qmm_t_f32_mtl supports only 4-bit and 6-bit weights");
  }
  if (!(group_size == 32 || group_size == 64 || group_size == 128)) {
    return set_error(
        "edge_cmlx_encode_affine_qmm_t_f32_mtl received an unsupported group size");
  }

  const int expanded_packed_cols = packed_cols * 32 / bits;
  if (lhs_cols != expanded_packed_cols) {
    return set_error(
        "edge_cmlx_encode_affine_qmm_t_f32_mtl lhs/weight mismatch");
  }
  if (scale_cols * group_size != expanded_packed_cols) {
    return set_error(
        "edge_cmlx_encode_affine_qmm_t_f32_mtl scale shape mismatch");
  }

  const size_t expected_count =
      static_cast<size_t>(lhs_rows) * static_cast<size_t>(packed_rows);
  if (output_count < expected_count) {
    return set_error(
        "edge_cmlx_encode_affine_qmm_t_f32_mtl output buffer is too small");
  }

  try {
    const size_t lhs_byte_count =
        static_cast<size_t>(lhs_rows) * static_cast<size_t>(lhs_cols) *
        sizeof(float);
    const size_t packed_byte_count =
        static_cast<size_t>(packed_rows) * static_cast<size_t>(packed_cols) *
        sizeof(uint32_t);
    const size_t scales_byte_count =
        static_cast<size_t>(scale_rows) * static_cast<size_t>(scale_cols) *
        sizeof(float);
    const size_t output_byte_count = expected_count * sizeof(float);

    auto* cb = static_cast<MTL::CommandBuffer*>(command_buffer);
    auto* lhs = checked_mtl_buffer(lhs_buffer, lhs_offset, lhs_byte_count, "lhs");
    auto* weights = checked_mtl_buffer(
        packed_weights_buffer,
        packed_weights_offset,
        packed_byte_count,
        "packed weights");
    auto* scales = checked_mtl_buffer(
        scales_buffer,
        scales_offset,
        scales_byte_count,
        "scales");
    auto* biases = checked_mtl_buffer(
        biases_buffer,
        biases_offset,
        scales_byte_count,
        "biases");
    auto* output = checked_mtl_buffer(
        output_buffer,
        output_offset,
        output_byte_count,
        "output");

    auto& d = mlx::core::metal::device(
        mlx::core::Device{mlx::core::Device::gpu});
    const int K = lhs_cols;
    const int N = packed_rows;
    const int M = lhs_rows;
    const bool aligned = (N % 32) == 0;
    const bool batched = false;
    const std::string mode = "affine";
    const std::string type_string = "float";
    std::string kname;
    kname.reserve(64);
    kname += mode + "_qmm_t_";
    kname += type_string;
    kname += "_gs_";
    kname += std::to_string(group_size);
    kname += "_b_";
    kname += std::to_string(bits);
    kname += aligned ? "_alN_true" : "_alN_false";
    kname += batched ? "_batch_1" : "_batch_0";
    auto* kernel = get_quantized_kernel_wrapped(
        d,
        kname,
        "qmm_t",
        mode,
        type_string,
        group_size,
        bits,
        aligned,
        batched);

    auto* encoder = cb->computeCommandEncoder();
    if (encoder == nullptr) {
      return set_error(
          "edge_cmlx_encode_affine_qmm_t_f32_mtl could not create a compute encoder");
    }
    encoder->setComputePipelineState(kernel);
    encoder->setBuffer(weights, packed_weights_offset, 0);
    encoder->setBuffer(scales, scales_offset, 1);
    encoder->setBuffer(biases, biases_offset, 2);
    encoder->setBuffer(lhs, lhs_offset, 3);
    encoder->setBuffer(output, output_offset, 4);
    encoder->setBytes(&K, sizeof(int), 5);
    encoder->setBytes(&N, sizeof(int), 6);
    encoder->setBytes(&M, sizeof(int), 7);

    constexpr int bm = 32;
    constexpr int bn = 32;
    MTL::Size group_dims(32, 2, 2);
    MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, 1);
    encoder->dispatchThreadgroups(grid_dims, group_dims);
    encoder->endEncoding();
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_encode_affine_qmm_t_f32_mtl failed with an unknown error");
  }
}

}

using namespace edge_cmlx::detail;

const char* edge_cmlx_vendor_version(void) {
  return "0.32.0";
}

int edge_cmlx_vendor_version_numeric(void) {
  return MLX_VERSION_NUMERIC;
}

int edge_cmlx_vendor_present(void) {
  return 1;
}

const char* edge_cmlx_last_error(void) {
  return edge_cmlx_error.empty() ? nullptr : edge_cmlx_error.c_str();
}

int edge_cmlx_default_metallib_available(void) {
  return default_metallib_available();
}

int edge_cmlx_run_state_boundary_probe(
    int steps,
    EdgeCmlxStateBoundaryProbeResult* result) {
  edge_cmlx_error.clear();
  if (result == nullptr) {
    return set_error(
        "edge_cmlx_run_state_boundary_probe received a null result pointer");
  }
  *result = EdgeCmlxStateBoundaryProbeResult{};
  if (steps <= 0) {
    return set_error(
        "edge_cmlx_run_state_boundary_probe received non-positive steps");
  }

  try {
    using namespace mlx::core;

    const auto stream = Device{Device::gpu};
    const auto run_probe = [&](bool async_schedule) {
      array state = zeros(Shape{4}, float32, stream);
      EdgeCmlxStateProbeSnapshot recurrent_after_schedule{};
      EdgeCmlxStateProbeSnapshot conv_after_schedule{};
      EdgeCmlxStateProbeSnapshot recurrent_stop_gradient_after_schedule{};
      EdgeCmlxStateProbeSnapshot conv_stop_gradient_after_schedule{};
      EdgeCmlxStateProbeSnapshot custom_recurrent_after_schedule{};
      EdgeCmlxStateProbeSnapshot custom_recurrent_stop_gradient_after_schedule{};
      array final_recurrent_raw = state;
      array final_conv_raw = state;
      array final_custom_recurrent_raw = state;

      for (int step = 0; step < steps; ++step) {
        auto delta = add(
            state,
            array(static_cast<float>(step + 1), float32),
            stream);
        auto conv_input = concatenate({state, delta}, 0, stream);
        auto conv_raw = slice(conv_input, Shape{4}, Shape{8}, stream);
        auto conv_stop_gradient = stop_gradient(conv_raw, stream);
        auto parts = split(conv_input, Shape{4}, 0, stream);
        auto recurrent_raw = parts[1];
        auto recurrent_stop_gradient = stop_gradient(recurrent_raw, stream);

        auto conv_work = multiply(
            conv_input,
            array(2.0f, float32),
            stream);
        auto conv_output = slice(conv_work, Shape{0}, Shape{4}, stream);
        const int custom_key_heads = 1;
        const int custom_value_heads = 1;
        const int custom_key_dim = 32;
        const int custom_value_dim = 4;
        auto custom_query = broadcast_to(
            reshape(sum(parts[0], stream), Shape{1, 1, 1, 1}, stream),
            Shape{1, 1, custom_key_heads, custom_key_dim},
            stream);
        auto custom_key = broadcast_to(
            reshape(sum(parts[1], stream), Shape{1, 1, 1, 1}, stream),
            Shape{1, 1, custom_key_heads, custom_key_dim},
            stream);
        auto custom_value = reshape(
            parts[1],
            Shape{1, 1, custom_value_heads, custom_value_dim},
            stream);
        auto custom_decay = full(
            Shape{1, custom_value_heads},
            0.875f,
            float32,
            stream);
        auto custom_beta = full(
            Shape{1, custom_value_heads},
            0.5f,
            float32,
            stream);
        auto custom_state = broadcast_to(
            reshape(state, Shape{custom_value_heads, custom_value_dim, 1}, stream),
            Shape{custom_value_heads, custom_value_dim, custom_key_dim},
            stream);
        auto custom_update = qwen35_gated_delta_update_array(
            custom_query,
            custom_key,
            custom_value,
            custom_decay,
            custom_beta,
            custom_state,
            custom_key_heads,
            custom_value_heads,
            custom_key_dim,
            custom_value_dim,
            stream);
        auto custom_output = custom_update.first;
        auto custom_recurrent_raw = custom_update.second;
        auto custom_recurrent_stop_gradient =
            stop_gradient(custom_recurrent_raw, stream);
        auto token = add(
            sum(add(parts[0], conv_output, stream), stream),
            sum(custom_output, stream),
            stream);

        if (async_schedule) {
          async_eval(token);
        } else {
          eval(token);
        }

        recurrent_after_schedule = state_probe_snapshot(recurrent_raw);
        conv_after_schedule = state_probe_snapshot(conv_raw);
        recurrent_stop_gradient_after_schedule =
            state_probe_snapshot(recurrent_stop_gradient);
        conv_stop_gradient_after_schedule =
            state_probe_snapshot(conv_stop_gradient);
        custom_recurrent_after_schedule =
            state_probe_snapshot(custom_recurrent_raw);
        custom_recurrent_stop_gradient_after_schedule =
            state_probe_snapshot(custom_recurrent_stop_gradient);

        state = add(recurrent_stop_gradient, conv_stop_gradient, stream);
        final_recurrent_raw = recurrent_raw;
        final_conv_raw = conv_raw;
        final_custom_recurrent_raw = custom_recurrent_raw;
      }

      eval(state);
      return std::make_tuple(
          recurrent_after_schedule,
          conv_after_schedule,
          recurrent_stop_gradient_after_schedule,
          conv_stop_gradient_after_schedule,
          custom_recurrent_after_schedule,
          custom_recurrent_stop_gradient_after_schedule,
          state_probe_snapshot(final_recurrent_raw),
          state_probe_snapshot(final_conv_raw),
          state_probe_snapshot(final_custom_recurrent_raw));
    };

    auto sync_result = run_probe(false);
    result->sync_steps = steps;
    result->sync_recurrent_after_token_eval = std::get<0>(sync_result);
    result->sync_conv_after_token_eval = std::get<1>(sync_result);
    result->sync_recurrent_stop_gradient_after_token_eval =
        std::get<2>(sync_result);
    result->sync_conv_stop_gradient_after_token_eval =
        std::get<3>(sync_result);
    result->sync_custom_recurrent_after_token_eval =
        std::get<4>(sync_result);
    result->sync_custom_recurrent_stop_gradient_after_token_eval =
        std::get<5>(sync_result);

    auto async_result = run_probe(true);
    result->async_steps = steps;
    result->async_recurrent_after_schedule = std::get<0>(async_result);
    result->async_conv_after_schedule = std::get<1>(async_result);
    result->async_recurrent_stop_gradient_after_schedule =
        std::get<2>(async_result);
    result->async_conv_stop_gradient_after_schedule =
        std::get<3>(async_result);
    result->async_custom_recurrent_after_schedule =
        std::get<4>(async_result);
    result->async_custom_recurrent_stop_gradient_after_schedule =
        std::get<5>(async_result);
    result->async_recurrent_after_final_eval = std::get<6>(async_result);
    result->async_conv_after_final_eval = std::get<7>(async_result);
    result->async_custom_recurrent_after_final_eval =
        std::get<8>(async_result);
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_run_state_boundary_probe failed with an unknown error");
  }
}

int edge_cmlx_run_cross_thread_stream_probe(void) {
  edge_cmlx_error.clear();

  try {
    using namespace mlx::core;

    const auto stream = Device{Device::gpu};
    auto state = zeros(Shape{4}, float32, stream);
    auto update = add(state, array(1.0f, float32), stream);
    auto token = sum(update, stream);
    async_eval(token);

    std::exception_ptr thread_error = nullptr;
    std::thread worker([update, &thread_error]() mutable {
      try {
        eval(update);
      } catch (...) {
        thread_error = std::current_exception();
      }
    });
    worker.join();
    if (thread_error != nullptr) {
      std::rethrow_exception(thread_error);
    }
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_run_cross_thread_stream_probe failed with an unknown error");
  }
}

int edge_cmlx_set_command_buffer_limits(int max_ops, int max_mb) {
  edge_cmlx_error.clear();
  if (max_ops <= 0 || max_mb <= 0) {
    return set_error(
        "edge_cmlx_set_command_buffer_limits received non-positive limits");
  }
  try {
    mlx::core::metal::device(mlx::core::Device::gpu)
        .set_max_ops_mb_per_buffer(max_ops, max_mb);
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_set_command_buffer_limits failed with an unknown error");
  }
}

int edge_cmlx_get_command_buffer_limits(int* max_ops, int* max_mb) {
  edge_cmlx_error.clear();
  if (max_ops == nullptr || max_mb == nullptr) {
    return set_error(
        "edge_cmlx_get_command_buffer_limits received a null pointer");
  }
  try {
    auto limits = mlx::core::metal::device(mlx::core::Device::gpu)
        .get_max_ops_mb_per_buffer();
    *max_ops = std::get<0>(limits);
    *max_mb = std::get<1>(limits);
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_get_command_buffer_limits failed with an unknown error");
  }
}

int edge_cmlx_matmul_f32_cpu(
    const float* lhs,
    int lhs_rows,
    int lhs_cols,
    const float* rhs,
    int rhs_rows,
    int rhs_cols,
    float* output,
    size_t output_count) {
  return matmul_f32(
      lhs,
      lhs_rows,
      lhs_cols,
      rhs,
      rhs_rows,
      rhs_cols,
      output,
      output_count,
      mlx::core::Device{mlx::core::Device::cpu});
}

int edge_cmlx_matmul_f32_gpu(
    const float* lhs,
    int lhs_rows,
    int lhs_cols,
    const float* rhs,
    int rhs_rows,
    int rhs_cols,
    float* output,
    size_t output_count) {
  return matmul_f32(
      lhs,
      lhs_rows,
      lhs_cols,
      rhs,
      rhs_rows,
      rhs_cols,
      output,
      output_count,
      mlx::core::Device{mlx::core::Device::gpu});
}

int edge_cmlx_softmax_f32_gpu(
    const float* input,
    int rows,
    int columns,
    float* output,
    size_t output_count) {
  return softmax_f32_gpu(input, rows, columns, output, output_count);
}

int edge_cmlx_sample_token_f32_gpu(
    const float* logits,
    int vocabulary_size,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    uint64_t seed,
    int* output_token_id) {
  return sample_token_f32_gpu(
      logits,
      vocabulary_size,
      temperature,
      top_k,
      top_p,
      min_p,
      seed,
      output_token_id);
}

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
    size_t output_count) {
  return affine_quantized_matmul_f32_gpu(
      lhs,
      lhs_rows,
      lhs_cols,
      packed_weights,
      packed_rows,
      packed_cols,
      scales,
      scale_rows,
      scale_cols,
      biases,
      group_size,
      bits,
      transpose,
      output,
      output_count);
}

int edge_cmlx_fast_rms_norm_f32_gpu(
    const float* input,
    int rows,
    int columns,
    const float* weight,
    float epsilon,
    float* output,
    size_t output_count) {
  return fast_rms_norm_f32_gpu(
      input,
      rows,
      columns,
      weight,
      epsilon,
      output,
      output_count);
}

int edge_cmlx_rms_norm_scale_f32_gpu(
    const float* input,
    int rows,
    int columns,
    float epsilon,
    float scale,
    float* output,
    size_t output_count) {
  return rms_norm_scale_f32_gpu(
      input,
      rows,
      columns,
      epsilon,
      scale,
      output,
      output_count);
}

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
    size_t output_count) {
  return encode_fast_rms_norm_f32_mtl(
      command_buffer,
      input_buffer,
      input_offset,
      rows,
      columns,
      weight_buffer,
      weight_offset,
      epsilon,
      output_buffer,
      output_offset,
      output_count);
}

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
    size_t output_count) {
  return affine_quantized_matmul_f32_mtl(
      lhs_buffer,
      lhs_rows,
      lhs_cols,
      packed_weights_buffer,
      packed_rows,
      packed_cols,
      scales_buffer,
      scale_rows,
      scale_cols,
      biases_buffer,
      group_size,
      bits,
      transpose,
      output_buffer,
      output_count);
}

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
    size_t output_count) {
  return rms_norm_affine_quantized_matmul_f32_mtl(
      lhs_buffer,
      lhs_rows,
      lhs_cols,
      norm_weight_buffer,
      epsilon,
      packed_weights_buffer,
      packed_rows,
      packed_cols,
      scales_buffer,
      scale_rows,
      scale_cols,
      biases_buffer,
      group_size,
      bits,
      transpose,
      output_buffer,
      output_count);
}

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
    size_t output_count) {
  return encode_affine_qmm_t_f32_mtl(
      command_buffer,
      lhs_buffer,
      lhs_offset,
      lhs_rows,
      lhs_cols,
      packed_weights_buffer,
      packed_weights_offset,
      packed_rows,
      packed_cols,
      scales_buffer,
      scales_offset,
      scale_rows,
      scale_cols,
      biases_buffer,
      biases_offset,
      group_size,
      bits,
      output_buffer,
      output_offset,
      output_count);
}

void* edge_cmlx_qwen35_session_create(const EdgeCmlxQwen35Config* config) {
  edge_cmlx_error.clear();
  if (config == nullptr) {
    set_error("edge_cmlx_qwen35_session_create received a null config");
    return nullptr;
  }
  try {
    validate_qwen35_config(*config);
    return new EdgeCmlxQwen35Session(*config);
  } catch (const std::exception& error) {
    set_error(error.what());
    return nullptr;
  } catch (...) {
    set_error("edge_cmlx_qwen35_session_create failed with an unknown error");
    return nullptr;
  }
}

void edge_cmlx_qwen35_session_destroy(void* session) {
  delete static_cast<EdgeCmlxQwen35Session*>(session);
  mlx::core::clear_cache();
}

int edge_cmlx_qwen35_session_set_layer_kind(
    void* session,
    int layer_index,
    int layer_kind) {
  edge_cmlx_error.clear();
  try {
    auto* qwen_session = checked_qwen35_session(session);
    if (layer_index < 0 || layer_index >= qwen_session->config.layer_count) {
      return set_error(
          "edge_cmlx_qwen35_session_set_layer_kind received an invalid layer index");
    }
    if (layer_kind != EdgeCmlxQwen35LayerKindFullAttention &&
        layer_kind != EdgeCmlxQwen35LayerKindGDN) {
      return set_error(
          "edge_cmlx_qwen35_session_set_layer_kind received an invalid layer kind");
    }
    qwen_session->layer_kinds[static_cast<size_t>(layer_index)] = layer_kind;
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_set_layer_kind failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_clear_dsr_policies(void* session) {
  edge_cmlx_error.clear();
  try {
    auto* qwen_session = checked_qwen35_session(session);
    qwen_session->attention_dsr_policies.clear();
    qwen_session->attention_score_states.clear();
    qwen_session->attention_active_lengths.clear();
    qwen_session->attention_dsr_tokens_since_eviction.clear();
    qwen_session->attention_quantized_key_states.clear();
    qwen_session->attention_quantized_value_states.clear();
    qwen_session->prefill_fp16_attention_materialized_pending_clear = false;
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_clear_dsr_policies failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_set_dsr_policy(
    void* session,
    int layer_index,
    const EdgeCmlxQwen35DSRPolicy* policy) {
  edge_cmlx_error.clear();
  if (policy == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_set_dsr_policy received a null policy");
  }
  try {
    auto* qwen_session = checked_qwen35_session(session);
    if (layer_index < 0 || layer_index >= qwen_session->config.layer_count) {
      return set_error(
          "edge_cmlx_qwen35_session_set_dsr_policy received an invalid layer index");
    }
    if (qwen_session->layer_kinds[static_cast<size_t>(layer_index)] !=
        EdgeCmlxQwen35LayerKindFullAttention) {
      return set_error(
          "edge_cmlx_qwen35_session_set_dsr_policy received a non-attention layer");
    }
    validate_qwen35_dsr_policy(*policy);
    qwen_session->attention_dsr_policies.insert_or_assign(layer_index, *policy);
    qwen_session->attention_key_states.erase(layer_index);
    qwen_session->attention_value_states.erase(layer_index);
    qwen_session->attention_quantized_key_states.erase(layer_index);
    qwen_session->attention_quantized_value_states.erase(layer_index);
    qwen_session->attention_score_states.erase(layer_index);
    qwen_session->attention_active_lengths.erase(layer_index);
    qwen_session->attention_dsr_tokens_since_eviction.erase(layer_index);
    qwen_session->prefill_fp16_attention_materialized_pending_clear = false;
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_set_dsr_policy failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_update_dsr_policy_fields(
    void* session,
    int layer_index,
    const EdgeCmlxQwen35DSRPolicy* policy) {
  edge_cmlx_error.clear();
  if (policy == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_update_dsr_policy_fields received a null policy");
  }
  try {
    auto* qwen_session = checked_qwen35_session(session);
    if (layer_index < 0 || layer_index >= qwen_session->config.layer_count) {
      return set_error(
          "edge_cmlx_qwen35_session_update_dsr_policy_fields received an invalid layer index");
    }
    if (qwen_session->layer_kinds[static_cast<size_t>(layer_index)] !=
        EdgeCmlxQwen35LayerKindFullAttention) {
      return set_error(
          "edge_cmlx_qwen35_session_update_dsr_policy_fields received a non-attention layer");
    }
    validate_qwen35_dsr_policy(*policy);
    qwen_session->attention_dsr_policies.insert_or_assign(layer_index, *policy);
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_update_dsr_policy_fields failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_set_attention_cache_quantization(
    void* session,
    int group_size,
    int bits) {
  edge_cmlx_error.clear();
  if ((group_size == 0 && bits != 0) || (group_size != 0 && bits == 0)) {
    return set_error(
        "edge_cmlx_qwen35_session_set_attention_cache_quantization received a partial configuration");
  }
  if (group_size < 0 || bits < 0) {
    return set_error(
        "edge_cmlx_qwen35_session_set_attention_cache_quantization received a negative configuration");
  }
  if (bits != 0 &&
      bits != 2 &&
      bits != 3 &&
      bits != 4 &&
      bits != 5 &&
      bits != 6 &&
      bits != 8) {
    return set_error(
        "edge_cmlx_qwen35_session_set_attention_cache_quantization received unsupported bits");
  }
  try {
    auto* qwen_session = checked_qwen35_session(session);
    if (qwen_session->decoded_token_count != 0) {
      return set_error(
          "edge_cmlx_qwen35_session_set_attention_cache_quantization cannot reconfigure an active decode cache");
    }
    qwen_session->attention_cache_quantization_group_size = group_size;
    qwen_session->attention_cache_quantization_bits = bits;
    qwen_session->attention_key_states.clear();
    qwen_session->attention_value_states.clear();
    qwen_session->attention_quantized_key_states.clear();
    qwen_session->attention_quantized_value_states.clear();
    qwen_session->prefill_fp16_attention_materialized_pending_clear = false;
    auto gpu_device = mlx::core::Device{mlx::core::Device::gpu};
    qwen35_preallocate_attention_cache(*qwen_session, gpu_device);
    mlx::core::clear_cache();
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_set_attention_cache_quantization failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_set_frog_jump_mask(
    void* session,
    uint64_t layer_mask) {
  edge_cmlx_error.clear();
  try {
    auto* qwen_session = checked_qwen35_session(session);
    if (qwen_session->decoded_token_count != 0) {
      return set_error(
          "edge_cmlx_qwen35_session_set_frog_jump_mask cannot reconfigure an active decode cache");
    }
    if (qwen_session->config.layer_count > 64) {
      return set_error(
          "edge_cmlx_qwen35_session_set_frog_jump_mask supports at most 64 layers");
    }
    const uint64_t valid_mask = qwen_session->config.layer_count == 64
        ? UINT64_MAX
        : ((uint64_t{1} << qwen_session->config.layer_count) - 1);
    if ((layer_mask & ~valid_mask) != 0) {
      return set_error(
          "edge_cmlx_qwen35_session_set_frog_jump_mask received an out-of-range layer");
    }
    for (int layer = 0; layer < qwen_session->config.layer_count; ++layer) {
      if ((layer_mask & (uint64_t{1} << layer)) == 0) {
        continue;
      }
      const int kind = qwen_session->layer_kinds[static_cast<size_t>(layer)];
      if (kind != EdgeCmlxQwen35LayerKindGDN) {
        return set_error(
            "edge_cmlx_qwen35_session_set_frog_jump_mask can only skip GDN layers");
      }
      if (layer + 1 < qwen_session->config.layer_count &&
          qwen_session->layer_kinds[static_cast<size_t>(layer + 1)] ==
              EdgeCmlxQwen35LayerKindFullAttention) {
        return set_error(
            "edge_cmlx_qwen35_session_set_frog_jump_mask cannot skip a GDN layer immediately before full attention");
      }
    }
    qwen_session->frog_jump_layer_mask = layer_mask;
    qwen_session->frog_jump_logged = false;
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_set_frog_jump_mask failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_set_float_tensor(
    void* session,
    int tensor_id,
    const EdgeCmlxFloatTensorDescriptor* descriptor) {
  edge_cmlx_error.clear();
  if (descriptor == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_set_float_tensor received a null descriptor");
  }
  try {
    auto* qwen_session = checked_qwen35_session(session);
    auto shape = shape_from_descriptor(*descriptor);
    auto tensor = array_from_mtl_buffer(
        descriptor->buffer,
        std::move(shape),
        mlx::core::float32,
        "qwen35 float tensor");
    auto decode_tensor = qwen35_float_tensor_for_decode(
        tensor_id, tensor, mlx::core::Device{mlx::core::Device::gpu});
    if (decode_tensor.dtype() != tensor.dtype()) {
      mlx::core::eval(decode_tensor);
    }
    qwen_session->float_tensors.insert_or_assign(
        tensor_id, std::move(decode_tensor));
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_set_float_tensor failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_set_quantized_tensor(
    void* session,
    int tensor_id,
    const EdgeCmlxQuantizedTensorDescriptor* descriptor) {
  edge_cmlx_error.clear();
  if (descriptor == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_set_quantized_tensor received a null descriptor");
  }
  if (descriptor->packed_rows <= 0 || descriptor->packed_cols <= 0 ||
      descriptor->scale_rows <= 0 || descriptor->scale_cols <= 0 ||
      descriptor->group_size <= 0 || descriptor->bits <= 0) {
    return set_error(
        "edge_cmlx_qwen35_session_set_quantized_tensor received an invalid shape");
  }
  if (descriptor->scale_rows != descriptor->packed_rows) {
    return set_error(
        "edge_cmlx_qwen35_session_set_quantized_tensor scale row mismatch");
  }
  const int expanded_packed_cols =
      descriptor->packed_cols * 32 / descriptor->bits;
  const int logical_cols = descriptor->scale_cols * descriptor->group_size;
  if (logical_cols <= 0 || logical_cols > expanded_packed_cols) {
    return set_error(
        "edge_cmlx_qwen35_session_set_quantized_tensor scale shape mismatch");
  }

  try {
    auto* qwen_session = checked_qwen35_session(session);
    auto scales = qwen35_quantized_aux_tensor_for_decode(
        array_from_mtl_buffer(
            descriptor->scales_buffer,
            descriptor->scales_offset,
            mlx::core::Shape{
                static_cast<mlx::core::ShapeElem>(descriptor->scale_rows),
                static_cast<mlx::core::ShapeElem>(descriptor->scale_cols)},
            mlx::core::float32,
            "qwen35 quantized scales"),
        mlx::core::Device{mlx::core::Device::gpu});
    auto biases = qwen35_quantized_aux_tensor_for_decode(
        array_from_mtl_buffer(
            descriptor->biases_buffer,
            descriptor->biases_offset,
            mlx::core::Shape{
                static_cast<mlx::core::ShapeElem>(descriptor->scale_rows),
                static_cast<mlx::core::ShapeElem>(descriptor->scale_cols)},
            mlx::core::float32,
            "qwen35 quantized biases"),
        mlx::core::Device{mlx::core::Device::gpu});
    mlx::core::eval(scales, biases);
    EdgeCmlxQuantizedArray quantized{
        array_from_mtl_buffer(
            descriptor->packed_buffer,
            descriptor->packed_offset,
            mlx::core::Shape{
                static_cast<mlx::core::ShapeElem>(descriptor->packed_rows),
                static_cast<mlx::core::ShapeElem>(descriptor->packed_cols)},
            mlx::core::uint32,
            "qwen35 packed quantized tensor"),
        std::move(scales),
        std::move(biases),
        descriptor->group_size,
        descriptor->bits};
    qwen_session->quantized_tensors.insert_or_assign(
        tensor_id, std::move(quantized));
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_set_quantized_tensor failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_load_safetensors(
    void* session,
    const char* const* shard_paths,
    int shard_count,
    const char* model_prefix,
    int group_size,
    int bits) {
  edge_cmlx_error.clear();
  if (shard_paths == nullptr || shard_count <= 0 || model_prefix == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_load_safetensors received invalid arguments");
  }
  if (group_size <= 0 || bits <= 0) {
    return set_error(
        "edge_cmlx_qwen35_session_load_safetensors received invalid quantization");
  }
  try {
    auto* qwen_session = checked_qwen35_session(session);
    std::unordered_map<std::string, mlx::core::array> tensors;
    for (int i = 0; i < shard_count; ++i) {
      if (shard_paths[i] == nullptr) {
        return set_error(
            "edge_cmlx_qwen35_session_load_safetensors received a null shard path");
      }
      auto loaded = mlx::core::load_safetensors(std::string(shard_paths[i]));
      for (auto& item : loaded.first) {
        if (qwen35_should_skip_text_decode_tensor(item.first)) {
          continue;
        }
        tensors.insert_or_assign(item.first, std::move(item.second));
      }
    }

    const std::string prefix(model_prefix);
    register_loaded_weight_tensor(
        *qwen_session,
        qwen35_embedding_id(),
        tensors,
        prefix + ".embed_tokens.weight",
        group_size,
        bits);
    register_loaded_float_tensor(
        *qwen_session,
        qwen35_final_norm_id(),
        tensors,
        prefix + ".norm.weight");
    register_loaded_weight_tensor(
        *qwen_session,
        qwen35_lm_head_id(),
        tensors,
        qwen35_lm_head_weight_name(tensors, prefix),
        group_size,
        bits);

    for (int layer = 0; layer < qwen_session->config.layer_count; ++layer) {
      const auto layer_prefix = prefix + ".layers." + std::to_string(layer);
      const int layer_kind =
          qwen_session->layer_kinds[static_cast<size_t>(layer)];
      if (layer_kind == EdgeCmlxQwen35LayerKindFullAttention) {
        register_qwen35_full_attention_layer_tensors(
            *qwen_session,
            tensors,
            layer_prefix,
            layer,
            group_size,
            bits);
      } else if (layer_kind == EdgeCmlxQwen35LayerKindGDN) {
        register_qwen35_gdn_layer_tensors(
            *qwen_session,
            tensors,
            layer_prefix,
            layer,
            group_size,
            bits);
      } else {
        return set_error(
            "edge_cmlx_qwen35_session_load_safetensors encountered an unset layer kind");
      }
    }
    mlx::core::clear_cache();
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_load_safetensors failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_restore_neural_imprint_cache(
    void* session,
    const char* artifact_path,
    int prefix_token_count) {
  edge_cmlx_error.clear();
  if (artifact_path == nullptr || artifact_path[0] == '\0') {
    return set_error(
        "edge_cmlx_qwen35_session_restore_neural_imprint_cache received an empty artifact path");
  }
  if (prefix_token_count <= 0) {
    return set_error(
        "edge_cmlx_qwen35_session_restore_neural_imprint_cache received an invalid prefix token count");
  }

  try {
    auto* qwen_session = checked_qwen35_session(session);
    if (!qwen_session->attention_dsr_policies.empty()) {
      return set_error(
          "edge_cmlx_qwen35_session_restore_neural_imprint_cache does not support active DSR policies");
    }

    auto loaded = mlx::core::load_safetensors(std::string(artifact_path));
    const auto& tensors = loaded.first;
    auto gpu_device = mlx::core::Device{mlx::core::Device::gpu};
    std::vector<mlx::core::array> materialized_states;

    qwen35_reset_decode_cache(*qwen_session);
    qwen_session->attention_key_states.clear();
    qwen_session->attention_value_states.clear();
    qwen_session->attention_quantized_key_states.clear();
    qwen_session->attention_quantized_value_states.clear();
    qwen_session->attention_active_lengths.clear();
    qwen_session->prefill_fp16_attention_materialized_pending_clear = false;

    const int conv_state_tokens =
        qwen_session->config.linear_conv_kernel_size - 1;
    const int conv_hidden =
        qwen_session->config.linear_key_head_count *
            qwen_session->config.linear_key_head_dimension * 2 +
        qwen_session->config.linear_value_head_count *
            qwen_session->config.linear_value_head_dimension;

    for (int layer = 0; layer < qwen_session->config.layer_count; ++layer) {
      const int kind = qwen_session->layer_kinds[static_cast<size_t>(layer)];
      if (kind == EdgeCmlxQwen35LayerKindUnknown) {
        return set_error(
            "edge_cmlx_qwen35_session_restore_neural_imprint_cache encountered an unset layer kind");
      }

      if (kind == EdgeCmlxQwen35LayerKindGDN) {
        auto conv_state = qwen35_restore_neural_imprint_batched_tensor(
            tensors,
            qwen35_neural_imprint_state_name(layer, 0),
            mlx::core::Shape{1, conv_state_tokens, conv_hidden},
            mlx::core::Shape{conv_state_tokens, conv_hidden},
            gpu_device);
        auto recurrent_state = qwen35_restore_neural_imprint_batched_tensor(
            tensors,
            qwen35_neural_imprint_state_name(layer, 1),
            mlx::core::Shape{
                1,
                qwen_session->config.linear_value_head_count,
                qwen_session->config.linear_value_head_dimension,
                qwen_session->config.linear_key_head_dimension},
            mlx::core::Shape{
                qwen_session->config.linear_value_head_count,
                qwen_session->config.linear_value_head_dimension,
                qwen_session->config.linear_key_head_dimension},
            gpu_device);
        qwen_session->gdn_conv_states.insert_or_assign(layer, conv_state);
        qwen_session->gdn_recurrent_states.insert_or_assign(layer, recurrent_state);
        materialized_states.push_back(std::move(conv_state));
        materialized_states.push_back(std::move(recurrent_state));
        continue;
      }

      if (kind == EdgeCmlxQwen35LayerKindFullAttention) {
        auto key_state = qwen35_restore_neural_imprint_tensor(
            tensors,
            qwen35_neural_imprint_state_name(layer, 0),
            mlx::core::Shape{
                1,
                qwen_session->config.key_value_head_count,
                prefix_token_count,
                qwen_session->config.attention_head_dimension},
            gpu_device);
        auto value_state = qwen35_restore_neural_imprint_tensor(
            tensors,
            qwen35_neural_imprint_state_name(layer, 1),
            mlx::core::Shape{
                1,
                qwen_session->config.key_value_head_count,
                prefix_token_count,
                qwen_session->config.attention_head_dimension},
            gpu_device);
        qwen_session->attention_key_states.insert_or_assign(layer, key_state);
        qwen_session->attention_value_states.insert_or_assign(layer, value_state);
        qwen_session->attention_active_lengths[layer] = prefix_token_count;
        materialized_states.push_back(std::move(key_state));
        materialized_states.push_back(std::move(value_state));
        continue;
      }

      return set_error(
          "edge_cmlx_qwen35_session_restore_neural_imprint_cache encountered an invalid layer kind");
    }

    if (!materialized_states.empty()) {
      mlx::core::eval(std::move(materialized_states));
    }
    qwen_session->decoded_token_count = prefix_token_count;
    qwen_session->attention_cache_base_position = 0;
    qwen_session->attention_cache_base_index = 0;
    mlx::core::clear_cache();
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_restore_neural_imprint_cache failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_save_neural_imprint_cache(
    void* session,
    const char* artifact_path,
    const char* const* metadata_keys,
    const char* const* metadata_values,
    int metadata_count) {
  edge_cmlx_error.clear();
  if (artifact_path == nullptr || artifact_path[0] == '\0') {
    return set_error(
        "edge_cmlx_qwen35_session_save_neural_imprint_cache received an empty artifact path");
  }
  if (metadata_count < 0) {
    return set_error(
        "edge_cmlx_qwen35_session_save_neural_imprint_cache received an invalid metadata count");
  }
  if (metadata_count > 0 &&
      (metadata_keys == nullptr || metadata_values == nullptr)) {
    return set_error(
        "edge_cmlx_qwen35_session_save_neural_imprint_cache received null metadata arrays");
  }

  try {
    auto* qwen_session = checked_qwen35_session(session);
    if (qwen_session->decoded_token_count <= 0) {
      return set_error(
          "edge_cmlx_qwen35_session_save_neural_imprint_cache requires a non-empty decode cache");
    }
    if (!qwen_session->attention_dsr_policies.empty()) {
      return set_error(
          "edge_cmlx_qwen35_session_save_neural_imprint_cache does not support active DSR policies");
    }
    if (!qwen_session->attention_quantized_key_states.empty() ||
        !qwen_session->attention_quantized_value_states.empty()) {
      return set_error(
          "edge_cmlx_qwen35_session_save_neural_imprint_cache does not support quantized attention cache export");
    }
    if (qwen_session->attention_cache_base_position != 0 ||
        qwen_session->attention_cache_base_index != 0) {
      return set_error(
          "edge_cmlx_qwen35_session_save_neural_imprint_cache cannot export a trimmed or ring attention cache");
    }

    using namespace mlx::core;
    auto gpu_device = Device{Device::gpu};
    std::vector<Qwen35NeuralImprintExportEntry> entries;
    entries.reserve(static_cast<size_t>(qwen_session->config.layer_count) * 2);
    size_t payload_offset = 0;

    const int prefix_token_count = qwen_session->decoded_token_count;
    const int conv_state_tokens =
        qwen_session->config.linear_conv_kernel_size - 1;
    const int conv_hidden =
        qwen_session->config.linear_key_head_count *
            qwen_session->config.linear_key_head_dimension * 2 +
        qwen_session->config.linear_value_head_count *
            qwen_session->config.linear_value_head_dimension;

    for (int layer = 0; layer < qwen_session->config.layer_count; ++layer) {
      const int kind = qwen_session->layer_kinds[static_cast<size_t>(layer)];
      if (kind == EdgeCmlxQwen35LayerKindUnknown) {
        return set_error(
            "edge_cmlx_qwen35_session_save_neural_imprint_cache encountered an unset layer kind");
      }

      if (kind == EdgeCmlxQwen35LayerKindGDN) {
        const auto conv_state_item = qwen_session->gdn_conv_states.find(layer);
        const auto recurrent_state_item =
            qwen_session->gdn_recurrent_states.find(layer);
        if (conv_state_item == qwen_session->gdn_conv_states.end() ||
            recurrent_state_item == qwen_session->gdn_recurrent_states.end()) {
          return set_error(
              "edge_cmlx_qwen35_session_save_neural_imprint_cache missing GDN cache state");
        }
        const auto conv_name = qwen35_neural_imprint_state_name(layer, 0);
        const auto recurrent_name = qwen35_neural_imprint_state_name(layer, 1);
        Qwen35NeuralImprintExportEntry conv_entry{
            Qwen35NeuralImprintExportKind::batched,
            layer,
            0,
            conv_name,
            Shape{1, conv_state_tokens, conv_hidden},
            Shape{conv_state_tokens, conv_hidden},
            conv_state_item->second.dtype()};
        qwen35_finalize_neural_imprint_export_entry(
            *qwen_session,
            conv_entry,
            gpu_device,
            payload_offset);
        entries.push_back(std::move(conv_entry));

        Qwen35NeuralImprintExportEntry recurrent_entry{
            Qwen35NeuralImprintExportKind::batched,
            layer,
            1,
            recurrent_name,
            Shape{
                1,
                qwen_session->config.linear_value_head_count,
                qwen_session->config.linear_value_head_dimension,
                qwen_session->config.linear_key_head_dimension},
            Shape{
                qwen_session->config.linear_value_head_count,
                qwen_session->config.linear_value_head_dimension,
                qwen_session->config.linear_key_head_dimension},
            recurrent_state_item->second.dtype()};
        qwen35_finalize_neural_imprint_export_entry(
            *qwen_session,
            recurrent_entry,
            gpu_device,
            payload_offset);
        entries.push_back(std::move(recurrent_entry));
        continue;
      }

      if (kind == EdgeCmlxQwen35LayerKindFullAttention) {
        const auto key_state_item =
            qwen_session->attention_key_states.find(layer);
        const auto value_state_item =
            qwen_session->attention_value_states.find(layer);
        if (key_state_item == qwen_session->attention_key_states.end() ||
            value_state_item == qwen_session->attention_value_states.end()) {
          return set_error(
              "edge_cmlx_qwen35_session_save_neural_imprint_cache missing full-attention cache state");
        }
        const auto expected_shape = Shape{
            1,
            qwen_session->config.key_value_head_count,
            prefix_token_count,
            qwen_session->config.attention_head_dimension};
        const auto key_name = qwen35_neural_imprint_state_name(layer, 0);
        const auto value_name = qwen35_neural_imprint_state_name(layer, 1);
        Qwen35NeuralImprintExportEntry key_entry{
            Qwen35NeuralImprintExportKind::attention,
            layer,
            0,
            key_name,
            expected_shape,
            expected_shape,
            key_state_item->second.dtype()};
        qwen35_finalize_neural_imprint_export_entry(
            *qwen_session,
            key_entry,
            gpu_device,
            payload_offset);
        entries.push_back(std::move(key_entry));

        Qwen35NeuralImprintExportEntry value_entry{
            Qwen35NeuralImprintExportKind::attention,
            layer,
            1,
            value_name,
            expected_shape,
            expected_shape,
            value_state_item->second.dtype()};
        qwen35_finalize_neural_imprint_export_entry(
            *qwen_session,
            value_entry,
            gpu_device,
            payload_offset);
        entries.push_back(std::move(value_entry));
        continue;
      }

      return set_error(
          "edge_cmlx_qwen35_session_save_neural_imprint_cache encountered an invalid layer kind");
    }

    std::unordered_map<std::string, std::string> metadata;
    metadata.reserve(static_cast<size_t>(metadata_count));
    for (int index = 0; index < metadata_count; ++index) {
      const char* key = metadata_keys[index];
      const char* value = metadata_values[index];
      if (key == nullptr || key[0] == '\0' || value == nullptr) {
        return set_error(
            "edge_cmlx_qwen35_session_save_neural_imprint_cache received invalid metadata");
      }
      metadata[std::string(key)] = std::string(value);
    }

    qwen35_save_neural_imprint_safetensors_low_peak(
        *qwen_session,
        std::string(artifact_path),
        entries,
        metadata,
        gpu_device);
    mlx::core::clear_cache();
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_save_neural_imprint_cache failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_decoded_token_count(const void* session) {
  edge_cmlx_error.clear();
  try {
    return checked_qwen35_session(session)->decoded_token_count;
  } catch (const std::exception& error) {
    set_error(error.what());
    return -1;
  } catch (...) {
    set_error(
        "edge_cmlx_qwen35_session_decoded_token_count failed with an unknown error");
    return -1;
  }
}

int edge_cmlx_qwen35_session_registered_float_tensor_count(
    const void* session) {
  edge_cmlx_error.clear();
  try {
    return static_cast<int>(
        checked_qwen35_session(session)->float_tensors.size());
  } catch (const std::exception& error) {
    set_error(error.what());
    return -1;
  } catch (...) {
    set_error(
        "edge_cmlx_qwen35_session_registered_float_tensor_count failed with an unknown error");
    return -1;
  }
}

int edge_cmlx_qwen35_session_registered_quantized_tensor_count(
    const void* session) {
  edge_cmlx_error.clear();
  try {
    return static_cast<int>(
        checked_qwen35_session(session)->quantized_tensors.size());
  } catch (const std::exception& error) {
    set_error(error.what());
    return -1;
  } catch (...) {
    set_error(
        "edge_cmlx_qwen35_session_registered_quantized_tensor_count failed with an unknown error");
    return -1;
  }
}

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
    size_t output_count) {
  edge_cmlx_error.clear();
  if (input_rows <= 0 || input_cols <= 0 || output_cols <= 0) {
    return set_error(
        "edge_cmlx_qwen35_session_eval_quantized_mlp_f32_mtl received an invalid shape");
  }
  const size_t expected_output_count =
      static_cast<size_t>(input_rows) * static_cast<size_t>(output_cols);
  if (output_count < expected_output_count) {
    return set_error(
        "edge_cmlx_qwen35_session_eval_quantized_mlp_f32_mtl output buffer is too small");
  }

  try {
    using namespace mlx::core;

    auto* qwen_session = checked_qwen35_session(session);
    const auto& gate = checked_qwen35_quantized_tensor(
        *qwen_session, gate_tensor_id);
    const auto& up = checked_qwen35_quantized_tensor(
        *qwen_session, up_tensor_id);
    const auto& down = checked_qwen35_quantized_tensor(
        *qwen_session, down_tensor_id);

    if (quantized_inner_columns(gate, true) != input_cols ||
        quantized_inner_columns(up, true) != input_cols ||
        quantized_output_columns(gate, true) !=
            quantized_output_columns(up, true) ||
        quantized_inner_columns(down, true) !=
            quantized_output_columns(gate, true) ||
        quantized_output_columns(down, true) != output_cols) {
      return set_error(
          "edge_cmlx_qwen35_session_eval_quantized_mlp_f32_mtl tensor shape mismatch");
    }

    {
      auto gpu_device = Device{Device::gpu};
      auto input = array_from_mtl_buffer(
          input_buffer,
          Shape{
              static_cast<ShapeElem>(input_rows),
              static_cast<ShapeElem>(input_cols)},
          float32,
          "qwen35 mlp input");
      auto output = array_from_mtl_buffer(
          output_buffer,
          Shape{
              static_cast<ShapeElem>(input_rows),
              static_cast<ShapeElem>(output_cols)},
          float32,
          "qwen35 mlp output");

      auto gate_output = affine_quantized_matmul_array(
          input, gate, true, gpu_device);
      auto up_output = affine_quantized_matmul_array(
          input, up, true, gpu_device);
      auto activation = edge_cmlx::blocks::swiglu_activation(
          gate_output, up_output, gpu_device);
      auto result = affine_quantized_matmul_array(
          activation, down, true, gpu_device);
      result.copy_shared_buffer(output);
      eval(result);
    }
    mlx::core::clear_cache();
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_eval_quantized_mlp_f32_mtl failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_eval_quantized_gdn_decode_layer_f32_mtl(
    void* session,
    int layer_index,
    const void* input_buffer,
    const void* conv_state_buffer,
    const void* recurrent_state_buffer,
    void* output_buffer,
    void* next_conv_state_buffer,
    void* next_recurrent_state_buffer,
    size_t output_count) {
  edge_cmlx_error.clear();
  try {
    using namespace mlx::core;

    auto* qwen_session = checked_qwen35_session(session);
    const auto& config = qwen_session->config;
    if (layer_index < 0 || layer_index >= config.layer_count) {
      return set_error(
          "edge_cmlx_qwen35_session_eval_quantized_gdn_decode_layer_f32_mtl received an invalid layer index");
    }
    const int hidden_size = config.hidden_size;
    const int key_hidden =
        config.linear_key_head_count * config.linear_key_head_dimension;
    const int value_hidden =
        config.linear_value_head_count * config.linear_value_head_dimension;
    const int conv_hidden = key_hidden * 2 + value_hidden;
    const int conv_state_tokens = config.linear_conv_kernel_size - 1;
    const size_t expected_output_count = static_cast<size_t>(hidden_size);
    if (output_count < expected_output_count) {
      return set_error(
          "edge_cmlx_qwen35_session_eval_quantized_gdn_decode_layer_f32_mtl output buffer is too small");
    }

    {
      auto gpu_device = Device{Device::gpu};
      auto input = array_from_mtl_buffer(
          input_buffer,
          Shape{1, static_cast<ShapeElem>(hidden_size)},
          float32,
          "qwen35 gdn decode input");
      auto conv_state = array_from_mtl_buffer(
          conv_state_buffer,
          Shape{
              static_cast<ShapeElem>(conv_state_tokens),
              static_cast<ShapeElem>(conv_hidden)},
          float32,
          "qwen35 gdn decode conv state");
      auto recurrent_state = array_from_mtl_buffer(
          recurrent_state_buffer,
          Shape{
              static_cast<ShapeElem>(config.linear_value_head_count),
              static_cast<ShapeElem>(config.linear_value_head_dimension),
              static_cast<ShapeElem>(config.linear_key_head_dimension)},
          float32,
          "qwen35 gdn decode recurrent state");
      auto output = array_from_mtl_buffer(
          output_buffer,
          Shape{1, static_cast<ShapeElem>(hidden_size)},
          float32,
          "qwen35 gdn decode output");
      auto next_conv_output = array_from_mtl_buffer(
          next_conv_state_buffer,
          Shape{
              static_cast<ShapeElem>(conv_state_tokens),
              static_cast<ShapeElem>(conv_hidden)},
          float32,
          "qwen35 gdn decode next conv state");
      auto next_recurrent_output = array_from_mtl_buffer(
          next_recurrent_state_buffer,
          Shape{
              static_cast<ShapeElem>(config.linear_value_head_count),
              static_cast<ShapeElem>(config.linear_value_head_dimension),
              static_cast<ShapeElem>(config.linear_key_head_dimension)},
          float32,
          "qwen35 gdn decode next recurrent state");

      auto attention_input = qwen35_rms_norm(
          input,
          checked_qwen35_float_tensor(
              *qwen_session, qwen35_layer_input_norm_id(layer_index)),
          config.rms_norm_epsilon,
          gpu_device);
      auto attention = qwen35_gdn_decode_attention_array(
          *qwen_session,
          layer_index,
          attention_input,
          conv_state,
          recurrent_state,
          gpu_device);
      auto attention_residual = add(input, attention.output, gpu_device);
      auto mlp_input = qwen35_rms_norm(
          attention_residual,
          checked_qwen35_float_tensor(
              *qwen_session,
              qwen35_layer_post_attention_norm_id(layer_index)),
          config.rms_norm_epsilon,
          gpu_device);
      auto mlp_output = qwen35_mlp_array(
          mlp_input,
          *qwen_session,
          layer_index,
          gpu_device);
      auto result = add(attention_residual, mlp_output, gpu_device);
      eval(result, attention.next_conv_state, attention.next_recurrent_state);
      auto copy_stream = default_stream(Device{Device::gpu});
      copy_gpu_inplace(result, output, CopyType::General, copy_stream);
      copy_gpu_inplace(
          attention.next_conv_state,
          next_conv_output,
          CopyType::General,
          copy_stream);
      copy_gpu_inplace(
          attention.next_recurrent_state,
          next_recurrent_output,
          CopyType::General,
          copy_stream);
      synchronize(copy_stream);
    }
    mlx::core::clear_cache();
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_eval_quantized_gdn_decode_layer_f32_mtl failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_eval_quantized_full_attention_decode_layer_f32_mtl(
    void* session,
    int layer_index,
    const void* input_buffer,
    int position_offset,
    void* output_buffer,
    size_t output_count) {
  edge_cmlx_error.clear();
  try {
    using namespace mlx::core;

    auto* qwen_session = checked_qwen35_session(session);
    const auto& config = qwen_session->config;
    if (layer_index < 0 || layer_index >= config.layer_count) {
      return set_error(
          "edge_cmlx_qwen35_session_eval_quantized_full_attention_decode_layer_f32_mtl received an invalid layer index");
    }
    if (position_offset < 0) {
      return set_error(
          "edge_cmlx_qwen35_session_eval_quantized_full_attention_decode_layer_f32_mtl received a negative position offset");
    }
    const int hidden_size = config.hidden_size;
    const size_t expected_output_count = static_cast<size_t>(hidden_size);
    if (output_count < expected_output_count) {
      return set_error(
          "edge_cmlx_qwen35_session_eval_quantized_full_attention_decode_layer_f32_mtl output buffer is too small");
    }

    {
      auto gpu_device = Device{Device::gpu};
      auto input = array_from_mtl_buffer(
          input_buffer,
          Shape{1, static_cast<ShapeElem>(hidden_size)},
          float32,
          "qwen35 full attention decode input");
      auto output = array_from_mtl_buffer(
          output_buffer,
          Shape{1, static_cast<ShapeElem>(hidden_size)},
          float32,
          "qwen35 full attention decode output");
      std::vector<array> eval_outputs;

      auto attention_input = qwen35_rms_norm(
          input,
          checked_qwen35_float_tensor(
              *qwen_session, qwen35_layer_input_norm_id(layer_index)),
          config.rms_norm_epsilon,
          gpu_device);
      auto attention_output = qwen35_full_attention_decode_array(
          *qwen_session,
          layer_index,
          attention_input,
          position_offset,
          eval_outputs,
          nullptr,
          gpu_device);
      auto attention_residual = add(input, attention_output, gpu_device);
      auto mlp_input = qwen35_rms_norm(
          attention_residual,
          checked_qwen35_float_tensor(
              *qwen_session,
              qwen35_layer_post_attention_norm_id(layer_index)),
          config.rms_norm_epsilon,
          gpu_device);
      auto mlp_output = qwen35_mlp_array(
          mlp_input,
          *qwen_session,
          layer_index,
          gpu_device);
      auto result = add(attention_residual, mlp_output, gpu_device);
      eval_outputs.push_back(result);
      eval(std::move(eval_outputs));
      auto copy_stream = default_stream(Device{Device::gpu});
      copy_gpu_inplace(result, output, CopyType::General, copy_stream);
      synchronize(copy_stream);
    }
    mlx::core::clear_cache();
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_eval_quantized_full_attention_decode_layer_f32_mtl failed with an unknown error");
  }
}

void validate_qwen35_decode_layer_kinds(
    const EdgeCmlxQwen35Session& qwen_session,
    const char* caller) {
  for (int layer = 0; layer < qwen_session.config.layer_count; ++layer) {
    if (qwen_session.layer_kinds[static_cast<size_t>(layer)] == 0) {
      throw std::runtime_error(std::string(caller) + " has an unset layer kind");
    }
  }
}

mlx::core::array qwen35_apply_additive_sampling_penalties(
    mlx::core::array logits,
    const EdgeCmlxQwen35Session& qwen_session,
    mlx::core::StreamOrDevice stream) {
  using namespace mlx::core;

  if (qwen_session.presence_penalty == 0.0f &&
      qwen_session.frequency_penalty == 0.0f &&
      !qwen_session.eos_sampling_suppressed &&
      qwen_session.eos_sampling_logit_penalty == 0.0f) {
    return logits;
  }

  const int vocab = static_cast<int>(logits.shape(-1));
  std::unordered_map<int, float> penalties;
  if (qwen_session.presence_penalty != 0.0f) {
    for (int token_id : qwen_session.presence_context_tokens) {
      if (token_id >= 0 && token_id < vocab) {
        penalties[token_id] += qwen_session.presence_penalty;
      }
    }
  }
  if (qwen_session.frequency_penalty != 0.0f) {
    for (int token_id : qwen_session.frequency_context_tokens) {
      if (token_id >= 0 && token_id < vocab) {
        penalties[token_id] += qwen_session.frequency_penalty;
      }
    }
  }
  if (qwen_session.eos_sampling_suppressed ||
      qwen_session.eos_sampling_logit_penalty != 0.0f) {
    const float penalty = qwen_session.eos_sampling_suppressed
        ? 1.0e9f
        : qwen_session.eos_sampling_logit_penalty;
    for (int token_id : qwen_session.eos_sampling_token_ids) {
      if (token_id >= 0 && token_id < vocab) {
        penalties[token_id] += penalty;
      }
    }
  }
  if (penalties.empty()) {
    return logits;
  }

  std::vector<int32_t> indices;
  std::vector<float> values;
  indices.reserve(penalties.size());
  values.reserve(penalties.size());
  for (const auto& entry : penalties) {
    indices.push_back(static_cast<int32_t>(entry.first));
    values.push_back(entry.second);
  }

  const bool logits_are_2d = logits.ndim() == 2;
  auto last_row = logits_are_2d
      ? slice(logits, Shape{logits.shape(0) - 1, 0}, Shape{logits.shape(0), vocab}, stream)
      : reshape(logits, Shape{1, vocab}, stream);
  auto flat_penalties = reshape(
      zeros_like(last_row, stream),
      Shape{vocab},
      stream);
  auto idx = array(
      indices.data(),
      Shape{static_cast<ShapeElem>(indices.size())},
      int32);
  auto penalty_values = array(
      values.data(),
      Shape{static_cast<ShapeElem>(values.size())},
      float32);
  auto delta = scatter(
      flat_penalties,
      idx,
      reshape(penalty_values, Shape{static_cast<ShapeElem>(values.size()), 1}, stream),
      0,
      stream);
  auto modified_last = subtract(
      last_row,
      reshape(delta, last_row.shape(), stream),
      stream);
  if (logits_are_2d && logits.shape(0) > 1) {
    auto prefix = slice(
        logits,
        Shape{0, 0},
        Shape{logits.shape(0) - 1, vocab},
        stream);
    return concatenate({prefix, modified_last}, 0, stream);
  }
  return logits_are_2d ? modified_last : reshape(modified_last, Shape{vocab}, stream);
}

Qwen35AdvanceResult qwen35_session_advance_hidden_with_state(
    EdgeCmlxQwen35Session& qwen_session,
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
    int capture_layer,
    bool stop_after_capture) {
  using namespace mlx::core;

  if (token_count <= 0) {
    throw std::runtime_error(std::string(caller) + " received an empty token sequence");
  }
  if (capture_layer >= qwen_session.config.layer_count) {
    throw std::runtime_error(std::string(caller) + " received an invalid capture layer");
  }

  auto gpu_device = Device{Device::gpu};
  std::vector<array> eval_outputs;
  const bool profile_eval = qwen35_any_eval_profile_enabled();
  Qwen35EvalOutputInventory eval_inventory;
  Qwen35EvalOutputInventory* eval_inventory_ptr =
      profile_eval ? &eval_inventory : nullptr;
  qwen35_prepare_attention_cache_for_append(
      qwen_session,
      token_count,
      gpu_device);
  if (hidden.dtype() != float16) {
    hidden = astype(hidden, float16, gpu_device);
  }
  const bool use_float32_mlp_activation =
      std::string(caller).find("code_predictor") != std::string::npos;

  const bool log_dtype_chain = !qwen_session.dtype_chain_logged;
  if (log_dtype_chain) {
    qwen_session.dtype_chain_logged = true;
    fprintf(stderr,
        "[CmlxShim] DTYPE CHAIN: embedding=%s tokens=%d\n",
        hidden.dtype() == float16 ? "f16" : (hidden.dtype() == float32 ? "f32" : "other"),
        token_count);
  }

  const int position_offset = qwen_session.decoded_token_count;
  const int conv_state_tokens =
      qwen_session.config.linear_conv_kernel_size - 1;
  const int conv_hidden =
      qwen_session.config.linear_key_head_count *
          qwen_session.config.linear_key_head_dimension * 2 +
      qwen_session.config.linear_value_head_count *
          qwen_session.config.linear_value_head_dimension;
  std::optional<array> captured_hidden;

  for (int layer = 0; layer < qwen_session.config.layer_count; ++layer) {
    if (qwen35_should_skip_frog_jump_layer(qwen_session, layer)) {
      if (!qwen_session.frog_jump_logged) {
        qwen_session.frog_jump_logged = true;
        fprintf(stderr,
            "[CmlxShim] FROG JUMP: enabled mask=0x%llx tokens=%d\n",
            static_cast<unsigned long long>(qwen_session.frog_jump_layer_mask),
            token_count);
      }
      continue;
    }
    auto residual = hidden;
    auto attention_input = qwen35_rms_norm(
        hidden,
        checked_qwen35_float_tensor(
            qwen_session, qwen35_layer_input_norm_id(layer)),
        qwen_session.config.rms_norm_epsilon,
        gpu_device);
    if (log_dtype_chain && layer == 0) {
      fprintf(stderr,
          "[CmlxShim] DTYPE CHAIN: rms_norm_out=%s\n",
          attention_input.dtype() == float16 ? "f16" : (attention_input.dtype() == float32 ? "f32" : "other"));
    }
    const int kind =
        qwen_session.layer_kinds[static_cast<size_t>(layer)];
    auto attention_output = [&]() -> array {
      if (kind == EdgeCmlxQwen35LayerKindGDN) {
        auto conv_state_item = qwen_session.gdn_conv_states.find(layer);
        auto recurrent_state_item =
            qwen_session.gdn_recurrent_states.find(layer);
        auto conv_state =
            conv_state_item == qwen_session.gdn_conv_states.end()
            ? zeros(
                  Shape{conv_state_tokens, conv_hidden},
                  attention_input.dtype(),
                  gpu_device)
            : conv_state_item->second;
        auto recurrent_state =
            recurrent_state_item == qwen_session.gdn_recurrent_states.end()
            ? zeros(
                  Shape{
                      qwen_session.config.linear_value_head_count,
                      qwen_session.config.linear_value_head_dimension,
                      qwen_session.config.linear_key_head_dimension},
                  float32,
                  gpu_device)
            : recurrent_state_item->second;
        if (profile_eval) {
          const auto conv_snapshot = state_probe_snapshot(conv_state);
          const auto recurrent_snapshot = state_probe_snapshot(recurrent_state);
          fprintf(stderr,
              "[CmlxShim] GDN STATE caller=%s layer=%d decoded_before=%d "
              "conv={cached=%d status=%d avail=%d prim=%d sib=%d} "
              "recur={cached=%d status=%d avail=%d prim=%d sib=%d}\n",
              caller,
              layer,
              qwen_session.decoded_token_count,
              conv_state_item == qwen_session.gdn_conv_states.end() ? 0 : 1,
              conv_snapshot.status,
              conv_snapshot.is_available,
              conv_snapshot.has_primitive,
              conv_snapshot.sibling_count,
              recurrent_state_item == qwen_session.gdn_recurrent_states.end() ? 0 : 1,
              recurrent_snapshot.status,
              recurrent_snapshot.is_available,
              recurrent_snapshot.has_primitive,
              recurrent_snapshot.sibling_count);
        }
        if (qwen35_fused_rms_scale_enabled()) {
          qwen_session.eval_profile_fused_rms_scale_hits += 1;
          qwen_session.eval_profile_fused_rms_scale_tokens +=
              static_cast<uint64_t>(token_count);
          qwen_session.eval_profile_fused_rms_scale_last_layer = layer;
          qwen_session.eval_profile_fused_rms_scale_last_token_count =
              token_count;
        }
        auto attention = qwen35_gdn_decode_attention_array(
            qwen_session,
            layer,
            attention_input,
            conv_state,
            recurrent_state,
            gpu_device);
        const bool skip_gdn_stop_gradient =
            qwen35_skip_gdn_stop_gradient_enabled();
        auto stored_conv_state = skip_gdn_stop_gradient
            ? attention.next_conv_state
            : stop_gradient(attention.next_conv_state, gpu_device);
        auto stored_recurrent_state = skip_gdn_stop_gradient
            ? attention.next_recurrent_state
            : stop_gradient(attention.next_recurrent_state, gpu_device);
        qwen_session.gdn_conv_states.insert_or_assign(
            layer, stored_conv_state);
        qwen_session.gdn_recurrent_states.insert_or_assign(
            layer, stored_recurrent_state);
        qwen35_eval_output_push(
            eval_outputs,
            stored_conv_state,
            eval_inventory_ptr,
            "gdn_conv_state");
        qwen35_eval_output_push(
            eval_outputs,
            stored_recurrent_state,
            eval_inventory_ptr,
            "gdn_recurrent_state");
        return attention.output;
      }
      if (kind == EdgeCmlxQwen35LayerKindFullAttention) {
        return qwen35_full_attention_decode_array(
            qwen_session,
            layer,
            attention_input,
            position_offset,
            eval_outputs,
            eval_inventory_ptr,
            gpu_device);
      }
      throw std::runtime_error(
          std::string(caller) + " received an invalid layer kind");
    }();

    hidden = add(residual, attention_output, gpu_device);
    auto mlp_input = qwen35_rms_norm(
        hidden,
        checked_qwen35_float_tensor(
            qwen_session, qwen35_layer_post_attention_norm_id(layer)),
        qwen_session.config.rms_norm_epsilon,
        gpu_device);
    auto mlp_output = qwen35_mlp_array(
        mlp_input,
        qwen_session,
        layer,
        gpu_device,
        use_float32_mlp_activation);
    hidden = add(hidden, mlp_output, gpu_device);
    if (layer == capture_layer) {
      auto last_layer_hidden = token_count == 1
          ? hidden
          : slice(
                hidden,
                Shape{token_count - 1, 0},
                Shape{token_count, qwen_session.config.hidden_size},
                gpu_device);
      captured_hidden = astype(last_layer_hidden, float32, gpu_device);
      if (stop_after_capture) {
        eval(*captured_hidden);
        return Qwen35AdvanceResult{
            *captured_hidden,
            *captured_hidden,
            *captured_hidden,
            captured_hidden};
      }
      qwen35_eval_output_push(
          eval_outputs,
          *captured_hidden,
          eval_inventory_ptr,
          "captured_hidden");
    }
  }

  auto last_hidden = token_count == 1
      ? hidden
      : slice(
            hidden,
            Shape{token_count - 1, 0},
            Shape{token_count, qwen_session.config.hidden_size},
            gpu_device);
  auto normalized = qwen35_rms_norm(
      last_hidden,
      checked_qwen35_float_tensor(qwen_session, qwen35_final_norm_id()),
      qwen_session.config.rms_norm_epsilon,
      gpu_device);
  auto logits = [&]() -> array {
    if (const auto* lm_head =
            optional_qwen35_float_tensor(qwen_session, lm_head_id)) {
      if (lm_head->shape(0) == normalized.shape(-1)) {
        return matmul(normalized, *lm_head, gpu_device);
      }
      return matmul(
          normalized, transpose(*lm_head, {1, 0}, gpu_device), gpu_device);
    }
    return affine_quantized_matmul_array(
        normalized,
        checked_qwen35_quantized_tensor(qwen_session, lm_head_id),
        true,
        gpu_device);
  }();
  if (qwen_session.repetition_penalty != 1.0f &&
      !qwen_session.repetition_context_tokens.empty()) {
    const float penalty = qwen_session.repetition_penalty;
    const int vocab = static_cast<int>(logits.shape(-1));
    std::vector<int32_t> indices;
    indices.reserve(qwen_session.repetition_context_tokens.size());
    for (int tid : qwen_session.repetition_context_tokens) {
      if (tid >= 0 && tid < vocab) {
        indices.push_back(static_cast<int32_t>(tid));
      }
    }
    if (!indices.empty()) {
      auto idx = array(indices.data(),
                       Shape{static_cast<ShapeElem>(indices.size())},
                       int32);
      auto last_row = logits.ndim() == 2
          ? slice(logits, Shape{logits.shape(0) - 1, 0}, Shape{logits.shape(0), vocab}, gpu_device)
          : reshape(logits, Shape{1, vocab}, gpu_device);
      auto gathered = take(reshape(last_row, Shape{vocab}, gpu_device), idx, gpu_device);
      auto positive_mask = greater(gathered, array(0.0f, float32), gpu_device);
      auto divided = divide(gathered, array(penalty, float32), gpu_device);
      auto multiplied = multiply(gathered, array(penalty, float32), gpu_device);
      auto penalized = where(positive_mask, divided, multiplied, gpu_device);
      const auto N = static_cast<ShapeElem>(indices.size());
      auto modified_last = reshape(
          scatter(
              reshape(last_row, Shape{vocab}, gpu_device),
              idx,
              reshape(penalized, Shape{N, 1}, gpu_device),
              0,
              gpu_device),
          last_row.shape(),
          gpu_device);
      if (logits.ndim() == 2 && logits.shape(0) > 1) {
        auto prefix = slice(logits, Shape{0, 0}, Shape{logits.shape(0) - 1, vocab}, gpu_device);
        logits = concatenate({prefix, modified_last}, 0, gpu_device);
      } else {
        if (logits.ndim() != 2) {
          logits = reshape(modified_last, Shape{vocab}, gpu_device);
        } else {
          logits = modified_last;
        }
      }
    }
  }
  logits = qwen35_apply_additive_sampling_penalties(
      logits,
      qwen_session,
      gpu_device);
  auto token = qwen35_sample_token_from_logits(
      logits,
      temperature,
      top_k,
      top_p,
      min_p,
      seed,
      gpu_device);
  qwen35_record_sample_diagnostics(
      qwen_session,
      logits,
      token,
      temperature,
      top_k,
      top_p,
      min_p,
      gpu_device);
  qwen35_eval_output_push(
      eval_outputs,
      token,
      eval_inventory_ptr,
      "token");
  if (profile_eval &&
      qwen35_should_record_graph_profile(qwen_session, caller, token_count)) {
    qwen_session.eval_profile_last_graph_summary =
        qwen35_graph_profile_summary(eval_outputs);
  }
  const bool clear_prefill_fp16_cache =
      qwen_session.prefill_fp16_attention_materialized_pending_clear;
  const size_t profile_output_count = profile_eval ? eval_outputs.size() : 0;
  const size_t profile_output_bytes =
      profile_eval ? qwen35_eval_outputs_bytes(eval_outputs) : 0;
  std::chrono::steady_clock::time_point profile_start;
  mlx::core::metal::EdgeMetalProfileSnapshot metal_profile_before;
  if (profile_eval) {
    profile_start = qwen35_profile_now();
    metal_profile_before = mlx::core::metal::edge_metal_profile_snapshot();
  }
  if (async_schedule) {
    async_eval(std::move(eval_outputs));
  } else {
    eval(std::move(eval_outputs));
  }
  if (profile_eval) {
    const auto metal_profile_delta = mlx::core::metal::edge_metal_profile_delta(
        metal_profile_before,
        mlx::core::metal::edge_metal_profile_snapshot());
    const std::string inventory_summary = eval_inventory.summary();
    const double elapsed_ms = qwen35_profile_elapsed_ms(profile_start);
    qwen35_record_eval_profile_barrier(
        qwen_session,
        caller,
        async_schedule ? "async_eval" : "eval",
        token_count,
        qwen_session.decoded_token_count,
        profile_output_count,
        profile_output_bytes,
        elapsed_ms,
        eval_inventory,
        metal_profile_delta);
    const std::string metal_summary =
        mlx::core::metal::edge_metal_profile_summary(metal_profile_delta);
    fprintf(stderr,
        "[CmlxShim] EVAL PROFILE caller=%s mode=%s token_count=%d "
        "decoded_before=%d outputs=%zu bytes=%zu elapsed_ms=%.3f "
        "inventory=%s metal=%s\n",
        caller,
        async_schedule ? "async_eval" : "eval",
        token_count,
        qwen_session.decoded_token_count,
        profile_output_count,
        profile_output_bytes,
        elapsed_ms,
        inventory_summary.c_str(),
        metal_summary.c_str());
  }
  if (clear_prefill_fp16_cache) {
    qwen_session.prefill_fp16_attention_materialized_pending_clear = false;
    mlx::core::clear_cache();
  }

  const int previous_token_count = qwen_session.decoded_token_count;
  qwen_session.decoded_token_count += token_count;
  const int clear_cache_interval = qwen35_decode_clear_cache_interval();
  if (clear_cache_interval > 0 &&
      qwen_session.decoded_token_count / clear_cache_interval !=
          previous_token_count / clear_cache_interval) {
    mlx::core::clear_cache();
  }
  return Qwen35AdvanceResult{token, normalized, logits, captured_hidden};
}

mlx::core::array qwen35_session_advance_hidden(
    EdgeCmlxQwen35Session& qwen_session,
    mlx::core::array hidden,
    int token_count,
    bool async_schedule,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    uint64_t seed,
    const char* caller) {
  return qwen35_session_advance_hidden_with_state(
      qwen_session,
      std::move(hidden),
      token_count,
      async_schedule,
      temperature,
      top_k,
      top_p,
      min_p,
      seed,
      caller,
      qwen35_lm_head_id()).token;
}

mlx::core::array qwen35_session_advance_token_indices(
    EdgeCmlxQwen35Session& qwen_session,
    const mlx::core::array& token_indices,
    int token_count,
    bool async_schedule,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    uint64_t seed,
    const char* caller) {
  auto gpu_device = mlx::core::Device{mlx::core::Device::gpu};
  return qwen35_session_advance_hidden(
      qwen_session,
      qwen35_embedding_for_indices(qwen_session, token_indices, gpu_device),
      token_count,
      async_schedule,
      temperature,
      top_k,
      top_p,
      min_p,
      seed,
      caller);
}
void qwen35_session_advance_tokens(
    EdgeCmlxQwen35Session& qwen_session,
    const int* token_ids,
    int token_count,
    int* output_token_id,
    const char* caller) {
  using namespace mlx::core;

  if (output_token_id == nullptr) {
    throw std::runtime_error(std::string(caller) + " received a null output pointer");
  }
  auto token = qwen35_session_advance_token_indices(
      qwen_session,
      array(token_ids, Shape{token_count}, int32),
      token_count,
      false,
      0.0f,
      0,
      1.0f,
      0.0f,
      0,
      caller);
  qwen_session.pending_token.reset();
  *output_token_id = static_cast<int>(token.data<uint32_t>()[0]);
}

int edge_cmlx_qwen35_session_decode_step(
    void* session,
    const int* token_ids,
    int token_count,
    int* output_token_id) {
  edge_cmlx_error.clear();
  if (token_ids == nullptr || output_token_id == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_decode_step received a null pointer");
  }
  if (token_count != 1) {
    return set_error(
        "edge_cmlx_qwen35_session_decode_step currently supports one token");
  }

  try {
    auto* qwen_session = checked_qwen35_session(session);
    validate_qwen35_decode_layer_kinds(
        *qwen_session,
        "edge_cmlx_qwen35_session_decode_step");
    qwen35_session_advance_tokens(
        *qwen_session,
        token_ids,
        token_count,
        output_token_id,
        "edge_cmlx_qwen35_session_decode_step");
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_decode_step failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_prefill(
    void* session,
    const int* token_ids,
    int token_count,
    int* output_token_id) {
  edge_cmlx_error.clear();
  if (token_ids == nullptr || output_token_id == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_prefill received a null pointer");
  }
  if (token_count <= 0) {
    return set_error(
        "edge_cmlx_qwen35_session_prefill received an empty token sequence");
  }

  try {
    auto* qwen_session = checked_qwen35_session(session);
    if (qwen_session->decoded_token_count == 0) {
      fprintf(stderr,
          "[CmlxShim] FIRST PREFILL: tokens=%d "
          "dsr_layers=%zu cache_limit=%d "
          "quant_bits=%d quant_gs=%d "
          "can_use_fused_check=(queries_will_be_f16=%s)\n",
          token_count,
          qwen_session->attention_dsr_policies.size(),
          qwen_session->attention_cache_limit,
          qwen_session->attention_cache_quantization_bits,
          qwen_session->attention_cache_quantization_group_size,
          "yes_after_embedding_f16_cast");
      for (const auto& [layer, policy] : qwen_session->attention_dsr_policies) {
        if (layer == qwen_session->attention_dsr_policies.begin()->first) {
          fprintf(stderr,
              "[CmlxShim] DSR policy sample layer=%d: "
              "max_size=%d heavy=%d recent=%d sink=%d evict_interval=%d\n",
              layer, policy.max_size, policy.heavy_budget,
              policy.recent_budget, policy.sink_size,
              policy.eviction_interval);
        }
      }
    }
    validate_qwen35_decode_layer_kinds(
        *qwen_session,
        "edge_cmlx_qwen35_session_prefill");
    qwen35_session_advance_tokens(
        *qwen_session,
        token_ids,
        token_count,
        output_token_id,
        "edge_cmlx_qwen35_session_prefill");
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_prefill failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_capture_last_hidden(
    void* session,
    const int* token_ids,
    int token_count,
    int target_layer,
    float* output,
    int output_count) {
  edge_cmlx_error.clear();
  if (token_ids == nullptr || output == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_capture_last_hidden received a null pointer");
  }
  if (token_count <= 0) {
    return set_error(
        "edge_cmlx_qwen35_session_capture_last_hidden received an empty token sequence");
  }

  try {
    using namespace mlx::core;
    auto* qwen_session = checked_qwen35_session(session);
    if (target_layer < 0 || target_layer >= qwen_session->config.layer_count) {
      return set_error(
          "edge_cmlx_qwen35_session_capture_last_hidden received an invalid target layer");
    }
    if (output_count < qwen_session->config.hidden_size) {
      return set_error(
          "edge_cmlx_qwen35_session_capture_last_hidden output buffer is too small");
    }
    validate_qwen35_decode_layer_kinds(
        *qwen_session,
        "edge_cmlx_qwen35_session_capture_last_hidden");

    Qwen35DecodeStateGuard state_guard(*qwen_session);
    qwen35_reset_decode_cache(*qwen_session);
    qwen_session->frog_jump_layer_mask = 0;
    {
      auto token_indices = array(token_ids, Shape{token_count}, int32);
      auto hidden = qwen35_embedding_for_indices(
          *qwen_session,
          token_indices,
          Device{Device::gpu});
      auto result = qwen35_session_advance_hidden_with_state(
          *qwen_session,
          std::move(hidden),
          token_count,
          false,
          0.0f,
          0,
          1.0f,
          0.0f,
          0,
          "edge_cmlx_qwen35_session_capture_last_hidden",
          qwen35_lm_head_id(),
          target_layer,
          true);
      if (!result.captured_hidden.has_value()) {
        return set_error(
            "edge_cmlx_qwen35_session_capture_last_hidden missing captured hidden state");
      }
      const auto& captured = *result.captured_hidden;
      validate_array_element_count(
          captured,
          static_cast<size_t>(qwen_session->config.hidden_size),
          "edge_cmlx_qwen35_session_capture_last_hidden");
      std::memcpy(
          output,
          captured.data<float>(),
          static_cast<size_t>(qwen_session->config.hidden_size) * sizeof(float));
    }
    state_guard.restore();
    mlx::core::clear_cache();
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_capture_last_hidden failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_prefill_async(
    void* session,
    const int* token_ids,
    int token_count) {
  edge_cmlx_error.clear();
  if (token_ids == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_prefill_async received a null pointer");
  }
  if (token_count <= 0) {
    return set_error(
        "edge_cmlx_qwen35_session_prefill_async received an empty token sequence");
  }

  try {
    auto* qwen_session = checked_qwen35_session(session);
    validate_qwen35_decode_layer_kinds(
        *qwen_session,
        "edge_cmlx_qwen35_session_prefill_async");
    qwen_session->pending_token = qwen35_session_advance_token_indices(
        *qwen_session,
        mlx::core::array(token_ids, mlx::core::Shape{token_count}, mlx::core::int32),
        token_count,
        true,
        0.0f,
        0,
        1.0f,
        0.0f,
        0,
        "edge_cmlx_qwen35_session_prefill_async");
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_prefill_async failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_synchronize(void* session) {
  edge_cmlx_error.clear();
  try {
    auto* qwen_session = checked_qwen35_session(session);
    if (qwen_session->pending_token.has_value()) {
      mlx::core::eval(*qwen_session->pending_token);
    }
    mlx::core::synchronize();
    mlx::core::clear_cache();
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_synchronize failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_prefill_sampled_async(
    void* session,
    const int* token_ids,
    int token_count,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    uint64_t seed) {
  edge_cmlx_error.clear();
  if (token_ids == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_prefill_sampled_async received a null pointer");
  }
  if (token_count <= 0) {
    return set_error(
        "edge_cmlx_qwen35_session_prefill_sampled_async received an empty token sequence");
  }
  if (!(temperature > 0.0f) || !std::isfinite(temperature)) {
    return set_error(
        "edge_cmlx_qwen35_session_prefill_sampled_async received an invalid temperature");
  }
  if (top_k < 0) {
    return set_error(
        "edge_cmlx_qwen35_session_prefill_sampled_async received an invalid top_k");
  }
  if (!(top_p > 0.0f && top_p <= 1.0f) || !std::isfinite(top_p)) {
    return set_error(
        "edge_cmlx_qwen35_session_prefill_sampled_async received an invalid top_p");
  }
  if (min_p < 0.0f || !std::isfinite(min_p)) {
    return set_error(
        "edge_cmlx_qwen35_session_prefill_sampled_async received an invalid min_p");
  }

  try {
    auto* qwen_session = checked_qwen35_session(session);
    validate_qwen35_decode_layer_kinds(
        *qwen_session,
        "edge_cmlx_qwen35_session_prefill_sampled_async");
    qwen_session->pending_token = qwen35_session_advance_token_indices(
        *qwen_session,
        mlx::core::array(token_ids, mlx::core::Shape{token_count}, mlx::core::int32),
        token_count,
        true,
        temperature,
        top_k,
        top_p,
        min_p,
        seed,
        "edge_cmlx_qwen35_session_prefill_sampled_async");
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_prefill_sampled_async failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_prefill_embeddings(
    void* session,
    const float* embeddings,
    int token_count,
    int hidden_size,
    int* output_token_id) {
  edge_cmlx_error.clear();
  if (embeddings == nullptr || output_token_id == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_prefill_embeddings received a null pointer");
  }
  if (token_count <= 0 || hidden_size <= 0) {
    return set_error(
        "edge_cmlx_qwen35_session_prefill_embeddings received an invalid shape");
  }

  try {
    using namespace mlx::core;
    auto* qwen_session = checked_qwen35_session(session);
    if (hidden_size != qwen_session->config.hidden_size) {
      return set_error(
          "edge_cmlx_qwen35_session_prefill_embeddings hidden size mismatch");
    }
    validate_qwen35_decode_layer_kinds(
        *qwen_session,
        "edge_cmlx_qwen35_session_prefill_embeddings");
    auto token = qwen35_session_advance_hidden(
        *qwen_session,
        array(embeddings, Shape{token_count, hidden_size}, float32),
        token_count,
        false,
        0.0f,
        0,
        1.0f,
        0.0f,
        0,
        "edge_cmlx_qwen35_session_prefill_embeddings");
    qwen_session->pending_token.reset();
    *output_token_id = static_cast<int>(token.data<uint32_t>()[0]);
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_prefill_embeddings failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_prefill_media_features(
    void* session,
    const int* token_ids,
    int token_count,
    const float* media_features,
    int media_feature_count,
    int hidden_size,
    int media_token_id,
    int* output_token_id) {
  edge_cmlx_error.clear();
  if (token_ids == nullptr || media_features == nullptr ||
      output_token_id == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_prefill_media_features received a null pointer");
  }
  if (token_count <= 0 || media_feature_count <= 0 || hidden_size <= 0) {
    return set_error(
        "edge_cmlx_qwen35_session_prefill_media_features received an invalid shape");
  }

  try {
    using namespace mlx::core;
    auto* qwen_session = checked_qwen35_session(session);
    if (hidden_size != qwen_session->config.hidden_size) {
      return set_error(
          "edge_cmlx_qwen35_session_prefill_media_features hidden size mismatch");
    }
    validate_qwen35_decode_layer_kinds(
        *qwen_session,
        "edge_cmlx_qwen35_session_prefill_media_features");
    std::vector<int> media_positions;
    media_positions.reserve(static_cast<size_t>(media_feature_count));
    for (int i = 0; i < token_count; ++i) {
      if (token_ids[i] == media_token_id) {
        media_positions.push_back(i);
      }
    }
    if (static_cast<int>(media_positions.size()) != media_feature_count) {
      return set_error(
          "edge_cmlx_qwen35_session_prefill_media_features media token count mismatch");
    }

    auto gpu_device = Device{Device::gpu};
    auto hidden = qwen35_embedding_for_indices(
        *qwen_session,
        array(token_ids, Shape{token_count}, int32),
        gpu_device);
    auto features = array(
        media_features,
        Shape{media_feature_count, hidden_size},
        float32);
    if (features.dtype() != hidden.dtype()) {
      features = astype(features, hidden.dtype(), gpu_device);
    }
    auto indices = array(
        media_positions.data(),
        Shape{media_feature_count},
        int32);
    hidden = scatter(
        hidden,
        indices,
        reshape(features, Shape{media_feature_count, 1, hidden_size}, gpu_device),
        0,
        gpu_device);
    auto token = qwen35_session_advance_hidden(
        *qwen_session,
        hidden,
        token_count,
        false,
        0.0f,
        0,
        1.0f,
        0.0f,
        0,
        "edge_cmlx_qwen35_session_prefill_media_features");
    qwen_session->pending_token.reset();
    *output_token_id = static_cast<int>(token.data<uint32_t>()[0]);
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_prefill_media_features failed with an unknown error");
  }
}
int edge_cmlx_qwen35_session_next_token(
    void* session,
    int* output_token_id) {
  edge_cmlx_error.clear();
  if (output_token_id == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_next_token received a null output pointer");
  }

  try {
    auto* qwen_session = checked_qwen35_session(session);
    validate_qwen35_decode_layer_kinds(
        *qwen_session,
        "edge_cmlx_qwen35_session_next_token");
    if (!qwen_session->pending_token.has_value()) {
      return set_error(
          "edge_cmlx_qwen35_session_next_token has no pending token");
    }

    auto current_token = *qwen_session->pending_token;
    qwen_session->emitted_sample_diagnostics =
        qwen_session->pending_sample_diagnostics;
    auto next_token = qwen35_session_advance_token_indices(
        *qwen_session,
        current_token,
        1,
        true,
        0.0f,
        0,
        1.0f,
        0.0f,
        0,
        "edge_cmlx_qwen35_session_next_token");
    qwen_session->pending_token = next_token;
    const bool profile_eval = qwen35_eval_profile_enabled();
    std::chrono::steady_clock::time_point profile_start;
    if (profile_eval) {
      profile_start = qwen35_profile_now();
    }
    current_token.eval();
    if (profile_eval) {
      const double elapsed_ms = qwen35_profile_elapsed_ms(profile_start);
      qwen35_record_token_read_profile(
          *qwen_session,
          "edge_cmlx_qwen35_session_next_token",
          qwen_session->decoded_token_count,
          current_token.nbytes(),
          elapsed_ms);
      fprintf(stderr,
          "[CmlxShim] EVAL PROFILE caller=edge_cmlx_qwen35_session_next_token "
          "mode=token_read decoded_before=%d outputs=1 bytes=%zu "
          "elapsed_ms=%.3f\n",
          qwen_session->decoded_token_count,
          current_token.nbytes(),
          elapsed_ms);
    }
    const int emitted_token_id =
        static_cast<int>(current_token.data<uint32_t>()[0]);
    *output_token_id = emitted_token_id;
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_next_token failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_next_sampled_token(
    void* session,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    uint64_t seed,
    int* output_token_id) {
  edge_cmlx_error.clear();
  if (output_token_id == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_next_sampled_token received a null output pointer");
  }
  if (!(temperature > 0.0f) || !std::isfinite(temperature)) {
    return set_error(
        "edge_cmlx_qwen35_session_next_sampled_token received an invalid temperature");
  }
  if (top_k < 0) {
    return set_error(
        "edge_cmlx_qwen35_session_next_sampled_token received an invalid top_k");
  }
  if (!(top_p > 0.0f && top_p <= 1.0f) || !std::isfinite(top_p)) {
    return set_error(
        "edge_cmlx_qwen35_session_next_sampled_token received an invalid top_p");
  }
  if (min_p < 0.0f || !std::isfinite(min_p)) {
    return set_error(
        "edge_cmlx_qwen35_session_next_sampled_token received an invalid min_p");
  }

  try {
    auto* qwen_session = checked_qwen35_session(session);
    validate_qwen35_decode_layer_kinds(
        *qwen_session,
        "edge_cmlx_qwen35_session_next_sampled_token");
    if (!qwen_session->pending_token.has_value()) {
      return set_error(
          "edge_cmlx_qwen35_session_next_sampled_token has no pending token");
    }

    auto current_token = *qwen_session->pending_token;
    const bool profile_eval = qwen35_eval_profile_enabled();
    std::chrono::steady_clock::time_point profile_start;
    if (profile_eval) {
      profile_start = qwen35_profile_now();
    }
    current_token.eval();
    if (profile_eval) {
      const double elapsed_ms = qwen35_profile_elapsed_ms(profile_start);
      qwen35_record_token_read_profile(
          *qwen_session,
          "edge_cmlx_qwen35_session_next_sampled_token",
          qwen_session->decoded_token_count,
          current_token.nbytes(),
          elapsed_ms);
      fprintf(stderr,
          "[CmlxShim] EVAL PROFILE "
          "caller=edge_cmlx_qwen35_session_next_sampled_token "
          "mode=token_read decoded_before=%d outputs=1 bytes=%zu "
          "elapsed_ms=%.3f\n",
          qwen_session->decoded_token_count,
          current_token.nbytes(),
          elapsed_ms);
    }
    const int emitted_token_id =
        static_cast<int>(current_token.data<uint32_t>()[0]);
    if (qwen_session->repetition_penalty != 1.0f) {
      qwen_session->repetition_context_tokens.insert(emitted_token_id);
    }
    if (qwen_session->presence_penalty != 0.0f) {
      qwen_session->presence_context_tokens.insert(emitted_token_id);
    }
    if (qwen_session->frequency_penalty != 0.0f) {
      qwen_session->frequency_context_tokens.push_back(emitted_token_id);
    }
    qwen_session->emitted_sample_diagnostics =
        qwen_session->pending_sample_diagnostics;
    auto next_token = qwen35_session_advance_token_indices(
        *qwen_session,
        current_token,
        1,
        true,
        temperature,
        top_k,
        top_p,
        min_p,
        seed,
        "edge_cmlx_qwen35_session_next_sampled_token");
    qwen_session->pending_token = next_token;
    *output_token_id = emitted_token_id;
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_next_sampled_token failed with an unknown error");
  }
}

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
    int frequency_context_token_count) {
  edge_cmlx_error.clear();
  try {
    auto* qwen_session = checked_qwen35_session(session);
    if (!std::isfinite(repetition_penalty) || repetition_penalty <= 0.0f) {
      return set_error(
          "edge_cmlx_qwen35_session_set_sampling_penalties received an invalid repetition penalty");
    }
    if (!std::isfinite(presence_penalty)) {
      return set_error(
          "edge_cmlx_qwen35_session_set_sampling_penalties received an invalid presence penalty");
    }
    if (!std::isfinite(frequency_penalty)) {
      return set_error(
          "edge_cmlx_qwen35_session_set_sampling_penalties received an invalid frequency penalty");
    }
    if ((repetition_context_token_count > 0 && repetition_context_token_ids == nullptr) ||
        (presence_context_token_count > 0 && presence_context_token_ids == nullptr) ||
        (frequency_context_token_count > 0 && frequency_context_token_ids == nullptr)) {
      return set_error(
          "edge_cmlx_qwen35_session_set_sampling_penalties received null context tokens");
    }

    qwen_session->repetition_penalty = repetition_penalty;
    qwen_session->repetition_context_tokens.clear();
    if (repetition_context_token_count > 0) {
      qwen_session->repetition_context_tokens.insert(
          repetition_context_token_ids,
          repetition_context_token_ids + repetition_context_token_count);
    }

    qwen_session->presence_penalty = presence_penalty;
    qwen_session->presence_context_tokens.clear();
    if (presence_context_token_count > 0) {
      qwen_session->presence_context_tokens.insert(
          presence_context_token_ids,
          presence_context_token_ids + presence_context_token_count);
    }

    qwen_session->frequency_penalty = frequency_penalty;
    qwen_session->frequency_context_tokens.clear();
    if (frequency_context_token_count > 0) {
      qwen_session->frequency_context_tokens.assign(
          frequency_context_token_ids,
          frequency_context_token_ids + frequency_context_token_count);
    }
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_set_sampling_penalties failed");
  }
}

int edge_cmlx_qwen35_session_set_eos_sampling_bias(
    void* session,
    const int* token_ids,
    int token_count,
    int suppress,
    float logit_penalty) {
  edge_cmlx_error.clear();
  try {
    if (token_count < 0) {
      return set_error(
          "edge_cmlx_qwen35_session_set_eos_sampling_bias received an invalid token count");
    }
    if (token_count > 0 && token_ids == nullptr) {
      return set_error(
          "edge_cmlx_qwen35_session_set_eos_sampling_bias received null token ids");
    }
    if (logit_penalty < 0.0f || !std::isfinite(logit_penalty)) {
      return set_error(
          "edge_cmlx_qwen35_session_set_eos_sampling_bias received an invalid logit penalty");
    }
    auto* qwen_session = checked_qwen35_session(session);
    qwen_session->eos_sampling_token_ids.clear();
    for (int index = 0; index < token_count; ++index) {
      qwen_session->eos_sampling_token_ids.insert(token_ids[index]);
    }
    qwen_session->eos_sampling_suppressed = suppress != 0;
    qwen_session->eos_sampling_logit_penalty = logit_penalty;
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_set_eos_sampling_bias failed");
  }
}

int edge_cmlx_qwen35_session_clear_eos_sampling_bias(void* session) {
  edge_cmlx_error.clear();
  try {
    auto* qwen_session = checked_qwen35_session(session);
    qwen_session->eos_sampling_token_ids.clear();
    qwen_session->eos_sampling_suppressed = false;
    qwen_session->eos_sampling_logit_penalty = 0.0f;
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_clear_eos_sampling_bias failed");
  }
}

int edge_cmlx_qwen35_session_set_repetition_penalty(
    void* session,
    float penalty,
    const int* context_token_ids,
    int context_token_count) {
  edge_cmlx_error.clear();
  try {
    auto* qwen_session = checked_qwen35_session(session);
    if (!std::isfinite(penalty) || penalty <= 0.0f) {
      return set_error(
          "edge_cmlx_qwen35_session_set_repetition_penalty received an invalid penalty");
    }
    qwen_session->repetition_penalty = penalty;
    qwen_session->repetition_context_tokens.clear();
    qwen_session->presence_penalty = 0.0f;
    qwen_session->presence_context_tokens.clear();
    qwen_session->frequency_penalty = 0.0f;
    qwen_session->frequency_context_tokens.clear();
    if (context_token_ids != nullptr && context_token_count > 0) {
      qwen_session->repetition_context_tokens.insert(
          context_token_ids, context_token_ids + context_token_count);
    }
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_set_repetition_penalty failed");
  }
}

int edge_cmlx_qwen35_session_clear_repetition_penalty(void* session) {
  edge_cmlx_error.clear();
  try {
    auto* qwen_session = checked_qwen35_session(session);
    qwen_session->repetition_penalty = 1.0f;
    qwen_session->repetition_context_tokens.clear();
    qwen_session->presence_penalty = 0.0f;
    qwen_session->presence_context_tokens.clear();
    qwen_session->frequency_penalty = 0.0f;
    qwen_session->frequency_context_tokens.clear();
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_clear_repetition_penalty failed");
  }
}

int edge_cmlx_qwen35_session_copy_last_sample_diagnostics(
    void* session,
    char* output,
    int output_capacity) {
  edge_cmlx_error.clear();
  if (output == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_copy_last_sample_diagnostics received a null output pointer");
  }
  if (output_capacity <= 0) {
    return set_error(
        "edge_cmlx_qwen35_session_copy_last_sample_diagnostics received an invalid output capacity");
  }
  try {
    const auto* qwen_session = checked_qwen35_session(session);
    output[0] = '\0';
    if (!qwen_session->emitted_sample_diagnostics.empty()) {
      std::snprintf(
          output,
          static_cast<size_t>(output_capacity),
          "%s",
          qwen_session->emitted_sample_diagnostics.c_str());
    }
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_copy_last_sample_diagnostics failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_reset_eval_profile(void* session) {
  edge_cmlx_error.clear();
  try {
    auto* qwen_session = checked_qwen35_session(session);
    qwen35_reset_eval_profile(*qwen_session);
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_reset_eval_profile failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_copy_eval_profile(
    void* session,
    char* output,
    int output_capacity) {
  edge_cmlx_error.clear();
  if (output == nullptr) {
    return set_error(
        "edge_cmlx_qwen35_session_copy_eval_profile received a null output pointer");
  }
  if (output_capacity <= 0) {
    return set_error(
        "edge_cmlx_qwen35_session_copy_eval_profile received an invalid output capacity");
  }
  try {
    const auto* qwen_session = checked_qwen35_session(session);
    const std::string summary = qwen35_eval_profile_summary(*qwen_session);
    std::snprintf(
        output,
        static_cast<size_t>(output_capacity),
        "%s",
        summary.c_str());
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_copy_eval_profile failed with an unknown error");
  }
}

int edge_cmlx_set_memory_limit(size_t bytes) {
  edge_cmlx_error.clear();
  if (bytes == 0) {
    return set_error("edge_cmlx_set_memory_limit received a zero byte limit");
  }
  try {
    mlx::core::set_memory_limit(bytes);
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_set_memory_limit failed with an unknown error");
  }
}

int edge_cmlx_get_memory_limit(size_t* bytes) {
  edge_cmlx_error.clear();
  if (bytes == nullptr) {
    return set_error("edge_cmlx_get_memory_limit received a null pointer");
  }
  try {
    *bytes = mlx::core::get_memory_limit();
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_get_memory_limit failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_set_attention_cache_limit(
    void* session,
    int max_tokens) {
  edge_cmlx_error.clear();
  if (max_tokens < 0) {
    return set_error(
        "edge_cmlx_qwen35_session_set_attention_cache_limit received a negative limit");
  }

  try {
    auto* qwen_session = checked_qwen35_session(session);
    qwen_session->attention_cache_limit = max_tokens;
    auto gpu_device = mlx::core::Device{mlx::core::Device::gpu};
    qwen35_trim_attention_cache_if_needed(*qwen_session, gpu_device);
    if (qwen_session->decoded_token_count == 0 &&
        qwen_session->attention_key_states.empty() &&
        qwen_session->attention_value_states.empty()) {
      qwen35_preallocate_attention_cache(*qwen_session, gpu_device);
    }
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_set_attention_cache_limit failed with an unknown error");
  }
}

int edge_cmlx_qwen35_session_reset_decode_cache(void* session) {
  edge_cmlx_error.clear();
  try {
    auto* qwen_session = checked_qwen35_session(session);
    qwen35_reset_decode_cache(*qwen_session);
    if (qwen_session->tts_code_predictor_session) {
      qwen35_reset_decode_cache(*qwen_session->tts_code_predictor_session);
    }
    mlx::core::clear_cache();
    return 0;
  } catch (const std::exception& error) {
    return set_error(error.what());
  } catch (...) {
    return set_error(
        "edge_cmlx_qwen35_session_reset_decode_cache failed with an unknown error");
  }
}

uint64_t edge_cmlx_mtl_buffer_retain_count(const void* buffer) {
  if (buffer == nullptr) {
    return 0;
  }
  return static_cast<const MTL::Buffer*>(buffer)->retainCount();
}
