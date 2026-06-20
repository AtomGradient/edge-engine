// Copyright © 2023-2024 Apple Inc.
// Modified by AtomGradient: see Sources/Cmlx/PATCHES.md.

#include "mlx/backend/common/broadcasting.h"
#include "mlx/backend/common/compiled.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/reduce.h"
#include "mlx/backend/metal/unary.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace mlx::core {

namespace {

struct EdgeQMMTile {
  int bm;
  int bn;
  int bk;
  int wm;
  int wn;
};

inline const char* edge_quantized_getenv(const char* name) {
  const char* value = std::getenv(name);
  return (value && value[0] != '\0') ? value : nullptr;
}

inline const char* edge_quantized_getenv(
    const char* primary,
    const char* fallback) {
  if (const char* value = edge_quantized_getenv(primary)) {
    return value;
  }
  return edge_quantized_getenv(fallback);
}

inline bool edge_quantized_env_enabled(
    const char* primary,
    const char* fallback) {
  const char* value = edge_quantized_getenv(primary, fallback);
  if (!value) {
    return false;
  }
  return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
      std::strcmp(value, "FALSE") != 0 && std::strcmp(value, "off") != 0 &&
      std::strcmp(value, "OFF") != 0;
}

inline int edge_quantized_env_int(
    const char* primary,
    const char* fallback,
    int default_value) {
  const char* value = edge_quantized_getenv(primary, fallback);
  if (!value) {
    return default_value;
  }

  errno = 0;
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed <= 0 ||
      parsed > 1024) {
    return default_value;
  }
  return static_cast<int>(parsed);
}

inline int edge_qmv_group_y() {
  int value =
      edge_quantized_env_int("EDGE_QMV_GROUP_Y", "EDGE_CMLX_QMV_GROUP_Y", 2);
  return (value == 1 || value == 2 || value == 4) ? value : 2;
}

inline int edge_qmm_splitk_override() {
  // Diagnostic-only experiment knob.
  // review-by: 2026-06-02
  // delete unless a named follow-up experiment still needs it.
  return edge_quantized_env_int(
      "EDGE_QMM_SPLITK_PARTITIONS",
      "EDGE_CMLX_QMM_SPLITK_PARTITIONS",
      0);
}

inline bool edge_parse_qmm_tile(
    const char* value,
    EdgeQMMTile& tile,
    bool require_threadgroup_shape) {
  if (!value) {
    return false;
  }

  int bm = 0;
  int bn = 0;
  int bk = 0;
  int wm = tile.wm;
  int wn = tile.wn;
  int consumed = 0;
  int parsed = std::sscanf(value, "%dx%dx%dx%dx%d%n", &bm, &bn, &bk, &wm, &wn, &consumed);
  if (parsed != 5 || value[consumed] != '\0') {
    consumed = 0;
    parsed = std::sscanf(value, "%dx%dx%d%n", &bm, &bn, &bk, &consumed);
    if (parsed != 3 || value[consumed] != '\0' || require_threadgroup_shape) {
      return false;
    }
  }

  const auto valid_block = [](int value) {
    return value == 16 || value == 32 || value == 64 || value == 128;
  };
  const auto valid_warp = [](int value) {
    return value == 1 || value == 2 || value == 4;
  };
  if (!valid_block(bm) || !valid_block(bn) || !valid_block(bk) ||
      !valid_warp(wm) || !valid_warp(wn) || wm * wn > 8) {
    return false;
  }

  tile = EdgeQMMTile{bm, bn, bk, wm, wn};
  return true;
}

inline bool edge_parse_qmm_splitk_tile(const char* value, EdgeQMMTile& tile) {
  if (!value) {
    return false;
  }

  int bm = 0;
  int bn = 0;
  int bk = 0;
  int consumed = 0;
  int parsed = std::sscanf(value, "%dx%dx%d%n", &bm, &bn, &bk, &consumed);
  if (parsed != 3 || value[consumed] != '\0') {
    std::fprintf(
        stderr,
        "edge_qmm_splitk_tile_invalid value=%s expected=bmxbnxbk\n",
        value);
    return false;
  }

  const auto valid_block = [](int value) {
    return value == 16 || value == 32 || value == 64 || value == 128;
  };
  if (!valid_block(bm) || !valid_block(bn) || !valid_block(bk)) {
    std::fprintf(
        stderr,
        "edge_qmm_splitk_tile_invalid value=%s reason=unsupported_block\n",
        value);
    return false;
  }

  tile = EdgeQMMTile{bm, bn, bk, 2, 2};
  return true;
}

inline EdgeQMMTile edge_qmm_nax_tile() {
  // Diagnostic-only experiment knob.
  // review-by: 2026-06-02
  // delete unless a named follow-up experiment still needs it.
  EdgeQMMTile tile{64, 64, 64, 2, 2};
  const char* value =
      edge_quantized_getenv("EDGE_QMM_NAX_TILE", "EDGE_CMLX_QMM_NAX_TILE");
  if (!value) {
    value = edge_quantized_getenv("EDGE_QMM_TILE", "EDGE_CMLX_QMM_TILE");
  }
  edge_parse_qmm_tile(value, tile, true);
  return tile;
}

inline bool edge_qmm_splitk_tile(EdgeQMMTile& tile) {
  // Diagnostic-only experiment knob.
  // review-by: 2026-06-02
  // delete unless a named follow-up experiment still needs it.
  tile = EdgeQMMTile{32, 32, 32, 2, 2};
  const char* value =
      edge_quantized_getenv("EDGE_QMM_SPLITK_TILE", "EDGE_CMLX_QMM_SPLITK_TILE");
  if (!value) {
    value = edge_quantized_getenv("EDGE_QMM_TILE", "EDGE_CMLX_QMM_TILE");
  }
  return edge_parse_qmm_splitk_tile(value, tile);
}

inline bool edge_quantized_diagnostics_enabled() {
  return edge_quantized_env_enabled(
      "EDGE_QUANTIZED_DISPATCH_DIAGNOSTICS",
      "EDGE_CMLX_QUANTIZED_DISPATCH_DIAGNOSTICS");
}

inline void edge_log_quantized_dispatch(
    const char* event,
    const char* path,
    const std::string& mode,
    int M,
    int N,
    int K,
    int B,
    bool transpose,
    int group_size,
    int bits,
    int vector_limit,
    int split_k,
    const MTL::Size& group_dims,
    const MTL::Size& grid_dims,
    metal::Device& d) {
  if (!edge_quantized_diagnostics_enabled()) {
    return;
  }

  std::fprintf(
      stderr,
      "edge_quantized_dispatch event=%s path=%s mode=%s arch=%s M=%d N=%d "
      "K=%d B=%d transpose=%d group_size=%d bits=%d vector_limit=%d "
      "split_k=%d group_dims=%llux%llux%llu grid_dims=%llux%llux%llu\n",
      event,
      path,
      mode.c_str(),
      d.get_architecture().c_str(),
      M,
      N,
      K,
      B,
      transpose ? 1 : 0,
      group_size,
      bits,
      vector_limit,
      split_k,
      static_cast<unsigned long long>(group_dims.width),
      static_cast<unsigned long long>(group_dims.height),
      static_cast<unsigned long long>(group_dims.depth),
      static_cast<unsigned long long>(grid_dims.width),
      static_cast<unsigned long long>(grid_dims.height),
      static_cast<unsigned long long>(grid_dims.depth));
}

inline void edge_log_quantized_route(
    const char* path,
    const std::string& mode,
    int M,
    int N,
    int K,
    int B,
    bool transpose,
    int group_size,
    int bits,
    int vector_limit,
    metal::Device& d) {
  edge_log_quantized_dispatch(
      "route",
      path,
      mode,
      M,
      N,
      K,
      B,
      transpose,
      group_size,
      bits,
      vector_limit,
      0,
      MTL::Size(0, 0, 0),
      MTL::Size(0, 0, 0),
      d);
}

template <typename... Args>
auto get_quantized_kernel_wrapped(
    metal::Device& d,
    const std::string& name,
    const std::string& func,
    const std::string& mode,
    const std::string& type,
    int group_size,
    int bits,
    Args... args) {
  std::string template_def;
  std::string fname = ((mode == "affine") ? "affine_" : "fp_") + func;
  template_def = get_template_definition(
      name, fname, type, group_size, bits, std::forward<Args>(args)...);
  return get_quantized_kernel(d, name, template_def, mode);
}

template <typename... Args>
auto get_qmm_nax_kernel_wrapped(
    metal::Device& d,
    const std::string& name,
    const std::string& func,
    const std::string& mode,
    const std::string& type,
    int group_size,
    int bits,
    Args... args) {
  std::string template_def;
  std::string fname = ((mode == "affine") ? "affine_" : "fp_") + func;
  template_def = get_template_definition(
      name, fname, type, group_size, bits, std::forward<Args>(args)...);
  return get_qmm_nax_kernel(d, name, template_def, mode);
}

inline array
ensure_row_contiguous(const array& x, metal::Device& d, const Stream& s) {
  if (!x.flags().row_contiguous) {
    array x_copy = contiguous_copy_gpu(x, s);
    metal::get_command_encoder(s).add_temporary(x_copy);
    return x_copy;
  } else {
    return x;
  }
}

inline array ensure_row_contiguous_matrix(
    const array& x,
    metal::Device& d,
    const Stream& s) {
  if (x.ndim() < 2) {
    if (x.strides()[0] == 1) {
      return x;
    }
  } else {
    auto stride_0 = x.strides()[x.ndim() - 2];
    auto stride_1 = x.strides()[x.ndim() - 1];
    if (stride_0 == x.shape(-1) && stride_1 == 1) {
      return x;
    }
  }
  array x_copy = contiguous_copy_gpu(x, s);
  metal::get_command_encoder(s).add_temporary(x_copy);
  return x_copy;
}

inline int get_qmv_batch_limit(int D, int O, metal::Device& d) {
  auto arch_size = d.get_architecture().back();
  auto arch_gen = d.get_architecture_gen();
  if (arch_gen == 13 || arch_gen == 14) {
    switch (arch_size) {
      case 'd':
        if (D <= 2048 && O <= 2048) {
          return 32;
        } else if (D <= 4096 && O <= 4096) {
          return 18;
        } else {
          return 12;
        }
      default:
        if (D <= 2048 && O <= 2048) {
          return 14;
        } else if (D <= 4096 && O <= 4096) {
          return 10;
        } else {
          return 6;
        }
    }
  } else {
    switch (arch_size) {
      case 'd':
        if (D <= 2048 && O <= 2048) {
          return 32;
        } else if (D <= 4096 && O <= 4096) {
          return 18;
        } else {
          return 12;
        }
      default:
        if (D <= 2048 && O <= 2048) {
          return 18;
        } else if (D <= 4096 && O <= 4096) {
          return 12;
        } else {
          return 10;
        }
    }
  }
}

inline int add_strides_and_shapes(
    CommandEncoder& compute_encoder,
    bool skip,
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    int offset) {
  if (skip) {
    return 0;
  }

  // TODO: Collapse batch dimensions

  int x_batch_ndims = x.ndim() - 2;
  int w_batch_ndims = w.ndim() - 2;
  compute_encoder.set_bytes(x_batch_ndims, offset++);
  compute_encoder.set_vector_bytes(x.shape(), offset++);
  compute_encoder.set_vector_bytes(x.strides(), offset++);
  compute_encoder.set_bytes(w_batch_ndims, offset++);
  compute_encoder.set_vector_bytes(w.shape(), offset++);
  compute_encoder.set_vector_bytes(w.strides(), offset++);
  compute_encoder.set_vector_bytes(scales.strides(), offset++);
  if (biases) {
    compute_encoder.set_vector_bytes(biases->strides(), offset++);
  }

  return offset;
}

inline int add_gather_strides_and_shapes(
    CommandEncoder& compute_encoder,
    const array& lhs_indices,
    const array& rhs_indices,
    int offset) {
  auto [shape, strides] = collapse_contiguous_dims(
      lhs_indices.shape(), {lhs_indices.strides(), rhs_indices.strides()});
  int ndims = shape.size();

  compute_encoder.set_bytes(ndims, offset++);
  compute_encoder.set_vector_bytes(shape, offset++);
  compute_encoder.set_vector_bytes(strides[0], offset++);
  compute_encoder.set_vector_bytes(strides[1], offset++);

  return offset;
}

} // namespace

void qmv_quad(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  constexpr int quads_per_simd = 8;
  constexpr int results_per_quadgroup = 8;
  int bn = quads_per_simd * results_per_quadgroup;
  int simdgroup_size = 32;
  MTL::Size group_dims(simdgroup_size, 1, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);
  edge_log_quantized_dispatch(
      "kernel",
      "qmv_quad",
      mode,
      M,
      N,
      K,
      B,
      true,
      group_size,
      bits,
      -1,
      0,
      group_dims,
      grid_dims,
      d);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());

  concatenate(
      kname,
      mode + "_qmv_quad_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_d_",
      K,
      B > 1 ? "_batch_1" : "_batch_0");
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "qmv_quad", mode, type_string, group_size, bits, K, B > 1);
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  int qmv_group_y = edge_qmv_group_y();
  int bn = qmv_group_y * 4;
  int bk = 32;
  MTL::Size group_dims(bk, qmv_group_y, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);
  edge_log_quantized_dispatch(
      "kernel",
      "qmv",
      mode,
      M,
      N,
      K,
      B,
      true,
      group_size,
      bits,
      -1,
      0,
      group_dims,
      grid_dims,
      d);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  bool fast = N % bn == 0 && K % 512 == 0;

  concatenate(
      kname,
      mode + (fast ? "_qmv_fast_" : "_qmv_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      B > 1 ? "_batch_1" : "_batch_0");
  auto kernel = get_quantized_kernel_wrapped(
      d,
      kname,
      (fast ? "qmv_fast" : "qmv"),
      mode,
      type_string,
      group_size,
      bits,
      B > 1);

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(qmv_group_y, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qvm_split_k(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  auto& compute_encoder = metal::get_command_encoder(s);

  int split_k = K > 8192 ? 32 : 8;
  int split_D = (K + split_k - 1) / split_k;
  int B = out.size() / M / N;
  B *= split_k;

  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = std::min(group_size, 32) * num_simdgroups;
  MTL::Size group_dims = MTL::Size(bk, num_simdgroups, 1);
  MTL::Size grid_dims = MTL::Size(M, N / bn, B);
  edge_log_quantized_dispatch(
      "kernel",
      "qvm_split_k",
      mode,
      M,
      N,
      K,
      B / split_k,
      false,
      group_size,
      bits,
      -1,
      split_k,
      group_dims,
      grid_dims,
      d);

  auto x_shape = x.shape();
  auto x_strides = x.strides();
  if (x_shape.size() == 1) {
    x_shape.insert(x_shape.begin(), 1);
    x_strides.insert(x_strides.begin(), 0);
  }

  int x_ndim = x_shape.size();
  int x_batch_ndims = x_ndim - 2;
  int w_batch_ndims = w.ndim() - 2;
  auto w_shape = w.shape();
  auto w_strides = w.strides();
  auto s_strides = scales.strides();

  // Add split_k dim with reshapes
  x_shape.insert(x_shape.end() - 2, split_k);
  x_shape.back() /= split_k;
  x_strides.insert(x_strides.end() - 2, split_D);
  x_strides[x_ndim - 1] = split_D;
  x_batch_ndims += 1;

  w_shape.insert(w_shape.end() - 2, split_k);
  w_shape[w.ndim() - 1] /= split_k;
  w_strides.insert(w_strides.end() - 2, split_D * w.shape(-1));
  w_batch_ndims += 1;
  s_strides.insert(s_strides.end() - 2, split_D * scales.shape(-1));

  int final_block_size = K - (split_k - 1) * split_D;

  auto temp_shape = out.shape();
  if (temp_shape.size() == 1) {
    temp_shape.insert(temp_shape.begin(), 1);
  }
  temp_shape.insert(temp_shape.end() - 2, split_k);
  array intermediate(temp_shape, x.dtype(), nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  compute_encoder.add_temporary(intermediate);

  std::string type_string = get_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  concatenate(
      kname,
      mode + "_qvm_split_k_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_spk_",
      split_k);

  // Encode and dispatch kernel
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "qvm_split_k", mode, type_string, group_size, bits, split_k);

  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(intermediate, c++);
  compute_encoder.set_bytes(split_D, c++);
  compute_encoder.set_bytes(N, c++);

  compute_encoder.set_bytes(x_batch_ndims, c++);
  compute_encoder.set_vector_bytes(x_shape, c++);
  compute_encoder.set_vector_bytes(x_strides, c++);
  compute_encoder.set_bytes(w_batch_ndims, c++);
  compute_encoder.set_vector_bytes(w_shape, c++);
  compute_encoder.set_vector_bytes(w_strides, c++);
  compute_encoder.set_vector_bytes(s_strides, c++);
  if (biases) {
    auto b_strides = biases->strides();
    b_strides.insert(b_strides.end() - 2, split_D * biases->shape(-1));
    compute_encoder.set_vector_bytes(b_strides, c++);
  }
  compute_encoder.set_bytes(final_block_size, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  int axis = intermediate.ndim() - 3;
  ReductionPlan plan(
      ReductionOpType::ContiguousStridedReduce,
      {intermediate.shape(axis)},
      {intermediate.strides(axis)});
  strided_reduce_general_dispatch(
      intermediate, out, "sum", plan, {axis}, compute_encoder, d, s);
}

void qvm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = std::min(group_size, 32) * num_simdgroups;
  MTL::Size group_dims(bk, num_simdgroups, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);
  edge_log_quantized_dispatch(
      "kernel",
      "qvm",
      mode,
      M,
      N,
      K,
      B,
      false,
      group_size,
      bits,
      -1,
      0,
      group_dims,
      grid_dims,
      d);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + "_qvm_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      B > 1 ? "_batch_1" : "_batch_0");
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "qvm", mode, type_string, group_size, bits, B > 1);
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qmm_nax(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  EdgeQMMTile tile = edge_qmm_nax_tile();
  int wm = tile.wm;
  int wn = tile.wn;
  int bm = tile.bm;
  int bn = tile.bn;
  int bk = tile.bk;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);
  edge_log_quantized_dispatch(
      "kernel",
      "qmm_nax",
      mode,
      M,
      N,
      K,
      B,
      transpose,
      group_size,
      bits,
      -1,
      0,
      group_dims,
      grid_dims,
      d);

  std::string kname;
  kname.reserve(64);
  bool aligned = N % bn == 0;
  bool batched = B > 1;
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + (transpose ? "_qmm_t_nax_" : "_qmm_n_nax_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm",
      bm,
      "_bn",
      bn,
      "_bk",
      bk,
      "_wm",
      wm,
      "_wn",
      wn,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "",
      batched ? "_batch_1" : "_batch_0");
  std::string template_def;
  MTL::ComputePipelineState* kernel;
  if (transpose) {
    kernel = get_qmm_nax_kernel_wrapped(
        d,
        kname,
        "qmm_t_nax",
        mode,
        type_string,
        group_size,
        bits,
        aligned,
        batched,
        bm,
        bk,
        bn,
        wm,
        wn);
  } else {
    kernel = get_qmm_nax_kernel_wrapped(
        d,
        kname,
        "qmm_n_nax",
        mode,
        type_string,
        group_size,
        bits,
        batched,
        bm,
        bk,
        bn,
        wm,
        wn);
  }
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qmm_nax(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  int wm = 2;
  int wn = 2;
  int bm = 64;
  int bn = 64;
  int bk = 32;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  std::string kname;
  kname.reserve(64);
  bool aligned = N % 64 == 0;
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + (transpose ? "_gather_qmm_t_nax_" : "_gather_qmm_n_nax_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm",
      bm,
      "_bn",
      bn,
      "_bk",
      bk,
      "_wm",
      wm,
      "_wn",
      wn,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "");
  MTL::ComputePipelineState* kernel;
  if (transpose) {
    kernel = get_qmm_nax_kernel_wrapped(
        d,
        kname,
        "gather_qmm_t_nax_",
        mode,
        type_string,
        group_size,
        bits,
        aligned,
        bm,
        bk,
        bn,
        wm,
        wn);
  } else {
    kernel = get_qmm_nax_kernel_wrapped(
        d,
        kname,
        "gather_qmm_n_nax_",
        mode,
        type_string,
        group_size,
        bits,
        bm,
        bk,
        bn,
        wm,
        wn);
  }

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(lhs_indices, c++);
  compute_encoder.set_input_array(rhs_indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  c = add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, c);
  add_gather_strides_and_shapes(compute_encoder, lhs_indices, rhs_indices, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qmm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  if (metal::is_nax_available() && transpose && (K % 64 == 0) &&
      (env::enable_tf32() || x.dtype() != float32)) {
    return qmm_nax(
        /* const array& x = */ x,
        /* const array& w = */ w,
        /* const array& scales = */ scales,
        /* const std::optional<array>& biases = */ biases,
        /* array& out = */ out,
        /* bool transpose = */ transpose,
        /* int group_size = */ group_size,
        /* int bits = */ bits,
        /* int M = */ M,
        /* int N = */ N,
        /* int K = */ K,
        /* metal::Device& d = */ d,
        /* const Stream& s = */ s,
        /* const std::string& mode = */ mode);
  }

  int B = out.size() / M / N;

  int wm = 2;
  int wn = 2;
  int bm = 32;
  int bn = 32;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);
  edge_log_quantized_dispatch(
      "kernel",
      "qmm",
      mode,
      M,
      N,
      K,
      B,
      transpose,
      group_size,
      bits,
      -1,
      0,
      group_dims,
      grid_dims,
      d);

  std::string kname;
  kname.reserve(64);
  bool aligned = N % bn == 0;
  bool batched = B > 1;
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + (transpose ? "_qmm_t_" : "_qmm_n_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "",
      batched ? "_batch_1" : "_batch_0");
  std::string template_def;
  MTL::ComputePipelineState* kernel;
  if (transpose) {
    kernel = get_quantized_kernel_wrapped(
        d,
        kname,
        "qmm_t",
        mode,
        type_string,
        group_size,
        bits,
        aligned,
        batched);
  } else {
    kernel = get_quantized_kernel_wrapped(
        d, kname, "qmm_n", mode, type_string, group_size, bits, batched);
  }
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qmm_splitk(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  // Choose split_k to target ~512 threadgroups
  EdgeQMMTile tile;
  bool tile_override = edge_qmm_splitk_tile(tile);
  int bm = tile.bm, bn = tile.bn, bk = tile.bk;
  int n_tiles = (N + bn - 1) / bn;
  int m_tiles = (M + bm - 1) / bm;
  int current_tgs = n_tiles * m_tiles;
  int split_k = std::max(1, 512 / current_tgs);
  int requested_split_k = edge_qmm_splitk_override();
  if (requested_split_k > 0) {
    split_k = requested_split_k;
  }

  // Cap split_k by the number of quantization groups
  split_k = std::min(split_k, K / group_size);

  // Ensure K divides evenly by split_k * group_size
  while (split_k > 1 && (K % (split_k * group_size) != 0)) {
    split_k--;
  }
  if (split_k <= 1) {
    return qmm(
        x, w, scales, biases, out, true, group_size, bits, M, N, K, d, s, mode);
  }

  int k_partition_size = K / split_k;
  while (split_k > 1 && (k_partition_size % bk != 0)) {
    split_k--;
    k_partition_size = K / split_k;
  }
  if (split_k <= 1) {
    return qmm(
        x, w, scales, biases, out, true, group_size, bits, M, N, K, d, s, mode);
  }
  int split_k_partition_stride = M * N;

  // Allocate intermediate buffer: insert split_k at the front so that
  // partition_stride = M * N matches the leading stride of the buffer.
  auto& compute_encoder = metal::get_command_encoder(s);
  auto temp_shape = out.shape();
  if (temp_shape.size() == 1) {
    temp_shape.insert(temp_shape.begin(), 1);
  }
  temp_shape.insert(temp_shape.begin(), split_k);
  array intermediate(temp_shape, x.dtype(), nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  compute_encoder.add_temporary(intermediate);

  // Grid: (N_tiles, M_tiles, split_k)
  MTL::Size group_dims(32, 2, 2);
  MTL::Size grid_dims(n_tiles, m_tiles, split_k);
  edge_log_quantized_dispatch(
      "kernel",
      "qmm_splitk",
      mode,
      M,
      N,
      K,
      1,
      true,
      group_size,
      bits,
      -1,
      split_k,
      group_dims,
      grid_dims,
      d);

  bool aligned = N % bn == 0;
  std::string type_string = get_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  concatenate(
      kname,
      mode + "_qmm_t_splitk_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      aligned ? "_alN_true" : "_alN_false");
  if (tile_override) {
    concatenate(kname, "_bm", bm, "_bn", bn, "_bk", bk);
  }
  auto kernel = tile_override
      ? get_quantized_kernel_wrapped(
            d,
            kname,
            "qmm_t_splitk",
            mode,
            type_string,
            group_size,
            bits,
            aligned,
            bm,
            bk,
            bn)
      : get_quantized_kernel_wrapped(
            d, kname, "qmm_t_splitk", mode, type_string, group_size, bits, aligned);

  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(intermediate, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  compute_encoder.set_bytes(k_partition_size, c++);
  compute_encoder.set_bytes(split_k_partition_stride, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  // Sum across split_k dimension (axis 0)
  ReductionPlan plan(
      ReductionOpType::ContiguousStridedReduce,
      {intermediate.shape(0)},
      {intermediate.strides(0)});
  strided_reduce_general_dispatch(
      intermediate, out, "sum", plan, {0}, compute_encoder, d, s);
}

void gather_qmm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  if (metal::is_nax_available() && transpose && (K % 64 == 0) &&
      (env::enable_tf32() || x.dtype() != float32)) {
    return gather_qmm_nax(
        /* const array& x = */ x,
        /* const array& w = */ w,
        /* const array& scales = */ scales,
        /* const std::optional<array>& biases = */ biases,
        /* const array& lhs_indices = */ lhs_indices,
        /* const array& rhs_indices = */ rhs_indices,
        /* array& out = */ out,
        /* bool transpose = */ transpose,
        /* int group_size = */ group_size,
        /* int bits = */ bits,
        /* int M = */ M,
        /* int N = */ N,
        /* int K = */ K,
        /* metal::Device& d = */ d,
        /* const Stream& s = */ s,
        /* const std::string& mode = */ mode);
  }

  int B = out.size() / M / N;

  int wm = 2;
  int wn = 2;
  int bm = 32;
  int bn = 32;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  std::string kname;
  kname.reserve(64);
  bool aligned = N % 32 == 0;
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + (transpose ? "_gather_qmm_t_" : "_gather_qmm_n_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "");
  MTL::ComputePipelineState* kernel;
  if (transpose) {
    kernel = get_quantized_kernel_wrapped(
        d, kname, "gather_qmm_t", mode, type_string, group_size, bits, aligned);
  } else {
    kernel = get_quantized_kernel_wrapped(
        d, kname, "gather_qmm_n", mode, type_string, group_size, bits);
  }

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(lhs_indices, c++);
  compute_encoder.set_input_array(rhs_indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  c = add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, c);
  add_gather_strides_and_shapes(compute_encoder, lhs_indices, rhs_indices, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  int qmv_group_y = edge_qmv_group_y();
  int bn = qmv_group_y * 4;
  int bk = 32;
  MTL::Size group_dims(bk, qmv_group_y, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);
  edge_log_quantized_dispatch(
      "kernel",
      "gather_qmv",
      mode,
      M,
      N,
      K,
      B,
      true,
      group_size,
      bits,
      -1,
      0,
      group_dims,
      grid_dims,
      d);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  bool fast = N % bn == 0 && K % 512 == 0;
  concatenate(
      kname,
      mode + (fast ? "_gather_qmv_fast_" : "_gather_qmv_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits);

  auto kernel = get_quantized_kernel_wrapped(
      d,
      kname,
      (fast ? "gather_qmv_fast" : "gather_qmv"),
      mode,
      type_string,
      group_size,
      bits);

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(lhs_indices, c++);
  compute_encoder.set_input_array(rhs_indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(qmv_group_y, c++);
  c = add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, c);
  add_gather_strides_and_shapes(compute_encoder, lhs_indices, rhs_indices, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qvm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = std::min(group_size, 32) * num_simdgroups;
  MTL::Size group_dims(bk, num_simdgroups, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + "_gather_qvm_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits);
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "gather_qvm", mode, type_string, group_size, bits);
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(lhs_indices, c++);
  compute_encoder.set_input_array(rhs_indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  c = add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, c++);
  add_gather_strides_and_shapes(compute_encoder, lhs_indices, rhs_indices, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qmm_rhs_nax(
    const array& x_,
    const array& w_,
    const array& scales_,
    const std::optional<array>& biases_,
    const array& indices_,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string mode) {
  // Start by normalizing the indices
  array indices = ensure_row_contiguous(indices_, d, s);

  // Broadcast x with indices. If we are here that means lhs_indices were not
  // provided so the lhs_indices are implied to be the shape of x broadcasted
  // with rhs_indices. We need only broadcast x and copy it as if applying the
  // lhs_indices.
  auto broadcast_with_indices = [&d, &s, &indices](const array& x) {
    if (x.size() / x.shape(-2) / x.shape(-1) == indices.size()) {
      return ensure_row_contiguous(x, d, s);
    }

    auto x_shape = indices.shape();
    x_shape.push_back(x.shape(-2));
    x_shape.push_back(x.shape(-1));
    array new_x(std::move(x_shape), x.dtype(), nullptr, {});
    broadcast(x, new_x);
    return ensure_row_contiguous(new_x, d, s);
  };

  // Normalize the input arrays
  array x = broadcast_with_indices(x_);
  array w = ensure_row_contiguous(w_, d, s);
  array scales = ensure_row_contiguous(scales_, d, s);

  // TODO: Tune the block sizes
  int bm = 64, bn = 64, bk = 64;
  int wm = 2, wn = 2;

  const bool align_M = (M % bm) == 0;
  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;

  // Make the kernel name
  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode +
          (transpose ? "_gather_qmm_rhs_nax_nt_" : "_gather_qmm_rhs_nax_nn_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm_",
      bm,
      "_bn_",
      bn,
      "_bk_",
      bk,
      "_wm_",
      wm,
      "_wn_",
      wn);

  metal::MTLFCList func_consts = {
      {&align_M, MTL::DataType::DataTypeBool, 200},
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
  };

  // And the kernel hash that includes the function constants
  std::string hash_name;
  hash_name.reserve(128);
  concatenate(
      hash_name,
      kname,
      "_align_M_",
      align_M ? 't' : 'n',
      "_align_N_",
      align_N ? 't' : 'n',
      "_align_K_",
      align_K ? 't' : 'n');

  // Get and set the kernel
  auto& compute_encoder = metal::get_command_encoder(s);
  auto kernel = get_gather_qmm_nax_kernel(
      d,
      kname,
      hash_name,
      func_consts,
      x,
      group_size,
      bits,
      mode,
      bm,
      bn,
      bk,
      wm,
      wn,
      transpose);
  compute_encoder.set_compute_pipeline_state(kernel);

  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, 1);

  int c = 0;
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases_) {
    array biases = ensure_row_contiguous(*biases_, d, s);
    compute_encoder.set_input_array(biases, c++);
  }
  compute_encoder.set_input_array(indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(M, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(K, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qmm_rhs(
    const array& x_,
    const array& w_,
    const array& scales_,
    const std::optional<array>& biases_,
    const array& indices_,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string mode) {
  if (metal::is_nax_available() && transpose &&
      (env::enable_tf32() || x_.dtype() != float32)) {
    return gather_qmm_rhs_nax(
        /* const array& x_ = */ x_,
        /* const array& w_ = */ w_,
        /* const array& scales_ = */ scales_,
        /* const std::optional<array>& biases_ = */ biases_,
        /* const array& indices_ = */ indices_,
        /* array& out = */ out,
        /* bool transpose = */ transpose,
        /* int group_size = */ group_size,
        /* int bits = */ bits,
        /* int M = */ M,
        /* int N = */ N,
        /* int K = */ K,
        /* metal::Device& d = */ d,
        /* const Stream& s = */ s,
        /* const std::string mode = */ mode);
  }

  // Start by normalizing the indices
  array indices = ensure_row_contiguous(indices_, d, s);

  // Broadcast x with indices. If we are here that means lhs_indices were not
  // provided so the lhs_indices are implied to be the shape of x broadcasted
  // with rhs_indices. We need only broadcast x and copy it as if applying the
  // lhs_indices.
  auto broadcast_with_indices = [&d, &s, &indices](const array& x) {
    if (x.size() / x.shape(-2) / x.shape(-1) == indices.size()) {
      return ensure_row_contiguous(x, d, s);
    }

    auto x_shape = indices.shape();
    x_shape.push_back(x.shape(-2));
    x_shape.push_back(x.shape(-1));
    array new_x(std::move(x_shape), x.dtype(), nullptr, {});
    broadcast(x, new_x);
    return ensure_row_contiguous(new_x, d, s);
  };

  // Normalize the input arrays
  array x = broadcast_with_indices(x_);
  array w = ensure_row_contiguous(w_, d, s);
  array scales = ensure_row_contiguous(scales_, d, s);

  // TODO: Tune the block sizes
  int bm = 16, bn = 32, bk = 32;
  int wm = 1, wn = 2;

  const bool align_M = (M % bm) == 0;
  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;

  // Make the kernel name
  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + (transpose ? "_gather_qmm_rhs_nt_" : "_gather_qmm_rhs_nn_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm_",
      bm,
      "_bn_",
      bn,
      "_bk_",
      bk,
      "_wm_",
      wm,
      "_wn_",
      wn);

  metal::MTLFCList func_consts = {
      {&align_M, MTL::DataType::DataTypeBool, 200},
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
  };

  // And the kernel hash that includes the function constants
  std::string hash_name;
  hash_name.reserve(128);
  concatenate(
      hash_name,
      kname,
      "_align_M_",
      align_M ? 't' : 'n',
      "_align_N_",
      align_N ? 't' : 'n',
      "_align_K_",
      align_K ? 't' : 'n');

  // Get and set the kernel
  auto& compute_encoder = metal::get_command_encoder(s);
  auto kernel = get_gather_qmm_kernel(
      d,
      kname,
      hash_name,
      func_consts,
      x,
      group_size,
      bits,
      mode,
      bm,
      bn,
      bk,
      wm,
      wn,
      transpose);
  compute_encoder.set_compute_pipeline_state(kernel);

  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, 1);

  int c = 0;
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases_) {
    array biases = ensure_row_contiguous(*biases_, d, s);
    compute_encoder.set_input_array(biases, c++);
  }
  compute_encoder.set_input_array(indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(M, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(K, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void dispatch_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  // It is a qmv with a small inner dimension so route to qmv_quad kernel
  if ((K == 128 || K == 64) && is_power_of_2(bits)) {
    qmv_quad(x, w, scales, biases, out, group_size, bits, M, N, K, d, s, mode);
    return;
  }
  qmv(x, w, scales, biases, out, group_size, bits, M, N, K, d, s, mode);
}

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  if (!out.data_shared_ptr()) {
    out.set_data(allocator::malloc(out.nbytes()));
  }

  // Make sure the last two dims of x and w, s, b are contiguous. This should
  // be relaxed for x.
  array x = ensure_row_contiguous_matrix(inputs[0], d, s);
  array w = ensure_row_contiguous_matrix(inputs[1], d, s);
  array scales = ensure_row_contiguous_matrix(inputs[2], d, s);
  std::optional<array> biases = std::nullopt;
  if (inputs.size() == 4) {
    biases = ensure_row_contiguous_matrix(inputs[3], d, s);
  }

  // Extract the matmul shapes
  bool non_batched = w.ndim() == 2 && x.flags().row_contiguous;
  int K = x.shape(-1);
  int M = non_batched ? x.size() / K : x.shape(-2);
  int N = out.shape(-1);

  int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;
  auto mode = quantization_mode_to_string(mode_);
  int B = out.size() / M / N;
  // It is a matrix matrix product.
  if (M >= vector_limit) {
    // Use split-K qmm for small M with transposed weights (non-batched only)
    if (transpose_ && B == 1) {
      edge_log_quantized_route(
          "qmm_splitk_candidate",
          mode,
          M,
          N,
          K,
          B,
          transpose_,
          group_size_,
          bits_,
          vector_limit,
          d);
      qmm_splitk(
          x, w, scales, biases, out, group_size_, bits_, M, N, K, d, s, mode);
      return;
    }
    edge_log_quantized_route(
        "qmm",
        mode,
        M,
        N,
        K,
        B,
        transpose_,
        group_size_,
        bits_,
        vector_limit,
        d);
    qmm(x,
        w,
        scales,
        biases,
        out,
        transpose_,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        mode);
    return;
  }

  // Run of the mill qmv
  if (transpose_) {
    edge_log_quantized_route(
        "qmv",
        mode,
        M,
        N,
        K,
        B,
        transpose_,
        group_size_,
        bits_,
        vector_limit,
        d);
    dispatch_qmv(
        x, w, scales, biases, out, group_size_, bits_, M, N, K, d, s, mode);
    return;
  }

  // Run of the mill qvm
  if (K < 1024) {
    edge_log_quantized_route(
        "qvm",
        mode,
        M,
        N,
        K,
        B,
        transpose_,
        group_size_,
        bits_,
        vector_limit,
        d);
    qvm(x, w, scales, biases, out, group_size_, bits_, M, N, K, d, s, mode);
    return;
  }

  // Qvm with large dimension so route to a split K kernel for more parallelism
  edge_log_quantized_route(
      "qvm_split_k",
      mode,
      M,
      N,
      K,
      B,
      transpose_,
      group_size_,
      bits_,
      vector_limit,
      d);
  qvm_split_k(
      x, w, scales, biases, out, group_size_, bits_, M, N, K, d, s, mode);
  return;
}

void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc(out.nbytes()));

  array x = ensure_row_contiguous_matrix(inputs[0], d, s);
  array w = ensure_row_contiguous_matrix(inputs[1], d, s);
  array scales = ensure_row_contiguous_matrix(inputs[2], d, s);
  std::optional<array> biases = std::nullopt;
  if (inputs.size() == 6) {
    biases = ensure_row_contiguous_matrix(inputs[3], d, s);
  }
  const array& lhs_indices = inputs[inputs.size() - 2];
  const array& rhs_indices = inputs[inputs.size() - 1];

  int K = x.shape(-1);
  int M = x.shape(-2);
  int N = out.shape(-1);
  int B = out.size() / M / N;
  int E = w.size() / w.shape(-1) / w.shape(-2);
  int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;
  auto mode = quantization_mode_to_string(mode_);

  // We are walking x in order and w is also in order so we can batch up the
  // matmuls and reuse reading x and w.
  //
  // TODO: Tune 16 and 4 here a bit better.
  if (M == 1 && B >= 16 && right_sorted_ == true && B / E >= 4) {
    gather_qmm_rhs(
        x,
        w,
        scales,
        biases,
        rhs_indices,
        out,
        transpose_,
        group_size_,
        bits_,
        x.size() / K,
        N,
        K,
        d,
        s,
        mode);
    return;
  }

  // It is a matrix matrix product
  if (M >= vector_limit) {
    gather_qmm(
        x,
        w,
        scales,
        biases,
        lhs_indices,
        rhs_indices,
        out,
        transpose_,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        mode);
    return;
  }

  if (transpose_) {
    gather_qmv(
        x,
        w,
        scales,
        biases,
        lhs_indices,
        rhs_indices,
        out,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        mode);
    return;
  }

  gather_qvm(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      out,
      group_size_,
      bits_,
      M,
      N,
      K,
      d,
      s,
      mode);
}

void quantize_dequantize(
    const array& in,
    array& out,
    std::string mode,
    int group_size,
    int bits,
    metal::Device& d,
    const Stream& s) {
  auto& compute_encoder = metal::get_command_encoder(s);

  auto w = ensure_row_contiguous(in, d, s);
  compute_encoder.set_input_array(w, 0);
  compute_encoder.set_output_array(out, 1);
  auto type_string = get_type_string(in.dtype());
  std::string kname;
  concatenate(
      kname,
      mode + "_quantize_dequantize_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits);
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "quantize_dequantize", mode, type_string, group_size, bits);

  compute_encoder.set_compute_pipeline_state(kernel);

  constexpr int uint8_per_uint32 = 4;
  constexpr int simd_size = 32;
  int packs_per_int = (bits == 3 || bits == 5) ? 8 : bits == 6 ? 4 : 8 / bits;
  int per_thread = std::max(group_size / simd_size, 1);
  size_t nthreads = w.size() / per_thread;

  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  if (thread_group_size > nthreads) {
    thread_group_size = nthreads;
  }
  auto group_dims = MTL::Size(thread_group_size, 1, 1);
  bool use_2d = nthreads > UINT_MAX;
  auto grid_shape = w.shape();
  grid_shape.back() /= per_thread;
  MTL::Size grid_dims = use_2d ? get_2d_grid_dims(grid_shape, w.strides())
                               : MTL::Size(nthreads, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void QQMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto mode = quantization_mode_to_string(mode_);
  bool w_quantized = (inputs[1].dtype() == uint32);
  if (w_quantized && inputs[0].shape(-2) == 1) {
    out.set_data(allocator::malloc(out.nbytes()));

    bool donate_x = inputs[0].is_donatable();
    array x = ensure_row_contiguous(inputs[0], d, s);
    // If x is a copy it should be donatable
    donate_x |= x.is_donatable();
    auto xhat = donate_x
        ? x
        : array(allocator::malloc(x.nbytes()), x.shape(), x.dtype());
    quantize_dequantize(x, xhat, mode, group_size_, bits_, d, s);

    // Make sure the last two dims of w and s are contiguous
    array w = ensure_row_contiguous_matrix(inputs[1], d, s);
    array scales = ensure_row_contiguous_matrix(inputs[2], d, s);

    bool non_batched = w.ndim() == 2;
    int K = x.shape(-1);
    int M = non_batched ? x.size() / K : x.shape(-2);
    int N = out.shape(-1);
    dispatch_qmv(
        xhat,
        w,
        scales,
        std::nullopt,
        out,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        mode);
    return;
  } else {
    throw std::runtime_error("[QQMatmul] NYI for the general case");
  }
}

void fast::Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& w_pre = inputs[0];
  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& compute_encoder = metal::get_command_encoder(s);

  auto w = ensure_row_contiguous(w_pre, d, s);
  if (dequantize_) {
    auto scales = ensure_row_contiguous(inputs[1], d, s);
    if (mode_ == QuantizationMode::Affine) {
      auto biases = ensure_row_contiguous(inputs[2], d, s);
      compute_encoder.set_input_array(biases, 2);
    }
    compute_encoder.set_input_array(w, 0);
    compute_encoder.set_input_array(scales, 1);
    compute_encoder.set_output_array(out, 3);
  } else {
    auto& scales = outputs[1];
    scales.set_data(allocator::malloc(scales.nbytes()));
    if (mode_ == QuantizationMode::Affine) {
      auto& biases = outputs[2];
      biases.set_data(allocator::malloc(biases.nbytes()));
      compute_encoder.set_output_array(biases, 3);
    }
    compute_encoder.set_input_array(w, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder.set_output_array(scales, 2);
  }

  auto type_string = dequantize_ ? get_type_string(out.dtype())
                                 : get_type_string(w_pre.dtype());
  auto mode = quantization_mode_to_string(mode_);
  std::string kname;
  concatenate(
      kname,
      mode + (dequantize_ ? "_dequantize" : "_quantize"),
      "_",
      type_string,
      "_gs_",
      group_size_,
      "_b_",
      bits_);
  auto kernel = get_quantized_kernel_wrapped(
      d,
      kname,
      dequantize_ ? "dequantize" : "quantize",
      mode,
      type_string,
      group_size_,
      bits_);

  compute_encoder.set_compute_pipeline_state(kernel);

  // Treat uint32 as uint8 in kernel
  constexpr int uint8_per_uint32 = 4;
  constexpr int simd_size = 32;
  int packs_per_int = (bits_ == 3 || bits_ == 5) ? 8
      : bits_ == 6                               ? 4
                                                 : 8 / bits_;
  int per_thread =
      dequantize_ ? packs_per_int : std::max(group_size_ / simd_size, 1);
  size_t nthreads =
      dequantize_ ? out.size() / packs_per_int : w.size() / per_thread;

  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  if (thread_group_size > nthreads) {
    thread_group_size = nthreads;
  }
  auto group_dims = MTL::Size(thread_group_size, 1, 1);
  bool use_2d = nthreads > UINT_MAX;
  auto grid_shape = w.shape();
  if (dequantize_) {
    grid_shape.back() *= uint8_per_uint32;
  } else {
    grid_shape.back() /= per_thread;
  }
  MTL::Size grid_dims = use_2d ? get_2d_grid_dims(grid_shape, w.strides())
                               : MTL::Size(nthreads, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void fast::ConvertFP8::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& in = inputs[0];
  auto& out = outputs[0];
  unary_op_gpu(inputs, out, name(), stream());
}

} // namespace mlx::core
