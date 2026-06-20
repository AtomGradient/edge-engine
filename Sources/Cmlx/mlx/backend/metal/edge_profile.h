#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx::core::metal {

struct EdgeMetalProfilePrimitiveCounters {
  uint64_t evals = 0;
  uint64_t dispatches = 0;
  double cpu_ms = 0.0;
};

struct EdgeMetalProfileSnapshot {
  bool enabled = false;
  uint64_t primitive_evals = 0;
  uint64_t dispatches = 0;
  uint64_t command_buffers = 0;
  uint64_t command_buffer_completions = 0;
  uint64_t command_buffer_errors = 0;
  uint64_t empty_command_buffers = 0;
  uint64_t skipped_empty_commits = 0;
  uint64_t committed_ops = 0;
  uint64_t committed_bytes = 0;
  uint64_t max_ops_per_command_buffer = 0;
  uint64_t max_bytes_per_command_buffer = 0;
  uint64_t last_completed_command_buffer = 0;
  double primitive_cpu_ms = 0.0;
  double command_buffer_completion_ms = 0.0;
  double max_command_buffer_completion_ms = 0.0;
  double last_command_buffer_completion_ms = 0.0;
  std::unordered_map<std::string, EdgeMetalProfilePrimitiveCounters> primitives;
  std::vector<uint64_t> command_buffer_ops_samples;
  std::vector<uint64_t> command_buffer_bytes_samples;
  std::vector<double> command_buffer_completion_ms_samples;
};

class EdgeMetalProfilePrimitiveScope {
 public:
  explicit EdgeMetalProfilePrimitiveScope(const char* name);
  ~EdgeMetalProfilePrimitiveScope();

  EdgeMetalProfilePrimitiveScope(const EdgeMetalProfilePrimitiveScope&) =
      delete;
  EdgeMetalProfilePrimitiveScope& operator=(
      const EdgeMetalProfilePrimitiveScope&) = delete;

 private:
  const char* previous_ = nullptr;
  bool enabled_ = false;
};

bool edge_metal_profile_enabled();
bool edge_metal_completion_probe_enabled();
void edge_metal_profile_reset();
void edge_metal_profile_record_primitive_eval(
    const char* name,
    double cpu_ms);
void edge_metal_profile_record_dispatch();
uint64_t edge_metal_profile_record_command_buffer_commit(
    int ops,
    size_t bytes);
void edge_metal_profile_record_command_buffer_completion(
    uint64_t sequence,
    int ops,
    size_t bytes,
    double elapsed_ms,
    int status,
    bool is_error,
    const char* error_message);
void edge_metal_profile_record_skipped_empty_commit();
EdgeMetalProfileSnapshot edge_metal_profile_snapshot();
EdgeMetalProfileSnapshot edge_metal_profile_delta(
    const EdgeMetalProfileSnapshot& before,
    const EdgeMetalProfileSnapshot& after);
std::string edge_metal_profile_summary(
    const EdgeMetalProfileSnapshot& snapshot,
    size_t top_n = 12);

} // namespace mlx::core::metal
