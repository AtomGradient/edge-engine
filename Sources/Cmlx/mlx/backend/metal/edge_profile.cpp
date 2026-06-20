#include "mlx/backend/metal/edge_profile.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

namespace mlx::core::metal {
namespace {

std::mutex profile_mutex;
EdgeMetalProfileSnapshot profile_total;
uint64_t next_command_buffer_sequence = 1;
thread_local const char* current_primitive_name = nullptr;

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

size_t env_top_n(size_t fallback) {
  const char* value = std::getenv("EDGE_CMLX_METAL_PROFILE_TOP_N");
  if (value == nullptr || value[0] == '\0') {
    value = std::getenv("CMLX_METAL_PROFILE_TOP_N");
  }
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || parsed == 0) {
    return fallback;
  }
  return std::min<size_t>(static_cast<size_t>(parsed), 128);
}

const char* profile_env_value() {
  const char* value = std::getenv("EDGE_CMLX_METAL_PROFILE");
  if (value == nullptr || value[0] == '\0') {
    value = std::getenv("CMLX_METAL_PROFILE");
  }
  return value;
}

const char* completion_probe_env_value() {
  const char* value = std::getenv("EDGE_CMLX_METAL_COMPLETION_PROBE");
  if (value == nullptr || value[0] == '\0') {
    value = std::getenv("CMLX_METAL_COMPLETION_PROBE");
  }
  return value;
}

const char* command_buffer_trace_env_value() {
  const char* value = std::getenv("EDGE_CMLX_METAL_PROFILE_CB_TRACE");
  if (value == nullptr || value[0] == '\0') {
    value = std::getenv("CMLX_METAL_PROFILE_CB_TRACE");
  }
  return value;
}

double command_buffer_slow_ms() {
  const char* value = std::getenv("EDGE_CMLX_METAL_PROFILE_SLOW_CB_MS");
  if (value == nullptr || value[0] == '\0') {
    value = std::getenv("CMLX_METAL_PROFILE_SLOW_CB_MS");
  }
  if (value == nullptr || value[0] == '\0') {
    return 100.0;
  }
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value || parsed <= 0.0) {
    return 100.0;
  }
  return parsed;
}

bool command_buffer_trace_enabled() {
  static const bool enabled = env_truthy(command_buffer_trace_env_value());
  return enabled;
}

EdgeMetalProfilePrimitiveCounters primitive_delta(
    const EdgeMetalProfilePrimitiveCounters& before,
    const EdgeMetalProfilePrimitiveCounters& after) {
  return EdgeMetalProfilePrimitiveCounters{
      after.evals >= before.evals ? after.evals - before.evals : 0,
      after.dispatches >= before.dispatches
          ? after.dispatches - before.dispatches
          : 0,
      after.cpu_ms >= before.cpu_ms ? after.cpu_ms - before.cpu_ms : 0.0};
}

} // namespace

bool edge_metal_profile_enabled() {
  static const bool enabled = env_truthy(profile_env_value());
  return enabled;
}

bool edge_metal_completion_probe_enabled() {
  // Default-on production fix for iPad M3 command-buffer freeze observed around
  // T24-T27 without any completion handler; empty handler runs reached T100+.
  // Set EDGE_CMLX_METAL_COMPLETION_PROBE=0 only for debug/control runs.
  static const bool enabled = !env_falsey(completion_probe_env_value());
  return enabled;
}

EdgeMetalProfilePrimitiveScope::EdgeMetalProfilePrimitiveScope(
    const char* name)
    : enabled_(edge_metal_profile_enabled()) {
  if (!enabled_) {
    return;
  }
  previous_ = current_primitive_name;
  current_primitive_name = name == nullptr ? "unknown" : name;
}

EdgeMetalProfilePrimitiveScope::~EdgeMetalProfilePrimitiveScope() {
  if (enabled_) {
    current_primitive_name = previous_;
  }
}

void edge_metal_profile_reset() {
  if (!edge_metal_profile_enabled()) {
    return;
  }
  std::lock_guard lock(profile_mutex);
  profile_total = {};
  profile_total.enabled = true;
}

void edge_metal_profile_record_primitive_eval(
    const char* name,
    double cpu_ms) {
  if (!edge_metal_profile_enabled()) {
    return;
  }
  const std::string primitive_name =
      name == nullptr || name[0] == '\0' ? "unknown" : name;
  std::lock_guard lock(profile_mutex);
  profile_total.enabled = true;
  profile_total.primitive_evals += 1;
  profile_total.primitive_cpu_ms += cpu_ms;
  auto& bucket = profile_total.primitives[primitive_name];
  bucket.evals += 1;
  bucket.cpu_ms += cpu_ms;
}

void edge_metal_profile_record_dispatch() {
  if (!edge_metal_profile_enabled()) {
    return;
  }
  const std::string primitive_name =
      current_primitive_name == nullptr || current_primitive_name[0] == '\0'
      ? "unknown"
      : current_primitive_name;
  std::lock_guard lock(profile_mutex);
  profile_total.enabled = true;
  profile_total.dispatches += 1;
  profile_total.primitives[primitive_name].dispatches += 1;
}

uint64_t edge_metal_profile_record_command_buffer_commit(
    int ops,
    size_t bytes) {
  if (!edge_metal_profile_enabled()) {
    return 0;
  }
  const uint64_t committed_ops = ops > 0 ? static_cast<uint64_t>(ops) : 0;
  const uint64_t committed_bytes = static_cast<uint64_t>(bytes);
  uint64_t sequence = 0;
  std::lock_guard lock(profile_mutex);
  sequence = next_command_buffer_sequence++;
  profile_total.enabled = true;
  profile_total.command_buffers += 1;
  if (committed_ops == 0 && committed_bytes == 0) {
    profile_total.empty_command_buffers += 1;
  }
  profile_total.committed_ops += committed_ops;
  profile_total.committed_bytes += committed_bytes;
  profile_total.max_ops_per_command_buffer =
      std::max(profile_total.max_ops_per_command_buffer, committed_ops);
  profile_total.max_bytes_per_command_buffer =
      std::max(profile_total.max_bytes_per_command_buffer, committed_bytes);
  profile_total.command_buffer_ops_samples.push_back(committed_ops);
  profile_total.command_buffer_bytes_samples.push_back(committed_bytes);
  if (command_buffer_trace_enabled()) {
    std::fprintf(
        stderr,
        "[CmlxMetal] COMMAND_BUFFER_COMMIT id=%llu ops=%llu bytes=%llu\n",
        static_cast<unsigned long long>(sequence),
        static_cast<unsigned long long>(committed_ops),
        static_cast<unsigned long long>(committed_bytes));
  }
  return sequence;
}

void edge_metal_profile_record_command_buffer_completion(
    uint64_t sequence,
    int ops,
    size_t bytes,
    double elapsed_ms,
    int status,
    bool is_error,
    const char* error_message) {
  if (!edge_metal_profile_enabled() || sequence == 0) {
    return;
  }
  const uint64_t committed_ops = ops > 0 ? static_cast<uint64_t>(ops) : 0;
  const uint64_t committed_bytes = static_cast<uint64_t>(bytes);
  {
    std::lock_guard lock(profile_mutex);
    profile_total.enabled = true;
    profile_total.command_buffer_completions += 1;
    if (is_error) {
      profile_total.command_buffer_errors += 1;
    }
    profile_total.command_buffer_completion_ms += elapsed_ms;
    profile_total.max_command_buffer_completion_ms =
        std::max(profile_total.max_command_buffer_completion_ms, elapsed_ms);
    profile_total.last_command_buffer_completion_ms = elapsed_ms;
    profile_total.last_completed_command_buffer = sequence;
    profile_total.command_buffer_completion_ms_samples.push_back(elapsed_ms);
  }
  if (command_buffer_trace_enabled() ||
      is_error ||
      elapsed_ms >= command_buffer_slow_ms()) {
    std::fprintf(
        stderr,
        "[CmlxMetal] COMMAND_BUFFER_COMPLETE id=%llu ops=%llu bytes=%llu "
        "elapsed_ms=%.3f status=%d%s%s\n",
        static_cast<unsigned long long>(sequence),
        static_cast<unsigned long long>(committed_ops),
        static_cast<unsigned long long>(committed_bytes),
        elapsed_ms,
        status,
        error_message != nullptr && error_message[0] != '\0' ? " error=" : "",
        error_message != nullptr && error_message[0] != '\0' ? error_message : "");
  }
}

void edge_metal_profile_record_skipped_empty_commit() {
  if (!edge_metal_profile_enabled()) {
    return;
  }
  std::lock_guard lock(profile_mutex);
  profile_total.enabled = true;
  profile_total.skipped_empty_commits += 1;
}

EdgeMetalProfileSnapshot edge_metal_profile_snapshot() {
  if (!edge_metal_profile_enabled()) {
    return {};
  }
  std::lock_guard lock(profile_mutex);
  auto snapshot = profile_total;
  snapshot.enabled = true;
  return snapshot;
}

EdgeMetalProfileSnapshot edge_metal_profile_delta(
    const EdgeMetalProfileSnapshot& before,
    const EdgeMetalProfileSnapshot& after) {
  if (!after.enabled) {
    return {};
  }
  EdgeMetalProfileSnapshot delta;
  delta.enabled = true;
  delta.primitive_evals = after.primitive_evals >= before.primitive_evals
      ? after.primitive_evals - before.primitive_evals
      : 0;
  delta.dispatches = after.dispatches >= before.dispatches
      ? after.dispatches - before.dispatches
      : 0;
  delta.command_buffers = after.command_buffers >= before.command_buffers
      ? after.command_buffers - before.command_buffers
      : 0;
  delta.command_buffer_completions =
      after.command_buffer_completions >= before.command_buffer_completions
      ? after.command_buffer_completions - before.command_buffer_completions
      : 0;
  delta.command_buffer_errors =
      after.command_buffer_errors >= before.command_buffer_errors
      ? after.command_buffer_errors - before.command_buffer_errors
      : 0;
  delta.empty_command_buffers =
      after.empty_command_buffers >= before.empty_command_buffers
      ? after.empty_command_buffers - before.empty_command_buffers
      : 0;
  delta.skipped_empty_commits =
      after.skipped_empty_commits >= before.skipped_empty_commits
      ? after.skipped_empty_commits - before.skipped_empty_commits
      : 0;
  delta.committed_ops = after.committed_ops >= before.committed_ops
      ? after.committed_ops - before.committed_ops
      : 0;
  delta.committed_bytes = after.committed_bytes >= before.committed_bytes
      ? after.committed_bytes - before.committed_bytes
      : 0;
  delta.primitive_cpu_ms = after.primitive_cpu_ms >= before.primitive_cpu_ms
      ? after.primitive_cpu_ms - before.primitive_cpu_ms
      : 0.0;
  delta.command_buffer_completion_ms =
      after.command_buffer_completion_ms >= before.command_buffer_completion_ms
      ? after.command_buffer_completion_ms - before.command_buffer_completion_ms
      : 0.0;
  delta.last_completed_command_buffer = after.last_completed_command_buffer;
  delta.last_command_buffer_completion_ms =
      after.last_command_buffer_completion_ms;

  if (after.command_buffer_ops_samples.size() >
      before.command_buffer_ops_samples.size()) {
    delta.command_buffer_ops_samples.assign(
        after.command_buffer_ops_samples.begin() +
            static_cast<std::ptrdiff_t>(
                before.command_buffer_ops_samples.size()),
        after.command_buffer_ops_samples.end());
    delta.max_ops_per_command_buffer = *std::max_element(
        delta.command_buffer_ops_samples.begin(),
        delta.command_buffer_ops_samples.end());
  }
  if (after.command_buffer_bytes_samples.size() >
      before.command_buffer_bytes_samples.size()) {
    delta.command_buffer_bytes_samples.assign(
        after.command_buffer_bytes_samples.begin() +
            static_cast<std::ptrdiff_t>(
                before.command_buffer_bytes_samples.size()),
        after.command_buffer_bytes_samples.end());
    delta.max_bytes_per_command_buffer = *std::max_element(
        delta.command_buffer_bytes_samples.begin(),
        delta.command_buffer_bytes_samples.end());
  }
  if (after.command_buffer_completion_ms_samples.size() >
      before.command_buffer_completion_ms_samples.size()) {
    delta.command_buffer_completion_ms_samples.assign(
        after.command_buffer_completion_ms_samples.begin() +
            static_cast<std::ptrdiff_t>(
                before.command_buffer_completion_ms_samples.size()),
        after.command_buffer_completion_ms_samples.end());
    delta.max_command_buffer_completion_ms = *std::max_element(
        delta.command_buffer_completion_ms_samples.begin(),
        delta.command_buffer_completion_ms_samples.end());
    delta.last_command_buffer_completion_ms =
        delta.command_buffer_completion_ms_samples.back();
  }

  for (const auto& entry : after.primitives) {
    const auto before_it = before.primitives.find(entry.first);
    const auto before_value = before_it == before.primitives.end()
        ? EdgeMetalProfilePrimitiveCounters{}
        : before_it->second;
    auto primitive = primitive_delta(before_value, entry.second);
    if (primitive.evals > 0 || primitive.dispatches > 0 ||
        primitive.cpu_ms > 0.0) {
      delta.primitives.emplace(entry.first, primitive);
    }
  }
  return delta;
}

std::string edge_metal_profile_summary(
    const EdgeMetalProfileSnapshot& snapshot,
    size_t top_n) {
  if (!snapshot.enabled) {
    return "disabled";
  }
  top_n = env_top_n(top_n);

  std::vector<
      std::pair<std::string, EdgeMetalProfilePrimitiveCounters>>
      primitives(snapshot.primitives.begin(), snapshot.primitives.end());
  std::sort(
      primitives.begin(),
      primitives.end(),
      [](const auto& lhs, const auto& rhs) {
        if (lhs.second.dispatches != rhs.second.dispatches) {
          return lhs.second.dispatches > rhs.second.dispatches;
        }
        if (lhs.second.evals != rhs.second.evals) {
          return lhs.second.evals > rhs.second.evals;
        }
        return lhs.first < rhs.first;
      });

  std::ostringstream out;
  out << "primitiveEvals=" << snapshot.primitive_evals
      << ",dispatches=" << snapshot.dispatches
      << ",commandBuffers=" << snapshot.command_buffers
      << ",commandBufferCompletions=" << snapshot.command_buffer_completions
      << ",commandBuffersInFlight="
      << (snapshot.command_buffers >= snapshot.command_buffer_completions
              ? snapshot.command_buffers - snapshot.command_buffer_completions
              : 0)
      << ",commandBufferErrors=" << snapshot.command_buffer_errors
      << ",emptyCommandBuffers=" << snapshot.empty_command_buffers
      << ",skippedEmptyCommits=" << snapshot.skipped_empty_commits
      << ",committedOps=" << snapshot.committed_ops
      << ",committedMB="
      << static_cast<double>(snapshot.committed_bytes) / (1024.0 * 1024.0)
      << ",maxOpsPerCB=" << snapshot.max_ops_per_command_buffer
      << ",maxMBPerCB="
      << static_cast<double>(snapshot.max_bytes_per_command_buffer) /
          (1024.0 * 1024.0)
      << ",cbCompletionAvgMs="
      << (snapshot.command_buffer_completions > 0
              ? snapshot.command_buffer_completion_ms /
                  static_cast<double>(snapshot.command_buffer_completions)
              : 0.0)
      << ",cbCompletionMaxMs="
      << snapshot.max_command_buffer_completion_ms
      << ",cbCompletionLastMs="
      << snapshot.last_command_buffer_completion_ms
      << ",lastCompletedCB=" << snapshot.last_completed_command_buffer
      << ",primitiveCpuMs=" << snapshot.primitive_cpu_ms
      << ",top=";
  if (primitives.empty() || top_n == 0) {
    out << "empty";
  } else {
    bool first = true;
    size_t emitted = 0;
    for (const auto& entry : primitives) {
      if (emitted >= top_n) {
        break;
      }
      if (!first) {
        out << "|";
      }
      first = false;
      emitted += 1;
      out << entry.first << ":eval=" << entry.second.evals
          << ",dispatch=" << entry.second.dispatches
          << ",cpuMs=" << entry.second.cpu_ms;
    }
  }
  return out.str();
}

} // namespace mlx::core::metal
