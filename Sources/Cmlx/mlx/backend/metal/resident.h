// Copyright © 2024 Apple Inc.

#pragma once

#include <mutex>
#include <unordered_set>

#include <Metal/Metal.hpp>

namespace mlx::core::metal {

class ResidencySet {
 public:
  ResidencySet(MTL::Device* d);
  ~ResidencySet();

  ResidencySet(const ResidencySet&) = delete;
  ResidencySet& operator=(const ResidencySet&) = delete;

  const MTL::ResidencySet* mtl_residency_set() {
    return wired_set_.get();
  }

  void insert(MTL::Allocation* buf);
  void erase(MTL::Allocation* buf);

  void resize(size_t size);

 private:
  std::mutex mutex_;
  NS::SharedPtr<MTL::ResidencySet> wired_set_;
  std::unordered_set<const MTL::Allocation*> unwired_set_;
  size_t capacity_{0};
};

} // namespace mlx::core::metal
