// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Testing
@testable import EdgeEngine

@Test func qwenKVCacheShapeUsesFullAttentionLayers() throws {
    let architecture = try smallHybridArchitecture()
    let shape = try QwenKVCacheShape.shape(for: architecture, layerIndex: 0, capacity: 8)

    #expect(shape.layerIndex == 0)
    #expect(shape.capacity == 8)
    #expect(shape.tensorShape == EdgeTensorShape([8, 2]))
}

@Test func qwenKVCacheShapeRejectsGDNLayers() throws {
    let architecture = try smallHybridArchitecture()
    var rejected = false

    do {
        _ = try QwenKVCacheShape.shape(for: architecture, layerIndex: 1, capacity: 8)
    } catch QwenKVCacheError.layerIsNotFullAttention(layerIndex: 1, kind: .gdn) {
        rejected = true
    }

    #expect(rejected)
}

@Test func qwenKVCacheAppendsRowsOnGPU() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let cache = try QwenKVCache(
        shape: try QwenKVCacheShape(
            layerIndex: 0,
            capacity: 3,
            keyValueHeadCount: 1,
            headDimension: 2
        ),
        runtime: runtime
    )
    let keys = try EdgeTensor(
        float32: [
            1, 2,
            3, 4,
        ],
        shape: EdgeTensorShape([2, 2]),
        runtime: runtime
    )
    let values = try EdgeTensor(
        float32: [
            10, 20,
            30, 40,
        ],
        shape: EdgeTensorShape([2, 2]),
        runtime: runtime
    )

    try cache.append(keys: keys, values: values, executor: executor)

    #expect(cache.tokenCount == 2)
    #expect(cache.remainingCapacity == 1)
    #expect(try cache.keys.readFloat32() == [
        1, 2,
        3, 4,
        0, 0,
    ])
    #expect(try cache.values.readFloat32() == [
        10, 20,
        30, 40,
        0, 0,
    ])
}

@Test func qwenDSRKVCacheEvictsToSinkHeavyAndRecentRows() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let cache = try QwenKVCache(
        shape: try QwenKVCacheShape(
            layerIndex: 0,
            capacity: 6,
            keyValueHeadCount: 1,
            headDimension: 1
        ),
        runtime: runtime,
        dsrPolicy: QwenDSRKVCachePolicy(
            maxSize: 4,
            heavyBudget: 1,
            recentBudget: 1,
            sinkSize: 1,
            evictionInterval: 1,
            scoreActivationRatio: 0,
            scoreDecay: 0.95
        )
    )
    let keys = try EdgeTensor(
        float32: [1, 2, 10, 3, 4],
        shape: EdgeTensorShape([5, 1]),
        runtime: runtime
    )
    let values = try EdgeTensor(
        float32: [10, 20, 100, 30, 40],
        shape: EdgeTensorShape([5, 1]),
        runtime: runtime
    )
    let query = try EdgeTensor(float32: [1], shape: EdgeTensorShape([1, 1]), runtime: runtime)

    try cache.append(keys: keys, values: values, executor: executor)
    try cache.updateDSRScores(
        query: query,
        executor: executor,
        queryHeadCount: 1,
        headDimension: 1
    )

    #expect(cache.tokenCount == 5)
    #expect(cache.activeTokenCount == 3)
    #expect(cache.remainingCapacity == 3)
    #expect(Array(try cache.keys.readFloat32().prefix(3)) == [1, 10, 4])
    #expect(Array(try cache.values.readFloat32().prefix(3)) == [10, 100, 40])
}

@Test func qwenDSRPolicyBuilderUsesSceneAndLayerScaling() throws {
    let architecture = try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 16,
        hiddenSize: 4,
        intermediateSize: 8,
        attentionHeadCount: 2,
        keyValueHeadCount: 1,
        headDimension: 2,
        contextLength: 16,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000,
        partialRotaryFactor: 1,
        layerKinds: [
            .fullAttention, .fullAttention, .gdn,
            .fullAttention, .fullAttention, .gdn,
            .fullAttention, .fullAttention,
        ]
    )

    let policies = try QwenDSRKVCachePolicy.layerAwarePolicies(
        for: architecture,
        maxSize: 14,
        sinkSize: 2,
        scene: .summary,
        evictionInterval: 16
    )

    #expect(policies.keys.sorted() == [0, 1, 3, 4, 6, 7])
    #expect(policies[0]?.heavyBudget == 5)
    #expect(policies[0]?.recentBudget == 2)
    #expect(policies[3]?.heavyBudget == 8)
    #expect(policies[3]?.recentBudget == 4)
    #expect(policies[6]?.heavyBudget == 8)
    #expect(policies[6]?.recentBudget == 4)
    #expect(policies[7]?.evictionInterval == 16)
}

@Test func qwenKVCacheRejectsOverflow() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let cache = try QwenKVCache(
        shape: try QwenKVCacheShape(
            layerIndex: 0,
            capacity: 1,
            keyValueHeadCount: 1,
            headDimension: 1
        ),
        runtime: runtime
    )
    let tensor = try EdgeTensor(float32: [1, 2], shape: EdgeTensorShape([2, 1]), runtime: runtime)
    var rejected = false

    do {
        try cache.append(keys: tensor, values: tensor, executor: executor)
    } catch QwenKVCacheError.cacheOverflow(capacity: 1, requestedTokenCount: 2) {
        rejected = true
    }

    #expect(rejected)
}

@Test func qwenKVCacheResetClearsBuffers() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let cache = try QwenKVCache(
        shape: try QwenKVCacheShape(
            layerIndex: 0,
            capacity: 2,
            keyValueHeadCount: 1,
            headDimension: 1
        ),
        runtime: runtime
    )
    let tensor = try EdgeTensor(float32: [7], shape: EdgeTensorShape([1, 1]), runtime: runtime)

    try cache.append(keys: tensor, values: tensor, executor: executor)
    cache.reset()

    #expect(cache.tokenCount == 0)
    #expect(try cache.keys.readFloat32() == [0, 0])
    #expect(try cache.values.readFloat32() == [0, 0])
}

private func smallHybridArchitecture() throws -> QwenHybridArchitecture {
    try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 16,
        hiddenSize: 4,
        intermediateSize: 8,
        attentionHeadCount: 2,
        keyValueHeadCount: 1,
        headDimension: 2,
        contextLength: 16,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000,
        partialRotaryFactor: 1,
        layerKinds: [.fullAttention, .gdn]
    )
}
