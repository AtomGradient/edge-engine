// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Testing
@testable import EdgeEngine

@Test func qwenHybridDecoderCachesAllocateByLayerKind() throws {
    let runtime = try EdgeMetalRuntime()
    let architecture = try makeHybridDecoderCachesArchitecture()
    let caches = try QwenHybridDecoderCaches(
        architecture: architecture,
        runtime: runtime,
        kvCapacity: 12
    )

    #expect(caches.kvLayerIndices == [0, 2])
    #expect(caches.gdnLayerIndices == [1, 3])
    #expect(try caches.kvCache(layerIndex: 0).shape.tensorShape == EdgeTensorShape([12, 4]))
    #expect(try caches.kvCache(layerIndex: 2).shape.tensorShape == EdgeTensorShape([12, 4]))
    #expect(try caches.gdnCache(layerIndex: 1).shape.convStateTensorShape == EdgeTensorShape([3, 12]))
    #expect(try caches.gdnCache(layerIndex: 3).shape.recurrentStateTensorShape == EdgeTensorShape([1, 4, 4]))
}

@Test func qwenHybridDecoderCachesRejectWrongLayerLookup() throws {
    let caches = try QwenHybridDecoderCaches(
        architecture: makeHybridDecoderCachesArchitecture(),
        runtime: try EdgeMetalRuntime()
    )
    var rejectedKV = false
    var rejectedGDN = false

    do {
        _ = try caches.kvCache(layerIndex: 1)
    } catch QwenHybridDecoderCachesError.missingKVCache(layerIndex: 1) {
        rejectedKV = true
    }

    do {
        _ = try caches.gdnCache(layerIndex: 0)
    } catch QwenHybridDecoderCachesError.missingGDNCache(layerIndex: 0) {
        rejectedGDN = true
    }

    #expect(rejectedKV)
    #expect(rejectedGDN)
}

@Test func qwenHybridDecoderCachesResetAllLayerCaches() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let caches = try QwenHybridDecoderCaches(
        architecture: makeHybridDecoderCachesArchitecture(),
        runtime: runtime,
        kvCapacity: 4
    )
    let kvCache = try caches.kvCache(layerIndex: 0)
    let gdnCache = try caches.gdnCache(layerIndex: 1)
    let kvTensor = try EdgeTensor(
        float32: [
            1, 2, 3, 4,
            5, 6, 7, 8,
        ],
        shape: EdgeTensorShape([2, 4]),
        runtime: runtime
    )
    let nextConvState = try EdgeTensor(
        float32: Array(repeating: Float(11), count: 36),
        shape: EdgeTensorShape([3, 12]),
        runtime: runtime
    )
    let nextRecurrentState = try EdgeTensor(
        float32: Array(repeating: Float(17), count: 16),
        shape: EdgeTensorShape([1, 4, 4]),
        runtime: runtime
    )

    try kvCache.append(keys: kvTensor, values: kvTensor, executor: executor)
    try gdnCache.update(
        nextConvState: nextConvState,
        nextRecurrentState: nextRecurrentState,
        tokenCount: 2
    )
    caches.reset()

    #expect(kvCache.tokenCount == 0)
    #expect(gdnCache.tokenPosition == 0)
    #expect(try kvCache.keys.readFloat32() == Array(repeating: Float.zero, count: 16))
    #expect(try kvCache.values.readFloat32() == Array(repeating: Float.zero, count: 16))
    #expect(try gdnCache.convState.readFloat32() == Array(repeating: Float.zero, count: 36))
    #expect(try gdnCache.recurrentState.readFloat32() == Array(repeating: Float.zero, count: 16))
}

@Test func qwenHybridDecoderCachesReportsSharedTokenPosition() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let caches = try QwenHybridDecoderCaches(
        architecture: makeHybridDecoderCachesArchitecture(),
        runtime: runtime,
        kvCapacity: 4
    )
    let kvTensor = try EdgeTensor(
        float32: [
            1, 2, 3, 4,
            5, 6, 7, 8,
        ],
        shape: EdgeTensorShape([2, 4]),
        runtime: runtime
    )
    let nextConvState = try EdgeTensor(
        float32: Array(repeating: Float(11), count: 36),
        shape: EdgeTensorShape([3, 12]),
        runtime: runtime
    )
    let nextRecurrentState = try EdgeTensor(
        float32: Array(repeating: Float(17), count: 16),
        shape: EdgeTensorShape([1, 4, 4]),
        runtime: runtime
    )

    try caches.kvCache(layerIndex: 0).append(keys: kvTensor, values: kvTensor, executor: executor)
    try caches.kvCache(layerIndex: 2).append(keys: kvTensor, values: kvTensor, executor: executor)
    try caches.gdnCache(layerIndex: 1).update(
        nextConvState: nextConvState,
        nextRecurrentState: nextRecurrentState,
        tokenCount: 2
    )
    try caches.gdnCache(layerIndex: 3).update(
        nextConvState: nextConvState,
        nextRecurrentState: nextRecurrentState,
        tokenCount: 2
    )

    #expect(try caches.tokenPosition() == 2)
}

@Test func qwenHybridDecoderCachesRejectMismatchedTokenPositions() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let caches = try QwenHybridDecoderCaches(
        architecture: makeHybridDecoderCachesArchitecture(),
        runtime: runtime,
        kvCapacity: 4
    )
    let kvTensor = try EdgeTensor(
        float32: [
            1, 2, 3, 4,
            5, 6, 7, 8,
        ],
        shape: EdgeTensorShape([2, 4]),
        runtime: runtime
    )
    var rejected = false

    try caches.kvCache(layerIndex: 0).append(keys: kvTensor, values: kvTensor, executor: executor)

    do {
        _ = try caches.tokenPosition()
    } catch QwenHybridDecoderCachesError.tokenPositionMismatch(
        expected: 2,
        actual: 0,
        layerIndex: 2
    ) {
        rejected = true
    }

    #expect(rejected)
}

private func makeHybridDecoderCachesArchitecture() throws -> QwenHybridArchitecture {
    try QwenHybridArchitecture(
        family: .qwen36,
        vocabularySize: 128,
        hiddenSize: 8,
        intermediateSize: 32,
        attentionHeadCount: 2,
        keyValueHeadCount: 1,
        headDimension: 4,
        linearValueHeadCount: 1,
        linearKeyHeadCount: 1,
        linearKeyHeadDimension: 4,
        linearValueHeadDimension: 4,
        linearConvKernelSize: 4,
        contextLength: 64,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000,
        layerKinds: [.fullAttention, .gdn, .fullAttention, .gdn]
    )
}
