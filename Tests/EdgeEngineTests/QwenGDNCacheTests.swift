// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Testing
@testable import EdgeEngine

@Test func qwenGDNCacheShapeUsesConvAndRecurrentStateDimensions() throws {
    let architecture = try makeGDNCacheArchitecture()

    let shape = try QwenGDNCacheShape.shape(for: architecture, layerIndex: 1)

    #expect(shape.layerIndex == 1)
    #expect(shape.convStateTensorShape == EdgeTensorShape([3, 10]))
    #expect(shape.convStateElementCount == 30)
    #expect(shape.recurrentStateTensorShape == EdgeTensorShape([2, 2, 3]))
    #expect(shape.recurrentStateElementCount == 12)
}

@Test func qwenGDNCacheShapesFollowAllGDNLayers() throws {
    let shapes = try QwenGDNCacheShape.shapes(for: makeGDNCacheArchitecture())

    #expect(shapes.map(\.layerIndex) == [1, 2])
    #expect(shapes.map(\.convStateTensorShape) == [
        EdgeTensorShape([3, 10]),
        EdgeTensorShape([3, 10]),
    ])
}

@Test func qwenGDNCacheShapeRejectsFullAttentionLayer() throws {
    do {
        _ = try QwenGDNCacheShape.shape(for: makeGDNCacheArchitecture(), layerIndex: 0)
        Issue.record("GDN cache shape should reject full-attention layers.")
    } catch QwenGDNCacheError.layerIsNotGDN(layerIndex: 0, kind: .fullAttention) {
        return
    }
    Issue.record("GDN cache shape threw the wrong error for a full-attention layer.")
}

@Test func qwenGDNCacheAllocatesAndResetsConvAndRecurrentState() throws {
    let runtime = try EdgeMetalRuntime()
    let cache = try QwenGDNCache(
        shape: QwenGDNCacheShape.shape(for: makeGDNCacheArchitecture(), layerIndex: 1),
        runtime: runtime
    )

    #expect(try cache.convState.readFloat32() == Array(repeating: 0, count: 30))
    #expect(try cache.recurrentState.readFloat32() == Array(repeating: 0, count: 12))
    cache.advanceToken()
    cache.advanceToken()
    #expect(cache.tokenPosition == 2)

    cache.convState.buffer.contents()
        .bindMemory(to: Float.self, capacity: 30)
        .update(repeating: 11, count: 30)
    cache.recurrentState.buffer.contents()
        .bindMemory(to: Float.self, capacity: 12)
        .update(repeating: 17, count: 12)
    cache.reset()

    #expect(cache.tokenPosition == 0)
    #expect(try cache.convState.readFloat32() == Array(repeating: 0, count: 30))
    #expect(try cache.recurrentState.readFloat32() == Array(repeating: 0, count: 12))
}

@Test func qwenGDNCacheUpdatesConvAndRecurrentStateBuffers() throws {
    let runtime = try EdgeMetalRuntime()
    let cache = try QwenGDNCache(
        shape: QwenGDNCacheShape.shape(for: makeGDNCacheArchitecture(), layerIndex: 1),
        runtime: runtime
    )
    let nextConvStateValues = (0..<30).map { Float($0 + 1) }
    let nextRecurrentStateValues = (0..<12).map { Float($0 + 101) }
    let nextConvState = try EdgeTensor(
        float32: nextConvStateValues,
        shape: EdgeTensorShape([3, 10]),
        runtime: runtime
    )
    let nextRecurrentState = try EdgeTensor(
        float32: nextRecurrentStateValues,
        shape: EdgeTensorShape([2, 2, 3]),
        runtime: runtime
    )

    try cache.update(
        nextConvState: nextConvState,
        nextRecurrentState: nextRecurrentState,
        tokenCount: 2
    )

    #expect(cache.tokenPosition == 2)
    #expect(try cache.convState.readFloat32() == nextConvStateValues)
    #expect(try cache.recurrentState.readFloat32() == nextRecurrentStateValues)
}

@Test func qwenGDNCacheUsesExecutorNextStateWithoutCopying() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let cache = try QwenGDNCache(
        shape: QwenGDNCacheShape.shape(for: makeGDNCacheArchitecture(), layerIndex: 1),
        runtime: runtime
    )
    let nextConvStateValues = (0..<30).map { Float($0 + 1) }
    let nextRecurrentStateValues = (0..<12).map { Float($0 + 101) }
    let nextConvState = try EdgeTensor(
        float32: nextConvStateValues,
        shape: EdgeTensorShape([3, 10]),
        runtime: runtime
    )
    let nextRecurrentState = try EdgeTensor(
        float32: nextRecurrentStateValues,
        shape: EdgeTensorShape([2, 2, 3]),
        runtime: runtime
    )

    try cache.update(
        nextConvState: nextConvState,
        nextRecurrentState: nextRecurrentState,
        executor: executor
    )

    #expect(cache.convState === nextConvState)
    #expect(cache.recurrentState === nextRecurrentState)
    #expect(cache.tokenPosition == 1)
    #expect(try cache.convState.readFloat32() == nextConvStateValues)
    #expect(try cache.recurrentState.readFloat32() == nextRecurrentStateValues)
}

@Test func qwenGDNCacheRejectsInvalidNextStateShapes() throws {
    let runtime = try EdgeMetalRuntime()
    let cache = try QwenGDNCache(
        shape: QwenGDNCacheShape.shape(for: makeGDNCacheArchitecture(), layerIndex: 1),
        runtime: runtime
    )
    let wrongConvState = try EdgeTensor(
        float32: Array(repeating: Float.zero, count: 20),
        shape: EdgeTensorShape([2, 10]),
        runtime: runtime
    )
    let recurrentState = try EdgeTensor(
        float32: Array(repeating: Float.zero, count: 12),
        shape: EdgeTensorShape([2, 2, 3]),
        runtime: runtime
    )

    do {
        try cache.update(nextConvState: wrongConvState, nextRecurrentState: recurrentState)
        Issue.record("GDN cache should reject a mismatched next conv state.")
    } catch QwenGDNCacheError.invalidNextConvStateShape(expected: [3, 10], actual: [2, 10]) {
        return
    }
    Issue.record("GDN cache threw the wrong error for a mismatched next conv state.")
}

private func makeGDNCacheArchitecture() throws -> QwenHybridArchitecture {
    try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 128,
        hiddenSize: 8,
        intermediateSize: 32,
        attentionHeadCount: 2,
        keyValueHeadCount: 1,
        linearValueHeadCount: 2,
        linearKeyHeadCount: 1,
        linearKeyHeadDimension: 3,
        linearValueHeadDimension: 2,
        linearConvKernelSize: 4,
        contextLength: 64,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000,
        layerKinds: [.fullAttention, .gdn, .gdn]
    )
}
