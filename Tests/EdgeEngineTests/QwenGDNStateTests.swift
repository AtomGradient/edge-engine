// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Testing
@testable import EdgeEngine

@Test func qwenGDNStateShapesFollowHybridArchitectureGDNLayers() throws {
    let architecture = try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 128,
        hiddenSize: 8,
        intermediateSize: 32,
        attentionHeadCount: 2,
        keyValueHeadCount: 1,
        contextLength: 64,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000,
        layerKinds: [.fullAttention, .gdn, .fullAttention, .gdn]
    )

    let shapes = try QwenGDNStateShape.shapes(for: architecture)

    #expect(shapes.map(\.layerIndex) == [1, 3])
    #expect(shapes.map(\.tensorShape) == [EdgeTensorShape([1, 4, 4]), EdgeTensorShape([1, 4, 4])])
}

@Test func qwenGDNStateShapesUseLinearAttentionDimensions() throws {
    let architecture = try QwenHybridArchitecture(
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
        layerKinds: [.fullAttention, .gdn]
    )

    let shapes = try QwenGDNStateShape.shapes(for: architecture)
    let expected = try QwenGDNStateShape(
        layerIndex: 1,
        valueHeadCount: 2,
        valueHeadDimension: 2,
        keyHeadDimension: 3
    )

    #expect(shapes == [expected])
    #expect(shapes[0].tensorShape == EdgeTensorShape([2, 2, 3]))
    #expect(shapes[0].elementCount == 12)
}

@Test func qwenGDNStateAllocatesZeroedRecurrentStorageAndResetsPosition() throws {
    let runtime = try EdgeMetalRuntime()
    let shape = try QwenGDNStateShape(layerIndex: 1, headCount: 1, headDimension: 2)
    let state = try QwenGDNState(shape: shape, runtime: runtime)

    #expect(try state.tensor.readFloat32() == [0, 0, 0, 0])
    state.advanceToken()
    state.advanceToken()
    #expect(state.tokenPosition == 2)

    state.tensor.buffer.contents()
        .bindMemory(to: Float.self, capacity: 4)
        .update(repeating: 42, count: 4)
    state.reset()

    #expect(state.tokenPosition == 0)
    #expect(try state.tensor.readFloat32() == [0, 0, 0, 0])
}
