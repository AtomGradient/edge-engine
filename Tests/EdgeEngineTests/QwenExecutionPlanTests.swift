// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Testing
@testable import EdgeEngine

@Test func qwenExecutionPlanMapsHybridLayersToDispatchSteps() throws {
    let architecture = try QwenHybridArchitecture(
        family: .qwen36,
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

    let plan = try QwenExecutionPlan(architecture: architecture)

    #expect(plan.steps.map(\.layerIndex) == [0, 1, 2, 3])
    #expect(plan.fullAttentionSteps.map(\.layerIndex) == [0, 2])
    #expect(plan.gdnSteps.map(\.layerIndex) == [1, 3])
    #expect(try plan.step(layerIndex: 0).requiresKVCache)
    #expect(try plan.step(layerIndex: 1).requiresGDNState)
    #expect(plan.gdnStateShapes.map(\.layerIndex) == [1, 3])
    #expect(plan.gdnCacheShapes.map(\.layerIndex) == [1, 3])
    #expect(plan.gdnCacheShapes.map(\.convStateTensorShape) == [
        EdgeTensorShape([3, 12]),
        EdgeTensorShape([3, 12]),
    ])
}

@Test func qwenExecutionPlanRejectsUnknownLayerLookup() throws {
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
        layerKinds: [.fullAttention, .gdn]
    )
    let plan = try QwenExecutionPlan(architecture: architecture)

    var rejectedUnknownLayer = false
    do {
        _ = try plan.step(layerIndex: 99)
        Issue.record("Execution plan must reject unknown layer lookups.")
    } catch QwenExecutionPlanError.layerNotFound {
        rejectedUnknownLayer = true
    }
    #expect(rejectedUnknownLayer)
}
