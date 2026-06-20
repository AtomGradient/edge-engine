// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Testing
@testable import EdgeEngine

@Test func qwenHybridArchitectureRejectsFullAttentionOnlyPlans() throws {
    var rejectedFAOnlyPlan = false
    do {
        _ = try QwenHybridArchitecture(
            family: .qwen35,
            vocabularySize: 151_936,
            hiddenSize: 2_048,
            intermediateSize: 5_504,
            attentionHeadCount: 16,
            keyValueHeadCount: 4,
            contextLength: 32_768,
            rmsNormEpsilon: 1e-6,
            ropeTheta: 1_000_000,
            layerKinds: [.fullAttention, .fullAttention, .fullAttention]
        )
        Issue.record("Qwen3.5/3.6 layer plan must not be FA-only.")
    } catch QwenArchitectureError.missingGDNLayers {
        rejectedFAOnlyPlan = true
    }
    #expect(rejectedFAOnlyPlan)
}

@Test func qwenHybridArchitectureExposesFullAttentionAndGDNLayerIndices() throws {
    let architecture = try QwenHybridArchitecture(
        family: .qwen36,
        vocabularySize: 151_936,
        hiddenSize: 3_072,
        intermediateSize: 8_192,
        attentionHeadCount: 24,
        keyValueHeadCount: 8,
        contextLength: 65_536,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 1_000_000,
        layerKinds: [.fullAttention, .gdn, .fullAttention, .gdn]
    )

    #expect(architecture.layerCount == 4)
    #expect(architecture.usesHybridAttentionAndGDN)
    #expect(architecture.fullAttentionLayerIndices == [0, 2])
    #expect(architecture.gdnLayerIndices == [1, 3])
}

@Test func qwenHybridArchitectureUsesExplicitAttentionHeadDimension() throws {
    let architecture = try QwenHybridArchitecture(
        family: .qwen36,
        vocabularySize: 248_320,
        hiddenSize: 5_120,
        intermediateSize: 17_408,
        attentionHeadCount: 24,
        keyValueHeadCount: 4,
        headDimension: 256,
        contextLength: 262_144,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000_000,
        layerKinds: [.gdn, .gdn, .gdn, .fullAttention]
    )

    #expect(architecture.attentionHeadDimension == 256)
    #expect(architecture.attentionHiddenSize == 6_144)
    #expect(architecture.queryHiddenSize == 6_144)
    #expect(architecture.queryGateHiddenSize == 6_144)
    #expect(architecture.queryProjectionHiddenSize == 12_288)
    #expect(architecture.keyValueHiddenSize == 1_024)
    #expect(architecture.rotaryDimension == 64)
}

@Test func qwenHybridArchitectureExposesLinearAttentionDimensions() throws {
    let architecture = try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 248_320,
        hiddenSize: 4_096,
        intermediateSize: 12_288,
        attentionHeadCount: 16,
        keyValueHeadCount: 4,
        headDimension: 256,
        linearValueHeadCount: 64,
        linearKeyHeadCount: 16,
        linearKeyHeadDimension: 192,
        linearValueHeadDimension: 128,
        linearConvKernelSize: 4,
        contextLength: 262_144,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000_000,
        layerKinds: [.gdn, .gdn, .gdn, .fullAttention]
    )

    #expect(architecture.linearKeyHiddenSize == 3_072)
    #expect(architecture.linearValueHiddenSize == 8_192)
    #expect(architecture.linearQKVHiddenSize == 14_336)
    #expect(architecture.linearConvHiddenSize == 14_336)
    #expect(architecture.linearConvKernelSize == 4)
}

@Test func qwenFrogJumpPlanComputesModerateV2Layers() throws {
    let qwen9BPattern = Array(
        repeating: [
            QwenHybridLayerKind.gdn,
            .gdn,
            .gdn,
            .fullAttention,
        ],
        count: 8
    ).flatMap { $0 }
    let qwen9B = try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 248_320,
        hiddenSize: 4_096,
        intermediateSize: 12_288,
        attentionHeadCount: 16,
        keyValueHeadCount: 4,
        headDimension: 256,
        contextLength: 262_144,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000_000,
        layerKinds: qwen9BPattern
    )
    let qwen9BPlan = QwenFrogJumpPlan.compute(
        architecture: qwen9B,
        requestedEnabled: true,
        thinkingEnabled: false
    )
    #expect(qwen9BPlan.enabled)
    #expect(qwen9BPlan.skipLayers == [8, 9, 12, 13, 16, 17])
    #expect(qwen9BPlan.layerMask == 0x0003_3300)

    let qwen35BPattern = Array(
        repeating: [
            QwenHybridLayerKind.gdn,
            .gdn,
            .gdn,
            .fullAttention,
        ],
        count: 10
    ).flatMap { $0 }
    let qwen35B = try QwenHybridArchitecture(
        family: .qwen36,
        vocabularySize: 248_320,
        hiddenSize: 5_120,
        intermediateSize: 17_408,
        attentionHeadCount: 24,
        keyValueHeadCount: 4,
        headDimension: 256,
        contextLength: 262_144,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000_000,
        layerKinds: qwen35BPattern
    )
    let qwen35BPlan = QwenFrogJumpPlan.compute(
        architecture: qwen35B,
        requestedEnabled: true,
        thinkingEnabled: false
    )
    #expect(qwen35BPlan.enabled)
    #expect(qwen35BPlan.skipLayers == [8, 9, 12, 13, 16, 17, 20, 21])
}

@Test func qwenFrogJumpPlanStaysDisabledForThinkingOrNonFourIntervalPlans() throws {
    let pattern = Array(
        repeating: [
            QwenHybridLayerKind.gdn,
            .gdn,
            .gdn,
            .fullAttention,
        ],
        count: 8
    ).flatMap { $0 }
    let architecture = try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 248_320,
        hiddenSize: 4_096,
        intermediateSize: 12_288,
        attentionHeadCount: 16,
        keyValueHeadCount: 4,
        headDimension: 256,
        contextLength: 262_144,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000_000,
        layerKinds: pattern
    )
    #expect(!QwenFrogJumpPlan.compute(
        architecture: architecture,
        requestedEnabled: true,
        thinkingEnabled: true
    ).enabled)
    #expect(!QwenFrogJumpPlan.compute(
        architecture: architecture,
        requestedEnabled: false,
        thinkingEnabled: false
    ).enabled)
    #expect(!QwenFrogJumpPlan.compute(layerKinds: [.gdn, .gdn, .fullAttention, .gdn]).enabled)
}

@Test func qwenHybridArchitectureValidatesQuantizationProfile() throws {
    let architecture = try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 248_320,
        hiddenSize: 2_560,
        intermediateSize: 9_216,
        attentionHeadCount: 16,
        keyValueHeadCount: 4,
        headDimension: 256,
        contextLength: 262_144,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000_000,
        quantization: QwenQuantizationProfile(groupSize: 64, bits: 8),
        layerKinds: [.gdn, .gdn, .gdn, .fullAttention]
    )

    #expect(architecture.quantization == QwenQuantizationProfile(groupSize: 64, bits: 8))

    do {
        _ = try QwenHybridArchitecture(
            family: .qwen35,
            vocabularySize: 248_320,
            hiddenSize: 2_560,
            intermediateSize: 9_216,
            attentionHeadCount: 16,
            keyValueHeadCount: 4,
            headDimension: 256,
            contextLength: 262_144,
            rmsNormEpsilon: 1e-6,
            ropeTheta: 10_000_000,
            quantization: QwenQuantizationProfile(groupSize: 64, bits: 7),
            layerKinds: [.gdn, .fullAttention]
        )
        Issue.record("Qwen architecture should reject unsupported quantization bit widths.")
    } catch QwenArchitectureError.unsupportedQuantizationBits(7) {
        return
    }
    Issue.record("Qwen architecture threw the wrong error for unsupported quantization bits.")
}

@Test func qwenHybridArchitectureRejectsInvalidLinearHeadMapping() throws {
    do {
        _ = try QwenHybridArchitecture(
            family: .qwen35,
            vocabularySize: 248_320,
            hiddenSize: 4_096,
            intermediateSize: 12_288,
            attentionHeadCount: 16,
            keyValueHeadCount: 4,
            headDimension: 256,
            linearValueHeadCount: 63,
            linearKeyHeadCount: 16,
            linearKeyHeadDimension: 192,
            linearValueHeadDimension: 128,
            contextLength: 262_144,
            rmsNormEpsilon: 1e-6,
            ropeTheta: 10_000_000,
            layerKinds: [.gdn, .fullAttention]
        )
        Issue.record("Qwen linear value heads must be divisible by linear key heads.")
    } catch QwenArchitectureError.linearValueHeadsNotDivisibleByKeyHeads(63, 16) {
        return
    }
    Issue.record("Qwen architecture threw the wrong error for invalid linear head mapping.")
}

@Test func qwenHybridArchitectureDecodesLegacyPayloadWithoutLinearFields() throws {
    let json = """
    {
      "family": "qwen35",
      "vocabularySize": 128,
      "hiddenSize": 8,
      "intermediateSize": 32,
      "attentionHeadCount": 2,
      "keyValueHeadCount": 1,
      "contextLength": 64,
      "rmsNormEpsilon": 1e-6,
      "ropeTheta": 10000,
      "partialRotaryFactor": 0.25,
      "layerPlan": [
        { "index": 0, "kind": "fullAttention" },
        { "index": 1, "kind": "gdn" }
      ]
    }
    """

    let architecture = try JSONDecoder().decode(
        QwenHybridArchitecture.self,
        from: try #require(json.data(using: .utf8))
    )

    #expect(architecture.linearValueHeadCount == 1)
    #expect(architecture.linearKeyHeadCount == 1)
    #expect(architecture.linearKeyHeadDimension == 4)
    #expect(architecture.linearValueHeadDimension == 4)
    #expect(architecture.linearConvKernelSize == 4)
}
