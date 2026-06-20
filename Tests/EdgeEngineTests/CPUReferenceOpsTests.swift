// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Testing
@testable import EdgeEngine

@Test func cpuReferenceMatmulMatchesSmokeFixture() throws {
    let result = try CPUReferenceOps.matmul(
        [
            1, 2, 3,
            4, 5, 6,
        ],
        rows: 2,
        inner: 3,
        [
            7, 8,
            9, 10,
            11, 12,
        ],
        columns: 2
    )

    #expect(result == [58, 64, 139, 154])
}

@Test func cpuReferenceSoftmaxIsStable() throws {
    let result = try CPUReferenceOps.softmax([1, 2, 3])

    #expect(abs(result.reduce(0, +) - 1) < 1e-6)
    #expect(result[2] > result[1])
    #expect(result[1] > result[0])
}

@Test func cpuReferenceRMSNormMatchesManualCalculation() throws {
    let result = try CPUReferenceOps.rmsNorm([3, 4], weight: [1, 1], epsilon: 0)
    let expectedScale = Float(1.0 / Double(12.5).squareRoot())

    #expect(abs(result[0] - 3 * expectedScale) < 1e-6)
    #expect(abs(result[1] - 4 * expectedScale) < 1e-6)
}

@Test func cpuReferenceRMSNormByHeadNormalizesEachHeadIndependently() throws {
    let result = try CPUReferenceOps.rmsNormByHead(
        [
            3, 4, 0, 5,
            6, 8, -5, 0,
        ],
        weight: [1, 2],
        tokenCount: 2,
        headCount: 2,
        headDimension: 2,
        epsilon: 0
    )

    let firstHeadScale = Float(1.0 / Double(12.5).squareRoot())
    let secondHeadScale = Float(1.0 / Double(12.5).squareRoot())
    #expect(abs(result[0] - 3 * firstHeadScale) < 1e-6)
    #expect(abs(result[1] - 8 * firstHeadScale) < 1e-6)
    #expect(abs(result[2] - 0 * secondHeadScale) < 1e-6)
    #expect(abs(result[3] - 10 * secondHeadScale) < 1e-6)
}

@Test func cpuReferenceSwiGLUAppliesSiluGate() throws {
    let result = try CPUReferenceOps.swiglu(gate: [0, 1], up: [2, 3])

    #expect(result[0] == 0)
    #expect(result[1] > 2)
}

@Test func cpuReferenceGDNDepthwiseConvUpdatesRollingState() throws {
    let result = try CPUReferenceOps.gdnDepthwiseConv1D(
        [
            5, 6,
            7, 8,
        ],
        convState: [
            1, 2,
            3, 4,
        ],
        weights: [
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6,
        ],
        tokenCount: 2,
        channelCount: 2,
        kernelSize: 3
    )

    let expectedConvolution = [
        siluForCPUReferenceTests(2.2),
        siluForCPUReferenceTests(6.4),
        siluForCPUReferenceTests(3.4),
        siluForCPUReferenceTests(9.4),
    ]
    let error = try NumericComparison.maxAbsoluteError(result.activated, expectedConvolution)

    #expect(error < 1e-6)
    #expect(result.nextConvState == [
        5, 6,
        7, 8,
    ])
}

@Test func cpuReferenceAddMatchesElementwiseSum() throws {
    let result = try CPUReferenceOps.add([1, -2, 3], [4, 5, -6])

    #expect(result == [5, 3, -3])
}

@Test func cpuReferenceEmbeddingLookupCopiesRows() throws {
    let result = try CPUReferenceOps.embeddingLookup(
        tokenIds: [2, 0, 2],
        embeddings: [
            1, 2,
            3, 4,
            5, 6,
        ],
        vocabularySize: 3,
        hiddenSize: 2
    )

    #expect(result == [
        5, 6,
        1, 2,
        5, 6,
    ])
}

@Test func scaledDotProductAttentionAppliesCausalMask() throws {
    let output = try CPUReferenceOps.scaledDotProductAttention(
        query: [1, 1],
        key: [1, 2],
        value: [10, 20],
        tokenCount: 2,
        queryHeadCount: 1,
        keyValueHeadCount: 1,
        headDimension: 1
    )

    #expect(output[0] == 10)
    #expect(abs(output[1] - 17.310585) < 1e-5)
}

@Test func scaledDotProductAttentionSharesKVHeadsAcrossQueryGroups() throws {
    let output = try CPUReferenceOps.scaledDotProductAttention(
        query: [1, 2],
        key: [3],
        value: [5],
        tokenCount: 1,
        queryHeadCount: 2,
        keyValueHeadCount: 1,
        headDimension: 1
    )

    #expect(output == [5, 5])
}

@Test func scaledDotProductAttentionSupportsCachedKeyValues() throws {
    let output = try CPUReferenceOps.scaledDotProductAttention(
        query: [1, 1],
        key: [1, 2, 3],
        value: [10, 20, 30],
        queryTokenCount: 1,
        keyValueTokenCount: 3,
        queryPositionOffset: 2,
        queryHeadCount: 2,
        keyValueHeadCount: 1,
        headDimension: 1
    )
    let probabilities = try CPUReferenceOps.softmax([1, 2, 3])
    let expected = probabilities[0] * 10 + probabilities[1] * 20 + probabilities[2] * 30

    #expect(abs(output[0] - expected) < 1e-5)
    #expect(abs(output[1] - expected) < 1e-5)
}

@Test func cosineSimilarityReportsIdenticalVectors() throws {
    let similarity = try NumericComparison.cosineSimilarity([1, 2, 3], [1, 2, 3])

    #expect(abs(similarity - 1) < 1e-6)
}

@Test func maxAbsoluteErrorReportsLargestDifference() throws {
    let error = try NumericComparison.maxAbsoluteError([1, 2, 3], [1, 4, -1])

    #expect(error == 4)
}

private func siluForCPUReferenceTests(_ value: Float) -> Float {
    value / (1.0 + Foundation.exp(-value))
}
