// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Testing
@testable import EdgeEngine

@Test func q4QuantizationRoundTripsSmallValues() throws {
    let values: [Float] = [-1, -0.5, 0, 0.5, 1]
    let weights = try Q4WeightMatrix.quantizeSymmetric(
        values,
        rows: 1,
        columns: 5,
        groupSize: 5
    )
    let dequantized = weights.dequantizedValues()

    let similarity = try NumericComparison.cosineSimilarity(values, dequantized)
    #expect(similarity > 0.99)
}

@Test func q4MatmulUsesDequantizedWeights() throws {
    let lhs: [Float] = [
        1, 2, 3,
        4, 5, 6,
    ]
    let denseWeights: [Float] = [
        7, 8,
        9, 10,
        11, 12,
    ]
    let weights = try Q4WeightMatrix.quantizeSymmetric(
        denseWeights,
        rows: 3,
        columns: 2,
        groupSize: 6
    )
    let result = try CPUReferenceOps.q4Matmul(lhs, rows: 2, inner: 3, weights: weights)
    let expected = try CPUReferenceOps.matmul(
        lhs,
        rows: 2,
        inner: 3,
        weights.dequantizedValues(),
        columns: 2
    )

    #expect(result == expected)
}
