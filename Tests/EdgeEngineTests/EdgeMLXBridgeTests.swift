// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeEngine
import Darwin
import Foundation
import Testing
import XCTest

@Test func cmlxVendorMetadataIsExposed() {
    #expect(EdgeMLXBridge.isVendorPresent)
    #expect(EdgeMLXBridge.vendorVersion == "0.32.0")
    #expect(EdgeMLXBridge.vendorVersionNumeric > 0)
}

@Test func cmlxDefaultMetallibAvailabilityIsProbeable() {
    #expect(EdgeMLXBridge.isDefaultMetallibAvailable)
}

@Test func cmlxStateBoundaryProbeDistinguishesSiblingAndSliceState() throws {
    let result = try EdgeMLXBridge.runStateBoundaryProbe(steps: 4)

    assertCmlxStateBoundaryProbe(result, expectedSteps: 4)
}

@Test func cmlxStateBoundaryProbeRunsLongEnoughToCatchSmallGraphGrowth() throws {
    let result = try EdgeMLXBridge.runStateBoundaryProbe(steps: 100)

    assertCmlxStateBoundaryProbe(result, expectedSteps: 100)
}

@Test func cmlxCrossThreadStreamProbeEvaluatesLazyGraphFromWorkerThread() throws {
    try EdgeMLXBridge.runCrossThreadStreamProbe()
}

private func assertCmlxStateBoundaryProbe(
    _ result: EdgeMLXStateBoundaryProbeResult,
    expectedSteps: Int
) {
    #expect(result.syncSteps == expectedSteps)
    #expect(result.asyncSteps == expectedSteps)

    #expect(result.syncRecurrentAfterTokenEval.isAvailable)
    #expect(result.syncRecurrentAfterTokenEval.hasPrimitive == false)
    #expect(result.syncRecurrentAfterTokenEval.siblingCount == 0)

    #expect(result.syncCustomRecurrentAfterTokenEval.isAvailable)
    #expect(result.syncCustomRecurrentAfterTokenEval.hasPrimitive == false)
    #expect(result.syncCustomRecurrentAfterTokenEval.siblingCount == 0)

    #expect(result.syncConvAfterTokenEval.hasPrimitive)
    #expect(result.syncConvAfterTokenEval.isAvailable == false)

    #expect(result.syncRecurrentStopGradientAfterTokenEval.hasPrimitive)
    #expect(result.syncCustomRecurrentStopGradientAfterTokenEval.hasPrimitive)
    #expect(result.syncConvStopGradientAfterTokenEval.hasPrimitive)

    #expect(result.asyncRecurrentAfterSchedule.hasPrimitive == false)
    #expect(result.asyncCustomRecurrentAfterSchedule.hasPrimitive == false)
    #expect(result.asyncConvAfterSchedule.hasPrimitive)
    #expect(result.asyncRecurrentStopGradientAfterSchedule.hasPrimitive)
    #expect(result.asyncCustomRecurrentStopGradientAfterSchedule.hasPrimitive)
    #expect(result.asyncConvStopGradientAfterSchedule.hasPrimitive)

    #expect(result.asyncRecurrentAfterFinalEval.isAvailable)
    #expect(result.asyncRecurrentAfterFinalEval.hasPrimitive == false)
    #expect(result.asyncCustomRecurrentAfterFinalEval.isAvailable)
    #expect(result.asyncCustomRecurrentAfterFinalEval.hasPrimitive == false)
    #expect(result.asyncConvAfterFinalEval.isAvailable)
    #expect(result.asyncConvAfterFinalEval.hasPrimitive == false)
}

@Test func cmlxCPUFloatMatmulMatchesReference() throws {
    let lhs: [Float] = [
        1, 2, 3,
        -4, 5, 6,
    ]
    let rhs: [Float] = [
        7, -8,
        9, 10,
        -11, 12,
    ]

    let result = try EdgeMLXBridge.matmulFloat32CPU(
        lhs: lhs,
        rows: 2,
        inner: 3,
        rhs: rhs,
        columns: 2
    )
    let expected = try CPUReferenceOps.matmul(lhs, rows: 2, inner: 3, rhs, columns: 2)

    #expect(result.count == expected.count)
    for index in result.indices {
        #expect(abs(result[index] - expected[index]) < 1e-5)
    }
}

@Test func cmlxGPUFloatMatmulMatchesReference() throws {
    let lhs: [Float] = [
        1, 2, 3,
        -4, 5, 6,
    ]
    let rhs: [Float] = [
        7, -8,
        9, 10,
        -11, 12,
    ]

    let result = try EdgeMLXBridge.matmulFloat32GPU(
        lhs: lhs,
        rows: 2,
        inner: 3,
        rhs: rhs,
        columns: 2
    )
    let expected = try CPUReferenceOps.matmul(lhs, rows: 2, inner: 3, rhs, columns: 2)

    #expect(result.count == expected.count)
    for index in result.indices {
        #expect(abs(result[index] - expected[index]) < 1e-5)
    }
}

@Test func cmlxAffineQuantizedMatmulGPUValidatesEightBitVisionCandidateGroupSizes() throws {
    let rows = 3
    let inner = 64
    let columns = 5
    let lhs = (0..<(rows * inner)).map { Float(Int($0 % 19) - 9) / 7.0 }
    let rawRows: [[UInt32]] = (0..<columns).map { row in
        (0..<inner).map { column in
            UInt32((row * 17 + column * 11) % 256)
        }
    }

    for groupSize in [32, 64] {
        let groupCount = inner / groupSize
        let scales = (0..<(columns * groupCount)).map {
            Float(($0 % 13) + 1) / 255.0
        }
        let biases = (0..<(columns * groupCount)).map {
            Float(($0 % 7) - 3) / 128.0
        }
        let weights = try EdgeQuantizedTensor(
            shape: [columns, inner],
            packedShape: [columns, (inner * 8 + 31) / 32],
            scaleShape: [columns, groupCount],
            groupSize: groupSize,
            bits: 8,
            packedValues: edgeMLXBridgePackQuantizedRows(rawRows, bits: 8),
            scales: scales,
            biases: biases
        )

        let result = try EdgeMLXBridge.affineQuantizedMatmulFloat32GPU(
            lhs: lhs,
            rows: rows,
            inner: inner,
            weights: weights,
            transpose: true
        )
        let expected = edgeMLXBridgeDenseMatmulTransposed(
            lhs,
            rows: rows,
            inner: inner,
            weights: edgeMLXBridgeDenseAffineRows(
                rawRows: rawRows,
                scales: scales,
                biases: biases,
                groupSize: groupSize
            ),
            outputs: columns
        )
        let error = try NumericComparison.maxAbsoluteError(result, expected)
        #expect(error < 1e-3)
    }

    let unsupportedGroupSize = 16
    let unsupportedGroupCount = inner / unsupportedGroupSize
    let unsupportedWeights = try EdgeQuantizedTensor(
        shape: [columns, inner],
        packedShape: [columns, (inner * 8 + 31) / 32],
        scaleShape: [columns, unsupportedGroupCount],
        groupSize: unsupportedGroupSize,
        bits: 8,
        packedValues: edgeMLXBridgePackQuantizedRows(rawRows, bits: 8),
        scales: Array(repeating: 1.0 / 255.0, count: columns * unsupportedGroupCount),
        biases: Array(repeating: 0, count: columns * unsupportedGroupCount)
    )
    #expect(throws: EdgeMLXBridgeError.invalidShape) {
        _ = try EdgeMLXBridge.affineQuantizedMatmulFloat32GPU(
            lhs: lhs,
            rows: rows,
            inner: inner,
            weights: unsupportedWeights,
            transpose: true
        )
    }
}

@Test func cmlxGPUSoftmaxMatchesReferenceByRow() throws {
    let input: [Float] = [
        1, 2, 3,
        -4, 0, 5,
    ]

    let result = try EdgeMLXBridge.softmaxFloat32GPU(input, rows: 2, columns: 3)
    let expected = try CPUReferenceOps.softmax([1, 2, 3])
        + CPUReferenceOps.softmax([-4, 0, 5])

    #expect(result.count == expected.count)
    for index in result.indices {
        #expect(abs(result[index] - expected[index]) < 1e-5)
    }
}

private func edgeMLXBridgePackQuantizedRows(_ rows: [[UInt32]], bits: Int) -> [UInt32] {
    rows.flatMap { edgeMLXBridgePackQuantizedWords($0, bits: bits) }
}

private func edgeMLXBridgePackQuantizedWords(_ values: [UInt32], bits: Int) -> [UInt32] {
    var words = Array(repeating: UInt32.zero, count: (values.count * bits + 31) / 32)
    let mask = UInt32((1 << bits) - 1)
    for (index, value) in values.enumerated() {
        let bitOffset = index * bits
        let wordIndex = bitOffset / 32
        let shift = bitOffset % 32
        words[wordIndex] |= (value & mask) << UInt32(shift)
        if shift + bits > 32 {
            words[wordIndex + 1] |= (value & mask) >> UInt32(32 - shift)
        }
    }
    return words
}

private func edgeMLXBridgeDenseAffineRows(
    rawRows: [[UInt32]],
    scales: [Float],
    biases: [Float],
    groupSize: Int
) -> [[Float]] {
    rawRows.enumerated().map { rowIndex, row in
        row.enumerated().map { columnIndex, raw in
            let groupIndex = rowIndex * (row.count / groupSize) + columnIndex / groupSize
            return Float(raw) * scales[groupIndex] + biases[groupIndex]
        }
    }
}

private func edgeMLXBridgeDenseMatmulTransposed(
    _ lhs: [Float],
    rows: Int,
    inner: Int,
    weights: [[Float]],
    outputs: Int
) -> [Float] {
    var result = Array(repeating: Float.zero, count: rows * outputs)
    for row in 0..<rows {
        for output in 0..<outputs {
            var sum = Float.zero
            for column in 0..<inner {
                sum += lhs[row * inner + column] * weights[output][column]
            }
            result[row * outputs + output] = sum
        }
    }
    return result
}

@Test func cmlxSampleTokenAppliesTopPBeforeTopKWithoutRenormalizingTopK() throws {
    let logits = [Float(10), Float(7.6)] + Array(repeating: Float(7.5), count: 62)
    var sampledSecondToken = false

    for seed in UInt64(1)...128 {
        let token = try EdgeMLXBridge.sampleTokenFloat32GPU(
            logits: logits,
            temperature: 1,
            topK: 2,
            topP: 0.9,
            seed: seed
        )
        #expect(token == 0 || token == 1)
        if token == 1 {
            sampledSecondToken = true
            break
        }
    }

    #expect(sampledSecondToken)
}

@Test func cmlxSampleTokenAppliesMinPBeforeSampling() throws {
    let logits: [Float] = [0, 0, 4]

    for seed in UInt64(1)...16 {
        let token = try EdgeMLXBridge.sampleTokenFloat32GPU(
            logits: logits,
            temperature: 1,
            topK: nil,
            topP: 1,
            minP: 0.95,
            seed: seed
        )
        #expect(token == 2)
    }
}

@Test func cmlxSampleTokenAppliesMinPWithTopKBeforeSampling() throws {
    let logits: [Float] = [10, 8, 0, -1]

    for seed in UInt64(1)...16 {
        let token = try EdgeMLXBridge.sampleTokenFloat32GPU(
            logits: logits,
            temperature: 1,
            topK: 3,
            topP: 1,
            minP: 0.5,
            seed: seed
        )
        #expect(token == 0)
    }
}

@Test func cmlxFastRMSNormMatchesReferenceWhenDefaultMetallibIsAvailable() throws {
    #expect(EdgeMLXBridge.isDefaultMetallibAvailable)

    let rows = 2
    let columns = 8
    let input = (0..<(rows * columns)).map { Float(Int($0 % 11) - 5) / 4.0 }
    let weight = (0..<columns).map { 0.75 + Float($0 % 5) * 0.1 }
    let result = try EdgeMLXBridge.fastRMSNormFloat32GPU(
        input,
        rows: rows,
        columns: columns,
        weight: weight,
        epsilon: 1e-6
    )
    let expected = try (0..<rows).flatMap { row in
        let start = row * columns
        let end = start + columns
        return try CPUReferenceOps.rmsNorm(
            Array(input[start..<end]),
            weight: weight,
            epsilon: 1e-6
        )
    }

    #expect(result.count == expected.count)
    for index in result.indices {
        #expect(abs(result[index] - expected[index]) < 1e-5)
    }
}

@Test func cmlxRMSNormScaleMatchesReferenceWhenDefaultMetallibIsAvailable() throws {
    #expect(EdgeMLXBridge.isDefaultMetallibAvailable)

    let rows = 2
    let columns = 8
    let epsilon: Float = 1e-6
    let scale: Float = 0.125
    let input = (0..<(rows * columns)).map { Float(Int($0 % 17) - 8) / 7.0 }
    let result = try EdgeMLXBridge.rmsNormScaleFloat32GPU(
        input,
        rows: rows,
        columns: columns,
        epsilon: epsilon,
        scale: scale
    )
    let expected = try (0..<rows).flatMap { row in
        let start = row * columns
        let end = start + columns
        return try CPUReferenceOps.rmsNorm(
            Array(input[start..<end]),
            weight: Array(repeating: scale, count: columns),
            epsilon: epsilon
        )
    }

    #expect(result.count == expected.count)
    for index in result.indices {
        #expect(abs(result[index] - expected[index]) < 1e-5)
    }
}

@Test func cmlxFastRMSNormMTLMatchesReferenceWhenEnabled() throws {
    var configuration = MetalRuntimeConfiguration(commandBufferBatchingEnabled: true)
    configuration.useCmlxFastRMSNorm = true
    let runtime = try EdgeMetalRuntime(configuration: configuration)
    let executor = try MetalKernelExecutor(runtime: runtime)
    let rows = 3
    let columns = 16
    let input = (0..<(rows * columns)).map { Float(Int($0 % 13) - 6) / 5.0 }
    let weight = (0..<columns).map { 0.8 + Float($0 % 7) * 0.05 }
    let inputTensor = try executor.makeFloat32Tensor(
        input,
        shape: EdgeTensorShape([rows, columns])
    )
    let weightTensor = try executor.makeFloat32Tensor(
        weight,
        shape: EdgeTensorShape([columns])
    )

    let result = try executor.rmsNorm(
        inputTensor,
        weight: weightTensor,
        epsilon: 1e-6
    )
    #expect(executor.lastExecutionStats?.operationName == "edge_cmlx_fast_rms_norm")

    let expected = try (0..<rows).flatMap { row in
        let start = row * columns
        let end = start + columns
        return try CPUReferenceOps.rmsNorm(
            Array(input[start..<end]),
            weight: weight,
            epsilon: 1e-6
        )
    }
    let actual = try result.readFloat32()
    #expect(actual.count == expected.count)
    for index in actual.indices {
        #expect(abs(actual[index] - expected[index]) < 1e-5)
    }
}

@Test func cmlxGPUAffineQuantizedMatmulMatchesReference() throws {
    let lhsValues: [Float] = [
        1, 2, 3,
        4, 5, 6,
    ]
    let rawRows: [[UInt32]] = (0..<3).map { row in
        (0..<32).map { UInt32(($0 + row) % 16) }
    }
    let scales: [Float] = [0.5, 0.25, 0.125]
    let biases: [Float] = [0, 1, -1]
    let weights = try EdgeQuantizedTensor(
        shape: [3, 32],
        packedShape: [3, 4],
        scaleShape: [3, 1],
        groupSize: 32,
        bits: 4,
        packedValues: cmlxPackQuantizedRows(rawRows, bits: 4),
        scales: scales,
        biases: biases
    )

    let result = try EdgeMLXBridge.affineQuantizedMatmulFloat32GPU(
        lhs: lhsValues,
        rows: 2,
        inner: 3,
        weights: weights
    )
    let expected = try CPUReferenceOps.matmul(
        lhsValues,
        rows: 2,
        inner: 3,
        weights.dequantizedValues(),
        columns: 32
    )

    #expect(result.count == expected.count)
    for index in result.indices {
        #expect(abs(result[index] - expected[index]) < 0.05)
    }
}

@Test func cmlxGPUAffineQuantizedMatmulMatchesReferenceForSixBitPacking() throws {
    let rows = 2
    let inner = 64
    let columns = 8
    let lhsValues = (0..<(rows * inner)).map { Float(Int($0 % 19) - 9) / 8.0 }
    let rawRows: [[UInt32]] = (0..<columns).map { row in
        (0..<inner).map { UInt32((row * 11 + $0 * 7) % 64) }
    }
    let scales = (0..<columns).map { Float($0 + 1) / 64.0 }
    let biases = (0..<columns).map { Float($0 - 3) / 16.0 }
    let weights = try EdgeQuantizedTensor(
        shape: [columns, inner],
        packedShape: [columns, (inner * 6 + 31) / 32],
        scaleShape: [columns, 1],
        groupSize: 64,
        bits: 6,
        packedValues: cmlxPackQuantizedRows(rawRows, bits: 6),
        scales: scales,
        biases: biases
    )

    let result = try EdgeMLXBridge.affineQuantizedMatmulFloat32GPU(
        lhs: lhsValues,
        rows: rows,
        inner: inner,
        weights: weights,
        transpose: true
    )
    let expected = try CPUReferenceOps.matmul(
        lhsValues,
        rows: rows,
        inner: inner,
        cmlxTransposeRowsToColumns(weights.dequantizedValues(), rows: columns, columns: inner),
        columns: columns
    )

    #expect(result.count == expected.count)
    for index in result.indices {
        #expect(abs(result[index] - expected[index]) < 0.05)
    }
}

@Test func cmlxGPUAffineQuantizedMatmulCoversSplitKPrefill() throws {
    let rows = 2
    let inner = 128
    let columns = 64
    let lhsValues = (0..<(rows * inner)).map { Float(Int($0 % 17) - 8) / 8.0 }
    let rawRows: [[UInt32]] = (0..<columns).map { row in
        (0..<inner).map { UInt32((row * 3 + $0 * 5) % 16) }
    }
    let scaleColumns = inner / 32
    let scales = (0..<(columns * scaleColumns)).map { index in
        0.0625 * (1.0 + Float(index % 5) / 10.0)
    }
    let biases = Array(repeating: Float.zero, count: columns * scaleColumns)
    let weights = try EdgeQuantizedTensor(
        shape: [columns, inner],
        packedShape: [columns, inner * 4 / 32],
        scaleShape: [columns, scaleColumns],
        groupSize: 32,
        bits: 4,
        packedValues: cmlxPackQuantizedRows(rawRows, bits: 4),
        scales: scales,
        biases: biases
    )

    let result = try EdgeMLXBridge.affineQuantizedMatmulFloat32GPU(
        lhs: lhsValues,
        rows: rows,
        inner: inner,
        weights: weights,
        transpose: true
    )
    let expected = try CPUReferenceOps.matmul(
        lhsValues,
        rows: rows,
        inner: inner,
        cmlxTransposeRowsToColumns(weights.dequantizedValues(), rows: columns, columns: inner),
        columns: columns
    )

    #expect(result.count == expected.count)
    for index in result.indices {
        #expect(abs(result[index] - expected[index]) < 0.05)
    }
}

/// Tests that mutate process-wide env vars stay in XCTest so they run outside
/// Swift Testing's concurrent scheduler.
final class EdgeMLXBridgeEnvironmentOverrideTests: XCTestCase {

    func testSplitKTileOverrideHandlesPartialNBlock() throws {
        let previousTile = getenv("EDGE_QMM_SPLITK_TILE").map { String(cString: $0) }
        let previousPartitions = getenv("EDGE_QMM_SPLITK_PARTITIONS").map { String(cString: $0) }
        defer {
            if let previousTile {
                setenv("EDGE_QMM_SPLITK_TILE", previousTile, 1)
            } else {
                unsetenv("EDGE_QMM_SPLITK_TILE")
            }
            if let previousPartitions {
                setenv("EDGE_QMM_SPLITK_PARTITIONS", previousPartitions, 1)
            } else {
                unsetenv("EDGE_QMM_SPLITK_PARTITIONS")
            }
        }
        setenv("EDGE_QMM_SPLITK_TILE", "32x64x32", 1)
        setenv("EDGE_QMM_SPLITK_PARTITIONS", "4", 1)

        let rows = 32
        let inner = 128
        let columns = 96 // Multiple of 32, but not of the overridden bn=64.
        let lhsValues = (0..<(rows * inner)).map { Float(Int($0 % 23) - 11) / 12.0 }
        let rawRows: [[UInt32]] = (0..<columns).map { row in
            (0..<inner).map { UInt32((row * 7 + $0 * 5) % 16) }
        }
        let scaleColumns = inner / 32
        let scales = (0..<(columns * scaleColumns)).map { index in
            0.04 * (1.0 + Float(index % 7) / 10.0)
        }
        let biases = Array(repeating: Float.zero, count: columns * scaleColumns)
        let weights = try EdgeQuantizedTensor(
            shape: [columns, inner],
            packedShape: [columns, inner * 4 / 32],
            scaleShape: [columns, scaleColumns],
            groupSize: 32,
            bits: 4,
            packedValues: cmlxPackQuantizedRows(rawRows, bits: 4),
            scales: scales,
            biases: biases
        )

        let result = try EdgeMLXBridge.affineQuantizedMatmulFloat32GPU(
            lhs: lhsValues,
            rows: rows,
            inner: inner,
            weights: weights,
            transpose: true
        )
        let expected = try CPUReferenceOps.matmul(
            lhsValues,
            rows: rows,
            inner: inner,
            cmlxTransposeRowsToColumns(weights.dequantizedValues(), rows: columns, columns: inner),
            columns: columns
        )

        XCTAssertEqual(result.count, expected.count)
        for index in result.indices {
            XCTAssertLessThan(abs(result[index] - expected[index]), 0.08)
        }
    }
}

@Test func cmlxLazyRMSNormAffineQuantizedMatmulMatchesReference() throws {
    let runtime = try EdgeMetalRuntime()
    let rows = 2
    let inner = 32
    let columns = 64
    let lhsValues = (0..<(rows * inner)).map { Float(Int($0 % 13) - 6) / 4.0 }
    let normWeightValues = (0..<inner).map { 0.75 + Float($0 % 7) * 0.05 }
    let rawRows: [[UInt32]] = (0..<columns).map { row in
        (0..<inner).map { UInt32((row * 5 + $0 * 3) % 16) }
    }
    let scales = (0..<columns).map { 0.05 * (1.0 + Float($0 % 3) * 0.25) }
    let biases = (0..<columns).map { Float(Int($0 % 5) - 2) * 0.02 }
    let weights = try EdgeQuantizedTensor(
        shape: [columns, inner],
        packedShape: [columns, inner * 4 / 32],
        scaleShape: [columns, 1],
        groupSize: 32,
        bits: 4,
        packedValues: cmlxPackQuantizedRows(rawRows, bits: 4),
        scales: scales,
        biases: biases
    )
    let lhs = try EdgeTensor(
        float32: lhsValues,
        shape: EdgeTensorShape([rows, inner]),
        runtime: runtime
    )
    let normWeight = try EdgeTensor(
        float32: normWeightValues,
        shape: EdgeTensorShape([inner]),
        runtime: runtime
    )

    let result = try EdgeMLXBridge.rmsNormAffineQuantizedMatmulFloat32MTL(
        lhs: lhs,
        normWeight: normWeight,
        epsilon: 1e-6,
        weights: weights,
        runtime: runtime,
        transpose: true
    ).readFloat32()
    let normalized = try (0..<rows).flatMap { row in
        let start = row * inner
        let end = start + inner
        return try CPUReferenceOps.rmsNorm(
            Array(lhsValues[start..<end]),
            weight: normWeightValues,
            epsilon: 1e-6
        )
    }
    let expected = try CPUReferenceOps.matmul(
        normalized,
        rows: rows,
        inner: inner,
        cmlxTransposeRowsToColumns(weights.dequantizedValues(), rows: columns, columns: inner),
        columns: columns
    )

    #expect(result.count == expected.count)
    for index in result.indices {
        #expect(abs(result[index] - expected[index]) < 0.08)
    }
}

@Test func cmlxGPUAffineQuantizedMatmulRejectsPartialPackedRows() throws {
    let weights = try EdgeQuantizedTensor(
        shape: [3, 4],
        packedShape: [3, 1],
        scaleShape: [3, 2],
        groupSize: 2,
        bits: 4,
        packedValues: cmlxPackQuantizedRows([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ], bits: 4),
        scales: [
            1, 10,
            0.5, 2,
            1, 0.25,
        ],
        biases: [
            0, -1,
            1, 0,
            -1, 10,
        ]
    )

    #expect(throws: EdgeMLXBridgeError.invalidShape) {
        try EdgeMLXBridge.affineQuantizedMatmulFloat32GPU(
            lhs: [
                1, 2, 3,
                4, 5, 6,
            ],
            rows: 2,
            inner: 3,
            weights: weights
        )
    }
}

@Test func cmlxQwen35SessionRegistersPersistentTensors() throws {
    let runtime = try EdgeMetalRuntime()
    let architecture = try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 64,
        hiddenSize: 32,
        intermediateSize: 64,
        attentionHeadCount: 4,
        keyValueHeadCount: 2,
        headDimension: 8,
        linearValueHeadCount: 2,
        linearKeyHeadCount: 1,
        linearKeyHeadDimension: 8,
        linearValueHeadDimension: 8,
        linearConvKernelSize: 4,
        contextLength: 128,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 1_000_000,
        partialRotaryFactor: 0.5,
        quantization: QwenQuantizationProfile(groupSize: 32, bits: 4),
        layerKinds: [.fullAttention, .gdn]
    )
    let session = try EdgeMLXQwen35Session(
        architecture: architecture,
        runtime: runtime
    )
    let norm = try EdgeTensor(
        float32: Array(repeating: Float(1), count: 32),
        shape: EdgeTensorShape([32]),
        runtime: runtime
    )
    let rawRows = (0..<4).map { row in
        (0..<32).map { UInt32((row + $0) % 16) }
    }
    let weights = try EdgeQuantizedTensor(
        shape: [4, 32],
        packedShape: [4, 4],
        scaleShape: [4, 1],
        groupSize: 32,
        bits: 4,
        packedValues: cmlxPackQuantizedRows(rawRows, bits: 4),
        scales: Array(repeating: Float(0.125), count: 4),
        biases: Array(repeating: Float.zero, count: 4)
    )

    try session.registerFloatTensor(id: 10, tensor: norm)
    try session.registerQuantizedTensor(id: 11, tensor: weights)

    #expect(session.registeredFloatTensorCount == 1)
    #expect(session.registeredQuantizedTensorCount == 1)
}

@Test func cmlxQwen35SessionMaterializesEmptyDecoderWeightsAsNoop() throws {
    let runtime = try EdgeMetalRuntime()
    let architecture = try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 32,
        hiddenSize: 64,
        intermediateSize: 64,
        attentionHeadCount: 1,
        keyValueHeadCount: 1,
        headDimension: 64,
        linearValueHeadCount: 1,
        linearKeyHeadCount: 1,
        linearKeyHeadDimension: 64,
        linearValueHeadDimension: 64,
        linearConvKernelSize: 4,
        contextLength: 128,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 1_000_000,
        partialRotaryFactor: 1.0,
        quantization: QwenQuantizationProfile(groupSize: 64, bits: 4),
        layerKinds: [.fullAttention, .gdn]
    )
    let session = try EdgeMLXQwen35Session(architecture: architecture, runtime: runtime)

    try session.materializeDecoderWeights()

    let summary = try #require(try session.memorySummary())
    #expect(summary.contains("weightsPresent=0"))
    #expect(summary.contains("decoderWeightMaterializeArrays=0"))
    #expect(summary.contains("decoderWeightMaterializeBytes=0"))
    #expect(summary.contains("decoderWeightMaterializeBatches=0"))
}

@Test func cmlxQwen35SessionQuantizedDSRScoresCollapseGQAHeads() throws {
    let runtime = try EdgeMetalRuntime()
    let hidden = 256
    let intermediate = 64
    let attentionHeadCount = 2
    let keyValueHeadCount = 1
    let headDimension = 128
    let attentionHidden = attentionHeadCount * headDimension
    let keyValueHidden = keyValueHeadCount * headDimension
    let architecture = try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 32,
        hiddenSize: hidden,
        intermediateSize: intermediate,
        attentionHeadCount: attentionHeadCount,
        keyValueHeadCount: keyValueHeadCount,
        headDimension: headDimension,
        linearValueHeadCount: keyValueHeadCount,
        linearKeyHeadCount: keyValueHeadCount,
        linearKeyHeadDimension: headDimension,
        linearValueHeadDimension: headDimension,
        linearConvKernelSize: 4,
        contextLength: 128,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 1_000_000,
        partialRotaryFactor: 0.5,
        quantization: QwenQuantizationProfile(groupSize: 64, bits: 4),
        layerKinds: [.fullAttention, .gdn]
    )
    let session = try EdgeMLXQwen35Session(architecture: architecture, runtime: runtime)
    let norm = try EdgeTensor(
        float32: Array(repeating: Float(1), count: hidden),
        shape: EdgeTensorShape([hidden]),
        runtime: runtime
    )
    let headNorm = try EdgeTensor(
        float32: Array(repeating: Float(1), count: headDimension),
        shape: EdgeTensorShape([headDimension]),
        runtime: runtime
    )
    func zeroQuantized(rows: Int, columns: Int) throws -> EdgeQuantizedTensor {
        try makeCmlxQuantizedTensor(
            rows: Array(repeating: Array(repeating: UInt32.zero, count: columns), count: rows),
            groupSize: 64,
            bits: 4
        )
    }

    try session.registerFloatTensor(id: QwenCmlxLazyDecodeTensorID.layerInputNorm(0), tensor: norm)
    try session.registerFloatTensor(id: QwenCmlxLazyDecodeTensorID.layerPostAttentionNorm(0), tensor: norm)
    try session.registerFloatTensor(id: QwenCmlxLazyDecodeTensorID.layerAttentionQueryNorm(0), tensor: headNorm)
    try session.registerFloatTensor(id: QwenCmlxLazyDecodeTensorID.layerAttentionKeyNorm(0), tensor: headNorm)
    try session.registerQuantizedTensor(
        id: QwenCmlxLazyDecodeTensorID.layerAttentionQuery(0),
        tensor: zeroQuantized(rows: attentionHidden * 2, columns: hidden)
    )
    try session.registerQuantizedTensor(
        id: QwenCmlxLazyDecodeTensorID.layerAttentionKey(0),
        tensor: zeroQuantized(rows: keyValueHidden, columns: hidden)
    )
    try session.registerQuantizedTensor(
        id: QwenCmlxLazyDecodeTensorID.layerAttentionValue(0),
        tensor: zeroQuantized(rows: keyValueHidden, columns: hidden)
    )
    try session.registerQuantizedTensor(
        id: QwenCmlxLazyDecodeTensorID.layerAttentionOutput(0),
        tensor: zeroQuantized(rows: hidden, columns: attentionHidden)
    )
    try session.registerQuantizedTensor(
        id: QwenCmlxLazyDecodeTensorID.layerMLPGate(0),
        tensor: zeroQuantized(rows: intermediate, columns: hidden)
    )
    try session.registerQuantizedTensor(
        id: QwenCmlxLazyDecodeTensorID.layerMLPUp(0),
        tensor: zeroQuantized(rows: intermediate, columns: hidden)
    )
    try session.registerQuantizedTensor(
        id: QwenCmlxLazyDecodeTensorID.layerMLPDown(0),
        tensor: zeroQuantized(rows: hidden, columns: intermediate)
    )
    try session.setDSRPolicy(
        QwenDSRKVCachePolicy(
            maxSize: 4,
            heavyBudget: 1,
            recentBudget: 1,
            sinkSize: 0,
            evictionInterval: 1,
            scoreActivationRatio: 0,
            scoreDecay: 0
        ),
        layerIndex: 0
    )
    try session.setAttentionCacheQuantization(groupSize: 128, bits: 4)
    let input = try EdgeTensor(
        float32: Array(repeating: Float(0.125), count: hidden),
        shape: EdgeTensorShape([1, hidden]),
        runtime: runtime
    )

    let output = try session.evalQuantizedFullAttentionDecodeLayer(
        input: input,
        layerIndex: 0
    )

    #expect(output.shape == EdgeTensorShape([1, hidden]))
    #expect(try output.readFloat32().count == hidden)
}

@Test func cmlxQwen35SessionEvaluatesQuantizedMLPWithPersistentWeights() throws {
    let runtime = try EdgeMetalRuntime()
    let architecture = try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 64,
        hiddenSize: 32,
        intermediateSize: 32,
        attentionHeadCount: 4,
        keyValueHeadCount: 2,
        headDimension: 8,
        linearValueHeadCount: 2,
        linearKeyHeadCount: 1,
        linearKeyHeadDimension: 8,
        linearValueHeadDimension: 8,
        linearConvKernelSize: 4,
        contextLength: 128,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 1_000_000,
        partialRotaryFactor: 0.5,
        quantization: QwenQuantizationProfile(groupSize: 32, bits: 4),
        layerKinds: [.fullAttention, .gdn]
    )
    let session = try EdgeMLXQwen35Session(
        architecture: architecture,
        runtime: runtime
    )
    let rows = 2
    let hidden = 32
    let intermediate = 32
    let lhsValues = (0..<(rows * hidden)).map { Float(Int($0 % 17) - 8) / 16.0 }
    let input = try EdgeTensor(
        float32: lhsValues,
        shape: EdgeTensorShape([rows, hidden]),
        runtime: runtime
    )
    let gateRows = cmlxMakeRawRows(rowCount: intermediate, columns: hidden, rowScale: 3, columnScale: 5, bias: 0)
    let upRows = cmlxMakeRawRows(rowCount: intermediate, columns: hidden, rowScale: 7, columnScale: 2, bias: 1)
    let downRows = cmlxMakeRawRows(rowCount: hidden, columns: intermediate, rowScale: 5, columnScale: 3, bias: 2)
    let gate = try makeCmlxQuantizedTensor(rows: gateRows, groupSize: 32, bits: 4)
    let up = try makeCmlxQuantizedTensor(rows: upRows, groupSize: 32, bits: 4)
    let down = try makeCmlxQuantizedTensor(rows: downRows, groupSize: 32, bits: 4)
    try session.registerQuantizedTensor(id: 100, tensor: gate)
    try session.registerQuantizedTensor(id: 101, tensor: up)
    try session.registerQuantizedTensor(id: 102, tensor: down)

    let output = try session.evalQuantizedMLP(
        input: input,
        gateTensorID: 100,
        upTensorID: 101,
        downTensorID: 102,
        outputColumns: hidden
    ).readFloat32()
    let gateOutput = try CPUReferenceOps.matmul(
        lhsValues,
        rows: rows,
        inner: hidden,
        cmlxTransposeRowsToColumns(gate.dequantizedValues(), rows: intermediate, columns: hidden),
        columns: intermediate
    )
    let upOutput = try CPUReferenceOps.matmul(
        lhsValues,
        rows: rows,
        inner: hidden,
        cmlxTransposeRowsToColumns(up.dequantizedValues(), rows: intermediate, columns: hidden),
        columns: intermediate
    )
    let activation = try CPUReferenceOps.swiglu(gate: gateOutput, up: upOutput)
    let expected = try CPUReferenceOps.matmul(
        activation,
        rows: rows,
        inner: intermediate,
        cmlxTransposeRowsToColumns(down.dequantizedValues(), rows: hidden, columns: intermediate),
        columns: hidden
    )

    #expect(output.count == expected.count)
    for index in output.indices {
        #expect(abs(output[index] - expected[index]) < 0.12)
    }
}

@Test func cmlxQwen35SessionEvaluatesQuantizedGDNDecodeLayerWithPersistentWeights() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let hidden = 32
    let intermediate = 32
    let keyHeadCount = 2
    let valueHeadCount = 4
    let headDimension = 8
    let convKernel = 4
    let keyHidden = keyHeadCount * headDimension
    let valueHidden = valueHeadCount * headDimension
    let convHidden = keyHidden * 2 + valueHidden
    let architecture = try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 128,
        hiddenSize: hidden,
        intermediateSize: intermediate,
        attentionHeadCount: 4,
        keyValueHeadCount: 2,
        headDimension: headDimension,
        linearValueHeadCount: valueHeadCount,
        linearKeyHeadCount: keyHeadCount,
        linearKeyHeadDimension: headDimension,
        linearValueHeadDimension: headDimension,
        linearConvKernelSize: convKernel,
        contextLength: 128,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 1_000_000,
        partialRotaryFactor: 0.5,
        quantization: QwenQuantizationProfile(groupSize: 32, bits: 4),
        layerKinds: [.gdn, .fullAttention]
    )
    let inputNorm = try EdgeTensor(
        float32: (0..<hidden).map { 0.8 + Float($0 % 5) * 0.03 },
        shape: EdgeTensorShape([hidden]),
        runtime: runtime
    )
    let postNorm = try EdgeTensor(
        float32: (0..<hidden).map { 0.9 + Float($0 % 7) * 0.02 },
        shape: EdgeTensorShape([hidden]),
        runtime: runtime
    )
    let gdnNorm = try EdgeTensor(
        float32: (0..<headDimension).map { 0.85 + Float($0 % 3) * 0.04 },
        shape: EdgeTensorShape([headDimension]),
        runtime: runtime
    )
    let conv1D = try EdgeTensor(
        float32: cmlxGDNConvWeights(channelCount: convHidden, kernelSize: convKernel),
        shape: EdgeTensorShape([convHidden, convKernel, 1]),
        runtime: runtime
    )
    let aLog = try EdgeTensor(
        float32: (0..<valueHeadCount).map { -2.0 + Float($0) * 0.17 },
        shape: EdgeTensorShape([valueHeadCount]),
        runtime: runtime
    )
    let dtBias = try EdgeTensor(
        float32: (0..<valueHeadCount).map { -0.05 + Float($0) * 0.02 },
        shape: EdgeTensorShape([valueHeadCount]),
        runtime: runtime
    )
    let inProjQKV = try makeCmlxQuantizedTensor(
        rows: cmlxMakeRawRows(rowCount: convHidden, columns: hidden, rowScale: 3, columnScale: 5, bias: 1),
        groupSize: 32,
        bits: 4
    )
    let inProjZ = try makeCmlxQuantizedTensor(
        rows: cmlxMakeRawRows(rowCount: valueHidden, columns: hidden, rowScale: 5, columnScale: 7, bias: 2),
        groupSize: 32,
        bits: 4
    )
    let inProjA = try makeCmlxQuantizedTensor(
        rows: cmlxMakeRawRows(rowCount: valueHeadCount, columns: hidden, rowScale: 2, columnScale: 3, bias: 0),
        groupSize: 32,
        bits: 4
    )
    let inProjB = try makeCmlxQuantizedTensor(
        rows: cmlxMakeRawRows(rowCount: valueHeadCount, columns: hidden, rowScale: 7, columnScale: 1, bias: 4),
        groupSize: 32,
        bits: 4
    )
    let outProj = try makeCmlxQuantizedTensor(
        rows: cmlxMakeRawRows(rowCount: hidden, columns: valueHidden, rowScale: 4, columnScale: 3, bias: 5),
        groupSize: 32,
        bits: 4
    )
    let mlpGate = try makeCmlxQuantizedTensor(
        rows: cmlxMakeRawRows(rowCount: intermediate, columns: hidden, rowScale: 3, columnScale: 1, bias: 3),
        groupSize: 32,
        bits: 4
    )
    let mlpUp = try makeCmlxQuantizedTensor(
        rows: cmlxMakeRawRows(rowCount: intermediate, columns: hidden, rowScale: 6, columnScale: 5, bias: 1),
        groupSize: 32,
        bits: 4
    )
    let mlpDown = try makeCmlxQuantizedTensor(
        rows: cmlxMakeRawRows(rowCount: hidden, columns: intermediate, rowScale: 5, columnScale: 2, bias: 6),
        groupSize: 32,
        bits: 4
    )

    let linearAttention = QwenQuantizedGDNWeights(
        layerIndex: 0,
        inProjQKV: inProjQKV,
        inProjZ: inProjZ,
        inProjB: inProjB,
        inProjA: inProjA,
        conv1D: conv1D,
        convWeightLayout: .mlxSanitizedDepthwise,
        aLog: aLog,
        dtBias: dtBias,
        norm: gdnNorm,
        outProj: outProj,
        linearKeyHeadCount: keyHeadCount,
        linearValueHeadCount: valueHeadCount,
        linearKeyHeadDimension: headDimension,
        linearValueHeadDimension: headDimension,
        linearKeyHiddenSize: keyHidden,
        linearValueHiddenSize: valueHidden,
        rmsNormEpsilon: architecture.rmsNormEpsilon
    )
    let layer = try QwenQuantizedGDNDecoderLayerReference(
        linearAttention: linearAttention,
        mlp: QwenQuantizedMLPWeights(
            layerIndex: 0,
            gate: mlpGate,
            up: mlpUp,
            down: mlpDown
        ),
        inputLayerNorm: inputNorm,
        postAttentionLayerNorm: postNorm,
        rmsNormEpsilon: architecture.rmsNormEpsilon
    )
    let input = try EdgeTensor(
        float32: (0..<hidden).map { Float(Int($0 % 13) - 6) / 13.0 },
        shape: EdgeTensorShape([1, hidden]),
        runtime: runtime
    )
    let convState = try EdgeTensor(
        float32: (0..<((convKernel - 1) * convHidden)).map { Float(Int($0 % 9) - 4) / 80.0 },
        shape: EdgeTensorShape([convKernel - 1, convHidden]),
        runtime: runtime
    )
    let recurrentState = try EdgeTensor(
        float32: (0..<(valueHeadCount * headDimension * headDimension)).map { Float(Int($0 % 11) - 5) / 100.0 },
        shape: EdgeTensorShape([valueHeadCount, headDimension, headDimension]),
        runtime: runtime
    )
    let expected = try layer.outputTensor(
        hiddenStates: input,
        convState: convState,
        recurrentState: recurrentState,
        executor: executor
    )

    let session = try EdgeMLXQwen35Session(
        architecture: architecture,
        runtime: runtime
    )
    try session.registerFloatTensor(id: QwenCmlxLazyDecodeTensorID.layerInputNorm(0), tensor: inputNorm)
    try session.registerFloatTensor(id: QwenCmlxLazyDecodeTensorID.layerPostAttentionNorm(0), tensor: postNorm)
    try session.registerQuantizedTensor(id: QwenCmlxLazyDecodeTensorID.layerGDNQKV(0), tensor: inProjQKV)
    try session.registerQuantizedTensor(id: QwenCmlxLazyDecodeTensorID.layerGDNZ(0), tensor: inProjZ)
    try session.registerQuantizedTensor(id: QwenCmlxLazyDecodeTensorID.layerGDNA(0), tensor: inProjA)
    try session.registerQuantizedTensor(id: QwenCmlxLazyDecodeTensorID.layerGDNB(0), tensor: inProjB)
    try session.registerFloatTensor(id: QwenCmlxLazyDecodeTensorID.layerGDNConv1D(0), tensor: conv1D)
    try session.registerFloatTensor(id: QwenCmlxLazyDecodeTensorID.layerGDNAlog(0), tensor: aLog)
    try session.registerFloatTensor(id: QwenCmlxLazyDecodeTensorID.layerGDNDTBias(0), tensor: dtBias)
    try session.registerFloatTensor(id: QwenCmlxLazyDecodeTensorID.layerGDNNorm(0), tensor: gdnNorm)
    try session.registerQuantizedTensor(id: QwenCmlxLazyDecodeTensorID.layerGDNOutput(0), tensor: outProj)
    try session.registerQuantizedTensor(id: QwenCmlxLazyDecodeTensorID.layerMLPGate(0), tensor: mlpGate)
    try session.registerQuantizedTensor(id: QwenCmlxLazyDecodeTensorID.layerMLPUp(0), tensor: mlpUp)
    try session.registerQuantizedTensor(id: QwenCmlxLazyDecodeTensorID.layerMLPDown(0), tensor: mlpDown)

    let cmlx = try session.evalQuantizedGDNDecodeLayer(
        input: input,
        layerIndex: 0,
        convState: convState,
        recurrentState: recurrentState
    )
    let actualHidden = try cmlx.hiddenStates.readFloat32()
    let expectedHidden = try expected.hiddenStates.readFloat32()
    let actualConv = try cmlx.nextConvState.readFloat32()
    let expectedConv = try expected.nextConvState.readFloat32()
    let actualRecurrent = try cmlx.nextRecurrentState.readFloat32()
    let expectedRecurrent = try expected.nextRecurrentState.readFloat32()
    #expect(actualHidden.count == expectedHidden.count)
    for index in actualHidden.indices {
        #expect(abs(actualHidden[index] - expectedHidden[index]) < 0.45)
    }
    #expect(actualConv.count == expectedConv.count)
    for index in actualConv.indices {
        #expect(abs(actualConv[index] - expectedConv[index]) < 1e-3)
    }
    #expect(actualRecurrent.count == expectedRecurrent.count)
    for index in actualRecurrent.indices {
        #expect(abs(actualRecurrent[index] - expectedRecurrent[index]) < 0.12)
    }
}

@Test func cmlxQwen35SessionDecodeStepRunsMixedFAGDNModelWithPersistentWeights() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let hidden = 128
    let intermediate = 128
    let vocabulary = 16
    let attentionHeadCount = 4
    let keyValueHeadCount = 2
    let headDimension = 32
    let linearKeyHeadCount = 2
    let linearValueHeadCount = 4
    let convKernel = 4
    let attentionHidden = attentionHeadCount * headDimension
    let keyValueHidden = keyValueHeadCount * headDimension
    let linearKeyHidden = linearKeyHeadCount * headDimension
    let linearValueHidden = linearValueHeadCount * headDimension
    let convHidden = linearKeyHidden * 2 + linearValueHidden
    let architecture = try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: vocabulary,
        hiddenSize: hidden,
        intermediateSize: intermediate,
        attentionHeadCount: attentionHeadCount,
        keyValueHeadCount: keyValueHeadCount,
        headDimension: headDimension,
        linearValueHeadCount: linearValueHeadCount,
        linearKeyHeadCount: linearKeyHeadCount,
        linearKeyHeadDimension: headDimension,
        linearValueHeadDimension: headDimension,
        linearConvKernelSize: convKernel,
        contextLength: 64,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 1_000_000,
        partialRotaryFactor: 0.5,
        quantization: QwenQuantizationProfile(groupSize: hidden, bits: 4),
        layerKinds: [.fullAttention, .gdn]
    )

    func tensor(_ values: [Float], _ shape: [Int]) throws -> EdgeTensor {
        try EdgeTensor(float32: values, shape: EdgeTensorShape(shape), runtime: runtime)
    }

    func zeroQuantized(rows: Int, columns: Int) throws -> EdgeQuantizedTensor {
        try makeCmlxQuantizedTensor(
            rows: Array(repeating: Array(repeating: UInt32.zero, count: columns), count: rows),
            groupSize: hidden,
            bits: 4
        )
    }

    func mlp(layerIndex: Int) throws -> QwenQuantizedMLPWeights {
        QwenQuantizedMLPWeights(
            layerIndex: layerIndex,
            gate: try zeroQuantized(rows: intermediate, columns: hidden),
            up: try zeroQuantized(rows: intermediate, columns: hidden),
            down: try zeroQuantized(rows: hidden, columns: intermediate)
        )
    }

    let norm = try tensor(Array(repeating: Float(1), count: hidden), [hidden])
    let headNorm = try tensor(Array(repeating: Float(1), count: headDimension), [headDimension])
    let embeddings = try tensor(
        (0..<(vocabulary * hidden)).map { 0.25 + Float($0 % hidden) / 128.0 },
        [vocabulary, hidden]
    )
    let lmHead = try makeCmlxQuantizedTensor(
        rows: (0..<vocabulary).map { row in
            Array(repeating: UInt32(row), count: hidden)
        },
        groupSize: hidden,
        bits: 4
    )
    let fullAttentionLayer = try QwenQuantizedFullAttentionDecoderLayerReference(
        attention: QwenQuantizedFullAttentionReference(
            architecture: architecture,
            layerIndex: 0,
            projectionWeights: QwenQuantizedAttentionProjectionWeights(
                layerIndex: 0,
                query: try zeroQuantized(rows: attentionHidden * 2, columns: hidden),
                key: try zeroQuantized(rows: keyValueHidden, columns: hidden),
                value: try zeroQuantized(rows: keyValueHidden, columns: hidden),
                queryHiddenSize: attentionHidden,
                queryHeadCount: attentionHeadCount,
                queryHeadDimension: headDimension
            ),
            normalizationWeights: QwenAttentionNormWeights(
                layerIndex: 0,
                query: headNorm,
                key: headNorm
            ),
            outputProjectionWeights: QwenQuantizedAttentionOutputProjectionWeights(
                layerIndex: 0,
                weight: try zeroQuantized(rows: hidden, columns: attentionHidden)
            )
        ),
        mlp: try mlp(layerIndex: 0),
        inputLayerNorm: norm,
        postAttentionLayerNorm: norm,
        rmsNormEpsilon: architecture.rmsNormEpsilon
    )
    let gdnLayer = try QwenQuantizedGDNDecoderLayerReference(
        linearAttention: QwenQuantizedGDNWeights(
            layerIndex: 1,
            inProjQKV: try zeroQuantized(rows: convHidden, columns: hidden),
            inProjZ: try zeroQuantized(rows: linearValueHidden, columns: hidden),
            inProjB: try zeroQuantized(rows: linearValueHeadCount, columns: hidden),
            inProjA: try zeroQuantized(rows: linearValueHeadCount, columns: hidden),
            conv1D: try tensor(cmlxGDNConvWeights(channelCount: convHidden, kernelSize: convKernel), [convHidden, convKernel, 1]),
            convWeightLayout: .mlxSanitizedDepthwise,
            aLog: try tensor(Array(repeating: Float(-1), count: linearValueHeadCount), [linearValueHeadCount]),
            dtBias: try tensor(Array(repeating: Float.zero, count: linearValueHeadCount), [linearValueHeadCount]),
            norm: headNorm,
            outProj: try zeroQuantized(rows: hidden, columns: linearValueHidden),
            linearKeyHeadCount: linearKeyHeadCount,
            linearValueHeadCount: linearValueHeadCount,
            linearKeyHeadDimension: headDimension,
            linearValueHeadDimension: headDimension,
            linearKeyHiddenSize: linearKeyHidden,
            linearValueHiddenSize: linearValueHidden,
            rmsNormEpsilon: architecture.rmsNormEpsilon
        ),
        mlp: try mlp(layerIndex: 1),
        inputLayerNorm: norm,
        postAttentionLayerNorm: norm,
        rmsNormEpsilon: architecture.rmsNormEpsilon
    )
    let model = try QwenHybridModelReference(
        architecture: architecture,
        embeddings: QwenTokenEmbeddingWeights(embeddings: embeddings),
        decoderLayers: [
            .quantizedFullAttention(fullAttentionLayer),
            .quantizedGDN(gdnLayer),
        ],
        outputWeights: QwenQuantizedModelOutputWeights(
            finalNorm: norm,
            lmHead: lmHead,
            rmsNormEpsilon: architecture.rmsNormEpsilon
        )
    )
    let cmlx = try QwenCmlxLazyDecodeSession(model: model, runtime: runtime)
    let referenceCaches = try QwenHybridDecoderCaches(architecture: architecture, runtime: runtime, kvCapacity: 8)

    let firstExpected = try model.lastTokenGreedyToken(
        tokenIds: [5],
        caches: referenceCaches,
        executor: executor
    )
    let firstActual = try cmlx.decodeStep(tokenID: 5)
    #expect(firstActual == firstExpected.tokenId)
    #expect(cmlx.tokenPosition == 1)

    let secondExpected = try model.lastTokenGreedyToken(
        tokenIds: [firstActual],
        caches: referenceCaches,
        executor: executor
    )
    let secondActual = try cmlx.decodeStep(tokenID: firstActual)
    #expect(secondActual == secondExpected.tokenId)
    #expect(cmlx.tokenPosition == 2)

    try cmlx.reset()
    let preservedFirst = try cmlx.decodeStep(tokenID: 5)
    #expect(preservedFirst == firstExpected.tokenId)
    #expect(try cmlx.hasDecoderWeights())
    let loadedSummary = try #require(try cmlx.memorySummary())
    #expect(loadedSummary.contains("weightsPresent=1"))
    #expect(loadedSummary.contains("decoded=1"))
    try cmlx.unloadDecoderWeightsPreservingState()
    #expect(!(try cmlx.hasDecoderWeights()))
    #expect(cmlx.tokenPosition == 1)
    #expect(cmlx.decodedTokenCount == 1)
    let unloadedSummary = try #require(try cmlx.memorySummary())
    #expect(unloadedSummary.contains("weightsPresent=0"))
    #expect(unloadedSummary.contains("decoded=1"))
    try cmlx.reloadDecoderWeights(model: model)
    #expect(try cmlx.hasDecoderWeights())
    let preservedSecond = try cmlx.decodeStep(tokenID: preservedFirst)
    #expect(preservedSecond == secondExpected.tokenId)
    #expect(cmlx.tokenPosition == 2)

    try cmlx.reset()
    #expect(cmlx.tokenPosition == 0)
    let resetCaches = try QwenHybridDecoderCaches(architecture: architecture, runtime: runtime, kvCapacity: 8)
    let resetExpected = try model.lastTokenGreedyToken(
        tokenIds: [5],
        caches: resetCaches,
        executor: executor
    )
    let resetActual = try cmlx.decodeStep(tokenID: 5)
    #expect(resetActual == resetExpected.tokenId)

    try cmlx.reset()
    let prefillCaches = try QwenHybridDecoderCaches(architecture: architecture, runtime: runtime, kvCapacity: 8)
    let prefillPrompt = [5, firstExpected.tokenId]
    let prefillExpected = try model.lastTokenGreedyToken(
        tokenIds: prefillPrompt,
        caches: prefillCaches,
        executor: executor
    )
    let prefillActual = try cmlx.prefill(tokenIDs: prefillPrompt)
    #expect(prefillActual == prefillExpected.tokenId)
    #expect(cmlx.tokenPosition == prefillPrompt.count)

    try cmlx.reset()
    let imageFeaturePrompt = [5, 6]
    let imageFeatureCaches = try QwenHybridDecoderCaches(architecture: architecture, runtime: runtime, kvCapacity: 8)
    let imageFeatureExpected = try model.lastTokenGreedyToken(
        tokenIds: imageFeaturePrompt,
        caches: imageFeatureCaches,
        executor: executor
    )
    let embeddingValues = try embeddings.readFloat32()
    let imageFeatureOffset = imageFeaturePrompt[1] * hidden
    let imageFeatureActual = try cmlx.prefillImageFeatures(
        tokenIDs: imageFeaturePrompt,
        imageFeatures: Array(embeddingValues[imageFeatureOffset..<(imageFeatureOffset + hidden)]),
        imageFeatureShape: [1, hidden],
        imageTokenID: imageFeaturePrompt[1]
    )
    #expect(imageFeatureActual == imageFeatureExpected.tokenId)
    #expect(cmlx.tokenPosition == imageFeaturePrompt.count)

    try cmlx.reset()
    let mediaFeatureActual = try cmlx.prefillMediaFeatures(
        tokenIDs: imageFeaturePrompt,
        mediaFeatures: Array(embeddingValues[imageFeatureOffset..<(imageFeatureOffset + hidden)]),
        mediaFeatureShape: [1, hidden],
        mediaTokenID: imageFeaturePrompt[1]
    )
    #expect(mediaFeatureActual == imageFeatureExpected.tokenId)
    #expect(cmlx.tokenPosition == imageFeaturePrompt.count)

    let generated = try cmlx.generateNextTokens(
        promptTokenIds: [5],
        maxTokenCount: 2
    )
    #expect(generated == [firstExpected.tokenId, secondExpected.tokenId])
    #expect(cmlx.tokenPosition == 3)

    try cmlx.reset()
    try cmlx.prefillAsync(tokenIDs: [5])
    let iteratorFirst = try cmlx.nextToken()
    #expect(iteratorFirst == firstExpected.tokenId)
    let iteratorSecond = try cmlx.nextToken()
    #expect(iteratorSecond == secondExpected.tokenId)
    #expect(cmlx.tokenPosition == 3)

    do {
        let previousEvalProfileEnv = getenv("EDGE_CMLX_EVAL_PROFILE")
            .map { String(cString: $0) }
        let previousGraphProfileEnv = getenv("EDGE_CMLX_GRAPH_PROFILE")
            .map { String(cString: $0) }
        setenv("EDGE_CMLX_EVAL_PROFILE", "1", 1)
        setenv("EDGE_CMLX_GRAPH_PROFILE", "1", 1)
        defer {
            if let previousEvalProfileEnv {
                setenv("EDGE_CMLX_EVAL_PROFILE", previousEvalProfileEnv, 1)
            } else {
                unsetenv("EDGE_CMLX_EVAL_PROFILE")
            }
            if let previousGraphProfileEnv {
                setenv("EDGE_CMLX_GRAPH_PROFILE", previousGraphProfileEnv, 1)
            } else {
                unsetenv("EDGE_CMLX_GRAPH_PROFILE")
            }
        }
        try cmlx.reset()
        try cmlx.resetEvalProfile()
        try cmlx.prefillAsync(tokenIDs: [5])
        _ = try cmlx.nextToken()
        let evalProfile = try #require(try cmlx.evalProfileSummary())
        #expect(evalProfile.contains("total={barriers=2"))
        #expect(evalProfile.contains("prefill={barriers=1"))
        #expect(evalProfile.contains("decode={barriers=1"))
        #expect(evalProfile.contains("tokenRead={barriers=0"))
        #expect(evalProfile.contains("tokenReads=1"))
        #expect(evalProfile.contains("inventoryTotal={"))
        #expect(evalProfile.contains("inventoryPrefill={"))
        #expect(evalProfile.contains("inventoryDecode={"))
        #expect(evalProfile.contains("last={caller=edge_cmlx_qwen35_session_next_token,mode=async_eval"))
        #expect(evalProfile.contains(",graph=roots="))
        #expect(evalProfile.contains(",ops="))
    }

    try cmlx.reset()
    try cmlx.setRepetitionPenalty(1.1, contextTokenIds: [5])
    try cmlx.prefillSampledAsync(
        tokenIDs: [5],
        temperature: 0.6,
        topK: 4,
        topP: 0.9,
        seed: 42
    )
    let sampledFirst = try cmlx.nextSampledToken(
        temperature: 0.6,
        topK: 4,
        topP: 0.9,
        seed: 43
    )
    #expect(sampledFirst >= 0)
    #expect(sampledFirst < architecture.vocabularySize)
    #expect(cmlx.tokenPosition == 2)

    let quantizedCacheCmlx = try QwenCmlxLazyDecodeSession(
        model: model,
        runtime: runtime,
        attentionCacheQuantizationGroupSize: 32,
        attentionCacheQuantizationBits: 4
    )
    try quantizedCacheCmlx.prefillAsync(tokenIDs: prefillPrompt)
    let quantizedCacheFirst = try quantizedCacheCmlx.nextToken()
    #expect(quantizedCacheFirst == prefillExpected.tokenId)
    let quantizedCacheSecond = try quantizedCacheCmlx.nextToken()
    #expect(quantizedCacheSecond >= 0)
    #expect(quantizedCacheSecond < architecture.vocabularySize)
    #expect(quantizedCacheCmlx.tokenPosition == prefillPrompt.count + 2)

    try cmlx.setAttentionCacheLimit(2)
    try cmlx.reset()
    try cmlx.prefillAsync(tokenIDs: [5, firstExpected.tokenId])
    let limitedFirst = try cmlx.nextToken()
    let limitedSecond = try cmlx.nextToken()
    let limitedThird = try cmlx.nextToken()
    #expect([limitedFirst, limitedSecond, limitedThird].count == 3)
    #expect(cmlx.tokenPosition == 5)
}

@Test func cmlxCPUFloatMatmulRejectsInvalidShapes() throws {
    #expect(throws: EdgeMLXBridgeError.invalidShape) {
        try EdgeMLXBridge.matmulFloat32CPU(
            lhs: [1, 2, 3],
            rows: 1,
            inner: 2,
            rhs: [1, 2, 3, 4],
            columns: 2
        )
    }
}

private func cmlxPackQuantizedRows(_ rows: [[UInt32]], bits: Int) -> [UInt32] {
    rows.flatMap { cmlxPackQuantizedWords($0, bits: bits) }
}

private func cmlxMakeRawRows(
    rowCount: Int,
    columns: Int,
    rowScale: Int,
    columnScale: Int,
    bias: Int
) -> [[UInt32]] {
    (0..<rowCount).map { row in
        (0..<columns).map { column in
            UInt32((row * rowScale + column * columnScale + bias) % 16)
        }
    }
}

private func makeCmlxQuantizedTensor(
    rows: [[UInt32]],
    groupSize: Int,
    bits: Int
) throws -> EdgeQuantizedTensor {
    let rowCount = rows.count
    let columns = rows.first?.count ?? 0
    let scaleColumns = columns / groupSize
    return try EdgeQuantizedTensor(
        shape: [rowCount, columns],
        packedShape: [rowCount, (columns * bits + 31) / 32],
        scaleShape: [rowCount, scaleColumns],
        groupSize: groupSize,
        bits: bits,
        packedValues: cmlxPackQuantizedRows(rows, bits: bits),
        scales: Array(repeating: Float(0.0625), count: rowCount * scaleColumns),
        biases: Array(repeating: Float.zero, count: rowCount * scaleColumns)
    )
}

private func cmlxGDNConvWeights(channelCount: Int, kernelSize: Int) -> [Float] {
    (0..<channelCount).flatMap { channel in
        (0..<kernelSize).map { kernel in
            if kernel == kernelSize - 1 {
                return 0.55 + Float(channel % 5) * 0.01
            }
            return Float((channel + kernel) % 7 - 3) / 70.0
        }
    }
}

private func cmlxTransposeRowsToColumns(_ values: [Float], rows: Int, columns: Int) -> [Float] {
    var transposed = Array(repeating: Float.zero, count: values.count)
    for row in 0..<rows {
        for column in 0..<columns {
            transposed[column * rows + row] = values[row * columns + column]
        }
    }
    return transposed
}

private func cmlxPackQuantizedWords(_ values: [UInt32], bits: Int) -> [UInt32] {
    var words = Array(repeating: UInt32.zero, count: (values.count * bits + 31) / 32)
    let mask = UInt32((1 << bits) - 1)
    for (index, value) in values.enumerated() {
        let bitOffset = index * bits
        let wordIndex = bitOffset / 32
        let shift = bitOffset % 32
        words[wordIndex] |= (value & mask) << UInt32(shift)
        if shift + bits > 32 {
            words[wordIndex + 1] |= (value & mask) >> UInt32(32 - shift)
        }
    }
    return words
}
