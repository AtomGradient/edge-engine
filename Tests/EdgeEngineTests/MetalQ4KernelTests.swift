// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Darwin
import Foundation
import Testing
import XCTest
@testable import EdgeEngine

final class MetalQ4MLXBridgeStressTests: XCTestCase {
    func testVendoredMLXBridgeStressDoesNotGrowMemoryOrRetainInputBuffers() throws {
        guard ProcessInfo.processInfo.environment["EDGE_RUN_MLX_STRESS_TEST"] == "1" else {
            throw XCTSkip("Set EDGE_RUN_MLX_STRESS_TEST=1 to run the 1000-iteration MLX bridge stress test.")
        }

        let runtime = try EdgeMetalRuntime(
            configuration: MetalRuntimeConfiguration(useMLXQuantizedMatmul: true)
        )
        let executor = try MetalKernelExecutor(runtime: runtime)
        let rows = 64
        let inner = 128
        let columns = 64
        let lhsValues = (0..<(rows * inner)).map { Float(Int($0 % 17) - 8) / 11.0 }
        let rawRows: [[UInt32]] = (0..<columns).map { row in
            (0..<inner).map { UInt32((row * 7 + $0 * 5) % 16) }
        }
        let scales = (0..<columns * 2).map { Float($0 % 7 + 1) / 16.0 }
        let biases = Array(repeating: Float.zero, count: columns * 2)
        let denseWeights = affineDenseValues(
            rawRows: rawRows,
            scales: scales,
            biases: biases,
            groupSize: 64
        )
        let weights = try EdgeQuantizedTensor(
            shape: [columns, inner],
            packedShape: [columns, inner * 4 / 32],
            scaleShape: [columns, 2],
            groupSize: 64,
            bits: 4,
            packedValues: packQuantizedRows(rawRows, bits: 4),
            scales: scales,
            biases: biases
        )
        let lhs = try EdgeTensor(float32: lhsValues, shape: EdgeTensorShape([rows, inner]), runtime: runtime)

        for _ in 0..<20 {
            _ = try executor.affineQuantizedMatmul(lhs, weights: weights, transpose: true)
        }
        runtime.waitForPendingWork()

        let footprintBefore = currentPhysicalFootprintBytes()
        let lhsRetainBefore = EdgeMLXBridge.mtlBufferRetainCountForTesting(lhs.buffer)
        var finalOutput: [Float] = []
        for iteration in 0..<1_000 {
            try autoreleasepool {
                let result = try executor.affineQuantizedMatmul(lhs, weights: weights, transpose: true)
                if iteration == 999 {
                    finalOutput = try result.readFloat32()
                }
            }
        }
        runtime.waitForPendingWork()

        let lhsRetainAfter = EdgeMLXBridge.mtlBufferRetainCountForTesting(lhs.buffer)
        XCTAssertLessThanOrEqual(lhsRetainAfter, lhsRetainBefore + 1)
        if let footprintBefore, let footprintAfter = currentPhysicalFootprintBytes(), footprintAfter > footprintBefore {
            XCTAssertLessThanOrEqual(footprintAfter - footprintBefore, 50 * 1_048_576)
        }

        let expected = denseMatmulTransposed(
            lhsValues,
            rows: rows,
            inner: inner,
            weights: denseWeights,
            outputs: columns
        )
        let error = try NumericComparison.maxAbsoluteError(finalOutput, expected)
        XCTAssertLessThan(error, 0.05)
        XCTAssertEqual(executor.lastExecutionStats?.operationName, "edge_cmlx_affine_quantized_matmul")
    }
}

@Test func customMetalQ4MatmulMatchesCPUOracle() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let lhsValues: [Float] = [
        1, 2, 3,
        4, 5, 6,
    ]
    let denseWeights: [Float] = [
        -1, 0.5,
        2, -3,
        4, 1,
    ]
    let weights = try Q4WeightMatrix.quantizeSymmetric(
        denseWeights,
        rows: 3,
        columns: 2,
        groupSize: 4
    )
    let lhs = try EdgeTensor(float32: lhsValues, shape: EdgeTensorShape([2, 3]), runtime: runtime)

    let result = try executor.q4Matmul(lhs, weights: weights)
    let expected = try CPUReferenceOps.q4Matmul(lhsValues, rows: 2, inner: 3, weights: weights)
    let error = try NumericComparison.maxAbsoluteError(try result.readFloat32(), expected)

    #expect(result.shape == EdgeTensorShape([2, 2]))
    #expect(error < 1e-4)
}

@Test func customMetalAffineQuantizedMatmulMatchesCPUOracle() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let lhsValues: [Float] = [
        1, 2, 3,
        4, 5, 6,
    ]
    let rawRows: [[UInt32]] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ]
    let scales: [Float] = [
        1, 10,
        0.5, 2,
        1, 0.25,
    ]
    let biases: [Float] = [
        0, -1,
        1, 0,
        -1, 10,
    ]
    let denseWeights = affineDenseValues(
        rawRows: rawRows,
        scales: scales,
        biases: biases,
        groupSize: 2
    )
    let weights = try EdgeQuantizedTensor(
        shape: [3, 4],
        packedShape: [3, 1],
        scaleShape: [3, 2],
        groupSize: 2,
        bits: 4,
        packedValues: packQuantizedRows(rawRows, bits: 4),
        scales: scales,
        biases: biases
    )
    let lhs = try EdgeTensor(float32: lhsValues, shape: EdgeTensorShape([2, 3]), runtime: runtime)

    let result = try executor.affineQuantizedMatmul(lhs, weights: weights)
    let expected = denseMatmul(lhsValues, rows: 2, inner: 3, weights: denseWeights, columns: 4)
    let error = try NumericComparison.maxAbsoluteError(try result.readFloat32(), expected)

    #expect(result.shape == EdgeTensorShape([2, 4]))
    #expect(error < 1e-4)
    #expect(executor.lastExecutionStats?.operationName == "edge_affine_quantized_matmul")
}

@Test func customMetalAffineQuantizedMatmulReusesWeightBuffers() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let lhsValues: [Float] = [1, 2]
    let rawRows: [[UInt32]] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    let weights = try EdgeQuantizedTensor(
        shape: [2, 4],
        packedShape: [2, 1],
        scaleShape: [2, 1],
        groupSize: 4,
        bits: 4,
        packedValues: packQuantizedRows(rawRows, bits: 4),
        scales: [1, 1]
    )
    let lhs = try EdgeTensor(float32: lhsValues, shape: EdgeTensorShape([1, 2]), runtime: runtime)

    let first = try executor.affineQuantizedMatmul(lhs, weights: weights)
    let firstOutput = try first.readFloat32()
    let afterFirst = executor.affineQuantizedBufferCacheStats
    let second = try executor.affineQuantizedMatmul(lhs, weights: weights)
    let secondOutput = try second.readFloat32()
    let afterSecond = executor.affineQuantizedBufferCacheStats

    #expect(firstOutput == secondOutput)
    #expect(afterFirst.entryCount == 1)
    #expect(afterFirst.missCount == 1)
    #expect(afterFirst.hitCount == 0)
    #expect(afterFirst.uploadedByteCount == 24)
    #expect(afterFirst.cachedByteCount == 24)
    #expect(afterSecond.entryCount == 1)
    #expect(afterSecond.missCount == 1)
    #expect(afterSecond.hitCount == 1)
    #expect(afterSecond.uploadedByteCount == 24)
    #expect(afterSecond.cachedByteCount == 24)
}

@Test func customMetalAffineQuantizedMatmulCanDisableWeightBufferCache() throws {
    let runtime = try EdgeMetalRuntime(
        configuration: MetalRuntimeConfiguration(quantizedBufferCacheLimitBytes: 0)
    )
    let executor = try MetalKernelExecutor(runtime: runtime)
    let lhsValues: [Float] = [1, 2]
    let rawRows: [[UInt32]] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    let weights = try EdgeQuantizedTensor(
        shape: [2, 4],
        packedShape: [2, 1],
        scaleShape: [2, 1],
        groupSize: 4,
        bits: 4,
        packedValues: packQuantizedRows(rawRows, bits: 4),
        scales: [1, 1]
    )
    let lhs = try EdgeTensor(float32: lhsValues, shape: EdgeTensorShape([1, 2]), runtime: runtime)

    let first = try executor.affineQuantizedMatmul(lhs, weights: weights)
    let firstOutput = try first.readFloat32()
    let afterFirst = executor.affineQuantizedBufferCacheStats
    let second = try executor.affineQuantizedMatmul(lhs, weights: weights)
    let secondOutput = try second.readFloat32()
    let afterSecond = executor.affineQuantizedBufferCacheStats

    #expect(firstOutput == secondOutput)
    #expect(afterFirst.entryCount == 0)
    #expect(afterFirst.missCount == 1)
    #expect(afterFirst.hitCount == 0)
    #expect(afterFirst.uploadedByteCount == 24)
    #expect(afterFirst.cachedByteCount == 0)
    #expect(afterSecond.entryCount == 0)
    #expect(afterSecond.missCount == 2)
    #expect(afterSecond.hitCount == 0)
    #expect(afterSecond.uploadedByteCount == 48)
    #expect(afterSecond.cachedByteCount == 0)
}

@Test func customMetalAffineQuantizedMatmulCanReleaseHostStorageAfterCachedUpload() throws {
    let runtime = try EdgeMetalRuntime(
        configuration: MetalRuntimeConfiguration(
            releaseQuantizedHostStorageAfterUpload: true
        )
    )
    let executor = try MetalKernelExecutor(runtime: runtime)
    let lhsValues: [Float] = [1, 2]
    let rawRows: [[UInt32]] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    let weights = try EdgeQuantizedTensor(
        shape: [2, 4],
        packedShape: [2, 1],
        scaleShape: [2, 1],
        groupSize: 4,
        bits: 4,
        packedValues: packQuantizedRows(rawRows, bits: 4),
        scales: [1, 1]
    )
    let lhs = try EdgeTensor(float32: lhsValues, shape: EdgeTensorShape([1, 2]), runtime: runtime)
    #expect(weights.hostStorageByteCount == 16)

    let first = try executor.affineQuantizedMatmul(lhs, weights: weights)
    let firstOutput = try first.readFloat32()
    let afterFirst = executor.affineQuantizedBufferCacheStats
    #expect(weights.hostStorageReleased)
    #expect(weights.hostStorageByteCount == 0)

    let second = try executor.affineQuantizedMatmul(lhs, weights: weights)
    let secondOutput = try second.readFloat32()
    let afterSecond = executor.affineQuantizedBufferCacheStats

    #expect(firstOutput == secondOutput)
    #expect(afterFirst.entryCount == 1)
    #expect(afterFirst.missCount == 1)
    #expect(afterFirst.hitCount == 0)
    #expect(afterFirst.uploadedByteCount == 24)
    #expect(afterFirst.cachedByteCount == 24)
    #expect(afterFirst.releasedHostStorageByteCount == 16)
    #expect(afterFirst.releasedHostStorageCount == 1)
    #expect(afterSecond.entryCount == 1)
    #expect(afterSecond.missCount == 1)
    #expect(afterSecond.hitCount == 1)
    #expect(afterSecond.uploadedByteCount == 24)
    #expect(afterSecond.cachedByteCount == 24)
    #expect(afterSecond.releasedHostStorageByteCount == 16)
    #expect(afterSecond.releasedHostStorageCount == 1)
}

@Test func customMetalAffineQuantizedMatmulKeepsHostStorageWhenUploadIsNotCached() throws {
    let runtime = try EdgeMetalRuntime(
        configuration: MetalRuntimeConfiguration(
            quantizedBufferCacheLimitBytes: 0,
            releaseQuantizedHostStorageAfterUpload: true
        )
    )
    let executor = try MetalKernelExecutor(runtime: runtime)
    let lhsValues: [Float] = [1, 2]
    let rawRows: [[UInt32]] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    let weights = try EdgeQuantizedTensor(
        shape: [2, 4],
        packedShape: [2, 1],
        scaleShape: [2, 1],
        groupSize: 4,
        bits: 4,
        packedValues: packQuantizedRows(rawRows, bits: 4),
        scales: [1, 1]
    )
    let lhs = try EdgeTensor(float32: lhsValues, shape: EdgeTensorShape([1, 2]), runtime: runtime)

    _ = try executor.affineQuantizedMatmul(lhs, weights: weights)
    let stats = executor.affineQuantizedBufferCacheStats

    #expect(!weights.hostStorageReleased)
    #expect(weights.hostStorageByteCount == 16)
    #expect(stats.entryCount == 0)
    #expect(stats.cachedByteCount == 0)
    #expect(stats.releasedHostStorageByteCount == 0)
    #expect(stats.releasedHostStorageCount == 0)
}

@Test func customMetalAffineQuantizedMatmulHandlesSixBitPackingAcrossWordBoundary() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let lhsValues: [Float] = [1, 0.5, 0.25]
    let rawRows: [[UInt32]] = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20, 21, 22, 23],
    ]
    let scales: [Float] = [1, 1, 1]
    let denseWeights = affineDenseValues(
        rawRows: rawRows,
        scales: scales,
        biases: nil,
        groupSize: 8
    )
    let weights = try EdgeQuantizedTensor(
        shape: [3, 8],
        packedShape: [3, 2],
        scaleShape: [3, 1],
        groupSize: 8,
        bits: 6,
        packedValues: packQuantizedRows(rawRows, bits: 6),
        scales: scales
    )
    let lhs = try EdgeTensor(float32: lhsValues, shape: EdgeTensorShape([1, 3]), runtime: runtime)

    let result = try executor.affineQuantizedMatmul(lhs, weights: weights)
    let expected = denseMatmul(lhsValues, rows: 1, inner: 3, weights: denseWeights, columns: 8)
    let error = try NumericComparison.maxAbsoluteError(try result.readFloat32(), expected)

    #expect(result.shape == EdgeTensorShape([1, 8]))
    #expect(error < 1e-4)
}

@Test func customMetalAffineQuantizedMatmulSupportsTransposedMLXWeights() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let lhsValues: [Float] = [
        1, 2, 3,
        4, 5, 6,
    ]
    let rawRows: [[UInt32]] = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
    ]
    let scales: [Float] = [1, 2, 0.5, 0.25]
    let biases: [Float] = [0, -1, 10, 20]
    let denseWeights = affineDenseValues(
        rawRows: rawRows,
        scales: scales,
        biases: biases,
        groupSize: 3
    )
    let weights = try EdgeQuantizedTensor(
        shape: [4, 3],
        packedShape: [4, 1],
        scaleShape: [4, 1],
        groupSize: 3,
        bits: 4,
        packedValues: packQuantizedRows(rawRows, bits: 4),
        scales: scales,
        biases: biases
    )
    let lhs = try EdgeTensor(float32: lhsValues, shape: EdgeTensorShape([2, 3]), runtime: runtime)

    let result = try executor.affineQuantizedMatmul(lhs, weights: weights, transpose: true)
    let expected = denseMatmulTransposed(lhsValues, rows: 2, inner: 3, weights: denseWeights, outputs: 4)
    let error = try NumericComparison.maxAbsoluteError(try result.readFloat32(), expected)

    #expect(result.shape == EdgeTensorShape([2, 4]))
    #expect(error < 1e-4)
}

@Test func customMetalAffineQuantizedMatmulCanUseVendoredMLXBridge() throws {
    let runtime = try EdgeMetalRuntime(
        configuration: MetalRuntimeConfiguration(useMLXQuantizedMatmul: true)
    )
    let executor = try MetalKernelExecutor(runtime: runtime)
    let rows = 2
    let inner = 32
    let columns = 4
    let lhsValues = (0..<(rows * inner)).map { Float(Int($0 % 11) - 5) / 7.0 }
    let rawRows: [[UInt32]] = (0..<columns).map { row in
        (0..<inner).map { UInt32((row * 5 + $0 * 3) % 16) }
    }
    let scales = (0..<columns).map { Float($0 + 1) / 16.0 }
    let biases = (0..<columns).map { Float($0 - 1) / 8.0 }
    let denseWeights = affineDenseValues(
        rawRows: rawRows,
        scales: scales,
        biases: biases,
        groupSize: inner
    )
    let weights = try EdgeQuantizedTensor(
        shape: [columns, inner],
        packedShape: [columns, inner * 4 / 32],
        scaleShape: [columns, 1],
        groupSize: inner,
        bits: 4,
        packedValues: packQuantizedRows(rawRows, bits: 4),
        scales: scales,
        biases: biases
    )
    let lhs = try EdgeTensor(float32: lhsValues, shape: EdgeTensorShape([rows, inner]), runtime: runtime)

    let result = try executor.affineQuantizedMatmul(lhs, weights: weights, transpose: true)
    let expected = denseMatmulTransposed(lhsValues, rows: rows, inner: inner, weights: denseWeights, outputs: columns)
    let error = try NumericComparison.maxAbsoluteError(try result.readFloat32(), expected)

    #expect(result.shape == EdgeTensorShape([rows, columns]))
    #expect(error < 0.05)
    #expect(executor.lastExecutionStats?.operationName == "edge_cmlx_affine_quantized_matmul")
}

@Test func customMetalAffineQuantizedMatmulCanUseVendoredMLXBridgeForSingleToken() throws {
    let runtime = try EdgeMetalRuntime(
        configuration: MetalRuntimeConfiguration(useMLXQuantizedMatmul: true)
    )
    let executor = try MetalKernelExecutor(runtime: runtime)
    let inner = 64
    let columns = 8
    let lhsValues = (0..<inner).map { Float(Int($0 % 13) - 6) / 9.0 }
    let rawRows: [[UInt32]] = (0..<columns).map { row in
        (0..<inner).map { UInt32((row * 7 + $0 * 3) % 16) }
    }
    let scales = (0..<columns).map { Float($0 + 1) / 16.0 }
    let biases = Array(repeating: Float.zero, count: columns)
    let denseWeights = affineDenseValues(
        rawRows: rawRows,
        scales: scales,
        biases: biases,
        groupSize: inner
    )
    let weights = try EdgeQuantizedTensor(
        shape: [columns, inner],
        packedShape: [columns, inner * 4 / 32],
        scaleShape: [columns, 1],
        groupSize: inner,
        bits: 4,
        packedValues: packQuantizedRows(rawRows, bits: 4),
        scales: scales,
        biases: biases
    )
    let lhs = try EdgeTensor(float32: lhsValues, shape: EdgeTensorShape([1, inner]), runtime: runtime)

    let result = try executor.affineQuantizedMatmul(lhs, weights: weights, transpose: true)
    let expected = denseMatmulTransposed(lhsValues, rows: 1, inner: inner, weights: denseWeights, outputs: columns)
    let error = try NumericComparison.maxAbsoluteError(try result.readFloat32(), expected)

    #expect(result.shape == EdgeTensorShape([1, columns]))
    #expect(error < 0.05)
    #expect(executor.lastExecutionStats?.operationName == "edge_cmlx_affine_quantized_matmul")
}

@Test func customMetalAffineQuantizedMatmulUsesVendoredMLXBridgeForLargePrefillWhenEnabled() throws {
    let runtime = try EdgeMetalRuntime(
        configuration: MetalRuntimeConfiguration(
            useMLXQuantizedMatmul: false,
            useMLXQuantizedPrefillMatmul: true
        )
    )
    let executor = try MetalKernelExecutor(runtime: runtime)
    let rows = 64
    let inner = 64
    let columns = 64
    let lhsValues = (0..<(rows * inner)).map { Float(Int($0 % 17) - 8) / 11.0 }
    let rawRows: [[UInt32]] = (0..<columns).map { row in
        (0..<inner).map { UInt32((row * 7 + $0 * 5) % 16) }
    }
    let scales = (0..<columns).map { Float($0 + 1) / 16.0 }
    let biases = Array(repeating: Float.zero, count: columns)
    let denseWeights = affineDenseValues(
        rawRows: rawRows,
        scales: scales,
        biases: biases,
        groupSize: inner
    )
    let weights = try EdgeQuantizedTensor(
        shape: [columns, inner],
        packedShape: [columns, inner * 4 / 32],
        scaleShape: [columns, 1],
        groupSize: inner,
        bits: 4,
        packedValues: packQuantizedRows(rawRows, bits: 4),
        scales: scales,
        biases: biases
    )
    let lhs = try EdgeTensor(float32: lhsValues, shape: EdgeTensorShape([rows, inner]), runtime: runtime)

    let result = try executor.affineQuantizedMatmul(lhs, weights: weights, transpose: true)
    let expected = denseMatmulTransposed(lhsValues, rows: rows, inner: inner, weights: denseWeights, outputs: columns)
    let error = try NumericComparison.maxAbsoluteError(try result.readFloat32(), expected)

    #expect(result.shape == EdgeTensorShape([rows, columns]))
    #expect(error < 0.05)
    #expect(executor.lastExecutionStats?.operationName == "edge_cmlx_affine_quantized_matmul")
}

@Test func customMetalAffineQuantizedMatmulCanEncodeVendoredQMMOnRuntimeCommandBuffer() throws {
    let runtime = try EdgeMetalRuntime(
        configuration: MetalRuntimeConfiguration(
            commandBufferBatchingEnabled: true,
            useMLXQuantizedMatmul: false,
            useMLXQuantizedPrefillMatmul: false,
            useVendoredCommandBufferPrefillQMM: true
        )
    )
    let executor = try MetalKernelExecutor(runtime: runtime)
    let rows = 64
    let inner = 64
    let columns = 64
    let lhsValues = (0..<(rows * inner)).map { Float(Int($0 % 17) - 8) / 11.0 }
    let rawRows: [[UInt32]] = (0..<columns).map { row in
        (0..<inner).map { UInt32((row * 7 + $0 * 5) % 16) }
    }
    let scales = (0..<columns).map { Float($0 + 1) / 16.0 }
    let biases = Array(repeating: Float.zero, count: columns)
    let denseWeights = affineDenseValues(
        rawRows: rawRows,
        scales: scales,
        biases: biases,
        groupSize: inner
    )
    let weights = try EdgeQuantizedTensor(
        shape: [columns, inner],
        packedShape: [columns, inner * 4 / 32],
        scaleShape: [columns, 1],
        groupSize: inner,
        bits: 4,
        packedValues: packQuantizedRows(rawRows, bits: 4),
        scales: scales,
        biases: biases
    )
    let lhs = try EdgeTensor(float32: lhsValues, shape: EdgeTensorShape([rows, inner]), runtime: runtime)

    let result = try executor.affineQuantizedMatmul(lhs, weights: weights, transpose: true)
    let expected = denseMatmulTransposed(lhsValues, rows: rows, inner: inner, weights: denseWeights, outputs: columns)
    let error = try NumericComparison.maxAbsoluteError(try result.readFloat32(), expected)

    #expect(result.shape == EdgeTensorShape([rows, columns]))
    #expect(error < 0.05)
    #expect(executor.lastExecutionStats?.operationName == "edge_cmlx_affine_qmm_t_command_buffer")
}

@Test func customMetalAffineQuantizedMatmulKeepsSingleTokenDecodeOnNativeQMV() throws {
    let runtime = try EdgeMetalRuntime(
        configuration: MetalRuntimeConfiguration(
            useMLXQuantizedMatmul: false,
            useMLXQuantizedPrefillMatmul: true
        )
    )
    let executor = try MetalKernelExecutor(runtime: runtime)
    let inner = 64
    let columns = 8
    let lhsValues = (0..<inner).map { Float(Int($0 % 13) - 6) / 9.0 }
    let rawRows: [[UInt32]] = (0..<columns).map { row in
        (0..<inner).map { UInt32((row * 7 + $0 * 3) % 16) }
    }
    let scales = (0..<columns).map { Float($0 + 1) / 16.0 }
    let biases = Array(repeating: Float.zero, count: columns)
    let denseWeights = affineDenseValues(
        rawRows: rawRows,
        scales: scales,
        biases: biases,
        groupSize: inner
    )
    let weights = try EdgeQuantizedTensor(
        shape: [columns, inner],
        packedShape: [columns, inner * 4 / 32],
        scaleShape: [columns, 1],
        groupSize: inner,
        bits: 4,
        packedValues: packQuantizedRows(rawRows, bits: 4),
        scales: scales,
        biases: biases
    )
    let lhs = try EdgeTensor(float32: lhsValues, shape: EdgeTensorShape([1, inner]), runtime: runtime)

    let result = try executor.affineQuantizedMatmul(lhs, weights: weights, transpose: true)
    let expected = denseMatmulTransposed(lhsValues, rows: 1, inner: inner, weights: denseWeights, outputs: columns)
    let error = try NumericComparison.maxAbsoluteError(try result.readFloat32(), expected)

    #expect(result.shape == EdgeTensorShape([1, columns]))
    #expect(error < 1e-3)
    #expect(executor.lastExecutionStats?.operationName == "edge_affine_qmv_transposed")
}

@Test func customMetalAffineQuantizedMatmulFallsBackForUnsupportedVendoredMLXGroupSize() throws {
    let runtime = try EdgeMetalRuntime(
        configuration: MetalRuntimeConfiguration(useMLXQuantizedMatmul: true)
    )
    let executor = try MetalKernelExecutor(runtime: runtime)
    let rows = 1
    let inner = 2
    let columns = 4
    let lhsValues: [Float] = [1, 2]
    let rawRows: [[UInt32]] = [
        [1, 0],
        [0, 1],
        [2, 0],
        [0, 2],
    ]
    let scales = Array(repeating: Float(1), count: columns)
    let biases = Array(repeating: Float.zero, count: columns)
    let denseWeights = affineDenseValues(
        rawRows: rawRows,
        scales: scales,
        biases: biases,
        groupSize: inner
    )
    let weights = try EdgeQuantizedTensor(
        shape: [columns, inner],
        packedShape: [columns, 1],
        scaleShape: [columns, 1],
        groupSize: inner,
        bits: 4,
        packedValues: packQuantizedRows(rawRows, bits: 4),
        scales: scales,
        biases: biases
    )
    let lhs = try EdgeTensor(float32: lhsValues, shape: EdgeTensorShape([rows, inner]), runtime: runtime)

    let result = try executor.affineQuantizedMatmul(lhs, weights: weights, transpose: true)
    let expected = denseMatmulTransposed(lhsValues, rows: rows, inner: inner, weights: denseWeights, outputs: columns)
    let error = try NumericComparison.maxAbsoluteError(try result.readFloat32(), expected)

    #expect(result.shape == EdgeTensorShape([rows, columns]))
    #expect(error < 1e-4)
    #expect(executor.lastExecutionStats?.operationName == "edge_affine_quantized_matmul")
}

@Test func customMetalAffineQuantizedMatmulFallsBackWhenVendoredMLXCannotRepresentShape() throws {
    let runtime = try EdgeMetalRuntime(
        configuration: MetalRuntimeConfiguration(useMLXQuantizedMatmul: true)
    )
    let executor = try MetalKernelExecutor(runtime: runtime)
    let lhsValues: [Float] = [
        1, 2, 3,
        4, 5, 6,
    ]
    let rawRows: [[UInt32]] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ]
    let scales: [Float] = [
        1, 10,
        0.5, 2,
        1, 0.25,
    ]
    let biases: [Float] = [
        0, -1,
        1, 0,
        -1, 10,
    ]
    let denseWeights = affineDenseValues(
        rawRows: rawRows,
        scales: scales,
        biases: biases,
        groupSize: 2
    )
    let weights = try EdgeQuantizedTensor(
        shape: [3, 4],
        packedShape: [3, 1],
        scaleShape: [3, 2],
        groupSize: 2,
        bits: 4,
        packedValues: packQuantizedRows(rawRows, bits: 4),
        scales: scales,
        biases: biases
    )
    let lhs = try EdgeTensor(float32: lhsValues, shape: EdgeTensorShape([2, 3]), runtime: runtime)

    let result = try executor.affineQuantizedMatmul(lhs, weights: weights)
    let expected = denseMatmul(lhsValues, rows: 2, inner: 3, weights: denseWeights, columns: 4)
    let error = try NumericComparison.maxAbsoluteError(try result.readFloat32(), expected)

    #expect(result.shape == EdgeTensorShape([2, 4]))
    #expect(error < 1e-4)
    #expect(executor.lastExecutionStats?.operationName == "edge_affine_quantized_matmul")
}

@Test func customMetalAffineQuantizedQMVTransposedMatchesCPUOracle() throws {
    let runtime = try EdgeMetalRuntime(
        configuration: MetalRuntimeConfiguration(useMLXQuantizedMatmul: false)
    )
    let executor = try MetalKernelExecutor(runtime: runtime)
    let lhsValues = (0..<128).map { Float(Int($0 % 17) - 8) / 9.0 }

    for bits in [4, 6] {
        let maxRaw = UInt32((1 << bits) - 1)
        let rawRows: [[UInt32]] = (0..<8).map { row in
            (0..<64).map { column in
                UInt32((row * 11 + column * 7) % Int(maxRaw + 1))
            }
        }
        let scales = (0..<8).map { Float($0 + 1) / Float(maxRaw + 1) }
        let biases = (0..<8).map { Float($0 - 3) / 32.0 }
        let denseWeights = affineDenseValues(
            rawRows: rawRows,
            scales: scales,
            biases: biases,
            groupSize: 64
        )
        let packedWordsPerRow = (64 * bits + 31) / 32
        let weights = try EdgeQuantizedTensor(
            shape: [8, 64],
            packedShape: [8, packedWordsPerRow],
            scaleShape: [8, 1],
            groupSize: 64,
            bits: bits,
            packedValues: packQuantizedRows(rawRows, bits: bits),
            scales: scales,
            biases: biases
        )
        let lhs = try EdgeTensor(float32: lhsValues, shape: EdgeTensorShape([2, 64]), runtime: runtime)

        let result = try executor.affineQuantizedMatmul(lhs, weights: weights, transpose: true)
        let expected = denseMatmulTransposed(lhsValues, rows: 2, inner: 64, weights: denseWeights, outputs: 8)
        let error = try NumericComparison.maxAbsoluteError(try result.readFloat32(), expected)

        #expect(result.shape == EdgeTensorShape([2, 8]))
        #expect(error < 1e-3)
        if runtime.configuration.useMLXQuantizedMatmul == false {
            #expect(executor.lastExecutionStats?.operationName == "edge_affine_qmv_transposed")
        }
    }
}

@Test func customMetalAffineQuantizedQMVTransposedArgmaxMatchesFullLogits() throws {
    let runtime = try EdgeMetalRuntime(
        configuration: MetalRuntimeConfiguration(useMLXQuantizedMatmul: false)
    )
    let executor = try MetalKernelExecutor(runtime: runtime)

    for bits in [4, 6] {
        let inner = bits == 4 ? 512 : 256
        let columns = 17
        let groupSize = 64
        let groupCount = inner / groupSize
        let maxRaw = UInt32((1 << bits) - 1)
        let lhsValues = (0..<inner).map { Float(Int($0 % 23) - 11) / 13.0 }
        let rawRows: [[UInt32]] = (0..<columns).map { row in
            (0..<inner).map { column in
                UInt32((row * 13 + column * 5 + bits) % Int(maxRaw + 1))
            }
        }
        let scales = (0..<(columns * groupCount)).map { Float(($0 % 11) + 1) / Float(maxRaw + 1) }
        let biases = (0..<(columns * groupCount)).map { Float(($0 % 7) - 3) / 64.0 }
        let weights = try EdgeQuantizedTensor(
            shape: [columns, inner],
            packedShape: [columns, (inner * bits + 31) / 32],
            scaleShape: [columns, groupCount],
            groupSize: groupSize,
            bits: bits,
            packedValues: packQuantizedRows(rawRows, bits: bits),
            scales: scales,
            biases: biases
        )
        let lhs = try EdgeTensor(float32: lhsValues, shape: EdgeTensorShape([1, inner]), runtime: runtime)

        let logits = try executor.affineQuantizedMatmul(lhs, weights: weights, transpose: true)
        let values = try logits.readFloat32()
        let expected = values.enumerated().max { lhs, rhs in
            if lhs.element == rhs.element {
                return lhs.offset > rhs.offset
            }
            return lhs.element < rhs.element
        }!
        let token = try executor.affineQuantizedMatmulArgmax(lhs, weights: weights, transpose: true)

        #expect(token.tokenId == expected.offset)
        #expect(abs(token.logit - expected.element) < 1e-3)
        #expect(executor.lastExecutionStats?.operationName == "edge_affine_qmv_transposed_argmax")
    }
}

@Test func customMetalAffineQuantizedMatmulSupportsEightBitWeights() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let lhsValues: [Float] = [1, 2]
    let rawRows: [[UInt32]] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    let scales: [Float] = [0.5, 2]
    let biases: [Float] = [0, -1]
    let denseWeights = affineDenseValues(
        rawRows: rawRows,
        scales: scales,
        biases: biases,
        groupSize: 4
    )
    let weights = try EdgeQuantizedTensor(
        shape: [2, 4],
        packedShape: [2, 1],
        scaleShape: [2, 1],
        groupSize: 4,
        bits: 8,
        packedValues: packQuantizedRows(rawRows, bits: 8),
        scales: scales,
        biases: biases
    )
    let lhs = try EdgeTensor(float32: lhsValues, shape: EdgeTensorShape([1, 2]), runtime: runtime)

    let result = try executor.affineQuantizedMatmul(lhs, weights: weights)
    let expected = denseMatmul(lhsValues, rows: 1, inner: 2, weights: denseWeights, columns: 4)
    let error = try NumericComparison.maxAbsoluteError(try result.readFloat32(), expected)

    #expect(result.shape == EdgeTensorShape([1, 4]))
    #expect(error < 1e-4)
}

private func affineDenseValues(
    rawRows: [[UInt32]],
    scales: [Float],
    biases: [Float]?,
    groupSize: Int
) -> [Float] {
    let scaleColumns = rawRows[0].count / groupSize
    return rawRows.enumerated().flatMap { row, values in
        values.enumerated().map { column, value in
            let scaleIndex = row * scaleColumns + column / groupSize
            return Float(value) * scales[scaleIndex] + (biases?[scaleIndex] ?? 0)
        }
    }
}

private func denseMatmul(
    _ lhs: [Float],
    rows: Int,
    inner: Int,
    weights: [Float],
    columns: Int
) -> [Float] {
    var output = Array(repeating: Float.zero, count: rows * columns)
    for row in 0..<rows {
        for column in 0..<columns {
            var acc = Float.zero
            for index in 0..<inner {
                acc += lhs[row * inner + index] * weights[index * columns + column]
            }
            output[row * columns + column] = acc
        }
    }
    return output
}

private func denseMatmulTransposed(
    _ lhs: [Float],
    rows: Int,
    inner: Int,
    weights: [Float],
    outputs: Int
) -> [Float] {
    var output = Array(repeating: Float.zero, count: rows * outputs)
    for row in 0..<rows {
        for column in 0..<outputs {
            var acc = Float.zero
            for index in 0..<inner {
                acc += lhs[row * inner + index] * weights[column * inner + index]
            }
            output[row * outputs + column] = acc
        }
    }
    return output
}

private func packQuantizedRows(_ rows: [[UInt32]], bits: Int) -> [UInt32] {
    rows.flatMap { packQuantizedWords($0, bits: bits) }
}

private func packQuantizedWords(_ values: [UInt32], bits: Int) -> [UInt32] {
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

private func currentPhysicalFootprintBytes() -> UInt64? {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.stride / MemoryLayout<natural_t>.stride
    )
    let result = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
            task_info(
                mach_task_self_,
                task_flavor_t(TASK_VM_INFO),
                rebound,
                &count
            )
        }
    }
    guard result == KERN_SUCCESS else {
        return nil
    }
    return UInt64(info.phys_footprint)
}
