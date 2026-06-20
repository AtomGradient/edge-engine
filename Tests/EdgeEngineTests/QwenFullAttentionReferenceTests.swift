// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Testing
@testable import EdgeEngine

@Test func qwenFullAttentionReferenceRunsProjectionAndAttentionSmoke() throws {
    let architecture = try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 128,
        hiddenSize: 2,
        intermediateSize: 8,
        attentionHeadCount: 2,
        keyValueHeadCount: 1,
        contextLength: 32,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000,
        layerKinds: [.fullAttention, .gdn]
    )
    let file = try SafeTensorsFile(data: makeFullAttentionReferenceWeightsFileData())
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let projectionWeights = try QwenAttentionProjectionWeights.loadRuntimeLayout(
        layerIndex: 0,
        architecture: architecture,
        weights: file,
        runtime: runtime
    )
    let reference = try QwenFullAttentionReference(
        architecture: architecture,
        layerIndex: 0,
        projectionWeights: projectionWeights
    )
    let hiddenStates = try EdgeTensor(
        float32: [1, 2],
        shape: EdgeTensorShape([1, 2]),
        runtime: runtime
    )

    let output = try reference.attentionOutput(hiddenStates: hiddenStates, executor: executor)

    #expect(output == [3, 3])
}

@Test func qwenFullAttentionReferenceReturnsTensorOutput() throws {
    let architecture = try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 128,
        hiddenSize: 2,
        intermediateSize: 8,
        attentionHeadCount: 2,
        keyValueHeadCount: 1,
        contextLength: 32,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000,
        layerKinds: [.fullAttention, .gdn]
    )
    let file = try SafeTensorsFile(data: makeFullAttentionReferenceWeightsFileData())
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let projectionWeights = try QwenAttentionProjectionWeights.loadRuntimeLayout(
        layerIndex: 0,
        architecture: architecture,
        weights: file,
        runtime: runtime
    )
    let reference = try QwenFullAttentionReference(
        architecture: architecture,
        layerIndex: 0,
        projectionWeights: projectionWeights
    )
    let hiddenStates = try EdgeTensor(
        float32: [1, 2],
        shape: EdgeTensorShape([1, 2]),
        runtime: runtime
    )

    let output = try reference.attentionTensor(hiddenStates: hiddenStates, executor: executor)

    #expect(output.shape == EdgeTensorShape([1, 2]))
    #expect(try output.readFloat32() == [3, 3])
}

@Test func qwenFullAttentionReferenceUsesKVCacheForIncrementalDecode() throws {
    let architecture = try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 128,
        hiddenSize: 2,
        intermediateSize: 8,
        attentionHeadCount: 2,
        keyValueHeadCount: 1,
        contextLength: 32,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000,
        layerKinds: [.fullAttention, .gdn]
    )
    let file = try SafeTensorsFile(data: makeFullAttentionReferenceWeightsFileData())
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let projectionWeights = try QwenAttentionProjectionWeights.loadRuntimeLayout(
        layerIndex: 0,
        architecture: architecture,
        weights: file,
        runtime: runtime
    )
    let reference = try QwenFullAttentionReference(
        architecture: architecture,
        layerIndex: 0,
        projectionWeights: projectionWeights
    )
    let fullHiddenStates = try EdgeTensor(
        float32: [
            1, 2,
            2, 1,
        ],
        shape: EdgeTensorShape([2, 2]),
        runtime: runtime
    )
    let cache = try QwenKVCache(
        shape: try QwenKVCacheShape.shape(for: architecture, layerIndex: 0, capacity: 4),
        runtime: runtime
    )

    let fullOutput = try reference.attentionOutput(hiddenStates: fullHiddenStates, executor: executor)
    let firstTokenOutput = try reference.attentionOutput(
        hiddenStates: try EdgeTensor(float32: [1, 2], shape: EdgeTensorShape([1, 2]), runtime: runtime),
        executor: executor,
        kvCache: cache
    )
    let secondTokenOutput = try reference.attentionOutput(
        hiddenStates: try EdgeTensor(float32: [2, 1], shape: EdgeTensorShape([1, 2]), runtime: runtime),
        executor: executor,
        kvCache: cache
    )
    let error = try NumericComparison.maxAbsoluteError(fullOutput, firstTokenOutput + secondTokenOutput)

    #expect(cache.tokenCount == 2)
    #expect(error < 1e-5)
}

@Test func qwenFullAttentionReferenceMasksFutureTokensDuringPrefill() throws {
    let architecture = try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 128,
        hiddenSize: 2,
        intermediateSize: 8,
        attentionHeadCount: 2,
        keyValueHeadCount: 1,
        contextLength: 32,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000,
        layerKinds: [.fullAttention, .gdn]
    )
    let file = try SafeTensorsFile(data: makeFullAttentionReferenceWeightsFileData())
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let projectionWeights = try QwenAttentionProjectionWeights.loadRuntimeLayout(
        layerIndex: 0,
        architecture: architecture,
        weights: file,
        runtime: runtime
    )
    let reference = try QwenFullAttentionReference(
        architecture: architecture,
        layerIndex: 0,
        projectionWeights: projectionWeights
    )
    let baselineHiddenStates = try EdgeTensor(
        float32: [
            1, 2,
            2, 1,
        ],
        shape: EdgeTensorShape([2, 2]),
        runtime: runtime
    )
    let alteredFutureHiddenStates = try EdgeTensor(
        float32: [
            1, 2,
            100, -100,
        ],
        shape: EdgeTensorShape([2, 2]),
        runtime: runtime
    )

    let baselineOutput = try reference.attentionOutput(
        hiddenStates: baselineHiddenStates,
        executor: executor
    )
    let alteredFutureOutput = try reference.attentionOutput(
        hiddenStates: alteredFutureHiddenStates,
        executor: executor
    )
    let firstTokenError = try NumericComparison.maxAbsoluteError(
        Array(baselineOutput[0..<2]),
        Array(alteredFutureOutput[0..<2])
    )

    #expect(firstTokenError < 1e-5)
}

@Test func qwenFullAttentionReferenceAppliesQueryGateAndOutputProjection() throws {
    let architecture = try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 128,
        hiddenSize: 2,
        intermediateSize: 8,
        attentionHeadCount: 2,
        keyValueHeadCount: 1,
        contextLength: 32,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000,
        layerKinds: [.fullAttention, .gdn]
    )
    let file = try SafeTensorsFile(data: makeFullAttentionReferenceWeightsFileData())
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let projectionWeights = try QwenAttentionProjectionWeights.loadRuntimeLayout(
        layerIndex: 0,
        architecture: architecture,
        weights: file,
        runtime: runtime
    )
    let outputProjectionWeights = try QwenAttentionOutputProjectionWeights.loadRuntimeLayout(
        layerIndex: 0,
        architecture: architecture,
        weights: file,
        runtime: runtime
    )
    let reference = try QwenFullAttentionReference(
        architecture: architecture,
        layerIndex: 0,
        projectionWeights: projectionWeights,
        outputProjectionWeights: outputProjectionWeights
    )
    let hiddenStates = try EdgeTensor(
        float32: [1, 2],
        shape: EdgeTensorShape([1, 2]),
        runtime: runtime
    )

    let output = try reference.attentionOutput(hiddenStates: hiddenStates, executor: executor)
    let expected = try CPUReferenceOps.sigmoidMultiply([3, 3], gate: [2, 4])
    let error = try NumericComparison.maxAbsoluteError(output, expected)

    #expect(error < 1e-5)
}

@Test func qwenQuantizedFullAttentionReferenceMatchesFloatReference() throws {
    let architecture = try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 128,
        hiddenSize: 2,
        intermediateSize: 8,
        attentionHeadCount: 2,
        keyValueHeadCount: 1,
        contextLength: 32,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000,
        layerKinds: [.fullAttention, .gdn]
    )
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let query = try EdgeTensor(
        float32: [
            1, 0, 3, 0,
            0, 1, 0, 2,
        ],
        shape: EdgeTensorShape([2, 4]),
        runtime: runtime
    )
    let key = try EdgeTensor(
        float32: [1, 2],
        shape: EdgeTensorShape([2, 1]),
        runtime: runtime
    )
    let value = try EdgeTensor(
        float32: [2, 1],
        shape: EdgeTensorShape([2, 1]),
        runtime: runtime
    )
    let outputProjection = try EdgeTensor(
        float32: [
            1, 0,
            0, 1,
        ],
        shape: EdgeTensorShape([2, 2]),
        runtime: runtime
    )
    let floatReference = try QwenFullAttentionReference(
        architecture: architecture,
        layerIndex: 0,
        projectionWeights: QwenAttentionProjectionWeights(
            layerIndex: 0,
            query: query,
            key: key,
            value: value,
            queryHiddenSize: architecture.queryHiddenSize,
            queryHeadCount: architecture.attentionHeadCount,
            queryHeadDimension: architecture.attentionHeadDimension
        ),
        outputProjectionWeights: QwenAttentionOutputProjectionWeights(
            layerIndex: 0,
            weight: outputProjection
        )
    )
    let quantizedReference = try QwenQuantizedFullAttentionReference(
        architecture: architecture,
        layerIndex: 0,
        projectionWeights: QwenQuantizedAttentionProjectionWeights(
            layerIndex: 0,
            query: exactFullAttentionQuantizedRows([
                [1, 0],
                [0, 1],
                [3, 0],
                [0, 2],
            ]),
            key: exactFullAttentionQuantizedRows([[1, 2]]),
            value: exactFullAttentionQuantizedRows([[2, 1]]),
            queryHiddenSize: architecture.queryHiddenSize,
            queryHeadCount: architecture.attentionHeadCount,
            queryHeadDimension: architecture.attentionHeadDimension
        ),
        outputProjectionWeights: QwenQuantizedAttentionOutputProjectionWeights(
            layerIndex: 0,
            weight: exactFullAttentionQuantizedRows([
                [1, 0],
                [0, 1],
            ])
        )
    )
    let hiddenStates = try EdgeTensor(
        float32: [
            1, 2,
            2, 1,
        ],
        shape: EdgeTensorShape([2, 2]),
        runtime: runtime
    )

    let expected = try floatReference.attentionOutput(hiddenStates: hiddenStates, executor: executor)
    let actual = try quantizedReference.attentionOutput(hiddenStates: hiddenStates, executor: executor)
    let error = try NumericComparison.maxAbsoluteError(actual, expected)

    #expect(error < 1e-5)
}

@Test func qwenQuantizedFullAttentionReferenceUsesKVCacheLikeFloatReference() throws {
    let architecture = try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 128,
        hiddenSize: 2,
        intermediateSize: 8,
        attentionHeadCount: 2,
        keyValueHeadCount: 1,
        contextLength: 32,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000,
        layerKinds: [.fullAttention, .gdn]
    )
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let floatReference = try QwenFullAttentionReference(
        architecture: architecture,
        layerIndex: 0,
        projectionWeights: QwenAttentionProjectionWeights(
            layerIndex: 0,
            query: EdgeTensor(
                float32: [
                    1, 0, 3, 0,
                    0, 1, 0, 2,
                ],
                shape: EdgeTensorShape([2, 4]),
                runtime: runtime
            ),
            key: EdgeTensor(float32: [1, 2], shape: EdgeTensorShape([2, 1]), runtime: runtime),
            value: EdgeTensor(float32: [2, 1], shape: EdgeTensorShape([2, 1]), runtime: runtime),
            queryHiddenSize: architecture.queryHiddenSize,
            queryHeadCount: architecture.attentionHeadCount,
            queryHeadDimension: architecture.attentionHeadDimension
        ),
        outputProjectionWeights: QwenAttentionOutputProjectionWeights(
            layerIndex: 0,
            weight: EdgeTensor(
                float32: [
                    1, 0,
                    0, 1,
                ],
                shape: EdgeTensorShape([2, 2]),
                runtime: runtime
            )
        )
    )
    let quantizedReference = try QwenQuantizedFullAttentionReference(
        architecture: architecture,
        layerIndex: 0,
        projectionWeights: QwenQuantizedAttentionProjectionWeights(
            layerIndex: 0,
            query: exactFullAttentionQuantizedRows([
                [1, 0],
                [0, 1],
                [3, 0],
                [0, 2],
            ]),
            key: exactFullAttentionQuantizedRows([[1, 2]]),
            value: exactFullAttentionQuantizedRows([[2, 1]]),
            queryHiddenSize: architecture.queryHiddenSize,
            queryHeadCount: architecture.attentionHeadCount,
            queryHeadDimension: architecture.attentionHeadDimension
        ),
        outputProjectionWeights: QwenQuantizedAttentionOutputProjectionWeights(
            layerIndex: 0,
            weight: exactFullAttentionQuantizedRows([
                [1, 0],
                [0, 1],
            ])
        )
    )
    let fullHiddenStates = try EdgeTensor(
        float32: [
            1, 2,
            2, 1,
        ],
        shape: EdgeTensorShape([2, 2]),
        runtime: runtime
    )
    let kvCache = try QwenKVCache(
        shape: try QwenKVCacheShape.shape(for: architecture, layerIndex: 0, capacity: 4),
        runtime: runtime
    )

    let expected = try floatReference.attentionOutput(hiddenStates: fullHiddenStates, executor: executor)
    let firstTokenOutput = try quantizedReference.attentionOutput(
        hiddenStates: EdgeTensor(float32: [1, 2], shape: EdgeTensorShape([1, 2]), runtime: runtime),
        executor: executor,
        kvCache: kvCache
    )
    let secondTokenOutput = try quantizedReference.attentionOutput(
        hiddenStates: EdgeTensor(float32: [2, 1], shape: EdgeTensorShape([1, 2]), runtime: runtime),
        executor: executor,
        kvCache: kvCache
    )
    let error = try NumericComparison.maxAbsoluteError(expected, firstTokenOutput + secondTokenOutput)

    #expect(kvCache.tokenCount == 2)
    #expect(error < 1e-5)
}

@Test func qwenFullAttentionReferenceRejectsGDNLayer() throws {
    let architecture = try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 128,
        hiddenSize: 2,
        intermediateSize: 8,
        attentionHeadCount: 2,
        keyValueHeadCount: 1,
        contextLength: 32,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000,
        layerKinds: [.fullAttention, .gdn]
    )
    let runtime = try EdgeMetalRuntime()
    let empty = try EdgeTensor(float32: [0, 0], shape: EdgeTensorShape([2, 1]), runtime: runtime)
    let projectionWeights = QwenAttentionProjectionWeights(
        layerIndex: 1,
        query: empty,
        key: empty,
        value: empty
    )

    var rejectedGDNLayer = false
    do {
        _ = try QwenFullAttentionReference(
            architecture: architecture,
            layerIndex: 1,
            projectionWeights: projectionWeights
        )
        Issue.record("Full-attention reference path must not accept GDN layers.")
    } catch QwenFullAttentionReferenceError.layerIsNotFullAttention {
        rejectedGDNLayer = true
    }
    #expect(rejectedGDNLayer)
}

private func makeFullAttentionReferenceWeightsFileData() -> Data {
    var payload = Data()
    payload.append(fullAttentionFloatData([
        1, 0, 3, 0,
        0, 1, 0, 2,
    ]))
    payload.append(fullAttentionFloatData([3, 4]))
    payload.append(fullAttentionFloatData([-1, 2]))
    payload.append(fullAttentionFloatData([
        1, 0,
        0, 1,
    ]))

    let headerJSON = """
    {
      "model.layers.0.self_attn.q_proj.weight": {
        "dtype": "F32",
        "shape": [2, 4],
        "data_offsets": [0, 32]
      },
      "model.layers.0.self_attn.k_proj.weight": {
        "dtype": "F32",
        "shape": [2, 1],
        "data_offsets": [32, 40]
      },
      "model.layers.0.self_attn.v_proj.weight": {
        "dtype": "F32",
        "shape": [2, 1],
        "data_offsets": [40, 48]
      },
      "model.layers.0.self_attn.o_proj.weight": {
        "dtype": "F32",
        "shape": [2, 2],
        "data_offsets": [48, 64]
      }
    }
    """
    let headerData = headerJSON.data(using: .utf8)!
    var headerLength = UInt64(headerData.count).littleEndian
    var fileData = withUnsafeBytes(of: &headerLength) { Data($0) }
    fileData.append(headerData)
    fileData.append(payload)
    return fileData
}

private func fullAttentionFloatData(_ values: [Float]) -> Data {
    values.withUnsafeBufferPointer { buffer in
        Data(buffer: buffer)
    }
}

private func exactFullAttentionQuantizedRows(_ rows: [[UInt32]]) throws -> EdgeQuantizedTensor {
    let columns = rows.first?.count ?? 0
    return try EdgeQuantizedTensor(
        shape: [rows.count, columns],
        packedShape: [rows.count, (columns * 8 + 31) / 32],
        scaleShape: [rows.count, 1],
        groupSize: columns,
        bits: 8,
        packedValues: fullAttentionPackQuantizedRows(rows, bits: 8),
        scales: Array(repeating: 1, count: rows.count),
        biases: Array(repeating: 0, count: rows.count)
    )
}

private func fullAttentionPackQuantizedRows(_ rows: [[UInt32]], bits: Int) -> [UInt32] {
    rows.flatMap { fullAttentionPackQuantizedWords($0, bits: bits) }
}

private func fullAttentionPackQuantizedWords(_ values: [UInt32], bits: Int) -> [UInt32] {
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
