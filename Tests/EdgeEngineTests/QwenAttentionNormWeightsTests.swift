// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Testing
@testable import EdgeEngine

@Test func qwenAttentionNormWeightsLoadAndApplyRuntimeLayout() throws {
    let architecture = try normTestArchitecture()
    let file = try SafeTensorsFile(data: makeNormWeightsFileData())
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let query = try EdgeTensor(
        float32: [3, 4, 0, 5],
        shape: EdgeTensorShape([1, 4]),
        runtime: runtime
    )
    let key = try EdgeTensor(
        float32: [6, 8],
        shape: EdgeTensorShape([1, 2]),
        runtime: runtime
    )

    let weights = try QwenAttentionNormWeights.loadRuntimeLayout(
        layerIndex: 0,
        architecture: architecture,
        weights: file,
        runtime: runtime
    )
    let output = try weights.apply(
        query: query,
        key: key,
        architecture: architecture,
        executor: executor
    )
    let expectedQuery = try CPUReferenceOps.rmsNormByHead(
        [3, 4, 0, 5],
        weight: [1, 2],
        tokenCount: 1,
        headCount: 2,
        headDimension: 2,
        epsilon: 0
    )
    let expectedKey = try CPUReferenceOps.rmsNormByHead(
        [6, 8],
        weight: [3, 4],
        tokenCount: 1,
        headCount: 1,
        headDimension: 2,
        epsilon: 0
    )

    #expect(weights.query.shape == EdgeTensorShape([2]))
    #expect(weights.key.shape == EdgeTensorShape([2]))
    #expect(try NumericComparison.maxAbsoluteError(try output.query.readFloat32(), expectedQuery) < 1e-5)
    #expect(try NumericComparison.maxAbsoluteError(try output.key.readFloat32(), expectedKey) < 1e-5)
}

@Test func qwenAttentionNormWeightsRejectWrongShape() throws {
    let architecture = try normTestArchitecture()
    let file = try SafeTensorsFile(data: makeNormWeightsFileData(queryShape: [3]))
    let runtime = try EdgeMetalRuntime()

    do {
        _ = try QwenAttentionNormWeights.loadRuntimeLayout(
            layerIndex: 0,
            architecture: architecture,
            weights: file,
            runtime: runtime
        )
        Issue.record("Norm loader should reject q_norm weights that do not match head_dim.")
    } catch QwenAttentionNormWeightError.invalidWeightShape {
        return
    }
    Issue.record("Norm loader threw the wrong error for an invalid q_norm shape.")
}

private func normTestArchitecture() throws -> QwenHybridArchitecture {
    try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 128,
        hiddenSize: 4,
        intermediateSize: 16,
        attentionHeadCount: 2,
        keyValueHeadCount: 1,
        headDimension: 2,
        contextLength: 32,
        rmsNormEpsilon: 0,
        ropeTheta: 10_000,
        layerKinds: [.fullAttention, .gdn]
    )
}

private func makeNormWeightsFileData(queryShape: [Int] = [2]) -> Data {
    let queryValues: [Float] = queryShape == [2] ? [1, 2] : [1, 2, 3]
    var payload = Data()
    payload.append(normFloatData(queryValues))
    payload.append(normFloatData([3, 4]))
    let queryEnd = queryValues.count * MemoryLayout<Float>.stride
    let keyEnd = queryEnd + 2 * MemoryLayout<Float>.stride

    let headerJSON = """
    {
      "model.layers.0.self_attn.q_norm.weight": {
        "dtype": "F32",
        "shape": \(normShapeJSON(queryShape)),
        "data_offsets": [0, \(queryEnd)]
      },
      "model.layers.0.self_attn.k_norm.weight": {
        "dtype": "F32",
        "shape": [2],
        "data_offsets": [\(queryEnd), \(keyEnd)]
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

private func normShapeJSON(_ shape: [Int]) -> String {
    "[" + shape.map(String.init).joined(separator: ", ") + "]"
}

private func normFloatData(_ values: [Float]) -> Data {
    values.withUnsafeBufferPointer { buffer in
        Data(buffer: buffer)
    }
}
