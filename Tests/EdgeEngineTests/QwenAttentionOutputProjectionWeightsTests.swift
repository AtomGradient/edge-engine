// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Testing
@testable import EdgeEngine

@Test func qwenAttentionOutputProjectionWeightsLoadAndProjectRuntimeLayoutWeights() throws {
    let architecture = try makeOutputProjectionArchitecture()
    let file = try SafeTensorsFile(data: makeOutputProjectionWeightsFileData())
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let attentionOutput = try EdgeTensor(float32: [1, 2], shape: EdgeTensorShape([1, 2]), runtime: runtime)
    let gate = try EdgeTensor(float32: [0, 0], shape: EdgeTensorShape([1, 2]), runtime: runtime)

    let weights = try QwenAttentionOutputProjectionWeights.loadRuntimeLayout(
        layerIndex: 0,
        architecture: architecture,
        weights: file,
        runtime: runtime
    )
    let output = try weights.project(attentionOutput: attentionOutput, queryGate: gate, executor: executor)

    #expect(weights.weight.shape == EdgeTensorShape([2, 2]))
    #expect(try output.readFloat32() == [3.5, 5])
}

@Test func qwenAttentionOutputProjectionWeightsLoadAndProjectHuggingFaceLayoutWeights() throws {
    let architecture = try makeOutputProjectionArchitecture()
    let file = try SafeTensorsFile(data: makeOutputProjectionWeightsFileData(
        shape: [2, 2],
        values: [
            1, 3,
            2, 4,
        ]
    ))
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let attentionOutput = try EdgeTensor(float32: [1, 2], shape: EdgeTensorShape([1, 2]), runtime: runtime)
    let gate = try EdgeTensor(float32: [0, 0], shape: EdgeTensorShape([1, 2]), runtime: runtime)

    let weights = try QwenAttentionOutputProjectionWeights.loadHuggingFaceLayout(
        layerIndex: 0,
        architecture: architecture,
        weights: file,
        runtime: runtime
    )
    let output = try weights.project(attentionOutput: attentionOutput, queryGate: gate, executor: executor)

    #expect(weights.weight.shape == EdgeTensorShape([2, 2]))
    #expect(try weights.weight.readFloat32() == [1, 2, 3, 4])
    #expect(try output.readFloat32() == [3.5, 5])
}

@Test func qwenAttentionOutputProjectionWeightsRejectWrongShape() throws {
    let architecture = try makeOutputProjectionArchitecture()
    let file = try SafeTensorsFile(data: makeOutputProjectionWeightsFileData(
        shape: [2, 1],
        values: [1, 2]
    ))
    let runtime = try EdgeMetalRuntime()

    var rejectedShape = false
    do {
        _ = try QwenAttentionOutputProjectionWeights.loadRuntimeLayout(
            layerIndex: 0,
            architecture: architecture,
            weights: file,
            runtime: runtime
        )
        Issue.record("Output projection loader must reject non-runtime-layout shapes.")
    } catch QwenProjectionWeightError.invalidWeightShape {
        rejectedShape = true
    }
    #expect(rejectedShape)
}

private func makeOutputProjectionArchitecture() throws -> QwenHybridArchitecture {
    try QwenHybridArchitecture(
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
}

private func makeOutputProjectionWeightsFileData(
    shape: [Int] = [2, 2],
    values: [Float] = [
        1, 2,
        3, 4,
    ]
) -> Data {
    let payload = outputProjectionFloatData(values)
    let headerJSON = """
    {
      "model.layers.0.self_attn.o_proj.weight": {
        "dtype": "F32",
        "shape": \(outputProjectionShapeJSON(shape)),
        "data_offsets": [0, \(payload.count)]
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

private func outputProjectionShapeJSON(_ shape: [Int]) -> String {
    "[" + shape.map(String.init).joined(separator: ", ") + "]"
}

private func outputProjectionFloatData(_ values: [Float]) -> Data {
    values.withUnsafeBufferPointer { buffer in
        Data(buffer: buffer)
    }
}
