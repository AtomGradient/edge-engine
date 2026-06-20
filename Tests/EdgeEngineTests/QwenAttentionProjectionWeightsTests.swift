// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Testing
@testable import EdgeEngine

@Test func qwenAttentionProjectionWeightsLoadAndProjectRuntimeLayoutWeights() throws {
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
    let file = try SafeTensorsFile(data: makeProjectionWeightsFileData())
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let hiddenStates = try EdgeTensor(
        float32: [1, 2],
        shape: EdgeTensorShape([1, 2]),
        runtime: runtime
    )

    let weights = try QwenAttentionProjectionWeights.loadRuntimeLayout(
        layerIndex: 0,
        architecture: architecture,
        weights: file,
        runtime: runtime
    )
    let outputs = try weights.project(hiddenStates: hiddenStates, executor: executor)

    #expect(try outputs.query.readFloat32() == [1, 3])
    #expect(try outputs.queryGate?.readFloat32() == [2, 4])
    #expect(try outputs.key.readFloat32() == [11])
    #expect(try outputs.value.readFloat32() == [3])
}

@Test func qwenAttentionProjectionWeightsRejectWrongRuntimeLayoutShape() throws {
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
    let file = try SafeTensorsFile(data: makeProjectionWeightsFileData(kShape: [1, 2]))
    let runtime = try EdgeMetalRuntime()

    var rejectedShape = false
    do {
        _ = try QwenAttentionProjectionWeights.loadRuntimeLayout(
            layerIndex: 0,
            architecture: architecture,
            weights: file,
            runtime: runtime
        )
        Issue.record("Projection loader must reject non-runtime-layout shapes.")
    } catch QwenProjectionWeightError.invalidWeightShape {
        rejectedShape = true
    }
    #expect(rejectedShape)
}

@Test func qwenAttentionProjectionWeightsLoadAndProjectHuggingFaceLayoutWeights() throws {
    let architecture = try QwenHybridArchitecture(
        family: .qwen36,
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
    let file = try SafeTensorsFile(data: makeHuggingFaceProjectionWeightsFileData())
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let hiddenStates = try EdgeTensor(
        float32: [1, 2],
        shape: EdgeTensorShape([1, 2]),
        runtime: runtime
    )

    let weights = try QwenAttentionProjectionWeights.loadHuggingFaceLayout(
        layerIndex: 0,
        architecture: architecture,
        weights: file,
        runtime: runtime
    )
    let outputs = try weights.project(hiddenStates: hiddenStates, executor: executor)

    #expect(try outputs.query.readFloat32() == [5, 17])
    #expect(try outputs.queryGate?.readFloat32() == [11, 23])
    #expect(try outputs.key.readFloat32() == [11])
    #expect(try outputs.value.readFloat32() == [3])
}

@Test func qwenAttentionProjectionWeightsAcceptQwen36DecoupledHeadDimension() throws {
    let architecture = try QwenHybridArchitecture(
        family: .qwen36,
        vocabularySize: 128,
        hiddenSize: 3,
        intermediateSize: 12,
        attentionHeadCount: 2,
        keyValueHeadCount: 1,
        headDimension: 2,
        contextLength: 32,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000,
        layerKinds: [.fullAttention, .gdn]
    )
    let file = try SafeTensorsFile(data: makeDecoupledQwen36HuggingFaceProjectionWeightsFileData())
    let runtime = try EdgeMetalRuntime()

    let weights = try QwenAttentionProjectionWeights.loadHuggingFaceLayout(
        layerIndex: 0,
        architecture: architecture,
        weights: file,
        runtime: runtime
    )

    #expect(architecture.attentionHiddenSize == 4)
    #expect(architecture.queryProjectionHiddenSize == 8)
    #expect(weights.query.shape == EdgeTensorShape([3, 8]))
    #expect(weights.key.shape == EdgeTensorShape([3, 2]))
    #expect(weights.value.shape == EdgeTensorShape([3, 2]))
}


private func makeProjectionWeightsFileData(kShape: [Int] = [2, 1]) -> Data {
    var payload = Data()
    payload.append(projectionFloatData([
        1, 0, 3, 0,
        0, 1, 0, 2,
    ]))
    payload.append(projectionFloatData([3, 4]))
    payload.append(projectionFloatData([-1, 2]))

    let headerJSON = """
    {
      "model.layers.0.self_attn.q_proj.weight": {
        "dtype": "F32",
        "shape": [2, 4],
        "data_offsets": [0, 32]
      },
      "model.layers.0.self_attn.k_proj.weight": {
        "dtype": "F32",
        "shape": \(shapeJSON(kShape)),
        "data_offsets": [32, 40]
      },
      "model.layers.0.self_attn.v_proj.weight": {
        "dtype": "F32",
        "shape": [2, 1],
        "data_offsets": [40, 48]
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

private func makeHuggingFaceProjectionWeightsFileData() -> Data {
    var payload = Data()
    payload.append(projectionFloatData([
        1, 2,
        3, 4,
        5, 6,
        7, 8,
    ]))
    payload.append(projectionFloatData([3, 4]))
    payload.append(projectionFloatData([-1, 2]))

    let headerJSON = """
    {
      "model.layers.0.self_attn.q_proj.weight": {
        "dtype": "F32",
        "shape": [4, 2],
        "data_offsets": [0, 32]
      },
      "model.layers.0.self_attn.k_proj.weight": {
        "dtype": "F32",
        "shape": [1, 2],
        "data_offsets": [32, 40]
      },
      "model.layers.0.self_attn.v_proj.weight": {
        "dtype": "F32",
        "shape": [1, 2],
        "data_offsets": [40, 48]
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

private func makeDecoupledQwen36HuggingFaceProjectionWeightsFileData() -> Data {
    var payload = Data()
    payload.append(projectionFloatData([
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15,
        16, 17, 18,
        19, 20, 21,
        22, 23, 24,
    ]))
    payload.append(projectionFloatData([
        25, 26, 27,
        28, 29, 30,
    ]))
    payload.append(projectionFloatData([
        31, 32, 33,
        34, 35, 36,
    ]))

    let headerJSON = """
    {
      "model.layers.0.self_attn.q_proj.weight": {
        "dtype": "F32",
        "shape": [8, 3],
        "data_offsets": [0, 96]
      },
      "model.layers.0.self_attn.k_proj.weight": {
        "dtype": "F32",
        "shape": [2, 3],
        "data_offsets": [96, 120]
      },
      "model.layers.0.self_attn.v_proj.weight": {
        "dtype": "F32",
        "shape": [2, 3],
        "data_offsets": [120, 144]
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

private func shapeJSON(_ shape: [Int]) -> String {
    "[" + shape.map(String.init).joined(separator: ", ") + "]"
}

private func projectionFloatData(_ values: [Float]) -> Data {
    values.withUnsafeBufferPointer { buffer in
        Data(buffer: buffer)
    }
}
