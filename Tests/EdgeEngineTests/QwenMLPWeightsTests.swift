// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Testing
@testable import EdgeEngine

@Test func qwenMLPWeightsLoadAndRunRuntimeLayoutWeights() throws {
    let architecture = try makeMLPArchitecture(intermediateSize: 3)
    let file = try SafeTensorsFile(data: makeRuntimeMLPWeightsFileData())
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let hiddenStates = try EdgeTensor(float32: [1, 2], shape: EdgeTensorShape([1, 2]), runtime: runtime)

    let weights = try QwenMLPWeights.loadRuntimeLayout(
        layerIndex: 0,
        architecture: architecture,
        weights: file,
        runtime: runtime
    )
    let output = try weights(hiddenStates: hiddenStates, executor: executor)
    let expected = try CPUReferenceOps.swiglu(gate: [1, 2, 0], up: [1, 4, 6])
    let error = try NumericComparison.maxAbsoluteError(try output.readFloat32(), [expected[0], expected[1]])

    #expect(weights.gate.shape == EdgeTensorShape([2, 3]))
    #expect(weights.up.shape == EdgeTensorShape([2, 3]))
    #expect(weights.down.shape == EdgeTensorShape([3, 2]))
    #expect(output.shape == EdgeTensorShape([1, 2]))
    #expect(error < 1e-5)
}

@Test func qwenMLPWeightsLoadAndRunHuggingFaceLayoutWeights() throws {
    let architecture = try makeMLPArchitecture(intermediateSize: 3)
    let file = try SafeTensorsFile(data: makeHuggingFaceMLPWeightsFileData())
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let hiddenStates = try EdgeTensor(float32: [1, 2], shape: EdgeTensorShape([1, 2]), runtime: runtime)

    let weights = try QwenMLPWeights.loadHuggingFaceLayout(
        layerIndex: 0,
        architecture: architecture,
        weights: file,
        runtime: runtime
    )
    let output = try weights(hiddenStates: hiddenStates, executor: executor)
    let expected = try CPUReferenceOps.swiglu(gate: [1, 2, 0], up: [1, 4, 6])
    let error = try NumericComparison.maxAbsoluteError(try output.readFloat32(), [expected[0], expected[1]])

    #expect(try weights.gate.readFloat32() == [
        1, 0, 2,
        0, 1, -1,
    ])
    #expect(output.shape == EdgeTensorShape([1, 2]))
    #expect(error < 1e-5)
}

@Test func qwenMLPWeightsRejectWrongShape() throws {
    let architecture = try makeMLPArchitecture(intermediateSize: 3)
    let file = try SafeTensorsFile(data: makeRuntimeMLPWeightsFileData(upShape: [2, 2]))
    let runtime = try EdgeMetalRuntime()

    var rejectedShape = false
    do {
        _ = try QwenMLPWeights.loadRuntimeLayout(
            layerIndex: 0,
            architecture: architecture,
            weights: file,
            runtime: runtime
        )
        Issue.record("MLP loader must reject non-runtime-layout shapes.")
    } catch QwenProjectionWeightError.invalidWeightShape {
        rejectedShape = true
    }
    #expect(rejectedShape)
}

private func makeMLPArchitecture(intermediateSize: Int) throws -> QwenHybridArchitecture {
    try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 128,
        hiddenSize: 2,
        intermediateSize: intermediateSize,
        attentionHeadCount: 2,
        keyValueHeadCount: 1,
        contextLength: 32,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000,
        layerKinds: [.fullAttention, .gdn]
    )
}

private func makeRuntimeMLPWeightsFileData(upShape: [Int] = [2, 3]) -> Data {
    makeMLPWeightsFileData(
        gateShape: [2, 3],
        gateValues: [
            1, 0, 2,
            0, 1, -1,
        ],
        upShape: upShape,
        upValues: upShape == [2, 3] ? [
            1, 2, 0,
            0, 1, 3,
        ] : [
            1, 2,
            0, 1,
        ],
        downShape: [3, 2],
        downValues: [
            1, 0,
            0, 1,
            2, 3,
        ]
    )
}

private func makeHuggingFaceMLPWeightsFileData() -> Data {
    makeMLPWeightsFileData(
        gateShape: [3, 2],
        gateValues: [
            1, 0,
            0, 1,
            2, -1,
        ],
        upShape: [3, 2],
        upValues: [
            1, 0,
            2, 1,
            0, 3,
        ],
        downShape: [2, 3],
        downValues: [
            1, 0, 2,
            0, 1, 3,
        ]
    )
}

private func makeMLPWeightsFileData(
    gateShape: [Int],
    gateValues: [Float],
    upShape: [Int],
    upValues: [Float],
    downShape: [Int],
    downValues: [Float]
) -> Data {
    var payload = Data()
    let gateData = mlpFloatData(gateValues)
    let upData = mlpFloatData(upValues)
    let downData = mlpFloatData(downValues)
    payload.append(gateData)
    payload.append(upData)
    payload.append(downData)

    let gateEnd = gateData.count
    let upEnd = gateEnd + upData.count
    let downEnd = upEnd + downData.count
    let headerJSON = """
    {
      "model.layers.0.mlp.gate_proj.weight": {
        "dtype": "F32",
        "shape": \(mlpShapeJSON(gateShape)),
        "data_offsets": [0, \(gateEnd)]
      },
      "model.layers.0.mlp.up_proj.weight": {
        "dtype": "F32",
        "shape": \(mlpShapeJSON(upShape)),
        "data_offsets": [\(gateEnd), \(upEnd)]
      },
      "model.layers.0.mlp.down_proj.weight": {
        "dtype": "F32",
        "shape": \(mlpShapeJSON(downShape)),
        "data_offsets": [\(upEnd), \(downEnd)]
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

private func mlpShapeJSON(_ shape: [Int]) -> String {
    "[" + shape.map(String.init).joined(separator: ", ") + "]"
}

private func mlpFloatData(_ values: [Float]) -> Data {
    values.withUnsafeBufferPointer { buffer in
        Data(buffer: buffer)
    }
}
