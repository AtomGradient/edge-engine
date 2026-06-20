// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Testing
@testable import EdgeEngine

@Test func qwenModelOutputWeightsLoadExplicitLMHeadAndProjectLogits() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-output-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try writeOutputConfig(to: root)

    let embedName = "language_model.model.embed_tokens.weight"
    let normName = "language_model.model.norm.weight"
    let lmHeadName = "language_model.lm_head.weight"
    try writeOutputIndex(
        weightMap: outputWeightMap([
            embedName: "model-00001-of-00001.safetensors",
            normName: "model-00001-of-00001.safetensors",
            lmHeadName: "model-00001-of-00001.safetensors",
        ]),
        to: root
    )
    try writeOutputSafeTensorsEntries(
        [
            (embedName, "F32", [4, 2], outputFloatData(Array(repeating: 0, count: 8))),
            (normName, "F32", [2], outputFloatData([1, 1])),
            (lmHeadName, "F32", [4, 2], outputFloatData([
                1, 0,
                0, 1,
                1, 1,
                -1, 2,
            ])),
        ],
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )

    let store = QwenModelWeightStore(bundleIndex: try QwenModelBundleIndex.load(from: root))
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let hiddenStates = try EdgeTensor(float32: [3, 4], shape: EdgeTensorShape([1, 2]), runtime: runtime)

    let weights = try QwenModelOutputWeights.loadHuggingFaceLayout(weightStore: store, runtime: runtime)
    let logits = try weights.logits(hiddenStates: hiddenStates, executor: executor)
    let normalized = try CPUReferenceOps.rmsNorm([3, 4], weight: [1, 1], epsilon: 0)
    let expected = try CPUReferenceOps.matmul(
        normalized,
        rows: 1,
        inner: 2,
        [
            1, 0, 1, -1,
            0, 1, 1, 2,
        ],
        columns: 4
    )
    let error = try NumericComparison.maxAbsoluteError(try logits.readFloat32(), expected)

    #expect(!weights.usesTiedEmbeddings)
    #expect(weights.lmHead.shape == EdgeTensorShape([2, 4]))
    #expect(logits.shape == EdgeTensorShape([1, 4]))
    #expect(error < 1e-5)
}

@Test func qwenModelOutputWeightsFallsBackToTiedEmbeddings() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-output-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try writeOutputConfig(to: root)

    let embedName = "language_model.model.embed_tokens.weight"
    let normName = "language_model.model.norm.weight"
    try writeOutputIndex(
        weightMap: outputWeightMap([
            embedName: "model-00001-of-00001.safetensors",
            normName: "model-00001-of-00001.safetensors",
        ]),
        to: root
    )
    try writeOutputSafeTensorsEntries(
        [
            (embedName, "F32", [4, 2], outputFloatData([
                1, 0,
                0, 1,
                1, 1,
                -1, 2,
            ])),
            (normName, "F32", [2], outputFloatData([1, 1])),
        ],
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )

    let store = QwenModelWeightStore(bundleIndex: try QwenModelBundleIndex.load(from: root))
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let hiddenStates = try EdgeTensor(float32: [3, 4], shape: EdgeTensorShape([1, 2]), runtime: runtime)

    let weights = try QwenModelOutputWeights.loadHuggingFaceLayout(weightStore: store, runtime: runtime)
    let logits = try weights.logits(hiddenStates: hiddenStates, executor: executor)

    #expect(weights.usesTiedEmbeddings)
    #expect(weights.lmHead.shape == EdgeTensorShape([2, 4]))
    #expect(logits.shape == EdgeTensorShape([1, 4]))
}

@Test func qwenQuantizedModelOutputWeightsLoadLMHeadAndProjectLogits() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-output-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try writeOutputConfig(quantization: QwenQuantizationProfile(groupSize: 2, bits: 4), to: root)

    let embedName = "language_model.model.embed_tokens.weight"
    let normName = "language_model.model.norm.weight"
    let lmHeadName = "language_model.lm_head.weight"
    try writeOutputIndex(
        weightMap: outputWeightMap([
            embedName: "model-00001-of-00001.safetensors",
            normName: "model-00001-of-00001.safetensors",
            lmHeadName: "model-00001-of-00001.safetensors",
            "language_model.lm_head.scales": "model-00001-of-00001.safetensors",
        ]),
        to: root
    )
    try writeOutputSafeTensorsEntries(
        [
            (embedName, "F32", [4, 2], outputFloatData(Array(repeating: 0, count: 8))),
            (normName, "F32", [2], outputFloatData([1, 1])),
            (lmHeadName, "U32", [4, 1], outputUInt32Data([
                outputPackQuantizedValues([1, 0], bits: 4),
                outputPackQuantizedValues([0, 1], bits: 4),
                outputPackQuantizedValues([1, 1], bits: 4),
                outputPackQuantizedValues([0, 2], bits: 4),
            ])),
            ("language_model.lm_head.scales", "F32", [4, 1], outputFloatData([1, 1, 1, 1])),
        ],
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )

    let store = QwenModelWeightStore(bundleIndex: try QwenModelBundleIndex.load(from: root))
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let hiddenStates = try EdgeTensor(float32: [3, 4], shape: EdgeTensorShape([1, 2]), runtime: runtime)

    let weights = try QwenQuantizedModelOutputWeights.loadHuggingFaceLayout(
        weightStore: store,
        runtime: runtime
    )
    let logits = try weights.logits(hiddenStates: hiddenStates, executor: executor)
    let normalized = try CPUReferenceOps.rmsNorm([3, 4], weight: [1, 1], epsilon: 0)
    let expected = try CPUReferenceOps.matmul(
        normalized,
        rows: 1,
        inner: 2,
        [
            1, 0, 1, 0,
            0, 1, 1, 2,
        ],
        columns: 4
    )
    let error = try NumericComparison.maxAbsoluteError(try logits.readFloat32(), expected)

    #expect(weights.lmHead.shape == [4, 2])
    #expect(logits.shape == EdgeTensorShape([1, 4]))
    #expect(error < 1e-5)
}

private func writeOutputConfig(
    quantization: QwenQuantizationProfile? = nil,
    to root: URL
) throws {
    let quantizationJSON = quantization.map { profile in
        """
        ,
        "quantization": {
          "group_size": \(profile.groupSize),
          "bits": \(profile.bits)
        }
        """
    } ?? ""
    let configJSON = """
    {
      "model_type": "qwen3_5",
      "text_config": {
        "model_type": "qwen3_5_text",
        "vocab_size": 4,
        "hidden_size": 2,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 32,
        "rms_norm_eps": 0,
        "rope_parameters": {
          "rope_theta": 10000
        },
        "layer_types": ["full_attention", "linear_attention"]\(quantizationJSON)
      }
    }
    """
    try configJSON.write(
        to: root.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
}

private func outputWeightMap(_ modelLevel: [String: String]) -> [String: String] {
    var weightMap = modelLevel
    weightMap["language_model.model.layers.0.input_layernorm.weight"] = "model-00001-of-00001.safetensors"
    return weightMap
}

private func writeOutputIndex(weightMap: [String: String], to root: URL) throws {
    let fields = weightMap.keys.sorted().map { key in
        "\"\(key)\":\"\(weightMap[key]!)\""
    }
    let indexJSON = """
    {
      "weight_map": {\(fields.joined(separator: ","))}
    }
    """
    try indexJSON.write(
        to: root.appendingPathComponent("model.safetensors.index.json"),
        atomically: true,
        encoding: .utf8
    )
}

private func writeOutputSafeTensorsEntries(
    _ entries: [(name: String, dtype: String, shape: [Int], data: Data)],
    to url: URL
) throws {
    var payload = Data()
    var fields: [String] = []
    var offset = 0
    for entry in entries {
        let end = offset + entry.data.count
        fields.append(
            """
            "\(entry.name)": {
              "dtype": "\(entry.dtype)",
              "shape": \(outputShapeJSON(entry.shape)),
              "data_offsets": [\(offset), \(end)]
            }
            """
        )
        payload.append(entry.data)
        offset = end
    }
    let headerJSON = "{\(fields.joined(separator: ","))}"
    let headerData = headerJSON.data(using: .utf8)!
    var headerLength = UInt64(headerData.count).littleEndian
    var fileData = withUnsafeBytes(of: &headerLength) { Data($0) }
    fileData.append(headerData)
    fileData.append(payload)
    try fileData.write(to: url)
}

private func outputShapeJSON(_ shape: [Int]) -> String {
    "[" + shape.map(String.init).joined(separator: ",") + "]"
}

private func outputFloatData(_ values: [Float]) -> Data {
    values.withUnsafeBufferPointer { buffer in
        Data(buffer: buffer)
    }
}

private func outputUInt32Data(_ values: [UInt32]) -> Data {
    var data = Data()
    for value in values {
        var littleEndianValue = value.littleEndian
        data.append(withUnsafeBytes(of: &littleEndianValue) { Data($0) })
    }
    return data
}

private func outputPackQuantizedValues(_ values: [UInt32], bits: Int) -> UInt32 {
    values.enumerated().reduce(UInt32.zero) { result, element in
        result | (element.element << UInt32(element.offset * bits))
    }
}
