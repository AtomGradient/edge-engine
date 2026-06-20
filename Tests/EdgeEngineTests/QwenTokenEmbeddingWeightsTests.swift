// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Testing
@testable import EdgeEngine

@Test func qwenTokenEmbeddingWeightsLookupUsesMetalKernel() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let weights = QwenTokenEmbeddingWeights(
        embeddings: try EdgeTensor(
            float32: [
                1, 2,
                3, 4,
                5, 6,
            ],
            shape: EdgeTensorShape([3, 2]),
            runtime: runtime
        )
    )

    let hiddenStates = try weights.hiddenStates(tokenIds: [2, 0, 2], executor: executor)

    #expect(hiddenStates.shape == EdgeTensorShape([3, 2]))
    #expect(try hiddenStates.readFloat32() == [
        5, 6,
        1, 2,
        5, 6,
    ])
    #expect(executor.lastExecutionStats?.operationName == "edge_embedding_lookup")
}

@Test func qwenTokenEmbeddingWeightsLoadHuggingFaceLayoutFromBundleShard() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-token-embedding-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let architecture = try makeTokenEmbeddingArchitecture()
    let embedName = "model.embed_tokens.weight"
    let index = try QwenModelBundleIndex(
        rootURL: root,
        architecture: architecture,
        weightMap: completeTokenEmbeddingWeightMap()
    )
    try writeTokenEmbeddingSafeTensorsShard(
        name: embedName,
        shape: [3, 2],
        values: [
            1, 2,
            3, 4,
            5, 6,
        ],
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )

    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let weights = try QwenTokenEmbeddingWeights.loadHuggingFaceLayout(
        weightStore: QwenModelWeightStore(bundleIndex: index),
        runtime: runtime
    )
    let hiddenStates = try weights.hiddenStates(tokenIds: [1], executor: executor)

    #expect(weights.embeddings.shape == EdgeTensorShape([3, 2]))
    #expect(try hiddenStates.readFloat32() == [3, 4])
}

@Test func qwenTokenEmbeddingWeightsLoadQuantizedHuggingFaceLayoutFromBundleShard() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-quantized-token-embedding-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let architecture = try makeTokenEmbeddingArchitecture(
        quantization: QwenQuantizationProfile(groupSize: 1, bits: 8)
    )
    let embedName = "model.embed_tokens.weight"
    var weightMap = completeTokenEmbeddingWeightMap()
    weightMap["model.embed_tokens.scales"] = "model-00001-of-00001.safetensors"
    let index = try QwenModelBundleIndex(
        rootURL: root,
        architecture: architecture,
        weightMap: weightMap
    )
    try writeTokenEmbeddingSafeTensorsEntries(
        [
            (
                embedName,
                "U32",
                [3, 1],
                tokenEmbeddingUInt32Data([
                    packTokenEmbeddingQuantizedValues([1, 2], bits: 8),
                    packTokenEmbeddingQuantizedValues([3, 4], bits: 8),
                    packTokenEmbeddingQuantizedValues([5, 6], bits: 8),
                ])
            ),
            (
                "model.embed_tokens.scales",
                "F32",
                [3, 2],
                tokenEmbeddingFloatData(Array(repeating: 1, count: 6))
            ),
        ],
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )

    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let weights = try QwenTokenEmbeddingWeights.loadHuggingFaceLayout(
        weightStore: QwenModelWeightStore(bundleIndex: index),
        runtime: runtime
    )
    let hiddenStates = try weights.hiddenStates(tokenIds: [2, 0], executor: executor)

    guard case .quantized = weights.embeddings else {
        Issue.record("Quantized token embeddings should keep packed storage.")
        return
    }
    #expect(weights.embeddings.shape == EdgeTensorShape([3, 2]))
    #expect(try hiddenStates.readFloat32() == [
        5, 6,
        1, 2,
    ])
    #expect(executor.lastExecutionStats?.operationName == "edge_affine_quantized_embedding_lookup")
}

private func makeTokenEmbeddingArchitecture() throws -> QwenHybridArchitecture {
    try makeTokenEmbeddingArchitecture(quantization: nil)
}

private func makeTokenEmbeddingArchitecture(
    quantization: QwenQuantizationProfile?
) throws -> QwenHybridArchitecture {
    try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 3,
        hiddenSize: 2,
        intermediateSize: 2,
        attentionHeadCount: 2,
        keyValueHeadCount: 1,
        contextLength: 8,
        rmsNormEpsilon: 0,
        ropeTheta: 10_000,
        quantization: quantization,
        layerKinds: [.gdn, .fullAttention]
    )
}

private func writeTokenEmbeddingSafeTensorsShard(
    name: String,
    shape: [Int],
    values: [Float],
    to url: URL
) throws {
    try writeTokenEmbeddingSafeTensorsEntries(
        [
            (name, "F32", shape, tokenEmbeddingFloatData(values)),
        ],
        to: url
    )
}

private func writeTokenEmbeddingSafeTensorsEntries(
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
              "shape": \(jsonIntArrayForTokenEmbedding(entry.shape)),
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

private func tokenEmbeddingFloatData(_ values: [Float]) -> Data {
    var data = Data()
    for value in values {
        var littleEndianValue = value.bitPattern.littleEndian
        data.append(withUnsafeBytes(of: &littleEndianValue) { Data($0) })
    }
    return data
}

private func tokenEmbeddingUInt32Data(_ values: [UInt32]) -> Data {
    var data = Data()
    for value in values {
        var littleEndianValue = value.littleEndian
        data.append(withUnsafeBytes(of: &littleEndianValue) { Data($0) })
    }
    return data
}

private func packTokenEmbeddingQuantizedValues(_ values: [UInt32], bits: Int) -> UInt32 {
    var word = UInt32.zero
    let mask = UInt32((1 << bits) - 1)
    for (index, value) in values.enumerated() {
        word |= (value & mask) << UInt32(index * bits)
    }
    return word
}

private func jsonIntArrayForTokenEmbedding(_ values: [Int]) -> String {
    "[\(values.map(String.init).joined(separator: ","))]"
}

private func completeTokenEmbeddingWeightMap() -> [String: String] {
    var weightMap: [String: String] = [:]
    for name in requiredTokenEmbeddingModelLevel() {
        weightMap[name] = "model-00001-of-00001.safetensors"
    }
    for name in requiredTokenEmbeddingGDNLayer0() {
        weightMap[name] = "model-00001-of-00001.safetensors"
    }
    for name in requiredTokenEmbeddingFullAttentionLayer1() {
        weightMap[name] = "model-00001-of-00001.safetensors"
    }
    return weightMap
}

private func requiredTokenEmbeddingModelLevel() -> [String] {
    [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.lm_head.weight",
    ]
}

private func requiredTokenEmbeddingGDNLayer0() -> [String] {
    let prefix = "model.layers.0"
    let attentionPrefix = "\(prefix).linear_attn"
    return [
        "\(prefix).input_layernorm.weight",
        "\(prefix).post_attention_layernorm.weight",
        "\(prefix).mlp.gate_proj.weight",
        "\(prefix).mlp.up_proj.weight",
        "\(prefix).mlp.down_proj.weight",
        "\(attentionPrefix).A_log",
        "\(attentionPrefix).conv1d.weight",
        "\(attentionPrefix).dt_bias",
        "\(attentionPrefix).in_proj_a.weight",
        "\(attentionPrefix).in_proj_b.weight",
        "\(attentionPrefix).in_proj_qkv.weight",
        "\(attentionPrefix).in_proj_z.weight",
        "\(attentionPrefix).norm.weight",
        "\(attentionPrefix).out_proj.weight",
    ]
}

private func requiredTokenEmbeddingFullAttentionLayer1() -> [String] {
    let prefix = "model.layers.1"
    let attentionPrefix = "\(prefix).self_attn"
    return [
        "\(prefix).input_layernorm.weight",
        "\(prefix).post_attention_layernorm.weight",
        "\(prefix).mlp.gate_proj.weight",
        "\(prefix).mlp.up_proj.weight",
        "\(prefix).mlp.down_proj.weight",
        "\(attentionPrefix).q_proj.weight",
        "\(attentionPrefix).k_proj.weight",
        "\(attentionPrefix).v_proj.weight",
        "\(attentionPrefix).o_proj.weight",
        "\(attentionPrefix).q_norm.weight",
        "\(attentionPrefix).k_norm.weight",
    ]
}
