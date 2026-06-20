// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Testing
@testable import EdgeEngine

@Test func qwenModelWeightStoreLoadsProjectionWeightsFromBundleShard() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-weight-store-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try writeConfig(layerTypes: ["linear_attention", "full_attention"], to: root)

    var weightMap = completeWeightMapForWeightStore(modelPrefix: "language_model.model")
    let qName = "language_model.model.layers.1.self_attn.q_proj.weight"
    let kName = "language_model.model.layers.1.self_attn.k_proj.weight"
    let vName = "language_model.model.layers.1.self_attn.v_proj.weight"
    let oName = "language_model.model.layers.1.self_attn.o_proj.weight"
    let mlpPrefix = "language_model.model.layers.1.mlp"
    let gateName = "\(mlpPrefix).gate_proj.weight"
    let upName = "\(mlpPrefix).up_proj.weight"
    let downName = "\(mlpPrefix).down_proj.weight"
    weightMap[qName] = "model-00002-of-00002.safetensors"
    weightMap[kName] = "model-00002-of-00002.safetensors"
    weightMap[vName] = "model-00002-of-00002.safetensors"
    weightMap[oName] = "model-00002-of-00002.safetensors"
    weightMap[gateName] = "model-00002-of-00002.safetensors"
    weightMap[upName] = "model-00002-of-00002.safetensors"
    weightMap[downName] = "model-00002-of-00002.safetensors"
    try writeIndex(weightMap: weightMap, to: root)
    try writeSafeTensorsShard(
        tensors: [
            qName: (shape: [4, 2], values: [
                1, 2,
                3, 4,
                5, 6,
                7, 8,
            ]),
            kName: (shape: [1, 2], values: [3, 4]),
            vName: (shape: [1, 2], values: [-1, 2]),
            oName: (shape: [2, 2], values: [
                1, 3,
                2, 4,
            ]),
            gateName: (shape: [8, 2], values: [
                1, 0,
                0, 1,
                0, 0,
                0, 0,
                0, 0,
                0, 0,
                0, 0,
                0, 0,
            ]),
            upName: (shape: [8, 2], values: [
                1, 0,
                2, 1,
                0, 0,
                0, 0,
                0, 0,
                0, 0,
                0, 0,
                0, 0,
            ]),
            downName: (shape: [2, 8], values: [
                1, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0, 0, 0,
            ]),
        ],
        to: root.appendingPathComponent("model-00002-of-00002.safetensors")
    )

    let bundle = try QwenModelBundleIndex.load(from: root)
    let store = QwenModelWeightStore(bundleIndex: bundle)
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let hiddenStates = try EdgeTensor(
        float32: [1, 2],
        shape: EdgeTensorShape([1, 2]),
        runtime: runtime
    )

    let weights = try QwenAttentionProjectionWeights.loadHuggingFaceLayout(
        layerIndex: 1,
        weightStore: store,
        runtime: runtime
    )
    let outputs = try weights.project(hiddenStates: hiddenStates, executor: executor)
    let outputWeights = try QwenAttentionOutputProjectionWeights.loadHuggingFaceLayout(
        layerIndex: 1,
        weightStore: store,
        runtime: runtime
    )
    let attentionOutput = try EdgeTensor(
        float32: [1, 2],
        shape: EdgeTensorShape([1, 2]),
        runtime: runtime
    )
    let gate = try EdgeTensor(float32: [0, 0], shape: EdgeTensorShape([1, 2]), runtime: runtime)
    let projectedOutput = try outputWeights.project(
        attentionOutput: attentionOutput,
        queryGate: gate,
        executor: executor
    )
    let mlpWeights = try QwenMLPWeights.loadHuggingFaceLayout(
        layerIndex: 1,
        weightStore: store,
        runtime: runtime
    )
    let mlpOutput = try mlpWeights(hiddenStates: hiddenStates, executor: executor)
    let expectedMLP = try CPUReferenceOps.swiglu(
        gate: [1, 2, 0, 0, 0, 0, 0, 0],
        up: [1, 4, 0, 0, 0, 0, 0, 0]
    )
    let mlpError = try NumericComparison.maxAbsoluteError(
        try mlpOutput.readFloat32(),
        [expectedMLP[0], expectedMLP[1]]
    )

    #expect(try outputs.query.readFloat32() == [5, 17])
    #expect(try outputs.queryGate?.readFloat32() == [11, 23])
    #expect(try outputs.key.readFloat32() == [11])
    #expect(try outputs.value.readFloat32() == [3])
    #expect(try outputWeights.weight.readFloat32() == [1, 2, 3, 4])
    #expect(try projectedOutput.readFloat32() == [3.5, 5])
    #expect(mlpWeights.gate.shape == EdgeTensorShape([2, 8]))
    #expect(mlpOutput.shape == EdgeTensorShape([1, 2]))
    #expect(mlpError < 1e-5)
}

@Test func qwenModelWeightStoreRejectsProjectionLoadForGDNLayer() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-weight-store-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try writeConfig(layerTypes: ["linear_attention", "full_attention"], to: root)
    try writeIndex(weightMap: completeWeightMapForWeightStore(modelPrefix: "language_model.model"), to: root)

    let bundle = try QwenModelBundleIndex.load(from: root)
    let store = QwenModelWeightStore(bundleIndex: bundle)
    let runtime = try EdgeMetalRuntime()

    do {
        _ = try QwenAttentionProjectionWeights.loadHuggingFaceLayout(
            layerIndex: 0,
            weightStore: store,
            runtime: runtime
        )
        Issue.record("Projection weight store loader must reject GDN layers.")
    } catch QwenProjectionWeightError.layerIsNotFullAttention(layerIndex: 0, kind: .gdn) {
        return
    }
    Issue.record("Projection weight store loader threw the wrong error for a GDN layer.")
}

@Test func qwenModelWeightStoreLoadsQuantizedWeightGroupFromBundleShard() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-weight-store-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try writeConfig(layerTypes: ["linear_attention", "full_attention"], to: root)

    let qName = "language_model.model.layers.1.self_attn.q_proj.weight"
    let scalesName = "language_model.model.layers.1.self_attn.q_proj.scales"
    let biasesName = "language_model.model.layers.1.self_attn.q_proj.biases"
    var weightMap = completeWeightMapForWeightStore(modelPrefix: "language_model.model")
    weightMap[qName] = "model-00002-of-00002.safetensors"
    weightMap[scalesName] = "model-00002-of-00002.safetensors"
    weightMap[biasesName] = "model-00002-of-00002.safetensors"
    try writeIndex(weightMap: weightMap, to: root)
    try writeQuantizedSafeTensorsShard(
        weightName: qName,
        scalesName: scalesName,
        biasesName: biasesName,
        to: root.appendingPathComponent("model-00002-of-00002.safetensors")
    )

    let bundle = try QwenModelBundleIndex.load(from: root)
    let store = QwenModelWeightStore(bundleIndex: bundle)

    let quantized = try store.loadQuantizedTensor(
        weightName: qName,
        scalesName: scalesName,
        biasesName: biasesName,
        groupSize: 2,
        bits: 4
    )

    #expect(quantized.shape == [2, 2])
    #expect(quantized.dequantizedValues() == [1, 2, 16, 18])
}

@Test func qwenQuantizedAttentionProjectionWeightsLoadAndProjectFromBundleShard() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-weight-store-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try writeConfig(
        layerTypes: ["linear_attention", "full_attention"],
        quantization: QwenQuantizationProfile(groupSize: 2, bits: 4),
        to: root
    )

    let prefix = "language_model.model.layers.1.self_attn"
    let qName = "\(prefix).q_proj.weight"
    let kName = "\(prefix).k_proj.weight"
    let vName = "\(prefix).v_proj.weight"
    var weightMap = completeWeightMapForWeightStore(modelPrefix: "language_model.model")
    for name in [qName, kName, vName] {
        let base = String(name.dropLast(".weight".count))
        weightMap[name] = "model-00002-of-00002.safetensors"
        weightMap["\(base).scales"] = "model-00002-of-00002.safetensors"
        weightMap["\(base).biases"] = "model-00002-of-00002.safetensors"
    }
    try writeIndex(weightMap: weightMap, to: root)
    try writeQuantizedProjectionShard(
        qName: qName,
        kName: kName,
        vName: vName,
        to: root.appendingPathComponent("model-00002-of-00002.safetensors")
    )

    let bundle = try QwenModelBundleIndex.load(from: root)
    let store = QwenModelWeightStore(bundleIndex: bundle)
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let hiddenStates = try EdgeTensor(
        float32: [1, 2],
        shape: EdgeTensorShape([1, 2]),
        runtime: runtime
    )

    let weights = try QwenQuantizedAttentionProjectionWeights.loadHuggingFaceLayout(
        layerIndex: 1,
        weightStore: store
    )
    let outputs = try weights.project(hiddenStates: hiddenStates, executor: executor)

    #expect(weights.query.shape == [4, 2])
    #expect(weights.key.shape == [1, 2])
    #expect(weights.value.shape == [1, 2])
    #expect(try outputs.query.readFloat32() == [1, 3])
    #expect(try outputs.queryGate?.readFloat32() == [2, 4])
    #expect(try outputs.key.readFloat32() == [11])
    #expect(try outputs.value.readFloat32() == [5.5])
}

@Test func qwenQuantizedAttentionOutputProjectionWeightsLoadAndProjectFromBundleShard() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-weight-store-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try writeConfig(
        layerTypes: ["linear_attention", "full_attention"],
        quantization: QwenQuantizationProfile(groupSize: 2, bits: 4),
        to: root
    )

    let prefix = "language_model.model.layers.1.self_attn"
    let oName = "\(prefix).o_proj.weight"
    let oBase = String(oName.dropLast(".weight".count))
    var weightMap = completeWeightMapForWeightStore(modelPrefix: "language_model.model")
    weightMap[oName] = "model-00002-of-00002.safetensors"
    weightMap["\(oBase).scales"] = "model-00002-of-00002.safetensors"
    weightMap["\(oBase).biases"] = "model-00002-of-00002.safetensors"
    try writeIndex(weightMap: weightMap, to: root)
    try writeSafeTensorsEntries(
        [
            (
                oName,
                "U32",
                [2, 1],
                uint32DataForWeightStore([
                    packQuantizedValuesForWeightStore([1, 2], bits: 4),
                    packQuantizedValuesForWeightStore([3, 4], bits: 4),
                ])
            ),
            ("\(oBase).scales", "F32", [2, 1], floatDataForWeightStore([1, 1])),
            ("\(oBase).biases", "F32", [2, 1], floatDataForWeightStore([0, 0])),
        ],
        to: root.appendingPathComponent("model-00002-of-00002.safetensors")
    )

    let bundle = try QwenModelBundleIndex.load(from: root)
    let store = QwenModelWeightStore(bundleIndex: bundle)
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let attentionOutput = try EdgeTensor(
        float32: [1, 2],
        shape: EdgeTensorShape([1, 2]),
        runtime: runtime
    )
    let gate = try EdgeTensor(float32: [0, 0], shape: EdgeTensorShape([1, 2]), runtime: runtime)

    let weights = try QwenQuantizedAttentionOutputProjectionWeights.loadHuggingFaceLayout(
        layerIndex: 1,
        weightStore: store
    )
    let output = try weights.project(attentionOutput: attentionOutput, queryGate: gate, executor: executor)

    #expect(weights.weight.shape == [2, 2])
    #expect(weights.weight.dequantizedValues() == [1, 2, 3, 4])
    #expect(try output.readFloat32() == [2.5, 5.5])
}

@Test func qwenQuantizedMLPWeightsLoadAndProjectFromBundleShard() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-weight-store-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try writeConfig(
        layerTypes: ["linear_attention", "full_attention"],
        quantization: QwenQuantizationProfile(groupSize: 2, bits: 4),
        to: root
    )

    let prefix = "language_model.model.layers.1.mlp"
    let gateName = "\(prefix).gate_proj.weight"
    let upName = "\(prefix).up_proj.weight"
    let downName = "\(prefix).down_proj.weight"
    var weightMap = completeWeightMapForWeightStore(modelPrefix: "language_model.model")
    for name in [gateName, upName, downName] {
        let base = String(name.dropLast(".weight".count))
        weightMap[name] = "model-00002-of-00002.safetensors"
        weightMap["\(base).scales"] = "model-00002-of-00002.safetensors"
    }
    try writeIndex(weightMap: weightMap, to: root)
    try writeQuantizedMLPShard(
        gateName: gateName,
        upName: upName,
        downName: downName,
        to: root.appendingPathComponent("model-00002-of-00002.safetensors")
    )

    let bundle = try QwenModelBundleIndex.load(from: root)
    let store = QwenModelWeightStore(bundleIndex: bundle)
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let hiddenStates = try EdgeTensor(
        float32: [1, 2],
        shape: EdgeTensorShape([1, 2]),
        runtime: runtime
    )

    let weights = try QwenQuantizedMLPWeights.loadHuggingFaceLayout(
        layerIndex: 1,
        weightStore: store
    )
    let output = try weights(hiddenStates: hiddenStates, executor: executor)
    let expected = try CPUReferenceOps.swiglu(
        gate: [1, 2, 0, 0, 0, 0, 0, 0],
        up: [1, 4, 0, 0, 0, 0, 0, 0]
    )
    let error = try NumericComparison.maxAbsoluteError(try output.readFloat32(), [expected[0], expected[1]])

    #expect(weights.gate.shape == [8, 2])
    #expect(weights.up.shape == [8, 2])
    #expect(weights.down.shape == [2, 8])
    #expect(output.shape == EdgeTensorShape([1, 2]))
    #expect(error < 1e-5)
}

@Test func qwenQuantizedAttentionProjectionWeightsRequireQuantizationProfile() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-weight-store-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try writeConfig(layerTypes: ["linear_attention", "full_attention"], to: root)
    try writeIndex(weightMap: completeWeightMapForWeightStore(modelPrefix: "language_model.model"), to: root)

    let bundle = try QwenModelBundleIndex.load(from: root)
    let store = QwenModelWeightStore(bundleIndex: bundle)

    do {
        _ = try QwenQuantizedAttentionProjectionWeights.loadHuggingFaceLayout(
            layerIndex: 1,
            weightStore: store
        )
        Issue.record("Quantized projection loading should require config quantization metadata.")
    } catch QwenProjectionWeightError.missingQuantizationProfile {
        return
    }
    Issue.record("Quantized projection loading threw the wrong error for missing quantization metadata.")
}

private func writeConfig(
    layerTypes: [String],
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
        "vocab_size": 128,
        "hidden_size": 2,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
          "rope_theta": 10000
        },
        "layer_types": \(jsonArrayForWeightStore(layerTypes))\(quantizationJSON)
      }
    }
    """
    try configJSON.write(
        to: root.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
}

private func writeIndex(weightMap: [String: String], to root: URL) throws {
    let indexJSON = """
    {
      "weight_map": \(jsonObjectForWeightStore(weightMap))
    }
    """
    try indexJSON.write(
        to: root.appendingPathComponent("model.safetensors.index.json"),
        atomically: true,
        encoding: .utf8
    )
}

private func completeWeightMapForWeightStore(modelPrefix: String) -> [String: String] {
    var weightMap: [String: String] = [:]
    for name in requiredModelLevelForWeightStore(modelPrefix: modelPrefix) {
        weightMap[name] = "model-00001-of-00002.safetensors"
    }
    for name in requiredLayer0GDNForWeightStore(modelPrefix: modelPrefix) {
        weightMap[name] = "model-00001-of-00002.safetensors"
    }
    for name in requiredLayer1FAForWeightStore(modelPrefix: modelPrefix) {
        weightMap[name] = "model-00002-of-00002.safetensors"
    }
    return weightMap
}

private func requiredModelLevelForWeightStore(modelPrefix: String) -> [String] {
    [
        "\(modelPrefix).embed_tokens.weight",
        "\(modelPrefix).norm.weight",
        "language_model.lm_head.weight",
    ]
}

private func requiredLayer0GDNForWeightStore(modelPrefix: String) -> [String] {
    let prefix = "\(modelPrefix).layers.0"
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

private func requiredLayer1FAForWeightStore(modelPrefix: String) -> [String] {
    let prefix = "\(modelPrefix).layers.1"
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

private func writeSafeTensorsShard(
    tensors: [String: (shape: [Int], values: [Float])],
    to url: URL
) throws {
    var payload = Data()
    var fields: [String] = []
    var offset = 0
    for name in tensors.keys.sorted() {
        let tensor = tensors[name]!
        let data = floatDataForWeightStore(tensor.values)
        let end = offset + data.count
        fields.append(
            """
            "\(name)": {
              "dtype": "F32",
              "shape": \(jsonIntArrayForWeightStore(tensor.shape)),
              "data_offsets": [\(offset), \(end)]
            }
            """
        )
        payload.append(data)
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

private func writeQuantizedSafeTensorsShard(
    weightName: String,
    scalesName: String,
    biasesName: String,
    to url: URL
) throws {
    var payload = Data()
    payload.append(uint32DataForWeightStore([
        packQuantizedValuesForWeightStore([1, 2], bits: 4),
        packQuantizedValuesForWeightStore([3, 4], bits: 4),
    ]))
    payload.append(floatDataForWeightStore([1, 2]))
    payload.append(floatDataForWeightStore([0, 10]))

    let headerJSON = """
    {
      "\(weightName)": {
        "dtype": "U32",
        "shape": [2, 1],
        "data_offsets": [0, 8]
      },
      "\(scalesName)": {
        "dtype": "F32",
        "shape": [2, 1],
        "data_offsets": [8, 16]
      },
      "\(biasesName)": {
        "dtype": "F32",
        "shape": [2, 1],
        "data_offsets": [16, 24]
      }
    }
    """
    let headerData = headerJSON.data(using: .utf8)!
    var headerLength = UInt64(headerData.count).littleEndian
    var fileData = withUnsafeBytes(of: &headerLength) { Data($0) }
    fileData.append(headerData)
    fileData.append(payload)
    try fileData.write(to: url)
}

private func writeQuantizedProjectionShard(
    qName: String,
    kName: String,
    vName: String,
    to url: URL
) throws {
    let qBase = String(qName.dropLast(".weight".count))
    let kBase = String(kName.dropLast(".weight".count))
    let vBase = String(vName.dropLast(".weight".count))
    let entries: [(name: String, dtype: String, shape: [Int], data: Data)] = [
        (
            qName,
            "U32",
            [4, 1],
            uint32DataForWeightStore([
                packQuantizedValuesForWeightStore([1, 0], bits: 4),
                packQuantizedValuesForWeightStore([0, 1], bits: 4),
                packQuantizedValuesForWeightStore([3, 0], bits: 4),
                packQuantizedValuesForWeightStore([0, 2], bits: 4),
            ])
        ),
        ("\(qBase).scales", "F32", [4, 1], floatDataForWeightStore([1, 1, 1, 1])),
        ("\(qBase).biases", "F32", [4, 1], floatDataForWeightStore([0, 0, 0, 0])),
        (kName, "U32", [1, 1], uint32DataForWeightStore([packQuantizedValuesForWeightStore([3, 4], bits: 4)])),
        ("\(kBase).scales", "F32", [1, 1], floatDataForWeightStore([1])),
        ("\(kBase).biases", "F32", [1, 1], floatDataForWeightStore([0])),
        (vName, "U32", [1, 1], uint32DataForWeightStore([packQuantizedValuesForWeightStore([3, 4], bits: 4)])),
        ("\(vBase).scales", "F32", [1, 1], floatDataForWeightStore([0.5])),
        ("\(vBase).biases", "F32", [1, 1], floatDataForWeightStore([0])),
    ]
    try writeSafeTensorsEntries(entries, to: url)
}

private func writeQuantizedMLPShard(
    gateName: String,
    upName: String,
    downName: String,
    to url: URL
) throws {
    let gateBase = String(gateName.dropLast(".weight".count))
    let upBase = String(upName.dropLast(".weight".count))
    let downBase = String(downName.dropLast(".weight".count))
    let entries: [(name: String, dtype: String, shape: [Int], data: Data)] = [
        (
            gateName,
            "U32",
            [8, 1],
            uint32DataForWeightStore([
                packQuantizedValuesForWeightStore([1, 0], bits: 4),
                packQuantizedValuesForWeightStore([0, 1], bits: 4),
                packQuantizedValuesForWeightStore([0, 0], bits: 4),
                packQuantizedValuesForWeightStore([0, 0], bits: 4),
                packQuantizedValuesForWeightStore([0, 0], bits: 4),
                packQuantizedValuesForWeightStore([0, 0], bits: 4),
                packQuantizedValuesForWeightStore([0, 0], bits: 4),
                packQuantizedValuesForWeightStore([0, 0], bits: 4),
            ])
        ),
        ("\(gateBase).scales", "F32", [8, 1], floatDataForWeightStore(Array(repeating: 1, count: 8))),
        (
            upName,
            "U32",
            [8, 1],
            uint32DataForWeightStore([
                packQuantizedValuesForWeightStore([1, 0], bits: 4),
                packQuantizedValuesForWeightStore([2, 1], bits: 4),
                packQuantizedValuesForWeightStore([0, 0], bits: 4),
                packQuantizedValuesForWeightStore([0, 0], bits: 4),
                packQuantizedValuesForWeightStore([0, 0], bits: 4),
                packQuantizedValuesForWeightStore([0, 0], bits: 4),
                packQuantizedValuesForWeightStore([0, 0], bits: 4),
                packQuantizedValuesForWeightStore([0, 0], bits: 4),
            ])
        ),
        ("\(upBase).scales", "F32", [8, 1], floatDataForWeightStore(Array(repeating: 1, count: 8))),
        (
            downName,
            "U32",
            [2, 1],
            uint32DataForWeightStore([
                packQuantizedValuesForWeightStore([1, 0, 0, 0, 0, 0, 0, 0], bits: 4),
                packQuantizedValuesForWeightStore([0, 1, 0, 0, 0, 0, 0, 0], bits: 4),
            ])
        ),
        ("\(downBase).scales", "F32", [2, 4], floatDataForWeightStore(Array(repeating: 1, count: 8))),
    ]
    try writeSafeTensorsEntries(entries, to: url)
}

private func writeSafeTensorsEntries(
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
              "shape": \(jsonIntArrayForWeightStore(entry.shape)),
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

private func jsonArrayForWeightStore(_ values: [String]) -> String {
    "[" + values.map { "\"\($0)\"" }.joined(separator: ",") + "]"
}

private func jsonIntArrayForWeightStore(_ values: [Int]) -> String {
    "[" + values.map(String.init).joined(separator: ",") + "]"
}

private func jsonObjectForWeightStore(_ values: [String: String]) -> String {
    let fields = values.keys.sorted().map { key in
        "\"\(key)\":\"\(values[key]!)\""
    }
    return "{\(fields.joined(separator: ","))}"
}

private func floatDataForWeightStore(_ values: [Float]) -> Data {
    values.withUnsafeBufferPointer { buffer in
        Data(buffer: buffer)
    }
}

private func uint32DataForWeightStore(_ values: [UInt32]) -> Data {
    var data = Data()
    for value in values {
        var littleEndianValue = value.littleEndian
        data.append(withUnsafeBytes(of: &littleEndianValue) { Data($0) })
    }
    return data
}

private func packQuantizedValuesForWeightStore(_ values: [UInt32], bits: Int) -> UInt32 {
    values.enumerated().reduce(UInt32.zero) { result, element in
        result | (element.element << UInt32(element.offset * bits))
    }
}
