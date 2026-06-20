// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Testing
@testable import EdgeEngine

@Test func qwenModelBundleIndexLoadsPublicConfigAndWeightIndex() throws {
    let root = try makeTemporaryQwenBundle(
        layerTypes: ["linear_attention", "full_attention"],
        weightMap: completeWeightMap(modelPrefix: "language_model.model")
    )

    let index = try QwenModelBundleIndex.load(from: root)

    #expect(index.architecture.family == .qwen35)
    #expect(index.architecture.layerCount == 2)
    #expect(index.modelPrefix == "language_model.model")
    #expect(index.executionPlan.gdnSteps.map(\.layerIndex) == [0])
    #expect(index.executionPlan.fullAttentionSteps.map(\.layerIndex) == [1])
    #expect(index.missingRequiredTensorNames == [])

    #expect(index.modelLevelManifest.embedTokensName == "language_model.model.embed_tokens.weight")
    #expect(index.modelLevelManifest.finalNormName == "language_model.model.norm.weight")
    #expect(index.modelLevelManifest.lmHeadName == "language_model.lm_head.weight")
    #expect(!index.modelLevelManifest.isWeightTied)

    let gdn = try index.layerManifest(0)
    #expect(gdn.kind == .gdn)
    #expect(gdn.attentionNamespace == .linearAttention)
    #expect(gdn.layerPrefix == "language_model.model.layers.0")
    #expect(gdn.requiredTensorNames.contains("language_model.model.layers.0.linear_attn.in_proj_qkv.weight"))
    #expect(gdn.requiredTensorNames.contains("language_model.model.layers.0.mlp.gate_proj.weight"))
    #expect(gdn.requiredTensorNames.contains("language_model.model.layers.0.mlp.up_proj.weight"))
    #expect(gdn.requiredTensorNames.contains("language_model.model.layers.0.mlp.down_proj.weight"))
    #expect(gdn.optionalTensorNames.contains("language_model.model.layers.0.linear_attn.in_proj_qkv.scales"))
    #expect(gdn.optionalTensorNames.contains("language_model.model.layers.0.linear_attn.in_proj_qkv.biases"))

    let qkvGroup = try #require(
        gdn.quantizedWeightGroups.first {
            $0.weightName == "language_model.model.layers.0.linear_attn.in_proj_qkv.weight"
        }
    )
    #expect(qkvGroup.isQuantized)
    #expect(qkvGroup.scalesName == "language_model.model.layers.0.linear_attn.in_proj_qkv.scales")
    #expect(qkvGroup.biasesName == "language_model.model.layers.0.linear_attn.in_proj_qkv.biases")

    let fa = try index.layerManifest(1)
    #expect(fa.kind == .fullAttention)
    #expect(fa.attentionNamespace == .selfAttention)
    #expect(fa.requiredTensorNames.contains("language_model.model.layers.1.self_attn.q_proj.weight"))
    #expect(fa.requiredTensorNames.contains("language_model.model.layers.1.mlp.gate_proj.weight"))

    #expect(try index.shardFileName(containing: "language_model.model.layers.1.self_attn.q_proj.weight") == "model-00002-of-00002.safetensors")
    #expect(
        try index.shardFileURL(containing: "language_model.model.layers.1.self_attn.q_proj.weight")
            == root.appendingPathComponent("model-00002-of-00002.safetensors")
    )
}

@Test func qwenModelBundleIndexSupportsBareModelPrefix() throws {
    let root = try makeTemporaryQwenBundle(
        layerTypes: ["linear_attention", "full_attention"],
        weightMap: completeWeightMap(modelPrefix: "model")
    )

    let index = try QwenModelBundleIndex.load(from: root, family: .qwen36)

    #expect(index.architecture.family == .qwen36)
    #expect(index.modelPrefix == "model")
    #expect(index.missingRequiredTensorNames == [])
    #expect(index.modelLevelManifest.embedTokensName == "model.embed_tokens.weight")
    #expect(index.modelLevelManifest.finalNormName == "model.norm.weight")
    #expect(index.modelLevelManifest.lmHeadName == "lm_head.weight")
    #expect(try index.layerManifest(0).attentionNamespace == .linearAttention)
    #expect(try index.layerManifest(1).attentionNamespace == .selfAttention)
}

@Test func qwenModelBundleIndexTreatsAbsentLMHeadAsWeightTied() throws {
    let root = try makeTemporaryQwenBundle(
        layerTypes: ["linear_attention", "full_attention"],
        weightMap: completeWeightMap(modelPrefix: "language_model.model", includeLMHead: false)
    )

    let index = try QwenModelBundleIndex.load(from: root)

    #expect(index.modelLevelManifest.lmHeadName == nil)
    #expect(index.modelLevelManifest.isWeightTied)
    #expect(index.modelLevelManifest.requiredTensorNames == [
        "language_model.model.embed_tokens.weight",
        "language_model.model.norm.weight",
    ])
    #expect(index.missingRequiredTensorNames == [])
}

@Test func qwenModelBundleIndexCoversQuantizedAttentionAndMLPWeights() throws {
    let root = try makeTemporaryQwenBundle(
        layerTypes: ["linear_attention", "full_attention"],
        weightMap: completeWeightMap(
            modelPrefix: "language_model.model",
            quantizeAllLinearWeights: true
        )
    )

    let index = try QwenModelBundleIndex.load(from: root)
    let linearGroups = index.layerManifests.flatMap(\.quantizedWeightGroups).filter { group in
        group.weightName.contains(".linear_attn.")
            || group.weightName.contains(".self_attn.")
            || group.weightName.contains(".mlp.")
    }

    #expect(!linearGroups.isEmpty)
    #expect(linearGroups.allSatisfy { $0.isQuantized })
    #expect(linearGroups.allSatisfy { $0.scalesName != nil && $0.biasesName != nil })
}

@Test func qwenModelBundleIndexUsesSparseMoEManifestWhenConfigIsMoE() throws {
    let root = try makeTemporaryQwenBundle(
        layerTypes: ["linear_attention", "full_attention"],
        weightMap: completeMoEWeightMap(modelPrefix: "language_model.model"),
        sparseMoE: true
    )

    let index = try QwenModelBundleIndex.load(from: root)

    #expect(index.architecture.usesSparseMoEMLP)
    #expect(index.missingRequiredTensorNames == [])
    let gdn = try index.layerManifest(0)
    #expect(gdn.requiredTensorNames.contains("language_model.model.layers.0.mlp.gate.weight"))
    #expect(gdn.requiredTensorNames.contains("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"))
    #expect(gdn.requiredTensorNames.contains("language_model.model.layers.0.mlp.shared_expert_gate.weight"))
    #expect(!gdn.requiredTensorNames.contains("language_model.model.layers.0.mlp.gate_proj.weight"))

    let routerGroup = try #require(
        gdn.quantizedWeightGroups.first {
            $0.weightName == "language_model.model.layers.0.mlp.gate.weight"
        }
    )
    #expect(routerGroup.scalesName == "language_model.model.layers.0.mlp.gate.scales")
    #expect(routerGroup.biasesName == "language_model.model.layers.0.mlp.gate.biases")
}

@Test func qwenModelBundleIndexReportsMissingRequiredTensor() throws {
    var weightMap = completeWeightMap(modelPrefix: "language_model.model")
    weightMap.removeValue(forKey: "language_model.model.layers.1.self_attn.k_norm.weight")
    let root = try makeTemporaryQwenBundle(
        layerTypes: ["linear_attention", "full_attention"],
        weightMap: weightMap
    )

    let index = try QwenModelBundleIndex.load(from: root)

    #expect(index.missingRequiredTensorNames == ["language_model.model.layers.1.self_attn.k_norm.weight"])
    do {
        try index.validateRequiredTensorCoverage()
        Issue.record("Bundle index must fail fast when a required runtime tensor is missing.")
    } catch QwenModelBundleIndexError.missingTensor("language_model.model.layers.1.self_attn.k_norm.weight") {
        return
    }
    Issue.record("Bundle index threw the wrong error for missing tensor coverage.")
}

@Test func qwenModelBundleIndexRejectsWeightIndexWithoutModelLayers() throws {
    let root = try makeTemporaryQwenBundle(
        layerTypes: ["linear_attention", "full_attention"],
        weightMap: ["visual.encoder.weight": "model.safetensors"]
    )

    do {
        _ = try QwenModelBundleIndex.load(from: root)
        Issue.record("Bundle index must reject unknown Qwen model prefixes.")
    } catch QwenModelBundleIndexError.unsupportedModelPrefix {
        return
    }
    Issue.record("Bundle index threw the wrong error for unsupported prefixes.")
}

private func makeTemporaryQwenBundle(
    layerTypes: [String],
    weightMap: [String: String],
    sparseMoE: Bool = false
) throws -> URL {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-bundle-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

    let configJSON = """
    {
      "model_type": "qwen3_5",
      "text_config": {
        "model_type": "qwen3_5_text",
        "vocab_size": 248320,
        "hidden_size": 2560,
        \(sparseMoE ? #"""
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        """# : #"""
        "intermediate_size": 9216,
        """#)
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
          "rope_theta": 10000000
        },
        "layer_types": \(jsonArray(layerTypes))
      }
    }
    """
    try configJSON.write(
        to: root.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let indexJSON = """
    {
      "metadata": {
        "total_size": 0
      },
      "weight_map": \(jsonObject(weightMap))
    }
    """
    try indexJSON.write(
        to: root.appendingPathComponent("model.safetensors.index.json"),
        atomically: true,
        encoding: .utf8
    )
    return root
}

private func completeWeightMap(
    modelPrefix: String,
    includeLMHead: Bool = true,
    quantizeAllLinearWeights: Bool = false
) -> [String: String] {
    var weightMap: [String: String] = [:]
    for name in requiredModelLevel(modelPrefix: modelPrefix, includeLMHead: includeLMHead) {
        weightMap[name] = "model-00001-of-00002.safetensors"
    }
    for name in requiredLayer0GDN(modelPrefix: modelPrefix) {
        weightMap[name] = "model-00001-of-00002.safetensors"
    }
    for name in requiredLayer1FA(modelPrefix: modelPrefix) {
        weightMap[name] = "model-00002-of-00002.safetensors"
    }

    let quantizedBase = "\(modelPrefix).layers.0.linear_attn.in_proj_qkv"
    weightMap["\(quantizedBase).scales"] = "model-00001-of-00002.safetensors"
    weightMap["\(quantizedBase).biases"] = "model-00001-of-00002.safetensors"
    if quantizeAllLinearWeights {
        for name in requiredLayer0GDN(modelPrefix: modelPrefix) + requiredLayer1FA(modelPrefix: modelPrefix) {
            guard isLinearWeightName(name) else { continue }
            let base = String(name.dropLast(".weight".count))
            weightMap["\(base).scales"] = weightMap[name]
            weightMap["\(base).biases"] = weightMap[name]
        }
    }
    return weightMap
}

private func completeMoEWeightMap(modelPrefix: String) -> [String: String] {
    var weightMap: [String: String] = [:]
    for name in requiredModelLevel(modelPrefix: modelPrefix, includeLMHead: true) {
        weightMap[name] = "model-00001-of-00002.safetensors"
    }
    for name in requiredLayer0GDN(modelPrefix: modelPrefix, sparseMoE: true) {
        weightMap[name] = "model-00001-of-00002.safetensors"
    }
    for name in requiredLayer1FA(modelPrefix: modelPrefix, sparseMoE: true) {
        weightMap[name] = "model-00002-of-00002.safetensors"
    }
    for name in requiredLayer0GDN(modelPrefix: modelPrefix, sparseMoE: true)
        + requiredLayer1FA(modelPrefix: modelPrefix, sparseMoE: true)
    {
        guard isLinearWeightName(name) else { continue }
        let base = String(name.dropLast(".weight".count))
        weightMap["\(base).scales"] = weightMap[name]
        weightMap["\(base).biases"] = weightMap[name]
    }
    return weightMap
}

private func requiredModelLevel(modelPrefix: String, includeLMHead: Bool) -> [String] {
    var names = [
        "\(modelPrefix).embed_tokens.weight",
        "\(modelPrefix).norm.weight",
    ]
    if includeLMHead {
        names.append(lmHeadName(modelPrefix: modelPrefix))
    }
    return names
}

private func lmHeadName(modelPrefix: String) -> String {
    if modelPrefix == "language_model.model" {
        return "language_model.lm_head.weight"
    }
    return "lm_head.weight"
}

private func requiredLayer0GDN(modelPrefix: String, sparseMoE: Bool = false) -> [String] {
    let prefix = "\(modelPrefix).layers.0"
    let attentionPrefix = "\(prefix).linear_attn"
    return commonLayerNames(prefix: prefix, sparseMoE: sparseMoE) + [
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

private func requiredLayer1FA(modelPrefix: String, sparseMoE: Bool = false) -> [String] {
    let prefix = "\(modelPrefix).layers.1"
    let attentionPrefix = "\(prefix).self_attn"
    return commonLayerNames(prefix: prefix, sparseMoE: sparseMoE) + [
        "\(attentionPrefix).q_proj.weight",
        "\(attentionPrefix).k_proj.weight",
        "\(attentionPrefix).v_proj.weight",
        "\(attentionPrefix).o_proj.weight",
        "\(attentionPrefix).q_norm.weight",
        "\(attentionPrefix).k_norm.weight",
    ]
}

private func commonLayerNames(prefix: String, sparseMoE: Bool) -> [String] {
    var names = [
        "\(prefix).input_layernorm.weight",
        "\(prefix).post_attention_layernorm.weight",
    ]
    if sparseMoE {
        names += [
            "\(prefix).mlp.gate.weight",
            "\(prefix).mlp.switch_mlp.gate_proj.weight",
            "\(prefix).mlp.switch_mlp.up_proj.weight",
            "\(prefix).mlp.switch_mlp.down_proj.weight",
            "\(prefix).mlp.shared_expert.gate_proj.weight",
            "\(prefix).mlp.shared_expert.up_proj.weight",
            "\(prefix).mlp.shared_expert.down_proj.weight",
            "\(prefix).mlp.shared_expert_gate.weight",
        ]
    } else {
        names += [
            "\(prefix).mlp.gate_proj.weight",
            "\(prefix).mlp.up_proj.weight",
            "\(prefix).mlp.down_proj.weight",
        ]
    }
    return names
}

private func isLinearWeightName(_ name: String) -> Bool {
    name.hasSuffix(".weight")
        && (
            name.contains(".linear_attn.")
                || name.contains(".self_attn.")
                || name.contains(".mlp.")
        )
}

private func jsonArray(_ values: [String]) -> String {
    "[" + values.map { "\"\($0)\"" }.joined(separator: ",") + "]"
}

private func jsonObject(_ values: [String: String]) -> String {
    let fields = values.keys.sorted().map { key in
        "\"\(key)\":\"\(values[key]!)\""
    }
    return "{\(fields.joined(separator: ","))}"
}
