// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Testing
@testable import EdgeEngine

@Test func qwenVLMBundlePreflightLoadsQwen35Metadata() throws {
    let root = try makeTemporaryVLMBundle()
    try writeVLMConfig(to: root)
    try writeVLMPreprocessorConfig(to: root)
    try writeVLMTokenizerJSON(to: root)
    try writeVLMIndex(weightMap: makeVLMWeightMap(), to: root)

    let result = try QwenVLMBundlePreflightRunner.run(
        configuration: QwenVLMBundlePreflightConfiguration(modelRootURL: root)
    )

    #expect(result.modelType == "qwen3_5_vl")
    #expect(result.plan.modelFamily == .qwen35VLM)
    #expect(result.plan.languageArchitecture.family == .qwen35)
    #expect(result.plan.languageArchitecture.layerCount == 2)
    #expect(result.plan.languageArchitecture.usesHybridAttentionAndGDN)
    #expect(result.plan.visionConfiguration.hiddenSize == 4)
    #expect(result.plan.visionConfiguration.layerCount == 2)
    #expect(result.plan.imageProcessorConfiguration.imageProcessorType == "Qwen3VLImageProcessor")
    #expect(result.plan.imageProcessorConfiguration.minPixels == 3_136)
    #expect(result.visionTensorPrefixes == ["visual"])
    #expect(result.passesPreflight)
    #expect(result.failureReasons.isEmpty)
    #expect(result.missingRequiredResourceNames.isEmpty)
    #expect(result.missingLanguageTensorNames.isEmpty)
    #expect(result.tokenizerSpecialTokenChecks.map(\.matches) == [true, true, true])

    let encoded = try JSONEncoder().encode(result)
    let decoded = try JSONDecoder().decode(QwenVLMBundlePreflightResult.self, from: encoded)
    #expect(decoded == result)
}

@Test func qwenVLMBundlePreflightAcceptsExportedVisionDepthFields() throws {
    let root = try makeTemporaryVLMBundle()
    try writeVLMConfig(
        to: root,
        visionLayerCountKey: "depth",
        visionHeadCountKey: "num_heads",
        modelType: "qwen3_5"
    )
    try writeVLMPreprocessorConfig(to: root)
    try writeVLMTokenizerJSON(to: root)
    try writeVLMIndex(weightMap: makeVLMWeightMap(visionPrefix: "vision_tower"), to: root)

    let result = try QwenVLMBundlePreflightRunner.run(
        configuration: QwenVLMBundlePreflightConfiguration(modelRootURL: root)
    )

    #expect(result.modelType == "qwen3_5")
    #expect(result.plan.modelFamily == .qwen35VLM)
    #expect(result.plan.visionConfiguration.layerCount == 2)
    #expect(result.plan.visionConfiguration.attentionHeadCount == 2)
    #expect(result.visionTensorPrefixes == ["vision_tower"])
    #expect(result.passesPreflight)
}

@Test func qwenVLMBundlePreflightReportsVisionAndTokenizerFailures() throws {
    let root = try makeTemporaryVLMBundle()
    try writeVLMConfig(to: root)
    try writeVLMPreprocessorConfig(to: root)
    try writeVLMTokenizerJSON(
        to: root,
        imageTokenContent: "<|wrong_image_token|>"
    )
    try writeVLMIndex(weightMap: makeVLMWeightMap(includeVision: false), to: root)

    let result = try QwenVLMBundlePreflightRunner.run(
        configuration: QwenVLMBundlePreflightConfiguration(modelRootURL: root)
    )

    #expect(!result.passesPreflight)
    #expect(result.failureReasons == [
        .missingVisionTensors,
        .tokenizerSpecialTokenMismatch,
    ])
    #expect(result.visionTensorPrefixes.isEmpty)
    #expect(result.missingLanguageTensorNames.isEmpty)
    #expect(result.tokenizerSpecialTokenChecks.map(\.matches) == [true, true, false])
}

@Test func qwenVLMBundlePreflightRejectsFamilyMismatch() throws {
    let root = try makeTemporaryVLMBundle()
    try writeVLMConfig(to: root)
    try writeVLMPreprocessorConfig(to: root)
    try writeVLMTokenizerJSON(to: root)
    try writeVLMIndex(weightMap: makeVLMWeightMap(), to: root)

    var rejected = false
    do {
        _ = try QwenVLMBundlePreflightRunner.run(
            configuration: QwenVLMBundlePreflightConfiguration(
                modelRootURL: root,
                modelFamily: .qwen36VLM
            )
        )
    } catch QwenVLMBundlePreflightError.modelFamilyMismatch(expected: .qwen36VLM, actual: .qwen35VLM) {
        rejected = true
    }
    #expect(rejected)
}

private func makeTemporaryVLMBundle() throws -> URL {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-vlm-preflight-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
}

private func writeVLMConfig(
    to root: URL,
    visionLayerCountKey: String = "num_hidden_layers",
    visionHeadCountKey: String = "num_attention_heads",
    modelType: String = "qwen3_5_vl"
) throws {
    try writeVLMJSON(
        """
        {
          "model_type": "\(modelType)",
          "image_token_id": 151655,
          "vision_start_token_id": 151652,
          "vision_end_token_id": 151653,
          "text_config": {
            "model_type": "qwen3_5",
            "vocab_size": 3,
            "hidden_size": 2,
            "intermediate_size": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 1,
            "linear_num_value_heads": 1,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 1,
            "linear_value_head_dim": 1,
            "linear_conv_kernel_dim": 4,
            "context_length": 8,
            "rms_norm_eps": 0.000001,
            "rope_theta": 10000,
            "partial_rotary_factor": 0.25,
            "layer_types": ["full_attention", "linear_attention"]
          },
          "vision_config": {
            "hidden_size": 4,
            "intermediate_size": 8,
            "\(visionLayerCountKey)": 2,
            "\(visionHeadCountKey)": 2,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "image_size": 448
          }
        }
        """,
        to: root.appendingPathComponent("config.json")
    )
}

private func writeVLMPreprocessorConfig(to root: URL) throws {
    try writeVLMJSON(
        """
        {
          "image_processor_type": "Qwen3VLImageProcessor",
          "min_pixels": 3136,
          "max_pixels": 1003520,
          "patch_size": 14,
          "merge_size": 2,
          "temporal_patch_size": 2,
          "image_mean": [0.48145466, 0.4578275, 0.40821073],
          "image_std": [0.26862954, 0.26130258, 0.27577711]
        }
        """,
        to: root.appendingPathComponent("preprocessor_config.json")
    )
}

private func writeVLMTokenizerJSON(
    to root: URL,
    imageTokenContent: String = "<|image_pad|>"
) throws {
    try writeVLMJSON(
        """
        {
          "added_tokens": [
            {"id": 151652, "content": "<|vision_start|>", "special": true},
            {"id": 151653, "content": "<|vision_end|>", "special": true},
            {"id": 151655, "content": "\(imageTokenContent)", "special": true}
          ]
        }
        """,
        to: root.appendingPathComponent("tokenizer.json")
    )
}

private func writeVLMIndex(weightMap: [String: String], to root: URL) throws {
    let payload: [String: Any] = [
        "weight_map": weightMap,
    ]
    let data = try JSONSerialization.data(
        withJSONObject: payload,
        options: [.prettyPrinted, .sortedKeys]
    )
    try data.write(to: root.appendingPathComponent("model.safetensors.index.json"))
}

private func writeVLMJSON(_ json: String, to url: URL) throws {
    try json.data(using: .utf8)!.write(to: url)
}

private func makeVLMWeightMap(
    includeVision: Bool = true,
    visionPrefix: String = "visual"
) -> [String: String] {
    var map = Dictionary(
        uniqueKeysWithValues: languageTensorNames().map {
            ($0, "model-00001-of-00001.safetensors")
        }
    )
    if includeVision {
        map["\(visionPrefix).patch_embed.proj.weight"] = "model-00001-of-00001.safetensors"
        map["\(visionPrefix).blocks.0.attn.qkv.weight"] = "model-00001-of-00001.safetensors"
    }
    return map
}

private func languageTensorNames() -> [String] {
    [
        "language_model.model.embed_tokens.weight",
        "language_model.model.norm.weight",
        "language_model.lm_head.weight",
        "language_model.model.layers.0.input_layernorm.weight",
        "language_model.model.layers.0.post_attention_layernorm.weight",
        "language_model.model.layers.0.mlp.gate_proj.weight",
        "language_model.model.layers.0.mlp.up_proj.weight",
        "language_model.model.layers.0.mlp.down_proj.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.0.self_attn.k_proj.weight",
        "language_model.model.layers.0.self_attn.v_proj.weight",
        "language_model.model.layers.0.self_attn.o_proj.weight",
        "language_model.model.layers.0.self_attn.q_norm.weight",
        "language_model.model.layers.0.self_attn.k_norm.weight",
        "language_model.model.layers.1.input_layernorm.weight",
        "language_model.model.layers.1.post_attention_layernorm.weight",
        "language_model.model.layers.1.mlp.gate_proj.weight",
        "language_model.model.layers.1.mlp.up_proj.weight",
        "language_model.model.layers.1.mlp.down_proj.weight",
        "language_model.model.layers.1.linear_attn.A_log",
        "language_model.model.layers.1.linear_attn.conv1d.weight",
        "language_model.model.layers.1.linear_attn.dt_bias",
        "language_model.model.layers.1.linear_attn.in_proj_a.weight",
        "language_model.model.layers.1.linear_attn.in_proj_b.weight",
        "language_model.model.layers.1.linear_attn.in_proj_qkv.weight",
        "language_model.model.layers.1.linear_attn.in_proj_z.weight",
        "language_model.model.layers.1.linear_attn.norm.weight",
        "language_model.model.layers.1.linear_attn.out_proj.weight",
    ]
}
