// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Testing
@testable import EdgeEngine

@Test func qwenVLMModelBundleIndexClassifiesLanguageAndVisionTensors() throws {
    let root = try makeTemporaryVLMIndexBundle()
    let languageNames = makeLanguageTensorNames(layerTypes: ["full_attention", "linear_attention"])
    let visionNames = [
        "vision_tower.patch_embed.proj.weight",
        "vision_tower.blocks.0.attn.qkv.weight",
    ]
    try writeVLMIndexBundle(
        root: root,
        tensorNames: languageNames + visionNames
    )

    let index = try QwenVLMModelBundleIndex.load(from: root)

    #expect(index.preflightResult.passesPreflight)
    #expect(index.languageManifest.role == .language)
    #expect(index.languageManifest.prefix == "language_model")
    #expect(index.languageManifest.tensorNames == languageNames.sorted())
    #expect(index.visionManifest.role == .vision)
    #expect(index.visionManifest.prefix == "vision_tower")
    #expect(index.visionManifest.tensorNames == visionNames.sorted())
}

@Test func qwenVLMModelBundleIndexClassifiesBareLanguageModelHead() throws {
    let root = try makeTemporaryVLMIndexBundle()
    let languageNames = makeLanguageTensorNames(
        layerTypes: ["full_attention", "linear_attention"],
        modelPrefix: "model"
    )
    let visionNames = [
        "vision_tower.patch_embed.proj.weight",
        "vision_tower.blocks.0.attn.qkv.weight",
    ]
    try writeVLMIndexBundle(
        root: root,
        tensorNames: languageNames + visionNames
    )

    let index = try QwenVLMModelBundleIndex.load(from: root)

    #expect(index.languageManifest.prefix == "model")
    #expect(index.languageManifest.tensorNames == languageNames.sorted())
    #expect(index.visionManifest.tensorNames == visionNames.sorted())
}

@Test func qwenVLMModelBundleIndexBuildsNativePhasedLoadingPlan() throws {
    let root = try makeTemporaryVLMIndexBundle()
    let languageNames = makeLanguageTensorNames(layerTypes: ["full_attention", "linear_attention"])
    let visionNames = [
        "vision_tower.patch_embed.proj.weight",
        "vision_tower.blocks.0.attn.qkv.weight",
    ]
    try writeVLMIndexBundle(
        root: root,
        tensorNames: languageNames + visionNames
    )
    let index = try QwenVLMModelBundleIndex.load(from: root)

    let footprint = try index.makeWeightFootprint()
    #expect(footprint.languageWeightBytes == languageNames.count * 4)
    #expect(footprint.visionWeightBytes == visionNames.count * 4)

    let tightPlan = try index.makePhasedLoadingPlan(
        jetsamLimitMB: 0,
        appReserveMB: 0,
        activationReserveMB: 0
    )
    #expect(tightPlan.requiresPhasedLoading)
    #expect(!tightPlan.isPhasedFeasible)
    #expect(tightPlan.textTurnStages == [
        .validateBundle,
        .loadStructureOnly,
        .loadDecoderWeights,
        .decode,
    ])
    #expect(tightPlan.imageTurnStages == [
        .validateBundle,
        .loadStructureOnly,
        .loadVisionWeights,
        .precomputeVisionFeatures,
        .unloadVisionWeights,
        .loadDecoderWeights,
        .decode,
    ])

    let roomyPlan = try index.makePhasedLoadingPlan(
        jetsamLimitMB: 4_096,
        appReserveMB: 0,
        activationReserveMB: 0
    )
    #expect(!roomyPlan.requiresPhasedLoading)
    #expect(roomyPlan.isPhasedFeasible)
    #expect(roomyPlan.imageTurnStages == [
        .validateBundle,
        .loadStructureOnly,
        .loadDecoderWeights,
        .loadVisionWeights,
        .precomputeVisionFeatures,
        .decode,
    ])
}

@Test func qwenVLMNativeContainerLoadsStructureOnly() throws {
    let root = try makeTemporaryVLMIndexBundle()
    let languageNames = makeLanguageTensorNames(layerTypes: ["full_attention", "linear_attention"])
    let visionNames = [
        "vision_tower.patch_embed.proj.weight",
        "vision_tower.blocks.0.attn.qkv.weight",
    ]
    try writeVLMIndexBundle(
        root: root,
        tensorNames: languageNames + visionNames
    )

    let container = try QwenVLMNativeContainer.loadStructureOnly(from: root)

    #expect(container.state.isStructureLoaded)
    #expect(!container.state.isDecoderLoaded)
    #expect(!container.state.isVisionLoaded)
    #expect(container.state.languageTensorCount == languageNames.count)
    #expect(container.state.visionTensorCount == visionNames.count)
    #expect(container.state.loadedVisionWeightBytes == 0)
    #expect(throws: QwenVLMNativeContainerError.decoderNotLoaded) {
        try container.makeGreedyDecodeSession(kvCapacity: 1)
    }
    #expect(throws: QwenVLMNativeContainerError.decoderNotLoaded) {
        try container.makeDecoderCaches(kvCapacity: 1)
    }
    #expect(throws: QwenVLMNativeContainerError.decoderNotLoaded) {
        try container.prefillCmlxTokens(tokenIDs: [1])
    }

    let unloadReport = container.unloadDecoderWeights()
    #expect(!unloadReport.hadDecoderLoaded)
    #expect(unloadReport.purgedQuantizedBufferStats.entryCount == 0)
}

@Test func qwenVisionWeightStoreMaterializesOnlyVisionTensors() throws {
    let root = try makeTemporaryVLMIndexBundle()
    let languageNames = makeLanguageTensorNames(layerTypes: ["full_attention", "linear_attention"])
    let visionNames = [
        "vision_tower.patch_embed.proj.weight",
        "vision_tower.blocks.0.attn.qkv.weight",
    ]
    try writeVLMIndexBundle(
        root: root,
        tensorNames: languageNames + visionNames
    )
    let index = try QwenVLMModelBundleIndex.load(from: root)

    let store = QwenVisionWeightStore(index: index)
    let snapshot = try store.materializeAllTensors()

    #expect(store.tensorNames == visionNames.sorted())
    #expect(snapshot.tensorNames == visionNames.sorted())
    #expect(snapshot.tensors.map(\.byteCount) == [4, 4])
    #expect(snapshot.totalByteCount == 8)
    #expect(try store.metadata(named: visionNames[0]).shape == [1])
    #expect(throws: QwenVisionWeightStoreError.tensorNotInVisionManifest(languageNames[0])) {
        _ = try store.tensorData(named: languageNames[0])
    }
}

@Test func qwenVLMNativeContainerLoadsAndOffloadsVisionWeights() throws {
    let root = try makeTemporaryVLMIndexBundle()
    let languageNames = makeLanguageTensorNames(layerTypes: ["full_attention", "linear_attention"])
    let visionNames = [
        "vision_tower.patch_embed.proj.weight",
        "vision_tower.blocks.0.attn.qkv.weight",
    ]
    try writeVLMIndexBundle(
        root: root,
        tensorNames: languageNames + visionNames
    )
    let container = try QwenVLMNativeContainer.loadStructureOnly(from: root)

    try container.loadVisionWeights()

    #expect(container.isVisionLoaded)
    #expect(container.state.isVisionLoaded)
    #expect(container.state.loadedVisionWeightBytes == 8)
    #expect(throws: QwenVLMNativeContainerError.visionAlreadyLoaded) {
        try container.loadVisionWeights()
    }

    let report = container.unloadVisionWeights()
    #expect(report.hadVisionLoaded)
    #expect(report.releasedTensorCount == visionNames.count)
    #expect(report.releasedByteCount == 8)
    #expect(!container.isVisionLoaded)
    #expect(!container.state.isVisionLoaded)
    #expect(container.state.loadedVisionWeightBytes == 0)
}

@Test func qwenVLMModelBundleIndexRejectsUnclassifiedTensors() throws {
    let root = try makeTemporaryVLMIndexBundle()
    let languageNames = makeLanguageTensorNames(layerTypes: ["full_attention", "linear_attention"])
    let visionNames = [
        "vision_tower.patch_embed.proj.weight",
        "vision_tower.blocks.0.attn.qkv.weight",
    ]
    let unclassifiedName = "merger.weight"
    try writeVLMIndexBundle(
        root: root,
        tensorNames: languageNames + visionNames + [unclassifiedName]
    )

    #expect(throws: QwenVLMModelBundleIndexError.unclassifiedTensors([unclassifiedName])) {
        try QwenVLMModelBundleIndex.load(from: root)
    }
}

@Test func nativeRuntimeBridgeLoadsQwenVLMStructureOnlyContainer() throws {
    let root = try makeTemporaryVLMIndexBundle()
    let languageNames = makeLanguageTensorNames(layerTypes: ["full_attention", "linear_attention"])
    let visionNames = [
        "vision_tower.patch_embed.proj.weight",
        "vision_tower.blocks.0.attn.qkv.weight",
    ]
    try writeVLMIndexBundle(
        root: root,
        tensorNames: languageNames + visionNames
    )

    let container = try NativeRuntimeBridge.loadQwenVLMStructureOnly(modelRootURL: root)

    #expect(container.index.languageManifest.tensorNames == languageNames.sorted())
    #expect(container.index.visionManifest.tensorNames == visionNames.sorted())
    #expect(!container.isDecoderLoaded)
}

private func makeTemporaryVLMIndexBundle() throws -> URL {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edge-engine-qwen-vlm-index-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
}

private func writeVLMIndexBundle(root: URL, tensorNames: [String]) throws {
    try writeVLMIndexConfig(to: root)
    try writeVLMIndexPreprocessorConfig(to: root)
    try writeVLMIndexTokenizerJSON(to: root)
    try writeVLMIndex(weightMap: Dictionary(uniqueKeysWithValues: tensorNames.map {
        ($0, "model.safetensors")
    }), to: root)
    try writeSafeTensorsFile(
        tensorNames: tensorNames,
        to: root.appendingPathComponent("model.safetensors")
    )
}

private func writeVLMIndexConfig(to root: URL) throws {
    try writeText(
        """
        {
          "model_type": "qwen3_5",
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
            "depth": 2,
            "num_heads": 2,
            "patch_size": 14,
            "spatial_merge_size": 2
          }
        }
        """,
        to: root.appendingPathComponent("config.json")
    )
}

private func writeVLMIndexPreprocessorConfig(to root: URL) throws {
    try writeText(
        """
        {
          "image_processor_type": "Qwen2VLImageProcessorFast",
          "patch_size": 14,
          "merge_size": 2,
          "temporal_patch_size": 2,
          "image_mean": [0.5, 0.5, 0.5],
          "image_std": [0.5, 0.5, 0.5]
        }
        """,
        to: root.appendingPathComponent("preprocessor_config.json")
    )
}

private func writeVLMIndexTokenizerJSON(to root: URL) throws {
    try writeText(
        """
        {
          "added_tokens": [
            {"id": 151652, "content": "<|vision_start|>", "special": true},
            {"id": 151653, "content": "<|vision_end|>", "special": true},
            {"id": 151655, "content": "<|image_pad|>", "special": true}
          ]
        }
        """,
        to: root.appendingPathComponent("tokenizer.json")
    )
}

private func writeVLMIndex(weightMap: [String: String], to root: URL) throws {
    let data = try JSONSerialization.data(
        withJSONObject: ["weight_map": weightMap],
        options: [.prettyPrinted, .sortedKeys]
    )
    try data.write(to: root.appendingPathComponent("model.safetensors.index.json"))
}

private func writeSafeTensorsFile(tensorNames: [String], to url: URL) throws {
    var offset = 0
    var header: [String: Any] = [:]
    for name in tensorNames {
        header[name] = [
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [offset, offset + 4],
        ]
        offset += 4
    }
    let headerData = try JSONSerialization.data(
        withJSONObject: header,
        options: [.sortedKeys]
    )
    var data = Data()
    var headerLength = UInt64(headerData.count).littleEndian
    withUnsafeBytes(of: &headerLength) { data.append(contentsOf: $0) }
    data.append(headerData)
    data.append(Data(repeating: 0, count: offset))
    try data.write(to: url)
}

private func writeText(_ text: String, to url: URL) throws {
    try text.data(using: .utf8)!.write(to: url)
}

private func makeLanguageTensorNames(
    layerTypes: [String],
    modelPrefix: String = "language_model.model"
) -> [String] {
    let languageRoot = modelPrefix.hasSuffix(".model")
        ? String(modelPrefix.dropLast(".model".count))
        : modelPrefix
    let lmHeadName = modelPrefix == "model"
        ? "lm_head.weight"
        : "\(languageRoot).lm_head.weight"
    var tensorNames = [
        "\(modelPrefix).embed_tokens.weight",
        "\(modelPrefix).norm.weight",
        lmHeadName,
    ]
    for (layerIndex, layerType) in layerTypes.enumerated() {
        let prefix = "\(modelPrefix).layers.\(layerIndex)"
        tensorNames.append(contentsOf: [
            "\(prefix).input_layernorm.weight",
            "\(prefix).post_attention_layernorm.weight",
            "\(prefix).mlp.gate_proj.weight",
            "\(prefix).mlp.up_proj.weight",
            "\(prefix).mlp.down_proj.weight",
        ])
        switch layerType {
        case "full_attention":
            tensorNames.append(contentsOf: [
                "\(prefix).self_attn.q_proj.weight",
                "\(prefix).self_attn.k_proj.weight",
                "\(prefix).self_attn.v_proj.weight",
                "\(prefix).self_attn.o_proj.weight",
                "\(prefix).self_attn.q_norm.weight",
                "\(prefix).self_attn.k_norm.weight",
            ])
        case "linear_attention":
            tensorNames.append(contentsOf: [
                "\(prefix).linear_attn.A_log",
                "\(prefix).linear_attn.conv1d.weight",
                "\(prefix).linear_attn.dt_bias",
                "\(prefix).linear_attn.in_proj_a.weight",
                "\(prefix).linear_attn.in_proj_b.weight",
                "\(prefix).linear_attn.in_proj_qkv.weight",
                "\(prefix).linear_attn.in_proj_z.weight",
                "\(prefix).linear_attn.norm.weight",
                "\(prefix).linear_attn.out_proj.weight",
            ])
        default:
            Issue.record("Unsupported fixture layer type: \(layerType)")
        }
    }
    return tensorNames
}
