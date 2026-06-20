// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum QwenModelBundleIndexError: Error, Equatable {
    case missingConfig(URL)
    case missingWeightIndex(URL)
    case invalidWeightIndex
    case unsupportedModelPrefix
    case missingTensor(String)
    case missingLayerManifest(Int)
}

public enum QwenLayerWeightNamespace: String, Codable, Equatable, Sendable {
    case selfAttention = "self_attn"
    case linearAttention = "linear_attn"
}

public struct QwenQuantizedWeightGroup: Codable, Equatable, Sendable {
    public var weightName: String
    public var scalesName: String?
    public var biasesName: String?

    public init(weightName: String, scalesName: String?, biasesName: String?) {
        self.weightName = weightName
        self.scalesName = scalesName
        self.biasesName = biasesName
    }

    public var isQuantized: Bool {
        scalesName != nil || biasesName != nil
    }
}

public struct QwenLayerWeightManifest: Codable, Equatable, Sendable {
    public var layerIndex: Int
    public var kind: QwenHybridLayerKind
    public var layerPrefix: String
    public var attentionNamespace: QwenLayerWeightNamespace
    public var requiredTensorNames: [String]
    public var optionalTensorNames: [String]
    public var quantizedWeightGroups: [QwenQuantizedWeightGroup]

    public init(
        layerIndex: Int,
        kind: QwenHybridLayerKind,
        layerPrefix: String,
        attentionNamespace: QwenLayerWeightNamespace,
        requiredTensorNames: [String],
        optionalTensorNames: [String],
        quantizedWeightGroups: [QwenQuantizedWeightGroup]
    ) {
        self.layerIndex = layerIndex
        self.kind = kind
        self.layerPrefix = layerPrefix
        self.attentionNamespace = attentionNamespace
        self.requiredTensorNames = requiredTensorNames
        self.optionalTensorNames = optionalTensorNames
        self.quantizedWeightGroups = quantizedWeightGroups
    }

    public func missingRequiredTensorNames(in weightMap: [String: String]) -> [String] {
        requiredTensorNames.filter { weightMap[$0] == nil }
    }
}

public struct QwenModelLevelManifest: Codable, Equatable, Sendable {
    public var embedTokensName: String
    public var finalNormName: String
    public var lmHeadName: String?
    public var isWeightTied: Bool
    public var requiredTensorNames: [String]
    public var optionalTensorNames: [String]
    public var quantizedWeightGroups: [QwenQuantizedWeightGroup]

    public init(
        embedTokensName: String,
        finalNormName: String,
        lmHeadName: String?,
        requiredTensorNames: [String],
        optionalTensorNames: [String],
        quantizedWeightGroups: [QwenQuantizedWeightGroup]
    ) {
        self.embedTokensName = embedTokensName
        self.finalNormName = finalNormName
        self.lmHeadName = lmHeadName
        self.isWeightTied = lmHeadName == nil
        self.requiredTensorNames = requiredTensorNames
        self.optionalTensorNames = optionalTensorNames
        self.quantizedWeightGroups = quantizedWeightGroups
    }

    public func missingRequiredTensorNames(in weightMap: [String: String]) -> [String] {
        requiredTensorNames.filter { weightMap[$0] == nil }
    }
}

public struct QwenModelBundleIndex: Equatable, Sendable {
    public var rootURL: URL
    public var architecture: QwenHybridArchitecture
    public var executionPlan: QwenExecutionPlan
    public var weightMap: [String: String]
    public var modelPrefix: String
    public var modelLevelManifest: QwenModelLevelManifest
    public var layerManifests: [QwenLayerWeightManifest]

    public init(
        rootURL: URL,
        architecture: QwenHybridArchitecture,
        weightMap: [String: String]
    ) throws {
        self.rootURL = rootURL
        self.architecture = architecture
        self.executionPlan = try QwenExecutionPlan(architecture: architecture)
        self.weightMap = weightMap
        self.modelPrefix = try Self.detectModelPrefix(weightMap: weightMap)
        self.modelLevelManifest = Self.makeModelLevelManifest(
            weightMap: weightMap,
            modelPrefix: modelPrefix
        )
        self.layerManifests = Self.makeLayerManifests(
            architecture: architecture,
            weightMap: weightMap,
            modelPrefix: modelPrefix
        )
    }

    public static func load(
        from rootURL: URL,
        family explicitFamily: QwenModelFamily? = nil
    ) throws -> QwenModelBundleIndex {
        let configURL = rootURL.appendingPathComponent("config.json")
        let indexURL = rootURL.appendingPathComponent("model.safetensors.index.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw QwenModelBundleIndexError.missingConfig(configURL)
        }
        guard FileManager.default.fileExists(atPath: indexURL.path) else {
            throw QwenModelBundleIndexError.missingWeightIndex(indexURL)
        }

        let architecture = try QwenConfigDecoder.decodeArchitecture(
            from: try Data(contentsOf: configURL),
            family: explicitFamily
        )
        let index = try JSONDecoder().decode(
            RawSafeTensorsIndex.self,
            from: try Data(contentsOf: indexURL)
        )
        guard !index.weightMap.isEmpty else {
            throw QwenModelBundleIndexError.invalidWeightIndex
        }
        return try QwenModelBundleIndex(
            rootURL: rootURL,
            architecture: architecture,
            weightMap: index.weightMap
        )
    }

    public var missingRequiredTensorNames: [String] {
        modelLevelManifest.missingRequiredTensorNames(in: weightMap)
            + layerManifests.flatMap { $0.missingRequiredTensorNames(in: weightMap) }
    }

    public func validateRequiredTensorCoverage() throws {
        if let missing = missingRequiredTensorNames.first {
            throw QwenModelBundleIndexError.missingTensor(missing)
        }
    }

    public func layerManifest(_ layerIndex: Int) throws -> QwenLayerWeightManifest {
        guard let manifest = layerManifests.first(where: { $0.layerIndex == layerIndex }) else {
            throw QwenModelBundleIndexError.missingLayerManifest(layerIndex)
        }
        return manifest
    }

    public func shardFileName(containing tensorName: String) throws -> String {
        guard let fileName = weightMap[tensorName] else {
            throw QwenModelBundleIndexError.missingTensor(tensorName)
        }
        return fileName
    }

    public func shardFileURL(containing tensorName: String) throws -> URL {
        rootURL.appendingPathComponent(try shardFileName(containing: tensorName))
    }

    public func tensorNames(prefix: String) -> [String] {
        weightMap.keys
            .filter { $0.hasPrefix(prefix) }
            .sorted()
    }

    /// Supports public Qwen3.5/Qwen3.6 bundle naming:
    /// - VLM/converted bundles: `language_model.model.layers.{i}.*`
    /// - Bare LLM bundles: `model.layers.{i}.*`
    /// New families must extend this prefix list explicitly.
    private static func detectModelPrefix(weightMap: [String: String]) throws -> String {
        let prefixes = ["language_model.model", "model"]
        for prefix in prefixes where weightMap.keys.contains(where: { $0.hasPrefix("\(prefix).layers.") }) {
            return prefix
        }
        throw QwenModelBundleIndexError.unsupportedModelPrefix
    }

    private static func makeModelLevelManifest(
        weightMap: [String: String],
        modelPrefix: String
    ) -> QwenModelLevelManifest {
        let embed = "\(modelPrefix).embed_tokens.weight"
        let finalNorm = "\(modelPrefix).norm.weight"
        let lmHead = lmHeadCandidates(modelPrefix: modelPrefix)
            .first { weightMap[$0] != nil }
        var required = [embed, finalNorm]
        if let lmHead {
            required.append(lmHead)
        }
        let optional = optionalTensorNames(requiredBaseNames: required, weightMap: weightMap)
        return QwenModelLevelManifest(
            embedTokensName: embed,
            finalNormName: finalNorm,
            lmHeadName: lmHead,
            requiredTensorNames: required,
            optionalTensorNames: optional,
            quantizedWeightGroups: quantizedGroups(
                requiredBaseNames: required,
                optionalTensorNames: optional
            )
        )
    }

    private static func lmHeadCandidates(modelPrefix: String) -> [String] {
        var candidates: [String] = []
        if modelPrefix.hasSuffix(".model") {
            candidates.append(String(modelPrefix.dropLast(".model".count)) + ".lm_head.weight")
        }
        candidates += [
            "\(modelPrefix).lm_head.weight",
            "lm_head.weight",
        ]
        return candidates
    }

    private static func makeLayerManifests(
        architecture: QwenHybridArchitecture,
        weightMap: [String: String],
        modelPrefix: String
    ) -> [QwenLayerWeightManifest] {
        architecture.layerPlan.map { layer in
            makeLayerManifest(
                layer: layer,
                usesSparseMoEMLP: architecture.usesSparseMoEMLP,
                weightMap: weightMap,
                modelPrefix: modelPrefix
            )
        }
    }

    private static func makeLayerManifest(
        layer: QwenHybridLayerPlan,
        usesSparseMoEMLP: Bool,
        weightMap: [String: String],
        modelPrefix: String
    ) -> QwenLayerWeightManifest {
        let layerPrefix = "\(modelPrefix).layers.\(layer.index)"
        let attentionNamespace: QwenLayerWeightNamespace = layer.kind == .fullAttention
            ? .selfAttention
            : .linearAttention
        let required = requiredTensorNames(
            layerPrefix: layerPrefix,
            attentionNamespace: attentionNamespace,
            usesSparseMoEMLP: usesSparseMoEMLP
        )
        let optional = optionalTensorNames(requiredBaseNames: required, weightMap: weightMap)
        return QwenLayerWeightManifest(
            layerIndex: layer.index,
            kind: layer.kind,
            layerPrefix: layerPrefix,
            attentionNamespace: attentionNamespace,
            requiredTensorNames: required,
            optionalTensorNames: optional,
            quantizedWeightGroups: quantizedGroups(
                requiredBaseNames: required,
                optionalTensorNames: optional
            )
        )
    }

    private static func requiredTensorNames(
        layerPrefix: String,
        attentionNamespace: QwenLayerWeightNamespace,
        usesSparseMoEMLP: Bool
    ) -> [String] {
        var names = [
            "\(layerPrefix).input_layernorm.weight",
            "\(layerPrefix).post_attention_layernorm.weight",
        ]
        if usesSparseMoEMLP {
            names += [
                "\(layerPrefix).mlp.gate.weight",
                "\(layerPrefix).mlp.switch_mlp.gate_proj.weight",
                "\(layerPrefix).mlp.switch_mlp.up_proj.weight",
                "\(layerPrefix).mlp.switch_mlp.down_proj.weight",
                "\(layerPrefix).mlp.shared_expert.gate_proj.weight",
                "\(layerPrefix).mlp.shared_expert.up_proj.weight",
                "\(layerPrefix).mlp.shared_expert.down_proj.weight",
                "\(layerPrefix).mlp.shared_expert_gate.weight",
            ]
        } else {
            names += [
                "\(layerPrefix).mlp.gate_proj.weight",
                "\(layerPrefix).mlp.up_proj.weight",
                "\(layerPrefix).mlp.down_proj.weight",
            ]
        }

        switch attentionNamespace {
        case .selfAttention:
            let prefix = "\(layerPrefix).self_attn"
            names += [
                "\(prefix).q_proj.weight",
                "\(prefix).k_proj.weight",
                "\(prefix).v_proj.weight",
                "\(prefix).o_proj.weight",
                "\(prefix).q_norm.weight",
                "\(prefix).k_norm.weight",
            ]
        case .linearAttention:
            let prefix = "\(layerPrefix).linear_attn"
            names += [
                "\(prefix).A_log",
                "\(prefix).conv1d.weight",
                "\(prefix).dt_bias",
                "\(prefix).in_proj_a.weight",
                "\(prefix).in_proj_b.weight",
                "\(prefix).in_proj_qkv.weight",
                "\(prefix).in_proj_z.weight",
                "\(prefix).norm.weight",
                "\(prefix).out_proj.weight",
            ]
        }

        return names
    }

    private static func optionalTensorNames(
        requiredBaseNames: [String],
        weightMap: [String: String]
    ) -> [String] {
        requiredBaseNames
            .filter { $0.hasSuffix(".weight") }
            .flatMap { weightName in
                ["scales", "biases"].map { suffix in
                    String(weightName.dropLast(".weight".count)) + ".\(suffix)"
                }
            }
            .filter { weightMap[$0] != nil }
            .sorted()
    }

    private static func quantizedGroups(
        requiredBaseNames: [String],
        optionalTensorNames: [String]
    ) -> [QwenQuantizedWeightGroup] {
        let optional = Set(optionalTensorNames)
        return requiredBaseNames
            .filter { $0.hasSuffix(".weight") }
            .map { weightName in
                let base = String(weightName.dropLast(".weight".count))
                let scales = "\(base).scales"
                let biases = "\(base).biases"
                return QwenQuantizedWeightGroup(
                    weightName: weightName,
                    scalesName: optional.contains(scales) ? scales : nil,
                    biasesName: optional.contains(biases) ? biases : nil
                )
            }
    }
}

private struct RawSafeTensorsIndex: Decodable {
    var weightMap: [String: String]

    private enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}
