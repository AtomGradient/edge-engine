// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum QwenConfigDecoderError: Error, Equatable {
    case missingFamily
    case unsupportedFamily(String)
    case missingHybridLayerPlan
    case unsupportedLayerKind(String)
}

public enum QwenConfigDecoder {
    public static func decodeArchitecture(
        from data: Data,
        family explicitFamily: QwenModelFamily? = nil
    ) throws -> QwenHybridArchitecture {
        let raw = try JSONDecoder().decode(RawQwenConfig.self, from: data)
        let family = try explicitFamily ?? decodeFamily(raw.familyHint)
        let layerKinds = try decodeLayerKinds(raw.layerPlan)

        return try QwenHybridArchitecture(
            family: family,
            vocabularySize: raw.vocabularySize,
            hiddenSize: raw.hiddenSize,
            intermediateSize: raw.intermediateSize,
            moeMLP: raw.moeMLP,
            attentionHeadCount: raw.attentionHeadCount,
            keyValueHeadCount: raw.keyValueHeadCount,
            headDimension: raw.headDimension,
            linearValueHeadCount: raw.linearValueHeadCount,
            linearKeyHeadCount: raw.linearKeyHeadCount,
            linearKeyHeadDimension: raw.linearKeyHeadDimension,
            linearValueHeadDimension: raw.linearValueHeadDimension,
            linearConvKernelSize: raw.linearConvKernelSize ?? 4,
            contextLength: raw.contextLength,
            rmsNormEpsilon: raw.rmsNormEpsilon,
            ropeTheta: raw.ropeTheta,
            partialRotaryFactor: raw.partialRotaryFactor,
            quantization: raw.quantization,
            layerKinds: layerKinds
        )
    }

    private static func decodeFamily(_ hint: String?) throws -> QwenModelFamily {
        guard let hint else {
            throw QwenConfigDecoderError.missingFamily
        }
        let normalized = hint
            .lowercased()
            .replacingOccurrences(of: "-", with: "")
            .replacingOccurrences(of: "_", with: "")
            .replacingOccurrences(of: ".", with: "")

        if normalized.contains("qwen35") || normalized.contains("qwen3p5") {
            return .qwen35
        }
        if normalized.contains("qwen36") || normalized.contains("qwen3p6") {
            return .qwen36
        }
        throw QwenConfigDecoderError.unsupportedFamily(hint)
    }

    private static func decodeLayerKinds(_ plan: [String]?) throws -> [QwenHybridLayerKind] {
        guard let plan, !plan.isEmpty else {
            throw QwenConfigDecoderError.missingHybridLayerPlan
        }
        return try plan.map { rawKind in
            let normalized = rawKind
                .lowercased()
                .replacingOccurrences(of: "-", with: "_")
            switch normalized {
            case "full_attention", "fullattention", "attention", "fa":
                return .fullAttention
            case "gdn", "gated_delta", "gateddelta", "gated_delta_net", "gateddeltanet",
                "linear_attention", "linearattention":
                return .gdn
            default:
                throw QwenConfigDecoderError.unsupportedLayerKind(rawKind)
            }
        }
    }
}

private struct RawQwenConfig: Decodable {
    var familyHint: String?
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeMLP: QwenMoEMLPConfiguration?
    var attentionHeadCount: Int
    var keyValueHeadCount: Int
    var headDimension: Int?
    var linearValueHeadCount: Int?
    var linearKeyHeadCount: Int?
    var linearKeyHeadDimension: Int?
    var linearValueHeadDimension: Int?
    var linearConvKernelSize: Int?
    var contextLength: Int
    var rmsNormEpsilon: Float
    var ropeTheta: Float
    var partialRotaryFactor: Float
    var quantization: QwenQuantizationProfile?
    var layerPlan: [String]?

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case family
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case normalizeTopKProbabilities = "norm_topk_prob"
        case attentionHeadCount = "num_attention_heads"
        case keyValueHeadCount = "num_key_value_heads"
        case headDimension = "head_dim"
        case linearValueHeadCount = "linear_num_value_heads"
        case linearKeyHeadCount = "linear_num_key_heads"
        case linearKeyHeadDimension = "linear_key_head_dim"
        case linearValueHeadDimension = "linear_value_head_dim"
        case linearConvKernelSize = "linear_conv_kernel_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case contextLength = "context_length"
        case rmsNormEpsilon = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeParameters = "rope_parameters"
        case quantization
        case edgeruntimeLayerPlan = "edgeruntime_layer_plan"
        case hybridLayerTypes = "hybrid_layer_types"
        case layerPlan = "layer_plan"
        case layerTypes = "layer_types"
        case textConfig = "text_config"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let textConfig = try container.decodeIfPresent(RawQwenTextConfig.self, forKey: .textConfig)
        let ropeParameters = try container.decodeIfPresent(RawQwenRopeParameters.self, forKey: .ropeParameters)
        familyHint = try container.decodeIfPresent(String.self, forKey: .family)
            ?? container.decodeIfPresent(String.self, forKey: .modelType)
            ?? textConfig?.familyHint
        vocabularySize = try Self.decodeRequiredInt(
            root: container,
            rootKey: .vocabularySize,
            nestedValue: textConfig?.vocabularySize
        )
        hiddenSize = try Self.decodeRequiredInt(
            root: container,
            rootKey: .hiddenSize,
            nestedValue: textConfig?.hiddenSize
        )
        let denseIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
            ?? textConfig?.intermediateSize
        let numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts)
            ?? textConfig?.numExperts
        let numExpertsPerToken = try container.decodeIfPresent(Int.self, forKey: .numExpertsPerToken)
            ?? textConfig?.numExpertsPerToken
        let moeIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
            ?? textConfig?.moeIntermediateSize
        let sharedExpertIntermediateSize = try container.decodeIfPresent(
            Int.self,
            forKey: .sharedExpertIntermediateSize
        ) ?? textConfig?.sharedExpertIntermediateSize
        let normalizeTopKProbabilities = try container.decodeIfPresent(
            Bool.self,
            forKey: .normalizeTopKProbabilities
        ) ?? textConfig?.normalizeTopKProbabilities ?? true
        if let numExperts,
           let numExpertsPerToken,
           let moeIntermediateSize,
           let sharedExpertIntermediateSize,
           numExperts > 0,
           numExpertsPerToken > 0,
           moeIntermediateSize > 0,
           sharedExpertIntermediateSize > 0 {
            moeMLP = QwenMoEMLPConfiguration(
                expertCount: numExperts,
                expertsPerToken: numExpertsPerToken,
                intermediateSize: moeIntermediateSize,
                sharedExpertIntermediateSize: sharedExpertIntermediateSize,
                normalizeTopKProbabilities: normalizeTopKProbabilities
            )
            intermediateSize = denseIntermediateSize ?? moeIntermediateSize * numExpertsPerToken
        } else {
            moeMLP = nil
            intermediateSize = try denseIntermediateSize ?? Self.throwMissing(.intermediateSize)
        }
        attentionHeadCount = try Self.decodeRequiredInt(
            root: container,
            rootKey: .attentionHeadCount,
            nestedValue: textConfig?.attentionHeadCount
        )
        keyValueHeadCount = try Self.decodeRequiredInt(
            root: container,
            rootKey: .keyValueHeadCount,
            nestedValue: textConfig?.keyValueHeadCount
        )
        headDimension = try container.decodeIfPresent(Int.self, forKey: .headDimension)
            ?? textConfig?.headDimension
        linearValueHeadCount = try container.decodeIfPresent(Int.self, forKey: .linearValueHeadCount)
            ?? textConfig?.linearValueHeadCount
        linearKeyHeadCount = try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadCount)
            ?? textConfig?.linearKeyHeadCount
        linearKeyHeadDimension = try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDimension)
            ?? textConfig?.linearKeyHeadDimension
        linearValueHeadDimension = try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDimension)
            ?? textConfig?.linearValueHeadDimension
        linearConvKernelSize = try container.decodeIfPresent(Int.self, forKey: .linearConvKernelSize)
            ?? textConfig?.linearConvKernelSize
        contextLength = try container.decodeIfPresent(Int.self, forKey: .contextLength)
            ?? container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? textConfig?.contextLength
            ?? Self.throwMissing(.contextLength)
        rmsNormEpsilon = try Self.decodeRequiredFloat(
            root: container,
            rootKey: .rmsNormEpsilon,
            nestedValue: textConfig?.rmsNormEpsilon
        )
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
            ?? ropeParameters?.ropeTheta
            ?? textConfig?.ropeTheta
            ?? Self.throwMissing(.ropeTheta)
        partialRotaryFactor = try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor)
            ?? ropeParameters?.partialRotaryFactor
            ?? textConfig?.partialRotaryFactor
            ?? 0.25
        quantization = try container.decodeIfPresent(QwenQuantizationProfile.self, forKey: .quantization)
            ?? textConfig?.quantization
        layerPlan = try container.decodeIfPresent([String].self, forKey: .edgeruntimeLayerPlan)
            ?? container.decodeIfPresent([String].self, forKey: .hybridLayerTypes)
            ?? container.decodeIfPresent([String].self, forKey: .layerPlan)
            ?? container.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? textConfig?.layerPlan
    }

    private static func decodeRequiredInt(
        root: KeyedDecodingContainer<CodingKeys>,
        rootKey: CodingKeys,
        nestedValue: Int?
    ) throws -> Int {
        try root.decodeIfPresent(Int.self, forKey: rootKey) ?? nestedValue ?? throwMissing(rootKey)
    }

    private static func decodeRequiredFloat(
        root: KeyedDecodingContainer<CodingKeys>,
        rootKey: CodingKeys,
        nestedValue: Float?
    ) throws -> Float {
        try root.decodeIfPresent(Float.self, forKey: rootKey) ?? nestedValue ?? throwMissing(rootKey)
    }

    private static func throwMissing<T>(_ key: CodingKeys) throws -> T {
        throw DecodingError.keyNotFound(
            key,
            DecodingError.Context(
                codingPath: [],
                debugDescription: "Missing required Qwen config field: \(key.rawValue)"
            )
        )
    }
}

private struct RawQwenTextConfig: Decodable {
    var familyHint: String?
    var vocabularySize: Int?
    var hiddenSize: Int?
    var intermediateSize: Int?
    var numExperts: Int?
    var numExpertsPerToken: Int?
    var moeIntermediateSize: Int?
    var sharedExpertIntermediateSize: Int?
    var normalizeTopKProbabilities: Bool?
    var attentionHeadCount: Int?
    var keyValueHeadCount: Int?
    var headDimension: Int?
    var linearValueHeadCount: Int?
    var linearKeyHeadCount: Int?
    var linearKeyHeadDimension: Int?
    var linearValueHeadDimension: Int?
    var linearConvKernelSize: Int?
    var contextLength: Int?
    var rmsNormEpsilon: Float?
    var ropeTheta: Float?
    var partialRotaryFactor: Float?
    var quantization: QwenQuantizationProfile?
    var layerPlan: [String]?

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case family
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case normalizeTopKProbabilities = "norm_topk_prob"
        case attentionHeadCount = "num_attention_heads"
        case keyValueHeadCount = "num_key_value_heads"
        case headDimension = "head_dim"
        case linearValueHeadCount = "linear_num_value_heads"
        case linearKeyHeadCount = "linear_num_key_heads"
        case linearKeyHeadDimension = "linear_key_head_dim"
        case linearValueHeadDimension = "linear_value_head_dim"
        case linearConvKernelSize = "linear_conv_kernel_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case contextLength = "context_length"
        case rmsNormEpsilon = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeParameters = "rope_parameters"
        case quantization
        case edgeruntimeLayerPlan = "edgeruntime_layer_plan"
        case hybridLayerTypes = "hybrid_layer_types"
        case layerPlan = "layer_plan"
        case layerTypes = "layer_types"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let ropeParameters = try container.decodeIfPresent(RawQwenRopeParameters.self, forKey: .ropeParameters)
        familyHint = try container.decodeIfPresent(String.self, forKey: .family)
            ?? container.decodeIfPresent(String.self, forKey: .modelType)
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize)
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
        numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts)
        numExpertsPerToken = try container.decodeIfPresent(Int.self, forKey: .numExpertsPerToken)
        moeIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
        sharedExpertIntermediateSize = try container.decodeIfPresent(
            Int.self,
            forKey: .sharedExpertIntermediateSize
        )
        normalizeTopKProbabilities = try container.decodeIfPresent(
            Bool.self,
            forKey: .normalizeTopKProbabilities
        )
        attentionHeadCount = try container.decodeIfPresent(Int.self, forKey: .attentionHeadCount)
        keyValueHeadCount = try container.decodeIfPresent(Int.self, forKey: .keyValueHeadCount)
        headDimension = try container.decodeIfPresent(Int.self, forKey: .headDimension)
        linearValueHeadCount = try container.decodeIfPresent(Int.self, forKey: .linearValueHeadCount)
        linearKeyHeadCount = try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadCount)
        linearKeyHeadDimension = try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDimension)
        linearValueHeadDimension = try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDimension)
        linearConvKernelSize = try container.decodeIfPresent(Int.self, forKey: .linearConvKernelSize)
        contextLength = try container.decodeIfPresent(Int.self, forKey: .contextLength)
            ?? container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        rmsNormEpsilon = try container.decodeIfPresent(Float.self, forKey: .rmsNormEpsilon)
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
            ?? ropeParameters?.ropeTheta
        partialRotaryFactor = try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor)
            ?? ropeParameters?.partialRotaryFactor
        quantization = try container.decodeIfPresent(QwenQuantizationProfile.self, forKey: .quantization)
        layerPlan = try container.decodeIfPresent([String].self, forKey: .edgeruntimeLayerPlan)
            ?? container.decodeIfPresent([String].self, forKey: .hybridLayerTypes)
            ?? container.decodeIfPresent([String].self, forKey: .layerPlan)
            ?? container.decodeIfPresent([String].self, forKey: .layerTypes)
    }
}

private struct RawQwenRopeParameters: Decodable {
    var ropeTheta: Float?
    var partialRotaryFactor: Float?

    private enum CodingKeys: String, CodingKey {
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
    }
}
