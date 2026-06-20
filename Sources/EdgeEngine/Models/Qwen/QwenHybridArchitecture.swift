// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public enum QwenModelFamily: String, Codable, Equatable, Sendable {
    case qwen35
    case qwen36
}

public enum QwenHybridLayerKind: String, Codable, Equatable, Sendable {
    case fullAttention
    case gdn
}

public struct QwenHybridLayerPlan: Codable, Equatable, Sendable {
    public var index: Int
    public var kind: QwenHybridLayerKind

    public init(index: Int, kind: QwenHybridLayerKind) {
        self.index = index
        self.kind = kind
    }
}

public struct QwenQuantizationProfile: Codable, Equatable, Sendable {
    public var groupSize: Int
    public var bits: Int

    private enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
    }

    public init(groupSize: Int, bits: Int) {
        self.groupSize = groupSize
        self.bits = bits
    }
}

public enum QwenArchitectureError: Error, Equatable {
    case invalidVocabularySize(Int)
    case invalidHiddenSize(Int)
    case invalidIntermediateSize(Int)
    case invalidMoEExpertCount(Int)
    case invalidMoEExpertsPerToken(Int)
    case invalidMoEIntermediateSize(Int)
    case invalidMoESharedExpertIntermediateSize(Int)
    case invalidAttentionHeadCount(Int)
    case invalidKeyValueHeadCount(Int)
    case invalidAttentionHeadDimension(Int)
    case invalidLinearValueHeadCount(Int)
    case invalidLinearKeyHeadCount(Int)
    case invalidLinearKeyHeadDimension(Int)
    case invalidLinearValueHeadDimension(Int)
    case invalidLinearConvKernelSize(Int)
    case invalidContextLength(Int)
    case invalidPartialRotaryFactor(Float)
    case hiddenSizeNotDivisibleByAttentionHeads(hiddenSize: Int, attentionHeadCount: Int)
    case attentionHeadsNotDivisibleByKeyValueHeads(attentionHeadCount: Int, keyValueHeadCount: Int)
    case linearValueHeadsNotDivisibleByKeyHeads(valueHeadCount: Int, keyHeadCount: Int)
    case emptyLayerPlan
    case layerIndexMismatch(expected: Int, actual: Int)
    case missingFullAttentionLayers
    case missingGDNLayers
    case invalidQuantizationGroupSize(Int)
    case unsupportedQuantizationBits(Int)
}

public struct QwenMoEMLPConfiguration: Codable, Equatable, Sendable {
    public var expertCount: Int
    public var expertsPerToken: Int
    public var intermediateSize: Int
    public var sharedExpertIntermediateSize: Int
    public var normalizeTopKProbabilities: Bool

    private enum CodingKeys: String, CodingKey {
        case expertCount
        case expertsPerToken
        case intermediateSize
        case sharedExpertIntermediateSize
        case normalizeTopKProbabilities
    }

    public init(
        expertCount: Int,
        expertsPerToken: Int,
        intermediateSize: Int,
        sharedExpertIntermediateSize: Int,
        normalizeTopKProbabilities: Bool = true
    ) {
        self.expertCount = expertCount
        self.expertsPerToken = expertsPerToken
        self.intermediateSize = intermediateSize
        self.sharedExpertIntermediateSize = sharedExpertIntermediateSize
        self.normalizeTopKProbabilities = normalizeTopKProbabilities
    }
}

public struct QwenHybridArchitecture: Codable, Equatable, Sendable {
    public var family: QwenModelFamily
    public var vocabularySize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var moeMLP: QwenMoEMLPConfiguration?
    public var attentionHeadCount: Int
    public var keyValueHeadCount: Int
    public var headDimension: Int?
    public var linearValueHeadCount: Int
    public var linearKeyHeadCount: Int
    public var linearKeyHeadDimension: Int
    public var linearValueHeadDimension: Int
    public var linearConvKernelSize: Int
    public var contextLength: Int
    public var rmsNormEpsilon: Float
    public var ropeTheta: Float
    public var partialRotaryFactor: Float
    public var quantization: QwenQuantizationProfile?
    public var layerPlan: [QwenHybridLayerPlan]

    private enum CodingKeys: String, CodingKey {
        case family
        case vocabularySize
        case hiddenSize
        case intermediateSize
        case moeMLP
        case attentionHeadCount
        case keyValueHeadCount
        case headDimension
        case linearValueHeadCount
        case linearKeyHeadCount
        case linearKeyHeadDimension
        case linearValueHeadDimension
        case linearConvKernelSize
        case contextLength
        case rmsNormEpsilon
        case ropeTheta
        case partialRotaryFactor
        case quantization
        case layerPlan
    }

    public init(
        family: QwenModelFamily,
        vocabularySize: Int,
        hiddenSize: Int,
        intermediateSize: Int,
        moeMLP: QwenMoEMLPConfiguration? = nil,
        attentionHeadCount: Int,
        keyValueHeadCount: Int,
        headDimension: Int? = nil,
        linearValueHeadCount: Int? = nil,
        linearKeyHeadCount: Int? = nil,
        linearKeyHeadDimension: Int? = nil,
        linearValueHeadDimension: Int? = nil,
        linearConvKernelSize: Int = 4,
        contextLength: Int,
        rmsNormEpsilon: Float,
        ropeTheta: Float,
        partialRotaryFactor: Float = 0.25,
        quantization: QwenQuantizationProfile? = nil,
        layerKinds: [QwenHybridLayerKind],
        allowFullAttentionOnly: Bool = false
    ) throws {
        self.family = family
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeMLP = moeMLP
        self.attentionHeadCount = attentionHeadCount
        self.keyValueHeadCount = keyValueHeadCount
        self.headDimension = headDimension
        let inferredHeadDimension = headDimension ?? (attentionHeadCount > 0 ? hiddenSize / attentionHeadCount : 0)
        self.linearValueHeadCount = linearValueHeadCount ?? keyValueHeadCount
        self.linearKeyHeadCount = linearKeyHeadCount ?? keyValueHeadCount
        self.linearKeyHeadDimension = linearKeyHeadDimension ?? inferredHeadDimension
        self.linearValueHeadDimension = linearValueHeadDimension ?? inferredHeadDimension
        self.linearConvKernelSize = linearConvKernelSize
        self.contextLength = contextLength
        self.rmsNormEpsilon = rmsNormEpsilon
        self.ropeTheta = ropeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.quantization = quantization
        self.layerPlan = layerKinds.enumerated().map { index, kind in
            QwenHybridLayerPlan(index: index, kind: kind)
        }
        try validate(allowFullAttentionOnly: allowFullAttentionOnly)
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        family = try container.decode(QwenModelFamily.self, forKey: .family)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        moeMLP = try container.decodeIfPresent(QwenMoEMLPConfiguration.self, forKey: .moeMLP)
        attentionHeadCount = try container.decode(Int.self, forKey: .attentionHeadCount)
        keyValueHeadCount = try container.decode(Int.self, forKey: .keyValueHeadCount)
        headDimension = try container.decodeIfPresent(Int.self, forKey: .headDimension)
        let inferredHeadDimension = headDimension ?? (attentionHeadCount > 0 ? hiddenSize / attentionHeadCount : 0)
        linearValueHeadCount = try container.decodeIfPresent(Int.self, forKey: .linearValueHeadCount)
            ?? keyValueHeadCount
        linearKeyHeadCount = try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadCount)
            ?? keyValueHeadCount
        linearKeyHeadDimension = try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDimension)
            ?? inferredHeadDimension
        linearValueHeadDimension = try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDimension)
            ?? inferredHeadDimension
        linearConvKernelSize = try container.decodeIfPresent(Int.self, forKey: .linearConvKernelSize)
            ?? 4
        contextLength = try container.decode(Int.self, forKey: .contextLength)
        rmsNormEpsilon = try container.decode(Float.self, forKey: .rmsNormEpsilon)
        ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        partialRotaryFactor = try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor)
            ?? 0.25
        quantization = try container.decodeIfPresent(QwenQuantizationProfile.self, forKey: .quantization)
        layerPlan = try container.decode([QwenHybridLayerPlan].self, forKey: .layerPlan)
        try validate()
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(family, forKey: .family)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encodeIfPresent(moeMLP, forKey: .moeMLP)
        try container.encode(attentionHeadCount, forKey: .attentionHeadCount)
        try container.encode(keyValueHeadCount, forKey: .keyValueHeadCount)
        try container.encodeIfPresent(headDimension, forKey: .headDimension)
        try container.encode(linearValueHeadCount, forKey: .linearValueHeadCount)
        try container.encode(linearKeyHeadCount, forKey: .linearKeyHeadCount)
        try container.encode(linearKeyHeadDimension, forKey: .linearKeyHeadDimension)
        try container.encode(linearValueHeadDimension, forKey: .linearValueHeadDimension)
        try container.encode(linearConvKernelSize, forKey: .linearConvKernelSize)
        try container.encode(contextLength, forKey: .contextLength)
        try container.encode(rmsNormEpsilon, forKey: .rmsNormEpsilon)
        try container.encode(ropeTheta, forKey: .ropeTheta)
        try container.encode(partialRotaryFactor, forKey: .partialRotaryFactor)
        try container.encodeIfPresent(quantization, forKey: .quantization)
        try container.encode(layerPlan, forKey: .layerPlan)
    }

    public var layerCount: Int {
        layerPlan.count
    }

    public var fullAttentionLayerIndices: [Int] {
        layerPlan.compactMap { $0.kind == .fullAttention ? $0.index : nil }
    }

    public var gdnLayerIndices: [Int] {
        layerPlan.compactMap { $0.kind == .gdn ? $0.index : nil }
    }

    public var usesHybridAttentionAndGDN: Bool {
        !fullAttentionLayerIndices.isEmpty && !gdnLayerIndices.isEmpty
    }

    public var usesSparseMoEMLP: Bool {
        moeMLP != nil
    }

    public var attentionHeadDimension: Int {
        headDimension ?? hiddenSize / attentionHeadCount
    }

    public var attentionHiddenSize: Int {
        attentionHeadCount * attentionHeadDimension
    }

    public var queryHiddenSize: Int {
        attentionHiddenSize
    }

    public var queryGateHiddenSize: Int {
        attentionHiddenSize
    }

    public var queryProjectionHiddenSize: Int {
        queryHiddenSize + queryGateHiddenSize
    }

    public var keyValueHiddenSize: Int {
        keyValueHeadCount * attentionHeadDimension
    }

    public var linearKeyHiddenSize: Int {
        linearKeyHeadCount * linearKeyHeadDimension
    }

    public var linearValueHiddenSize: Int {
        linearValueHeadCount * linearValueHeadDimension
    }

    public var linearQKVHiddenSize: Int {
        linearKeyHiddenSize * 2 + linearValueHiddenSize
    }

    public var linearConvHiddenSize: Int {
        linearQKVHiddenSize
    }

    public var rotaryDimension: Int {
        max(1, Int(Float(attentionHeadDimension) * partialRotaryFactor))
    }

    public func validate(allowFullAttentionOnly: Bool = false) throws {
        guard vocabularySize > 0 else {
            throw QwenArchitectureError.invalidVocabularySize(vocabularySize)
        }
        guard hiddenSize > 0 else {
            throw QwenArchitectureError.invalidHiddenSize(hiddenSize)
        }
        guard intermediateSize > 0 else {
            throw QwenArchitectureError.invalidIntermediateSize(intermediateSize)
        }
        if let moeMLP {
            guard moeMLP.expertCount > 0 else {
                throw QwenArchitectureError.invalidMoEExpertCount(moeMLP.expertCount)
            }
            guard moeMLP.expertsPerToken > 0,
                  moeMLP.expertsPerToken <= moeMLP.expertCount
            else {
                throw QwenArchitectureError.invalidMoEExpertsPerToken(moeMLP.expertsPerToken)
            }
            guard moeMLP.intermediateSize > 0 else {
                throw QwenArchitectureError.invalidMoEIntermediateSize(moeMLP.intermediateSize)
            }
            guard moeMLP.sharedExpertIntermediateSize > 0 else {
                throw QwenArchitectureError.invalidMoESharedExpertIntermediateSize(
                    moeMLP.sharedExpertIntermediateSize
                )
            }
        }
        guard attentionHeadCount > 0 else {
            throw QwenArchitectureError.invalidAttentionHeadCount(attentionHeadCount)
        }
        guard keyValueHeadCount > 0 else {
            throw QwenArchitectureError.invalidKeyValueHeadCount(keyValueHeadCount)
        }
        if let headDimension {
            guard headDimension > 0 else {
                throw QwenArchitectureError.invalidAttentionHeadDimension(headDimension)
            }
        } else {
            guard hiddenSize % attentionHeadCount == 0 else {
                throw QwenArchitectureError.hiddenSizeNotDivisibleByAttentionHeads(
                    hiddenSize: hiddenSize,
                    attentionHeadCount: attentionHeadCount
                )
            }
        }
        guard attentionHeadCount % keyValueHeadCount == 0 else {
            throw QwenArchitectureError.attentionHeadsNotDivisibleByKeyValueHeads(
                attentionHeadCount: attentionHeadCount,
                keyValueHeadCount: keyValueHeadCount
            )
        }
        guard linearValueHeadCount > 0 else {
            throw QwenArchitectureError.invalidLinearValueHeadCount(linearValueHeadCount)
        }
        guard linearKeyHeadCount > 0 else {
            throw QwenArchitectureError.invalidLinearKeyHeadCount(linearKeyHeadCount)
        }
        guard linearKeyHeadDimension > 0 else {
            throw QwenArchitectureError.invalidLinearKeyHeadDimension(linearKeyHeadDimension)
        }
        guard linearValueHeadDimension > 0 else {
            throw QwenArchitectureError.invalidLinearValueHeadDimension(linearValueHeadDimension)
        }
        guard linearConvKernelSize > 0 else {
            throw QwenArchitectureError.invalidLinearConvKernelSize(linearConvKernelSize)
        }
        guard linearValueHeadCount % linearKeyHeadCount == 0 else {
            throw QwenArchitectureError.linearValueHeadsNotDivisibleByKeyHeads(
                valueHeadCount: linearValueHeadCount,
                keyHeadCount: linearKeyHeadCount
            )
        }
        guard contextLength > 0 else {
            throw QwenArchitectureError.invalidContextLength(contextLength)
        }
        guard partialRotaryFactor > 0, partialRotaryFactor <= 1 else {
            throw QwenArchitectureError.invalidPartialRotaryFactor(partialRotaryFactor)
        }
        if let quantization {
            guard quantization.groupSize > 0 else {
                throw QwenArchitectureError.invalidQuantizationGroupSize(quantization.groupSize)
            }
            guard [2, 3, 4, 5, 6, 8].contains(quantization.bits) else {
                throw QwenArchitectureError.unsupportedQuantizationBits(quantization.bits)
            }
        }
        guard !layerPlan.isEmpty else {
            throw QwenArchitectureError.emptyLayerPlan
        }
        for (expected, layer) in layerPlan.enumerated() where layer.index != expected {
            throw QwenArchitectureError.layerIndexMismatch(expected: expected, actual: layer.index)
        }
        guard !fullAttentionLayerIndices.isEmpty else {
            throw QwenArchitectureError.missingFullAttentionLayers
        }
        guard allowFullAttentionOnly || !gdnLayerIndices.isEmpty else {
            throw QwenArchitectureError.missingGDNLayers
        }
    }
}
