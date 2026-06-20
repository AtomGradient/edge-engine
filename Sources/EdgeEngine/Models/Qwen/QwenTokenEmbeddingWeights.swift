// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public enum QwenTokenEmbeddingReference {
    case float(EdgeTensor)
    case quantized(EdgeQuantizedTensor)

    public var shape: EdgeTensorShape {
        switch self {
        case .float(let embeddings):
            embeddings.shape
        case .quantized(let embeddings):
            EdgeTensorShape(embeddings.shape)
        }
    }

    public func readFloat32() throws -> [Float] {
        switch self {
        case .float(let embeddings):
            try embeddings.readFloat32()
        case .quantized(let embeddings):
            embeddings.dequantizedValues()
        }
    }
}

public struct QwenTokenEmbeddingWeights {
    public var embeddings: QwenTokenEmbeddingReference

    public init(embeddings: EdgeTensor) {
        self.embeddings = .float(embeddings)
    }

    public init(embeddings: EdgeQuantizedTensor) {
        self.embeddings = .quantized(embeddings)
    }

    public init(embeddings: QwenTokenEmbeddingReference) {
        self.embeddings = embeddings
    }

    public static func loadHuggingFaceLayout(
        weightStore: QwenModelWeightStore,
        runtime: EdgeMetalRuntime
    ) throws -> QwenTokenEmbeddingWeights {
        let architecture = weightStore.bundleIndex.architecture
        let manifest = weightStore.bundleIndex.modelLevelManifest
        let name = manifest.embedTokensName
        let embeddings: QwenTokenEmbeddingReference
        if architecture.quantization != nil,
           let group = manifest.quantizedWeightGroups.first(where: { $0.weightName == name }),
           let scalesName = group.scalesName
        {
            let quantization = try resolveQuantization(architecture: architecture)
            let quantized = try weightStore.loadQuantizedTensor(
                weightName: name,
                scalesName: scalesName,
                biasesName: group.biasesName,
                groupSize: quantization.groupSize,
                bits: quantization.bits
            )
            try validate(quantized, name: name, architecture: architecture)
            embeddings = .quantized(quantized)
        } else {
            embeddings = .float(try weightStore.loadFloat32Tensor(named: name, runtime: runtime))
        }
        try validate(embeddings, name: name, architecture: architecture)
        return QwenTokenEmbeddingWeights(embeddings: embeddings)
    }

    public func hiddenStates(
        tokenIds: [Int],
        executor: MetalKernelExecutor
    ) throws -> EdgeTensor {
        switch embeddings {
        case .float(let embeddings):
            try executor.embeddingLookup(tokenIds: tokenIds, embeddings: embeddings)
        case .quantized(let embeddings):
            try executor.affineQuantizedEmbeddingLookup(tokenIds: tokenIds, embeddings: embeddings)
        }
    }

    private static func validate(
        _ embeddings: QwenTokenEmbeddingReference,
        name: String,
        architecture: QwenHybridArchitecture
    ) throws {
        switch embeddings {
        case .float(let embeddings):
            try validate(embeddings, name: name, architecture: architecture)
        case .quantized(let embeddings):
            try validate(embeddings, name: name, architecture: architecture)
        }
    }

    private static func validate(
        _ embeddings: EdgeTensor,
        name: String,
        architecture: QwenHybridArchitecture
    ) throws {
        guard embeddings.shape.dimensions == [architecture.vocabularySize, architecture.hiddenSize] else {
            throw QwenProjectionWeightError.invalidWeightShape(
                name: name,
                expected: [architecture.vocabularySize, architecture.hiddenSize],
                actual: embeddings.shape.dimensions
            )
        }
    }

    private static func validate(
        _ embeddings: EdgeQuantizedTensor,
        name: String,
        architecture: QwenHybridArchitecture
    ) throws {
        guard embeddings.shape == [architecture.vocabularySize, architecture.hiddenSize] else {
            throw QwenProjectionWeightError.invalidWeightShape(
                name: name,
                expected: [architecture.vocabularySize, architecture.hiddenSize],
                actual: embeddings.shape
            )
        }
    }

    private static func resolveQuantization(
        architecture: QwenHybridArchitecture
    ) throws -> QwenQuantizationProfile {
        guard let profile = architecture.quantization else {
            throw QwenProjectionWeightError.missingQuantizationProfile
        }
        return profile
    }
}
