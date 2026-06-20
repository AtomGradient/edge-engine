// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public struct QwenModelOutputWeights {
    public var finalNorm: EdgeTensor
    public var lmHead: EdgeTensor
    public var rmsNormEpsilon: Float
    public var usesTiedEmbeddings: Bool

    public init(
        finalNorm: EdgeTensor,
        lmHead: EdgeTensor,
        rmsNormEpsilon: Float,
        usesTiedEmbeddings: Bool = false
    ) {
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.rmsNormEpsilon = rmsNormEpsilon
        self.usesTiedEmbeddings = usesTiedEmbeddings
    }

    public static func loadHuggingFaceLayout(
        weightStore: QwenModelWeightStore,
        runtime: EdgeMetalRuntime
    ) throws -> QwenModelOutputWeights {
        let architecture = weightStore.bundleIndex.architecture
        let manifest = weightStore.bundleIndex.modelLevelManifest
        let finalNorm = try weightStore.loadFloat32Tensor(named: manifest.finalNormName, runtime: runtime)
        let lmHeadName = manifest.lmHeadName ?? manifest.embedTokensName
        let lmHead = try weightStore.loadFloat32TensorTransposed2D(named: lmHeadName, runtime: runtime)
        try validate(
            finalNorm: finalNorm,
            lmHead: lmHead,
            finalNormName: manifest.finalNormName,
            lmHeadName: lmHeadName,
            architecture: architecture
        )
        return QwenModelOutputWeights(
            finalNorm: finalNorm,
            lmHead: lmHead,
            rmsNormEpsilon: architecture.rmsNormEpsilon,
            usesTiedEmbeddings: manifest.lmHeadName == nil
        )
    }

    public func logits(hiddenStates: EdgeTensor, executor: MetalKernelExecutor) throws -> EdgeTensor {
        let normalized = try executor.rmsNorm(hiddenStates, weight: finalNorm, epsilon: rmsNormEpsilon)
        return try executor.matmul(normalized, lmHead)
    }

    public func greedyToken(hiddenStates: EdgeTensor, executor: MetalKernelExecutor) throws -> QwenGreedyToken {
        let logits = try logits(hiddenStates: hiddenStates, executor: executor)
        return try executor.argmaxLastRow(logits)
    }

    private static func validate(
        finalNorm: EdgeTensor,
        lmHead: EdgeTensor,
        finalNormName: String,
        lmHeadName: String,
        architecture: QwenHybridArchitecture
    ) throws {
        guard finalNorm.shape.dimensions == [architecture.hiddenSize] else {
            throw QwenProjectionWeightError.invalidWeightShape(
                name: finalNormName,
                expected: [architecture.hiddenSize],
                actual: finalNorm.shape.dimensions
            )
        }
        guard lmHead.shape.dimensions == [architecture.hiddenSize, architecture.vocabularySize] else {
            throw QwenProjectionWeightError.invalidWeightShape(
                name: lmHeadName,
                expected: [architecture.hiddenSize, architecture.vocabularySize],
                actual: lmHead.shape.dimensions
            )
        }
    }
}

public struct QwenQuantizedModelOutputWeights {
    public var finalNorm: EdgeTensor
    public var lmHead: EdgeQuantizedTensor
    public var rmsNormEpsilon: Float
    public var usesTiedEmbeddings: Bool

    public init(
        finalNorm: EdgeTensor,
        lmHead: EdgeQuantizedTensor,
        rmsNormEpsilon: Float,
        usesTiedEmbeddings: Bool = false
    ) {
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.rmsNormEpsilon = rmsNormEpsilon
        self.usesTiedEmbeddings = usesTiedEmbeddings
    }

    public static func loadHuggingFaceLayout(
        weightStore: QwenModelWeightStore,
        groupSize: Int? = nil,
        bits: Int? = nil,
        runtime: EdgeMetalRuntime,
        tiedLMHead: EdgeQuantizedTensor? = nil
    ) throws -> QwenQuantizedModelOutputWeights {
        let architecture = weightStore.bundleIndex.architecture
        let quantization = try resolveQuantization(
            architecture: architecture,
            groupSize: groupSize,
            bits: bits
        )
        let manifest = weightStore.bundleIndex.modelLevelManifest
        let finalNorm = try weightStore.loadFloat32Tensor(named: manifest.finalNormName, runtime: runtime)
        let lmHeadName = manifest.lmHeadName ?? manifest.embedTokensName
        let lmHead: EdgeQuantizedTensor
        if manifest.lmHeadName == nil, let tiedLMHead {
            lmHead = tiedLMHead
        } else {
            lmHead = try loadQuantizedWeight(
                lmHeadName,
                manifest: manifest,
                weightStore: weightStore,
                groupSize: quantization.groupSize,
                bits: quantization.bits
            )
        }
        try validate(
            finalNorm: finalNorm,
            lmHead: lmHead,
            finalNormName: manifest.finalNormName,
            lmHeadName: lmHeadName,
            architecture: architecture
        )
        return QwenQuantizedModelOutputWeights(
            finalNorm: finalNorm,
            lmHead: lmHead,
            rmsNormEpsilon: architecture.rmsNormEpsilon,
            usesTiedEmbeddings: manifest.lmHeadName == nil
        )
    }

    public func logits(hiddenStates: EdgeTensor, executor: MetalKernelExecutor) throws -> EdgeTensor {
        if executor.runtimeConfiguration.useCmlxLazyOutputHead {
            return try executor.rmsNormAffineQuantizedMatmul(
                hiddenStates,
                normWeight: finalNorm,
                epsilon: rmsNormEpsilon,
                weights: lmHead,
                transpose: true,
                diagnosticName: "logits_head_cmlx_lazy"
            )
        }
        let normalized = try executor.rmsNorm(hiddenStates, weight: finalNorm, epsilon: rmsNormEpsilon)
        return try executor.affineQuantizedMatmul(normalized, weights: lmHead, transpose: true)
    }

    public func greedyToken(hiddenStates: EdgeTensor, executor: MetalKernelExecutor) throws -> QwenGreedyToken {
        guard executor.runtimeConfiguration.useGreedyOutputHeadArgmax else {
            let logits = try logits(hiddenStates: hiddenStates, executor: executor)
            return try executor.argmaxLastRow(logits)
        }
        let normalized = try executor.rmsNorm(hiddenStates, weight: finalNorm, epsilon: rmsNormEpsilon)
        return try executor.affineQuantizedMatmulArgmax(normalized, weights: lmHead, transpose: true)
    }

    private static func resolveQuantization(
        architecture: QwenHybridArchitecture,
        groupSize: Int?,
        bits: Int?
    ) throws -> QwenQuantizationProfile {
        if let groupSize, let bits {
            return QwenQuantizationProfile(groupSize: groupSize, bits: bits)
        }
        guard let profile = architecture.quantization else {
            throw QwenProjectionWeightError.missingQuantizationProfile
        }
        return QwenQuantizationProfile(
            groupSize: groupSize ?? profile.groupSize,
            bits: bits ?? profile.bits
        )
    }

    private static func loadQuantizedWeight(
        _ weightName: String,
        manifest: QwenModelLevelManifest,
        weightStore: QwenModelWeightStore,
        groupSize: Int,
        bits: Int
    ) throws -> EdgeQuantizedTensor {
        guard let group = manifest.quantizedWeightGroups.first(where: { $0.weightName == weightName }),
              let scalesName = group.scalesName
        else {
            throw QwenProjectionWeightError.missingQuantizedCompanion(weightName: weightName)
        }
        return try weightStore.loadQuantizedTensor(
            weightName: weightName,
            scalesName: scalesName,
            biasesName: group.biasesName,
            groupSize: groupSize,
            bits: bits
        )
    }

    private static func validate(
        finalNorm: EdgeTensor,
        lmHead: EdgeQuantizedTensor,
        finalNormName: String,
        lmHeadName: String,
        architecture: QwenHybridArchitecture
    ) throws {
        guard finalNorm.shape.dimensions == [architecture.hiddenSize] else {
            throw QwenProjectionWeightError.invalidWeightShape(
                name: finalNormName,
                expected: [architecture.hiddenSize],
                actual: finalNorm.shape.dimensions
            )
        }
        guard lmHead.shape == [architecture.vocabularySize, architecture.hiddenSize] else {
            throw QwenProjectionWeightError.invalidWeightShape(
                name: lmHeadName,
                expected: [architecture.vocabularySize, architecture.hiddenSize],
                actual: lmHead.shape
            )
        }
    }
}
