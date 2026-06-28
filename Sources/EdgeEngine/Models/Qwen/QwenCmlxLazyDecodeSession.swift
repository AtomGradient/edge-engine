// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum QwenCmlxLazyDecodeSessionError: Error, Equatable {
    case unsupportedFloatDecoderLayer(layerIndex: Int)
    case unsupportedFloatOutputHead
    case missingQuantizationProfile
}

public enum QwenCmlxLazyDecodeTensorID {
    public static let embedding = 1
    public static let finalNorm = 2
    public static let lmHead = 3

    public static func layerInputNorm(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 1
    }

    public static func layerPostAttentionNorm(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 2
    }

    public static func layerAttentionQuery(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 10
    }

    public static func layerAttentionKey(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 11
    }

    public static func layerAttentionValue(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 12
    }

    public static func layerAttentionQueryNorm(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 13
    }

    public static func layerAttentionKeyNorm(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 14
    }

    public static func layerAttentionOutput(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 15
    }

    public static func layerGDNQKV(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 20
    }

    public static func layerGDNZ(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 21
    }

    public static func layerGDNA(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 22
    }

    public static func layerGDNB(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 23
    }

    public static func layerGDNConv1D(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 24
    }

    public static func layerGDNAlog(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 25
    }

    public static func layerGDNDTBias(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 26
    }

    public static func layerGDNNorm(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 27
    }

    public static func layerGDNOutput(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 28
    }

    public static func layerMLPGate(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 40
    }

    public static func layerMLPUp(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 41
    }

    public static func layerMLPDown(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 42
    }

    public static func layerMoEGate(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 50
    }

    public static func layerMoESwitchGate(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 51
    }

    public static func layerMoESwitchUp(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 52
    }

    public static func layerMoESwitchDown(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 53
    }

    public static func layerMoESharedGate(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 54
    }

    public static func layerMoESharedUp(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 55
    }

    public static func layerMoESharedDown(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 56
    }

    public static func layerMoESharedExpertGate(_ layerIndex: Int) -> Int {
        1_000 + layerIndex * 100 + 57
    }
}

/// Registers a Qwen hybrid model's tensors into a persistent Cmlx session.
///
/// This is intentionally separate from the public generation path until the
/// Cmlx decode-step implementation reaches parity with the existing Swift path.
public final class QwenCmlxLazyDecodeSession {
    public let session: EdgeMLXQwen35Session
    public private(set) var tokenPosition: Int

    public init(
        model: QwenHybridModelReference,
        runtime: EdgeMetalRuntime,
        executor: MetalKernelExecutor? = nil,
        attentionCacheLimit: Int? = nil,
        dsrPolicies: [Int: QwenDSRKVCachePolicy] = [:],
        attentionCacheQuantizationGroupSize: Int? = nil,
        attentionCacheQuantizationBits: Int? = nil,
        frogJumpLayerMask: UInt64 = 0
    ) throws {
        let session = try EdgeMLXQwen35Session(
            architecture: model.architecture,
            runtime: runtime,
            weightBufferOwner: executor
        )
        self.session = session
        self.tokenPosition = 0
        try session.setDSRPolicies(dsrPolicies)
        try session.setAttentionCacheQuantization(
            groupSize: attentionCacheQuantizationGroupSize,
            bits: attentionCacheQuantizationBits
        )
        try session.setFrogJumpLayerMask(frogJumpLayerMask)
        try session.setAttentionCacheLimit(maxTokens: attentionCacheLimit)
        try Self.register(model: model, into: session, executor: executor)
    }

    public init(
        bundleIndex: QwenModelBundleIndex,
        runtime: EdgeMetalRuntime,
        attentionCacheLimit: Int? = nil,
        dsrPolicies: [Int: QwenDSRKVCachePolicy] = [:],
        attentionCacheQuantizationGroupSize: Int? = nil,
        attentionCacheQuantizationBits: Int? = nil,
        frogJumpLayerMask: UInt64 = 0
    ) throws {
        guard let quantization = bundleIndex.architecture.quantization else {
            throw QwenCmlxLazyDecodeSessionError.missingQuantizationProfile
        }
        let session = try EdgeMLXQwen35Session(
            architecture: bundleIndex.architecture,
            runtime: runtime
        )
        self.session = session
        self.tokenPosition = 0
        try session.setDSRPolicies(dsrPolicies)
        try session.setAttentionCacheQuantization(
            groupSize: attentionCacheQuantizationGroupSize,
            bits: attentionCacheQuantizationBits
        )
        try session.setFrogJumpLayerMask(frogJumpLayerMask)
        try session.setAttentionCacheLimit(maxTokens: attentionCacheLimit)
        let shardURLs = Set(bundleIndex.weightMap.values)
            .sorted()
            .map { bundleIndex.rootURL.appendingPathComponent($0) }
        try session.loadSafetensors(
            shardURLs: shardURLs,
            modelPrefix: bundleIndex.modelPrefix,
            groupSize: quantization.groupSize,
            bits: quantization.bits
        )
    }

    public var registeredFloatTensorCount: Int {
        session.registeredFloatTensorCount
    }

    public var registeredQuantizedTensorCount: Int {
        session.registeredQuantizedTensorCount
    }

    public var decodedTokenCount: Int {
        session.decodedTokenCount
    }

    public func hasDecoderWeights() throws -> Bool {
        try session.hasDecoderWeights()
    }

    public static func configureCommandBufferLimits(
        maxOps: Int,
        maxMB: Int
    ) throws {
        try EdgeMLXQwen35Session.configureCommandBufferLimits(
            maxOps: maxOps,
            maxMB: maxMB
        )
    }

    public static func configureMemoryLimit(bytes: Int) throws {
        try EdgeMLXQwen35Session.configureMemoryLimit(bytes: bytes)
    }

    public static func currentMemoryLimitBytes() throws -> Int {
        try EdgeMLXQwen35Session.currentMemoryLimitBytes()
    }

    public func reset() throws {
        try session.resetDecodeCache()
        tokenPosition = 0
    }

    public func unloadDecoderWeightsPreservingState() throws {
        try session.unloadDecoderWeightsPreservingState()
    }

    public func reloadDecoderWeights(bundleIndex: QwenModelBundleIndex) throws {
        guard let quantization = bundleIndex.architecture.quantization else {
            throw QwenCmlxLazyDecodeSessionError.missingQuantizationProfile
        }
        let shardURLs = Set(bundleIndex.weightMap.values)
            .sorted()
            .map { bundleIndex.rootURL.appendingPathComponent($0) }
        try session.loadSafetensors(
            shardURLs: shardURLs,
            modelPrefix: bundleIndex.modelPrefix,
            groupSize: quantization.groupSize,
            bits: quantization.bits
        )
    }

    public func reloadDecoderWeights(
        model: QwenHybridModelReference,
        executor: MetalKernelExecutor? = nil
    ) throws {
        try Self.register(model: model, into: session, executor: executor)
    }

    public func restoreNeuralImprintCache(
        artifactURL: URL,
        prefixTokenCount: Int
    ) throws {
        try session.restoreNeuralImprintCache(
            artifactURL: artifactURL,
            prefixTokenCount: prefixTokenCount
        )
        tokenPosition = prefixTokenCount
    }

    public func setAttentionCacheLimit(_ maxTokens: Int?) throws {
        try session.setAttentionCacheLimit(maxTokens: maxTokens)
    }

    public func configureVision(index: QwenVLMModelBundleIndex) throws {
        try session.setVisionConfig(plan: index.preflightResult.plan)
        try session.loadVisionSafetensors(index: index)
    }

    public func visionEncode(
        pixelValues: [Float],
        pixelValuesShape: [Int],
        gridTHW: [QwenImageGridTHW],
        plan: QwenVLMRuntimePlan
    ) throws -> EdgeMLXQwen35VisionEncoding {
        try session.visionEncode(
            pixelValues: pixelValues,
            pixelValuesShape: pixelValuesShape,
            gridTHW: gridTHW,
            spatialMergeSize: plan.visionConfiguration.spatialMergeSize
                ?? plan.imageProcessorConfiguration.mergeSize
                ?? 2,
            outputHiddenSize: plan.languageArchitecture.hiddenSize
        )
    }

    public func updateDSRPoliciesInPlace(_ policies: [Int: QwenDSRKVCachePolicy]) throws {
        for (layerIndex, policy) in policies.sorted(by: { $0.key < $1.key }) {
            try session.updateDSRPolicyFields(policy, layerIndex: layerIndex)
        }
    }

    @discardableResult
    public func decodeStep(tokenID: Int) throws -> Int {
        let nextTokenID = try session.decodeStep(tokenID: tokenID)
        tokenPosition += 1
        return nextTokenID
    }

    @discardableResult
    public func prefill(tokenIDs: [Int]) throws -> Int {
        guard !tokenIDs.isEmpty else {
            throw QwenHybridModelReferenceError.emptyTokenIds
        }
        let nextTokenID = try session.prefill(tokenIDs: tokenIDs)
        tokenPosition += tokenIDs.count
        return nextTokenID
    }

    public func captureLastHidden(
        tokenIDs: [Int],
        targetLayer: Int
    ) throws -> [Float] {
        guard !tokenIDs.isEmpty else {
            throw QwenHybridModelReferenceError.emptyTokenIds
        }
        return try session.captureLastHidden(
            tokenIDs: tokenIDs,
            targetLayer: targetLayer
        )
    }

    public func prefillAsync(tokenIDs: [Int]) throws {
        guard !tokenIDs.isEmpty else {
            throw QwenHybridModelReferenceError.emptyTokenIds
        }
        try session.prefillAsync(tokenIDs: tokenIDs)
        tokenPosition += tokenIDs.count
    }

    public func prefillSampledAsync(
        tokenIDs: [Int],
        temperature: Float,
        topK: Int? = nil,
        topP: Float,
        minP: Float = 0,
        seed: UInt64
    ) throws {
        guard !tokenIDs.isEmpty else {
            throw QwenHybridModelReferenceError.emptyTokenIds
        }
        try session.prefillSampledAsync(
            tokenIDs: tokenIDs,
            temperature: temperature,
            topK: topK,
            topP: topP,
            minP: minP,
            seed: seed
        )
        tokenPosition += tokenIDs.count
    }

    public func synchronize() throws {
        try session.synchronize()
    }

    @discardableResult
    public func prefillMediaFeatures(
        tokenIDs: [Int],
        mediaFeatures: [Float],
        mediaFeatureShape: [Int],
        mediaTokenID: Int
    ) throws -> Int {
        let nextTokenID = try session.prefillMediaFeatures(
            tokenIDs: tokenIDs,
            mediaFeatures: mediaFeatures,
            mediaFeatureShape: mediaFeatureShape,
            mediaTokenID: mediaTokenID
        )
        tokenPosition += tokenIDs.count
        return nextTokenID
    }

    @discardableResult
    public func prefillImageFeatures(
        tokenIDs: [Int],
        imageFeatures: [Float],
        imageFeatureShape: [Int],
        imageTokenID: Int
    ) throws -> Int {
        try prefillMediaFeatures(
            tokenIDs: tokenIDs,
            mediaFeatures: imageFeatures,
            mediaFeatureShape: imageFeatureShape,
            mediaTokenID: imageTokenID
        )
    }

    @discardableResult
    public func nextToken() throws -> Int {
        let tokenID = try session.nextToken()
        tokenPosition += 1
        return tokenID
    }

    @discardableResult
    public func nextSampledToken(
        temperature: Float,
        topK: Int? = nil,
        topP: Float,
        minP: Float = 0,
        seed: UInt64
    ) throws -> Int {
        let tokenID = try session.nextSampledToken(
            temperature: temperature,
            topK: topK,
            topP: topP,
            minP: minP,
            seed: seed
        )
        tokenPosition += 1
        return tokenID
    }

    public func setRepetitionPenalty(
        _ penalty: Float,
        contextTokenIds: [Int]
    ) throws {
        try session.setRepetitionPenalty(penalty, contextTokenIds: contextTokenIds)
    }

    public func setSamplingPenalties(
        repetitionPenalty: Float,
        repetitionContextTokenIds: [Int],
        presencePenalty: Float,
        presenceContextTokenIds: [Int],
        frequencyPenalty: Float,
        frequencyContextTokenIds: [Int]
    ) throws {
        try session.setSamplingPenalties(
            repetitionPenalty: repetitionPenalty,
            repetitionContextTokenIds: repetitionContextTokenIds,
            presencePenalty: presencePenalty,
            presenceContextTokenIds: presenceContextTokenIds,
            frequencyPenalty: frequencyPenalty,
            frequencyContextTokenIds: frequencyContextTokenIds
        )
    }

    public func setEOSSamplingBias(
        tokenIds: [Int],
        suppress: Bool,
        logitPenalty: Float
    ) throws {
        try session.setEOSSamplingBias(
            tokenIds: tokenIds,
            suppress: suppress,
            logitPenalty: logitPenalty
        )
    }

    public func clearEOSSamplingBias() throws {
        try session.clearEOSSamplingBias()
    }

    public func clearRepetitionPenalty() throws {
        try session.clearRepetitionPenalty()
    }

    public func lastSampleDiagnostics() throws -> String? {
        try session.lastSampleDiagnostics()
    }

    public func memorySummary() throws -> String? {
        try session.memorySummary()
    }

    public func resetEvalProfile() throws {
        try session.resetEvalProfile()
    }

    public func evalProfileSummary() throws -> String? {
        try session.evalProfileSummary()
    }

    public func generateNextTokens(
        promptTokenIds: [Int],
        maxTokenCount: Int,
        endTokenIds: Set<Int> = []
    ) throws -> [Int] {
        guard !promptTokenIds.isEmpty else {
            throw QwenHybridModelReferenceError.emptyTokenIds
        }
        guard maxTokenCount >= 0 else {
            throw QwenGreedyDecoderError.invalidMaxTokenCount(maxTokenCount)
        }
        try reset()

        var nextTokenID: Int? = try prefill(tokenIDs: promptTokenIds)

        var generatedTokenIds: [Int] = []
        generatedTokenIds.reserveCapacity(maxTokenCount)
        while generatedTokenIds.count < maxTokenCount, let tokenID = nextTokenID {
            generatedTokenIds.append(tokenID)
            if endTokenIds.contains(tokenID) {
                break
            }
            nextTokenID = try decodeStep(tokenID: tokenID)
        }
        return generatedTokenIds
    }

    private static func register(
        model: QwenHybridModelReference,
        into session: EdgeMLXQwen35Session,
        executor: MetalKernelExecutor?
    ) throws {
        switch model.embeddings.embeddings {
        case .float(let embeddings):
            try session.registerFloatTensor(
                id: QwenCmlxLazyDecodeTensorID.embedding,
                tensor: embeddings
            )
        case .quantized(let embeddings):
            try registerQuantizedTensor(
                id: QwenCmlxLazyDecodeTensorID.embedding,
                tensor: embeddings,
                into: session,
                executor: executor
            )
        }

        switch model.outputWeights {
        case .float(let output):
            try session.registerFloatTensor(
                id: QwenCmlxLazyDecodeTensorID.finalNorm,
                tensor: output.finalNorm
            )
            try session.registerFloatTensor(
                id: QwenCmlxLazyDecodeTensorID.lmHead,
                tensor: output.lmHead
            )
        case .quantized(let output):
            try session.registerFloatTensor(
                id: QwenCmlxLazyDecodeTensorID.finalNorm,
                tensor: output.finalNorm
            )
            try registerQuantizedTensor(
                id: QwenCmlxLazyDecodeTensorID.lmHead,
                tensor: output.lmHead,
                into: session,
                executor: executor
            )
        }

        for layer in model.decoderLayers {
            switch layer {
            case .fullAttention(let layer):
                throw QwenCmlxLazyDecodeSessionError.unsupportedFloatDecoderLayer(
                    layerIndex: layer.attention.layerIndex
                )
            case .gdn(let layer):
                throw QwenCmlxLazyDecodeSessionError.unsupportedFloatDecoderLayer(
                    layerIndex: layer.linearAttention.layerIndex
                )
            case .quantizedFullAttention(let layer):
                try registerFullAttentionLayer(layer, into: session, executor: executor)
            case .quantizedGDN(let layer):
                try registerGDNLayer(layer, into: session, executor: executor)
            }
        }
    }

    private static func registerQuantizedTensor(
        id: Int,
        tensor: EdgeQuantizedTensor,
        into session: EdgeMLXQwen35Session,
        executor: MetalKernelExecutor?
    ) throws {
        if let executor {
            try session.registerQuantizedTensor(id: id, tensor: tensor, executor: executor)
        } else {
            try session.registerQuantizedTensor(id: id, tensor: tensor)
        }
    }

    private static func registerCommonLayerTensors(
        layerIndex: Int,
        inputLayerNorm: EdgeTensor,
        postAttentionLayerNorm: EdgeTensor,
        mlp: QwenQuantizedMLPWeights,
        into session: EdgeMLXQwen35Session,
        executor: MetalKernelExecutor?
    ) throws {
        try session.registerFloatTensor(
            id: QwenCmlxLazyDecodeTensorID.layerInputNorm(layerIndex),
            tensor: inputLayerNorm
        )
        try session.registerFloatTensor(
            id: QwenCmlxLazyDecodeTensorID.layerPostAttentionNorm(layerIndex),
            tensor: postAttentionLayerNorm
        )
        try registerQuantizedTensor(
            id: QwenCmlxLazyDecodeTensorID.layerMLPGate(layerIndex),
            tensor: mlp.gate,
            into: session,
            executor: executor
        )
        try registerQuantizedTensor(
            id: QwenCmlxLazyDecodeTensorID.layerMLPUp(layerIndex),
            tensor: mlp.up,
            into: session,
            executor: executor
        )
        try registerQuantizedTensor(
            id: QwenCmlxLazyDecodeTensorID.layerMLPDown(layerIndex),
            tensor: mlp.down,
            into: session,
            executor: executor
        )
    }

    private static func registerFullAttentionLayer(
        _ layer: QwenQuantizedFullAttentionDecoderLayerReference,
        into session: EdgeMLXQwen35Session,
        executor: MetalKernelExecutor?
    ) throws {
        let layerIndex = layer.attention.layerIndex
        try registerCommonLayerTensors(
            layerIndex: layerIndex,
            inputLayerNorm: layer.inputLayerNorm,
            postAttentionLayerNorm: layer.postAttentionLayerNorm,
            mlp: layer.mlp,
            into: session,
            executor: executor
        )
        try registerQuantizedTensor(
            id: QwenCmlxLazyDecodeTensorID.layerAttentionQuery(layerIndex),
            tensor: layer.attention.projectionWeights.query,
            into: session,
            executor: executor
        )
        try registerQuantizedTensor(
            id: QwenCmlxLazyDecodeTensorID.layerAttentionKey(layerIndex),
            tensor: layer.attention.projectionWeights.key,
            into: session,
            executor: executor
        )
        try registerQuantizedTensor(
            id: QwenCmlxLazyDecodeTensorID.layerAttentionValue(layerIndex),
            tensor: layer.attention.projectionWeights.value,
            into: session,
            executor: executor
        )
        if let normalizationWeights = layer.attention.normalizationWeights {
            try session.registerFloatTensor(
                id: QwenCmlxLazyDecodeTensorID.layerAttentionQueryNorm(layerIndex),
                tensor: normalizationWeights.query
            )
            try session.registerFloatTensor(
                id: QwenCmlxLazyDecodeTensorID.layerAttentionKeyNorm(layerIndex),
                tensor: normalizationWeights.key
            )
        }
        if let outputProjectionWeights = layer.attention.outputProjectionWeights {
            try registerQuantizedTensor(
                id: QwenCmlxLazyDecodeTensorID.layerAttentionOutput(layerIndex),
                tensor: outputProjectionWeights.weight,
                into: session,
                executor: executor
            )
        }
    }

    private static func registerGDNLayer(
        _ layer: QwenQuantizedGDNDecoderLayerReference,
        into session: EdgeMLXQwen35Session,
        executor: MetalKernelExecutor?
    ) throws {
        let layerIndex = layer.linearAttention.layerIndex
        try registerCommonLayerTensors(
            layerIndex: layerIndex,
            inputLayerNorm: layer.inputLayerNorm,
            postAttentionLayerNorm: layer.postAttentionLayerNorm,
            mlp: layer.mlp,
            into: session,
            executor: executor
        )
        try registerQuantizedTensor(
            id: QwenCmlxLazyDecodeTensorID.layerGDNQKV(layerIndex),
            tensor: layer.linearAttention.inProjQKV,
            into: session,
            executor: executor
        )
        try registerQuantizedTensor(
            id: QwenCmlxLazyDecodeTensorID.layerGDNZ(layerIndex),
            tensor: layer.linearAttention.inProjZ,
            into: session,
            executor: executor
        )
        try registerQuantizedTensor(
            id: QwenCmlxLazyDecodeTensorID.layerGDNA(layerIndex),
            tensor: layer.linearAttention.inProjA,
            into: session,
            executor: executor
        )
        try registerQuantizedTensor(
            id: QwenCmlxLazyDecodeTensorID.layerGDNB(layerIndex),
            tensor: layer.linearAttention.inProjB,
            into: session,
            executor: executor
        )
        try session.registerFloatTensor(
            id: QwenCmlxLazyDecodeTensorID.layerGDNConv1D(layerIndex),
            tensor: layer.linearAttention.conv1D
        )
        try session.registerFloatTensor(
            id: QwenCmlxLazyDecodeTensorID.layerGDNAlog(layerIndex),
            tensor: layer.linearAttention.aLog
        )
        try session.registerFloatTensor(
            id: QwenCmlxLazyDecodeTensorID.layerGDNDTBias(layerIndex),
            tensor: layer.linearAttention.dtBias
        )
        try session.registerFloatTensor(
            id: QwenCmlxLazyDecodeTensorID.layerGDNNorm(layerIndex),
            tensor: layer.linearAttention.norm
        )
        try registerQuantizedTensor(
            id: QwenCmlxLazyDecodeTensorID.layerGDNOutput(layerIndex),
            tensor: layer.linearAttention.outProj,
            into: session,
            executor: executor
        )
    }
}
