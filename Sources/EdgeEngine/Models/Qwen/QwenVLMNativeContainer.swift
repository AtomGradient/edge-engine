// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum QwenVLMNativeContainerError: Error, Equatable {
    case decoderAlreadyLoaded
    case visionAlreadyLoaded
    case decoderNotLoaded
    case visionNotLoaded
}

public struct QwenVLMNativeUnloadReport: Equatable, Sendable {
    public var hadDecoderLoaded: Bool
    public var purgedQuantizedBufferStats: MetalQuantizedBufferCacheStats

    public init(
        hadDecoderLoaded: Bool,
        purgedQuantizedBufferStats: MetalQuantizedBufferCacheStats
    ) {
        self.hadDecoderLoaded = hadDecoderLoaded
        self.purgedQuantizedBufferStats = purgedQuantizedBufferStats
    }
}

public struct QwenVLMNativeContainerState: Equatable, Sendable {
    public var isStructureLoaded: Bool
    public var isDecoderLoaded: Bool
    public var isVisionLoaded: Bool
    public var languageTensorCount: Int
    public var visionTensorCount: Int
    public var loadedVisionWeightBytes: Int

    public init(
        isStructureLoaded: Bool,
        isDecoderLoaded: Bool,
        isVisionLoaded: Bool = false,
        languageTensorCount: Int,
        visionTensorCount: Int,
        loadedVisionWeightBytes: Int = 0
    ) {
        self.isStructureLoaded = isStructureLoaded
        self.isDecoderLoaded = isDecoderLoaded
        self.isVisionLoaded = isVisionLoaded
        self.languageTensorCount = languageTensorCount
        self.visionTensorCount = visionTensorCount
        self.loadedVisionWeightBytes = loadedVisionWeightBytes
    }
}

public struct QwenVLMNativeVisionUnloadReport: Equatable, Sendable {
    public var hadVisionLoaded: Bool
    public var releasedTensorCount: Int
    public var releasedByteCount: Int

    public init(
        hadVisionLoaded: Bool,
        releasedTensorCount: Int,
        releasedByteCount: Int
    ) {
        self.hadVisionLoaded = hadVisionLoaded
        self.releasedTensorCount = releasedTensorCount
        self.releasedByteCount = releasedByteCount
    }
}

public final class QwenVLMNativeContainer {
    public let index: QwenVLMModelBundleIndex
    public let runtime: EdgeMetalRuntime
    public let executor: MetalKernelExecutor

    private var decoderModel: QwenHybridModelReference?
    private var cmlxDecoderSession: QwenCmlxLazyDecodeSession?
    private var cmlxVisionSession: EdgeMLXQwen35Session?
    private var visionWeightStore: QwenVisionWeightStore?
    private var visionWeights: QwenVisionWeightSnapshot?

    public init(
        index: QwenVLMModelBundleIndex,
        runtime: EdgeMetalRuntime,
        executor: MetalKernelExecutor
    ) {
        self.index = index
        self.runtime = runtime
        self.executor = executor
    }

    public static func loadStructureOnly(
        from modelRootURL: URL,
        family: QwenVLMModelFamily? = nil,
        runtimeConfiguration: MetalRuntimeConfiguration =
            EdgeEngineMetalConfigurationStore.shared.currentConfiguration
    ) throws -> QwenVLMNativeContainer {
        let runtime = try EdgeMetalRuntime(configuration: runtimeConfiguration)
        let executor = try MetalKernelExecutor(runtime: runtime)
        return try QwenVLMNativeContainer(
            index: QwenVLMModelBundleIndex.load(from: modelRootURL, family: family),
            runtime: runtime,
            executor: executor
        )
    }

    public var isDecoderLoaded: Bool {
        decoderModel != nil
    }

    public var isVisionLoaded: Bool {
        visionWeights != nil
    }

    public var isCmlxDecoderLoaded: Bool {
        cmlxDecoderSession != nil
    }

    public var isCmlxVisionLoaded: Bool {
        cmlxVisionSession != nil
    }

    public var state: QwenVLMNativeContainerState {
        QwenVLMNativeContainerState(
            isStructureLoaded: true,
            isDecoderLoaded: isDecoderLoaded,
            isVisionLoaded: isVisionLoaded,
            languageTensorCount: index.languageManifest.tensorNames.count,
            visionTensorCount: index.visionManifest.tensorNames.count,
            loadedVisionWeightBytes: visionWeights?.totalByteCount ?? 0
        )
    }

    public func loadDecoderWeights() throws {
        guard decoderModel == nil else {
            throw QwenVLMNativeContainerError.decoderAlreadyLoaded
        }
        decoderModel = try QwenHybridModelReference.loadHuggingFaceLayout(
            weightStore: QwenModelWeightStore(bundleIndex: index.languageIndex),
            runtime: runtime
        )
    }

    public func loadCmlxDecoderWeights(
        attentionCacheLimit: Int? = nil,
        dsrPolicies: [Int: QwenDSRKVCachePolicy] = [:],
        attentionCacheQuantizationGroupSize: Int? = nil,
        attentionCacheQuantizationBits: Int? = nil,
        frogJumpLayerMask: UInt64 = 0
    ) throws {
        guard cmlxDecoderSession == nil else {
            throw QwenVLMNativeContainerError.decoderAlreadyLoaded
        }
        cmlxDecoderSession = try QwenCmlxLazyDecodeSession(
            bundleIndex: index.languageIndex,
            runtime: runtime,
            attentionCacheLimit: attentionCacheLimit,
            dsrPolicies: dsrPolicies,
            attentionCacheQuantizationGroupSize: attentionCacheQuantizationGroupSize,
            attentionCacheQuantizationBits: attentionCacheQuantizationBits,
            frogJumpLayerMask: frogJumpLayerMask
        )
    }

    @discardableResult
    public func unloadDecoderWeights() -> QwenVLMNativeUnloadReport {
        let hadDecoderLoaded = decoderModel != nil
        decoderModel = nil
        let purgedStats = executor.removeAffineQuantizedBuffers(scope: .decoder)
        return QwenVLMNativeUnloadReport(
            hadDecoderLoaded: hadDecoderLoaded,
            purgedQuantizedBufferStats: purgedStats
        )
    }

    public func unloadCmlxDecoderWeights() {
        cmlxDecoderSession = nil
    }

    public func loadVisionWeights() throws {
        guard visionWeights == nil else {
            throw QwenVLMNativeContainerError.visionAlreadyLoaded
        }
        let store = QwenVisionWeightStore(index: index)
        visionWeights = try store.materializeAllTensors()
        visionWeightStore = store
    }

    @discardableResult
    public func unloadVisionWeights() -> QwenVLMNativeVisionUnloadReport {
        let hadVisionLoaded = visionWeights != nil
        let releasedTensorCount = visionWeights?.tensors.count ?? 0
        let releasedByteCount = visionWeights?.totalByteCount ?? 0
        visionWeights = nil
        visionWeightStore = nil
        return QwenVLMNativeVisionUnloadReport(
            hadVisionLoaded: hadVisionLoaded,
            releasedTensorCount: releasedTensorCount,
            releasedByteCount: releasedByteCount
        )
    }

    public func loadCmlxVisionWeights() throws {
        guard cmlxVisionSession == nil else {
            throw QwenVLMNativeContainerError.visionAlreadyLoaded
        }
        let session = try EdgeMLXQwen35Session(
            architecture: index.languageIndex.architecture,
            runtime: runtime
        )
        try session.setVisionConfig(plan: index.preflightResult.plan)
        try session.loadVisionSafetensors(index: index)
        cmlxVisionSession = session
    }

    public func unloadCmlxVisionWeights() {
        cmlxVisionSession = nil
    }

    public func visionEncode(
        pixelValues: [Float],
        pixelValuesShape: [Int],
        gridTHW: [QwenImageGridTHW]
    ) throws -> EdgeMLXQwen35VisionEncoding {
        guard let cmlxVisionSession else {
            throw QwenVLMNativeContainerError.visionNotLoaded
        }
        let plan = index.preflightResult.plan
        return try cmlxVisionSession.visionEncode(
            pixelValues: pixelValues,
            pixelValuesShape: pixelValuesShape,
            gridTHW: gridTHW,
            spatialMergeSize: plan.visionConfiguration.spatialMergeSize
                ?? plan.imageProcessorConfiguration.mergeSize
                ?? 2,
            outputHiddenSize: plan.languageArchitecture.hiddenSize
        )
    }

    @discardableResult
    public func resetCmlxDecoder() throws -> QwenCmlxLazyDecodeSession {
        guard let cmlxDecoderSession else {
            throw QwenVLMNativeContainerError.decoderNotLoaded
        }
        try cmlxDecoderSession.reset()
        return cmlxDecoderSession
    }

    @discardableResult
    public func prefillMediaFeatures(
        tokenIDs: [Int],
        mediaFeatures: [Float],
        mediaFeatureShape: [Int],
        mediaTokenID: Int
    ) throws -> Int {
        guard let cmlxDecoderSession else {
            throw QwenVLMNativeContainerError.decoderNotLoaded
        }
        return try cmlxDecoderSession.prefillMediaFeatures(
            tokenIDs: tokenIDs,
            mediaFeatures: mediaFeatures,
            mediaFeatureShape: mediaFeatureShape,
            mediaTokenID: mediaTokenID
        )
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
    public func prefillCmlxTokens(tokenIDs: [Int]) throws -> Int {
        guard let cmlxDecoderSession else {
            throw QwenVLMNativeContainerError.decoderNotLoaded
        }
        return try cmlxDecoderSession.prefill(tokenIDs: tokenIDs)
    }

    public func prefillCmlxTokensAsync(tokenIDs: [Int]) throws {
        guard let cmlxDecoderSession else {
            throw QwenVLMNativeContainerError.decoderNotLoaded
        }
        try cmlxDecoderSession.prefillAsync(tokenIDs: tokenIDs)
    }

    public func restoreCmlxNeuralImprintCache(
        artifactURL: URL,
        prefixTokenCount: Int
    ) throws {
        guard let cmlxDecoderSession else {
            throw QwenVLMNativeContainerError.decoderNotLoaded
        }
        try cmlxDecoderSession.restoreNeuralImprintCache(
            artifactURL: artifactURL,
            prefixTokenCount: prefixTokenCount
        )
    }

    @discardableResult
    public func nextCmlxToken() throws -> Int {
        guard let cmlxDecoderSession else {
            throw QwenVLMNativeContainerError.decoderNotLoaded
        }
        return try cmlxDecoderSession.nextToken()
    }

    @discardableResult
    public func decodeCmlxStep(tokenID: Int) throws -> Int {
        guard let cmlxDecoderSession else {
            throw QwenVLMNativeContainerError.decoderNotLoaded
        }
        return try cmlxDecoderSession.decodeStep(tokenID: tokenID)
    }

    @discardableResult
    public func nextSampledCmlxToken(
        temperature: Float,
        topK: Int? = nil,
        topP: Float,
        minP: Float = 0,
        seed: UInt64
    ) throws -> Int {
        guard let cmlxDecoderSession else {
            throw QwenVLMNativeContainerError.decoderNotLoaded
        }
        return try cmlxDecoderSession.nextSampledToken(
            temperature: temperature,
            topK: topK,
            topP: topP,
            minP: minP,
            seed: seed
        )
    }

    /// Sampled-aware async prefill for the text path. Mirrors the LLM cmlx
    /// lazy-decode sampled prefill so VLM text-only turns can use the fast
    /// Metal decode path instead of falling back to the Swift greedy session.
    public func prefillSampledCmlxTokensAsync(
        tokenIDs: [Int],
        temperature: Float,
        topK: Int? = nil,
        topP: Float,
        minP: Float = 0,
        seed: UInt64
    ) throws {
        guard let cmlxDecoderSession else {
            throw QwenVLMNativeContainerError.decoderNotLoaded
        }
        try cmlxDecoderSession.prefillSampledAsync(
            tokenIDs: tokenIDs,
            temperature: temperature,
            topK: topK,
            topP: topP,
            minP: minP,
            seed: seed
        )
    }

    public func setCmlxSamplingPenalties(
        repetitionPenalty: Float,
        repetitionContextTokenIds: [Int],
        presencePenalty: Float,
        presenceContextTokenIds: [Int],
        frequencyPenalty: Float,
        frequencyContextTokenIds: [Int]
    ) throws {
        guard let cmlxDecoderSession else {
            throw QwenVLMNativeContainerError.decoderNotLoaded
        }
        try cmlxDecoderSession.setSamplingPenalties(
            repetitionPenalty: repetitionPenalty,
            repetitionContextTokenIds: repetitionContextTokenIds,
            presencePenalty: presencePenalty,
            presenceContextTokenIds: presenceContextTokenIds,
            frequencyPenalty: frequencyPenalty,
            frequencyContextTokenIds: frequencyContextTokenIds
        )
    }

    public func clearCmlxRepetitionPenalty() throws {
        guard let cmlxDecoderSession else {
            throw QwenVLMNativeContainerError.decoderNotLoaded
        }
        try cmlxDecoderSession.clearRepetitionPenalty()
    }

    public func setCmlxEOSSamplingBias(
        tokenIds: [Int],
        suppress: Bool,
        logitPenalty: Float
    ) throws {
        guard let cmlxDecoderSession else {
            throw QwenVLMNativeContainerError.decoderNotLoaded
        }
        try cmlxDecoderSession.setEOSSamplingBias(
            tokenIds: tokenIds,
            suppress: suppress,
            logitPenalty: logitPenalty
        )
    }

    public func clearCmlxEOSSamplingBias() throws {
        guard let cmlxDecoderSession else {
            throw QwenVLMNativeContainerError.decoderNotLoaded
        }
        try cmlxDecoderSession.clearEOSSamplingBias()
    }

    public func makeDecoderCaches(
        kvCapacity: Int? = nil,
        dsrPolicies: [Int: QwenDSRKVCachePolicy] = [:]
    ) throws -> QwenHybridDecoderCaches {
        guard decoderModel != nil else {
            throw QwenVLMNativeContainerError.decoderNotLoaded
        }
        return try QwenHybridDecoderCaches(
            architecture: index.languageIndex.architecture,
            runtime: runtime,
            kvCapacity: kvCapacity,
            dsrPolicies: dsrPolicies
        )
    }

    public func makeGreedyDecodeSession(
        kvCapacity: Int? = nil,
        dsrPolicies: [Int: QwenDSRKVCachePolicy] = [:]
    ) throws -> QwenGreedyDecodeSession {
        guard let decoderModel else {
            throw QwenVLMNativeContainerError.decoderNotLoaded
        }
        return QwenGreedyDecodeSession(
            model: decoderModel,
            caches: try makeDecoderCaches(
                kvCapacity: kvCapacity,
                dsrPolicies: dsrPolicies
            ),
            executor: executor
        )
    }
}
