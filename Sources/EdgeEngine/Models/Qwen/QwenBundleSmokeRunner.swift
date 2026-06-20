// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Darwin
import Dispatch
import Foundation

public enum QwenBundleSmokeError: Error, Equatable {
    case emptyPromptTokenIds
    case invalidMaxNewTokenCount(Int)
    case invalidPrefillTopLogitCount(Int)
    case invalidStepTopLogitCount(Int)
    case kvCapacityTooSmall(minimum: Int, actual: Int)
}

public enum QwenBundleSmokeOutputKind: String, Codable, Equatable, Sendable {
    case float
    case quantized
}

public enum QwenBundleSmokeDecodeBackend: String, Codable, Equatable, Sendable {
    case swiftReference
    case cmlxDecodeStep
}

public struct QwenBundleSmokeConfiguration {
    public var modelRootURL: URL
    public var promptTokenIds: [Int]
    public var maxNewTokenCount: Int
    public var endTokenIds: Set<Int>
    public var kvCapacity: Int?
    public var family: QwenModelFamily?
    public var runtimeConfiguration: MetalRuntimeConfiguration
    public var prefillTopLogitCount: Int
    public var stepTopLogitCount: Int
    public var decodeBackend: QwenBundleSmokeDecodeBackend

    public init(
        modelRootURL: URL,
        promptTokenIds: [Int],
        maxNewTokenCount: Int = 1,
        endTokenIds: Set<Int> = [],
        kvCapacity: Int? = nil,
        family: QwenModelFamily? = nil,
        runtimeConfiguration: MetalRuntimeConfiguration = .init(),
        prefillTopLogitCount: Int = 5,
        stepTopLogitCount: Int? = nil,
        decodeBackend: QwenBundleSmokeDecodeBackend = .swiftReference
    ) {
        self.modelRootURL = modelRootURL
        self.promptTokenIds = promptTokenIds
        self.maxNewTokenCount = maxNewTokenCount
        self.endTokenIds = endTokenIds
        self.kvCapacity = kvCapacity
        self.family = family
        self.runtimeConfiguration = runtimeConfiguration
        self.prefillTopLogitCount = prefillTopLogitCount
        self.stepTopLogitCount = stepTopLogitCount ?? prefillTopLogitCount
        self.decodeBackend = decodeBackend
    }
}

public struct QwenBundleSmokeLogitSummary: Codable, Equatable, Sendable {
    public var tokenId: Int
    public var logit: Float

    public init(tokenId: Int, logit: Float) {
        self.tokenId = tokenId
        self.logit = logit
    }
}

public struct QwenBundleSmokeStep: Codable, Equatable, Sendable {
    public var tokenId: Int
    public var logit: Float
    public var topLogits: [QwenBundleSmokeLogitSummary]
    public var cacheTokenPosition: Int
    public var reachedEndToken: Bool

    public init(
        tokenId: Int,
        logit: Float,
        topLogits: [QwenBundleSmokeLogitSummary] = [],
        cacheTokenPosition: Int,
        reachedEndToken: Bool
    ) {
        self.tokenId = tokenId
        self.logit = logit
        self.topLogits = topLogits
        self.cacheTokenPosition = cacheTokenPosition
        self.reachedEndToken = reachedEndToken
    }

    private enum CodingKeys: String, CodingKey {
        case tokenId
        case logit
        case topLogits
        case cacheTokenPosition
        case reachedEndToken
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.tokenId = try container.decode(Int.self, forKey: .tokenId)
        self.logit = try container.decode(Float.self, forKey: .logit)
        self.topLogits = try container.decodeIfPresent(
            [QwenBundleSmokeLogitSummary].self,
            forKey: .topLogits
        ) ?? []
        self.cacheTokenPosition = try container.decode(Int.self, forKey: .cacheTokenPosition)
        self.reachedEndToken = try container.decode(Bool.self, forKey: .reachedEndToken)
    }
}

public struct QwenBundleSmokeRuntimeSnapshot: Codable, Equatable, Sendable {
    public var maxOpsPerCommandBuffer: Int
    public var effectiveMaxOpsPerCommandBuffer: Int
    public var maxMBPerCommandBuffer: Int
    public var contextLengthHint: Int
    public var dynamicOpsScheduleEnabled: Bool
    public var quantizedBufferCacheLimitBytes: Int?
    public var releaseQuantizedHostStorageAfterUpload: Bool
    public var logicalCommitCount: Int
    public var quantizedBufferCacheEntries: Int
    public var quantizedBufferCacheHits: Int
    public var quantizedBufferCacheMisses: Int
    public var quantizedBufferUploadedByteCount: Int
    public var quantizedBufferCachedByteCount: Int
    public var quantizedHostStorageReleasedByteCount: Int
    public var quantizedHostStorageReleaseCount: Int

    public init(
        maxOpsPerCommandBuffer: Int,
        effectiveMaxOpsPerCommandBuffer: Int,
        maxMBPerCommandBuffer: Int,
        contextLengthHint: Int,
        dynamicOpsScheduleEnabled: Bool,
        quantizedBufferCacheLimitBytes: Int? = nil,
        releaseQuantizedHostStorageAfterUpload: Bool = false,
        logicalCommitCount: Int,
        quantizedBufferCacheEntries: Int,
        quantizedBufferCacheHits: Int,
        quantizedBufferCacheMisses: Int,
        quantizedBufferUploadedByteCount: Int,
        quantizedBufferCachedByteCount: Int = 0,
        quantizedHostStorageReleasedByteCount: Int = 0,
        quantizedHostStorageReleaseCount: Int = 0
    ) {
        self.maxOpsPerCommandBuffer = maxOpsPerCommandBuffer
        self.effectiveMaxOpsPerCommandBuffer = effectiveMaxOpsPerCommandBuffer
        self.maxMBPerCommandBuffer = maxMBPerCommandBuffer
        self.contextLengthHint = contextLengthHint
        self.dynamicOpsScheduleEnabled = dynamicOpsScheduleEnabled
        self.quantizedBufferCacheLimitBytes = quantizedBufferCacheLimitBytes
        self.releaseQuantizedHostStorageAfterUpload = releaseQuantizedHostStorageAfterUpload
        self.logicalCommitCount = logicalCommitCount
        self.quantizedBufferCacheEntries = quantizedBufferCacheEntries
        self.quantizedBufferCacheHits = quantizedBufferCacheHits
        self.quantizedBufferCacheMisses = quantizedBufferCacheMisses
        self.quantizedBufferUploadedByteCount = quantizedBufferUploadedByteCount
        self.quantizedBufferCachedByteCount = quantizedBufferCachedByteCount
        self.quantizedHostStorageReleasedByteCount = quantizedHostStorageReleasedByteCount
        self.quantizedHostStorageReleaseCount = quantizedHostStorageReleaseCount
    }
}

public struct QwenBundleSmokeMemorySnapshot: Codable, Equatable, Sendable {
    public var residentAtStartBytes: UInt64?
    public var residentAfterModelLoadBytes: UInt64?
    public var residentAfterCacheAllocationBytes: UInt64?
    public var residentAfterPrefillBytes: UInt64?
    public var residentAfterDecodeBytes: UInt64?
    public var peakResidentBytes: UInt64?

    public init(
        residentAtStartBytes: UInt64?,
        residentAfterModelLoadBytes: UInt64?,
        residentAfterCacheAllocationBytes: UInt64?,
        residentAfterPrefillBytes: UInt64?,
        residentAfterDecodeBytes: UInt64?,
        peakResidentBytes: UInt64?
    ) {
        self.residentAtStartBytes = residentAtStartBytes
        self.residentAfterModelLoadBytes = residentAfterModelLoadBytes
        self.residentAfterCacheAllocationBytes = residentAfterCacheAllocationBytes
        self.residentAfterPrefillBytes = residentAfterPrefillBytes
        self.residentAfterDecodeBytes = residentAfterDecodeBytes
        self.peakResidentBytes = peakResidentBytes
    }
}

public struct QwenBundleSmokeResult: Codable, Equatable, Sendable {
    public var modelRootPath: String
    public var modelPrefix: String
    public var family: QwenModelFamily
    public var vocabularySize: Int
    public var hiddenSize: Int
    public var layerCount: Int
    public var fullAttentionLayerCount: Int
    public var gdnLayerCount: Int
    public var quantization: QwenQuantizationProfile?
    public var outputKind: QwenBundleSmokeOutputKind
    public var usesTiedEmbeddings: Bool
    public var promptTokenIds: [Int]
    public var maxNewTokenCount: Int
    public var generatedTokenIds: [Int]
    public var steps: [QwenBundleSmokeStep]
    public var prefillLogitsShape: [Int]
    public var prefillTopLogits: [QwenBundleSmokeLogitSummary]
    public var cacheTokenPosition: Int
    public var decodeBackend: QwenBundleSmokeDecodeBackend
    public var prefillDurationSeconds: Double?
    public var decodeDurationSeconds: Double?
    public var decodeTokensPerSecond: Double?
    public var runtime: QwenBundleSmokeRuntimeSnapshot
    public var memory: QwenBundleSmokeMemorySnapshot

    public init(
        modelRootPath: String,
        modelPrefix: String,
        family: QwenModelFamily,
        vocabularySize: Int,
        hiddenSize: Int,
        layerCount: Int,
        fullAttentionLayerCount: Int,
        gdnLayerCount: Int,
        quantization: QwenQuantizationProfile?,
        outputKind: QwenBundleSmokeOutputKind,
        usesTiedEmbeddings: Bool,
        promptTokenIds: [Int],
        maxNewTokenCount: Int,
        generatedTokenIds: [Int],
        steps: [QwenBundleSmokeStep],
        prefillLogitsShape: [Int],
        prefillTopLogits: [QwenBundleSmokeLogitSummary] = [],
        cacheTokenPosition: Int,
        decodeBackend: QwenBundleSmokeDecodeBackend = .swiftReference,
        prefillDurationSeconds: Double? = nil,
        decodeDurationSeconds: Double? = nil,
        decodeTokensPerSecond: Double? = nil,
        runtime: QwenBundleSmokeRuntimeSnapshot,
        memory: QwenBundleSmokeMemorySnapshot = .empty
    ) {
        self.modelRootPath = modelRootPath
        self.modelPrefix = modelPrefix
        self.family = family
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.fullAttentionLayerCount = fullAttentionLayerCount
        self.gdnLayerCount = gdnLayerCount
        self.quantization = quantization
        self.outputKind = outputKind
        self.usesTiedEmbeddings = usesTiedEmbeddings
        self.promptTokenIds = promptTokenIds
        self.maxNewTokenCount = maxNewTokenCount
        self.generatedTokenIds = generatedTokenIds
        self.steps = steps
        self.prefillLogitsShape = prefillLogitsShape
        self.prefillTopLogits = prefillTopLogits
        self.cacheTokenPosition = cacheTokenPosition
        self.decodeBackend = decodeBackend
        self.prefillDurationSeconds = prefillDurationSeconds
        self.decodeDurationSeconds = decodeDurationSeconds
        self.decodeTokensPerSecond = decodeTokensPerSecond
        self.runtime = runtime
        self.memory = memory
    }

    private enum CodingKeys: String, CodingKey {
        case modelRootPath
        case modelPrefix
        case family
        case vocabularySize
        case hiddenSize
        case layerCount
        case fullAttentionLayerCount
        case gdnLayerCount
        case quantization
        case outputKind
        case usesTiedEmbeddings
        case promptTokenIds
        case maxNewTokenCount
        case generatedTokenIds
        case steps
        case prefillLogitsShape
        case prefillTopLogits
        case cacheTokenPosition
        case decodeBackend
        case prefillDurationSeconds
        case decodeDurationSeconds
        case decodeTokensPerSecond
        case runtime
        case memory
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelRootPath = try container.decode(String.self, forKey: .modelRootPath)
        self.modelPrefix = try container.decode(String.self, forKey: .modelPrefix)
        self.family = try container.decode(QwenModelFamily.self, forKey: .family)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.layerCount = try container.decode(Int.self, forKey: .layerCount)
        self.fullAttentionLayerCount = try container.decode(Int.self, forKey: .fullAttentionLayerCount)
        self.gdnLayerCount = try container.decode(Int.self, forKey: .gdnLayerCount)
        self.quantization = try container.decodeIfPresent(QwenQuantizationProfile.self, forKey: .quantization)
        self.outputKind = try container.decode(QwenBundleSmokeOutputKind.self, forKey: .outputKind)
        self.usesTiedEmbeddings = try container.decode(Bool.self, forKey: .usesTiedEmbeddings)
        self.promptTokenIds = try container.decode([Int].self, forKey: .promptTokenIds)
        self.maxNewTokenCount = try container.decode(Int.self, forKey: .maxNewTokenCount)
        self.generatedTokenIds = try container.decode([Int].self, forKey: .generatedTokenIds)
        self.steps = try container.decode([QwenBundleSmokeStep].self, forKey: .steps)
        self.prefillLogitsShape = try container.decode([Int].self, forKey: .prefillLogitsShape)
        self.prefillTopLogits = try container.decodeIfPresent(
            [QwenBundleSmokeLogitSummary].self,
            forKey: .prefillTopLogits
        ) ?? []
        self.cacheTokenPosition = try container.decode(Int.self, forKey: .cacheTokenPosition)
        self.decodeBackend = try container.decodeIfPresent(
            QwenBundleSmokeDecodeBackend.self,
            forKey: .decodeBackend
        ) ?? .swiftReference
        self.prefillDurationSeconds = try container.decodeIfPresent(
            Double.self,
            forKey: .prefillDurationSeconds
        )
        self.decodeDurationSeconds = try container.decodeIfPresent(
            Double.self,
            forKey: .decodeDurationSeconds
        )
        self.decodeTokensPerSecond = try container.decodeIfPresent(
            Double.self,
            forKey: .decodeTokensPerSecond
        )
        self.runtime = try container.decode(QwenBundleSmokeRuntimeSnapshot.self, forKey: .runtime)
        self.memory = try container.decodeIfPresent(
            QwenBundleSmokeMemorySnapshot.self,
            forKey: .memory
        ) ?? .empty
    }
}

public enum QwenBundleSmokeRunner {
    public static func run(configuration: QwenBundleSmokeConfiguration) throws -> QwenBundleSmokeResult {
        guard !configuration.promptTokenIds.isEmpty else {
            throw QwenBundleSmokeError.emptyPromptTokenIds
        }
        guard configuration.maxNewTokenCount >= 0 else {
            throw QwenBundleSmokeError.invalidMaxNewTokenCount(configuration.maxNewTokenCount)
        }
        guard configuration.prefillTopLogitCount >= 0 else {
            throw QwenBundleSmokeError.invalidPrefillTopLogitCount(configuration.prefillTopLogitCount)
        }
        guard configuration.stepTopLogitCount >= 0 else {
            throw QwenBundleSmokeError.invalidStepTopLogitCount(configuration.stepTopLogitCount)
        }

        let requiredCapacity = max(1, configuration.promptTokenIds.count + configuration.maxNewTokenCount)
        if let kvCapacity = configuration.kvCapacity, kvCapacity < requiredCapacity {
            throw QwenBundleSmokeError.kvCapacityTooSmall(
                minimum: requiredCapacity,
                actual: kvCapacity
            )
        }

        let residentAtStartBytes = ProcessMemoryProbe.residentBytes()
        var runtimeConfiguration = configuration.runtimeConfiguration
        if runtimeConfiguration.contextLengthHint == 0 {
            runtimeConfiguration.contextLengthHint = requiredCapacity
        }
        let runtime = try EdgeMetalRuntime(configuration: runtimeConfiguration)
        let bundleIndex = try QwenModelBundleIndex.load(
            from: configuration.modelRootURL,
            family: configuration.family
        )
        if configuration.decodeBackend == .cmlxDecodeStep {
            return try runCmlxDecodeStep(
                configuration: configuration,
                bundleIndex: bundleIndex,
                runtime: runtime,
                residentAtStartBytes: residentAtStartBytes,
                residentAfterModelLoadBytes: ProcessMemoryProbe.residentBytes()
            )
        }
        let model = try QwenHybridModelReference.loadHuggingFaceLayout(
            weightStore: QwenModelWeightStore(bundleIndex: bundleIndex),
            runtime: runtime
        )
        let residentAfterModelLoadBytes = ProcessMemoryProbe.residentBytes()
        let executor = try MetalKernelExecutor(runtime: runtime)

        let caches = try QwenHybridDecoderCaches(
            architecture: bundleIndex.architecture,
            runtime: runtime,
            kvCapacity: configuration.kvCapacity ?? requiredCapacity
        )
        let residentAfterCacheAllocationBytes = ProcessMemoryProbe.residentBytes()
        let session = QwenGreedyDecodeSession(
            model: model,
            caches: caches,
            executor: executor
        )
        let prefillStartSeconds = monotonicSeconds()
        let prefillLogits = try session.prefill(promptTokenIds: configuration.promptTokenIds)
        let prefillDurationSeconds = monotonicSeconds() - prefillStartSeconds
        let residentAfterPrefillBytes = ProcessMemoryProbe.residentBytes()
        let prefillTopLogits = try topLogits(
            logits: prefillLogits,
            count: configuration.prefillTopLogitCount
        )
        var currentLogits = prefillLogits
        var steps: [QwenBundleSmokeStep] = []
        steps.reserveCapacity(configuration.maxNewTokenCount)

        let decodeStartSeconds = monotonicSeconds()
        for _ in 0..<configuration.maxNewTokenCount {
            let stepTopLogits = try topLogits(
                logits: currentLogits,
                count: configuration.stepTopLogitCount
            )
            let token = try QwenGreedyDecoder.nextToken(logits: currentLogits)
            let reachedEndToken = configuration.endTokenIds.contains(token.tokenId)
            if reachedEndToken {
                session.invalidateCurrentLogits()
            } else {
                currentLogits = try session.advance(with: token.tokenId)
            }
            steps.append(
                QwenBundleSmokeStep(
                    tokenId: token.tokenId,
                    logit: token.logit,
                    topLogits: stepTopLogits,
                    cacheTokenPosition: try session.tokenPosition(),
                    reachedEndToken: reachedEndToken
                )
            )
            if reachedEndToken {
                break
            }
        }
        let decodeDurationSeconds = monotonicSeconds() - decodeStartSeconds
        let residentAfterDecodeBytes = ProcessMemoryProbe.residentBytes()

        return QwenBundleSmokeResult(
            modelRootPath: configuration.modelRootURL.path,
            modelPrefix: bundleIndex.modelPrefix,
            family: bundleIndex.architecture.family,
            vocabularySize: bundleIndex.architecture.vocabularySize,
            hiddenSize: bundleIndex.architecture.hiddenSize,
            layerCount: bundleIndex.architecture.layerCount,
            fullAttentionLayerCount: bundleIndex.architecture.fullAttentionLayerIndices.count,
            gdnLayerCount: bundleIndex.architecture.gdnLayerIndices.count,
            quantization: bundleIndex.architecture.quantization,
            outputKind: outputKind(for: model.outputWeights),
            usesTiedEmbeddings: model.outputWeights.usesTiedEmbeddings,
            promptTokenIds: configuration.promptTokenIds,
            maxNewTokenCount: configuration.maxNewTokenCount,
            generatedTokenIds: steps.map(\.tokenId),
            steps: steps,
            prefillLogitsShape: prefillLogits.shape.dimensions,
            prefillTopLogits: prefillTopLogits,
            cacheTokenPosition: try session.tokenPosition(),
            decodeBackend: configuration.decodeBackend,
            prefillDurationSeconds: prefillDurationSeconds,
            decodeDurationSeconds: decodeDurationSeconds,
            decodeTokensPerSecond: tokensPerSecond(
                tokenCount: steps.count,
                durationSeconds: decodeDurationSeconds
            ),
            runtime: runtimeSnapshot(
                configuration: runtime.configuration,
                executor: executor
            ),
            memory: QwenBundleSmokeMemorySnapshot(
                residentAtStartBytes: residentAtStartBytes,
                residentAfterModelLoadBytes: residentAfterModelLoadBytes,
                residentAfterCacheAllocationBytes: residentAfterCacheAllocationBytes,
                residentAfterPrefillBytes: residentAfterPrefillBytes,
                residentAfterDecodeBytes: residentAfterDecodeBytes,
                peakResidentBytes: ProcessMemoryProbe.peakResidentBytes()
            )
        )
    }

    private static func runCmlxDecodeStep(
        configuration: QwenBundleSmokeConfiguration,
        bundleIndex: QwenModelBundleIndex,
        runtime: EdgeMetalRuntime,
        residentAtStartBytes: UInt64?,
        residentAfterModelLoadBytes: UInt64?
    ) throws -> QwenBundleSmokeResult {
        let session = try QwenCmlxLazyDecodeSession(
            bundleIndex: bundleIndex,
            runtime: runtime
        )
        let residentAfterCacheAllocationBytes = ProcessMemoryProbe.residentBytes()

        let prefillStartSeconds = monotonicSeconds()
        var nextTokenID: Int? = try session.prefill(tokenIDs: configuration.promptTokenIds)
        let prefillDurationSeconds = monotonicSeconds() - prefillStartSeconds
        let residentAfterPrefillBytes = ProcessMemoryProbe.residentBytes()

        var steps: [QwenBundleSmokeStep] = []
        steps.reserveCapacity(configuration.maxNewTokenCount)
        let decodeStartSeconds = monotonicSeconds()
        while steps.count < configuration.maxNewTokenCount, let tokenID = nextTokenID {
            let reachedEndToken = configuration.endTokenIds.contains(tokenID)
            if !reachedEndToken {
                nextTokenID = try session.decodeStep(tokenID: tokenID)
            }
            steps.append(
                QwenBundleSmokeStep(
                    tokenId: tokenID,
                    logit: 0,
                    topLogits: [],
                    cacheTokenPosition: session.tokenPosition,
                    reachedEndToken: reachedEndToken
                )
            )
            if reachedEndToken {
                break
            }
        }
        let decodeDurationSeconds = monotonicSeconds() - decodeStartSeconds
        let residentAfterDecodeBytes = ProcessMemoryProbe.residentBytes()

        return QwenBundleSmokeResult(
            modelRootPath: configuration.modelRootURL.path,
            modelPrefix: bundleIndex.modelPrefix,
            family: bundleIndex.architecture.family,
            vocabularySize: bundleIndex.architecture.vocabularySize,
            hiddenSize: bundleIndex.architecture.hiddenSize,
            layerCount: bundleIndex.architecture.layerCount,
            fullAttentionLayerCount: bundleIndex.architecture.fullAttentionLayerIndices.count,
            gdnLayerCount: bundleIndex.architecture.gdnLayerIndices.count,
            quantization: bundleIndex.architecture.quantization,
            outputKind: .quantized,
            usesTiedEmbeddings: bundleIndex.modelLevelManifest.isWeightTied,
            promptTokenIds: configuration.promptTokenIds,
            maxNewTokenCount: configuration.maxNewTokenCount,
            generatedTokenIds: steps.map(\.tokenId),
            steps: steps,
            prefillLogitsShape: [],
            prefillTopLogits: [],
            cacheTokenPosition: session.tokenPosition,
            decodeBackend: configuration.decodeBackend,
            prefillDurationSeconds: prefillDurationSeconds,
            decodeDurationSeconds: decodeDurationSeconds,
            decodeTokensPerSecond: tokensPerSecond(
                tokenCount: steps.count,
                durationSeconds: decodeDurationSeconds
            ),
            runtime: runtimeSnapshot(configuration: runtime.configuration),
            memory: QwenBundleSmokeMemorySnapshot(
                residentAtStartBytes: residentAtStartBytes,
                residentAfterModelLoadBytes: residentAfterModelLoadBytes,
                residentAfterCacheAllocationBytes: residentAfterCacheAllocationBytes,
                residentAfterPrefillBytes: residentAfterPrefillBytes,
                residentAfterDecodeBytes: residentAfterDecodeBytes,
                peakResidentBytes: ProcessMemoryProbe.peakResidentBytes()
            )
        )
    }

    private static func topLogits(
        logits: EdgeTensor,
        count: Int
    ) throws -> [QwenBundleSmokeLogitSummary] {
        guard count > 0 else {
            return []
        }
        guard logits.shape.rank == 2,
              logits.shape.dimensions[0] > 0,
              logits.shape.dimensions[1] > 0
        else {
            throw QwenTokenSamplerError.invalidLogitsShape(
                expected: [-1, -1],
                actual: logits.shape.dimensions
            )
        }

        let values = try logits.readFloat32()
        let vocabularySize = logits.shape.dimensions[1]
        let topCount = min(count, vocabularySize)
        let lastRowOffset = (logits.shape.dimensions[0] - 1) * vocabularySize
        var topLogits: [QwenBundleSmokeLogitSummary] = []
        topLogits.reserveCapacity(topCount)

        for tokenId in 0..<vocabularySize {
            let logit = values[lastRowOffset + tokenId]
            guard logit.isFinite else {
                throw QwenTokenSamplerError.nonFiniteLogit(tokenId: tokenId, value: logit)
            }
            insertTopLogit(
                QwenBundleSmokeLogitSummary(tokenId: tokenId, logit: logit),
                into: &topLogits,
                limit: topCount
            )
        }
        return topLogits
    }

    private static func insertTopLogit(
        _ candidate: QwenBundleSmokeLogitSummary,
        into topLogits: inout [QwenBundleSmokeLogitSummary],
        limit: Int
    ) {
        if topLogits.count == limit,
           let last = topLogits.last,
           !isBetterLogit(candidate, than: last) {
            return
        }

        var index = topLogits.count
        while index > 0, isBetterLogit(candidate, than: topLogits[index - 1]) {
            index -= 1
        }
        topLogits.insert(candidate, at: index)
        if topLogits.count > limit {
            topLogits.removeLast()
        }
    }

    private static func isBetterLogit(
        _ lhs: QwenBundleSmokeLogitSummary,
        than rhs: QwenBundleSmokeLogitSummary
    ) -> Bool {
        if lhs.logit == rhs.logit {
            return lhs.tokenId < rhs.tokenId
        }
        return lhs.logit > rhs.logit
    }

    private static func outputKind(for outputWeights: QwenHybridModelOutputReference) -> QwenBundleSmokeOutputKind {
        switch outputWeights {
        case .float:
            .float
        case .quantized:
            .quantized
        }
    }

    private static func runtimeSnapshot(
        configuration: MetalRuntimeConfiguration,
        executor: MetalKernelExecutor? = nil
    ) -> QwenBundleSmokeRuntimeSnapshot {
        let cacheStats = executor?.affineQuantizedBufferCacheStats
        return QwenBundleSmokeRuntimeSnapshot(
            maxOpsPerCommandBuffer: configuration.maxOpsPerCommandBuffer,
            effectiveMaxOpsPerCommandBuffer: configuration.effectiveMaxOpsPerCommandBuffer,
            maxMBPerCommandBuffer: configuration.maxMBPerCommandBuffer,
            contextLengthHint: configuration.contextLengthHint,
            dynamicOpsScheduleEnabled: configuration.dynamicOpsSchedule != nil,
            quantizedBufferCacheLimitBytes: configuration.quantizedBufferCacheLimitBytes,
            releaseQuantizedHostStorageAfterUpload: configuration.releaseQuantizedHostStorageAfterUpload,
            logicalCommitCount: executor?.lastExecutionStats?.logicalCommitCount
                ?? executor?.schedulingSnapshot.commitCount
                ?? 0,
            quantizedBufferCacheEntries: cacheStats?.entryCount ?? 0,
            quantizedBufferCacheHits: cacheStats?.hitCount ?? 0,
            quantizedBufferCacheMisses: cacheStats?.missCount ?? 0,
            quantizedBufferUploadedByteCount: cacheStats?.uploadedByteCount ?? 0,
            quantizedBufferCachedByteCount: cacheStats?.cachedByteCount ?? 0,
            quantizedHostStorageReleasedByteCount: cacheStats?.releasedHostStorageByteCount ?? 0,
            quantizedHostStorageReleaseCount: cacheStats?.releasedHostStorageCount ?? 0
        )
    }

    private static func monotonicSeconds() -> Double {
        Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000_000
    }

    private static func tokensPerSecond(tokenCount: Int, durationSeconds: Double) -> Double? {
        guard tokenCount > 0, durationSeconds > 0 else {
            return nil
        }
        return Double(tokenCount) / durationSeconds
    }
}

extension QwenBundleSmokeMemorySnapshot {
    public static let empty = QwenBundleSmokeMemorySnapshot(
        residentAtStartBytes: nil,
        residentAfterModelLoadBytes: nil,
        residentAfterCacheAllocationBytes: nil,
        residentAfterPrefillBytes: nil,
        residentAfterDecodeBytes: nil,
        peakResidentBytes: nil
    )
}

private enum ProcessMemoryProbe {
    static func residentBytes() -> UInt64? {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.stride / MemoryLayout<natural_t>.stride
        )
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    rebound,
                    &count
                )
            }
        }
        guard result == KERN_SUCCESS else {
            return nil
        }
        return UInt64(info.resident_size)
    }

    static func peakResidentBytes() -> UInt64? {
        var usage = rusage()
        guard getrusage(RUSAGE_SELF, &usage) == 0, usage.ru_maxrss > 0 else {
            return nil
        }
        return UInt64(usage.ru_maxrss)
    }
}
