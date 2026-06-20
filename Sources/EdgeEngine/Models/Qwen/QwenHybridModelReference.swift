// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum QwenHybridModelReferenceError: Error, Equatable {
    case layerCountMismatch(expected: Int, actual: Int)
    case layerIndexMismatch(expected: Int, actual: Int)
    case layerKindMismatch(layerIndex: Int, expected: QwenHybridLayerKind, actual: QwenHybridLayerKind)
    case emptyTokenIds
    case cacheTokenPositionMismatch(expected: Int, actual: Int)
    case invalidCaptureLayer(Int)
}

public enum QwenHybridModelOutputReference {
    case float(QwenModelOutputWeights)
    case quantized(QwenQuantizedModelOutputWeights)

    public var usesTiedEmbeddings: Bool {
        switch self {
        case .float(let weights):
            weights.usesTiedEmbeddings
        case .quantized(let weights):
            weights.usesTiedEmbeddings
        }
    }

    public func logits(hiddenStates: EdgeTensor, executor: MetalKernelExecutor) throws -> EdgeTensor {
        switch self {
        case .float(let weights):
            try weights.logits(hiddenStates: hiddenStates, executor: executor)
        case .quantized(let weights):
            try weights.logits(hiddenStates: hiddenStates, executor: executor)
        }
    }

    public func greedyToken(hiddenStates: EdgeTensor, executor: MetalKernelExecutor) throws -> QwenGreedyToken {
        switch self {
        case .float(let weights):
            try weights.greedyToken(hiddenStates: hiddenStates, executor: executor)
        case .quantized(let weights):
            try weights.greedyToken(hiddenStates: hiddenStates, executor: executor)
        }
    }
}

public struct QwenHybridModelReference {
    public var architecture: QwenHybridArchitecture
    public var embeddings: QwenTokenEmbeddingWeights
    public var decoderLayers: [QwenHybridDecoderLayerReference]
    public var outputWeights: QwenHybridModelOutputReference

    public init(
        architecture: QwenHybridArchitecture,
        embeddings: QwenTokenEmbeddingWeights,
        decoderLayers: [QwenHybridDecoderLayerReference],
        outputWeights: QwenHybridModelOutputReference
    ) throws {
        try architecture.validate()
        guard decoderLayers.count == architecture.layerCount else {
            throw QwenHybridModelReferenceError.layerCountMismatch(
                expected: architecture.layerCount,
                actual: decoderLayers.count
            )
        }
        for layerPlan in architecture.layerPlan {
            let layer = decoderLayers[layerPlan.index]
            guard layer.layerIndex == layerPlan.index else {
                throw QwenHybridModelReferenceError.layerIndexMismatch(
                    expected: layerPlan.index,
                    actual: layer.layerIndex
                )
            }
            guard layer.kind == layerPlan.kind else {
                throw QwenHybridModelReferenceError.layerKindMismatch(
                    layerIndex: layerPlan.index,
                    expected: layerPlan.kind,
                    actual: layer.kind
                )
            }
        }
        self.architecture = architecture
        self.embeddings = embeddings
        self.decoderLayers = decoderLayers
        self.outputWeights = outputWeights
    }

    public init(
        architecture: QwenHybridArchitecture,
        embeddings: QwenTokenEmbeddingWeights,
        decoderLayers: [QwenHybridDecoderLayerReference],
        outputWeights: QwenModelOutputWeights
    ) throws {
        try self.init(
            architecture: architecture,
            embeddings: embeddings,
            decoderLayers: decoderLayers,
            outputWeights: .float(outputWeights)
        )
    }

    public init(
        architecture: QwenHybridArchitecture,
        embeddings: QwenTokenEmbeddingWeights,
        decoderLayers: [QwenHybridDecoderLayerReference],
        outputWeights: QwenQuantizedModelOutputWeights
    ) throws {
        try self.init(
            architecture: architecture,
            embeddings: embeddings,
            decoderLayers: decoderLayers,
            outputWeights: .quantized(outputWeights)
        )
    }

    public static func loadHuggingFaceLayout(
        weightStore: QwenModelWeightStore,
        runtime: EdgeMetalRuntime
    ) throws -> QwenHybridModelReference {
        try weightStore.bundleIndex.validateRequiredTensorCoverage()
        let architecture = weightStore.bundleIndex.architecture
        let decoderLayers = try (0..<architecture.layerCount).map { layerIndex in
            try QwenHybridDecoderLayerReference.loadHuggingFaceLayout(
                layerIndex: layerIndex,
                weightStore: weightStore,
                runtime: runtime
            )
        }
        let embeddings = try QwenTokenEmbeddingWeights.loadHuggingFaceLayout(
            weightStore: weightStore,
            runtime: runtime
        )
        return try QwenHybridModelReference(
            architecture: architecture,
            embeddings: embeddings,
            decoderLayers: decoderLayers,
            outputWeights: loadOutputWeights(
                weightStore: weightStore,
                tokenEmbeddings: embeddings,
                runtime: runtime
            )
        )
    }

    private static func loadOutputWeights(
        weightStore: QwenModelWeightStore,
        tokenEmbeddings: QwenTokenEmbeddingWeights,
        runtime: EdgeMetalRuntime
    ) throws -> QwenHybridModelOutputReference {
        let architecture = weightStore.bundleIndex.architecture
        let manifest = weightStore.bundleIndex.modelLevelManifest
        let outputWeightName = manifest.lmHeadName ?? manifest.embedTokensName
        let hasQuantizedOutputCompanion = manifest.quantizedWeightGroups.contains { group in
            group.weightName == outputWeightName && group.scalesName != nil
        }
        if architecture.quantization != nil && hasQuantizedOutputCompanion {
            return .quantized(
                try QwenQuantizedModelOutputWeights.loadHuggingFaceLayout(
                    weightStore: weightStore,
                    runtime: runtime,
                    tiedLMHead: tiedQuantizedEmbedding(from: tokenEmbeddings, manifest: manifest)
                )
            )
        }
        return .float(
            try QwenModelOutputWeights.loadHuggingFaceLayout(
                weightStore: weightStore,
                runtime: runtime
            )
        )
    }

    private static func tiedQuantizedEmbedding(
        from tokenEmbeddings: QwenTokenEmbeddingWeights,
        manifest: QwenModelLevelManifest
    ) -> EdgeQuantizedTensor? {
        guard manifest.lmHeadName == nil,
              case .quantized(let embeddings) = tokenEmbeddings.embeddings
        else {
            return nil
        }
        return embeddings
    }

    public func hiddenStates(
        tokenIds: [Int],
        caches: QwenHybridDecoderCaches,
        executor: MetalKernelExecutor,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> EdgeTensor {
        guard !tokenIds.isEmpty else {
            throw QwenHybridModelReferenceError.emptyTokenIds
        }
        let initialTokenPosition = try caches.tokenPosition()
        diagnosticSink?("embedding_begin tokens=\(tokenIds.count) position=\(initialTokenPosition)")
        var hiddenStates = try autoreleasepool {
            try embeddings.hiddenStates(
                tokenIds: tokenIds,
                executor: executor
            )
        }
        diagnosticSink?("embedding_done shape=\(hiddenStates.shape.dimensions)")
        let shouldBatchPrefillLayers = tokenIds.count > 1
            && executor.runtimeConfiguration.usePrefillLayerCommandBufferBatching
            && !executor.runtimeConfiguration.useSingleCommandBufferPrefill
        if shouldBatchPrefillLayers {
            diagnosticSink?("prefill_layer_command_buffer_batching_begin layers=\(decoderLayers.count)")
        }
        for layer in decoderLayers {
            diagnosticSink?("layer_\(layer.layerIndex)_\(layer.kind.rawValue)_begin shape=\(hiddenStates.shape.dimensions)")
            let layerInput = hiddenStates
            hiddenStates = try autoreleasepool {
                let evaluateLayer = {
                    try self.outputTensor(
                        for: layer,
                        hiddenStates: layerInput,
                        initialTokenPosition: initialTokenPosition,
                        caches: caches,
                        executor: executor,
                        diagnosticSink: diagnosticSink
                    )
                }
                if shouldBatchPrefillLayers {
                    diagnosticSink?("layer_\(layer.layerIndex)_command_buffer_batch_begin")
                    let output = try executor.withUnboundedCommandBufferBatch(evaluateLayer)
                    diagnosticSink?("layer_\(layer.layerIndex)_command_buffer_batch_done")
                    return output
                }
                return try evaluateLayer()
            }
            diagnosticSink?("layer_\(layer.layerIndex)_\(layer.kind.rawValue)_done shape=\(hiddenStates.shape.dimensions)")
        }
        if shouldBatchPrefillLayers {
            diagnosticSink?("prefill_layer_command_buffer_batching_done shape=\(hiddenStates.shape.dimensions)")
        }
        let expectedTokenPosition = initialTokenPosition + tokenIds.count
        let actualTokenPosition = try caches.tokenPosition()
        guard actualTokenPosition == expectedTokenPosition else {
            throw QwenHybridModelReferenceError.cacheTokenPositionMismatch(
                expected: expectedTokenPosition,
                actual: actualTokenPosition
            )
        }
        return hiddenStates
    }

    public func lastTokenHiddenState(
        tokenIds: [Int],
        targetLayer: Int,
        caches: QwenHybridDecoderCaches,
        executor: MetalKernelExecutor,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> EdgeTensor {
        guard !tokenIds.isEmpty else {
            throw QwenHybridModelReferenceError.emptyTokenIds
        }
        guard targetLayer >= 0, targetLayer < decoderLayers.count else {
            throw QwenHybridModelReferenceError.invalidCaptureLayer(targetLayer)
        }

        let initialTokenPosition = try caches.tokenPosition()
        diagnosticSink?("capture_embedding_begin tokens=\(tokenIds.count) position=\(initialTokenPosition) targetLayer=\(targetLayer)")
        var hiddenStates = try autoreleasepool {
            try embeddings.hiddenStates(
                tokenIds: tokenIds,
                executor: executor
            )
        }
        diagnosticSink?("capture_embedding_done shape=\(hiddenStates.shape.dimensions)")
        for layer in decoderLayers {
            diagnosticSink?("capture_layer_\(layer.layerIndex)_\(layer.kind.rawValue)_begin shape=\(hiddenStates.shape.dimensions)")
            let layerInput = hiddenStates
            hiddenStates = try autoreleasepool {
                try self.outputTensor(
                    for: layer,
                    hiddenStates: layerInput,
                    initialTokenPosition: initialTokenPosition,
                    caches: caches,
                    executor: executor,
                    diagnosticSink: diagnosticSink
                )
            }
            diagnosticSink?("capture_layer_\(layer.layerIndex)_\(layer.kind.rawValue)_done shape=\(hiddenStates.shape.dimensions)")
            if layer.layerIndex == targetLayer {
                let tokenCount = hiddenStates.shape.rank == 2 ? hiddenStates.shape.dimensions[0] : 0
                guard tokenCount > 0 else {
                    throw QwenHybridModelReferenceError.emptyTokenIds
                }
                if tokenCount == 1 {
                    return hiddenStates
                }
                return try autoreleasepool {
                    try executor.gatherRows(
                        source: hiddenStates,
                        rowIndices: [tokenCount - 1]
                    )
                }
            }
        }
        throw QwenHybridModelReferenceError.invalidCaptureLayer(targetLayer)
    }

    private func outputTensor(
        for layer: QwenHybridDecoderLayerReference,
        hiddenStates: EdgeTensor,
        initialTokenPosition: Int,
        caches: QwenHybridDecoderCaches,
        executor: MetalKernelExecutor,
        diagnosticSink: ((String) -> Void)?
    ) throws -> EdgeTensor {
        switch layer.kind {
        case .fullAttention:
            return try layer.outputTensor(
                hiddenStates: hiddenStates,
                executor: executor,
                positionOffset: initialTokenPosition,
                kvCache: caches.kvCache(layerIndex: layer.layerIndex),
                diagnosticSink: diagnosticSink
            )
        case .gdn:
            return try layer.outputTensor(
                hiddenStates: hiddenStates,
                executor: executor,
                gdnCache: caches.gdnCache(layerIndex: layer.layerIndex),
                diagnosticSink: diagnosticSink
            )
        }
    }

    public func logits(
        tokenIds: [Int],
        caches: QwenHybridDecoderCaches,
        executor: MetalKernelExecutor,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> EdgeTensor {
        let outputHiddenStates = try hiddenStates(
            tokenIds: tokenIds,
            caches: caches,
            executor: executor,
            diagnosticSink: diagnosticSink
        )
        diagnosticSink?("logits_head_begin shape=\(outputHiddenStates.shape.dimensions)")
        let logits = try autoreleasepool {
            try outputWeights.logits(
                hiddenStates: outputHiddenStates,
                executor: executor
            )
        }
        diagnosticSink?("logits_head_done shape=\(logits.shape.dimensions)")
        return logits
    }

    public func lastTokenLogits(
        tokenIds: [Int],
        caches: QwenHybridDecoderCaches,
        executor: MetalKernelExecutor,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> EdgeTensor {
        let outputHiddenStates = try hiddenStates(
            tokenIds: tokenIds,
            caches: caches,
            executor: executor,
            diagnosticSink: diagnosticSink
        )
        let tokenCount = outputHiddenStates.shape.rank == 2 ? outputHiddenStates.shape.dimensions[0] : 0
        let lastHiddenStates: EdgeTensor
        if tokenCount == 1 {
            lastHiddenStates = outputHiddenStates
        } else {
            diagnosticSink?("last_token_gather_begin shape=\(outputHiddenStates.shape.dimensions)")
            lastHiddenStates = try autoreleasepool {
                try executor.gatherRows(
                    source: outputHiddenStates,
                    rowIndices: [tokenCount - 1]
                )
            }
            diagnosticSink?("last_token_gather_done shape=\(lastHiddenStates.shape.dimensions)")
        }
        diagnosticSink?("logits_head_begin shape=\(lastHiddenStates.shape.dimensions)")
        let logits = try autoreleasepool {
            try outputWeights.logits(
                hiddenStates: lastHiddenStates,
                executor: executor
            )
        }
        diagnosticSink?("logits_head_done shape=\(logits.shape.dimensions)")
        return logits
    }

    public func lastTokenGreedyToken(
        tokenIds: [Int],
        caches: QwenHybridDecoderCaches,
        executor: MetalKernelExecutor,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> QwenGreedyToken {
        let outputHiddenStates = try hiddenStates(
            tokenIds: tokenIds,
            caches: caches,
            executor: executor,
            diagnosticSink: diagnosticSink
        )
        let tokenCount = outputHiddenStates.shape.rank == 2 ? outputHiddenStates.shape.dimensions[0] : 0
        let lastHiddenStates: EdgeTensor
        if tokenCount == 1 {
            lastHiddenStates = outputHiddenStates
        } else {
            diagnosticSink?("last_token_gather_begin shape=\(outputHiddenStates.shape.dimensions)")
            lastHiddenStates = try autoreleasepool {
                try executor.gatherRows(
                    source: outputHiddenStates,
                    rowIndices: [tokenCount - 1]
                )
            }
            diagnosticSink?("last_token_gather_done shape=\(lastHiddenStates.shape.dimensions)")
        }
        diagnosticSink?("greedy_output_head_begin shape=\(lastHiddenStates.shape.dimensions)")
        let token = try autoreleasepool {
            try outputWeights.greedyToken(
                hiddenStates: lastHiddenStates,
                executor: executor
            )
        }
        diagnosticSink?("greedy_output_head_done token=\(token.tokenId)")
        return token
    }
}
