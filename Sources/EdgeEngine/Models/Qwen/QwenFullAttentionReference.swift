// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public enum QwenFullAttentionReferenceError: Error, Equatable {
    case invalidLayerIndex(Int)
    case layerIsNotFullAttention(layerIndex: Int, kind: QwenHybridLayerKind)
    case invalidHiddenStateShape(expected: [Int], actual: [Int])
    case kvCacheLayerMismatch(expected: Int, actual: Int)
    case invalidKVCacheShape(expected: [Int], actual: [Int])
}

public struct QwenFullAttentionReference {
    public var architecture: QwenHybridArchitecture
    public var layerIndex: Int
    public var projectionWeights: QwenAttentionProjectionWeights
    public var normalizationWeights: QwenAttentionNormWeights?
    public var outputProjectionWeights: QwenAttentionOutputProjectionWeights?

    public init(
        architecture: QwenHybridArchitecture,
        layerIndex: Int,
        projectionWeights: QwenAttentionProjectionWeights,
        normalizationWeights: QwenAttentionNormWeights? = nil,
        outputProjectionWeights: QwenAttentionOutputProjectionWeights? = nil
    ) throws {
        guard layerIndex >= 0, layerIndex < architecture.layerPlan.count else {
            throw QwenFullAttentionReferenceError.invalidLayerIndex(layerIndex)
        }
        let layerKind = architecture.layerPlan[layerIndex].kind
        guard layerKind == .fullAttention else {
            throw QwenFullAttentionReferenceError.layerIsNotFullAttention(
                layerIndex: layerIndex,
                kind: layerKind
            )
        }
        self.architecture = architecture
        self.layerIndex = layerIndex
        self.projectionWeights = projectionWeights
        self.normalizationWeights = normalizationWeights
        self.outputProjectionWeights = outputProjectionWeights
    }

    public func attentionOutput(
        hiddenStates: EdgeTensor,
        executor: MetalKernelExecutor,
        positionOffset: Int = 0,
        kvCache: QwenKVCache? = nil
    ) throws -> [Float] {
        try attentionTensor(
            hiddenStates: hiddenStates,
            executor: executor,
            positionOffset: positionOffset,
            kvCache: kvCache
        ).readFloat32()
    }

    public func attentionTensor(
        hiddenStates: EdgeTensor,
        executor: MetalKernelExecutor,
        positionOffset: Int = 0,
        kvCache: QwenKVCache? = nil,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> EdgeTensor {
        guard hiddenStates.shape.rank == 2,
              hiddenStates.shape.dimensions[1] == architecture.hiddenSize
        else {
            throw QwenFullAttentionReferenceError.invalidHiddenStateShape(
                expected: [-1, architecture.hiddenSize],
                actual: hiddenStates.shape.dimensions
            )
        }
        if let kvCache {
            try validate(kvCache: kvCache)
        }

        diagnosticSink?("fa_\(layerIndex)_projection_begin")
        let projectionOutputs = try projectionWeights.project(
            hiddenStates: hiddenStates,
            executor: executor,
            diagnosticSink: diagnosticSink
        )
        diagnosticSink?("fa_\(layerIndex)_projection_done")
        let normalizedOutputs: (query: EdgeTensor, key: EdgeTensor)
        if let normalizationWeights {
            diagnosticSink?("fa_\(layerIndex)_qk_norm_begin")
            normalizedOutputs = try normalizationWeights.apply(
                query: projectionOutputs.query,
                key: projectionOutputs.key,
                architecture: architecture,
                executor: executor
            )
            diagnosticSink?("fa_\(layerIndex)_qk_norm_done")
        } else {
            normalizedOutputs = (projectionOutputs.query, projectionOutputs.key)
        }
        let effectivePositionOffset = kvCache?.tokenCount ?? positionOffset
        diagnosticSink?("fa_\(layerIndex)_rope_q_begin")
        let queries = try executor.applyRoPE(
            normalizedOutputs.query,
            headCount: architecture.attentionHeadCount,
            headDimension: architecture.attentionHeadDimension,
            rotaryDimension: architecture.rotaryDimension,
            base: architecture.ropeTheta,
            offset: effectivePositionOffset
        )
        diagnosticSink?("fa_\(layerIndex)_rope_q_done shape=\(queries.shape.dimensions)")
        diagnosticSink?("fa_\(layerIndex)_rope_k_begin")
        let keys = try executor.applyRoPE(
            normalizedOutputs.key,
            headCount: architecture.keyValueHeadCount,
            headDimension: architecture.attentionHeadDimension,
            rotaryDimension: architecture.rotaryDimension,
            base: architecture.ropeTheta,
            offset: effectivePositionOffset
        )
        diagnosticSink?("fa_\(layerIndex)_rope_k_done shape=\(keys.shape.dimensions)")
        let rawAttentionOutput: EdgeTensor
        if let kvCache {
            diagnosticSink?("fa_\(layerIndex)_kv_append_begin")
            try kvCache.append(keys: keys, values: projectionOutputs.value, executor: executor)
            diagnosticSink?("fa_\(layerIndex)_kv_append_done active=\(kvCache.activeTokenCount)")
            diagnosticSink?("fa_\(layerIndex)_sdpa_begin")
            rawAttentionOutput = try executor.scaledDotProductAttention(
                query: queries,
                key: kvCache.keys,
                value: kvCache.values,
                keyValueTokenCount: kvCache.activeTokenCount,
                queryPositionOffset: effectivePositionOffset,
                queryHeadCount: architecture.attentionHeadCount,
                keyValueHeadCount: architecture.keyValueHeadCount,
                headDimension: architecture.attentionHeadDimension
            )
            diagnosticSink?("fa_\(layerIndex)_sdpa_done shape=\(rawAttentionOutput.shape.dimensions)")
            diagnosticSink?("fa_\(layerIndex)_dsr_scores_begin")
            try kvCache.updateDSRScores(
                query: queries,
                executor: executor,
                queryHeadCount: architecture.attentionHeadCount,
                headDimension: architecture.attentionHeadDimension
            )
            diagnosticSink?("fa_\(layerIndex)_dsr_scores_done")
        } else {
            diagnosticSink?("fa_\(layerIndex)_sdpa_begin")
            rawAttentionOutput = try executor.scaledDotProductAttention(
                query: queries,
                key: keys,
                value: projectionOutputs.value,
                queryHeadCount: architecture.attentionHeadCount,
                keyValueHeadCount: architecture.keyValueHeadCount,
                headDimension: architecture.attentionHeadDimension
            )
            diagnosticSink?("fa_\(layerIndex)_sdpa_done shape=\(rawAttentionOutput.shape.dimensions)")
        }
        guard let outputProjectionWeights else {
            return rawAttentionOutput
        }

        return try outputProjectionWeights.project(
            attentionOutput: rawAttentionOutput,
            queryGate: projectionOutputs.queryGate,
            executor: executor,
            diagnosticSink: diagnosticSink
        )
    }

    private func validate(kvCache: QwenKVCache) throws {
        guard kvCache.shape.layerIndex == layerIndex else {
            throw QwenFullAttentionReferenceError.kvCacheLayerMismatch(
                expected: layerIndex,
                actual: kvCache.shape.layerIndex
            )
        }
        guard kvCache.shape.keyValueHeadCount == architecture.keyValueHeadCount,
              kvCache.shape.headDimension == architecture.attentionHeadDimension
        else {
            throw QwenFullAttentionReferenceError.invalidKVCacheShape(
                expected: [architecture.keyValueHeadCount, architecture.attentionHeadDimension],
                actual: [kvCache.shape.keyValueHeadCount, kvCache.shape.headDimension]
            )
        }
    }
}

public struct QwenQuantizedFullAttentionReference {
    public var architecture: QwenHybridArchitecture
    public var layerIndex: Int
    public var projectionWeights: QwenQuantizedAttentionProjectionWeights
    public var normalizationWeights: QwenAttentionNormWeights?
    public var outputProjectionWeights: QwenQuantizedAttentionOutputProjectionWeights?

    public init(
        architecture: QwenHybridArchitecture,
        layerIndex: Int,
        projectionWeights: QwenQuantizedAttentionProjectionWeights,
        normalizationWeights: QwenAttentionNormWeights? = nil,
        outputProjectionWeights: QwenQuantizedAttentionOutputProjectionWeights? = nil
    ) throws {
        guard layerIndex >= 0, layerIndex < architecture.layerPlan.count else {
            throw QwenFullAttentionReferenceError.invalidLayerIndex(layerIndex)
        }
        let layerKind = architecture.layerPlan[layerIndex].kind
        guard layerKind == .fullAttention else {
            throw QwenFullAttentionReferenceError.layerIsNotFullAttention(
                layerIndex: layerIndex,
                kind: layerKind
            )
        }
        self.architecture = architecture
        self.layerIndex = layerIndex
        self.projectionWeights = projectionWeights
        self.normalizationWeights = normalizationWeights
        self.outputProjectionWeights = outputProjectionWeights
    }

    public static func loadHuggingFaceLayout(
        layerIndex: Int,
        weightStore: QwenModelWeightStore,
        runtime: EdgeMetalRuntime
    ) throws -> QwenQuantizedFullAttentionReference {
        let architecture = weightStore.bundleIndex.architecture
        return try QwenQuantizedFullAttentionReference(
            architecture: architecture,
            layerIndex: layerIndex,
            projectionWeights: QwenQuantizedAttentionProjectionWeights.loadHuggingFaceLayout(
                layerIndex: layerIndex,
                weightStore: weightStore
            ),
            normalizationWeights: QwenAttentionNormWeights.loadHuggingFaceLayout(
                layerIndex: layerIndex,
                weightStore: weightStore,
                runtime: runtime
            ),
            outputProjectionWeights: QwenQuantizedAttentionOutputProjectionWeights.loadHuggingFaceLayout(
                layerIndex: layerIndex,
                weightStore: weightStore
            )
        )
    }

    public func attentionOutput(
        hiddenStates: EdgeTensor,
        executor: MetalKernelExecutor,
        positionOffset: Int = 0,
        kvCache: QwenKVCache? = nil
    ) throws -> [Float] {
        try attentionTensor(
            hiddenStates: hiddenStates,
            executor: executor,
            positionOffset: positionOffset,
            kvCache: kvCache
        ).readFloat32()
    }

    public func attentionTensor(
        hiddenStates: EdgeTensor,
        executor: MetalKernelExecutor,
        positionOffset: Int = 0,
        kvCache: QwenKVCache? = nil,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> EdgeTensor {
        guard hiddenStates.shape.rank == 2,
              hiddenStates.shape.dimensions[1] == architecture.hiddenSize
        else {
            throw QwenFullAttentionReferenceError.invalidHiddenStateShape(
                expected: [-1, architecture.hiddenSize],
                actual: hiddenStates.shape.dimensions
            )
        }
        if let kvCache {
            try validate(kvCache: kvCache)
        }

        diagnosticSink?("fa_\(layerIndex)_projection_begin")
        let projectionOutputs = try projectionWeights.project(
            hiddenStates: hiddenStates,
            executor: executor,
            diagnosticSink: diagnosticSink
        )
        diagnosticSink?("fa_\(layerIndex)_projection_done")
        let normalizedOutputs: (query: EdgeTensor, key: EdgeTensor)
        if let normalizationWeights {
            diagnosticSink?("fa_\(layerIndex)_qk_norm_begin")
            normalizedOutputs = try normalizationWeights.apply(
                query: projectionOutputs.query,
                key: projectionOutputs.key,
                architecture: architecture,
                executor: executor
            )
            diagnosticSink?("fa_\(layerIndex)_qk_norm_done")
        } else {
            normalizedOutputs = (projectionOutputs.query, projectionOutputs.key)
        }
        let effectivePositionOffset = kvCache?.tokenCount ?? positionOffset
        diagnosticSink?("fa_\(layerIndex)_rope_q_begin")
        let queries = try executor.applyRoPE(
            normalizedOutputs.query,
            headCount: architecture.attentionHeadCount,
            headDimension: architecture.attentionHeadDimension,
            rotaryDimension: architecture.rotaryDimension,
            base: architecture.ropeTheta,
            offset: effectivePositionOffset
        )
        diagnosticSink?("fa_\(layerIndex)_rope_q_done shape=\(queries.shape.dimensions)")
        diagnosticSink?("fa_\(layerIndex)_rope_k_begin")
        let keys = try executor.applyRoPE(
            normalizedOutputs.key,
            headCount: architecture.keyValueHeadCount,
            headDimension: architecture.attentionHeadDimension,
            rotaryDimension: architecture.rotaryDimension,
            base: architecture.ropeTheta,
            offset: effectivePositionOffset
        )
        diagnosticSink?("fa_\(layerIndex)_rope_k_done shape=\(keys.shape.dimensions)")
        let rawAttentionOutput: EdgeTensor
        if let kvCache {
            diagnosticSink?("fa_\(layerIndex)_kv_append_begin")
            try kvCache.append(keys: keys, values: projectionOutputs.value, executor: executor)
            diagnosticSink?("fa_\(layerIndex)_kv_append_done active=\(kvCache.activeTokenCount)")
            diagnosticSink?("fa_\(layerIndex)_sdpa_begin")
            rawAttentionOutput = try executor.scaledDotProductAttention(
                query: queries,
                key: kvCache.keys,
                value: kvCache.values,
                keyValueTokenCount: kvCache.activeTokenCount,
                queryPositionOffset: effectivePositionOffset,
                queryHeadCount: architecture.attentionHeadCount,
                keyValueHeadCount: architecture.keyValueHeadCount,
                headDimension: architecture.attentionHeadDimension
            )
            diagnosticSink?("fa_\(layerIndex)_sdpa_done shape=\(rawAttentionOutput.shape.dimensions)")
            diagnosticSink?("fa_\(layerIndex)_dsr_scores_begin")
            try kvCache.updateDSRScores(
                query: queries,
                executor: executor,
                queryHeadCount: architecture.attentionHeadCount,
                headDimension: architecture.attentionHeadDimension
            )
            diagnosticSink?("fa_\(layerIndex)_dsr_scores_done")
        } else {
            diagnosticSink?("fa_\(layerIndex)_sdpa_begin")
            rawAttentionOutput = try executor.scaledDotProductAttention(
                query: queries,
                key: keys,
                value: projectionOutputs.value,
                queryHeadCount: architecture.attentionHeadCount,
                keyValueHeadCount: architecture.keyValueHeadCount,
                headDimension: architecture.attentionHeadDimension
            )
            diagnosticSink?("fa_\(layerIndex)_sdpa_done shape=\(rawAttentionOutput.shape.dimensions)")
        }
        guard let outputProjectionWeights else {
            return rawAttentionOutput
        }

        return try outputProjectionWeights.project(
            attentionOutput: rawAttentionOutput,
            queryGate: projectionOutputs.queryGate,
            executor: executor,
            diagnosticSink: diagnosticSink
        )
    }

    private func validate(kvCache: QwenKVCache) throws {
        guard kvCache.shape.layerIndex == layerIndex else {
            throw QwenFullAttentionReferenceError.kvCacheLayerMismatch(
                expected: layerIndex,
                actual: kvCache.shape.layerIndex
            )
        }
        guard kvCache.shape.keyValueHeadCount == architecture.keyValueHeadCount,
              kvCache.shape.headDimension == architecture.attentionHeadDimension
        else {
            throw QwenFullAttentionReferenceError.invalidKVCacheShape(
                expected: [architecture.keyValueHeadCount, architecture.attentionHeadDimension],
                actual: [kvCache.shape.keyValueHeadCount, kvCache.shape.headDimension]
            )
        }
    }
}
