// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public enum QwenHybridDecoderLayerError: Error, Equatable {
    case missingGDNCache(layerIndex: Int)
    case unexpectedKVCache(layerIndex: Int)
    case unexpectedGDNCache(layerIndex: Int)
    case gdnCacheLayerMismatch(expected: Int, actual: Int)
    case invalidHiddenStateShape(expected: [Int], actual: [Int])
}

public enum QwenHybridDecoderLayerReference {
    case fullAttention(QwenFullAttentionDecoderLayerReference)
    case quantizedFullAttention(QwenQuantizedFullAttentionDecoderLayerReference)
    case gdn(QwenGDNDecoderLayerReference)
    case quantizedGDN(QwenQuantizedGDNDecoderLayerReference)

    public var layerIndex: Int {
        switch self {
        case .fullAttention(let layer):
            return layer.attention.layerIndex
        case .quantizedFullAttention(let layer):
            return layer.attention.layerIndex
        case .gdn(let layer):
            return layer.linearAttention.layerIndex
        case .quantizedGDN(let layer):
            return layer.linearAttention.layerIndex
        }
    }

    public var kind: QwenHybridLayerKind {
        switch self {
        case .fullAttention, .quantizedFullAttention:
            return .fullAttention
        case .gdn, .quantizedGDN:
            return .gdn
        }
    }

    public static func loadHuggingFaceLayout(
        layerIndex: Int,
        weightStore: QwenModelWeightStore,
        runtime: EdgeMetalRuntime
    ) throws -> QwenHybridDecoderLayerReference {
        let architecture = weightStore.bundleIndex.architecture
        let manifest = try weightStore.bundleIndex.layerManifest(layerIndex)
        let useQuantizedWeights = architecture.quantization != nil
        switch manifest.kind {
        case .fullAttention:
            if useQuantizedWeights {
                return .quantizedFullAttention(
                    try QwenQuantizedFullAttentionDecoderLayerReference.loadHuggingFaceLayout(
                        layerIndex: layerIndex,
                        weightStore: weightStore,
                        runtime: runtime
                    )
                )
            }
            return .fullAttention(
                try QwenFullAttentionDecoderLayerReference.loadHuggingFaceLayout(
                    layerIndex: layerIndex,
                    weightStore: weightStore,
                    runtime: runtime
                )
            )
        case .gdn:
            if useQuantizedWeights {
                return .quantizedGDN(
                    try QwenQuantizedGDNDecoderLayerReference.loadHuggingFaceLayout(
                        layerIndex: layerIndex,
                        weightStore: weightStore,
                        runtime: runtime
                    )
                )
            }
            return .gdn(
                try QwenGDNDecoderLayerReference.loadHuggingFaceLayout(
                    layerIndex: layerIndex,
                    weightStore: weightStore,
                    runtime: runtime
                )
            )
        }
    }

    public func outputTensor(
        hiddenStates: EdgeTensor,
        executor: MetalKernelExecutor,
        positionOffset: Int = 0,
        kvCache: QwenKVCache? = nil,
        gdnCache: QwenGDNCache? = nil,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> EdgeTensor {
        switch self {
        case .fullAttention(let layer):
            guard gdnCache == nil else {
                throw QwenHybridDecoderLayerError.unexpectedGDNCache(layerIndex: layerIndex)
            }
            return try layer.outputTensor(
                hiddenStates: hiddenStates,
                executor: executor,
                positionOffset: positionOffset,
                kvCache: kvCache,
                diagnosticSink: diagnosticSink
            )

        case .quantizedFullAttention(let layer):
            guard gdnCache == nil else {
                throw QwenHybridDecoderLayerError.unexpectedGDNCache(layerIndex: layerIndex)
            }
            return try layer.outputTensor(
                hiddenStates: hiddenStates,
                executor: executor,
                positionOffset: positionOffset,
                kvCache: kvCache,
                diagnosticSink: diagnosticSink
            )

        case .gdn(let layer):
            guard kvCache == nil else {
                throw QwenHybridDecoderLayerError.unexpectedKVCache(layerIndex: layerIndex)
            }
            guard let gdnCache else {
                throw QwenHybridDecoderLayerError.missingGDNCache(layerIndex: layerIndex)
            }
            guard gdnCache.shape.layerIndex == layerIndex else {
                throw QwenHybridDecoderLayerError.gdnCacheLayerMismatch(
                    expected: layerIndex,
                    actual: gdnCache.shape.layerIndex
                )
            }
            let expectedHiddenSize = layer.linearAttention.inProjQKV.shape.dimensions[0]
            guard hiddenStates.shape.rank == 2,
                  hiddenStates.shape.dimensions[1] == expectedHiddenSize
            else {
                throw QwenHybridDecoderLayerError.invalidHiddenStateShape(
                    expected: [-1, expectedHiddenSize],
                    actual: hiddenStates.shape.dimensions
                )
            }

            let outputs = try layer.outputTensor(
                hiddenStates: hiddenStates,
                convState: gdnCache.convState,
                recurrentState: gdnCache.recurrentState,
                executor: executor,
                diagnosticSink: diagnosticSink
            )
            diagnosticSink?("layer_\(layerIndex)_gdn_cache_update_begin")
            try gdnCache.update(
                nextConvState: outputs.nextConvState,
                nextRecurrentState: outputs.nextRecurrentState,
                tokenCount: hiddenStates.shape.dimensions[0],
                executor: executor
            )
            diagnosticSink?("layer_\(layerIndex)_gdn_cache_update_done")
            return outputs.hiddenStates

        case .quantizedGDN(let layer):
            guard kvCache == nil else {
                throw QwenHybridDecoderLayerError.unexpectedKVCache(layerIndex: layerIndex)
            }
            guard let gdnCache else {
                throw QwenHybridDecoderLayerError.missingGDNCache(layerIndex: layerIndex)
            }
            guard gdnCache.shape.layerIndex == layerIndex else {
                throw QwenHybridDecoderLayerError.gdnCacheLayerMismatch(
                    expected: layerIndex,
                    actual: gdnCache.shape.layerIndex
                )
            }
            let expectedHiddenSize = layer.linearAttention.inProjQKV.shape[1]
            guard hiddenStates.shape.rank == 2,
                  hiddenStates.shape.dimensions[1] == expectedHiddenSize
            else {
                throw QwenHybridDecoderLayerError.invalidHiddenStateShape(
                    expected: [-1, expectedHiddenSize],
                    actual: hiddenStates.shape.dimensions
                )
            }

            let outputs = try layer.outputTensor(
                hiddenStates: hiddenStates,
                convState: gdnCache.convState,
                recurrentState: gdnCache.recurrentState,
                executor: executor,
                diagnosticSink: diagnosticSink
            )
            diagnosticSink?("layer_\(layerIndex)_gdn_cache_update_begin")
            try gdnCache.update(
                nextConvState: outputs.nextConvState,
                nextRecurrentState: outputs.nextRecurrentState,
                tokenCount: hiddenStates.shape.dimensions[0],
                executor: executor
            )
            diagnosticSink?("layer_\(layerIndex)_gdn_cache_update_done")
            return outputs.hiddenStates
        }
    }
}
