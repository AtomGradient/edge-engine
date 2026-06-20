// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public enum QwenAttentionNormWeightError: Error, Equatable {
    case invalidWeightShape(name: String, expected: [Int], actual: [Int])
    case layerIsNotFullAttention(layerIndex: Int, kind: QwenHybridLayerKind)
}

public struct QwenAttentionNormWeights {
    public var layerIndex: Int
    public var query: EdgeTensor
    public var key: EdgeTensor

    public init(layerIndex: Int, query: EdgeTensor, key: EdgeTensor) {
        self.layerIndex = layerIndex
        self.query = query
        self.key = key
    }

    public static func loadRuntimeLayout(
        layerIndex: Int,
        architecture: QwenHybridArchitecture,
        weights: SafeTensorsFile,
        runtime: EdgeMetalRuntime
    ) throws -> QwenAttentionNormWeights {
        let prefix = "model.layers.\(layerIndex).self_attn"
        let query = try weights.loadFloat32Tensor(named: "\(prefix).q_norm.weight", runtime: runtime)
        let key = try weights.loadFloat32Tensor(named: "\(prefix).k_norm.weight", runtime: runtime)

        try validate(query, name: "\(prefix).q_norm.weight", architecture: architecture)
        try validate(key, name: "\(prefix).k_norm.weight", architecture: architecture)

        return QwenAttentionNormWeights(layerIndex: layerIndex, query: query, key: key)
    }

    public static func loadHuggingFaceLayout(
        layerIndex: Int,
        architecture: QwenHybridArchitecture,
        weights: SafeTensorsFile,
        runtime: EdgeMetalRuntime
    ) throws -> QwenAttentionNormWeights {
        try loadRuntimeLayout(
            layerIndex: layerIndex,
            architecture: architecture,
            weights: weights,
            runtime: runtime
        )
    }

    public static func loadHuggingFaceLayout(
        layerIndex: Int,
        weightStore: QwenModelWeightStore,
        runtime: EdgeMetalRuntime
    ) throws -> QwenAttentionNormWeights {
        let architecture = weightStore.bundleIndex.architecture
        let manifest = try weightStore.bundleIndex.layerManifest(layerIndex)
        guard manifest.kind == .fullAttention else {
            throw QwenAttentionNormWeightError.layerIsNotFullAttention(
                layerIndex: layerIndex,
                kind: manifest.kind
            )
        }
        let prefix = "\(manifest.layerPrefix).self_attn"
        let query = try weightStore.loadFloat32Tensor(named: "\(prefix).q_norm.weight", runtime: runtime)
        let key = try weightStore.loadFloat32Tensor(named: "\(prefix).k_norm.weight", runtime: runtime)

        try validate(query, name: "\(prefix).q_norm.weight", architecture: architecture)
        try validate(key, name: "\(prefix).k_norm.weight", architecture: architecture)

        return QwenAttentionNormWeights(layerIndex: layerIndex, query: query, key: key)
    }

    public func apply(
        query queryTensor: EdgeTensor,
        key keyTensor: EdgeTensor,
        architecture: QwenHybridArchitecture,
        executor: MetalKernelExecutor
    ) throws -> (query: EdgeTensor, key: EdgeTensor) {
        let query = try executor.rmsNormByHead(
            queryTensor,
            weight: query,
            headCount: architecture.attentionHeadCount,
            headDimension: architecture.attentionHeadDimension,
            epsilon: architecture.rmsNormEpsilon
        )
        let key = try executor.rmsNormByHead(
            keyTensor,
            weight: key,
            headCount: architecture.keyValueHeadCount,
            headDimension: architecture.attentionHeadDimension,
            epsilon: architecture.rmsNormEpsilon
        )
        return (query, key)
    }

    private static func validate(
        _ tensor: EdgeTensor,
        name: String,
        architecture: QwenHybridArchitecture
    ) throws {
        let expectedShape = [architecture.attentionHeadDimension]
        guard tensor.shape.dimensions == expectedShape else {
            throw QwenAttentionNormWeightError.invalidWeightShape(
                name: name,
                expected: expectedShape,
                actual: tensor.shape.dimensions
            )
        }
    }
}
