// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public enum QwenProjectionWeightError: Error, Equatable {
    case invalidWeightShape(name: String, expected: [Int], actual: [Int])
    case layerIsNotFullAttention(layerIndex: Int, kind: QwenHybridLayerKind)
    case layerIsNotGDN(layerIndex: Int, kind: QwenHybridLayerKind)
    case missingQuantizedCompanion(weightName: String)
    case missingQuantizationProfile
}

public struct QwenAttentionProjectionOutputs {
    public var query: EdgeTensor
    /// Non-nil for the gated Qwen3.5/Qwen3.6 attention variants currently supported.
    public var queryGate: EdgeTensor?
    public var key: EdgeTensor
    public var value: EdgeTensor

    public init(query: EdgeTensor, key: EdgeTensor, value: EdgeTensor, queryGate: EdgeTensor? = nil) {
        self.query = query
        self.queryGate = queryGate
        self.key = key
        self.value = value
    }
}

public struct QwenQuantizedAttentionProjectionWeights {
    public var layerIndex: Int
    public var queryHiddenSize: Int
    public var queryHeadCount: Int
    public var queryHeadDimension: Int
    public var query: EdgeQuantizedTensor
    public var key: EdgeQuantizedTensor
    public var value: EdgeQuantizedTensor

    public init(
        layerIndex: Int,
        query: EdgeQuantizedTensor,
        key: EdgeQuantizedTensor,
        value: EdgeQuantizedTensor,
        queryHiddenSize: Int? = nil,
        queryHeadCount: Int? = nil,
        queryHeadDimension: Int? = nil
    ) {
        self.layerIndex = layerIndex
        self.queryHiddenSize = queryHiddenSize ?? query.shape[0] / 2
        self.queryHeadCount = queryHeadCount ?? 1
        self.queryHeadDimension = queryHeadDimension ?? self.queryHiddenSize
        self.query = query
        self.key = key
        self.value = value
    }

    public static func loadHuggingFaceLayout(
        layerIndex: Int,
        weightStore: QwenModelWeightStore,
        groupSize: Int? = nil,
        bits: Int? = nil
    ) throws -> QwenQuantizedAttentionProjectionWeights {
        let architecture = weightStore.bundleIndex.architecture
        let quantization = try resolveQuantization(
            architecture: architecture,
            groupSize: groupSize,
            bits: bits
        )
        let manifest = try weightStore.bundleIndex.layerManifest(layerIndex)
        guard manifest.kind == .fullAttention else {
            throw QwenProjectionWeightError.layerIsNotFullAttention(
                layerIndex: layerIndex,
                kind: manifest.kind
            )
        }
        let prefix = "\(manifest.layerPrefix).self_attn"
        let queryName = "\(prefix).q_proj.weight"
        let keyName = "\(prefix).k_proj.weight"
        let valueName = "\(prefix).v_proj.weight"

        let query = try loadQuantizedWeight(
            queryName,
            manifest: manifest,
            weightStore: weightStore,
            groupSize: quantization.groupSize,
            bits: quantization.bits
        )
        let key = try loadQuantizedWeight(
            keyName,
            manifest: manifest,
            weightStore: weightStore,
            groupSize: quantization.groupSize,
            bits: quantization.bits
        )
        let value = try loadQuantizedWeight(
            valueName,
            manifest: manifest,
            weightStore: weightStore,
            groupSize: quantization.groupSize,
            bits: quantization.bits
        )

        try validate(query, name: queryName, expectedShape: [
            architecture.queryProjectionHiddenSize,
            architecture.hiddenSize,
        ])
        try validate(key, name: keyName, expectedShape: [
            architecture.keyValueHiddenSize,
            architecture.hiddenSize,
        ])
        try validate(value, name: valueName, expectedShape: [
            architecture.keyValueHiddenSize,
            architecture.hiddenSize,
        ])

        return QwenQuantizedAttentionProjectionWeights(
            layerIndex: layerIndex,
            query: query,
            key: key,
            value: value,
            queryHiddenSize: architecture.queryHiddenSize,
            queryHeadCount: architecture.attentionHeadCount,
            queryHeadDimension: architecture.attentionHeadDimension
        )
    }

    public func project(
        hiddenStates: EdgeTensor,
        executor: MetalKernelExecutor,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> QwenAttentionProjectionOutputs {
        diagnosticSink?("fa_\(layerIndex)_q_proj_begin")
        let queryProjection = try executor.affineQuantizedMatmul(
            hiddenStates,
            weights: query,
            transpose: true,
            diagnosticSink: diagnosticSink,
            diagnosticName: "fa_\(layerIndex)_q_proj_affine"
        )
        diagnosticSink?("fa_\(layerIndex)_q_proj_done shape=\(queryProjection.shape.dimensions)")
        diagnosticSink?("fa_\(layerIndex)_q_split_begin")
        let querySplit = try executor.splitGatedQuery(
            queryProjection,
            headCount: queryHeadCount,
            headDimension: queryHeadDimension
        )
        diagnosticSink?("fa_\(layerIndex)_q_split_done")
        diagnosticSink?("fa_\(layerIndex)_k_proj_begin")
        let keyProjection = try executor.affineQuantizedMatmul(
            hiddenStates,
            weights: key,
            transpose: true,
            diagnosticSink: diagnosticSink,
            diagnosticName: "fa_\(layerIndex)_k_proj_affine"
        )
        diagnosticSink?("fa_\(layerIndex)_k_proj_done shape=\(keyProjection.shape.dimensions)")
        diagnosticSink?("fa_\(layerIndex)_v_proj_begin")
        let valueProjection = try executor.affineQuantizedMatmul(
            hiddenStates,
            weights: value,
            transpose: true,
            diagnosticSink: diagnosticSink,
            diagnosticName: "fa_\(layerIndex)_v_proj_affine"
        )
        diagnosticSink?("fa_\(layerIndex)_v_proj_done shape=\(valueProjection.shape.dimensions)")
        return QwenAttentionProjectionOutputs(
            query: querySplit.query,
            key: keyProjection,
            value: valueProjection,
            queryGate: querySplit.gate
        )
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
        manifest: QwenLayerWeightManifest,
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
        _ tensor: EdgeQuantizedTensor,
        name: String,
        expectedShape: [Int]
    ) throws {
        guard tensor.shape == expectedShape else {
            throw QwenProjectionWeightError.invalidWeightShape(
                name: name,
                expected: expectedShape,
                actual: tensor.shape
            )
        }
    }
}

public struct QwenAttentionProjectionWeights {
    public var layerIndex: Int
    public var queryHiddenSize: Int
    public var queryHeadCount: Int
    public var queryHeadDimension: Int
    public var query: EdgeTensor
    public var key: EdgeTensor
    public var value: EdgeTensor

    public init(
        layerIndex: Int,
        query: EdgeTensor,
        key: EdgeTensor,
        value: EdgeTensor,
        queryHiddenSize: Int? = nil,
        queryHeadCount: Int? = nil,
        queryHeadDimension: Int? = nil
    ) {
        self.layerIndex = layerIndex
        self.queryHiddenSize = queryHiddenSize ?? query.shape.dimensions[1] / 2
        self.queryHeadCount = queryHeadCount ?? 1
        self.queryHeadDimension = queryHeadDimension ?? self.queryHiddenSize
        self.query = query
        self.key = key
        self.value = value
    }

    public static func loadRuntimeLayout(
        layerIndex: Int,
        architecture: QwenHybridArchitecture,
        weights: SafeTensorsFile,
        runtime: EdgeMetalRuntime
    ) throws -> QwenAttentionProjectionWeights {
        let prefix = "model.layers.\(layerIndex).self_attn"
        let query = try weights.loadFloat32Tensor(named: "\(prefix).q_proj.weight", runtime: runtime)
        let key = try weights.loadFloat32Tensor(named: "\(prefix).k_proj.weight", runtime: runtime)
        let value = try weights.loadFloat32Tensor(named: "\(prefix).v_proj.weight", runtime: runtime)

        try validate(query, name: "\(prefix).q_proj.weight", expectedShape: [
            architecture.hiddenSize,
            architecture.queryProjectionHiddenSize,
        ])
        try validate(key, name: "\(prefix).k_proj.weight", expectedShape: [
            architecture.hiddenSize,
            architecture.keyValueHiddenSize,
        ])
        try validate(value, name: "\(prefix).v_proj.weight", expectedShape: [
            architecture.hiddenSize,
            architecture.keyValueHiddenSize,
        ])

        return QwenAttentionProjectionWeights(
            layerIndex: layerIndex,
            query: query,
            key: key,
            value: value,
            queryHiddenSize: architecture.queryHiddenSize,
            queryHeadCount: architecture.attentionHeadCount,
            queryHeadDimension: architecture.attentionHeadDimension
        )
    }

    public static func loadHuggingFaceLayout(
        layerIndex: Int,
        architecture: QwenHybridArchitecture,
        weights: SafeTensorsFile,
        runtime: EdgeMetalRuntime
    ) throws -> QwenAttentionProjectionWeights {
        let prefix = "model.layers.\(layerIndex).self_attn"
        let query = try weights.loadFloat32TensorTransposed2D(
            named: "\(prefix).q_proj.weight",
            runtime: runtime
        )
        let key = try weights.loadFloat32TensorTransposed2D(
            named: "\(prefix).k_proj.weight",
            runtime: runtime
        )
        let value = try weights.loadFloat32TensorTransposed2D(
            named: "\(prefix).v_proj.weight",
            runtime: runtime
        )

        try validate(query, name: "\(prefix).q_proj.weight", expectedShape: [
            architecture.hiddenSize,
            architecture.queryProjectionHiddenSize,
        ])
        try validate(key, name: "\(prefix).k_proj.weight", expectedShape: [
            architecture.hiddenSize,
            architecture.keyValueHiddenSize,
        ])
        try validate(value, name: "\(prefix).v_proj.weight", expectedShape: [
            architecture.hiddenSize,
            architecture.keyValueHiddenSize,
        ])

        return QwenAttentionProjectionWeights(
            layerIndex: layerIndex,
            query: query,
            key: key,
            value: value,
            queryHiddenSize: architecture.queryHiddenSize,
            queryHeadCount: architecture.attentionHeadCount,
            queryHeadDimension: architecture.attentionHeadDimension
        )
    }

    public static func loadHuggingFaceLayout(
        layerIndex: Int,
        weightStore: QwenModelWeightStore,
        runtime: EdgeMetalRuntime
    ) throws -> QwenAttentionProjectionWeights {
        let architecture = weightStore.bundleIndex.architecture
        let manifest = try weightStore.bundleIndex.layerManifest(layerIndex)
        guard manifest.kind == .fullAttention else {
            throw QwenProjectionWeightError.layerIsNotFullAttention(
                layerIndex: layerIndex,
                kind: manifest.kind
            )
        }
        let prefix = "\(manifest.layerPrefix).self_attn"
        let query = try weightStore.loadFloat32TensorTransposed2D(
            named: "\(prefix).q_proj.weight",
            runtime: runtime
        )
        let key = try weightStore.loadFloat32TensorTransposed2D(
            named: "\(prefix).k_proj.weight",
            runtime: runtime
        )
        let value = try weightStore.loadFloat32TensorTransposed2D(
            named: "\(prefix).v_proj.weight",
            runtime: runtime
        )

        try validate(query, name: "\(prefix).q_proj.weight", expectedShape: [
            architecture.hiddenSize,
            architecture.queryProjectionHiddenSize,
        ])
        try validate(key, name: "\(prefix).k_proj.weight", expectedShape: [
            architecture.hiddenSize,
            architecture.keyValueHiddenSize,
        ])
        try validate(value, name: "\(prefix).v_proj.weight", expectedShape: [
            architecture.hiddenSize,
            architecture.keyValueHiddenSize,
        ])

        return QwenAttentionProjectionWeights(
            layerIndex: layerIndex,
            query: query,
            key: key,
            value: value,
            queryHiddenSize: architecture.queryHiddenSize,
            queryHeadCount: architecture.attentionHeadCount,
            queryHeadDimension: architecture.attentionHeadDimension
        )
    }

    public func project(
        hiddenStates: EdgeTensor,
        executor: MetalKernelExecutor,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> QwenAttentionProjectionOutputs {
        diagnosticSink?("fa_\(layerIndex)_q_proj_begin")
        let queryProjection = try executor.matmul(hiddenStates, query)
        diagnosticSink?("fa_\(layerIndex)_q_proj_done shape=\(queryProjection.shape.dimensions)")
        diagnosticSink?("fa_\(layerIndex)_q_split_begin")
        let querySplit = try executor.splitGatedQuery(
            queryProjection,
            headCount: queryHeadCount,
            headDimension: queryHeadDimension
        )
        diagnosticSink?("fa_\(layerIndex)_q_split_done")
        diagnosticSink?("fa_\(layerIndex)_k_proj_begin")
        let keyProjection = try executor.matmul(hiddenStates, key)
        diagnosticSink?("fa_\(layerIndex)_k_proj_done shape=\(keyProjection.shape.dimensions)")
        diagnosticSink?("fa_\(layerIndex)_v_proj_begin")
        let valueProjection = try executor.matmul(hiddenStates, value)
        diagnosticSink?("fa_\(layerIndex)_v_proj_done shape=\(valueProjection.shape.dimensions)")
        return QwenAttentionProjectionOutputs(
            query: querySplit.query,
            key: keyProjection,
            value: valueProjection,
            queryGate: querySplit.gate
        )
    }

    private static func validate(
        _ tensor: EdgeTensor,
        name: String,
        expectedShape: [Int]
    ) throws {
        guard tensor.shape.dimensions == expectedShape else {
            throw QwenProjectionWeightError.invalidWeightShape(
                name: name,
                expected: expectedShape,
                actual: tensor.shape.dimensions
            )
        }
    }
}
