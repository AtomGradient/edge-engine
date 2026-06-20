// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public struct QwenAttentionOutputProjectionWeights {
    public var layerIndex: Int
    public var weight: EdgeTensor

    public init(layerIndex: Int, weight: EdgeTensor) {
        self.layerIndex = layerIndex
        self.weight = weight
    }

    public static func loadRuntimeLayout(
        layerIndex: Int,
        architecture: QwenHybridArchitecture,
        weights: SafeTensorsFile,
        runtime: EdgeMetalRuntime
    ) throws -> QwenAttentionOutputProjectionWeights {
        let name = "model.layers.\(layerIndex).self_attn.o_proj.weight"
        let weight = try weights.loadFloat32Tensor(named: name, runtime: runtime)
        try validate(weight, name: name, architecture: architecture)
        return QwenAttentionOutputProjectionWeights(layerIndex: layerIndex, weight: weight)
    }

    public static func loadHuggingFaceLayout(
        layerIndex: Int,
        architecture: QwenHybridArchitecture,
        weights: SafeTensorsFile,
        runtime: EdgeMetalRuntime
    ) throws -> QwenAttentionOutputProjectionWeights {
        let name = "model.layers.\(layerIndex).self_attn.o_proj.weight"
        let weight = try weights.loadFloat32TensorTransposed2D(named: name, runtime: runtime)
        try validate(weight, name: name, architecture: architecture)
        return QwenAttentionOutputProjectionWeights(layerIndex: layerIndex, weight: weight)
    }

    public static func loadHuggingFaceLayout(
        layerIndex: Int,
        weightStore: QwenModelWeightStore,
        runtime: EdgeMetalRuntime
    ) throws -> QwenAttentionOutputProjectionWeights {
        let architecture = weightStore.bundleIndex.architecture
        let manifest = try weightStore.bundleIndex.layerManifest(layerIndex)
        guard manifest.kind == .fullAttention else {
            throw QwenProjectionWeightError.layerIsNotFullAttention(
                layerIndex: layerIndex,
                kind: manifest.kind
            )
        }
        let name = "\(manifest.layerPrefix).self_attn.o_proj.weight"
        let weight = try weightStore.loadFloat32TensorTransposed2D(named: name, runtime: runtime)
        try validate(weight, name: name, architecture: architecture)
        return QwenAttentionOutputProjectionWeights(layerIndex: layerIndex, weight: weight)
    }

    public func project(
        attentionOutput: EdgeTensor,
        queryGate: EdgeTensor?,
        executor: MetalKernelExecutor,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> EdgeTensor {
        let gatedOutput: EdgeTensor
        if let queryGate {
            diagnosticSink?("fa_\(layerIndex)_o_gate_begin")
            gatedOutput = try executor.sigmoidMultiply(attentionOutput, gate: queryGate)
            diagnosticSink?("fa_\(layerIndex)_o_gate_done shape=\(gatedOutput.shape.dimensions)")
        } else {
            gatedOutput = attentionOutput
        }
        diagnosticSink?("fa_\(layerIndex)_o_proj_begin")
        let output = try executor.matmul(gatedOutput, weight)
        diagnosticSink?("fa_\(layerIndex)_o_proj_done shape=\(output.shape.dimensions)")
        return output
    }

    private static func validate(
        _ tensor: EdgeTensor,
        name: String,
        architecture: QwenHybridArchitecture
    ) throws {
        let expectedShape = [architecture.attentionHiddenSize, architecture.hiddenSize]
        guard tensor.shape.dimensions == expectedShape else {
            throw QwenProjectionWeightError.invalidWeightShape(
                name: name,
                expected: expectedShape,
                actual: tensor.shape.dimensions
            )
        }
    }
}

public struct QwenQuantizedAttentionOutputProjectionWeights {
    public var layerIndex: Int
    public var weight: EdgeQuantizedTensor

    public init(layerIndex: Int, weight: EdgeQuantizedTensor) {
        self.layerIndex = layerIndex
        self.weight = weight
    }

    public static func loadHuggingFaceLayout(
        layerIndex: Int,
        weightStore: QwenModelWeightStore,
        groupSize: Int? = nil,
        bits: Int? = nil
    ) throws -> QwenQuantizedAttentionOutputProjectionWeights {
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
        let name = "\(manifest.layerPrefix).self_attn.o_proj.weight"
        let weight = try loadQuantizedWeight(
            name,
            manifest: manifest,
            weightStore: weightStore,
            groupSize: quantization.groupSize,
            bits: quantization.bits
        )
        let expectedShape = [architecture.hiddenSize, architecture.attentionHiddenSize]
        guard weight.shape == expectedShape else {
            throw QwenProjectionWeightError.invalidWeightShape(
                name: name,
                expected: expectedShape,
                actual: weight.shape
            )
        }
        return QwenQuantizedAttentionOutputProjectionWeights(layerIndex: layerIndex, weight: weight)
    }

    public func project(
        attentionOutput: EdgeTensor,
        queryGate: EdgeTensor?,
        executor: MetalKernelExecutor,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> EdgeTensor {
        let gatedOutput: EdgeTensor
        if let queryGate {
            diagnosticSink?("fa_\(layerIndex)_o_gate_begin")
            gatedOutput = try executor.sigmoidMultiply(attentionOutput, gate: queryGate)
            diagnosticSink?("fa_\(layerIndex)_o_gate_done shape=\(gatedOutput.shape.dimensions)")
        } else {
            gatedOutput = attentionOutput
        }
        diagnosticSink?("fa_\(layerIndex)_o_proj_begin")
        let output = try executor.affineQuantizedMatmul(
            gatedOutput,
            weights: weight,
            transpose: true,
            diagnosticSink: diagnosticSink,
            diagnosticName: "fa_\(layerIndex)_o_proj_affine"
        )
        diagnosticSink?("fa_\(layerIndex)_o_proj_done shape=\(output.shape.dimensions)")
        return output
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
}
