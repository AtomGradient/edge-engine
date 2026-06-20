// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public struct QwenMLPWeights {
    public var layerIndex: Int
    public var gate: EdgeTensor
    public var up: EdgeTensor
    public var down: EdgeTensor

    public init(layerIndex: Int, gate: EdgeTensor, up: EdgeTensor, down: EdgeTensor) {
        self.layerIndex = layerIndex
        self.gate = gate
        self.up = up
        self.down = down
    }

    public static func loadRuntimeLayout(
        layerIndex: Int,
        architecture: QwenHybridArchitecture,
        weights: SafeTensorsFile,
        runtime: EdgeMetalRuntime
    ) throws -> QwenMLPWeights {
        let prefix = "model.layers.\(layerIndex).mlp"
        let gate = try weights.loadFloat32Tensor(named: "\(prefix).gate_proj.weight", runtime: runtime)
        let up = try weights.loadFloat32Tensor(named: "\(prefix).up_proj.weight", runtime: runtime)
        let down = try weights.loadFloat32Tensor(named: "\(prefix).down_proj.weight", runtime: runtime)
        try validate(gate: gate, up: up, down: down, prefix: prefix, architecture: architecture)
        return QwenMLPWeights(layerIndex: layerIndex, gate: gate, up: up, down: down)
    }

    public static func loadHuggingFaceLayout(
        layerIndex: Int,
        architecture: QwenHybridArchitecture,
        weights: SafeTensorsFile,
        runtime: EdgeMetalRuntime
    ) throws -> QwenMLPWeights {
        let prefix = "model.layers.\(layerIndex).mlp"
        let gate = try weights.loadFloat32TensorTransposed2D(
            named: "\(prefix).gate_proj.weight",
            runtime: runtime
        )
        let up = try weights.loadFloat32TensorTransposed2D(
            named: "\(prefix).up_proj.weight",
            runtime: runtime
        )
        let down = try weights.loadFloat32TensorTransposed2D(
            named: "\(prefix).down_proj.weight",
            runtime: runtime
        )
        try validate(gate: gate, up: up, down: down, prefix: prefix, architecture: architecture)
        return QwenMLPWeights(layerIndex: layerIndex, gate: gate, up: up, down: down)
    }

    public static func loadHuggingFaceLayout(
        layerIndex: Int,
        weightStore: QwenModelWeightStore,
        runtime: EdgeMetalRuntime
    ) throws -> QwenMLPWeights {
        let architecture = weightStore.bundleIndex.architecture
        let manifest = try weightStore.bundleIndex.layerManifest(layerIndex)
        let prefix = "\(manifest.layerPrefix).mlp"
        let gate = try weightStore.loadFloat32TensorTransposed2D(
            named: "\(prefix).gate_proj.weight",
            runtime: runtime
        )
        let up = try weightStore.loadFloat32TensorTransposed2D(
            named: "\(prefix).up_proj.weight",
            runtime: runtime
        )
        let down = try weightStore.loadFloat32TensorTransposed2D(
            named: "\(prefix).down_proj.weight",
            runtime: runtime
        )
        try validate(gate: gate, up: up, down: down, prefix: prefix, architecture: architecture)
        return QwenMLPWeights(layerIndex: layerIndex, gate: gate, up: up, down: down)
    }

    public func callAsFunction(
        hiddenStates: EdgeTensor,
        executor: MetalKernelExecutor,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> EdgeTensor {
        diagnosticSink?("mlp_\(layerIndex)_gate_proj_begin")
        let gateOutput = try executor.matmul(hiddenStates, gate)
        diagnosticSink?("mlp_\(layerIndex)_gate_proj_done shape=\(gateOutput.shape.dimensions)")
        diagnosticSink?("mlp_\(layerIndex)_up_proj_begin")
        let upOutput = try executor.matmul(hiddenStates, up)
        diagnosticSink?("mlp_\(layerIndex)_up_proj_done shape=\(upOutput.shape.dimensions)")
        diagnosticSink?("mlp_\(layerIndex)_activation_begin")
        let activation = try executor.siluMultiply(
            gate: gateOutput,
            up: upOutput,
            diagnosticSink: diagnosticSink,
            diagnosticName: "mlp_\(layerIndex)_activation"
        )
        diagnosticSink?("mlp_\(layerIndex)_activation_done shape=\(activation.shape.dimensions)")
        diagnosticSink?("mlp_\(layerIndex)_down_proj_begin")
        let downOutput = try executor.matmul(activation, down)
        diagnosticSink?("mlp_\(layerIndex)_down_proj_done shape=\(downOutput.shape.dimensions)")
        return downOutput
    }

    private static func validate(
        gate: EdgeTensor,
        up: EdgeTensor,
        down: EdgeTensor,
        prefix: String,
        architecture: QwenHybridArchitecture
    ) throws {
        let inputShape = [architecture.hiddenSize, architecture.intermediateSize]
        let outputShape = [architecture.intermediateSize, architecture.hiddenSize]
        try validate(gate, name: "\(prefix).gate_proj.weight", expectedShape: inputShape)
        try validate(up, name: "\(prefix).up_proj.weight", expectedShape: inputShape)
        try validate(down, name: "\(prefix).down_proj.weight", expectedShape: outputShape)
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

public struct QwenQuantizedMLPWeights {
    public var layerIndex: Int
    public var gate: EdgeQuantizedTensor
    public var up: EdgeQuantizedTensor
    public var down: EdgeQuantizedTensor

    public init(
        layerIndex: Int,
        gate: EdgeQuantizedTensor,
        up: EdgeQuantizedTensor,
        down: EdgeQuantizedTensor
    ) {
        self.layerIndex = layerIndex
        self.gate = gate
        self.up = up
        self.down = down
    }

    public static func loadHuggingFaceLayout(
        layerIndex: Int,
        weightStore: QwenModelWeightStore,
        groupSize: Int? = nil,
        bits: Int? = nil
    ) throws -> QwenQuantizedMLPWeights {
        let architecture = weightStore.bundleIndex.architecture
        let quantization = try resolveQuantization(
            architecture: architecture,
            groupSize: groupSize,
            bits: bits
        )
        let manifest = try weightStore.bundleIndex.layerManifest(layerIndex)
        let prefix = "\(manifest.layerPrefix).mlp"
        let gateName = "\(prefix).gate_proj.weight"
        let upName = "\(prefix).up_proj.weight"
        let downName = "\(prefix).down_proj.weight"

        let gate = try loadQuantizedWeight(
            gateName,
            manifest: manifest,
            weightStore: weightStore,
            groupSize: quantization.groupSize,
            bits: quantization.bits
        )
        let up = try loadQuantizedWeight(
            upName,
            manifest: manifest,
            weightStore: weightStore,
            groupSize: quantization.groupSize,
            bits: quantization.bits
        )
        let down = try loadQuantizedWeight(
            downName,
            manifest: manifest,
            weightStore: weightStore,
            groupSize: quantization.groupSize,
            bits: quantization.bits
        )

        try validate(gate, name: gateName, expectedShape: [
            architecture.intermediateSize,
            architecture.hiddenSize,
        ])
        try validate(up, name: upName, expectedShape: [
            architecture.intermediateSize,
            architecture.hiddenSize,
        ])
        try validate(down, name: downName, expectedShape: [
            architecture.hiddenSize,
            architecture.intermediateSize,
        ])

        return QwenQuantizedMLPWeights(layerIndex: layerIndex, gate: gate, up: up, down: down)
    }

    public func callAsFunction(
        hiddenStates: EdgeTensor,
        executor: MetalKernelExecutor,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> EdgeTensor {
        diagnosticSink?("mlp_\(layerIndex)_gate_proj_begin")
        let gateOutput = try executor.affineQuantizedMatmul(
            hiddenStates,
            weights: gate,
            transpose: true,
            diagnosticSink: diagnosticSink,
            diagnosticName: "mlp_\(layerIndex)_gate_proj_affine"
        )
        diagnosticSink?("mlp_\(layerIndex)_gate_proj_done shape=\(gateOutput.shape.dimensions)")
        diagnosticSink?("mlp_\(layerIndex)_up_proj_begin")
        let upOutput = try executor.affineQuantizedMatmul(
            hiddenStates,
            weights: up,
            transpose: true,
            diagnosticSink: diagnosticSink,
            diagnosticName: "mlp_\(layerIndex)_up_proj_affine"
        )
        diagnosticSink?("mlp_\(layerIndex)_up_proj_done shape=\(upOutput.shape.dimensions)")
        diagnosticSink?("mlp_\(layerIndex)_activation_begin")
        let activation = try executor.siluMultiply(
            gate: gateOutput,
            up: upOutput,
            diagnosticSink: diagnosticSink,
            diagnosticName: "mlp_\(layerIndex)_activation"
        )
        diagnosticSink?("mlp_\(layerIndex)_activation_done shape=\(activation.shape.dimensions)")
        diagnosticSink?("mlp_\(layerIndex)_down_proj_begin")
        let downOutput = try executor.affineQuantizedMatmul(
            activation,
            weights: down,
            transpose: true,
            diagnosticSink: diagnosticSink,
            diagnosticName: "mlp_\(layerIndex)_down_proj_affine"
        )
        diagnosticSink?("mlp_\(layerIndex)_down_proj_done shape=\(downOutput.shape.dimensions)")
        return downOutput
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
