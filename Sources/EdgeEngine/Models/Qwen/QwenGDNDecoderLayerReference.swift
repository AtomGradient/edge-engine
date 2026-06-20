// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public struct QwenGDNDecoderLayerOutputs {
    public var hiddenStates: EdgeTensor
    public var nextConvState: EdgeTensor
    public var nextRecurrentState: EdgeTensor

    public init(
        hiddenStates: EdgeTensor,
        nextConvState: EdgeTensor,
        nextRecurrentState: EdgeTensor
    ) {
        self.hiddenStates = hiddenStates
        self.nextConvState = nextConvState
        self.nextRecurrentState = nextRecurrentState
    }
}

public struct QwenGDNDecoderLayerReference {
    public var linearAttention: QwenGDNWeights
    public var mlp: QwenMLPWeights
    public var inputLayerNorm: EdgeTensor
    public var postAttentionLayerNorm: EdgeTensor
    public var rmsNormEpsilon: Float

    public init(
        linearAttention: QwenGDNWeights,
        mlp: QwenMLPWeights,
        inputLayerNorm: EdgeTensor,
        postAttentionLayerNorm: EdgeTensor,
        rmsNormEpsilon: Float
    ) throws {
        guard linearAttention.layerIndex == mlp.layerIndex else {
            throw QwenDecoderLayerReferenceError.layerIndexMismatch(
                attention: linearAttention.layerIndex,
                mlp: mlp.layerIndex
            )
        }
        let hiddenSize = linearAttention.outProj.shape.dimensions[1]
        try Self.validateNorm(inputLayerNorm, name: "input_layernorm.weight", hiddenSize: hiddenSize)
        try Self.validateNorm(postAttentionLayerNorm, name: "post_attention_layernorm.weight", hiddenSize: hiddenSize)
        self.linearAttention = linearAttention
        self.mlp = mlp
        self.inputLayerNorm = inputLayerNorm
        self.postAttentionLayerNorm = postAttentionLayerNorm
        self.rmsNormEpsilon = rmsNormEpsilon
    }

    public static func loadHuggingFaceLayout(
        layerIndex: Int,
        weightStore: QwenModelWeightStore,
        runtime: EdgeMetalRuntime
    ) throws -> QwenGDNDecoderLayerReference {
        let architecture = weightStore.bundleIndex.architecture
        let manifest = try weightStore.bundleIndex.layerManifest(layerIndex)
        let inputLayerNorm = try weightStore.loadFloat32Tensor(
            named: "\(manifest.layerPrefix).input_layernorm.weight",
            runtime: runtime
        )
        let postAttentionLayerNorm = try weightStore.loadFloat32Tensor(
            named: "\(manifest.layerPrefix).post_attention_layernorm.weight",
            runtime: runtime
        )
        return try QwenGDNDecoderLayerReference(
            linearAttention: QwenGDNWeights.loadHuggingFaceLayout(
                layerIndex: layerIndex,
                weightStore: weightStore,
                runtime: runtime
            ),
            mlp: QwenMLPWeights.loadHuggingFaceLayout(
                layerIndex: layerIndex,
                weightStore: weightStore,
                runtime: runtime
            ),
            inputLayerNorm: inputLayerNorm,
            postAttentionLayerNorm: postAttentionLayerNorm,
            rmsNormEpsilon: architecture.rmsNormEpsilon
        )
    }

    public func outputTensor(
        hiddenStates: EdgeTensor,
        convState: EdgeTensor,
        recurrentState: EdgeTensor,
        executor: MetalKernelExecutor,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> QwenGDNDecoderLayerOutputs {
        diagnosticSink?("gdn_\(linearAttention.layerIndex)_attention_norm_begin")
        let attentionInput = try executor.rmsNorm(
            hiddenStates,
            weight: inputLayerNorm,
            epsilon: rmsNormEpsilon
        )
        diagnosticSink?("gdn_\(linearAttention.layerIndex)_attention_norm_done shape=\(attentionInput.shape.dimensions)")
        diagnosticSink?("gdn_\(linearAttention.layerIndex)_linear_attention_begin")
        let attentionOutput = try linearAttention(
            hiddenStates: attentionInput,
            convState: convState,
            recurrentState: recurrentState,
            executor: executor
        )
        diagnosticSink?("gdn_\(linearAttention.layerIndex)_linear_attention_done shape=\(attentionOutput.hiddenStates.shape.dimensions)")
        diagnosticSink?("gdn_\(linearAttention.layerIndex)_attention_residual_begin")
        let attentionResidual = try executor.add(
            hiddenStates,
            attentionOutput.hiddenStates,
            diagnosticSink: diagnosticSink,
            diagnosticName: "gdn_\(linearAttention.layerIndex)_attention_residual_add"
        )
        diagnosticSink?("gdn_\(linearAttention.layerIndex)_attention_residual_done shape=\(attentionResidual.shape.dimensions)")
        diagnosticSink?("gdn_\(linearAttention.layerIndex)_mlp_norm_begin")
        let mlpInput = try executor.rmsNorm(
            attentionResidual,
            weight: postAttentionLayerNorm,
            epsilon: rmsNormEpsilon
        )
        diagnosticSink?("gdn_\(linearAttention.layerIndex)_mlp_norm_done shape=\(mlpInput.shape.dimensions)")
        diagnosticSink?("gdn_\(linearAttention.layerIndex)_mlp_begin")
        let mlpOutput = try mlp(
            hiddenStates: mlpInput,
            executor: executor,
            diagnosticSink: diagnosticSink
        )
        diagnosticSink?("gdn_\(linearAttention.layerIndex)_mlp_done shape=\(mlpOutput.shape.dimensions)")
        diagnosticSink?("gdn_\(linearAttention.layerIndex)_hidden_add_begin")
        let hiddenOutput = try executor.add(
            attentionResidual,
            mlpOutput,
            diagnosticSink: diagnosticSink,
            diagnosticName: "gdn_\(linearAttention.layerIndex)_hidden_add"
        )
        diagnosticSink?("gdn_\(linearAttention.layerIndex)_hidden_add_done shape=\(hiddenOutput.shape.dimensions)")
        return QwenGDNDecoderLayerOutputs(
            hiddenStates: hiddenOutput,
            nextConvState: attentionOutput.nextConvState,
            nextRecurrentState: attentionOutput.nextRecurrentState
        )
    }

    private static func validateNorm(_ tensor: EdgeTensor, name: String, hiddenSize: Int) throws {
        let expectedShape = [hiddenSize]
        guard tensor.shape.dimensions == expectedShape else {
            throw QwenDecoderLayerReferenceError.invalidNormShape(
                name: name,
                expected: expectedShape,
                actual: tensor.shape.dimensions
            )
        }
    }
}

public struct QwenQuantizedGDNDecoderLayerReference {
    public var linearAttention: QwenQuantizedGDNWeights
    public var mlp: QwenQuantizedMLPWeights
    public var inputLayerNorm: EdgeTensor
    public var postAttentionLayerNorm: EdgeTensor
    public var rmsNormEpsilon: Float

    public init(
        linearAttention: QwenQuantizedGDNWeights,
        mlp: QwenQuantizedMLPWeights,
        inputLayerNorm: EdgeTensor,
        postAttentionLayerNorm: EdgeTensor,
        rmsNormEpsilon: Float
    ) throws {
        guard linearAttention.layerIndex == mlp.layerIndex else {
            throw QwenDecoderLayerReferenceError.layerIndexMismatch(
                attention: linearAttention.layerIndex,
                mlp: mlp.layerIndex
            )
        }
        let hiddenSize = linearAttention.outProj.shape[0]
        try Self.validateNorm(inputLayerNorm, name: "input_layernorm.weight", hiddenSize: hiddenSize)
        try Self.validateNorm(postAttentionLayerNorm, name: "post_attention_layernorm.weight", hiddenSize: hiddenSize)
        self.linearAttention = linearAttention
        self.mlp = mlp
        self.inputLayerNorm = inputLayerNorm
        self.postAttentionLayerNorm = postAttentionLayerNorm
        self.rmsNormEpsilon = rmsNormEpsilon
    }

    public static func loadHuggingFaceLayout(
        layerIndex: Int,
        weightStore: QwenModelWeightStore,
        runtime: EdgeMetalRuntime
    ) throws -> QwenQuantizedGDNDecoderLayerReference {
        let architecture = weightStore.bundleIndex.architecture
        let manifest = try weightStore.bundleIndex.layerManifest(layerIndex)
        let inputLayerNorm = try weightStore.loadFloat32Tensor(
            named: "\(manifest.layerPrefix).input_layernorm.weight",
            runtime: runtime
        )
        let postAttentionLayerNorm = try weightStore.loadFloat32Tensor(
            named: "\(manifest.layerPrefix).post_attention_layernorm.weight",
            runtime: runtime
        )
        return try QwenQuantizedGDNDecoderLayerReference(
            linearAttention: QwenQuantizedGDNWeights.loadHuggingFaceLayout(
                layerIndex: layerIndex,
                weightStore: weightStore,
                runtime: runtime
            ),
            mlp: QwenQuantizedMLPWeights.loadHuggingFaceLayout(
                layerIndex: layerIndex,
                weightStore: weightStore
            ),
            inputLayerNorm: inputLayerNorm,
            postAttentionLayerNorm: postAttentionLayerNorm,
            rmsNormEpsilon: architecture.rmsNormEpsilon
        )
    }

    public func outputTensor(
        hiddenStates: EdgeTensor,
        convState: EdgeTensor,
        recurrentState: EdgeTensor,
        executor: MetalKernelExecutor,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> QwenGDNDecoderLayerOutputs {
        diagnosticSink?("gdn_\(linearAttention.layerIndex)_attention_norm_begin")
        let attentionInput = try executor.rmsNorm(
            hiddenStates,
            weight: inputLayerNorm,
            epsilon: rmsNormEpsilon
        )
        diagnosticSink?("gdn_\(linearAttention.layerIndex)_attention_norm_done shape=\(attentionInput.shape.dimensions)")
        diagnosticSink?("gdn_\(linearAttention.layerIndex)_linear_attention_begin")
        let attentionOutput = try linearAttention(
            hiddenStates: attentionInput,
            convState: convState,
            recurrentState: recurrentState,
            executor: executor,
            diagnosticSink: diagnosticSink
        )
        diagnosticSink?("gdn_\(linearAttention.layerIndex)_linear_attention_done shape=\(attentionOutput.hiddenStates.shape.dimensions)")
        diagnosticSink?("gdn_\(linearAttention.layerIndex)_attention_residual_begin")
        let attentionResidual = try executor.add(
            hiddenStates,
            attentionOutput.hiddenStates,
            diagnosticSink: diagnosticSink,
            diagnosticName: "gdn_\(linearAttention.layerIndex)_attention_residual_add"
        )
        diagnosticSink?("gdn_\(linearAttention.layerIndex)_attention_residual_done shape=\(attentionResidual.shape.dimensions)")
        diagnosticSink?("gdn_\(linearAttention.layerIndex)_mlp_norm_begin")
        let mlpInput = try executor.rmsNorm(
            attentionResidual,
            weight: postAttentionLayerNorm,
            epsilon: rmsNormEpsilon
        )
        diagnosticSink?("gdn_\(linearAttention.layerIndex)_mlp_norm_done shape=\(mlpInput.shape.dimensions)")
        diagnosticSink?("gdn_\(linearAttention.layerIndex)_mlp_begin")
        let mlpOutput = try mlp(
            hiddenStates: mlpInput,
            executor: executor,
            diagnosticSink: diagnosticSink
        )
        diagnosticSink?("gdn_\(linearAttention.layerIndex)_mlp_done shape=\(mlpOutput.shape.dimensions)")
        diagnosticSink?("gdn_\(linearAttention.layerIndex)_hidden_add_begin")
        let hiddenOutput = try executor.add(
            attentionResidual,
            mlpOutput,
            diagnosticSink: diagnosticSink,
            diagnosticName: "gdn_\(linearAttention.layerIndex)_hidden_add"
        )
        diagnosticSink?("gdn_\(linearAttention.layerIndex)_hidden_add_done shape=\(hiddenOutput.shape.dimensions)")
        return QwenGDNDecoderLayerOutputs(
            hiddenStates: hiddenOutput,
            nextConvState: attentionOutput.nextConvState,
            nextRecurrentState: attentionOutput.nextRecurrentState
        )
    }

    private static func validateNorm(_ tensor: EdgeTensor, name: String, hiddenSize: Int) throws {
        let expectedShape = [hiddenSize]
        guard tensor.shape.dimensions == expectedShape else {
            throw QwenDecoderLayerReferenceError.invalidNormShape(
                name: name,
                expected: expectedShape,
                actual: tensor.shape.dimensions
            )
        }
    }
}
