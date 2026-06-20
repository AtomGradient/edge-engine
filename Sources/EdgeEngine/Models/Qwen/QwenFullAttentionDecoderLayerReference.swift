// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public enum QwenDecoderLayerReferenceError: Error, Equatable {
    case layerIndexMismatch(attention: Int, mlp: Int)
    case invalidNormShape(name: String, expected: [Int], actual: [Int])
}

public struct QwenFullAttentionDecoderLayerReference {
    public var attention: QwenFullAttentionReference
    public var mlp: QwenMLPWeights
    public var inputLayerNorm: EdgeTensor
    public var postAttentionLayerNorm: EdgeTensor
    public var rmsNormEpsilon: Float

    public init(
        attention: QwenFullAttentionReference,
        mlp: QwenMLPWeights,
        inputLayerNorm: EdgeTensor,
        postAttentionLayerNorm: EdgeTensor,
        rmsNormEpsilon: Float
    ) throws {
        guard attention.layerIndex == mlp.layerIndex else {
            throw QwenDecoderLayerReferenceError.layerIndexMismatch(
                attention: attention.layerIndex,
                mlp: mlp.layerIndex
            )
        }
        let hiddenSize = attention.architecture.hiddenSize
        try Self.validateNorm(inputLayerNorm, name: "input_layernorm.weight", hiddenSize: hiddenSize)
        try Self.validateNorm(postAttentionLayerNorm, name: "post_attention_layernorm.weight", hiddenSize: hiddenSize)
        self.attention = attention
        self.mlp = mlp
        self.inputLayerNorm = inputLayerNorm
        self.postAttentionLayerNorm = postAttentionLayerNorm
        self.rmsNormEpsilon = rmsNormEpsilon
    }

    public static func loadHuggingFaceLayout(
        layerIndex: Int,
        weightStore: QwenModelWeightStore,
        runtime: EdgeMetalRuntime
    ) throws -> QwenFullAttentionDecoderLayerReference {
        let architecture = weightStore.bundleIndex.architecture
        let manifest = try weightStore.bundleIndex.layerManifest(layerIndex)
        let attention = try QwenFullAttentionReference(
            architecture: architecture,
            layerIndex: layerIndex,
            projectionWeights: QwenAttentionProjectionWeights.loadHuggingFaceLayout(
                layerIndex: layerIndex,
                weightStore: weightStore,
                runtime: runtime
            ),
            normalizationWeights: QwenAttentionNormWeights.loadHuggingFaceLayout(
                layerIndex: layerIndex,
                weightStore: weightStore,
                runtime: runtime
            ),
            outputProjectionWeights: QwenAttentionOutputProjectionWeights.loadHuggingFaceLayout(
                layerIndex: layerIndex,
                weightStore: weightStore,
                runtime: runtime
            )
        )
        let inputLayerNorm = try weightStore.loadFloat32Tensor(
            named: "\(manifest.layerPrefix).input_layernorm.weight",
            runtime: runtime
        )
        let postAttentionLayerNorm = try weightStore.loadFloat32Tensor(
            named: "\(manifest.layerPrefix).post_attention_layernorm.weight",
            runtime: runtime
        )
        return try QwenFullAttentionDecoderLayerReference(
            attention: attention,
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
        executor: MetalKernelExecutor,
        positionOffset: Int = 0,
        kvCache: QwenKVCache? = nil,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> EdgeTensor {
        diagnosticSink?("fa_\(attention.layerIndex)_attention_norm_begin")
        let attentionInput = try executor.rmsNorm(
            hiddenStates,
            weight: inputLayerNorm,
            epsilon: rmsNormEpsilon
        )
        diagnosticSink?("fa_\(attention.layerIndex)_attention_norm_done shape=\(attentionInput.shape.dimensions)")
        diagnosticSink?("fa_\(attention.layerIndex)_attention_begin")
        let attentionOutput = try attention.attentionTensor(
            hiddenStates: attentionInput,
            executor: executor,
            positionOffset: positionOffset,
            kvCache: kvCache,
            diagnosticSink: diagnosticSink
        )
        diagnosticSink?("fa_\(attention.layerIndex)_attention_done shape=\(attentionOutput.shape.dimensions)")
        diagnosticSink?("fa_\(attention.layerIndex)_attention_residual_begin")
        let attentionResidual = try executor.add(hiddenStates, attentionOutput)
        diagnosticSink?("fa_\(attention.layerIndex)_attention_residual_done shape=\(attentionResidual.shape.dimensions)")
        diagnosticSink?("fa_\(attention.layerIndex)_mlp_norm_begin")
        let mlpInput = try executor.rmsNorm(
            attentionResidual,
            weight: postAttentionLayerNorm,
            epsilon: rmsNormEpsilon
        )
        diagnosticSink?("fa_\(attention.layerIndex)_mlp_norm_done shape=\(mlpInput.shape.dimensions)")
        diagnosticSink?("fa_\(attention.layerIndex)_mlp_begin")
        let mlpOutput = try mlp(hiddenStates: mlpInput, executor: executor)
        diagnosticSink?("fa_\(attention.layerIndex)_mlp_done shape=\(mlpOutput.shape.dimensions)")
        diagnosticSink?("fa_\(attention.layerIndex)_hidden_add_begin")
        let output = try executor.add(attentionResidual, mlpOutput)
        diagnosticSink?("fa_\(attention.layerIndex)_hidden_add_done shape=\(output.shape.dimensions)")
        return output
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

public struct QwenQuantizedFullAttentionDecoderLayerReference {
    public var attention: QwenQuantizedFullAttentionReference
    public var mlp: QwenQuantizedMLPWeights
    public var inputLayerNorm: EdgeTensor
    public var postAttentionLayerNorm: EdgeTensor
    public var rmsNormEpsilon: Float

    public init(
        attention: QwenQuantizedFullAttentionReference,
        mlp: QwenQuantizedMLPWeights,
        inputLayerNorm: EdgeTensor,
        postAttentionLayerNorm: EdgeTensor,
        rmsNormEpsilon: Float
    ) throws {
        guard attention.layerIndex == mlp.layerIndex else {
            throw QwenDecoderLayerReferenceError.layerIndexMismatch(
                attention: attention.layerIndex,
                mlp: mlp.layerIndex
            )
        }
        let hiddenSize = attention.architecture.hiddenSize
        try Self.validateNorm(inputLayerNorm, name: "input_layernorm.weight", hiddenSize: hiddenSize)
        try Self.validateNorm(postAttentionLayerNorm, name: "post_attention_layernorm.weight", hiddenSize: hiddenSize)
        self.attention = attention
        self.mlp = mlp
        self.inputLayerNorm = inputLayerNorm
        self.postAttentionLayerNorm = postAttentionLayerNorm
        self.rmsNormEpsilon = rmsNormEpsilon
    }

    public static func loadHuggingFaceLayout(
        layerIndex: Int,
        weightStore: QwenModelWeightStore,
        runtime: EdgeMetalRuntime
    ) throws -> QwenQuantizedFullAttentionDecoderLayerReference {
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
        return try QwenQuantizedFullAttentionDecoderLayerReference(
            attention: QwenQuantizedFullAttentionReference.loadHuggingFaceLayout(
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
        executor: MetalKernelExecutor,
        positionOffset: Int = 0,
        kvCache: QwenKVCache? = nil,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> EdgeTensor {
        diagnosticSink?("fa_\(attention.layerIndex)_attention_norm_begin")
        let attentionInput = try executor.rmsNorm(
            hiddenStates,
            weight: inputLayerNorm,
            epsilon: rmsNormEpsilon
        )
        diagnosticSink?("fa_\(attention.layerIndex)_attention_norm_done shape=\(attentionInput.shape.dimensions)")
        diagnosticSink?("fa_\(attention.layerIndex)_attention_begin")
        let attentionOutput = try attention.attentionTensor(
            hiddenStates: attentionInput,
            executor: executor,
            positionOffset: positionOffset,
            kvCache: kvCache,
            diagnosticSink: diagnosticSink
        )
        diagnosticSink?("fa_\(attention.layerIndex)_attention_done shape=\(attentionOutput.shape.dimensions)")
        diagnosticSink?("fa_\(attention.layerIndex)_attention_residual_begin")
        let attentionResidual = try executor.add(hiddenStates, attentionOutput)
        diagnosticSink?("fa_\(attention.layerIndex)_attention_residual_done shape=\(attentionResidual.shape.dimensions)")
        diagnosticSink?("fa_\(attention.layerIndex)_mlp_norm_begin")
        let mlpInput = try executor.rmsNorm(
            attentionResidual,
            weight: postAttentionLayerNorm,
            epsilon: rmsNormEpsilon
        )
        diagnosticSink?("fa_\(attention.layerIndex)_mlp_norm_done shape=\(mlpInput.shape.dimensions)")
        diagnosticSink?("fa_\(attention.layerIndex)_mlp_begin")
        let mlpOutput = try mlp(
            hiddenStates: mlpInput,
            executor: executor,
            diagnosticSink: diagnosticSink
        )
        diagnosticSink?("fa_\(attention.layerIndex)_mlp_done shape=\(mlpOutput.shape.dimensions)")
        diagnosticSink?("fa_\(attention.layerIndex)_hidden_add_begin")
        let output = try executor.add(attentionResidual, mlpOutput)
        diagnosticSink?("fa_\(attention.layerIndex)_hidden_add_done shape=\(output.shape.dimensions)")
        return output
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
