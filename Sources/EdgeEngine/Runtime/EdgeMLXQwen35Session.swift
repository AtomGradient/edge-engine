// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeCmlx
import Metal

public enum EdgeMLXQwen35SessionError: Error, Equatable {
    case invalidConfiguration
    case invalidTensorShape
    case invalidQuantizedTensorShape
    case allocationFailed(byteCount: Int)
    case executionFailed(String)
}

extension EdgeMLXQwen35SessionError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration:
            "invalid Qwen3.5 Cmlx session configuration"
        case .invalidTensorShape:
            "invalid Qwen3.5 Cmlx tensor shape"
        case .invalidQuantizedTensorShape:
            "invalid Qwen3.5 Cmlx quantized tensor shape"
        case let .allocationFailed(byteCount):
            "failed to allocate \(byteCount) bytes for Qwen3.5 Cmlx session"
        case let .executionFailed(message):
            message
        }
    }
}

public struct EdgeMLXQwen35GDNDecodeLayerOutput {
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

public struct EdgeMLXQwen35VisionEncoding: Equatable, Sendable {
    public var values: [Float]
    public var shape: [Int]

    public init(values: [Float], shape: [Int]) {
        self.values = values
        self.shape = shape
    }
}

public struct EdgeMLXQwen35AudioTowerConfiguration: Equatable, Sendable {
    public var numMelBins: Int
    public var encoderLayers: Int
    public var encoderAttentionHeads: Int
    public var encoderFFNDim: Int
    public var dModel: Int
    public var maxSourcePositions: Int
    public var nWindow: Int
    public var outputDim: Int
    public var nWindowInfer: Int
    public var downsampleHiddenSize: Int
    public var layerNormEpsilon: Float

    public init(
        numMelBins: Int,
        encoderLayers: Int,
        encoderAttentionHeads: Int,
        encoderFFNDim: Int,
        dModel: Int,
        maxSourcePositions: Int,
        nWindow: Int,
        outputDim: Int,
        nWindowInfer: Int,
        downsampleHiddenSize: Int,
        layerNormEpsilon: Float = 1e-5
    ) {
        self.numMelBins = numMelBins
        self.encoderLayers = encoderLayers
        self.encoderAttentionHeads = encoderAttentionHeads
        self.encoderFFNDim = encoderFFNDim
        self.dModel = dModel
        self.maxSourcePositions = maxSourcePositions
        self.nWindow = nWindow
        self.outputDim = outputDim
        self.nWindowInfer = nWindowInfer
        self.downsampleHiddenSize = downsampleHiddenSize
        self.layerNormEpsilon = layerNormEpsilon
    }
}

public struct EdgeMLXQwen35AudioEncoding: Equatable, Sendable {
    public var values: [Float]
    public var shape: [Int]

    public init(values: [Float], shape: [Int]) {
        self.values = values
        self.shape = shape
    }
}

public struct EdgeMLXTTSCodes: Equatable, Sendable {
    public var values: [Int]
    public var shape: [Int]

    public init(values: [Int], shape: [Int]) {
        self.values = values
        self.shape = shape
    }
}

/// Persistent Cmlx-side storage for Qwen3.5/3.6 lazy decode work.
///
/// This is the first step toward matching the mlx-swift-lm execution model:
/// weights are wrapped once as MLX arrays, then decode-step graph construction
/// can reuse them without re-uploading quantized buffers every token.
public final class EdgeMLXQwen35Session {
    private let handle: UnsafeMutableRawPointer
    private let runtime: EdgeMetalRuntime
    private let architecture: QwenHybridArchitecture
    private let weightBufferOwner: MetalKernelExecutor?
    private var retainedBuffers: [MTLBuffer] = []
    private var audioTowerConfiguration: EdgeMLXQwen35AudioTowerConfiguration?

    public init(
        architecture: QwenHybridArchitecture,
        runtime: EdgeMetalRuntime,
        weightBufferOwner: MetalKernelExecutor? = nil
    ) throws {
        self.architecture = architecture
        self.weightBufferOwner = weightBufferOwner
        var config = EdgeCmlxQwen35Config(
            layer_count: Int32(architecture.layerCount),
            hidden_size: Int32(architecture.hiddenSize),
            vocabulary_size: Int32(architecture.vocabularySize),
            intermediate_size: Int32(architecture.intermediateSize),
            attention_head_count: Int32(architecture.attentionHeadCount),
            key_value_head_count: Int32(architecture.keyValueHeadCount),
            attention_head_dimension: Int32(architecture.attentionHeadDimension),
            linear_key_head_count: Int32(architecture.linearKeyHeadCount),
            linear_value_head_count: Int32(architecture.linearValueHeadCount),
            linear_key_head_dimension: Int32(architecture.linearKeyHeadDimension),
            linear_value_head_dimension: Int32(architecture.linearValueHeadDimension),
            linear_conv_kernel_size: Int32(architecture.linearConvKernelSize),
            rotary_dimension: Int32(architecture.rotaryDimension),
            rope_theta: architecture.ropeTheta,
            rms_norm_epsilon: architecture.rmsNormEpsilon,
            uses_sparse_moe: architecture.usesSparseMoEMLP ? 1 : 0,
            moe_num_experts: Int32(architecture.moeMLP?.expertCount ?? 0),
            moe_experts_per_token: Int32(architecture.moeMLP?.expertsPerToken ?? 0),
            moe_intermediate_size: Int32(architecture.moeMLP?.intermediateSize ?? 0),
            moe_shared_expert_intermediate_size: Int32(
                architecture.moeMLP?.sharedExpertIntermediateSize ?? 0
            ),
            moe_normalize_topk_probabilities: architecture.moeMLP?.normalizeTopKProbabilities == false
                ? 0
                : 1
        )
        guard let handle = edge_cmlx_qwen35_session_create(&config) else {
            throw Self.currentError(defaultMessage: "failed to create Qwen3.5 Cmlx session")
        }
        self.handle = handle
        self.runtime = runtime

        for layer in architecture.layerPlan {
            let layerKind: Int32
            switch layer.kind {
            case .fullAttention:
                layerKind = Int32(EdgeCmlxQwen35LayerKindFullAttention.rawValue)
            case .gdn:
                layerKind = Int32(EdgeCmlxQwen35LayerKindGDN.rawValue)
            }
            let status = edge_cmlx_qwen35_session_set_layer_kind(
                handle,
                Int32(layer.index),
                layerKind
            )
            guard status == 0 else {
                throw Self.currentError(defaultMessage: "failed to set Qwen3.5 Cmlx layer kind")
            }
        }
    }

    deinit {
        edge_cmlx_qwen35_session_destroy(handle)
    }

    public var registeredFloatTensorCount: Int {
        Int(edge_cmlx_qwen35_session_registered_float_tensor_count(handle))
    }

    public var registeredQuantizedTensorCount: Int {
        Int(edge_cmlx_qwen35_session_registered_quantized_tensor_count(handle))
    }

    public var decodedTokenCount: Int {
        Int(edge_cmlx_qwen35_session_decoded_token_count(handle))
    }

    public func hasDecoderWeights() throws -> Bool {
        let status = edge_cmlx_qwen35_session_has_decoder_weights(handle)
        guard status >= 0 else {
            throw Self.currentError(defaultMessage: "failed to inspect Qwen3.5 Cmlx decoder weights")
        }
        return status == 1
    }

    public static func configureCommandBufferLimits(
        maxOps: Int,
        maxMB: Int
    ) throws {
        let status = edge_cmlx_set_command_buffer_limits(
            Int32(max(1, maxOps)),
            Int32(max(1, maxMB))
        )
        guard status == 0 else {
            throw currentError(defaultMessage: "failed to configure Cmlx command buffer limits")
        }
    }

    public static func currentCommandBufferLimits() throws -> (maxOps: Int, maxMB: Int) {
        var maxOps = Int32(0)
        var maxMB = Int32(0)
        let status = edge_cmlx_get_command_buffer_limits(&maxOps, &maxMB)
        guard status == 0 else {
            throw currentError(defaultMessage: "failed to read Cmlx command buffer limits")
        }
        return (Int(maxOps), Int(maxMB))
    }

    public static func configureMemoryLimit(bytes: Int) throws {
        let status = edge_cmlx_set_memory_limit(max(1, bytes))
        guard status == 0 else {
            throw currentError(defaultMessage: "failed to configure Cmlx memory limit")
        }
    }

    public static func currentMemoryLimitBytes() throws -> Int {
        var bytes = 0
        let status = edge_cmlx_get_memory_limit(&bytes)
        guard status == 0 else {
            throw currentError(defaultMessage: "failed to read Cmlx memory limit")
        }
        return bytes
    }

    public func registerFloatTensor(
        id: Int,
        tensor: EdgeTensor
    ) throws {
        guard tensor.dataType == .float32,
              (1...4).contains(tensor.shape.rank)
        else {
            throw EdgeMLXQwen35SessionError.invalidTensorShape
        }
        var dimensions = [Int32](repeating: 1, count: 4)
        for (index, dimension) in tensor.shape.dimensions.enumerated() {
            dimensions[index] = Int32(dimension)
        }
        var descriptor = EdgeCmlxFloatTensorDescriptor(
            buffer: Self.opaquePointer(to: tensor.buffer),
            rank: Int32(tensor.shape.rank),
            dim0: dimensions[0],
            dim1: dimensions[1],
            dim2: dimensions[2],
            dim3: dimensions[3]
        )
        let status = edge_cmlx_qwen35_session_set_float_tensor(
            handle,
            Int32(id),
            &descriptor
        )
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to register Qwen3.5 float tensor")
        }
    }

    public func registerQuantizedTensor(
        id: Int,
        tensor: EdgeQuantizedTensor
    ) throws {
        guard tensor.shape.count == 2,
              tensor.packedShape.count == 2,
              tensor.scaleShape.count == 2
        else {
            throw EdgeMLXQwen35SessionError.invalidQuantizedTensorShape
        }

        let packed = try makeBuffer(
            values: tensor.packedValues,
            label: "edge.cmlx.qwen35.session.packed.\(id)"
        )
        let scales = try makeBuffer(
            values: tensor.scales,
            label: "edge.cmlx.qwen35.session.scales.\(id)"
        )
        let biases = try makeBuffer(
            values: tensor.biases ?? Array(repeating: Float.zero, count: tensor.scaleCount),
            label: "edge.cmlx.qwen35.session.biases.\(id)"
        )
        retainedBuffers.append(contentsOf: [packed, scales, biases])

        var descriptor = EdgeCmlxQuantizedTensorDescriptor(
            packed_buffer: Self.opaquePointer(to: packed),
            packed_offset: 0,
            packed_rows: Int32(tensor.packedShape[0]),
            packed_cols: Int32(tensor.packedShape[1]),
            scales_buffer: Self.opaquePointer(to: scales),
            scales_offset: 0,
            scale_rows: Int32(tensor.scaleShape[0]),
            scale_cols: Int32(tensor.scaleShape[1]),
            biases_buffer: Self.opaquePointer(to: biases),
            biases_offset: 0,
            group_size: Int32(tensor.groupSize),
            bits: Int32(tensor.bits)
        )
        let status = edge_cmlx_qwen35_session_set_quantized_tensor(
            handle,
            Int32(id),
            &descriptor
        )
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to register Qwen3.5 quantized tensor")
        }
    }

    public func registerQuantizedTensor(
        id: Int,
        tensor: EdgeQuantizedTensor,
        executor: MetalKernelExecutor
    ) throws {
        let buffers = try executor.sharedAffineQuantizedMetalBuffers(
            for: tensor,
            preferNoCopy: true
        )
        try registerQuantizedTensor(
            id: id,
            tensor: tensor,
            packedBuffer: buffers.packedBuffer,
            packedOffset: buffers.packedOffset,
            scalesBuffer: buffers.scalesBuffer,
            scalesOffset: buffers.scalesOffset,
            biasesBuffer: buffers.biasesBuffer,
            biasesOffset: buffers.biasesOffset
        )
    }

    public func loadSafetensors(
        shardURLs: [URL],
        modelPrefix: String,
        groupSize: Int,
        bits: Int
    ) throws {
        guard !shardURLs.isEmpty, groupSize > 0, bits > 0 else {
            throw EdgeMLXQwen35SessionError.invalidConfiguration
        }
        let duplicatedPaths = shardURLs.map { strdup($0.path)! }
        defer {
            for path in duplicatedPaths {
                free(path)
            }
        }
        let pathPointers: [UnsafePointer<CChar>?] = duplicatedPaths.map { UnsafePointer<CChar>($0) }
        let status = pathPointers.withUnsafeBufferPointer { paths in
            modelPrefix.withCString { prefix in
                edge_cmlx_qwen35_session_load_safetensors(
                    handle,
                    paths.baseAddress,
                    Int32(paths.count),
                    prefix,
                    Int32(groupSize),
                    Int32(bits)
                )
            }
        }
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to load Qwen3.5 Cmlx safetensors")
        }
    }

    public func materializeDecoderWeights() throws {
        let status = edge_cmlx_qwen35_session_materialize_decoder_weights(handle)
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to materialize Qwen3.5 Cmlx decoder weights")
        }
    }

    public func unloadDecoderWeightsPreservingState() throws {
        let status = edge_cmlx_qwen35_session_unload_decoder_weights_preserving_state(handle)
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to unload Qwen3.5 Cmlx decoder weights")
        }
    }

    public func restoreNeuralImprintCache(
        artifactURL: URL,
        prefixTokenCount: Int
    ) throws {
        guard prefixTokenCount > 0 else {
            throw EdgeMLXQwen35SessionError.invalidConfiguration
        }
        let status = artifactURL.path.withCString { path in
            edge_cmlx_qwen35_session_restore_neural_imprint_cache(
                handle,
                path,
                Int32(prefixTokenCount)
            )
        }
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to restore Qwen3.5 Cmlx Neural Imprint cache")
        }
    }

    public func saveNeuralImprintCache(
        artifactURL: URL,
        metadata: [String: String]
    ) throws {
        let entries = metadata.sorted { $0.key < $1.key }
        let duplicatedKeys = entries.map { strdup($0.key)! }
        let duplicatedValues = entries.map { strdup($0.value)! }
        defer {
            for key in duplicatedKeys {
                free(key)
            }
            for value in duplicatedValues {
                free(value)
            }
        }
        let keyPointers: [UnsafePointer<CChar>?] = duplicatedKeys.map {
            UnsafePointer<CChar>($0)
        }
        let valuePointers: [UnsafePointer<CChar>?] = duplicatedValues.map {
            UnsafePointer<CChar>($0)
        }
        let status = keyPointers.withUnsafeBufferPointer { keys in
            valuePointers.withUnsafeBufferPointer { values in
                artifactURL.path.withCString { path in
                    edge_cmlx_qwen35_session_save_neural_imprint_cache(
                        handle,
                        path,
                        keys.baseAddress,
                        values.baseAddress,
                        Int32(entries.count)
                    )
                }
            }
        }
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to save Qwen3.5 Cmlx Neural Imprint cache")
        }
    }

    public func loadTTSSafetensors(
        shardURLs: [URL],
        groupSize: Int,
        bits: Int,
        codePredictorArchitecture: QwenHybridArchitecture
    ) throws {
        guard !shardURLs.isEmpty, groupSize > 0, bits > 0 else {
            throw EdgeMLXQwen35SessionError.invalidConfiguration
        }
        let duplicatedPaths = shardURLs.map { strdup($0.path)! }
        defer {
            for path in duplicatedPaths {
                free(path)
            }
        }
        let pathPointers: [UnsafePointer<CChar>?] = duplicatedPaths.map { UnsafePointer<CChar>($0) }
        var codePredictorConfig = EdgeCmlxQwen35Config(
            layer_count: Int32(codePredictorArchitecture.layerCount),
            hidden_size: Int32(codePredictorArchitecture.hiddenSize),
            vocabulary_size: Int32(codePredictorArchitecture.vocabularySize),
            intermediate_size: Int32(codePredictorArchitecture.intermediateSize),
            attention_head_count: Int32(codePredictorArchitecture.attentionHeadCount),
            key_value_head_count: Int32(codePredictorArchitecture.keyValueHeadCount),
            attention_head_dimension: Int32(codePredictorArchitecture.attentionHeadDimension),
            linear_key_head_count: Int32(codePredictorArchitecture.linearKeyHeadCount),
            linear_value_head_count: Int32(codePredictorArchitecture.linearValueHeadCount),
            linear_key_head_dimension: Int32(codePredictorArchitecture.linearKeyHeadDimension),
            linear_value_head_dimension: Int32(codePredictorArchitecture.linearValueHeadDimension),
            linear_conv_kernel_size: Int32(codePredictorArchitecture.linearConvKernelSize),
            rotary_dimension: Int32(codePredictorArchitecture.rotaryDimension),
            rope_theta: codePredictorArchitecture.ropeTheta,
            rms_norm_epsilon: codePredictorArchitecture.rmsNormEpsilon,
            uses_sparse_moe: codePredictorArchitecture.usesSparseMoEMLP ? 1 : 0,
            moe_num_experts: Int32(codePredictorArchitecture.moeMLP?.expertCount ?? 0),
            moe_experts_per_token: Int32(codePredictorArchitecture.moeMLP?.expertsPerToken ?? 0),
            moe_intermediate_size: Int32(codePredictorArchitecture.moeMLP?.intermediateSize ?? 0),
            moe_shared_expert_intermediate_size: Int32(
                codePredictorArchitecture.moeMLP?.sharedExpertIntermediateSize ?? 0
            ),
            moe_normalize_topk_probabilities: codePredictorArchitecture.moeMLP?.normalizeTopKProbabilities == false
                ? 0
                : 1
        )
        let status = pathPointers.withUnsafeBufferPointer { paths in
            edge_cmlx_qwen35_session_load_tts_safetensors(
                handle,
                paths.baseAddress,
                Int32(paths.count),
                Int32(groupSize),
                Int32(bits),
                &codePredictorConfig
            )
        }
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to load Qwen3 TTS Cmlx safetensors")
        }
    }

    public func loadTTSSpeechTokenizerSafetensors(safetensorsURL: URL) throws {
        let status = safetensorsURL.path.withCString { path in
            edge_cmlx_qwen35_session_load_tts_speech_tokenizer_safetensors(
                handle,
                path
            )
        }
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to load Qwen3 TTS speech tokenizer safetensors")
        }
    }

    public func setVisionConfig(
        vision: QwenVisionConfiguration,
        imageProcessor: QwenImageProcessorConfiguration,
        outputHiddenSize: Int
    ) throws {
        let patchSize = vision.patchSize ?? imageProcessor.patchSize ?? 16
        let spatialMergeSize = vision.spatialMergeSize ?? imageProcessor.mergeSize ?? 2
        let temporalPatchSize = imageProcessor.temporalPatchSize ?? 2
        var config = EdgeCmlxQwen35VisionConfig(
            hidden_size: Int32(vision.hiddenSize),
            intermediate_size: Int32(vision.intermediateSize ?? vision.hiddenSize * 4),
            layer_count: Int32(vision.layerCount),
            head_count: Int32(vision.attentionHeadCount),
            patch_size: Int32(patchSize),
            spatial_merge_size: Int32(spatialMergeSize),
            temporal_patch_size: Int32(temporalPatchSize),
            output_hidden_size: Int32(outputHiddenSize),
            layer_norm_epsilon: 1e-6
        )
        let status = edge_cmlx_qwen35_session_set_vision_config(handle, &config)
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to configure Qwen3.5 Cmlx vision encoder")
        }
    }

    public func setVisionConfig(plan: QwenVLMRuntimePlan) throws {
        try setVisionConfig(
            vision: plan.visionConfiguration,
            imageProcessor: plan.imageProcessorConfiguration,
            outputHiddenSize: plan.languageArchitecture.hiddenSize
        )
    }

    public func loadVisionSafetensors(
        shardURLs: [URL],
        visionPrefix: String,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws {
        guard !shardURLs.isEmpty, !visionPrefix.isEmpty else {
            throw EdgeMLXQwen35SessionError.invalidConfiguration
        }
        diagnosticSink?(
            "vlm_cmlx_vision_safetensors_prepare shards=\(shardURLs.count) prefix=\(visionPrefix)"
        )
        let duplicatedPaths = shardURLs.map { strdup($0.path)! }
        defer {
            for path in duplicatedPaths {
                free(path)
            }
        }
        let pathPointers: [UnsafePointer<CChar>?] = duplicatedPaths.map { UnsafePointer<CChar>($0) }
        diagnosticSink?("vlm_cmlx_vision_safetensors_c_call_begin shards=\(pathPointers.count)")
        let status = pathPointers.withUnsafeBufferPointer { paths in
            visionPrefix.withCString { prefix in
                edge_cmlx_qwen35_session_load_vision_safetensors(
                    handle,
                    paths.baseAddress,
                    Int32(paths.count),
                    prefix
                )
            }
        }
        diagnosticSink?("vlm_cmlx_vision_safetensors_c_call_return status=\(status)")
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to load Qwen3.5 Cmlx vision safetensors")
        }
    }

    public func loadVisionSafetensors(
        index: QwenVLMModelBundleIndex,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws {
        diagnosticSink?(
            "vlm_cmlx_vision_safetensors_index_begin tensors=\(index.visionManifest.tensorNames.count)"
        )
        var shardFileNames = Set<String>()
        for tensorName in index.visionManifest.tensorNames {
            shardFileNames.insert(try index.shardFileName(containing: tensorName))
        }
        let shardURLs = shardFileNames
            .sorted()
            .map { index.rootURL.appendingPathComponent($0) }
        let shardFileList = shardURLs.map(\.lastPathComponent).joined(separator: ",")
        diagnosticSink?(
            "vlm_cmlx_vision_safetensors_index_done shards=\(shardURLs.count) files=\(shardFileList)"
        )
        try loadVisionSafetensors(
            shardURLs: shardURLs,
            visionPrefix: index.visionManifest.prefix,
            diagnosticSink: diagnosticSink
        )
    }

    public func setAudioConfig(_ configuration: EdgeMLXQwen35AudioTowerConfiguration) throws {
        var config = EdgeCmlxQwen35AudioConfig(
            num_mel_bins: Int32(configuration.numMelBins),
            encoder_layers: Int32(configuration.encoderLayers),
            encoder_attention_heads: Int32(configuration.encoderAttentionHeads),
            encoder_ffn_dim: Int32(configuration.encoderFFNDim),
            d_model: Int32(configuration.dModel),
            max_source_positions: Int32(configuration.maxSourcePositions),
            n_window: Int32(configuration.nWindow),
            output_dim: Int32(configuration.outputDim),
            n_window_infer: Int32(configuration.nWindowInfer),
            downsample_hidden_size: Int32(configuration.downsampleHiddenSize),
            layer_norm_epsilon: configuration.layerNormEpsilon
        )
        let status = edge_cmlx_qwen35_session_set_audio_config(handle, &config)
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to configure Qwen3 ASR Cmlx audio encoder")
        }
        audioTowerConfiguration = configuration
    }

    public func loadAudioSafetensors(
        shardURLs: [URL],
        audioPrefix: String
    ) throws {
        guard !shardURLs.isEmpty, !audioPrefix.isEmpty else {
            throw EdgeMLXQwen35SessionError.invalidConfiguration
        }
        let duplicatedPaths = shardURLs.map { strdup($0.path)! }
        defer {
            for path in duplicatedPaths {
                free(path)
            }
        }
        let pathPointers: [UnsafePointer<CChar>?] = duplicatedPaths.map { UnsafePointer<CChar>($0) }
        let status = pathPointers.withUnsafeBufferPointer { paths in
            audioPrefix.withCString { prefix in
                edge_cmlx_qwen35_session_load_audio_safetensors(
                    handle,
                    paths.baseAddress,
                    Int32(paths.count),
                    prefix
                )
            }
        }
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to load Qwen3 ASR Cmlx audio safetensors")
        }
    }

    public func audioEncode(
        logMelFeatures: [Float],
        featureShape: [Int]
    ) throws -> EdgeMLXQwen35AudioEncoding {
        guard let audioTowerConfiguration else {
            throw EdgeMLXQwen35SessionError.invalidConfiguration
        }
        guard featureShape.count == 2,
              featureShape[0] > 0,
              featureShape[1] == audioTowerConfiguration.numMelBins,
              logMelFeatures.count == featureShape[0] * featureShape[1]
        else {
            throw EdgeMLXQwen35SessionError.invalidTensorShape
        }

        let outputFrames = Self.qwenASRAudioOutputTokenCapacity(
            frameCount: featureShape[0],
            nWindow: audioTowerConfiguration.nWindow
        )
        var output = Array(
            repeating: Float.zero,
            count: outputFrames * audioTowerConfiguration.outputDim
        )
        var actualFrames = Int32(0)
        var actualHidden = Int32(0)
        let status = output.withUnsafeMutableBufferPointer { outputBuffer in
            logMelFeatures.withUnsafeBufferPointer { featureBuffer in
                edge_cmlx_qwen35_session_audio_encode(
                    handle,
                    featureBuffer.baseAddress,
                    Int32(featureShape[0]),
                    Int32(featureShape[1]),
                    outputBuffer.baseAddress,
                    outputBuffer.count,
                    &actualFrames,
                    &actualHidden
                )
            }
        }
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to run Qwen3 ASR Cmlx audio encoder")
        }
        let shape = [Int(actualFrames), Int(actualHidden)]
        guard shape[0] <= outputFrames,
              shape[1] == audioTowerConfiguration.outputDim
        else {
            throw EdgeMLXQwen35SessionError.invalidTensorShape
        }
        output.removeSubrange((shape[0] * shape[1])..<output.count)
        return EdgeMLXQwen35AudioEncoding(values: output, shape: shape)
    }

    public func visionEncode(
        pixelValues: [Float],
        pixelValuesShape: [Int],
        gridTHW: [QwenImageGridTHW],
        spatialMergeSize: Int,
        outputHiddenSize: Int
    ) throws -> EdgeMLXQwen35VisionEncoding {
        guard pixelValuesShape.count == 2,
              pixelValuesShape[0] > 0,
              pixelValuesShape[1] > 0,
              pixelValues.count == pixelValuesShape[0] * pixelValuesShape[1],
              !gridTHW.isEmpty,
              spatialMergeSize > 0,
              outputHiddenSize > 0
        else {
            throw EdgeMLXQwen35SessionError.invalidTensorShape
        }
        let outputPatches = gridTHW.reduce(0) { partial, grid in
            partial + grid.product / (spatialMergeSize * spatialMergeSize)
        }
        var output = Array(repeating: Float.zero, count: outputPatches * outputHiddenSize)
        var rawGrid = gridTHW.flatMap { [Int32($0.temporal), Int32($0.height), Int32($0.width)] }
        var actualPatches = Int32(0)
        var actualHidden = Int32(0)
        let status = output.withUnsafeMutableBufferPointer { outputBuffer in
            pixelValues.withUnsafeBufferPointer { pixelBuffer in
                rawGrid.withUnsafeMutableBufferPointer { gridBuffer in
                    edge_cmlx_qwen35_session_vision_encode(
                        handle,
                        pixelBuffer.baseAddress,
                        Int32(pixelValuesShape[0]),
                        Int32(pixelValuesShape[1]),
                        gridBuffer.baseAddress,
                        Int32(gridTHW.count),
                        outputBuffer.baseAddress,
                        &actualPatches,
                        &actualHidden
                    )
                }
            }
        }
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to run Qwen3.5 Cmlx vision encoder")
        }
        let shape = [Int(actualPatches), Int(actualHidden)]
        guard shape == [outputPatches, outputHiddenSize] else {
            throw EdgeMLXQwen35SessionError.invalidTensorShape
        }
        return EdgeMLXQwen35VisionEncoding(values: output, shape: shape)
    }

    public func registerQuantizedTensor(
        id: Int,
        tensor: EdgeQuantizedTensor,
        packedBuffer: MTLBuffer,
        packedOffset: Int,
        scalesBuffer: MTLBuffer,
        scalesOffset: Int,
        biasesBuffer: MTLBuffer,
        biasesOffset: Int
    ) throws {
        guard tensor.shape.count == 2,
              tensor.packedShape.count == 2,
              tensor.scaleShape.count == 2,
              packedOffset >= 0,
              scalesOffset >= 0,
              biasesOffset >= 0
        else {
            throw EdgeMLXQwen35SessionError.invalidQuantizedTensorShape
        }

        var descriptor = EdgeCmlxQuantizedTensorDescriptor(
            packed_buffer: Self.opaquePointer(to: packedBuffer),
            packed_offset: packedOffset,
            packed_rows: Int32(tensor.packedShape[0]),
            packed_cols: Int32(tensor.packedShape[1]),
            scales_buffer: Self.opaquePointer(to: scalesBuffer),
            scales_offset: scalesOffset,
            scale_rows: Int32(tensor.scaleShape[0]),
            scale_cols: Int32(tensor.scaleShape[1]),
            biases_buffer: Self.opaquePointer(to: biasesBuffer),
            biases_offset: biasesOffset,
            group_size: Int32(tensor.groupSize),
            bits: Int32(tensor.bits)
        )
        let status = edge_cmlx_qwen35_session_set_quantized_tensor(
            handle,
            Int32(id),
            &descriptor
        )
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to register Qwen3.5 quantized tensor")
        }
    }

    public func evalQuantizedMLP(
        input: EdgeTensor,
        gateTensorID: Int,
        upTensorID: Int,
        downTensorID: Int,
        outputColumns: Int
    ) throws -> EdgeTensor {
        guard input.dataType == .float32,
              input.shape.rank == 2,
              input.shape.dimensions[0] > 0,
              input.shape.dimensions[1] > 0,
              outputColumns > 0
        else {
            throw EdgeMLXQwen35SessionError.invalidTensorShape
        }
        let output = try EdgeTensor(
            shape: EdgeTensorShape([input.shape.dimensions[0], outputColumns]),
            dataType: .float32,
            runtime: runtime
        )
        let status = edge_cmlx_qwen35_session_eval_quantized_mlp_f32_mtl(
            handle,
            Self.opaquePointer(to: input.buffer),
            Int32(input.shape.dimensions[0]),
            Int32(input.shape.dimensions[1]),
            Int32(gateTensorID),
            Int32(upTensorID),
            Int32(downTensorID),
            Self.opaqueMutablePointer(to: output.buffer),
            Int32(outputColumns),
            output.shape.elementCount
        )
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to evaluate Qwen3.5 Cmlx quantized MLP")
        }
        return output
    }

    public func evalQuantizedGDNDecodeLayer(
        input: EdgeTensor,
        layerIndex: Int,
        convState: EdgeTensor,
        recurrentState: EdgeTensor
    ) throws -> EdgeMLXQwen35GDNDecodeLayerOutput {
        guard input.dataType == .float32,
              convState.dataType == .float32,
              recurrentState.dataType == .float32,
              input.shape.dimensions == [1, architecture.hiddenSize],
              convState.shape.dimensions == [
                  architecture.linearConvKernelSize - 1,
                  architecture.linearConvHiddenSize,
              ],
              recurrentState.shape.dimensions == [
                  architecture.linearValueHeadCount,
                  architecture.linearValueHeadDimension,
                  architecture.linearKeyHeadDimension,
              ],
              layerIndex >= 0,
              layerIndex < architecture.layerCount
        else {
            throw EdgeMLXQwen35SessionError.invalidTensorShape
        }

        let output = try EdgeTensor(
            shape: EdgeTensorShape([1, architecture.hiddenSize]),
            dataType: .float32,
            runtime: runtime
        )
        let nextConvState = try EdgeTensor(
            shape: convState.shape,
            dataType: .float32,
            runtime: runtime
        )
        let nextRecurrentState = try EdgeTensor(
            shape: recurrentState.shape,
            dataType: .float32,
            runtime: runtime
        )
        let status = edge_cmlx_qwen35_session_eval_quantized_gdn_decode_layer_f32_mtl(
            handle,
            Int32(layerIndex),
            Self.opaquePointer(to: input.buffer),
            Self.opaquePointer(to: convState.buffer),
            Self.opaquePointer(to: recurrentState.buffer),
            Self.opaqueMutablePointer(to: output.buffer),
            Self.opaqueMutablePointer(to: nextConvState.buffer),
            Self.opaqueMutablePointer(to: nextRecurrentState.buffer),
            output.shape.elementCount
        )
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to evaluate Qwen3.5 Cmlx GDN decode layer")
        }
        return EdgeMLXQwen35GDNDecodeLayerOutput(
            hiddenStates: output,
            nextConvState: nextConvState,
            nextRecurrentState: nextRecurrentState
        )
    }

    public func evalQuantizedFullAttentionDecodeLayer(
        input: EdgeTensor,
        layerIndex: Int,
        positionOffset: Int = 0
    ) throws -> EdgeTensor {
        guard input.dataType == .float32,
              input.shape.dimensions == [1, architecture.hiddenSize],
              layerIndex >= 0,
              layerIndex < architecture.layerCount,
              positionOffset >= 0
        else {
            throw EdgeMLXQwen35SessionError.invalidTensorShape
        }

        let output = try EdgeTensor(
            shape: EdgeTensorShape([1, architecture.hiddenSize]),
            dataType: .float32,
            runtime: runtime
        )
        let status = edge_cmlx_qwen35_session_eval_quantized_full_attention_decode_layer_f32_mtl(
            handle,
            Int32(layerIndex),
            Self.opaquePointer(to: input.buffer),
            Int32(positionOffset),
            Self.opaqueMutablePointer(to: output.buffer),
            output.shape.elementCount
        )
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to evaluate Qwen3.5 Cmlx full-attention decode layer")
        }
        return output
    }

    public func decodeStep(tokenID: Int) throws -> Int {
        var token = Int32(tokenID)
        var output = Int32(0)
        let status = edge_cmlx_qwen35_session_decode_step(
            handle,
            &token,
            1,
            &output
        )
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to decode Qwen3.5 Cmlx token")
        }
        return Int(output)
    }

    public func prefill(tokenIDs: [Int]) throws -> Int {
        guard !tokenIDs.isEmpty else {
            throw EdgeMLXQwen35SessionError.invalidConfiguration
        }
        var tokens = tokenIDs.map { Int32($0) }
        var output = Int32(0)
        let status = tokens.withUnsafeMutableBufferPointer { buffer in
            edge_cmlx_qwen35_session_prefill(
                handle,
                buffer.baseAddress,
                Int32(buffer.count),
                &output
            )
        }
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to prefill Qwen3.5 Cmlx tokens")
        }
        return Int(output)
    }

    public func captureLastHidden(
        tokenIDs: [Int],
        targetLayer: Int
    ) throws -> [Float] {
        guard !tokenIDs.isEmpty,
              targetLayer >= 0,
              targetLayer < architecture.layerCount
        else {
            throw EdgeMLXQwen35SessionError.invalidConfiguration
        }
        var tokens = tokenIDs.map { Int32($0) }
        var output = Array(repeating: Float(0), count: architecture.hiddenSize)
        let status = tokens.withUnsafeMutableBufferPointer { tokenBuffer in
            output.withUnsafeMutableBufferPointer { outputBuffer in
                edge_cmlx_qwen35_session_capture_last_hidden(
                    handle,
                    tokenBuffer.baseAddress,
                    Int32(tokenBuffer.count),
                    Int32(targetLayer),
                    outputBuffer.baseAddress,
                    Int32(outputBuffer.count)
                )
            }
        }
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to capture Qwen3.5 Cmlx hidden state")
        }
        return output
    }

    public func prefillEmbeddings(
        _ embeddings: [Float],
        shape: [Int]
    ) throws -> Int {
        guard shape.count == 2,
              shape[0] > 0,
              shape[1] == architecture.hiddenSize,
              embeddings.count == shape[0] * shape[1]
        else {
            throw EdgeMLXQwen35SessionError.invalidTensorShape
        }
        var output = Int32(0)
        let status = embeddings.withUnsafeBufferPointer { buffer in
            edge_cmlx_qwen35_session_prefill_embeddings(
                handle,
                buffer.baseAddress,
                Int32(shape[0]),
                Int32(shape[1]),
                &output
            )
        }
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to prefill Qwen3.5 Cmlx embeddings")
        }
        return Int(output)
    }

    public func prefillMediaFeatures(
        tokenIDs: [Int],
        mediaFeatures: [Float],
        mediaFeatureShape: [Int],
        mediaTokenID: Int
    ) throws -> Int {
        guard !tokenIDs.isEmpty,
              mediaFeatureShape.count == 2,
              mediaFeatureShape[0] > 0,
              mediaFeatureShape[1] == architecture.hiddenSize,
              mediaFeatures.count == mediaFeatureShape[0] * mediaFeatureShape[1]
        else {
            throw EdgeMLXQwen35SessionError.invalidTensorShape
        }
        var tokens = tokenIDs.map(Int32.init)
        var output = Int32(0)
        let status = tokens.withUnsafeMutableBufferPointer { tokenBuffer in
            mediaFeatures.withUnsafeBufferPointer { featureBuffer in
                edge_cmlx_qwen35_session_prefill_media_features(
                    handle,
                    tokenBuffer.baseAddress,
                    Int32(tokenBuffer.count),
                    featureBuffer.baseAddress,
                    Int32(mediaFeatureShape[0]),
                    Int32(mediaFeatureShape[1]),
                    Int32(mediaTokenID),
                    &output
                )
            }
        }
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to prefill Qwen3.5 Cmlx media features")
        }
        return Int(output)
    }

    public func prefillImageFeatures(
        tokenIDs: [Int],
        imageFeatures: [Float],
        imageFeatureShape: [Int],
        imageTokenID: Int
    ) throws -> Int {
        try prefillMediaFeatures(
            tokenIDs: tokenIDs,
            mediaFeatures: imageFeatures,
            mediaFeatureShape: imageFeatureShape,
            mediaTokenID: imageTokenID
        )
    }

    public func generateTTSCodes(
        targetTokenIDs: [Int],
        ttsBOSTokenID: Int,
        ttsEOSTokenID: Int,
        ttsPADTokenID: Int,
        codecPrefixTokenIDs: [Int],
        codecPADTokenID: Int,
        codecBOSTokenID: Int,
        codecEOSTokenID: Int,
        speakerID: Int?,
        maxTokens: Int,
        temperature: Float = 0.9,
        topK: Int = 50,
        seed: UInt64 = 0
    ) throws -> EdgeMLXTTSCodes {
        guard !targetTokenIDs.isEmpty,
              !codecPrefixTokenIDs.isEmpty,
              maxTokens > 0,
              temperature >= 0,
              temperature.isFinite,
              topK >= 0
        else {
            throw EdgeMLXQwen35SessionError.invalidConfiguration
        }
        var targetTokens = targetTokenIDs.map(Int32.init)
        var codecPrefixTokens = codecPrefixTokenIDs.map(Int32.init)
        var outputCodes = Array(repeating: Int32(0), count: maxTokens * 16)
        var outputSteps = Int32(0)
        var outputCodebooks = Int32(0)
        let status = targetTokens.withUnsafeMutableBufferPointer { targetBuffer in
            codecPrefixTokens.withUnsafeMutableBufferPointer { prefixBuffer in
                outputCodes.withUnsafeMutableBufferPointer { outputBuffer in
                    edge_cmlx_qwen35_session_tts_generate_codes(
                        handle,
                        targetBuffer.baseAddress,
                        Int32(targetBuffer.count),
                        Int32(ttsBOSTokenID),
                        Int32(ttsEOSTokenID),
                        Int32(ttsPADTokenID),
                        prefixBuffer.baseAddress,
                        Int32(prefixBuffer.count),
                        Int32(codecPADTokenID),
                        Int32(codecBOSTokenID),
                        Int32(codecEOSTokenID),
                        Int32(speakerID ?? -1),
                        Int32(maxTokens),
                        temperature,
                        Int32(topK),
                        seed,
                        outputBuffer.baseAddress,
                        Int32(outputBuffer.count),
                        &outputSteps,
                        &outputCodebooks
                    )
                }
            }
        }
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to generate Qwen3 TTS codec tokens")
        }
        let steps = Int(outputSteps)
        let codebooks = Int(outputCodebooks)
        let count = steps * codebooks
        return EdgeMLXTTSCodes(
            values: outputCodes.prefix(count).map(Int.init),
            shape: [steps, codebooks]
        )
    }

    public func decodeTTSAudio(
        codes: EdgeMLXTTSCodes,
        sampleRate: Int = 24_000,
        decodeUpsampleRate: Int = 1_920
    ) throws -> EdgeAudioBuffer {
        guard codes.shape.count == 2,
              codes.shape[0] > 0,
              codes.shape[1] == 16,
              codes.values.count == codes.shape[0] * codes.shape[1],
              sampleRate > 0,
              decodeUpsampleRate > 0
        else {
            throw EdgeMLXQwen35SessionError.invalidTensorShape
        }
        var rawCodes = codes.values.map(Int32.init)
        var samples = Array(repeating: Float.zero, count: codes.shape[0] * decodeUpsampleRate)
        var sampleCount = Int32(0)
        let status = rawCodes.withUnsafeMutableBufferPointer { codeBuffer in
            samples.withUnsafeMutableBufferPointer { sampleBuffer in
                edge_cmlx_qwen35_session_tts_decode_audio_codes(
                    handle,
                    codeBuffer.baseAddress,
                    Int32(codes.shape[0]),
                    Int32(codes.shape[1]),
                    sampleBuffer.baseAddress,
                    Int32(sampleBuffer.count),
                    &sampleCount
                )
            }
        }
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to decode Qwen3 TTS audio")
        }
        let count = Int(sampleCount)
        guard count > 0, count <= samples.count else {
            throw EdgeMLXQwen35SessionError.invalidTensorShape
        }
        samples.removeSubrange(count..<samples.count)
        return try EdgeAudioBuffer(
            sampleRate: sampleRate,
            channelCount: 1,
            interleavedSamples: samples
        )
    }

    public func prefillAsync(tokenIDs: [Int]) throws {
        guard !tokenIDs.isEmpty else {
            throw EdgeMLXQwen35SessionError.invalidConfiguration
        }
        var tokens = tokenIDs.map { Int32($0) }
        let status = tokens.withUnsafeMutableBufferPointer { buffer in
            edge_cmlx_qwen35_session_prefill_async(
                handle,
                buffer.baseAddress,
                Int32(buffer.count)
            )
        }
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to asynchronously prefill Qwen3.5 Cmlx tokens")
        }
    }

    public func prefillSampledAsync(
        tokenIDs: [Int],
        temperature: Float,
        topK: Int? = nil,
        topP: Float,
        minP: Float = 0,
        seed: UInt64
    ) throws {
        guard !tokenIDs.isEmpty else {
            throw EdgeMLXQwen35SessionError.invalidConfiguration
        }
        var tokens = tokenIDs.map { Int32($0) }
        let status = tokens.withUnsafeMutableBufferPointer { buffer in
            edge_cmlx_qwen35_session_prefill_sampled_async(
                handle,
                buffer.baseAddress,
                Int32(buffer.count),
                temperature,
                Int32(topK ?? 0),
                topP,
                minP,
                seed
            )
        }
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to asynchronously prefill sampled Qwen3.5 Cmlx tokens")
        }
    }

    public func synchronize() throws {
        let status = edge_cmlx_qwen35_session_synchronize(handle)
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to synchronize Qwen3.5 Cmlx session")
        }
    }

    public func nextToken() throws -> Int {
        var output = Int32(0)
        let status = edge_cmlx_qwen35_session_next_token(handle, &output)
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to advance Qwen3.5 Cmlx token iterator")
        }
        return Int(output)
    }

    public func nextSampledToken(
        temperature: Float,
        topK: Int? = nil,
        topP: Float,
        minP: Float = 0,
        seed: UInt64
    ) throws -> Int {
        var output = Int32(0)
        let status = edge_cmlx_qwen35_session_next_sampled_token(
            handle,
            temperature,
            Int32(topK ?? 0),
            topP,
            minP,
            seed,
            &output
        )
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to advance sampled Qwen3.5 Cmlx token iterator")
        }
        return Int(output)
    }

    public func setRepetitionPenalty(
        _ penalty: Float,
        contextTokenIds: [Int]
    ) throws {
        let cTokenIds = contextTokenIds.map { Int32($0) }
        let status = cTokenIds.withUnsafeBufferPointer { buffer in
            edge_cmlx_qwen35_session_set_repetition_penalty(
                handle,
                penalty,
                buffer.baseAddress,
                Int32(buffer.count)
            )
        }
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to set Qwen3.5 Cmlx repetition penalty")
        }
    }

    public func setSamplingPenalties(
        repetitionPenalty: Float,
        repetitionContextTokenIds: [Int],
        presencePenalty: Float,
        presenceContextTokenIds: [Int],
        frequencyPenalty: Float,
        frequencyContextTokenIds: [Int]
    ) throws {
        let repetitionTokenIds = repetitionContextTokenIds.map { Int32($0) }
        let presenceTokenIds = presenceContextTokenIds.map { Int32($0) }
        let frequencyTokenIds = frequencyContextTokenIds.map { Int32($0) }
        let status = repetitionTokenIds.withUnsafeBufferPointer { repetitionBuffer in
            presenceTokenIds.withUnsafeBufferPointer { presenceBuffer in
                frequencyTokenIds.withUnsafeBufferPointer { frequencyBuffer in
                    edge_cmlx_qwen35_session_set_sampling_penalties(
                        handle,
                        repetitionPenalty,
                        repetitionBuffer.baseAddress,
                        Int32(repetitionBuffer.count),
                        presencePenalty,
                        presenceBuffer.baseAddress,
                        Int32(presenceBuffer.count),
                        frequencyPenalty,
                        frequencyBuffer.baseAddress,
                        Int32(frequencyBuffer.count)
                    )
                }
            }
        }
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to set Qwen3.5 Cmlx sampling penalties")
        }
    }

    public func setEOSSamplingBias(
        tokenIds: [Int],
        suppress: Bool,
        logitPenalty: Float
    ) throws {
        let cTokenIds = tokenIds.map { Int32($0) }
        let status = cTokenIds.withUnsafeBufferPointer { buffer in
            edge_cmlx_qwen35_session_set_eos_sampling_bias(
                handle,
                buffer.baseAddress,
                Int32(buffer.count),
                suppress ? 1 : 0,
                logitPenalty
            )
        }
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to set Qwen3.5 Cmlx EOS sampling bias")
        }
    }

    public func clearEOSSamplingBias() throws {
        let status = edge_cmlx_qwen35_session_clear_eos_sampling_bias(handle)
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to clear Qwen3.5 Cmlx EOS sampling bias")
        }
    }

    public func clearRepetitionPenalty() throws {
        let status = edge_cmlx_qwen35_session_clear_repetition_penalty(handle)
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to clear Qwen3.5 Cmlx repetition penalty")
        }
    }

    public func lastSampleDiagnostics() throws -> String? {
        var output = Array(repeating: CChar(0), count: 1024)
        let status = output.withUnsafeMutableBufferPointer { buffer in
            edge_cmlx_qwen35_session_copy_last_sample_diagnostics(
                handle,
                buffer.baseAddress,
                Int32(buffer.count)
            )
        }
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to read Qwen3.5 Cmlx sample diagnostics")
        }
        let value = output.withUnsafeBufferPointer { buffer in
            String(cString: buffer.baseAddress!)
        }
        return value.isEmpty ? nil : value
    }

    public func memorySummary() throws -> String? {
        var output = Array(repeating: CChar(0), count: 8192)
        let status = output.withUnsafeMutableBufferPointer { buffer in
            edge_cmlx_qwen35_session_copy_memory_summary(
                handle,
                buffer.baseAddress,
                Int32(buffer.count)
            )
        }
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to read Qwen3.5 Cmlx memory summary")
        }
        let value = output.withUnsafeBufferPointer { buffer in
            String(cString: buffer.baseAddress!)
        }
        return value.isEmpty ? nil : value
    }

    public func resetEvalProfile() throws {
        let status = edge_cmlx_qwen35_session_reset_eval_profile(handle)
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to reset Qwen3.5 Cmlx eval profile")
        }
    }

    public func evalProfileSummary() throws -> String? {
        var output = Array(repeating: CChar(0), count: 4096)
        let status = output.withUnsafeMutableBufferPointer { buffer in
            edge_cmlx_qwen35_session_copy_eval_profile(
                handle,
                buffer.baseAddress,
                Int32(buffer.count)
            )
        }
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to read Qwen3.5 Cmlx eval profile")
        }
        let value = output.withUnsafeBufferPointer { buffer in
            String(cString: buffer.baseAddress!)
        }
        return value.isEmpty ? nil : value
    }

    public func clearDSRPolicies() throws {
        let status = edge_cmlx_qwen35_session_clear_dsr_policies(handle)
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to clear Qwen3.5 Cmlx DSR policies")
        }
    }

    public func setDSRPolicy(
        _ policy: QwenDSRKVCachePolicy,
        layerIndex: Int
    ) throws {
        var cPolicy = EdgeCmlxQwen35DSRPolicy(
            max_size: Int32(policy.maxSize),
            heavy_budget: Int32(policy.heavyBudget),
            recent_budget: Int32(policy.recentBudget),
            sink_size: Int32(policy.sinkSize),
            eviction_interval: Int32(policy.evictionInterval),
            score_activation_ratio: policy.scoreActivationRatio,
            score_decay: policy.scoreDecay
        )
        let status = edge_cmlx_qwen35_session_set_dsr_policy(
            handle,
            Int32(layerIndex),
            &cPolicy
        )
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to set Qwen3.5 Cmlx DSR policy")
        }
    }

    public func updateDSRPolicyFields(
        _ policy: QwenDSRKVCachePolicy,
        layerIndex: Int
    ) throws {
        var cPolicy = EdgeCmlxQwen35DSRPolicy(
            max_size: Int32(policy.maxSize),
            heavy_budget: Int32(policy.heavyBudget),
            recent_budget: Int32(policy.recentBudget),
            sink_size: Int32(policy.sinkSize),
            eviction_interval: Int32(policy.evictionInterval),
            score_activation_ratio: policy.scoreActivationRatio,
            score_decay: policy.scoreDecay
        )
        let status = edge_cmlx_qwen35_session_update_dsr_policy_fields(
            handle,
            Int32(layerIndex),
            &cPolicy
        )
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to update Qwen3.5 Cmlx DSR policy fields")
        }
    }

    public func setDSRPolicies(_ policies: [Int: QwenDSRKVCachePolicy]) throws {
        try clearDSRPolicies()
        for (layerIndex, policy) in policies.sorted(by: { $0.key < $1.key }) {
            try setDSRPolicy(policy, layerIndex: layerIndex)
        }
    }

    public func setAttentionCacheQuantization(
        groupSize: Int?,
        bits: Int?
    ) throws {
        let cGroupSize = Int32(max(0, groupSize ?? 0))
        let cBits = Int32(max(0, bits ?? 0))
        let status = edge_cmlx_qwen35_session_set_attention_cache_quantization(
            handle,
            cGroupSize,
            cBits
        )
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to configure Qwen3.5 Cmlx attention cache quantization")
        }
    }

    public func setFrogJumpLayerMask(_ layerMask: UInt64) throws {
        let status = edge_cmlx_qwen35_session_set_frog_jump_mask(
            handle,
            layerMask
        )
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to configure Qwen3.5 Cmlx frog jump mask")
        }
    }

    public func setAttentionCacheLimit(maxTokens: Int?) throws {
        let limit = Int32(max(0, maxTokens ?? 0))
        let status = edge_cmlx_qwen35_session_set_attention_cache_limit(handle, limit)
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to configure Qwen3.5 Cmlx attention cache limit")
        }
    }

    public func resetDecodeCache() throws {
        let status = edge_cmlx_qwen35_session_reset_decode_cache(handle)
        guard status == 0 else {
            throw Self.currentError(defaultMessage: "failed to reset Qwen3.5 Cmlx decode cache")
        }
    }

    private func makeBuffer<T>(
        values: [T],
        label: String
    ) throws -> MTLBuffer {
        let byteCount = values.count * MemoryLayout<T>.stride
        guard byteCount > 0 else {
            throw EdgeMLXQwen35SessionError.allocationFailed(byteCount: byteCount)
        }
        guard let buffer = values.withUnsafeBytes({ bytes in
            runtime.device.makeBuffer(
                bytes: bytes.baseAddress!,
                length: byteCount,
                options: [.storageModeShared]
            )
        }) else {
            throw EdgeMLXQwen35SessionError.allocationFailed(byteCount: byteCount)
        }
        buffer.label = label
        return buffer
    }

    private static func opaquePointer(to buffer: MTLBuffer) -> UnsafeRawPointer {
        UnsafeRawPointer(Unmanaged.passUnretained(buffer as AnyObject).toOpaque())
    }

    private static func opaqueMutablePointer(to buffer: MTLBuffer) -> UnsafeMutableRawPointer {
        Unmanaged.passUnretained(buffer as AnyObject).toOpaque()
    }

    private static func qwenASRAudioOutputTokenCapacity(frameCount: Int, nWindow: Int) -> Int {
        let chunkSize = nWindow * 2
        var offset = 0
        var total = 0
        while offset < frameCount {
            let length = min(chunkSize, frameCount - offset)
            var outputLength = length
            for _ in 0..<3 {
                outputLength = (outputLength + 1) / 2
            }
            total += max(1, outputLength)
            offset += length
        }
        return max(1, total)
    }

    private static func currentError(defaultMessage: String) -> EdgeMLXQwen35SessionError {
        let message = edge_cmlx_last_error().map { String(cString: $0) } ?? defaultMessage
        return .executionFailed(message)
    }
}
