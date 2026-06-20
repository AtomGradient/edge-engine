// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public enum QwenGDNConvWeightLayout: Equatable, Sendable {
    case mlxSanitizedDepthwise
    case huggingFaceDepthwise
}

public struct QwenGDNProjectionOutputs {
    /// Stacked pre-conv projection with `[q | k | v]` column layout.
    public var mixedQKV: EdgeTensor?
    /// Pre-conv query slice. GDN recurrent math consumes post-conv query from `QwenGDNConvolutionOutputs`.
    public var query: EdgeTensor
    /// Pre-conv key slice. GDN recurrent math consumes post-conv key from `QwenGDNConvolutionOutputs`.
    public var key: EdgeTensor
    /// Pre-conv value slice. GDN recurrent math consumes post-conv value from `QwenGDNConvolutionOutputs`.
    public var value: EdgeTensor
    public var z: EdgeTensor
    public var a: EdgeTensor
    public var b: EdgeTensor

    public init(
        query: EdgeTensor,
        key: EdgeTensor,
        value: EdgeTensor,
        z: EdgeTensor,
        a: EdgeTensor,
        b: EdgeTensor,
        mixedQKV: EdgeTensor? = nil
    ) {
        self.mixedQKV = mixedQKV
        self.query = query
        self.key = key
        self.value = value
        self.z = z
        self.a = a
        self.b = b
    }
}

public struct QwenGDNFusedProjectionOutputs {
    /// Stacked pre-conv projection with `[q | k | v]` column layout.
    public var mixedQKV: EdgeTensor
    public var z: EdgeTensor
    public var a: EdgeTensor
    public var b: EdgeTensor

    public init(
        mixedQKV: EdgeTensor,
        z: EdgeTensor,
        a: EdgeTensor,
        b: EdgeTensor
    ) {
        self.mixedQKV = mixedQKV
        self.z = z
        self.a = a
        self.b = b
    }
}

public struct QwenGDNConvolutionOutputs {
    /// Post-conv query after depthwise Conv1d + SiLU.
    public var query: EdgeTensor
    /// Post-conv key after depthwise Conv1d + SiLU.
    public var key: EdgeTensor
    /// Post-conv value after depthwise Conv1d + SiLU.
    public var value: EdgeTensor
    public var nextConvState: EdgeTensor

    public init(
        query: EdgeTensor,
        key: EdgeTensor,
        value: EdgeTensor,
        nextConvState: EdgeTensor
    ) {
        self.query = query
        self.key = key
        self.value = value
        self.nextConvState = nextConvState
    }
}

public struct QwenGDNNormalizedConvolutionOutputs {
    /// Post-conv query after per-head RMSNorm and Qwen GDN query scaling.
    public var query: EdgeTensor
    /// Post-conv key after per-head RMSNorm and Qwen GDN key scaling.
    public var key: EdgeTensor
    public var value: EdgeTensor
    public var nextConvState: EdgeTensor

    public init(
        query: EdgeTensor,
        key: EdgeTensor,
        value: EdgeTensor,
        nextConvState: EdgeTensor
    ) {
        self.query = query
        self.key = key
        self.value = value
        self.nextConvState = nextConvState
    }
}

public struct QwenGDNRecurrentOutputs {
    public var output: EdgeTensor
    public var nextRecurrentState: EdgeTensor

    public init(output: EdgeTensor, nextRecurrentState: EdgeTensor) {
        self.output = output
        self.nextRecurrentState = nextRecurrentState
    }
}

public struct QwenGDNForwardOutputs {
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

public enum QwenGDNForwardError: Error, Equatable {
    case missingMixedQKV
    case invalidMixedQKVShape(expected: [Int], actual: [Int])
}

public struct QwenGDNWeights {
    public var layerIndex: Int
    public var linearKeyHeadCount: Int
    public var linearValueHeadCount: Int
    public var linearKeyHeadDimension: Int
    public var linearValueHeadDimension: Int
    public var linearKeyHiddenSize: Int
    public var linearValueHiddenSize: Int
    public var rmsNormEpsilon: Float
    public var inProjQKV: EdgeTensor
    public var inProjZ: EdgeTensor
    public var inProjB: EdgeTensor
    public var inProjA: EdgeTensor
    public var conv1D: EdgeTensor
    public var convWeightLayout: QwenGDNConvWeightLayout
    public var aLog: EdgeTensor
    public var dtBias: EdgeTensor
    public var norm: EdgeTensor
    public var outProj: EdgeTensor

    public init(
        layerIndex: Int,
        inProjQKV: EdgeTensor,
        inProjZ: EdgeTensor,
        inProjB: EdgeTensor,
        inProjA: EdgeTensor,
        conv1D: EdgeTensor,
        convWeightLayout: QwenGDNConvWeightLayout,
        aLog: EdgeTensor,
        dtBias: EdgeTensor,
        norm: EdgeTensor,
        outProj: EdgeTensor,
        linearKeyHeadCount: Int? = nil,
        linearValueHeadCount: Int? = nil,
        linearKeyHeadDimension: Int? = nil,
        linearValueHeadDimension: Int? = nil,
        linearKeyHiddenSize: Int? = nil,
        linearValueHiddenSize: Int? = nil,
        rmsNormEpsilon: Float = 1e-6
    ) {
        self.layerIndex = layerIndex
        self.linearKeyHeadCount = linearKeyHeadCount ?? 1
        self.linearValueHeadCount = linearValueHeadCount ?? aLog.shape.dimensions[0]
        self.linearValueHeadDimension = linearValueHeadDimension ?? norm.shape.dimensions[0]
        self.linearValueHiddenSize = linearValueHiddenSize ?? inProjZ.shape.dimensions[1]
        self.linearKeyHiddenSize = linearKeyHiddenSize ?? ((inProjQKV.shape.dimensions[1] - self.linearValueHiddenSize) / 2)
        self.linearKeyHeadDimension = linearKeyHeadDimension ?? (self.linearKeyHiddenSize / self.linearKeyHeadCount)
        self.rmsNormEpsilon = rmsNormEpsilon
        self.inProjQKV = inProjQKV
        self.inProjZ = inProjZ
        self.inProjB = inProjB
        self.inProjA = inProjA
        self.conv1D = conv1D
        self.convWeightLayout = convWeightLayout
        self.aLog = aLog
        self.dtBias = dtBias
        self.norm = norm
        self.outProj = outProj
    }

    public static func loadHuggingFaceLayout(
        layerIndex: Int,
        weightStore: QwenModelWeightStore,
        runtime: EdgeMetalRuntime
    ) throws -> QwenGDNWeights {
        let architecture = weightStore.bundleIndex.architecture
        let manifest = try weightStore.bundleIndex.layerManifest(layerIndex)
        guard manifest.kind == .gdn else {
            throw QwenProjectionWeightError.layerIsNotGDN(layerIndex: layerIndex, kind: manifest.kind)
        }

        let prefix = "\(manifest.layerPrefix).linear_attn"
        let inProjQKVName = "\(prefix).in_proj_qkv.weight"
        let inProjZName = "\(prefix).in_proj_z.weight"
        let inProjBName = "\(prefix).in_proj_b.weight"
        let inProjAName = "\(prefix).in_proj_a.weight"
        let convName = "\(prefix).conv1d.weight"
        let aLogName = "\(prefix).A_log"
        let dtBiasName = "\(prefix).dt_bias"
        let normName = "\(prefix).norm.weight"
        let outProjName = "\(prefix).out_proj.weight"

        let inProjQKV = try weightStore.loadFloat32TensorTransposed2D(
            named: inProjQKVName,
            runtime: runtime
        )
        let inProjZ = try weightStore.loadFloat32TensorTransposed2D(
            named: inProjZName,
            runtime: runtime
        )
        let inProjB = try weightStore.loadFloat32TensorTransposed2D(
            named: inProjBName,
            runtime: runtime
        )
        let inProjA = try weightStore.loadFloat32TensorTransposed2D(
            named: inProjAName,
            runtime: runtime
        )
        let conv = try loadConv1DWeight(
            name: convName,
            weightStore: weightStore,
            architecture: architecture,
            runtime: runtime
        )
        let aLog = try weightStore.loadFloat32Tensor(named: aLogName, runtime: runtime)
        let dtBias = try weightStore.loadFloat32Tensor(named: dtBiasName, runtime: runtime)
        let norm = try weightStore.loadFloat32Tensor(named: normName, runtime: runtime)
        let outProj = try weightStore.loadFloat32TensorTransposed2D(
            named: outProjName,
            runtime: runtime
        )

        try validate(inProjQKV, name: inProjQKVName, expectedShape: [
            architecture.hiddenSize,
            architecture.linearQKVHiddenSize,
        ])
        try validate(inProjZ, name: inProjZName, expectedShape: [
            architecture.hiddenSize,
            architecture.linearValueHiddenSize,
        ])
        try validate(inProjB, name: inProjBName, expectedShape: [
            architecture.hiddenSize,
            architecture.linearValueHeadCount,
        ])
        try validate(inProjA, name: inProjAName, expectedShape: [
            architecture.hiddenSize,
            architecture.linearValueHeadCount,
        ])
        try validate(aLog, name: aLogName, expectedShape: [architecture.linearValueHeadCount])
        try validate(dtBias, name: dtBiasName, expectedShape: [architecture.linearValueHeadCount])
        try validate(norm, name: normName, expectedShape: [architecture.linearValueHeadDimension])
        try validate(outProj, name: outProjName, expectedShape: [
            architecture.linearValueHiddenSize,
            architecture.hiddenSize,
        ])

        return QwenGDNWeights(
            layerIndex: layerIndex,
            inProjQKV: inProjQKV,
            inProjZ: inProjZ,
            inProjB: inProjB,
            inProjA: inProjA,
            conv1D: conv.tensor,
            convWeightLayout: conv.layout,
            aLog: aLog,
            dtBias: dtBias,
            norm: norm,
            outProj: outProj,
            linearKeyHeadCount: architecture.linearKeyHeadCount,
            linearValueHeadCount: architecture.linearValueHeadCount,
            linearKeyHeadDimension: architecture.linearKeyHeadDimension,
            linearValueHeadDimension: architecture.linearValueHeadDimension,
            linearKeyHiddenSize: architecture.linearKeyHiddenSize,
            linearValueHiddenSize: architecture.linearValueHiddenSize,
            rmsNormEpsilon: architecture.rmsNormEpsilon
        )
    }

    public func project(
        hiddenStates: EdgeTensor,
        executor: MetalKernelExecutor
    ) throws -> QwenGDNProjectionOutputs {
        let qkv = try executor.matmul(hiddenStates, inProjQKV)
        let queryAndRest = try executor.splitColumns(qkv, firstColumnCount: linearKeyHiddenSize)
        let keyAndValue = try executor.splitColumns(
            queryAndRest.second,
            firstColumnCount: linearKeyHiddenSize
        )
        return QwenGDNProjectionOutputs(
            query: queryAndRest.first,
            key: keyAndValue.first,
            value: keyAndValue.second,
            z: try executor.matmul(hiddenStates, inProjZ),
            a: try executor.matmul(hiddenStates, inProjA),
            b: try executor.matmul(hiddenStates, inProjB),
            mixedQKV: qkv
        )
    }

    public func convolve(
        projections: QwenGDNProjectionOutputs,
        convState: EdgeTensor,
        executor: MetalKernelExecutor
    ) throws -> QwenGDNConvolutionOutputs {
        guard let mixedQKV = projections.mixedQKV else {
            throw QwenGDNForwardError.missingMixedQKV
        }
        let tokenCount = mixedQKV.shape.dimensions[0]
        let expectedShape = [
            tokenCount,
            linearKeyHiddenSize * 2 + linearValueHiddenSize,
        ]
        guard mixedQKV.shape.dimensions == expectedShape else {
            throw QwenGDNForwardError.invalidMixedQKVShape(
                expected: expectedShape,
                actual: mixedQKV.shape.dimensions
            )
        }

        let conv = try executor.gdnDepthwiseConv1D(
            input: mixedQKV,
            weights: conv1D,
            convState: convState
        )
        let queryAndRest = try executor.splitColumns(
            conv.activated,
            firstColumnCount: linearKeyHiddenSize
        )
        let keyAndValue = try executor.splitColumns(
            queryAndRest.second,
            firstColumnCount: linearKeyHiddenSize
        )
        return QwenGDNConvolutionOutputs(
            query: queryAndRest.first,
            key: keyAndValue.first,
            value: keyAndValue.second,
            nextConvState: conv.nextConvState
        )
    }

    public func normalize(
        convolutionOutputs: QwenGDNConvolutionOutputs,
        executor: MetalKernelExecutor
    ) throws -> QwenGDNNormalizedConvolutionOutputs {
        let normalized = try executor.gdnNormalizeQK(
            query: convolutionOutputs.query,
            key: convolutionOutputs.key,
            headCount: linearKeyHeadCount,
            headDimension: linearKeyHeadDimension
        )
        return QwenGDNNormalizedConvolutionOutputs(
            query: normalized.query,
            key: normalized.key,
            value: convolutionOutputs.value,
            nextConvState: convolutionOutputs.nextConvState
        )
    }

    public func recurrentUpdate(
        convolutionOutputs: QwenGDNConvolutionOutputs,
        projections: QwenGDNProjectionOutputs,
        recurrentState: EdgeTensor,
        executor: MetalKernelExecutor
    ) throws -> QwenGDNRecurrentOutputs {
        let normalizedOutputs = try normalize(
            convolutionOutputs: convolutionOutputs,
            executor: executor
        )
        return try recurrentUpdate(
            normalizedConvolutionOutputs: normalizedOutputs,
            projections: projections,
            recurrentState: recurrentState,
            executor: executor
        )
    }

    public func recurrentUpdate(
        normalizedConvolutionOutputs: QwenGDNNormalizedConvolutionOutputs,
        projections: QwenGDNProjectionOutputs,
        recurrentState: EdgeTensor,
        executor: MetalKernelExecutor
    ) throws -> QwenGDNRecurrentOutputs {
        let update = try executor.gdnRecurrentUpdate(
            query: normalizedConvolutionOutputs.query,
            key: normalizedConvolutionOutputs.key,
            value: normalizedConvolutionOutputs.value,
            a: projections.a,
            b: projections.b,
            aLog: aLog,
            dtBias: dtBias,
            state: recurrentState,
            keyHeadCount: linearKeyHeadCount,
            valueHeadCount: linearValueHeadCount,
            keyHeadDimension: linearKeyHeadDimension,
            valueHeadDimension: linearValueHeadDimension
        )
        return QwenGDNRecurrentOutputs(
            output: update.output,
            nextRecurrentState: update.nextState
        )
    }

    public func output(
        recurrentOutputs: QwenGDNRecurrentOutputs,
        projections: QwenGDNProjectionOutputs,
        executor: MetalKernelExecutor
    ) throws -> EdgeTensor {
        let normalized = try executor.rmsNormByHead(
            recurrentOutputs.output,
            weight: norm,
            headCount: linearValueHeadCount,
            headDimension: linearValueHeadDimension,
            epsilon: rmsNormEpsilon
        )
        let gated = try executor.siluMultiply(gate: projections.z, up: normalized)
        return try executor.matmul(gated, outProj)
    }

    public func callAsFunction(
        hiddenStates: EdgeTensor,
        convState: EdgeTensor,
        recurrentState: EdgeTensor,
        executor: MetalKernelExecutor
    ) throws -> QwenGDNForwardOutputs {
        let projections = try project(hiddenStates: hiddenStates, executor: executor)
        let convolved = try convolve(
            projections: projections,
            convState: convState,
            executor: executor
        )
        let recurrent = try recurrentUpdate(
            convolutionOutputs: convolved,
            projections: projections,
            recurrentState: recurrentState,
            executor: executor
        )
        let hiddenOutput = try output(
            recurrentOutputs: recurrent,
            projections: projections,
            executor: executor
        )
        return QwenGDNForwardOutputs(
            hiddenStates: hiddenOutput,
            nextConvState: convolved.nextConvState,
            nextRecurrentState: recurrent.nextRecurrentState
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

    fileprivate static func validateConv1D(
        _ tensor: EdgeTensor,
        name: String,
        architecture: QwenHybridArchitecture
    ) throws -> QwenGDNConvWeightLayout {
        let sanitizedShape = [
            architecture.linearConvHiddenSize,
            architecture.linearConvKernelSize,
            1,
        ]
        if tensor.shape.dimensions == sanitizedShape {
            return .mlxSanitizedDepthwise
        }

        let huggingFaceShape = [
            architecture.linearConvHiddenSize,
            1,
            architecture.linearConvKernelSize,
        ]
        if tensor.shape.dimensions == huggingFaceShape {
            return .huggingFaceDepthwise
        }

        throw QwenProjectionWeightError.invalidWeightShape(
            name: name,
            expected: sanitizedShape,
            actual: tensor.shape.dimensions
        )
    }

    fileprivate static func loadConv1DWeight(
        name: String,
        weightStore: QwenModelWeightStore,
        architecture: QwenHybridArchitecture,
        runtime: EdgeMetalRuntime
    ) throws -> (tensor: EdgeTensor, layout: QwenGDNConvWeightLayout) {
        let tensor = try weightStore.loadFloat32Tensor(named: name, runtime: runtime)
        let layout = try validateConv1D(tensor, name: name, architecture: architecture)
        switch layout {
        case .mlxSanitizedDepthwise:
            return (tensor, layout)
        case .huggingFaceDepthwise:
            return (
                try EdgeTensor(
                    float32: tensor.readFloat32(),
                    shape: EdgeTensorShape([
                        architecture.linearConvHiddenSize,
                        architecture.linearConvKernelSize,
                        1,
                    ]),
                    runtime: runtime
                ),
                layout
            )
        }
    }
}

public struct QwenQuantizedGDNWeights {
    public var layerIndex: Int
    public var linearKeyHeadCount: Int
    public var linearValueHeadCount: Int
    public var linearKeyHeadDimension: Int
    public var linearValueHeadDimension: Int
    public var linearKeyHiddenSize: Int
    public var linearValueHiddenSize: Int
    public var rmsNormEpsilon: Float
    public var inProjQKV: EdgeQuantizedTensor
    public var inProjZ: EdgeQuantizedTensor
    public var inProjB: EdgeQuantizedTensor
    public var inProjA: EdgeQuantizedTensor
    public var conv1D: EdgeTensor
    public var convWeightLayout: QwenGDNConvWeightLayout
    public var aLog: EdgeTensor
    public var dtBias: EdgeTensor
    public var norm: EdgeTensor
    public var outProj: EdgeQuantizedTensor

    public init(
        layerIndex: Int,
        inProjQKV: EdgeQuantizedTensor,
        inProjZ: EdgeQuantizedTensor,
        inProjB: EdgeQuantizedTensor,
        inProjA: EdgeQuantizedTensor,
        conv1D: EdgeTensor,
        convWeightLayout: QwenGDNConvWeightLayout,
        aLog: EdgeTensor,
        dtBias: EdgeTensor,
        norm: EdgeTensor,
        outProj: EdgeQuantizedTensor,
        linearKeyHeadCount: Int? = nil,
        linearValueHeadCount: Int? = nil,
        linearKeyHeadDimension: Int? = nil,
        linearValueHeadDimension: Int? = nil,
        linearKeyHiddenSize: Int? = nil,
        linearValueHiddenSize: Int? = nil,
        rmsNormEpsilon: Float = 1e-6
    ) {
        self.layerIndex = layerIndex
        self.linearKeyHeadCount = linearKeyHeadCount ?? 1
        self.linearValueHeadCount = linearValueHeadCount ?? aLog.shape.dimensions[0]
        self.linearValueHeadDimension = linearValueHeadDimension ?? norm.shape.dimensions[0]
        self.linearValueHiddenSize = linearValueHiddenSize ?? inProjZ.shape[0]
        self.linearKeyHiddenSize = linearKeyHiddenSize ?? ((inProjQKV.shape[0] - self.linearValueHiddenSize) / 2)
        self.linearKeyHeadDimension = linearKeyHeadDimension ?? (self.linearKeyHiddenSize / self.linearKeyHeadCount)
        self.rmsNormEpsilon = rmsNormEpsilon
        self.inProjQKV = inProjQKV
        self.inProjZ = inProjZ
        self.inProjB = inProjB
        self.inProjA = inProjA
        self.conv1D = conv1D
        self.convWeightLayout = convWeightLayout
        self.aLog = aLog
        self.dtBias = dtBias
        self.norm = norm
        self.outProj = outProj
    }

    public static func loadHuggingFaceLayout(
        layerIndex: Int,
        weightStore: QwenModelWeightStore,
        runtime: EdgeMetalRuntime,
        groupSize: Int? = nil,
        bits: Int? = nil
    ) throws -> QwenQuantizedGDNWeights {
        let architecture = weightStore.bundleIndex.architecture
        let quantization = try resolveQuantization(
            architecture: architecture,
            groupSize: groupSize,
            bits: bits
        )
        let manifest = try weightStore.bundleIndex.layerManifest(layerIndex)
        guard manifest.kind == .gdn else {
            throw QwenProjectionWeightError.layerIsNotGDN(layerIndex: layerIndex, kind: manifest.kind)
        }

        let prefix = "\(manifest.layerPrefix).linear_attn"
        let inProjQKVName = "\(prefix).in_proj_qkv.weight"
        let inProjZName = "\(prefix).in_proj_z.weight"
        let inProjBName = "\(prefix).in_proj_b.weight"
        let inProjAName = "\(prefix).in_proj_a.weight"
        let convName = "\(prefix).conv1d.weight"
        let aLogName = "\(prefix).A_log"
        let dtBiasName = "\(prefix).dt_bias"
        let normName = "\(prefix).norm.weight"
        let outProjName = "\(prefix).out_proj.weight"

        let inProjQKV = try loadQuantizedWeight(
            inProjQKVName,
            manifest: manifest,
            weightStore: weightStore,
            groupSize: quantization.groupSize,
            bits: quantization.bits
        )
        let inProjZ = try loadQuantizedWeight(
            inProjZName,
            manifest: manifest,
            weightStore: weightStore,
            groupSize: quantization.groupSize,
            bits: quantization.bits
        )
        let inProjB = try loadQuantizedWeight(
            inProjBName,
            manifest: manifest,
            weightStore: weightStore,
            groupSize: quantization.groupSize,
            bits: quantization.bits
        )
        let inProjA = try loadQuantizedWeight(
            inProjAName,
            manifest: manifest,
            weightStore: weightStore,
            groupSize: quantization.groupSize,
            bits: quantization.bits
        )
        let outProj = try loadQuantizedWeight(
            outProjName,
            manifest: manifest,
            weightStore: weightStore,
            groupSize: quantization.groupSize,
            bits: quantization.bits
        )
        let conv = try QwenGDNWeights.loadConv1DWeight(
            name: convName,
            weightStore: weightStore,
            architecture: architecture,
            runtime: runtime
        )
        let aLog = try weightStore.loadFloat32Tensor(named: aLogName, runtime: runtime)
        let dtBias = try weightStore.loadFloat32Tensor(named: dtBiasName, runtime: runtime)
        let norm = try weightStore.loadFloat32Tensor(named: normName, runtime: runtime)

        try validate(inProjQKV, name: inProjQKVName, expectedShape: [
            architecture.linearQKVHiddenSize,
            architecture.hiddenSize,
        ])
        try validate(inProjZ, name: inProjZName, expectedShape: [
            architecture.linearValueHiddenSize,
            architecture.hiddenSize,
        ])
        try validate(inProjB, name: inProjBName, expectedShape: [
            architecture.linearValueHeadCount,
            architecture.hiddenSize,
        ])
        try validate(inProjA, name: inProjAName, expectedShape: [
            architecture.linearValueHeadCount,
            architecture.hiddenSize,
        ])
        try validate(outProj, name: outProjName, expectedShape: [
            architecture.hiddenSize,
            architecture.linearValueHiddenSize,
        ])

        return QwenQuantizedGDNWeights(
            layerIndex: layerIndex,
            inProjQKV: inProjQKV,
            inProjZ: inProjZ,
            inProjB: inProjB,
            inProjA: inProjA,
            conv1D: conv.tensor,
            convWeightLayout: conv.layout,
            aLog: aLog,
            dtBias: dtBias,
            norm: norm,
            outProj: outProj,
            linearKeyHeadCount: architecture.linearKeyHeadCount,
            linearValueHeadCount: architecture.linearValueHeadCount,
            linearKeyHeadDimension: architecture.linearKeyHeadDimension,
            linearValueHeadDimension: architecture.linearValueHeadDimension,
            linearKeyHiddenSize: architecture.linearKeyHiddenSize,
            linearValueHiddenSize: architecture.linearValueHiddenSize,
            rmsNormEpsilon: architecture.rmsNormEpsilon
        )
    }

    public func project(
        hiddenStates: EdgeTensor,
        executor: MetalKernelExecutor,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> QwenGDNProjectionOutputs {
        diagnosticSink?("gdn_\(layerIndex)_project_qkv_begin")
        let qkv = try executor.affineQuantizedMatmul(
            hiddenStates,
            weights: inProjQKV,
            transpose: true,
            diagnosticSink: diagnosticSink,
            diagnosticName: "gdn_\(layerIndex)_project_qkv_affine"
        )
        diagnosticSink?("gdn_\(layerIndex)_project_qkv_done shape=\(qkv.shape.dimensions)")
        diagnosticSink?("gdn_\(layerIndex)_project_split_qkv_begin")
        let queryAndRest = try executor.splitColumns(qkv, firstColumnCount: linearKeyHiddenSize)
        let keyAndValue = try executor.splitColumns(
            queryAndRest.second,
            firstColumnCount: linearKeyHiddenSize
        )
        diagnosticSink?("gdn_\(layerIndex)_project_split_qkv_done")
        diagnosticSink?("gdn_\(layerIndex)_project_z_begin")
        let z = try executor.affineQuantizedMatmul(
            hiddenStates,
            weights: inProjZ,
            transpose: true,
            diagnosticSink: diagnosticSink,
            diagnosticName: "gdn_\(layerIndex)_project_z_affine"
        )
        diagnosticSink?("gdn_\(layerIndex)_project_z_done shape=\(z.shape.dimensions)")
        diagnosticSink?("gdn_\(layerIndex)_project_a_begin")
        let a = try executor.affineQuantizedMatmul(
            hiddenStates,
            weights: inProjA,
            transpose: true,
            diagnosticSink: diagnosticSink,
            diagnosticName: "gdn_\(layerIndex)_project_a_affine"
        )
        diagnosticSink?("gdn_\(layerIndex)_project_a_done shape=\(a.shape.dimensions)")
        diagnosticSink?("gdn_\(layerIndex)_project_b_begin")
        let b = try executor.affineQuantizedMatmul(
            hiddenStates,
            weights: inProjB,
            transpose: true,
            diagnosticSink: diagnosticSink,
            diagnosticName: "gdn_\(layerIndex)_project_b_affine"
        )
        diagnosticSink?("gdn_\(layerIndex)_project_b_done shape=\(b.shape.dimensions)")
        return QwenGDNProjectionOutputs(
            query: queryAndRest.first,
            key: keyAndValue.first,
            value: keyAndValue.second,
            z: z,
            a: a,
            b: b,
            mixedQKV: qkv
        )
    }

    public func projectForFusedDecode(
        hiddenStates: EdgeTensor,
        executor: MetalKernelExecutor,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> QwenGDNFusedProjectionOutputs {
        diagnosticSink?("gdn_\(layerIndex)_project_qkv_begin")
        let qkv = try executor.affineQuantizedMatmul(
            hiddenStates,
            weights: inProjQKV,
            transpose: true,
            diagnosticSink: diagnosticSink,
            diagnosticName: "gdn_\(layerIndex)_project_qkv_affine"
        )
        diagnosticSink?("gdn_\(layerIndex)_project_qkv_done shape=\(qkv.shape.dimensions)")
        diagnosticSink?("gdn_\(layerIndex)_project_z_begin")
        let z = try executor.affineQuantizedMatmul(
            hiddenStates,
            weights: inProjZ,
            transpose: true,
            diagnosticSink: diagnosticSink,
            diagnosticName: "gdn_\(layerIndex)_project_z_affine"
        )
        diagnosticSink?("gdn_\(layerIndex)_project_z_done shape=\(z.shape.dimensions)")
        diagnosticSink?("gdn_\(layerIndex)_project_a_begin")
        let a = try executor.affineQuantizedMatmul(
            hiddenStates,
            weights: inProjA,
            transpose: true,
            diagnosticSink: diagnosticSink,
            diagnosticName: "gdn_\(layerIndex)_project_a_affine"
        )
        diagnosticSink?("gdn_\(layerIndex)_project_a_done shape=\(a.shape.dimensions)")
        diagnosticSink?("gdn_\(layerIndex)_project_b_begin")
        let b = try executor.affineQuantizedMatmul(
            hiddenStates,
            weights: inProjB,
            transpose: true,
            diagnosticSink: diagnosticSink,
            diagnosticName: "gdn_\(layerIndex)_project_b_affine"
        )
        diagnosticSink?("gdn_\(layerIndex)_project_b_done shape=\(b.shape.dimensions)")
        return QwenGDNFusedProjectionOutputs(
            mixedQKV: qkv,
            z: z,
            a: a,
            b: b
        )
    }

    public func convolve(
        projections: QwenGDNProjectionOutputs,
        convState: EdgeTensor,
        executor: MetalKernelExecutor,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> QwenGDNConvolutionOutputs {
        guard let mixedQKV = projections.mixedQKV else {
            throw QwenGDNForwardError.missingMixedQKV
        }
        let tokenCount = mixedQKV.shape.dimensions[0]
        let expectedShape = [
            tokenCount,
            linearKeyHiddenSize * 2 + linearValueHiddenSize,
        ]
        guard mixedQKV.shape.dimensions == expectedShape else {
            throw QwenGDNForwardError.invalidMixedQKVShape(
                expected: expectedShape,
                actual: mixedQKV.shape.dimensions
            )
        }

        diagnosticSink?("gdn_\(layerIndex)_conv1d_begin")
        let conv = try executor.gdnDepthwiseConv1D(
            input: mixedQKV,
            weights: conv1D,
            convState: convState,
            diagnosticSink: diagnosticSink,
            diagnosticName: "gdn_\(layerIndex)_conv1d"
        )
        diagnosticSink?("gdn_\(layerIndex)_conv1d_done activatedShape=\(conv.activated.shape.dimensions)")
        diagnosticSink?("gdn_\(layerIndex)_conv_split_begin")
        let queryAndRest = try executor.splitColumns(
            conv.activated,
            firstColumnCount: linearKeyHiddenSize
        )
        let keyAndValue = try executor.splitColumns(
            queryAndRest.second,
            firstColumnCount: linearKeyHiddenSize
        )
        diagnosticSink?("gdn_\(layerIndex)_conv_split_done")
        return QwenGDNConvolutionOutputs(
            query: queryAndRest.first,
            key: keyAndValue.first,
            value: keyAndValue.second,
            nextConvState: conv.nextConvState
        )
    }

    public func normalize(
        convolutionOutputs: QwenGDNConvolutionOutputs,
        executor: MetalKernelExecutor,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> QwenGDNNormalizedConvolutionOutputs {
        diagnosticSink?("gdn_\(layerIndex)_normalize_qk_begin")
        let normalized = try executor.gdnNormalizeQK(
            query: convolutionOutputs.query,
            key: convolutionOutputs.key,
            headCount: linearKeyHeadCount,
            headDimension: linearKeyHeadDimension,
            diagnosticSink: diagnosticSink,
            diagnosticName: "gdn_\(layerIndex)_normalize_qk"
        )
        diagnosticSink?("gdn_\(layerIndex)_normalize_qk_done")
        return QwenGDNNormalizedConvolutionOutputs(
            query: normalized.query,
            key: normalized.key,
            value: convolutionOutputs.value,
            nextConvState: convolutionOutputs.nextConvState
        )
    }

    public func recurrentUpdate(
        convolutionOutputs: QwenGDNConvolutionOutputs,
        projections: QwenGDNProjectionOutputs,
        recurrentState: EdgeTensor,
        executor: MetalKernelExecutor,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> QwenGDNRecurrentOutputs {
        let normalizedOutputs = try normalize(
            convolutionOutputs: convolutionOutputs,
            executor: executor,
            diagnosticSink: diagnosticSink
        )
        return try recurrentUpdate(
            normalizedConvolutionOutputs: normalizedOutputs,
            projections: projections,
            recurrentState: recurrentState,
            executor: executor,
            diagnosticSink: diagnosticSink
        )
    }

    public func recurrentUpdate(
        normalizedConvolutionOutputs: QwenGDNNormalizedConvolutionOutputs,
        projections: QwenGDNProjectionOutputs,
        recurrentState: EdgeTensor,
        executor: MetalKernelExecutor,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> QwenGDNRecurrentOutputs {
        diagnosticSink?("gdn_\(layerIndex)_recurrent_update_begin")
        let update = try executor.gdnRecurrentUpdate(
            query: normalizedConvolutionOutputs.query,
            key: normalizedConvolutionOutputs.key,
            value: normalizedConvolutionOutputs.value,
            a: projections.a,
            b: projections.b,
            aLog: aLog,
            dtBias: dtBias,
            state: recurrentState,
            keyHeadCount: linearKeyHeadCount,
            valueHeadCount: linearValueHeadCount,
            keyHeadDimension: linearKeyHeadDimension,
            valueHeadDimension: linearValueHeadDimension,
            diagnosticSink: diagnosticSink,
            diagnosticName: "gdn_\(layerIndex)_recurrent_update"
        )
        diagnosticSink?("gdn_\(layerIndex)_recurrent_update_done outputShape=\(update.output.shape.dimensions)")
        return QwenGDNRecurrentOutputs(
            output: update.output,
            nextRecurrentState: update.nextState
        )
    }

    public func output(
        recurrentOutputs: QwenGDNRecurrentOutputs,
        projections: QwenGDNProjectionOutputs,
        executor: MetalKernelExecutor,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> EdgeTensor {
        try output(
            recurrentOutputs: recurrentOutputs,
            gate: projections.z,
            executor: executor,
            diagnosticSink: diagnosticSink
        )
    }

    public func output(
        recurrentOutputs: QwenGDNRecurrentOutputs,
        gate: EdgeTensor,
        executor: MetalKernelExecutor,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> EdgeTensor {
        diagnosticSink?("gdn_\(layerIndex)_output_norm_begin")
        let normalized = try executor.rmsNormByHead(
            recurrentOutputs.output,
            weight: norm,
            headCount: linearValueHeadCount,
            headDimension: linearValueHeadDimension,
            epsilon: rmsNormEpsilon
        )
        diagnosticSink?("gdn_\(layerIndex)_output_norm_done shape=\(normalized.shape.dimensions)")
        diagnosticSink?("gdn_\(layerIndex)_output_gate_begin")
        let gated = try executor.siluMultiply(
            gate: gate,
            up: normalized,
            diagnosticSink: diagnosticSink,
            diagnosticName: "gdn_\(layerIndex)_output_gate"
        )
        diagnosticSink?("gdn_\(layerIndex)_output_gate_done shape=\(gated.shape.dimensions)")
        diagnosticSink?("gdn_\(layerIndex)_output_proj_begin")
        let output = try executor.affineQuantizedMatmul(
            gated,
            weights: outProj,
            transpose: true,
            diagnosticSink: diagnosticSink,
            diagnosticName: "gdn_\(layerIndex)_output_proj_affine"
        )
        diagnosticSink?("gdn_\(layerIndex)_output_proj_done shape=\(output.shape.dimensions)")
        return output
    }

    public func callAsFunction(
        hiddenStates: EdgeTensor,
        convState: EdgeTensor,
        recurrentState: EdgeTensor,
        executor: MetalKernelExecutor,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> QwenGDNForwardOutputs {
        diagnosticSink?("gdn_\(layerIndex)_project_begin")
        if executor.runtimeConfiguration.useFusedGDNDecode,
           hiddenStates.shape.dimensions.first == 1 {
            let projections = try projectForFusedDecode(
                hiddenStates: hiddenStates,
                executor: executor,
                diagnosticSink: diagnosticSink
            )
            diagnosticSink?("gdn_\(layerIndex)_project_done")
            diagnosticSink?("gdn_\(layerIndex)_fused_decode_begin")
            let fused = try executor.gdnSingleTokenFusedUpdate(
                mixedQKV: projections.mixedQKV,
                weights: conv1D,
                convState: convState,
                a: projections.a,
                b: projections.b,
                aLog: aLog,
                dtBias: dtBias,
                recurrentState: recurrentState,
                keyHeadCount: linearKeyHeadCount,
                valueHeadCount: linearValueHeadCount,
                keyHeadDimension: linearKeyHeadDimension,
                valueHeadDimension: linearValueHeadDimension,
                epsilon: rmsNormEpsilon,
                diagnosticSink: diagnosticSink,
                diagnosticName: "gdn_\(layerIndex)_fused_decode"
            )
            diagnosticSink?("gdn_\(layerIndex)_fused_decode_done outputShape=\(fused.output.shape.dimensions)")
            diagnosticSink?("gdn_\(layerIndex)_output_begin")
            let hiddenOutput = try output(
                recurrentOutputs: QwenGDNRecurrentOutputs(
                    output: fused.output,
                    nextRecurrentState: fused.nextState
                ),
                gate: projections.z,
                executor: executor,
                diagnosticSink: diagnosticSink
            )
            diagnosticSink?("gdn_\(layerIndex)_output_done shape=\(hiddenOutput.shape.dimensions)")
            return QwenGDNForwardOutputs(
                hiddenStates: hiddenOutput,
                nextConvState: fused.nextConvState,
                nextRecurrentState: fused.nextState
            )
        }

        let projections = try project(
            hiddenStates: hiddenStates,
            executor: executor,
            diagnosticSink: diagnosticSink
        )
        diagnosticSink?("gdn_\(layerIndex)_project_done")
        diagnosticSink?("gdn_\(layerIndex)_convolve_begin")
        let convolved = try convolve(
            projections: projections,
            convState: convState,
            executor: executor,
            diagnosticSink: diagnosticSink
        )
        diagnosticSink?("gdn_\(layerIndex)_convolve_done")
        diagnosticSink?("gdn_\(layerIndex)_recurrent_begin")
        let recurrent = try recurrentUpdate(
            convolutionOutputs: convolved,
            projections: projections,
            recurrentState: recurrentState,
            executor: executor,
            diagnosticSink: diagnosticSink
        )
        diagnosticSink?("gdn_\(layerIndex)_recurrent_done")
        diagnosticSink?("gdn_\(layerIndex)_output_begin")
        let hiddenOutput = try output(
            recurrentOutputs: recurrent,
            projections: projections,
            executor: executor,
            diagnosticSink: diagnosticSink
        )
        diagnosticSink?("gdn_\(layerIndex)_output_done shape=\(hiddenOutput.shape.dimensions)")
        return QwenGDNForwardOutputs(
            hiddenStates: hiddenOutput,
            nextConvState: convolved.nextConvState,
            nextRecurrentState: recurrent.nextRecurrentState
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
