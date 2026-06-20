// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Testing
@testable import EdgeEngine

@Test func qwenGDNWeightsLoadHuggingFaceLayoutFromBundleShard() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-gdn-weights-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let architecture = try makeGDNWeightsArchitecture()
    let index = try QwenModelBundleIndex(
        rootURL: root,
        architecture: architecture,
        weightMap: gdnWeightsWeightMap()
    )
    try writeGDNSafeTensorsShard(
        entries: gdnWeightEntries(convShape: [10, 1, 4]),
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )

    let runtime = try EdgeMetalRuntime()
    let weights = try QwenGDNWeights.loadHuggingFaceLayout(
        layerIndex: 0,
        weightStore: QwenModelWeightStore(bundleIndex: index),
        runtime: runtime
    )

    #expect(weights.layerIndex == 0)
    #expect(weights.inProjQKV.shape == EdgeTensorShape([2, 10]))
    #expect(weights.inProjZ.shape == EdgeTensorShape([2, 4]))
    #expect(weights.inProjB.shape == EdgeTensorShape([2, 2]))
    #expect(weights.inProjA.shape == EdgeTensorShape([2, 2]))
    #expect(weights.conv1D.shape == EdgeTensorShape([10, 4, 1]))
    #expect(weights.convWeightLayout == .huggingFaceDepthwise)
    #expect(weights.aLog.shape == EdgeTensorShape([2]))
    #expect(weights.dtBias.shape == EdgeTensorShape([2]))
    #expect(weights.norm.shape == EdgeTensorShape([2]))
    #expect(weights.outProj.shape == EdgeTensorShape([4, 2]))
    #expect(weights.rmsNormEpsilon == architecture.rmsNormEpsilon)
    #expect(Array(try weights.inProjQKV.readFloat32().prefix(4)) == [0, 2, 4, 6])
    #expect(try weights.aLog.readFloat32() == [0.25, 0.5])
    #expect(try weights.dtBias.readFloat32() == [1, 2])
}

@Test func qwenGDNWeightsAcceptsSanitizedConvLayout() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-gdn-sanitized-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let index = try QwenModelBundleIndex(
        rootURL: root,
        architecture: try makeGDNWeightsArchitecture(),
        weightMap: gdnWeightsWeightMap()
    )
    try writeGDNSafeTensorsShard(
        entries: gdnWeightEntries(convShape: [10, 4, 1]),
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )

    let weights = try QwenGDNWeights.loadHuggingFaceLayout(
        layerIndex: 0,
        weightStore: QwenModelWeightStore(bundleIndex: index),
        runtime: try EdgeMetalRuntime()
    )

    #expect(weights.conv1D.shape == EdgeTensorShape([10, 4, 1]))
    #expect(weights.convWeightLayout == .mlxSanitizedDepthwise)
}

@Test func qwenGDNWeightsProjectHiddenStatesIntoGDNInputs() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-gdn-project-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let index = try QwenModelBundleIndex(
        rootURL: root,
        architecture: try makeGDNWeightsArchitecture(),
        weightMap: gdnWeightsWeightMap()
    )
    try writeGDNSafeTensorsShard(
        entries: gdnWeightEntries(convShape: [10, 4, 1]),
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )

    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let hiddenStates = try EdgeTensor(
        float32: [1, 2],
        shape: EdgeTensorShape([1, 2]),
        runtime: runtime
    )
    let weights = try QwenGDNWeights.loadHuggingFaceLayout(
        layerIndex: 0,
        weightStore: QwenModelWeightStore(bundleIndex: index),
        runtime: runtime
    )

    let outputs = try weights.project(hiddenStates: hiddenStates, executor: executor)

    #expect(outputs.query.shape == EdgeTensorShape([1, 3]))
    #expect(outputs.key.shape == EdgeTensorShape([1, 3]))
    #expect(outputs.value.shape == EdgeTensorShape([1, 4]))
    #expect(outputs.z.shape == EdgeTensorShape([1, 4]))
    #expect(outputs.a.shape == EdgeTensorShape([1, 2]))
    #expect(outputs.b.shape == EdgeTensorShape([1, 2]))
    #expect(outputs.mixedQKV?.shape == EdgeTensorShape([1, 10]))
    #expect(try outputs.mixedQKV?.readFloat32() == [2, 8, 14, 20, 26, 32, 38, 44, 50, 56])
    #expect(try outputs.query.readFloat32() == [2, 8, 14])
    #expect(try outputs.key.readFloat32() == [20, 26, 32])
    #expect(try outputs.value.readFloat32() == [38, 44, 50, 56])
    #expect(try outputs.z.readFloat32() == [2, 8, 14, 20])
    #expect(try outputs.a.readFloat32() == [2, 8])
    #expect(try outputs.b.readFloat32() == [2, 8])
}

@Test func qwenGDNWeightsConvolveProjectedQKVIntoPostConvInputs() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-gdn-convolve-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let index = try QwenModelBundleIndex(
        rootURL: root,
        architecture: try makeGDNWeightsArchitecture(),
        weightMap: gdnWeightsWeightMap()
    )
    try writeGDNSafeTensorsShard(
        entries: gdnWeightEntries(convShape: [10, 4, 1]),
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )

    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let hiddenStates = try EdgeTensor(
        float32: [1, 2],
        shape: EdgeTensorShape([1, 2]),
        runtime: runtime
    )
    let convStateValues = Array(repeating: Float.zero, count: 30)
    let convState = try EdgeTensor(
        float32: convStateValues,
        shape: EdgeTensorShape([3, 10]),
        runtime: runtime
    )
    let weights = try QwenGDNWeights.loadHuggingFaceLayout(
        layerIndex: 0,
        weightStore: QwenModelWeightStore(bundleIndex: index),
        runtime: runtime
    )
    let projections = try weights.project(hiddenStates: hiddenStates, executor: executor)
    guard let mixedQKV = projections.mixedQKV else {
        Issue.record("GDN projections should retain stacked qkv for conv.")
        return
    }

    let convolved = try weights.convolve(
        projections: projections,
        convState: convState,
        executor: executor
    )
    let expected = try CPUReferenceOps.gdnDepthwiseConv1D(
        try mixedQKV.readFloat32(),
        convState: convStateValues,
        weights: try weights.conv1D.readFloat32(),
        tokenCount: 1,
        channelCount: 10,
        kernelSize: 4
    )
    let activated = expected.activated
    let queryError = try NumericComparison.maxAbsoluteError(
        try convolved.query.readFloat32(),
        Array(activated[0..<3])
    )
    let keyError = try NumericComparison.maxAbsoluteError(
        try convolved.key.readFloat32(),
        Array(activated[3..<6])
    )
    let valueError = try NumericComparison.maxAbsoluteError(
        try convolved.value.readFloat32(),
        Array(activated[6..<10])
    )

    #expect(convolved.query.shape == EdgeTensorShape([1, 3]))
    #expect(convolved.key.shape == EdgeTensorShape([1, 3]))
    #expect(convolved.value.shape == EdgeTensorShape([1, 4]))
    #expect(convolved.nextConvState.shape == EdgeTensorShape([3, 10]))
    #expect(queryError < 1e-4)
    #expect(keyError < 1e-4)
    #expect(valueError < 1e-4)
    #expect(try convolved.nextConvState.readFloat32() == expected.nextConvState)
}

@Test func qwenGDNWeightsNormalizesPostConvQKForRecurrentUpdate() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-gdn-normalize-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let index = try QwenModelBundleIndex(
        rootURL: root,
        architecture: try makeGDNWeightsArchitecture(),
        weightMap: gdnWeightsWeightMap()
    )
    try writeGDNSafeTensorsShard(
        entries: gdnWeightEntries(convShape: [10, 4, 1]),
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )

    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let hiddenStates = try EdgeTensor(
        float32: [1, 2],
        shape: EdgeTensorShape([1, 2]),
        runtime: runtime
    )
    let convState = try EdgeTensor(
        float32: Array(repeating: Float.zero, count: 30),
        shape: EdgeTensorShape([3, 10]),
        runtime: runtime
    )
    let weights = try QwenGDNWeights.loadHuggingFaceLayout(
        layerIndex: 0,
        weightStore: QwenModelWeightStore(bundleIndex: index),
        runtime: runtime
    )
    let projections = try weights.project(hiddenStates: hiddenStates, executor: executor)
    let convolved = try weights.convolve(
        projections: projections,
        convState: convState,
        executor: executor
    )

    let normalized = try weights.normalize(convolutionOutputs: convolved, executor: executor)
    let expected = try CPUReferenceOps.gdnNormalizeQK(
        query: try convolved.query.readFloat32(),
        key: try convolved.key.readFloat32(),
        tokenCount: 1,
        headCount: 1,
        headDimension: 3
    )
    let queryError = try NumericComparison.maxAbsoluteError(
        try normalized.query.readFloat32(),
        expected.query
    )
    let keyError = try NumericComparison.maxAbsoluteError(
        try normalized.key.readFloat32(),
        expected.key
    )
    let normalizedValue = try normalized.value.readFloat32()
    let convolvedValue = try convolved.value.readFloat32()

    #expect(normalized.query.shape == EdgeTensorShape([1, 3]))
    #expect(normalized.key.shape == EdgeTensorShape([1, 3]))
    #expect(normalized.value.shape == EdgeTensorShape([1, 4]))
    #expect(normalized.nextConvState.shape == EdgeTensorShape([3, 10]))
    #expect(queryError < 1e-5)
    #expect(keyError < 1e-5)
    #expect(normalizedValue == convolvedValue)
}

@Test func qwenGDNWeightsRunsRecurrentUpdateFromPostConvInputs() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-gdn-recurrent-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let index = try QwenModelBundleIndex(
        rootURL: root,
        architecture: try makeGDNWeightsArchitecture(),
        weightMap: gdnWeightsWeightMap()
    )
    try writeGDNSafeTensorsShard(
        entries: gdnWeightEntries(convShape: [10, 4, 1]),
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )

    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let hiddenStates = try EdgeTensor(
        float32: [1, 2],
        shape: EdgeTensorShape([1, 2]),
        runtime: runtime
    )
    let convState = try EdgeTensor(
        float32: Array(repeating: Float.zero, count: 30),
        shape: EdgeTensorShape([3, 10]),
        runtime: runtime
    )
    let recurrentStateValues = Array(repeating: Float.zero, count: 12)
    let recurrentState = try EdgeTensor(
        float32: recurrentStateValues,
        shape: EdgeTensorShape([2, 2, 3]),
        runtime: runtime
    )
    let weights = try QwenGDNWeights.loadHuggingFaceLayout(
        layerIndex: 0,
        weightStore: QwenModelWeightStore(bundleIndex: index),
        runtime: runtime
    )
    let projections = try weights.project(hiddenStates: hiddenStates, executor: executor)
    let convolved = try weights.convolve(
        projections: projections,
        convState: convState,
        executor: executor
    )

    let recurrent = try weights.recurrentUpdate(
        convolutionOutputs: convolved,
        projections: projections,
        recurrentState: recurrentState,
        executor: executor
    )
    let normalized = try CPUReferenceOps.gdnNormalizeQK(
        query: try convolved.query.readFloat32(),
        key: try convolved.key.readFloat32(),
        tokenCount: 1,
        headCount: 1,
        headDimension: 3
    )
    let expected = try QwenGDNReference.gatedDeltaUpdate(
        query: normalized.query,
        key: normalized.key,
        value: try convolved.value.readFloat32(),
        a: try projections.a.readFloat32(),
        b: try projections.b.readFloat32(),
        aLog: try weights.aLog.readFloat32(),
        dtBias: try weights.dtBias.readFloat32(),
        initialState: recurrentStateValues,
        batchSize: 1,
        tokenCount: 1,
        keyHeadCount: 1,
        valueHeadCount: 2,
        keyHeadDimension: 3,
        valueHeadDimension: 2
    )
    let outputError = try NumericComparison.maxAbsoluteError(
        try recurrent.output.readFloat32(),
        expected.output
    )
    let stateError = try NumericComparison.maxAbsoluteError(
        try recurrent.nextRecurrentState.readFloat32(),
        expected.updatedState
    )
    let outputScale = expected.output.map { abs($0) }.max() ?? 1
    let stateScale = expected.updatedState.map { abs($0) }.max() ?? 1

    #expect(recurrent.output.shape == EdgeTensorShape([1, 4]))
    #expect(recurrent.nextRecurrentState.shape == EdgeTensorShape([2, 2, 3]))
    #expect(outputError < max(1e-3, outputScale * 1e-5))
    #expect(stateError < max(1e-3, stateScale * 1e-5))
}

@Test func qwenGDNWeightsProjectsGatedRecurrentOutputToHiddenStates() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-gdn-output-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let architecture = try makeGDNWeightsArchitecture()
    let index = try QwenModelBundleIndex(
        rootURL: root,
        architecture: architecture,
        weightMap: gdnWeightsWeightMap()
    )
    try writeGDNSafeTensorsShard(
        entries: gdnWeightEntries(convShape: [10, 4, 1]),
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )

    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let hiddenStates = try EdgeTensor(
        float32: [1, 2],
        shape: EdgeTensorShape([1, 2]),
        runtime: runtime
    )
    let convState = try EdgeTensor(
        float32: Array(repeating: Float.zero, count: 30),
        shape: EdgeTensorShape([3, 10]),
        runtime: runtime
    )
    let recurrentState = try EdgeTensor(
        float32: Array(repeating: Float.zero, count: 12),
        shape: EdgeTensorShape([2, 2, 3]),
        runtime: runtime
    )
    let weights = try QwenGDNWeights.loadHuggingFaceLayout(
        layerIndex: 0,
        weightStore: QwenModelWeightStore(bundleIndex: index),
        runtime: runtime
    )
    let projections = try weights.project(hiddenStates: hiddenStates, executor: executor)
    let convolved = try weights.convolve(
        projections: projections,
        convState: convState,
        executor: executor
    )
    let recurrent = try weights.recurrentUpdate(
        convolutionOutputs: convolved,
        projections: projections,
        recurrentState: recurrentState,
        executor: executor
    )

    let output = try weights.output(
        recurrentOutputs: recurrent,
        projections: projections,
        executor: executor
    )
    let recurrentValues = try recurrent.output.readFloat32()
    let normalized = try CPUReferenceOps.rmsNormByHead(
        recurrentValues,
        weight: try weights.norm.readFloat32(),
        tokenCount: 1,
        headCount: 2,
        headDimension: 2,
        epsilon: architecture.rmsNormEpsilon
    )
    let gated = try CPUReferenceOps.swiglu(
        gate: try projections.z.readFloat32(),
        up: normalized
    )
    let expected = try CPUReferenceOps.matmul(
        gated,
        rows: 1,
        inner: 4,
        try weights.outProj.readFloat32(),
        columns: 2
    )
    let outputError = try NumericComparison.maxAbsoluteError(
        try output.readFloat32(),
        expected
    )
    let outputScale = expected.map { abs($0) }.max() ?? 1

    #expect(output.shape == EdgeTensorShape([1, 2]))
    #expect(outputError < max(1e-3, outputScale * 1e-5))
}

@Test func qwenGDNWeightsRunsFullLinearAttentionForward() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-gdn-forward-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let index = try QwenModelBundleIndex(
        rootURL: root,
        architecture: try makeGDNWeightsArchitecture(),
        weightMap: gdnWeightsWeightMap()
    )
    try writeGDNSafeTensorsShard(
        entries: gdnWeightEntries(convShape: [10, 4, 1]),
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )

    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let hiddenStates = try EdgeTensor(
        float32: [1, 2],
        shape: EdgeTensorShape([1, 2]),
        runtime: runtime
    )
    let convState = try EdgeTensor(
        float32: Array(repeating: Float.zero, count: 30),
        shape: EdgeTensorShape([3, 10]),
        runtime: runtime
    )
    let recurrentState = try EdgeTensor(
        float32: Array(repeating: Float.zero, count: 12),
        shape: EdgeTensorShape([2, 2, 3]),
        runtime: runtime
    )
    let weights = try QwenGDNWeights.loadHuggingFaceLayout(
        layerIndex: 0,
        weightStore: QwenModelWeightStore(bundleIndex: index),
        runtime: runtime
    )

    let projections = try weights.project(hiddenStates: hiddenStates, executor: executor)
    let convolved = try weights.convolve(
        projections: projections,
        convState: convState,
        executor: executor
    )
    let recurrent = try weights.recurrentUpdate(
        convolutionOutputs: convolved,
        projections: projections,
        recurrentState: recurrentState,
        executor: executor
    )
    let manualOutput = try weights.output(
        recurrentOutputs: recurrent,
        projections: projections,
        executor: executor
    )
    let forward = try weights(
        hiddenStates: hiddenStates,
        convState: convState,
        recurrentState: recurrentState,
        executor: executor
    )
    let outputError = try NumericComparison.maxAbsoluteError(
        try forward.hiddenStates.readFloat32(),
        try manualOutput.readFloat32()
    )
    let convStateError = try NumericComparison.maxAbsoluteError(
        try forward.nextConvState.readFloat32(),
        try convolved.nextConvState.readFloat32()
    )
    let recurrentStateError = try NumericComparison.maxAbsoluteError(
        try forward.nextRecurrentState.readFloat32(),
        try recurrent.nextRecurrentState.readFloat32()
    )

    #expect(forward.hiddenStates.shape == EdgeTensorShape([1, 2]))
    #expect(forward.nextConvState.shape == EdgeTensorShape([3, 10]))
    #expect(forward.nextRecurrentState.shape == EdgeTensorShape([2, 2, 3]))
    #expect(outputError < 1e-5)
    #expect(convStateError < 1e-5)
    #expect(recurrentStateError < 1e-5)
}

@Test func qwenQuantizedGDNWeightsMatchesFloatForward() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeGDNWeightsArchitecture()
    let floatWeights = try makeExactGDNWeights(runtime: runtime, architecture: architecture)
    let quantizedWeights = try makeExactQuantizedGDNWeights(runtime: runtime, architecture: architecture)
    let hiddenStates = try EdgeTensor(
        float32: [
            1, 2,
            2, 1,
        ],
        shape: EdgeTensorShape([2, 2]),
        runtime: runtime
    )
    let convState = try EdgeTensor(
        float32: Array(repeating: Float.zero, count: 30),
        shape: EdgeTensorShape([3, 10]),
        runtime: runtime
    )
    let recurrentState = try EdgeTensor(
        float32: Array(repeating: Float.zero, count: 12),
        shape: EdgeTensorShape([2, 2, 3]),
        runtime: runtime
    )
    let expected = try floatWeights(
        hiddenStates: hiddenStates,
        convState: convState,
        recurrentState: recurrentState,
        executor: executor
    )
    let actual = try quantizedWeights(
        hiddenStates: hiddenStates,
        convState: convState,
        recurrentState: recurrentState,
        executor: executor
    )
    let hiddenValues = try expected.hiddenStates.readFloat32()
    let convValues = try expected.nextConvState.readFloat32()
    let recurrentValues = try expected.nextRecurrentState.readFloat32()
    let hiddenError = try NumericComparison.maxAbsoluteError(
        try actual.hiddenStates.readFloat32(),
        hiddenValues
    )
    let convStateError = try NumericComparison.maxAbsoluteError(
        try actual.nextConvState.readFloat32(),
        convValues
    )
    let recurrentStateError = try NumericComparison.maxAbsoluteError(
        try actual.nextRecurrentState.readFloat32(),
        recurrentValues
    )

    #expect(actual.hiddenStates.shape == EdgeTensorShape([2, 2]))
    #expect(actual.nextConvState.shape == EdgeTensorShape([3, 10]))
    #expect(actual.nextRecurrentState.shape == EdgeTensorShape([2, 2, 3]))
    #expect(hiddenError < max(1e-3, (hiddenValues.map { abs($0) }.max() ?? 1) * 1e-5))
    #expect(convStateError < max(1e-3, (convValues.map { abs($0) }.max() ?? 1) * 1e-5))
    #expect(recurrentStateError < max(1e-3, (recurrentValues.map { abs($0) }.max() ?? 1) * 1e-5))
}

@Test func qwenQuantizedGDNWeightsFusedDecodeMatchesDefaultSingleTokenForward() throws {
    let baselineRuntime = try EdgeMetalRuntime()
    let baselineExecutor = try MetalKernelExecutor(runtime: baselineRuntime)
    let fastRuntime = try EdgeMetalRuntime(configuration: MetalRuntimeConfiguration(useFusedGDNDecode: true))
    let fastExecutor = try MetalKernelExecutor(runtime: fastRuntime)
    let architecture = try makeGDNWeightsArchitecture()
    let baselineWeights = try makeExactQuantizedGDNWeights(runtime: baselineRuntime, architecture: architecture)
    let fastWeights = try makeExactQuantizedGDNWeights(runtime: fastRuntime, architecture: architecture)
    let hiddenValues: [Float] = [1, 2]
    let convValues = (0..<30).map { Float($0) * 0.01 }
    let recurrentValues = (0..<12).map { Float($0 - 3) * 0.02 }
    let baselineHiddenStates = try EdgeTensor(
        float32: hiddenValues,
        shape: EdgeTensorShape([1, 2]),
        runtime: baselineRuntime
    )
    let fastHiddenStates = try EdgeTensor(
        float32: hiddenValues,
        shape: EdgeTensorShape([1, 2]),
        runtime: fastRuntime
    )
    let baselineConvState = try EdgeTensor(
        float32: convValues,
        shape: EdgeTensorShape([3, 10]),
        runtime: baselineRuntime
    )
    let fastConvState = try EdgeTensor(
        float32: convValues,
        shape: EdgeTensorShape([3, 10]),
        runtime: fastRuntime
    )
    let baselineRecurrentState = try EdgeTensor(
        float32: recurrentValues,
        shape: EdgeTensorShape([2, 2, 3]),
        runtime: baselineRuntime
    )
    let fastRecurrentState = try EdgeTensor(
        float32: recurrentValues,
        shape: EdgeTensorShape([2, 2, 3]),
        runtime: fastRuntime
    )

    let expected = try baselineWeights(
        hiddenStates: baselineHiddenStates,
        convState: baselineConvState,
        recurrentState: baselineRecurrentState,
        executor: baselineExecutor
    )
    var diagnostics: [String] = []
    let actual = try fastWeights(
        hiddenStates: fastHiddenStates,
        convState: fastConvState,
        recurrentState: fastRecurrentState,
        executor: fastExecutor,
        diagnosticSink: { diagnostics.append($0) }
    )

    let expectedHidden = try expected.hiddenStates.readFloat32()
    let expectedConv = try expected.nextConvState.readFloat32()
    let expectedRecurrent = try expected.nextRecurrentState.readFloat32()
    let hiddenError = try NumericComparison.maxAbsoluteError(
        try actual.hiddenStates.readFloat32(),
        expectedHidden
    )
    let convStateError = try NumericComparison.maxAbsoluteError(
        try actual.nextConvState.readFloat32(),
        expectedConv
    )
    let recurrentStateError = try NumericComparison.maxAbsoluteError(
        try actual.nextRecurrentState.readFloat32(),
        expectedRecurrent
    )

    #expect(diagnostics.contains("gdn_0_fused_decode_begin"))
    #expect(actual.hiddenStates.shape == EdgeTensorShape([1, 2]))
    #expect(actual.nextConvState.shape == EdgeTensorShape([3, 10]))
    #expect(actual.nextRecurrentState.shape == EdgeTensorShape([2, 2, 3]))
    #expect(hiddenError < max(1e-3, (expectedHidden.map { abs($0) }.max() ?? 1) * 1e-5))
    #expect(convStateError < max(1e-3, (expectedConv.map { abs($0) }.max() ?? 1) * 1e-5))
    #expect(recurrentStateError < max(1e-3, (expectedRecurrent.map { abs($0) }.max() ?? 1) * 1e-5))
}

@Test func qwenQuantizedGDNWeightsLoadHuggingFaceLayoutAndRunForward() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-quantized-gdn-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let architecture = try makeQuantizedGDNWeightsArchitecture()
    let index = try QwenModelBundleIndex(
        rootURL: root,
        architecture: architecture,
        weightMap: quantizedGDNWeightsWeightMap()
    )
    try writeQuantizedGDNSafeTensorsShard(
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let hiddenStates = try EdgeTensor(
        float32: [
            1, 2,
            2, 1,
        ],
        shape: EdgeTensorShape([2, 2]),
        runtime: runtime
    )
    let convState = try EdgeTensor(
        float32: Array(repeating: Float.zero, count: 30),
        shape: EdgeTensorShape([3, 10]),
        runtime: runtime
    )
    let recurrentState = try EdgeTensor(
        float32: Array(repeating: Float.zero, count: 12),
        shape: EdgeTensorShape([2, 2, 3]),
        runtime: runtime
    )
    let expectedWeights = try makeExactGDNWeights(runtime: runtime, architecture: architecture)
    let weights = try QwenQuantizedGDNWeights.loadHuggingFaceLayout(
        layerIndex: 0,
        weightStore: QwenModelWeightStore(bundleIndex: index),
        runtime: runtime
    )

    let expected = try expectedWeights(
        hiddenStates: hiddenStates,
        convState: convState,
        recurrentState: recurrentState,
        executor: executor
    )
    let actual = try weights(
        hiddenStates: hiddenStates,
        convState: convState,
        recurrentState: recurrentState,
        executor: executor
    )
    let hiddenValues = try expected.hiddenStates.readFloat32()
    let convValues = try expected.nextConvState.readFloat32()
    let recurrentValues = try expected.nextRecurrentState.readFloat32()
    let hiddenError = try NumericComparison.maxAbsoluteError(
        try actual.hiddenStates.readFloat32(),
        hiddenValues
    )
    let convStateError = try NumericComparison.maxAbsoluteError(
        try actual.nextConvState.readFloat32(),
        convValues
    )
    let recurrentStateError = try NumericComparison.maxAbsoluteError(
        try actual.nextRecurrentState.readFloat32(),
        recurrentValues
    )

    #expect(weights.inProjQKV.shape == [10, 2])
    #expect(weights.inProjZ.shape == [4, 2])
    #expect(weights.inProjA.shape == [2, 2])
    #expect(weights.inProjB.shape == [2, 2])
    #expect(weights.outProj.shape == [2, 4])
    #expect(weights.convWeightLayout == .mlxSanitizedDepthwise)
    #expect(hiddenError < max(1e-3, (hiddenValues.map { abs($0) }.max() ?? 1) * 1e-5))
    #expect(convStateError < max(1e-3, (convValues.map { abs($0) }.max() ?? 1) * 1e-5))
    #expect(recurrentStateError < max(1e-3, (recurrentValues.map { abs($0) }.max() ?? 1) * 1e-5))
}

@Test func qwenQuantizedGDNDecoderLayerMatchesFloatReference() throws {
    let architecture = try makeGDNWeightsArchitecture()
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let hiddenStates = try EdgeTensor(
        float32: [
            1, 2,
            2, 1,
        ],
        shape: EdgeTensorShape([2, 2]),
        runtime: runtime
    )
    let convState = try EdgeTensor(
        float32: Array(repeating: Float.zero, count: 30),
        shape: EdgeTensorShape([3, 10]),
        runtime: runtime
    )
    let recurrentState = try EdgeTensor(
        float32: Array(repeating: Float.zero, count: 12),
        shape: EdgeTensorShape([2, 2, 3]),
        runtime: runtime
    )
    let floatLayer = try makeExactGDNDecoderLayer(runtime: runtime, architecture: architecture)
    let quantizedLayer = try makeExactQuantizedGDNDecoderLayer(runtime: runtime, architecture: architecture)

    let expected = try floatLayer.outputTensor(
        hiddenStates: hiddenStates,
        convState: convState,
        recurrentState: recurrentState,
        executor: executor
    )
    let actual = try quantizedLayer.outputTensor(
        hiddenStates: hiddenStates,
        convState: convState,
        recurrentState: recurrentState,
        executor: executor
    )
    let hiddenValues = try expected.hiddenStates.readFloat32()
    let convValues = try expected.nextConvState.readFloat32()
    let recurrentValues = try expected.nextRecurrentState.readFloat32()
    let hiddenError = try NumericComparison.maxAbsoluteError(
        try actual.hiddenStates.readFloat32(),
        hiddenValues
    )
    let convStateError = try NumericComparison.maxAbsoluteError(
        try actual.nextConvState.readFloat32(),
        convValues
    )
    let recurrentStateError = try NumericComparison.maxAbsoluteError(
        try actual.nextRecurrentState.readFloat32(),
        recurrentValues
    )

    #expect(actual.hiddenStates.shape == EdgeTensorShape([2, 2]))
    #expect(actual.nextConvState.shape == EdgeTensorShape([3, 10]))
    #expect(actual.nextRecurrentState.shape == EdgeTensorShape([2, 2, 3]))
    #expect(hiddenError < max(1e-3, (hiddenValues.map { abs($0) }.max() ?? 1) * 1e-5))
    #expect(convStateError < max(1e-3, (convValues.map { abs($0) }.max() ?? 1) * 1e-5))
    #expect(recurrentStateError < max(1e-3, (recurrentValues.map { abs($0) }.max() ?? 1) * 1e-5))
}

@Test func qwenQuantizedGDNDecoderLayerLoadsHuggingFaceLayoutAndRunsForward() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-quantized-gdn-decoder-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let architecture = try makeQuantizedGDNWeightsArchitecture()
    let index = try QwenModelBundleIndex(
        rootURL: root,
        architecture: architecture,
        weightMap: quantizedGDNDecoderLayerWeightMap()
    )
    try writeQuantizedGDNDecoderLayerSafeTensorsShard(
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let hiddenStates = try EdgeTensor(
        float32: [
            1, 2,
            2, 1,
        ],
        shape: EdgeTensorShape([2, 2]),
        runtime: runtime
    )
    let convState = try EdgeTensor(
        float32: Array(repeating: Float.zero, count: 30),
        shape: EdgeTensorShape([3, 10]),
        runtime: runtime
    )
    let recurrentState = try EdgeTensor(
        float32: Array(repeating: Float.zero, count: 12),
        shape: EdgeTensorShape([2, 2, 3]),
        runtime: runtime
    )
    let expectedLayer = try makeExactGDNDecoderLayer(runtime: runtime, architecture: architecture)
    let layer = try QwenQuantizedGDNDecoderLayerReference.loadHuggingFaceLayout(
        layerIndex: 0,
        weightStore: QwenModelWeightStore(bundleIndex: index),
        runtime: runtime
    )

    let expected = try expectedLayer.outputTensor(
        hiddenStates: hiddenStates,
        convState: convState,
        recurrentState: recurrentState,
        executor: executor
    )
    let actual = try layer.outputTensor(
        hiddenStates: hiddenStates,
        convState: convState,
        recurrentState: recurrentState,
        executor: executor
    )
    let hiddenValues = try expected.hiddenStates.readFloat32()
    let convValues = try expected.nextConvState.readFloat32()
    let recurrentValues = try expected.nextRecurrentState.readFloat32()
    let hiddenError = try NumericComparison.maxAbsoluteError(
        try actual.hiddenStates.readFloat32(),
        hiddenValues
    )
    let convStateError = try NumericComparison.maxAbsoluteError(
        try actual.nextConvState.readFloat32(),
        convValues
    )
    let recurrentStateError = try NumericComparison.maxAbsoluteError(
        try actual.nextRecurrentState.readFloat32(),
        recurrentValues
    )

    #expect(layer.linearAttention.inProjQKV.shape == [10, 2])
    #expect(layer.mlp.gate.shape == [4, 2])
    #expect(layer.mlp.down.shape == [2, 4])
    #expect(layer.inputLayerNorm.shape == EdgeTensorShape([2]))
    #expect(layer.postAttentionLayerNorm.shape == EdgeTensorShape([2]))
    #expect(layer.linearAttention.convWeightLayout == .mlxSanitizedDepthwise)
    #expect(hiddenError < max(1e-3, (hiddenValues.map { abs($0) }.max() ?? 1) * 1e-5))
    #expect(convStateError < max(1e-3, (convValues.map { abs($0) }.max() ?? 1) * 1e-5))
    #expect(recurrentStateError < max(1e-3, (recurrentValues.map { abs($0) }.max() ?? 1) * 1e-5))
}

@Test func qwenQuantizedGDNDecoderLayerCarriesStateLikeFloatReference() throws {
    let architecture = try makeGDNWeightsArchitecture()
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let fullHiddenStates = try EdgeTensor(
        float32: [
            1, 2,
            2, 1,
        ],
        shape: EdgeTensorShape([2, 2]),
        runtime: runtime
    )
    let zeroConvState = try EdgeTensor(
        float32: Array(repeating: Float.zero, count: 30),
        shape: EdgeTensorShape([3, 10]),
        runtime: runtime
    )
    let zeroRecurrentState = try EdgeTensor(
        float32: Array(repeating: Float.zero, count: 12),
        shape: EdgeTensorShape([2, 2, 3]),
        runtime: runtime
    )
    let floatLayer = try makeExactGDNDecoderLayer(runtime: runtime, architecture: architecture)
    let quantizedLayer = try makeExactQuantizedGDNDecoderLayer(runtime: runtime, architecture: architecture)

    let expectedFull = try floatLayer.outputTensor(
        hiddenStates: fullHiddenStates,
        convState: zeroConvState,
        recurrentState: zeroRecurrentState,
        executor: executor
    )
    let firstTokenOutput = try quantizedLayer.outputTensor(
        hiddenStates: EdgeTensor(float32: [1, 2], shape: EdgeTensorShape([1, 2]), runtime: runtime),
        convState: zeroConvState,
        recurrentState: zeroRecurrentState,
        executor: executor
    )
    let secondTokenOutput = try quantizedLayer.outputTensor(
        hiddenStates: EdgeTensor(float32: [2, 1], shape: EdgeTensorShape([1, 2]), runtime: runtime),
        convState: firstTokenOutput.nextConvState,
        recurrentState: firstTokenOutput.nextRecurrentState,
        executor: executor
    )
    let expectedHiddenValues = try expectedFull.hiddenStates.readFloat32()
    let actualHiddenValues = try firstTokenOutput.hiddenStates.readFloat32()
        + secondTokenOutput.hiddenStates.readFloat32()
    let hiddenError = try NumericComparison.maxAbsoluteError(
        actualHiddenValues,
        expectedHiddenValues
    )
    let convStateError = try NumericComparison.maxAbsoluteError(
        try secondTokenOutput.nextConvState.readFloat32(),
        try expectedFull.nextConvState.readFloat32()
    )
    let recurrentStateError = try NumericComparison.maxAbsoluteError(
        try secondTokenOutput.nextRecurrentState.readFloat32(),
        try expectedFull.nextRecurrentState.readFloat32()
    )

    #expect(firstTokenOutput.hiddenStates.shape == EdgeTensorShape([1, 2]))
    #expect(secondTokenOutput.hiddenStates.shape == EdgeTensorShape([1, 2]))
    #expect(secondTokenOutput.nextConvState.shape == EdgeTensorShape([3, 10]))
    #expect(secondTokenOutput.nextRecurrentState.shape == EdgeTensorShape([2, 2, 3]))
    #expect(hiddenError < max(1e-3, (expectedHiddenValues.map { abs($0) }.max() ?? 1) * 1e-5))
    #expect(convStateError < 1e-5)
    #expect(recurrentStateError < 1e-5)
}

@Test func qwenGDNDecoderLayerLoadsHuggingFaceLayoutAndRunsForward() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-gdn-decoder-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let index = try QwenModelBundleIndex(
        rootURL: root,
        architecture: try makeGDNWeightsArchitecture(),
        weightMap: gdnWeightsWeightMap()
    )
    try writeGDNSafeTensorsShard(
        entries: gdnDecoderLayerEntries(convShape: [10, 4, 1]),
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )

    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let hiddenStates = try EdgeTensor(
        float32: [1, 2],
        shape: EdgeTensorShape([1, 2]),
        runtime: runtime
    )
    let convState = try EdgeTensor(
        float32: Array(repeating: Float.zero, count: 30),
        shape: EdgeTensorShape([3, 10]),
        runtime: runtime
    )
    let recurrentState = try EdgeTensor(
        float32: Array(repeating: Float.zero, count: 12),
        shape: EdgeTensorShape([2, 2, 3]),
        runtime: runtime
    )
    let layer = try QwenGDNDecoderLayerReference.loadHuggingFaceLayout(
        layerIndex: 0,
        weightStore: QwenModelWeightStore(bundleIndex: index),
        runtime: runtime
    )

    let output = try layer.outputTensor(
        hiddenStates: hiddenStates,
        convState: convState,
        recurrentState: recurrentState,
        executor: executor
    )
    let attentionInput = try executor.rmsNorm(
        hiddenStates,
        weight: layer.inputLayerNorm,
        epsilon: layer.rmsNormEpsilon
    )
    let attentionOutput = try layer.linearAttention(
        hiddenStates: attentionInput,
        convState: convState,
        recurrentState: recurrentState,
        executor: executor
    )
    let attentionResidual = try executor.add(hiddenStates, attentionOutput.hiddenStates)
    let mlpInput = try executor.rmsNorm(
        attentionResidual,
        weight: layer.postAttentionLayerNorm,
        epsilon: layer.rmsNormEpsilon
    )
    let mlpOutput = try layer.mlp(hiddenStates: mlpInput, executor: executor)
    let expectedHiddenStates = try executor.add(attentionResidual, mlpOutput)
    let hiddenError = try NumericComparison.maxAbsoluteError(
        try output.hiddenStates.readFloat32(),
        try expectedHiddenStates.readFloat32()
    )
    let convStateError = try NumericComparison.maxAbsoluteError(
        try output.nextConvState.readFloat32(),
        try attentionOutput.nextConvState.readFloat32()
    )
    let recurrentStateError = try NumericComparison.maxAbsoluteError(
        try output.nextRecurrentState.readFloat32(),
        try attentionOutput.nextRecurrentState.readFloat32()
    )

    #expect(layer.inputLayerNorm.shape == EdgeTensorShape([2]))
    #expect(layer.postAttentionLayerNorm.shape == EdgeTensorShape([2]))
    #expect(layer.mlp.gate.shape == EdgeTensorShape([2, 4]))
    #expect(output.hiddenStates.shape == EdgeTensorShape([1, 2]))
    #expect(output.nextConvState.shape == EdgeTensorShape([3, 10]))
    #expect(output.nextRecurrentState.shape == EdgeTensorShape([2, 2, 3]))
    #expect(hiddenError < 1e-5)
    #expect(convStateError < 1e-5)
    #expect(recurrentStateError < 1e-5)
}

@Test func qwenGDNDecoderLayerRejectsMismatchedLayerIndices() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-gdn-decoder-mismatch-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let index = try QwenModelBundleIndex(
        rootURL: root,
        architecture: try makeGDNWeightsArchitecture(),
        weightMap: gdnWeightsWeightMap()
    )
    try writeGDNSafeTensorsShard(
        entries: gdnDecoderLayerEntries(convShape: [10, 4, 1]),
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )

    let runtime = try EdgeMetalRuntime()
    let linearAttention = try QwenGDNWeights.loadHuggingFaceLayout(
        layerIndex: 0,
        weightStore: QwenModelWeightStore(bundleIndex: index),
        runtime: runtime
    )
    let mlp = QwenMLPWeights(
        layerIndex: 1,
        gate: try EdgeTensor(float32: Array(repeating: 0, count: 8), shape: EdgeTensorShape([2, 4]), runtime: runtime),
        up: try EdgeTensor(float32: Array(repeating: 0, count: 8), shape: EdgeTensorShape([2, 4]), runtime: runtime),
        down: try EdgeTensor(float32: Array(repeating: 0, count: 8), shape: EdgeTensorShape([4, 2]), runtime: runtime)
    )
    let norm = try EdgeTensor(float32: [1, 1], shape: EdgeTensorShape([2]), runtime: runtime)
    var rejected = false

    do {
        _ = try QwenGDNDecoderLayerReference(
            linearAttention: linearAttention,
            mlp: mlp,
            inputLayerNorm: norm,
            postAttentionLayerNorm: norm,
            rmsNormEpsilon: 1e-6
        )
    } catch QwenDecoderLayerReferenceError.layerIndexMismatch(attention: 0, mlp: 1) {
        rejected = true
    }

    #expect(rejected)
}

@Test func qwenGDNWeightsRejectsFullAttentionLayer() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-gdn-reject-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let index = try QwenModelBundleIndex(
        rootURL: root,
        architecture: try makeGDNWeightsArchitecture(),
        weightMap: gdnWeightsWeightMap()
    )

    do {
        _ = try QwenGDNWeights.loadHuggingFaceLayout(
            layerIndex: 1,
            weightStore: QwenModelWeightStore(bundleIndex: index),
            runtime: try EdgeMetalRuntime()
        )
        Issue.record("GDN weights loader should reject full-attention layers.")
    } catch QwenProjectionWeightError.layerIsNotGDN(layerIndex: 1, kind: .fullAttention) {
        return
    }
    Issue.record("GDN weights loader threw the wrong error for a full-attention layer.")
}

private func makeGDNWeightsArchitecture() throws -> QwenHybridArchitecture {
    try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 16,
        hiddenSize: 2,
        intermediateSize: 4,
        attentionHeadCount: 2,
        keyValueHeadCount: 1,
        headDimension: 1,
        linearValueHeadCount: 2,
        linearKeyHeadCount: 1,
        linearKeyHeadDimension: 3,
        linearValueHeadDimension: 2,
        linearConvKernelSize: 4,
        contextLength: 8,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000,
        layerKinds: [.gdn, .fullAttention]
    )
}

private func makeQuantizedGDNWeightsArchitecture() throws -> QwenHybridArchitecture {
    try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 16,
        hiddenSize: 2,
        intermediateSize: 4,
        attentionHeadCount: 2,
        keyValueHeadCount: 1,
        headDimension: 1,
        linearValueHeadCount: 2,
        linearKeyHeadCount: 1,
        linearKeyHeadDimension: 3,
        linearValueHeadDimension: 2,
        linearConvKernelSize: 4,
        contextLength: 8,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000,
        quantization: QwenQuantizationProfile(groupSize: 2, bits: 8),
        layerKinds: [.gdn, .fullAttention]
    )
}

private func makeExactGDNWeights(
    runtime: EdgeMetalRuntime,
    architecture: QwenHybridArchitecture
) throws -> QwenGDNWeights {
    QwenGDNWeights(
        layerIndex: 0,
        inProjQKV: try EdgeTensor(
            float32: gdnFloatRows(gdnQKVProjectionRows()),
            shape: EdgeTensorShape([2, 10]),
            runtime: runtime
        ),
        inProjZ: try EdgeTensor(
            float32: gdnFloatRows(gdnValueProjectionRows()),
            shape: EdgeTensorShape([2, 4]),
            runtime: runtime
        ),
        inProjB: try EdgeTensor(
            float32: gdnFloatRows(gdnScalarProjectionRows()),
            shape: EdgeTensorShape([2, 2]),
            runtime: runtime
        ),
        inProjA: try EdgeTensor(
            float32: gdnFloatRows(gdnScalarProjectionRows()),
            shape: EdgeTensorShape([2, 2]),
            runtime: runtime
        ),
        conv1D: try EdgeTensor(
            float32: Array(repeating: Float(1), count: 40),
            shape: EdgeTensorShape([10, 4, 1]),
            runtime: runtime
        ),
        convWeightLayout: .mlxSanitizedDepthwise,
        aLog: try EdgeTensor(float32: [0.25, 0.5], shape: EdgeTensorShape([2]), runtime: runtime),
        dtBias: try EdgeTensor(float32: [1, 2], shape: EdgeTensorShape([2]), runtime: runtime),
        norm: try EdgeTensor(float32: [1, 1], shape: EdgeTensorShape([2]), runtime: runtime),
        outProj: try EdgeTensor(
            float32: gdnFloatRows(gdnOutputProjectionRows()),
            shape: EdgeTensorShape([4, 2]),
            runtime: runtime
        ),
        linearKeyHeadCount: architecture.linearKeyHeadCount,
        linearValueHeadCount: architecture.linearValueHeadCount,
        linearKeyHeadDimension: architecture.linearKeyHeadDimension,
        linearValueHeadDimension: architecture.linearValueHeadDimension,
        linearKeyHiddenSize: architecture.linearKeyHiddenSize,
        linearValueHiddenSize: architecture.linearValueHiddenSize,
        rmsNormEpsilon: architecture.rmsNormEpsilon
    )
}

private func makeExactQuantizedGDNWeights(
    runtime: EdgeMetalRuntime,
    architecture: QwenHybridArchitecture
) throws -> QwenQuantizedGDNWeights {
    QwenQuantizedGDNWeights(
        layerIndex: 0,
        inProjQKV: try exactGDNQuantizedRuntimeLayout(gdnQKVProjectionRows()),
        inProjZ: try exactGDNQuantizedRuntimeLayout(gdnValueProjectionRows()),
        inProjB: try exactGDNQuantizedRuntimeLayout(gdnScalarProjectionRows()),
        inProjA: try exactGDNQuantizedRuntimeLayout(gdnScalarProjectionRows()),
        conv1D: try EdgeTensor(
            float32: Array(repeating: Float(1), count: 40),
            shape: EdgeTensorShape([10, 4, 1]),
            runtime: runtime
        ),
        convWeightLayout: .mlxSanitizedDepthwise,
        aLog: try EdgeTensor(float32: [0.25, 0.5], shape: EdgeTensorShape([2]), runtime: runtime),
        dtBias: try EdgeTensor(float32: [1, 2], shape: EdgeTensorShape([2]), runtime: runtime),
        norm: try EdgeTensor(float32: [1, 1], shape: EdgeTensorShape([2]), runtime: runtime),
        outProj: try exactGDNQuantizedRuntimeLayout(gdnOutputProjectionRows()),
        linearKeyHeadCount: architecture.linearKeyHeadCount,
        linearValueHeadCount: architecture.linearValueHeadCount,
        linearKeyHeadDimension: architecture.linearKeyHeadDimension,
        linearValueHeadDimension: architecture.linearValueHeadDimension,
        linearKeyHiddenSize: architecture.linearKeyHiddenSize,
        linearValueHiddenSize: architecture.linearValueHiddenSize,
        rmsNormEpsilon: architecture.rmsNormEpsilon
    )
}

private func makeExactGDNDecoderLayer(
    runtime: EdgeMetalRuntime,
    architecture: QwenHybridArchitecture
) throws -> QwenGDNDecoderLayerReference {
    try QwenGDNDecoderLayerReference(
        linearAttention: makeExactGDNWeights(runtime: runtime, architecture: architecture),
        mlp: makeExactGDNMLPWeights(runtime: runtime),
        inputLayerNorm: EdgeTensor(float32: [1, 1], shape: EdgeTensorShape([2]), runtime: runtime),
        postAttentionLayerNorm: EdgeTensor(float32: [1, 1], shape: EdgeTensorShape([2]), runtime: runtime),
        rmsNormEpsilon: architecture.rmsNormEpsilon
    )
}

private func makeExactQuantizedGDNDecoderLayer(
    runtime: EdgeMetalRuntime,
    architecture: QwenHybridArchitecture
) throws -> QwenQuantizedGDNDecoderLayerReference {
    try QwenQuantizedGDNDecoderLayerReference(
        linearAttention: makeExactQuantizedGDNWeights(runtime: runtime, architecture: architecture),
        mlp: makeExactQuantizedGDNMLPWeights(),
        inputLayerNorm: EdgeTensor(float32: [1, 1], shape: EdgeTensorShape([2]), runtime: runtime),
        postAttentionLayerNorm: EdgeTensor(float32: [1, 1], shape: EdgeTensorShape([2]), runtime: runtime),
        rmsNormEpsilon: architecture.rmsNormEpsilon
    )
}

private func makeExactGDNMLPWeights(runtime: EdgeMetalRuntime) throws -> QwenMLPWeights {
    QwenMLPWeights(
        layerIndex: 0,
        gate: try EdgeTensor(
            float32: gdnFloatRows(gdnMLPProjectionRows()),
            shape: EdgeTensorShape([2, 4]),
            runtime: runtime
        ),
        up: try EdgeTensor(
            float32: gdnFloatRows(gdnMLPProjectionRows()),
            shape: EdgeTensorShape([2, 4]),
            runtime: runtime
        ),
        down: try EdgeTensor(
            float32: gdnFloatRows(gdnMLPOutputRows()),
            shape: EdgeTensorShape([4, 2]),
            runtime: runtime
        )
    )
}

private func makeExactQuantizedGDNMLPWeights() throws -> QwenQuantizedMLPWeights {
    try QwenQuantizedMLPWeights(
        layerIndex: 0,
        gate: exactGDNQuantizedRuntimeLayout(gdnMLPProjectionRows()),
        up: exactGDNQuantizedRuntimeLayout(gdnMLPProjectionRows()),
        down: exactGDNQuantizedRuntimeLayout(gdnMLPOutputRows())
    )
}

private func gdnQKVProjectionRows() -> [[UInt32]] {
    [
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ]
}

private func gdnValueProjectionRows() -> [[UInt32]] {
    [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]
}

private func gdnScalarProjectionRows() -> [[UInt32]] {
    [
        [1, 0],
        [0, 1],
    ]
}

private func gdnOutputProjectionRows() -> [[UInt32]] {
    [
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1],
    ]
}

private func gdnMLPProjectionRows() -> [[UInt32]] {
    [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]
}

private func gdnMLPOutputRows() -> [[UInt32]] {
    [
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1],
    ]
}

private func gdnFloatRows(_ rows: [[UInt32]]) -> [Float] {
    rows.flatMap { row in row.map { Float(Int($0)) } }
}

private func exactGDNQuantizedRuntimeLayout(_ rows: [[UInt32]]) throws -> EdgeQuantizedTensor {
    let quantizedRows = gdnQuantizedRowsForRuntimeLayout(rows)
    return try exactGDNQuantizedRows(quantizedRows)
}

private func gdnQuantizedRowsForRuntimeLayout(_ rows: [[UInt32]]) -> [[UInt32]] {
    let columns = rows.first?.count ?? 0
    return (0..<columns).map { column in
        rows.map { $0[column] }
    }
}

private func exactGDNQuantizedRows(_ rows: [[UInt32]]) throws -> EdgeQuantizedTensor {
    let columns = rows.first?.count ?? 0
    return try EdgeQuantizedTensor(
        shape: [rows.count, columns],
        packedShape: [rows.count, (columns * 8 + 31) / 32],
        scaleShape: [rows.count, 1],
        groupSize: columns,
        bits: 8,
        packedValues: gdnPackQuantizedRows(rows, bits: 8),
        scales: Array(repeating: 1, count: rows.count),
        biases: Array(repeating: 0, count: rows.count)
    )
}

private func gdnPackQuantizedRows(_ rows: [[UInt32]], bits: Int) -> [UInt32] {
    rows.flatMap { gdnPackQuantizedWords($0, bits: bits) }
}

private func gdnPackQuantizedWords(_ values: [UInt32], bits: Int) -> [UInt32] {
    var words = Array(repeating: UInt32.zero, count: (values.count * bits + 31) / 32)
    let mask = UInt32((1 << bits) - 1)
    for (index, value) in values.enumerated() {
        let bitOffset = index * bits
        let wordIndex = bitOffset / 32
        let shift = bitOffset % 32
        words[wordIndex] |= (value & mask) << UInt32(shift)
        if shift + bits > 32 {
            words[wordIndex + 1] |= (value & mask) >> UInt32(32 - shift)
        }
    }
    return words
}

private func quantizedGDNWeightsWeightMap() -> [String: String] {
    var weightMap = gdnWeightsWeightMap()
    for name in quantizedGDNWeightNames() {
        let base = String(name.dropLast(".weight".count))
        weightMap["\(base).scales"] = "model-00001-of-00001.safetensors"
    }
    return weightMap
}

private func quantizedGDNDecoderLayerWeightMap() -> [String: String] {
    var weightMap = quantizedGDNWeightsWeightMap()
    for name in quantizedGDNMLPWeightNames() {
        let base = String(name.dropLast(".weight".count))
        weightMap["\(base).scales"] = "model-00001-of-00001.safetensors"
    }
    return weightMap
}

private func quantizedGDNWeightNames() -> [String] {
    let prefix = "model.layers.0.linear_attn"
    return [
        "\(prefix).in_proj_qkv.weight",
        "\(prefix).in_proj_z.weight",
        "\(prefix).in_proj_b.weight",
        "\(prefix).in_proj_a.weight",
        "\(prefix).out_proj.weight",
    ]
}

private func quantizedGDNMLPWeightNames() -> [String] {
    let prefix = "model.layers.0.mlp"
    return [
        "\(prefix).gate_proj.weight",
        "\(prefix).up_proj.weight",
        "\(prefix).down_proj.weight",
    ]
}

private func gdnWeightsWeightMap() -> [String: String] {
    var weightMap: [String: String] = [:]
    for name in requiredGDNWeightNames() + requiredGDNWeightsModelLevelNames() + requiredGDNWeightsFullAttentionNames() {
        weightMap[name] = "model-00001-of-00001.safetensors"
    }
    return weightMap
}

private func requiredGDNWeightNames() -> [String] {
    let layerPrefix = "model.layers.0"
    let prefix = "model.layers.0.linear_attn"
    return [
        "\(layerPrefix).input_layernorm.weight",
        "\(layerPrefix).post_attention_layernorm.weight",
        "\(layerPrefix).mlp.gate_proj.weight",
        "\(layerPrefix).mlp.up_proj.weight",
        "\(layerPrefix).mlp.down_proj.weight",
        "\(prefix).A_log",
        "\(prefix).conv1d.weight",
        "\(prefix).dt_bias",
        "\(prefix).in_proj_a.weight",
        "\(prefix).in_proj_b.weight",
        "\(prefix).in_proj_qkv.weight",
        "\(prefix).in_proj_z.weight",
        "\(prefix).norm.weight",
        "\(prefix).out_proj.weight",
    ]
}

private func requiredGDNWeightsModelLevelNames() -> [String] {
    [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.lm_head.weight",
    ]
}

private func requiredGDNWeightsFullAttentionNames() -> [String] {
    let prefix = "model.layers.1.self_attn"
    let layerPrefix = "model.layers.1"
    return [
        "\(layerPrefix).input_layernorm.weight",
        "\(layerPrefix).post_attention_layernorm.weight",
        "\(layerPrefix).mlp.gate_proj.weight",
        "\(layerPrefix).mlp.up_proj.weight",
        "\(layerPrefix).mlp.down_proj.weight",
        "\(prefix).q_proj.weight",
        "\(prefix).k_proj.weight",
        "\(prefix).v_proj.weight",
        "\(prefix).o_proj.weight",
        "\(prefix).q_norm.weight",
        "\(prefix).k_norm.weight",
    ]
}

private func gdnWeightEntries(convShape: [Int]) -> [(name: String, shape: [Int], values: [Float])] {
    let prefix = "model.layers.0.linear_attn"
    return [
        ("\(prefix).in_proj_qkv.weight", [10, 2], sequentialFloats(count: 20)),
        ("\(prefix).in_proj_z.weight", [4, 2], sequentialFloats(count: 8)),
        ("\(prefix).in_proj_b.weight", [2, 2], sequentialFloats(count: 4)),
        ("\(prefix).in_proj_a.weight", [2, 2], sequentialFloats(count: 4)),
        ("\(prefix).conv1d.weight", convShape, sequentialFloats(count: 40)),
        ("\(prefix).A_log", [2], [0.25, 0.5]),
        ("\(prefix).dt_bias", [2], [1, 2]),
        ("\(prefix).norm.weight", [2], [3, 4]),
        ("\(prefix).out_proj.weight", [2, 4], sequentialFloats(count: 8)),
    ]
}

private func gdnDecoderLayerEntries(convShape: [Int]) -> [(name: String, shape: [Int], values: [Float])] {
    gdnWeightEntries(convShape: convShape) + [
        ("model.layers.0.input_layernorm.weight", [2], [1, 1]),
        ("model.layers.0.post_attention_layernorm.weight", [2], [1, 1]),
        ("model.layers.0.mlp.gate_proj.weight", [4, 2], Array(repeating: Float.zero, count: 8)),
        ("model.layers.0.mlp.up_proj.weight", [4, 2], Array(repeating: Float.zero, count: 8)),
        ("model.layers.0.mlp.down_proj.weight", [2, 4], Array(repeating: Float.zero, count: 8)),
    ]
}

private func writeGDNSafeTensorsShard(
    entries: [(name: String, shape: [Int], values: [Float])],
    to url: URL
) throws {
    var payload = Data()
    var fields: [String] = []
    var offset = 0
    for entry in entries {
        let data = floatDataForGDNWeights(entry.values)
        let end = offset + data.count
        fields.append(
            """
            "\(entry.name)": {
              "dtype": "F32",
              "shape": \(jsonIntArrayForGDNWeights(entry.shape)),
              "data_offsets": [\(offset), \(end)]
            }
            """
        )
        payload.append(data)
        offset = end
    }
    let headerJSON = "{\(fields.joined(separator: ","))}"
    let headerData = headerJSON.data(using: .utf8)!
    var headerLength = UInt64(headerData.count).littleEndian
    var fileData = withUnsafeBytes(of: &headerLength) { Data($0) }
    fileData.append(headerData)
    fileData.append(payload)
    try fileData.write(to: url)
}

private func writeQuantizedGDNSafeTensorsShard(to url: URL) throws {
    let prefix = "model.layers.0.linear_attn"
    var entries: [(name: String, dtype: String, shape: [Int], data: Data)] = []
    entries += quantizedGDNEntries(
        name: "\(prefix).in_proj_qkv.weight",
        runtimeLayoutRows: gdnQKVProjectionRows()
    )
    entries += quantizedGDNEntries(
        name: "\(prefix).in_proj_z.weight",
        runtimeLayoutRows: gdnValueProjectionRows()
    )
    entries += quantizedGDNEntries(
        name: "\(prefix).in_proj_b.weight",
        runtimeLayoutRows: gdnScalarProjectionRows()
    )
    entries += quantizedGDNEntries(
        name: "\(prefix).in_proj_a.weight",
        runtimeLayoutRows: gdnScalarProjectionRows()
    )
    entries += quantizedGDNEntries(
        name: "\(prefix).out_proj.weight",
        runtimeLayoutRows: gdnOutputProjectionRows()
    )
    entries += [
        ("\(prefix).conv1d.weight", "F32", [10, 4, 1], floatDataForGDNWeights(Array(repeating: Float(1), count: 40))),
        ("\(prefix).A_log", "F32", [2], floatDataForGDNWeights([0.25, 0.5])),
        ("\(prefix).dt_bias", "F32", [2], floatDataForGDNWeights([1, 2])),
        ("\(prefix).norm.weight", "F32", [2], floatDataForGDNWeights([1, 1])),
    ]
    try writeGDNSafeTensorsEntries(entries, to: url)
}

private func writeQuantizedGDNDecoderLayerSafeTensorsShard(to url: URL) throws {
    let linearPrefix = "model.layers.0.linear_attn"
    let mlpPrefix = "model.layers.0.mlp"
    var entries: [(name: String, dtype: String, shape: [Int], data: Data)] = []
    entries += quantizedGDNEntries(
        name: "\(linearPrefix).in_proj_qkv.weight",
        runtimeLayoutRows: gdnQKVProjectionRows()
    )
    entries += quantizedGDNEntries(
        name: "\(linearPrefix).in_proj_z.weight",
        runtimeLayoutRows: gdnValueProjectionRows()
    )
    entries += quantizedGDNEntries(
        name: "\(linearPrefix).in_proj_b.weight",
        runtimeLayoutRows: gdnScalarProjectionRows()
    )
    entries += quantizedGDNEntries(
        name: "\(linearPrefix).in_proj_a.weight",
        runtimeLayoutRows: gdnScalarProjectionRows()
    )
    entries += quantizedGDNEntries(
        name: "\(linearPrefix).out_proj.weight",
        runtimeLayoutRows: gdnOutputProjectionRows()
    )
    entries += quantizedGDNEntries(
        name: "\(mlpPrefix).gate_proj.weight",
        runtimeLayoutRows: gdnMLPProjectionRows()
    )
    entries += quantizedGDNEntries(
        name: "\(mlpPrefix).up_proj.weight",
        runtimeLayoutRows: gdnMLPProjectionRows()
    )
    entries += quantizedGDNEntries(
        name: "\(mlpPrefix).down_proj.weight",
        runtimeLayoutRows: gdnMLPOutputRows()
    )
    entries += [
        ("\(linearPrefix).conv1d.weight", "F32", [10, 4, 1], floatDataForGDNWeights(Array(repeating: Float(1), count: 40))),
        ("\(linearPrefix).A_log", "F32", [2], floatDataForGDNWeights([0.25, 0.5])),
        ("\(linearPrefix).dt_bias", "F32", [2], floatDataForGDNWeights([1, 2])),
        ("\(linearPrefix).norm.weight", "F32", [2], floatDataForGDNWeights([1, 1])),
        ("model.layers.0.input_layernorm.weight", "F32", [2], floatDataForGDNWeights([1, 1])),
        ("model.layers.0.post_attention_layernorm.weight", "F32", [2], floatDataForGDNWeights([1, 1])),
    ]
    try writeGDNSafeTensorsEntries(entries, to: url)
}

private func quantizedGDNEntries(
    name: String,
    runtimeLayoutRows: [[UInt32]],
    groupSize: Int = 2,
    bits: Int = 8
) -> [(name: String, dtype: String, shape: [Int], data: Data)] {
    let rows = gdnQuantizedRowsForRuntimeLayout(runtimeLayoutRows)
    let columns = rows.first?.count ?? 0
    let scaleColumns = columns / groupSize
    let base = String(name.dropLast(".weight".count))
    return [
        (
            name,
            "U32",
            [rows.count, (columns * bits + 31) / 32],
            uint32DataForGDNWeights(gdnPackQuantizedRows(rows, bits: bits))
        ),
        (
            "\(base).scales",
            "F32",
            [rows.count, scaleColumns],
            floatDataForGDNWeights(Array(repeating: Float(1), count: rows.count * scaleColumns))
        ),
    ]
}

private func writeGDNSafeTensorsEntries(
    _ entries: [(name: String, dtype: String, shape: [Int], data: Data)],
    to url: URL
) throws {
    var payload = Data()
    var fields: [String] = []
    var offset = 0
    for entry in entries {
        let end = offset + entry.data.count
        fields.append(
            """
            "\(entry.name)": {
              "dtype": "\(entry.dtype)",
              "shape": \(jsonIntArrayForGDNWeights(entry.shape)),
              "data_offsets": [\(offset), \(end)]
            }
            """
        )
        payload.append(entry.data)
        offset = end
    }
    let headerJSON = "{\(fields.joined(separator: ","))}"
    let headerData = headerJSON.data(using: .utf8)!
    var headerLength = UInt64(headerData.count).littleEndian
    var fileData = withUnsafeBytes(of: &headerLength) { Data($0) }
    fileData.append(headerData)
    fileData.append(payload)
    try fileData.write(to: url)
}

private func sequentialFloats(count: Int) -> [Float] {
    (0..<count).map(Float.init)
}

private func floatDataForGDNWeights(_ values: [Float]) -> Data {
    values.withUnsafeBufferPointer { buffer in
        Data(buffer: buffer)
    }
}

private func uint32DataForGDNWeights(_ values: [UInt32]) -> Data {
    var data = Data()
    for value in values {
        var littleEndianValue = value.littleEndian
        data.append(withUnsafeBytes(of: &littleEndianValue) { Data($0) })
    }
    return data
}

private func jsonIntArrayForGDNWeights(_ values: [Int]) -> String {
    "[\(values.map(String.init).joined(separator: ","))]"
}
