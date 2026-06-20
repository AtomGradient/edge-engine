// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Testing
@testable import EdgeEngine

@Test func qwenFullAttentionDecoderLayerRunsResidualAttentionAndMLP() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeDecoderLayerArchitecture()
    let inverseRoot2 = Float(1.0 / Double(2).squareRoot())
    let layer = try makeDecoderLayer(
        architecture: architecture,
        inputNorm: [inverseRoot2, 1],
        postAttentionNorm: [1, inverseRoot2],
        runtime: runtime
    )
    let hiddenStates = try EdgeTensor(float32: [1, 0], shape: EdgeTensorShape([1, 2]), runtime: runtime)

    let output = try layer.outputTensor(hiddenStates: hiddenStates, executor: executor)
    let mlpActivation = try CPUReferenceOps.swiglu(gate: [1, 0], up: [2, 0])
    let expected: [Float] = [mlpActivation[0], -1]
    let error = try NumericComparison.maxAbsoluteError(try output.readFloat32(), expected)

    #expect(output.shape == EdgeTensorShape([1, 2]))
    #expect(error < 1e-5)
}

@Test func qwenFullAttentionDecoderLayerLoadsHuggingFaceLayoutFromBundleShard() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-decoder-layer-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let architecture = try makeDecoderLayerArchitecture()
    let index = try QwenModelBundleIndex(
        rootURL: root,
        architecture: architecture,
        weightMap: completeDecoderLayerWeightMap()
    )
    let inverseRoot2 = Float(1.0 / Double(2).squareRoot())
    try writeDecoderLayerSafeTensorsShard(
        tensors: [
            "model.layers.0.input_layernorm.weight": (shape: [2], values: [inverseRoot2, 1]),
            "model.layers.0.post_attention_layernorm.weight": (shape: [2], values: [1, inverseRoot2]),
            "model.layers.0.self_attn.q_proj.weight": (shape: [4, 2], values: [
                1, 0,
                0, 1,
                3, 0,
                0, 2,
            ]),
            "model.layers.0.self_attn.k_proj.weight": (shape: [1, 2], values: [3, 4]),
            "model.layers.0.self_attn.v_proj.weight": (shape: [1, 2], values: [-1, 2]),
            "model.layers.0.self_attn.o_proj.weight": (shape: [2, 2], values: [
                2, 0,
                0, 2,
            ]),
            "model.layers.0.self_attn.q_norm.weight": (shape: [1], values: [1]),
            "model.layers.0.self_attn.k_norm.weight": (shape: [1], values: [1]),
            "model.layers.0.mlp.gate_proj.weight": (shape: [2, 2], values: [
                0, -1,
                0, 0,
            ]),
            "model.layers.0.mlp.up_proj.weight": (shape: [2, 2], values: [
                0, -2,
                0, 0,
            ]),
            "model.layers.0.mlp.down_proj.weight": (shape: [2, 2], values: [
                1, 0,
                0, 0,
            ]),
        ],
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let layer = try QwenFullAttentionDecoderLayerReference.loadHuggingFaceLayout(
        layerIndex: 0,
        weightStore: QwenModelWeightStore(bundleIndex: index),
        runtime: runtime
    )
    let hiddenStates = try EdgeTensor(float32: [1, 0], shape: EdgeTensorShape([1, 2]), runtime: runtime)

    let output = try layer.outputTensor(hiddenStates: hiddenStates, executor: executor)
    let mlpActivation = try CPUReferenceOps.swiglu(gate: [1, 0], up: [2, 0])
    let expected: [Float] = [mlpActivation[0], -1]
    let error = try NumericComparison.maxAbsoluteError(try output.readFloat32(), expected)

    #expect(layer.attention.outputProjectionWeights?.weight.shape == EdgeTensorShape([2, 2]))
    #expect(layer.inputLayerNorm.shape == EdgeTensorShape([2]))
    #expect(output.shape == EdgeTensorShape([1, 2]))
    #expect(error < 1e-5)
}

@Test func qwenQuantizedFullAttentionDecoderLayerMatchesFloatReference() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeDecoderLayerArchitecture()
    let floatLayer = try makePositiveDecoderLayer(architecture: architecture, runtime: runtime)
    let quantizedLayer = try makePositiveQuantizedDecoderLayer(architecture: architecture, runtime: runtime)
    let hiddenStates = try EdgeTensor(
        float32: [
            1, 2,
            2, 1,
        ],
        shape: EdgeTensorShape([2, 2]),
        runtime: runtime
    )

    let expected = try floatLayer.outputTensor(hiddenStates: hiddenStates, executor: executor)
    let actual = try quantizedLayer.outputTensor(hiddenStates: hiddenStates, executor: executor)
    let error = try NumericComparison.maxAbsoluteError(try actual.readFloat32(), try expected.readFloat32())

    #expect(actual.shape == EdgeTensorShape([2, 2]))
    #expect(error < 1e-5)
}

@Test func qwenQuantizedFullAttentionDecoderLayerUsesKVCacheLikeFloatReference() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeDecoderLayerArchitecture()
    let floatLayer = try makePositiveDecoderLayer(architecture: architecture, runtime: runtime)
    let quantizedLayer = try makePositiveQuantizedDecoderLayer(architecture: architecture, runtime: runtime)
    let fullHiddenStates = try EdgeTensor(
        float32: [
            1, 2,
            2, 1,
        ],
        shape: EdgeTensorShape([2, 2]),
        runtime: runtime
    )
    let kvCache = try QwenKVCache(
        shape: QwenKVCacheShape.shape(for: architecture, layerIndex: 0, capacity: 4),
        runtime: runtime
    )

    let expected = try floatLayer.outputTensor(hiddenStates: fullHiddenStates, executor: executor)
    let firstTokenOutput = try quantizedLayer.outputTensor(
        hiddenStates: EdgeTensor(float32: [1, 2], shape: EdgeTensorShape([1, 2]), runtime: runtime),
        executor: executor,
        kvCache: kvCache
    )
    let secondTokenOutput = try quantizedLayer.outputTensor(
        hiddenStates: EdgeTensor(float32: [2, 1], shape: EdgeTensorShape([1, 2]), runtime: runtime),
        executor: executor,
        kvCache: kvCache
    )
    let actual = try firstTokenOutput.readFloat32() + secondTokenOutput.readFloat32()
    let error = try NumericComparison.maxAbsoluteError(actual, try expected.readFloat32())

    #expect(kvCache.tokenCount == 2)
    #expect(error < 1e-5)
}

@Test func qwenFullAttentionDecoderLayerRejectsMismatchedLayerIndices() throws {
    let runtime = try EdgeMetalRuntime()
    let architecture = try makeDecoderLayerArchitecture()
    let attention = try makeDecoderAttention(architecture: architecture, runtime: runtime)
    let mlp = QwenMLPWeights(
        layerIndex: 1,
        gate: try EdgeTensor(
            float32: Array(repeating: 0, count: 4),
            shape: EdgeTensorShape([2, 2]),
            runtime: runtime
        ),
        up: try EdgeTensor(
            float32: Array(repeating: 0, count: 4),
            shape: EdgeTensorShape([2, 2]),
            runtime: runtime
        ),
        down: try EdgeTensor(
            float32: Array(repeating: 0, count: 4),
            shape: EdgeTensorShape([2, 2]),
            runtime: runtime
        )
    )
    let norm = try EdgeTensor(float32: [1, 1], shape: EdgeTensorShape([2]), runtime: runtime)
    var rejected = false

    do {
        _ = try QwenFullAttentionDecoderLayerReference(
            attention: attention,
            mlp: mlp,
            inputLayerNorm: norm,
            postAttentionLayerNorm: norm,
            rmsNormEpsilon: 0
        )
    } catch QwenDecoderLayerReferenceError.layerIndexMismatch(attention: 0, mlp: 1) {
        rejected = true
    }

    #expect(rejected)
}

@Test func qwenQuantizedFullAttentionDecoderLayerRejectsMismatchedLayerIndices() throws {
    let runtime = try EdgeMetalRuntime()
    let architecture = try makeDecoderLayerArchitecture()
    let attention = try makePositiveQuantizedAttention(architecture: architecture)
    let mlp = try QwenQuantizedMLPWeights(
        layerIndex: 1,
        gate: exactDecoderQuantizedRows([[1, 0], [0, 1]]),
        up: exactDecoderQuantizedRows([[1, 0], [0, 1]]),
        down: exactDecoderQuantizedRows([[1, 0], [0, 1]])
    )
    let norm = try EdgeTensor(float32: [1, 1], shape: EdgeTensorShape([2]), runtime: runtime)
    var rejected = false

    do {
        _ = try QwenQuantizedFullAttentionDecoderLayerReference(
            attention: attention,
            mlp: mlp,
            inputLayerNorm: norm,
            postAttentionLayerNorm: norm,
            rmsNormEpsilon: 0
        )
    } catch QwenDecoderLayerReferenceError.layerIndexMismatch(attention: 0, mlp: 1) {
        rejected = true
    }

    #expect(rejected)
}

@Test func qwenHybridDecoderLayerRoutesFullAttentionWithKVCache() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeDecoderLayerArchitecture()
    let layer = try makeDecoderLayer(
        architecture: architecture,
        inputNorm: [1, 1],
        postAttentionNorm: [1, 1],
        runtime: runtime
    )
    let cache = try QwenKVCache(
        shape: QwenKVCacheShape.shape(for: architecture, layerIndex: 0, capacity: 4),
        runtime: runtime
    )
    let hiddenStates = try EdgeTensor(float32: [1, 0], shape: EdgeTensorShape([1, 2]), runtime: runtime)
    let hybrid = QwenHybridDecoderLayerReference.fullAttention(layer)

    let output = try hybrid.outputTensor(
        hiddenStates: hiddenStates,
        executor: executor,
        kvCache: cache
    )

    #expect(hybrid.layerIndex == 0)
    #expect(hybrid.kind == .fullAttention)
    #expect(output.shape == EdgeTensorShape([1, 2]))
    #expect(cache.tokenCount == 1)
}

@Test func qwenHybridDecoderLayerRoutesQuantizedFullAttentionWithKVCache() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeDecoderLayerArchitecture()
    let layer = try makePositiveQuantizedDecoderLayer(
        architecture: architecture,
        runtime: runtime
    )
    let cache = try QwenKVCache(
        shape: QwenKVCacheShape.shape(for: architecture, layerIndex: 0, capacity: 4),
        runtime: runtime
    )
    let hiddenStates = try EdgeTensor(float32: [1, 2], shape: EdgeTensorShape([1, 2]), runtime: runtime)
    let hybrid = QwenHybridDecoderLayerReference.quantizedFullAttention(layer)

    let output = try hybrid.outputTensor(
        hiddenStates: hiddenStates,
        executor: executor,
        kvCache: cache
    )

    #expect(hybrid.layerIndex == 0)
    #expect(hybrid.kind == .fullAttention)
    #expect(output.shape == EdgeTensorShape([1, 2]))
    #expect(cache.tokenCount == 1)
}

@Test func qwenHybridDecoderLayerRoutesGDNAndUpdatesCache() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeDecoderLayerArchitecture()
    let layer = try makeHybridDecoderGDNLayer(runtime: runtime)
    let cache = try QwenGDNCache(
        shape: QwenGDNCacheShape.shape(for: architecture, layerIndex: 1),
        runtime: runtime
    )
    let hiddenStates = try EdgeTensor(float32: [1, 0], shape: EdgeTensorShape([1, 2]), runtime: runtime)
    let hybrid = QwenHybridDecoderLayerReference.gdn(layer)

    let output = try hybrid.outputTensor(
        hiddenStates: hiddenStates,
        executor: executor,
        gdnCache: cache
    )
    let convState = try cache.convState.readFloat32()

    #expect(hybrid.layerIndex == 1)
    #expect(hybrid.kind == .gdn)
    #expect(output.shape == EdgeTensorShape([1, 2]))
    #expect(cache.tokenPosition == 1)
    #expect(Array(convState.prefix(6)) == Array(repeating: Float.zero, count: 6))
    #expect(convState[6] > 1)
    #expect(convState[7] == 0)
    #expect(convState[8] > 1)
}

@Test func qwenHybridDecoderLayerRoutesQuantizedGDNAndUpdatesCache() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeDecoderLayerArchitecture()
    let layer = try makeHybridQuantizedDecoderGDNLayer(runtime: runtime)
    let cache = try QwenGDNCache(
        shape: QwenGDNCacheShape.shape(for: architecture, layerIndex: 1),
        runtime: runtime
    )
    let hiddenStates = try EdgeTensor(float32: [1, 0], shape: EdgeTensorShape([1, 2]), runtime: runtime)
    let hybrid = QwenHybridDecoderLayerReference.quantizedGDN(layer)

    let output = try hybrid.outputTensor(
        hiddenStates: hiddenStates,
        executor: executor,
        gdnCache: cache
    )
    let convState = try cache.convState.readFloat32()

    #expect(hybrid.layerIndex == 1)
    #expect(hybrid.kind == .gdn)
    #expect(output.shape == EdgeTensorShape([1, 2]))
    #expect(cache.tokenPosition == 1)
    #expect(Array(convState.prefix(6)) == Array(repeating: Float.zero, count: 6))
    #expect(convState[6] > 1)
    #expect(convState[7] == 0)
    #expect(convState[8] > 1)
}

@Test func qwenHybridDecoderLayerRejectsWrongCacheKind() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeDecoderLayerArchitecture()
    let fullLayer = try makeDecoderLayer(
        architecture: architecture,
        inputNorm: [1, 1],
        postAttentionNorm: [1, 1],
        runtime: runtime
    )
    let gdnLayer = try makeHybridDecoderGDNLayer(runtime: runtime)
    let kvCache = try QwenKVCache(
        shape: QwenKVCacheShape.shape(for: architecture, layerIndex: 0, capacity: 4),
        runtime: runtime
    )
    let gdnCache = try QwenGDNCache(
        shape: QwenGDNCacheShape.shape(for: architecture, layerIndex: 1),
        runtime: runtime
    )
    let hiddenStates = try EdgeTensor(float32: [1, 0], shape: EdgeTensorShape([1, 2]), runtime: runtime)
    var rejectedFullAttention = false
    var rejectedGDN = false
    var rejectedMissingGDNCache = false

    do {
        _ = try QwenHybridDecoderLayerReference.fullAttention(fullLayer).outputTensor(
            hiddenStates: hiddenStates,
            executor: executor,
            kvCache: kvCache,
            gdnCache: gdnCache
        )
    } catch QwenHybridDecoderLayerError.unexpectedGDNCache(layerIndex: 0) {
        rejectedFullAttention = true
    }

    do {
        _ = try QwenHybridDecoderLayerReference.gdn(gdnLayer).outputTensor(
            hiddenStates: hiddenStates,
            executor: executor,
            kvCache: kvCache,
            gdnCache: gdnCache
        )
    } catch QwenHybridDecoderLayerError.unexpectedKVCache(layerIndex: 1) {
        rejectedGDN = true
    }

    do {
        _ = try QwenHybridDecoderLayerReference.gdn(gdnLayer).outputTensor(
            hiddenStates: hiddenStates,
            executor: executor
        )
    } catch QwenHybridDecoderLayerError.missingGDNCache(layerIndex: 1) {
        rejectedMissingGDNCache = true
    }

    #expect(rejectedFullAttention)
    #expect(rejectedGDN)
    #expect(rejectedMissingGDNCache)
}

@Test func qwenHybridDecoderLayerRejectsMismatchedGDNCacheLayer() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let layer = try makeHybridDecoderGDNLayer(runtime: runtime)
    let wrongCache = try QwenGDNCache(
        shape: QwenGDNCacheShape(
            layerIndex: 2,
            convStateTokenCount: 3,
            convHiddenSize: 3,
            recurrentStateShape: QwenGDNStateShape(
                layerIndex: 2,
                valueHeadCount: 1,
                valueHeadDimension: 1,
                keyHeadDimension: 1
            )
        ),
        runtime: runtime
    )
    let hiddenStates = try EdgeTensor(float32: [1, 0], shape: EdgeTensorShape([1, 2]), runtime: runtime)
    var rejected = false

    do {
        _ = try QwenHybridDecoderLayerReference.gdn(layer).outputTensor(
            hiddenStates: hiddenStates,
            executor: executor,
            gdnCache: wrongCache
        )
    } catch QwenHybridDecoderLayerError.gdnCacheLayerMismatch(expected: 1, actual: 2) {
        rejected = true
    }

    #expect(rejected)
}

@Test func qwenHybridDecoderLayerLoadsHuggingFaceLayoutByManifest() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-hybrid-decoder-layer-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let index = try QwenModelBundleIndex(
        rootURL: root,
        architecture: try makeDecoderLayerArchitecture(),
        weightMap: completeDecoderLayerWeightMap()
    )
    var tensors = decoderFullAttentionLayer0TensorEntries()
    for (name, tensor) in decoderGDNLayer1TensorEntries() {
        tensors[name] = tensor
    }
    try writeDecoderLayerSafeTensorsShard(
        tensors: tensors,
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )
    let runtime = try EdgeMetalRuntime()
    let weightStore = QwenModelWeightStore(bundleIndex: index)

    let fullAttention = try QwenHybridDecoderLayerReference.loadHuggingFaceLayout(
        layerIndex: 0,
        weightStore: weightStore,
        runtime: runtime
    )
    let gdn = try QwenHybridDecoderLayerReference.loadHuggingFaceLayout(
        layerIndex: 1,
        weightStore: weightStore,
        runtime: runtime
    )

    #expect(fullAttention.layerIndex == 0)
    #expect(fullAttention.kind == .fullAttention)
    #expect(gdn.layerIndex == 1)
    #expect(gdn.kind == .gdn)
}

private func makeDecoderLayer(
    architecture: QwenHybridArchitecture,
    inputNorm: [Float],
    postAttentionNorm: [Float],
    runtime: EdgeMetalRuntime
) throws -> QwenFullAttentionDecoderLayerReference {
    try QwenFullAttentionDecoderLayerReference(
        attention: makeDecoderAttention(architecture: architecture, runtime: runtime),
        mlp: makeDecoderMLP(runtime: runtime),
        inputLayerNorm: EdgeTensor(float32: inputNorm, shape: EdgeTensorShape([2]), runtime: runtime),
        postAttentionLayerNorm: EdgeTensor(
            float32: postAttentionNorm,
            shape: EdgeTensorShape([2]),
            runtime: runtime
        ),
        rmsNormEpsilon: 0
    )
}

private func makeDecoderAttention(
    architecture: QwenHybridArchitecture,
    runtime: EdgeMetalRuntime
) throws -> QwenFullAttentionReference {
    let projectionWeights = QwenAttentionProjectionWeights(
        layerIndex: 0,
        query: try EdgeTensor(
            float32: [
                1, 0, 3, 0,
                0, 1, 0, 2,
            ],
            shape: EdgeTensorShape([2, 4]),
            runtime: runtime
        ),
        key: try EdgeTensor(float32: [3, 4], shape: EdgeTensorShape([2, 1]), runtime: runtime),
        value: try EdgeTensor(float32: [-1, 2], shape: EdgeTensorShape([2, 1]), runtime: runtime)
    )
    return try QwenFullAttentionReference(
        architecture: architecture,
        layerIndex: 0,
        projectionWeights: projectionWeights
    )
}

private func makeDecoderMLP(runtime: EdgeMetalRuntime) throws -> QwenMLPWeights {
    QwenMLPWeights(
        layerIndex: 0,
        gate: try EdgeTensor(
            float32: [
                0, 0,
                -1, 0,
            ],
            shape: EdgeTensorShape([2, 2]),
            runtime: runtime
        ),
        up: try EdgeTensor(
            float32: [
                0, 0,
                -2, 0,
            ],
            shape: EdgeTensorShape([2, 2]),
            runtime: runtime
        ),
        down: try EdgeTensor(
            float32: [
                1, 0,
                0, 0,
            ],
            shape: EdgeTensorShape([2, 2]),
            runtime: runtime
        )
    )
}

private func makePositiveDecoderLayer(
    architecture: QwenHybridArchitecture,
    runtime: EdgeMetalRuntime
) throws -> QwenFullAttentionDecoderLayerReference {
    try QwenFullAttentionDecoderLayerReference(
        attention: makePositiveDecoderAttention(architecture: architecture, runtime: runtime),
        mlp: makePositiveDecoderMLP(runtime: runtime),
        inputLayerNorm: EdgeTensor(float32: [1, 1], shape: EdgeTensorShape([2]), runtime: runtime),
        postAttentionLayerNorm: EdgeTensor(float32: [1, 1], shape: EdgeTensorShape([2]), runtime: runtime),
        rmsNormEpsilon: 1e-6
    )
}

private func makePositiveQuantizedDecoderLayer(
    architecture: QwenHybridArchitecture,
    runtime: EdgeMetalRuntime
) throws -> QwenQuantizedFullAttentionDecoderLayerReference {
    try QwenQuantizedFullAttentionDecoderLayerReference(
        attention: makePositiveQuantizedAttention(architecture: architecture),
        mlp: makePositiveQuantizedMLP(),
        inputLayerNorm: EdgeTensor(float32: [1, 1], shape: EdgeTensorShape([2]), runtime: runtime),
        postAttentionLayerNorm: EdgeTensor(float32: [1, 1], shape: EdgeTensorShape([2]), runtime: runtime),
        rmsNormEpsilon: 1e-6
    )
}

private func makePositiveDecoderAttention(
    architecture: QwenHybridArchitecture,
    runtime: EdgeMetalRuntime
) throws -> QwenFullAttentionReference {
    let projectionWeights = QwenAttentionProjectionWeights(
        layerIndex: 0,
        query: try EdgeTensor(
            float32: [
                1, 0, 3, 0,
                0, 1, 0, 2,
            ],
            shape: EdgeTensorShape([2, 4]),
            runtime: runtime
        ),
        key: try EdgeTensor(float32: [1, 2], shape: EdgeTensorShape([2, 1]), runtime: runtime),
        value: try EdgeTensor(float32: [2, 1], shape: EdgeTensorShape([2, 1]), runtime: runtime),
        queryHiddenSize: architecture.queryHiddenSize,
        queryHeadCount: architecture.attentionHeadCount,
        queryHeadDimension: architecture.attentionHeadDimension
    )
    return try QwenFullAttentionReference(
        architecture: architecture,
        layerIndex: 0,
        projectionWeights: projectionWeights,
        outputProjectionWeights: QwenAttentionOutputProjectionWeights(
            layerIndex: 0,
            weight: try EdgeTensor(
                float32: [
                    1, 0,
                    0, 1,
                ],
                shape: EdgeTensorShape([2, 2]),
                runtime: runtime
            )
        )
    )
}

private func makePositiveQuantizedAttention(
    architecture: QwenHybridArchitecture
) throws -> QwenQuantizedFullAttentionReference {
    try QwenQuantizedFullAttentionReference(
        architecture: architecture,
        layerIndex: 0,
        projectionWeights: QwenQuantizedAttentionProjectionWeights(
            layerIndex: 0,
            query: exactDecoderQuantizedRows([
                [1, 0],
                [0, 1],
                [3, 0],
                [0, 2],
            ]),
            key: exactDecoderQuantizedRows([[1, 2]]),
            value: exactDecoderQuantizedRows([[2, 1]]),
            queryHiddenSize: architecture.queryHiddenSize,
            queryHeadCount: architecture.attentionHeadCount,
            queryHeadDimension: architecture.attentionHeadDimension
        ),
        outputProjectionWeights: QwenQuantizedAttentionOutputProjectionWeights(
            layerIndex: 0,
            weight: exactDecoderQuantizedRows([
                [1, 0],
                [0, 1],
            ])
        )
    )
}

private func makePositiveDecoderMLP(runtime: EdgeMetalRuntime) throws -> QwenMLPWeights {
    QwenMLPWeights(
        layerIndex: 0,
        gate: try EdgeTensor(
            float32: [
                1, 0,
                0, 1,
            ],
            shape: EdgeTensorShape([2, 2]),
            runtime: runtime
        ),
        up: try EdgeTensor(
            float32: [
                2, 0,
                0, 1,
            ],
            shape: EdgeTensorShape([2, 2]),
            runtime: runtime
        ),
        down: try EdgeTensor(
            float32: [
                1, 0,
                0, 1,
            ],
            shape: EdgeTensorShape([2, 2]),
            runtime: runtime
        )
    )
}

private func makePositiveQuantizedMLP() throws -> QwenQuantizedMLPWeights {
    try QwenQuantizedMLPWeights(
        layerIndex: 0,
        gate: exactDecoderQuantizedRows([
            [1, 0],
            [0, 1],
        ]),
        up: exactDecoderQuantizedRows([
            [2, 0],
            [0, 1],
        ]),
        down: exactDecoderQuantizedRows([
            [1, 0],
            [0, 1],
        ])
    )
}

private func makeDecoderLayerArchitecture() throws -> QwenHybridArchitecture {
    try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 128,
        hiddenSize: 2,
        intermediateSize: 2,
        attentionHeadCount: 2,
        keyValueHeadCount: 1,
        contextLength: 32,
        rmsNormEpsilon: 0,
        ropeTheta: 10_000,
        layerKinds: [.fullAttention, .gdn]
    )
}

private func makeHybridDecoderGDNLayer(runtime: EdgeMetalRuntime) throws -> QwenGDNDecoderLayerReference {
    let linearAttention = QwenGDNWeights(
        layerIndex: 1,
        inProjQKV: try EdgeTensor(
            float32: [
                1, 0, 1,
                0, 1, 0,
            ],
            shape: EdgeTensorShape([2, 3]),
            runtime: runtime
        ),
        inProjZ: try EdgeTensor(float32: [1, 0], shape: EdgeTensorShape([2, 1]), runtime: runtime),
        inProjB: try EdgeTensor(float32: [1, 0], shape: EdgeTensorShape([2, 1]), runtime: runtime),
        inProjA: try EdgeTensor(float32: [1, 0], shape: EdgeTensorShape([2, 1]), runtime: runtime),
        conv1D: try EdgeTensor(
            float32: Array(repeating: Float(1), count: 12),
            shape: EdgeTensorShape([3, 4, 1]),
            runtime: runtime
        ),
        convWeightLayout: .mlxSanitizedDepthwise,
        aLog: try EdgeTensor(float32: [0.25], shape: EdgeTensorShape([1]), runtime: runtime),
        dtBias: try EdgeTensor(float32: [1], shape: EdgeTensorShape([1]), runtime: runtime),
        norm: try EdgeTensor(float32: [1], shape: EdgeTensorShape([1]), runtime: runtime),
        outProj: try EdgeTensor(float32: [1, 1], shape: EdgeTensorShape([1, 2]), runtime: runtime),
        linearKeyHeadCount: 1,
        linearValueHeadCount: 1,
        linearKeyHeadDimension: 1,
        linearValueHeadDimension: 1,
        linearKeyHiddenSize: 1,
        linearValueHiddenSize: 1,
        rmsNormEpsilon: 1e-6
    )
    return try QwenGDNDecoderLayerReference(
        linearAttention: linearAttention,
        mlp: QwenMLPWeights(
            layerIndex: 1,
            gate: try EdgeTensor(float32: Array(repeating: 0, count: 4), shape: EdgeTensorShape([2, 2]), runtime: runtime),
            up: try EdgeTensor(float32: Array(repeating: 0, count: 4), shape: EdgeTensorShape([2, 2]), runtime: runtime),
            down: try EdgeTensor(float32: Array(repeating: 0, count: 4), shape: EdgeTensorShape([2, 2]), runtime: runtime)
        ),
        inputLayerNorm: try EdgeTensor(float32: [1, 1], shape: EdgeTensorShape([2]), runtime: runtime),
        postAttentionLayerNorm: try EdgeTensor(float32: [1, 1], shape: EdgeTensorShape([2]), runtime: runtime),
        rmsNormEpsilon: 1e-6
    )
}

private func makeHybridQuantizedDecoderGDNLayer(runtime: EdgeMetalRuntime) throws -> QwenQuantizedGDNDecoderLayerReference {
    let linearAttention = try QwenQuantizedGDNWeights(
        layerIndex: 1,
        inProjQKV: exactDecoderQuantizedRows([
            [1, 0],
            [0, 1],
            [1, 0],
        ]),
        inProjZ: exactDecoderQuantizedRows([[1, 0]]),
        inProjB: exactDecoderQuantizedRows([[1, 0]]),
        inProjA: exactDecoderQuantizedRows([[1, 0]]),
        conv1D: EdgeTensor(
            float32: Array(repeating: Float(1), count: 12),
            shape: EdgeTensorShape([3, 4, 1]),
            runtime: runtime
        ),
        convWeightLayout: .mlxSanitizedDepthwise,
        aLog: EdgeTensor(float32: [0.25], shape: EdgeTensorShape([1]), runtime: runtime),
        dtBias: EdgeTensor(float32: [1], shape: EdgeTensorShape([1]), runtime: runtime),
        norm: EdgeTensor(float32: [1], shape: EdgeTensorShape([1]), runtime: runtime),
        outProj: exactDecoderQuantizedRows([
            [1],
            [1],
        ]),
        linearKeyHeadCount: 1,
        linearValueHeadCount: 1,
        linearKeyHeadDimension: 1,
        linearValueHeadDimension: 1,
        linearKeyHiddenSize: 1,
        linearValueHiddenSize: 1,
        rmsNormEpsilon: 1e-6
    )
    return try QwenQuantizedGDNDecoderLayerReference(
        linearAttention: linearAttention,
        mlp: QwenQuantizedMLPWeights(
            layerIndex: 1,
            gate: exactDecoderQuantizedRows([[0, 0], [0, 0]]),
            up: exactDecoderQuantizedRows([[0, 0], [0, 0]]),
            down: exactDecoderQuantizedRows([[0, 0], [0, 0]])
        ),
        inputLayerNorm: EdgeTensor(float32: [1, 1], shape: EdgeTensorShape([2]), runtime: runtime),
        postAttentionLayerNorm: EdgeTensor(float32: [1, 1], shape: EdgeTensorShape([2]), runtime: runtime),
        rmsNormEpsilon: 1e-6
    )
}

private func completeDecoderLayerWeightMap() -> [String: String] {
    var weightMap: [String: String] = [:]
    for name in requiredDecoderModelLevel() + requiredDecoderFullAttentionLayer0() + requiredDecoderGDNLayer1() {
        weightMap[name] = "model-00001-of-00001.safetensors"
    }
    return weightMap
}

private func requiredDecoderModelLevel() -> [String] {
    [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.lm_head.weight",
    ]
}

private func requiredDecoderFullAttentionLayer0() -> [String] {
    let prefix = "model.layers.0"
    let attentionPrefix = "\(prefix).self_attn"
    return [
        "\(prefix).input_layernorm.weight",
        "\(prefix).post_attention_layernorm.weight",
        "\(prefix).mlp.gate_proj.weight",
        "\(prefix).mlp.up_proj.weight",
        "\(prefix).mlp.down_proj.weight",
        "\(attentionPrefix).q_proj.weight",
        "\(attentionPrefix).k_proj.weight",
        "\(attentionPrefix).v_proj.weight",
        "\(attentionPrefix).o_proj.weight",
        "\(attentionPrefix).q_norm.weight",
        "\(attentionPrefix).k_norm.weight",
    ]
}

private func requiredDecoderGDNLayer1() -> [String] {
    let prefix = "model.layers.1"
    let attentionPrefix = "\(prefix).linear_attn"
    return [
        "\(prefix).input_layernorm.weight",
        "\(prefix).post_attention_layernorm.weight",
        "\(prefix).mlp.gate_proj.weight",
        "\(prefix).mlp.up_proj.weight",
        "\(prefix).mlp.down_proj.weight",
        "\(attentionPrefix).A_log",
        "\(attentionPrefix).conv1d.weight",
        "\(attentionPrefix).dt_bias",
        "\(attentionPrefix).in_proj_a.weight",
        "\(attentionPrefix).in_proj_b.weight",
        "\(attentionPrefix).in_proj_qkv.weight",
        "\(attentionPrefix).in_proj_z.weight",
        "\(attentionPrefix).norm.weight",
        "\(attentionPrefix).out_proj.weight",
    ]
}

private func decoderFullAttentionLayer0TensorEntries() -> [String: (shape: [Int], values: [Float])] {
    [
        "model.layers.0.input_layernorm.weight": ([2], [1, 1]),
        "model.layers.0.post_attention_layernorm.weight": ([2], [1, 1]),
        "model.layers.0.self_attn.q_proj.weight": ([4, 2], [
            1, 0,
            0, 1,
            3, 0,
            0, 2,
        ]),
        "model.layers.0.self_attn.k_proj.weight": ([1, 2], [3, 4]),
        "model.layers.0.self_attn.v_proj.weight": ([1, 2], [-1, 2]),
        "model.layers.0.self_attn.o_proj.weight": ([2, 2], [
            2, 0,
            0, 2,
        ]),
        "model.layers.0.self_attn.q_norm.weight": ([1], [1]),
        "model.layers.0.self_attn.k_norm.weight": ([1], [1]),
        "model.layers.0.mlp.gate_proj.weight": ([2, 2], [
            0, -1,
            0, 0,
        ]),
        "model.layers.0.mlp.up_proj.weight": ([2, 2], [
            0, -2,
            0, 0,
        ]),
        "model.layers.0.mlp.down_proj.weight": ([2, 2], [
            1, 0,
            0, 0,
        ]),
    ]
}

private func decoderGDNLayer1TensorEntries() -> [String: (shape: [Int], values: [Float])] {
    [
        "model.layers.1.input_layernorm.weight": ([2], [1, 1]),
        "model.layers.1.post_attention_layernorm.weight": ([2], [1, 1]),
        "model.layers.1.mlp.gate_proj.weight": ([2, 2], Array(repeating: Float.zero, count: 4)),
        "model.layers.1.mlp.up_proj.weight": ([2, 2], Array(repeating: Float.zero, count: 4)),
        "model.layers.1.mlp.down_proj.weight": ([2, 2], Array(repeating: Float.zero, count: 4)),
        "model.layers.1.linear_attn.in_proj_qkv.weight": ([3, 2], [
            1, 0,
            0, 1,
            1, 0,
        ]),
        "model.layers.1.linear_attn.in_proj_z.weight": ([1, 2], [1, 0]),
        "model.layers.1.linear_attn.in_proj_b.weight": ([1, 2], [1, 0]),
        "model.layers.1.linear_attn.in_proj_a.weight": ([1, 2], [1, 0]),
        "model.layers.1.linear_attn.conv1d.weight": ([3, 4, 1], Array(repeating: Float(1), count: 12)),
        "model.layers.1.linear_attn.A_log": ([1], [0.25]),
        "model.layers.1.linear_attn.dt_bias": ([1], [1]),
        "model.layers.1.linear_attn.norm.weight": ([1], [1]),
        "model.layers.1.linear_attn.out_proj.weight": ([2, 1], [1, 1]),
    ]
}

private func writeDecoderLayerSafeTensorsShard(
    tensors: [String: (shape: [Int], values: [Float])],
    to url: URL
) throws {
    var payload = Data()
    var fields: [String] = []
    var offset = 0
    for name in tensors.keys.sorted() {
        let tensor = tensors[name]!
        let data = tensor.values.withUnsafeBufferPointer { pointer in
            Data(buffer: pointer)
        }
        let end = offset + data.count
        fields.append(
            """
            "\(name)": {
              "dtype": "F32",
              "shape": \(jsonIntArrayForDecoderLayer(tensor.shape)),
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

private func jsonIntArrayForDecoderLayer(_ values: [Int]) -> String {
    "[\(values.map(String.init).joined(separator: ","))]"
}

private func exactDecoderQuantizedRows(_ rows: [[UInt32]]) throws -> EdgeQuantizedTensor {
    let columns = rows.first?.count ?? 0
    return try EdgeQuantizedTensor(
        shape: [rows.count, columns],
        packedShape: [rows.count, (columns * 8 + 31) / 32],
        scaleShape: [rows.count, 1],
        groupSize: columns,
        bits: 8,
        packedValues: decoderPackQuantizedRows(rows, bits: 8),
        scales: Array(repeating: 1, count: rows.count),
        biases: Array(repeating: 0, count: rows.count)
    )
}

private func decoderPackQuantizedRows(_ rows: [[UInt32]], bits: Int) -> [UInt32] {
    rows.flatMap { decoderPackQuantizedWords($0, bits: bits) }
}

private func decoderPackQuantizedWords(_ values: [UInt32], bits: Int) -> [UInt32] {
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
