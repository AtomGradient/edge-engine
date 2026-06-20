// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Testing
@testable import EdgeEngine

@Test func customMetalMatmulMatchesCPUReference() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let lhsValues: [Float] = [
        1, 2, 3,
        4, 5, 6,
    ]
    let rhsValues: [Float] = [
        7, 8,
        9, 10,
        11, 12,
    ]
    let lhs = try EdgeTensor(float32: lhsValues, shape: EdgeTensorShape([2, 3]), runtime: runtime)
    let rhs = try EdgeTensor(float32: rhsValues, shape: EdgeTensorShape([3, 2]), runtime: runtime)

    let result = try executor.matmul(lhs, rhs)
    let expected = try CPUReferenceOps.matmul(lhsValues, rows: 2, inner: 3, rhsValues, columns: 2)

    #expect(result.shape == EdgeTensorShape([2, 2]))
    #expect(try result.readFloat32() == expected)
}

@Test func customMetalSplitColumnsMatchesCPULayout() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let input = try EdgeTensor(
        float32: [
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10,
        ],
        shape: EdgeTensorShape([2, 5]),
        runtime: runtime
    )

    let split = try executor.splitColumns(input, firstColumnCount: 3)

    #expect(split.first.shape == EdgeTensorShape([2, 3]))
    #expect(split.second.shape == EdgeTensorShape([2, 2]))
    #expect(try split.first.readFloat32() == [1, 2, 3, 6, 7, 8])
    #expect(try split.second.readFloat32() == [4, 5, 9, 10])
}

@Test func customMetalSplitGatedQueryMatchesMLXPerHeadSemantics() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let input = try EdgeTensor(
        float32: [
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
        ],
        shape: EdgeTensorShape([2, 8]),
        runtime: runtime
    )

    let split = try executor.splitGatedQuery(input, headCount: 2, headDimension: 2)

    #expect(split.query.shape == EdgeTensorShape([2, 4]))
    #expect(split.gate.shape == EdgeTensorShape([2, 4]))
    #expect(try split.query.readFloat32() == [1, 2, 5, 6, 9, 10, 13, 14])
    #expect(try split.gate.readFloat32() == [3, 4, 7, 8, 11, 12, 15, 16])
}

@Test func customMetalGDNDepthwiseConv1DMatchesCPUReferenceAndReturnsNextState() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let inputValues: [Float] = [
        5, 6,
        7, 8,
    ]
    let convStateValues: [Float] = [
        1, 2,
        3, 4,
    ]
    let weightValues: [Float] = [
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6,
    ]
    let input = try EdgeTensor(
        float32: inputValues,
        shape: EdgeTensorShape([2, 2]),
        runtime: runtime
    )
    let convState = try EdgeTensor(
        float32: convStateValues,
        shape: EdgeTensorShape([2, 2]),
        runtime: runtime
    )
    let weights = try EdgeTensor(
        float32: weightValues,
        shape: EdgeTensorShape([2, 3, 1]),
        runtime: runtime
    )

    let result = try executor.gdnDepthwiseConv1D(
        input: input,
        weights: weights,
        convState: convState
    )
    let expected = try CPUReferenceOps.gdnDepthwiseConv1D(
        inputValues,
        convState: convStateValues,
        weights: weightValues,
        tokenCount: 2,
        channelCount: 2,
        kernelSize: 3
    )
    let outputError = try NumericComparison.maxAbsoluteError(
        try result.activated.readFloat32(),
        expected.activated
    )

    #expect(result.activated.shape == EdgeTensorShape([2, 2]))
    #expect(result.nextConvState.shape == EdgeTensorShape([2, 2]))
    #expect(outputError < 1e-5)
    #expect(try result.nextConvState.readFloat32() == expected.nextConvState)
    #expect(executor.lastExecutionStats?.operationName == "edge_gdn_depthwise_conv1d")
}

@Test func customMetalGDNNormalizeQKMatchesCPUReference() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let queryValues: [Float] = [
        3, 4, 1, 2,
        0, 5, -2, 6,
    ]
    let keyValues: [Float] = [
        6, 8, -3, 4,
        5, 12, 7, 24,
    ]
    let query = try EdgeTensor(float32: queryValues, shape: EdgeTensorShape([2, 4]), runtime: runtime)
    let key = try EdgeTensor(float32: keyValues, shape: EdgeTensorShape([2, 4]), runtime: runtime)

    let result = try executor.gdnNormalizeQK(
        query: query,
        key: key,
        headCount: 2,
        headDimension: 2
    )
    let expected = try CPUReferenceOps.gdnNormalizeQK(
        query: queryValues,
        key: keyValues,
        tokenCount: 2,
        headCount: 2,
        headDimension: 2
    )
    let queryError = try NumericComparison.maxAbsoluteError(
        try result.query.readFloat32(),
        expected.query
    )
    let keyError = try NumericComparison.maxAbsoluteError(
        try result.key.readFloat32(),
        expected.key
    )

    #expect(result.query.shape == EdgeTensorShape([2, 4]))
    #expect(result.key.shape == EdgeTensorShape([2, 4]))
    #expect(queryError < 1e-6)
    #expect(keyError < 1e-6)
    #expect(executor.lastExecutionStats?.operationName == "edge_gdn_normalize_qk")
}

@Test func customMetalGDNRecurrentUpdateMatchesCPUReference() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let queryValues: [Float] = [
        1, 0,
        1, 0,
    ]
    let keyValues: [Float] = [
        2, 3,
        2, 3,
    ]
    let valueValues: [Float] = [
        4, 5,
        4, 5,
    ]
    let aValues: [Float] = [0, 0]
    let bValues: [Float] = [0, 0]
    let query = try EdgeTensor(float32: queryValues, shape: EdgeTensorShape([2, 2]), runtime: runtime)
    let key = try EdgeTensor(float32: keyValues, shape: EdgeTensorShape([2, 2]), runtime: runtime)
    let value = try EdgeTensor(float32: valueValues, shape: EdgeTensorShape([2, 2]), runtime: runtime)
    let a = try EdgeTensor(float32: aValues, shape: EdgeTensorShape([2, 1]), runtime: runtime)
    let b = try EdgeTensor(float32: bValues, shape: EdgeTensorShape([2, 1]), runtime: runtime)
    let aLog = try EdgeTensor(float32: [0], shape: EdgeTensorShape([1]), runtime: runtime)
    let dtBias = try EdgeTensor(float32: [0], shape: EdgeTensorShape([1]), runtime: runtime)
    let state = try EdgeTensor(float32: [0, 0, 0, 0], shape: EdgeTensorShape([1, 2, 2]), runtime: runtime)

    let result = try executor.gdnRecurrentUpdate(
        query: query,
        key: key,
        value: value,
        a: a,
        b: b,
        aLog: aLog,
        dtBias: dtBias,
        state: state,
        keyHeadCount: 1,
        valueHeadCount: 1,
        keyHeadDimension: 2,
        valueHeadDimension: 2
    )
    let expected = try QwenGDNReference.gatedDeltaUpdate(
        query: queryValues,
        key: keyValues,
        value: valueValues,
        a: aValues,
        b: bValues,
        aLog: [0],
        dtBias: [0],
        batchSize: 1,
        tokenCount: 2,
        keyHeadCount: 1,
        valueHeadCount: 1,
        keyHeadDimension: 2,
        valueHeadDimension: 2
    )
    let outputError = try NumericComparison.maxAbsoluteError(
        try result.output.readFloat32(),
        expected.output
    )
    let stateError = try NumericComparison.maxAbsoluteError(
        try result.nextState.readFloat32(),
        expected.updatedState
    )

    #expect(result.output.shape == EdgeTensorShape([2, 2]))
    #expect(result.nextState.shape == EdgeTensorShape([1, 2, 2]))
    #expect(outputError < 1e-5)
    #expect(stateError < 1e-5)
    #expect(executor.lastExecutionStats?.operationName == "edge_gdn_recurrent_update")
}

@Test func customMetalGDNRecurrentUpdateRepeatsKeyHeadsAcrossValueHeads() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let queryValues: [Float] = [1, 0]
    let keyValues: [Float] = [2, 3]
    let valueValues: [Float] = [4, 5]
    let aValues: [Float] = [0, 0]
    let bValues: [Float] = [0, 0]
    let query = try EdgeTensor(float32: queryValues, shape: EdgeTensorShape([1, 2]), runtime: runtime)
    let key = try EdgeTensor(float32: keyValues, shape: EdgeTensorShape([1, 2]), runtime: runtime)
    let value = try EdgeTensor(float32: valueValues, shape: EdgeTensorShape([1, 2]), runtime: runtime)
    let a = try EdgeTensor(float32: aValues, shape: EdgeTensorShape([1, 2]), runtime: runtime)
    let b = try EdgeTensor(float32: bValues, shape: EdgeTensorShape([1, 2]), runtime: runtime)
    let aLog = try EdgeTensor(float32: [0, 0], shape: EdgeTensorShape([2]), runtime: runtime)
    let dtBias = try EdgeTensor(float32: [0, 0], shape: EdgeTensorShape([2]), runtime: runtime)
    let state = try EdgeTensor(
        float32: [0, 0, 0, 0],
        shape: EdgeTensorShape([2, 1, 2]),
        runtime: runtime
    )

    let result = try executor.gdnRecurrentUpdate(
        query: query,
        key: key,
        value: value,
        a: a,
        b: b,
        aLog: aLog,
        dtBias: dtBias,
        state: state,
        keyHeadCount: 1,
        valueHeadCount: 2,
        keyHeadDimension: 2,
        valueHeadDimension: 1
    )
    let expected = try QwenGDNReference.gatedDeltaUpdate(
        query: queryValues,
        key: keyValues,
        value: valueValues,
        a: aValues,
        b: bValues,
        aLog: [0, 0],
        dtBias: [0, 0],
        batchSize: 1,
        tokenCount: 1,
        keyHeadCount: 1,
        valueHeadCount: 2,
        keyHeadDimension: 2,
        valueHeadDimension: 1
    )

    #expect(try NumericComparison.maxAbsoluteError(try result.output.readFloat32(), expected.output) < 1e-5)
    #expect(try NumericComparison.maxAbsoluteError(try result.nextState.readFloat32(), expected.updatedState) < 1e-5)
}

@Test func customMetalGDNSingleTokenFusedUpdateMatchesUnfusedReference() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let mixedValues: [Float] = [1.0, -0.5, 0.25, 0.75, -1.25]
    let convStateValues: [Float] = [
        0.1, 0.2, -0.1, -0.2, 0.3,
        -0.4, 0.5, 0.6, -0.7, 0.8,
    ]
    let weightValues: [Float] = [
        0.2, 0.1, 0.4,
        -0.3, 0.2, 0.1,
        0.5, -0.2, 0.3,
        0.1, 0.7, -0.4,
        -0.6, 0.2, 0.5,
    ]
    let aValues: [Float] = [0.125]
    let bValues: [Float] = [-0.25]
    let aLogValues: [Float] = [0.2]
    let dtBiasValues: [Float] = [0.75]
    let initialState: [Float] = [0.4, -0.2]

    let mixedQKV = try EdgeTensor(float32: mixedValues, shape: EdgeTensorShape([1, 5]), runtime: runtime)
    let weights = try EdgeTensor(float32: weightValues, shape: EdgeTensorShape([5, 3, 1]), runtime: runtime)
    let convState = try EdgeTensor(
        float32: convStateValues,
        shape: EdgeTensorShape([2, 5]),
        runtime: runtime
    )
    let a = try EdgeTensor(float32: aValues, shape: EdgeTensorShape([1, 1]), runtime: runtime)
    let b = try EdgeTensor(float32: bValues, shape: EdgeTensorShape([1, 1]), runtime: runtime)
    let aLog = try EdgeTensor(float32: aLogValues, shape: EdgeTensorShape([1]), runtime: runtime)
    let dtBias = try EdgeTensor(float32: dtBiasValues, shape: EdgeTensorShape([1]), runtime: runtime)
    let state = try EdgeTensor(float32: initialState, shape: EdgeTensorShape([1, 1, 2]), runtime: runtime)

    let result = try executor.gdnSingleTokenFusedUpdate(
        mixedQKV: mixedQKV,
        weights: weights,
        convState: convState,
        a: a,
        b: b,
        aLog: aLog,
        dtBias: dtBias,
        recurrentState: state,
        keyHeadCount: 1,
        valueHeadCount: 1,
        keyHeadDimension: 2,
        valueHeadDimension: 1
    )
    let convolved = try CPUReferenceOps.gdnDepthwiseConv1D(
        mixedValues,
        convState: convStateValues,
        weights: weightValues,
        tokenCount: 1,
        channelCount: 5,
        kernelSize: 3
    )
    let normalized = try CPUReferenceOps.gdnNormalizeQK(
        query: Array(convolved.activated[0..<2]),
        key: Array(convolved.activated[2..<4]),
        tokenCount: 1,
        headCount: 1,
        headDimension: 2
    )
    let expected = try QwenGDNReference.gatedDeltaUpdate(
        query: normalized.query,
        key: normalized.key,
        value: Array(convolved.activated[4..<5]),
        a: aValues,
        b: bValues,
        aLog: aLogValues,
        dtBias: dtBiasValues,
        initialState: initialState,
        batchSize: 1,
        tokenCount: 1,
        keyHeadCount: 1,
        valueHeadCount: 1,
        keyHeadDimension: 2,
        valueHeadDimension: 1
    )

    #expect(result.output.shape == EdgeTensorShape([1, 1]))
    #expect(result.nextConvState.shape == EdgeTensorShape([2, 5]))
    #expect(result.nextState.shape == EdgeTensorShape([1, 1, 2]))
    #expect(try NumericComparison.maxAbsoluteError(try result.output.readFloat32(), expected.output) < 1e-5)
    #expect(try NumericComparison.maxAbsoluteError(try result.nextState.readFloat32(), expected.updatedState) < 1e-5)
    #expect(try result.nextConvState.readFloat32() == convolved.nextConvState)
    #expect(executor.lastExecutionStats?.operationName == "edge_gdn_single_token_fused_update")
}

@Test func customMetalRoPEMatchesCPUReference() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let values: [Float] = [
        1, 2, 3, 4,
        5, 6, 7, 8,
    ]
    let input = try EdgeTensor(float32: values, shape: EdgeTensorShape([2, 4]), runtime: runtime)

    let result = try executor.applyRoPE(
        input,
        headCount: 1,
        headDimension: 4,
        rotaryDimension: 4,
        base: 10_000,
        offset: 1
    )
    let expected = try CPUReferenceOps.rotaryEmbedding(
        values,
        tokenCount: 2,
        headCount: 1,
        headDimension: 4,
        rotaryDimension: 4,
        base: 10_000,
        offset: 1
    )
    let error = try NumericComparison.maxAbsoluteError(try result.readFloat32(), expected)

    #expect(result.shape == EdgeTensorShape([2, 4]))
    #expect(error < 1e-5)
}

@Test func customMetalRMSNormByHeadMatchesCPUReference() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let values: [Float] = [
        3, 4, 0, 5,
        6, 8, -5, 0,
    ]
    let input = try EdgeTensor(float32: values, shape: EdgeTensorShape([2, 4]), runtime: runtime)
    let weight = try EdgeTensor(float32: [1, 2], shape: EdgeTensorShape([2]), runtime: runtime)

    let result = try executor.rmsNormByHead(
        input,
        weight: weight,
        headCount: 2,
        headDimension: 2,
        epsilon: 0
    )
    let expected = try CPUReferenceOps.rmsNormByHead(
        values,
        weight: [1, 2],
        tokenCount: 2,
        headCount: 2,
        headDimension: 2,
        epsilon: 0
    )
    let error = try NumericComparison.maxAbsoluteError(try result.readFloat32(), expected)

    #expect(result.shape == EdgeTensorShape([2, 4]))
    #expect(error < 1e-5)
}

@Test func customMetalRMSNormMatchesCPUReference() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let values: [Float] = [
        3, 4,
        6, 8,
    ]
    let input = try EdgeTensor(float32: values, shape: EdgeTensorShape([2, 2]), runtime: runtime)
    let weight = try EdgeTensor(float32: [1, 2], shape: EdgeTensorShape([2]), runtime: runtime)

    let result = try executor.rmsNorm(input, weight: weight, epsilon: 0)
    let expected = try CPUReferenceOps.rmsNorm([3, 4], weight: [1, 2], epsilon: 0)
        + CPUReferenceOps.rmsNorm([6, 8], weight: [1, 2], epsilon: 0)
    let error = try NumericComparison.maxAbsoluteError(try result.readFloat32(), expected)

    #expect(result.shape == EdgeTensorShape([2, 2]))
    #expect(error < 1e-5)
    #expect(executor.lastExecutionStats?.operationName == "edge_rms_norm")
}

@Test func customMetalSigmoidMultiplyMatchesCPUReference() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let values: [Float] = [1, -2, 3, 4]
    let gateValues: [Float] = [0, 2, -4, -1]
    let input = try EdgeTensor(float32: values, shape: EdgeTensorShape([2, 2]), runtime: runtime)
    let gate = try EdgeTensor(float32: gateValues, shape: EdgeTensorShape([2, 2]), runtime: runtime)

    let result = try executor.sigmoidMultiply(input, gate: gate)
    let expected = try CPUReferenceOps.sigmoidMultiply(values, gate: gateValues)
    let error = try NumericComparison.maxAbsoluteError(try result.readFloat32(), expected)

    #expect(result.shape == EdgeTensorShape([2, 2]))
    #expect(error < 1e-6)
    #expect(executor.lastExecutionStats?.operationName == "edge_sigmoid_multiply")
}

@Test func customMetalSiluMultiplyMatchesCPUReference() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let gateValues: [Float] = [0, 1, -2, 3]
    let upValues: [Float] = [2, 3, -4, 5]
    let gate = try EdgeTensor(float32: gateValues, shape: EdgeTensorShape([2, 2]), runtime: runtime)
    let up = try EdgeTensor(float32: upValues, shape: EdgeTensorShape([2, 2]), runtime: runtime)

    let result = try executor.siluMultiply(gate: gate, up: up)
    let expected = try CPUReferenceOps.swiglu(gate: gateValues, up: upValues)
    let error = try NumericComparison.maxAbsoluteError(try result.readFloat32(), expected)

    #expect(result.shape == EdgeTensorShape([2, 2]))
    #expect(error < 1e-6)
    #expect(executor.lastExecutionStats?.operationName == "edge_silu_multiply")
}

@Test func customMetalAddMatchesCPUReference() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let lhsValues: [Float] = [1, -2, 3, 4]
    let rhsValues: [Float] = [4, 5, -6, 0.5]
    let lhs = try EdgeTensor(float32: lhsValues, shape: EdgeTensorShape([2, 2]), runtime: runtime)
    let rhs = try EdgeTensor(float32: rhsValues, shape: EdgeTensorShape([2, 2]), runtime: runtime)

    let result = try executor.add(lhs, rhs)
    let expected = try CPUReferenceOps.add(lhsValues, rhsValues)
    let error = try NumericComparison.maxAbsoluteError(try result.readFloat32(), expected)

    #expect(result.shape == EdgeTensorShape([2, 2]))
    #expect(error < 1e-6)
    #expect(executor.lastExecutionStats?.operationName == "edge_add")
}

@Test func customMetalEmbeddingLookupMatchesCPUReference() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let embeddingValues: [Float] = [
        1, 2,
        3, 4,
        5, 6,
    ]
    let embeddings = try EdgeTensor(
        float32: embeddingValues,
        shape: EdgeTensorShape([3, 2]),
        runtime: runtime
    )

    let result = try executor.embeddingLookup(tokenIds: [2, 0, 2], embeddings: embeddings)
    let expected = try CPUReferenceOps.embeddingLookup(
        tokenIds: [2, 0, 2],
        embeddings: embeddingValues,
        vocabularySize: 3,
        hiddenSize: 2
    )
    let error = try NumericComparison.maxAbsoluteError(try result.readFloat32(), expected)

    #expect(result.shape == EdgeTensorShape([3, 2]))
    #expect(error < 1e-6)
    #expect(executor.lastExecutionStats?.operationName == "edge_embedding_lookup")
}

@Test func customMetalEmbeddingLookupRejectsOutOfRangeToken() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let embeddings = try EdgeTensor(
        float32: [
            1, 2,
            3, 4,
        ],
        shape: EdgeTensorShape([2, 2]),
        runtime: runtime
    )

    var rejected = false
    do {
        _ = try executor.embeddingLookup(tokenIds: [0, 2], embeddings: embeddings)
    } catch MetalKernelExecutorError.invalidEmbeddingLookup(tokenIds: [0, 2], embeddings: [2, 2]) {
        rejected = true
    }

    #expect(rejected)
}

@Test func customMetalAffineQuantizedEmbeddingLookupMatchesDequantizedRows() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let rows: [[UInt32]] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ]
    let embeddings = try EdgeQuantizedTensor(
        shape: [3, 4],
        packedShape: [3, 1],
        scaleShape: [3, 2],
        groupSize: 2,
        bits: 4,
        packedValues: packEmbeddingQuantizedRows(rows, bits: 4),
        scales: [
            1, 10,
            2, 20,
            3, 30,
        ],
        biases: [
            0, 100,
            1, 200,
            2, 300,
        ]
    )

    let result = try executor.affineQuantizedEmbeddingLookup(
        tokenIds: [2, 0],
        embeddings: embeddings
    )
    let error = try NumericComparison.maxAbsoluteError(
        try result.readFloat32(),
        [
            29, 32, 630, 660,
            1, 2, 130, 140,
        ]
    )

    #expect(result.shape == EdgeTensorShape([2, 4]))
    #expect(error < 1e-6)
    #expect(executor.lastExecutionStats?.operationName == "edge_affine_quantized_embedding_lookup")
}

@Test func customMetalAffineQuantizedEmbeddingLookupRejectsOutOfRangeToken() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let embeddings = try EdgeQuantizedTensor(
        shape: [2, 4],
        packedShape: [2, 1],
        scaleShape: [2, 2],
        groupSize: 2,
        bits: 4,
        packedValues: packEmbeddingQuantizedRows(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ],
            bits: 4
        ),
        scales: Array(repeating: 1, count: 4)
    )

    var rejected = false
    do {
        _ = try executor.affineQuantizedEmbeddingLookup(tokenIds: [0, 2], embeddings: embeddings)
    } catch MetalKernelExecutorError.invalidEmbeddingLookup(tokenIds: [0, 2], embeddings: [2, 4]) {
        rejected = true
    }

    #expect(rejected)
}

@Test func customMetalScaledDotProductAttentionMatchesCPUReference() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let queryValues: [Float] = [
        2, 1,
        0, 1,
    ]
    let keyValues: [Float] = [
        1,
        2,
    ]
    let valueValues: [Float] = [
        10,
        20,
    ]
    let query = try EdgeTensor(float32: queryValues, shape: EdgeTensorShape([2, 2]), runtime: runtime)
    let key = try EdgeTensor(float32: keyValues, shape: EdgeTensorShape([2, 1]), runtime: runtime)
    let value = try EdgeTensor(float32: valueValues, shape: EdgeTensorShape([2, 1]), runtime: runtime)

    let result = try executor.scaledDotProductAttention(
        query: query,
        key: key,
        value: value,
        queryHeadCount: 2,
        keyValueHeadCount: 1,
        headDimension: 1
    )
    let expected = try CPUReferenceOps.scaledDotProductAttention(
        query: queryValues,
        key: keyValues,
        value: valueValues,
        tokenCount: 2,
        queryHeadCount: 2,
        keyValueHeadCount: 1,
        headDimension: 1
    )
    let error = try NumericComparison.maxAbsoluteError(try result.readFloat32(), expected)

    #expect(result.shape == EdgeTensorShape([2, 2]))
    #expect(error < 1e-5)
    #expect(executor.lastExecutionStats?.operationName == "edge_scaled_dot_product_attention")
}

@Test func customMetalScaledDotProductAttentionSupportsCachedKeyValues() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let queryValues: [Float] = [1, 1]
    let keyValues: [Float] = [1, 2, 3, 99]
    let valueValues: [Float] = [10, 20, 30, 999]
    let query = try EdgeTensor(float32: queryValues, shape: EdgeTensorShape([1, 2]), runtime: runtime)
    let key = try EdgeTensor(float32: keyValues, shape: EdgeTensorShape([4, 1]), runtime: runtime)
    let value = try EdgeTensor(float32: valueValues, shape: EdgeTensorShape([4, 1]), runtime: runtime)

    let result = try executor.scaledDotProductAttention(
        query: query,
        key: key,
        value: value,
        keyValueTokenCount: 3,
        queryPositionOffset: 2,
        queryHeadCount: 2,
        keyValueHeadCount: 1,
        headDimension: 1
    )
    let expected = try CPUReferenceOps.scaledDotProductAttention(
        query: queryValues,
        key: [1, 2, 3],
        value: [10, 20, 30],
        queryTokenCount: 1,
        keyValueTokenCount: 3,
        queryPositionOffset: 2,
        queryHeadCount: 2,
        keyValueHeadCount: 1,
        headDimension: 1
    )
    let error = try NumericComparison.maxAbsoluteError(try result.readFloat32(), expected)

    #expect(result.shape == EdgeTensorShape([1, 2]))
    #expect(error < 1e-5)
}

@Test func customMetalCopyRowsToPrefixCopiesIntoDestinationWindow() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let source = try EdgeTensor(
        float32: [
            1, 2,
            3, 4,
        ],
        shape: EdgeTensorShape([2, 2]),
        runtime: runtime
    )
    let destination = try EdgeTensor(
        float32: [
            0, 0,
            10, 10,
            20, 20,
            30, 30,
        ],
        shape: EdgeTensorShape([4, 2]),
        runtime: runtime
    )

    try executor.copyRowsToPrefix(source: source, destination: destination, startRow: 1)

    #expect(try destination.readFloat32() == [
        0, 0,
        1, 2,
        3, 4,
        30, 30,
    ])
    #expect(executor.lastExecutionStats?.operationName == "edge_copy_rows_to_prefix")
}

@Test func customMetalGatherRowsCopiesSelectedRows() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let source = try EdgeTensor(
        float32: [
            1, 2,
            3, 4,
            5, 6,
            7, 8,
        ],
        shape: EdgeTensorShape([4, 2]),
        runtime: runtime
    )

    let gathered = try executor.gatherRows(source: source, rowIndices: [0, 2, 3])

    #expect(gathered.shape == EdgeTensorShape([3, 2]))
    #expect(try gathered.readFloat32() == [
        1, 2,
        5, 6,
        7, 8,
    ])
    #expect(executor.lastExecutionStats?.operationName == "edge_gather_rows")
}

@Test func customMetalUpdateAttentionScoreEMAUpdatesPerKVHeadScores() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let query = try EdgeTensor(
        float32: [
            1, 0,
            0, 2,
        ],
        shape: EdgeTensorShape([1, 4]),
        runtime: runtime
    )
    let key = try EdgeTensor(
        float32: [
            3, 0,
            0, 4,
        ],
        shape: EdgeTensorShape([2, 2]),
        runtime: runtime
    )
    let scores = try EdgeTensor(
        float32: [0, 0],
        shape: EdgeTensorShape([2, 1]),
        runtime: runtime
    )

    try executor.updateAttentionScoreEMA(
        query: query,
        key: key,
        scores: scores,
        keyValueTokenCount: 2,
        queryHeadCount: 2,
        keyValueHeadCount: 1,
        headDimension: 2,
        scale: 1,
        hasExistingScores: false
    )

    #expect(try scores.readFloat32() == [1.5, 4.0])
    #expect(executor.lastExecutionStats?.operationName == "edge_update_attention_score_ema")
}

@Test func customMetalArgmaxLastLogitRowRunsOnGPU() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let logits = try EdgeTensor(
        float32: [
            0.1, 9.0, 3.0, 4.0,
            1.0, 2.0, 8.5, 8.5,
            -1.0, 12.0, 11.5, 0.0,
        ],
        shape: EdgeTensorShape([3, 4]),
        runtime: runtime
    )

    let token = try executor.argmaxLastRow(logits)

    #expect(token.tokenId == 1)
    #expect(token.logit == 12.0)
    #expect(executor.lastExecutionStats?.operationName == "edge_argmax_last_row")

    let scratch = try executor.makeArgmaxLastRowScratch()
    let nextLogits = try EdgeTensor(
        float32: [
            4.0, 3.0, 2.0, 1.0,
            -5.0, 7.0, 1.5, 9.25,
        ],
        shape: EdgeTensorShape([2, 4]),
        runtime: runtime
    )
    let nextToken = try executor.argmaxLastRow(nextLogits, scratch: scratch)
    #expect(nextToken.tokenId == 3)
    #expect(nextToken.logit == 9.25)
}

@Test func metalKernelExecutorRecordsOperationBudgetPressure() throws {
    let runtime = try EdgeMetalRuntime(
        configuration: .init(maxOpsPerCommandBuffer: 2, maxMBPerCommandBuffer: 40)
    )
    let executor = try MetalKernelExecutor(runtime: runtime)
    let lhs = try EdgeTensor(float32: [1, 2], shape: EdgeTensorShape([1, 2]), runtime: runtime)
    let rhs = try EdgeTensor(float32: [3, 4], shape: EdgeTensorShape([2, 1]), runtime: runtime)

    _ = try executor.matmul(lhs, rhs)
    #expect(executor.lastExecutionStats?.precommittedBeforeEncoding == false)
    #expect(executor.lastExecutionStats?.pendingOpsAfterRecord == 1)

    _ = try executor.matmul(lhs, rhs)
    #expect(executor.lastExecutionStats?.precommittedBeforeEncoding == false)
    #expect(executor.lastExecutionStats?.pendingOpsAfterRecord == 2)

    _ = try executor.matmul(lhs, rhs)
    #expect(executor.lastExecutionStats?.precommittedBeforeEncoding == true)
    #expect(executor.lastExecutionStats?.logicalCommitCount == 1)
    #expect(executor.schedulingSnapshot.pendingOps == 1)
}

@Test func metalKernelExecutorRecordsCommandBufferAcquisitionStats() throws {
    let runtime = try EdgeMetalRuntime(
        configuration: .init(
            maxOpsPerCommandBuffer: 2,
            maxMBPerCommandBuffer: 40,
            commandBufferBatchingEnabled: true,
            maxInFlightCommandBuffers: 4
        )
    )
    let executor = try MetalKernelExecutor(runtime: runtime)
    let lhs = try EdgeTensor(float32: [1, 2], shape: EdgeTensorShape([1, 2]), runtime: runtime)
    let rhs = try EdgeTensor(float32: [3, 4], shape: EdgeTensorShape([2, 1]), runtime: runtime)

    _ = try executor.matmul(lhs, rhs)
    var stats = try #require(runtime.lastCommandBufferAcquisitionStats)
    #expect(stats.waitedForBackpressure == false)
    #expect(stats.maxInFlightCommandBuffers == 4)
    #expect(stats.activeOperationCountAfterRecord == 1)
    #expect(stats.submittedBeforeWait == 0)

    _ = try executor.matmul(lhs, rhs)
    _ = try executor.matmul(lhs, rhs)
    stats = try #require(runtime.lastCommandBufferAcquisitionStats)
    #expect(stats.committedActiveBuffer == true)
    #expect(stats.submittedBeforeWait == 1)
    #expect(stats.activeOperationCountAfterRecord == 1)
}

@Test func metalKernelExecutorCanRecordUnboundedCommandBufferBatch() throws {
    let runtime = try EdgeMetalRuntime(
        configuration: .init(
            maxOpsPerCommandBuffer: 1,
            maxMBPerCommandBuffer: 1,
            commandBufferBatchingEnabled: true,
            maxInFlightCommandBuffers: 1
        )
    )
    let executor = try MetalKernelExecutor(runtime: runtime)
    let lhs = try EdgeTensor(float32: [1, 2], shape: EdgeTensorShape([1, 2]), runtime: runtime)
    let rhs = try EdgeTensor(float32: [3, 4], shape: EdgeTensorShape([2, 1]), runtime: runtime)

    try executor.withUnboundedCommandBufferBatch {
        _ = try executor.matmul(lhs, rhs)
        _ = try executor.matmul(lhs, rhs)
        let stats = try #require(runtime.lastCommandBufferAcquisitionStats)
        #expect(stats.committedActiveBuffer == false)
        #expect(stats.waitedForBackpressure == false)
        #expect(stats.activeOperationCountAfterRecord == 2)
    }
}

@Test func metalKernelExecutorKeepsNestedUnboundedCommandBufferBatchOpen() throws {
    let runtime = try EdgeMetalRuntime(
        configuration: .init(
            maxOpsPerCommandBuffer: 1,
            maxMBPerCommandBuffer: 1,
            commandBufferBatchingEnabled: true,
            maxInFlightCommandBuffers: 1
        )
    )
    let executor = try MetalKernelExecutor(runtime: runtime)
    let lhs = try EdgeTensor(float32: [1, 2], shape: EdgeTensorShape([1, 2]), runtime: runtime)
    let rhs = try EdgeTensor(float32: [3, 4], shape: EdgeTensorShape([2, 1]), runtime: runtime)

    try executor.withUnboundedCommandBufferBatch {
        _ = try executor.matmul(lhs, rhs)
        try executor.withUnboundedCommandBufferBatch {
            _ = try executor.matmul(lhs, rhs)
        }
        let stats = try #require(runtime.lastCommandBufferAcquisitionStats)
        #expect(stats.committedActiveBuffer == false)
        #expect(stats.waitedForBackpressure == false)
        #expect(stats.activeOperationCountAfterRecord == 2)
    }
}

@Test func metalKernelExecutorUsesRuntimeDynamicScheduleForStats() throws {
    let runtime = try EdgeMetalRuntime(
        configuration: .init(
            maxOpsPerCommandBuffer: 10,
            maxMBPerCommandBuffer: 40,
            contextLengthHint: 12_288,
            dynamicOpsSchedule: .init(floor: 1, contextLow: 4_096, contextHigh: 12_288)
        )
    )
    let executor = try MetalKernelExecutor(runtime: runtime)
    let lhs = try EdgeTensor(float32: [1, 2], shape: EdgeTensorShape([1, 2]), runtime: runtime)
    let rhs = try EdgeTensor(float32: [3, 4], shape: EdgeTensorShape([2, 1]), runtime: runtime)

    _ = try executor.matmul(lhs, rhs)
    #expect(executor.lastExecutionStats?.effectiveMaxOpsPerCommandBuffer == 1)

    _ = try executor.matmul(lhs, rhs)
    #expect(executor.lastExecutionStats?.precommittedBeforeEncoding == true)
    #expect(executor.lastExecutionStats?.logicalCommitCount == 1)
}

private func packEmbeddingQuantizedRows(_ rows: [[UInt32]], bits: Int) -> [UInt32] {
    rows.flatMap { packEmbeddingQuantizedWords($0, bits: bits) }
}

private func packEmbeddingQuantizedWords(_ values: [UInt32], bits: Int) -> [UInt32] {
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
