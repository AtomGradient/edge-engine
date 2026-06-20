// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Darwin
import Metal

public enum MetalKernelExecutorError: Error, Equatable {
    case libraryCompilationFailed
    case missingFunction(String)
    case pipelineCreationFailed(String)
    case commandBufferCreationFailed
    case commandEncoderCreationFailed
    case invalidMatrixRank
    case matrixDimensionMismatch
    case invalidColumnSplit(firstColumnCount: Int, columns: Int)
    case invalidGatedQuerySplit(headCount: Int, headDimension: Int, columns: Int)
    case invalidRoPEConfiguration(rotaryDimension: Int, headDimension: Int, base: Float)
    case invalidRMSNorm(columns: Int, weightCount: Int)
    case invalidRMSNormByHead(headCount: Int, headDimension: Int, columns: Int, weightCount: Int)
    case elementwiseShapeMismatch(lhs: [Int], rhs: [Int])
    case invalidEmbeddingLookup(tokenIds: [Int], embeddings: [Int])
    case invalidGDNDepthwiseConvShape(input: [Int], weights: [Int], convState: [Int])
    case invalidGDNQKNormalizationShape(query: [Int], key: [Int], headCount: Int, headDimension: Int)
    case invalidGDNRecurrentShape(
        query: [Int],
        key: [Int],
        value: [Int],
        a: [Int],
        b: [Int],
        aLog: [Int],
        dtBias: [Int],
        state: [Int]
    )
    case invalidGDNSingleTokenFusedUpdateShape(
        mixedQKV: [Int],
        weights: [Int],
        convState: [Int],
        a: [Int],
        b: [Int],
        aLog: [Int],
        dtBias: [Int],
        state: [Int]
    )
    case invalidAttentionShape(
        query: [Int],
        key: [Int],
        value: [Int],
        queryHeadCount: Int,
        keyValueHeadCount: Int,
        headDimension: Int
    )
    case invalidRowCopy(source: [Int], destination: [Int], startRow: Int)
    case invalidRowGather(source: [Int], rowIndices: [Int])
    case invalidAttentionScoreShape(
        query: [Int],
        key: [Int],
        scores: [Int],
        keyValueTokenCount: Int,
        queryHeadCount: Int,
        keyValueHeadCount: Int,
        headDimension: Int
    )
    case dtypeMismatch
}

final class MetalArgmaxLastRowScratch {
    let tokenBuffer: MTLBuffer
    let logitBuffer: MTLBuffer

    init(tokenBuffer: MTLBuffer, logitBuffer: MTLBuffer) {
        self.tokenBuffer = tokenBuffer
        self.logitBuffer = logitBuffer
    }
}

final class MetalAffineQMVArgmaxScratch {
    let partialTokenBuffer: MTLBuffer
    let partialLogitBuffer: MTLBuffer
    let tokenBuffer: MTLBuffer
    let logitBuffer: MTLBuffer
    let partialCount: Int

    init(
        partialTokenBuffer: MTLBuffer,
        partialLogitBuffer: MTLBuffer,
        tokenBuffer: MTLBuffer,
        logitBuffer: MTLBuffer,
        partialCount: Int
    ) {
        self.partialTokenBuffer = partialTokenBuffer
        self.partialLogitBuffer = partialLogitBuffer
        self.tokenBuffer = tokenBuffer
        self.logitBuffer = logitBuffer
        self.partialCount = partialCount
    }
}

/// Executes EdgeEngine-owned MSL kernels.
///
/// Phase 0 compiles source at runtime to keep the package simple. Once kernel
/// coverage stabilizes we can move these kernels into precompiled package
/// resources.
public final class MetalKernelExecutor {
    private let runtime: EdgeMetalRuntime
    private let library: MTLLibrary
    private let fp32MatmulPipeline: MTLComputePipelineState
    private let q4MatmulPipeline: MTLComputePipelineState
    private let affineQuantizedMatmulPipeline: MTLComputePipelineState
    private let affineQuantizedQMVTransposedPipeline: MTLComputePipelineState
    private let affineQuantizedQMVTransposedArgmaxPartialsPipeline: MTLComputePipelineState
    private let argmaxPartialsPipeline: MTLComputePipelineState
    private let splitColumnsPipeline: MTLComputePipelineState
    private let splitGatedQueryPipeline: MTLComputePipelineState
    private let ropePipeline: MTLComputePipelineState
    private let rmsNormPipeline: MTLComputePipelineState
    private let rmsNormByHeadPipeline: MTLComputePipelineState
    private let sigmoidMultiplyPipeline: MTLComputePipelineState
    private let siluMultiplyPipeline: MTLComputePipelineState
    private let gdnDepthwiseConv1DPipeline: MTLComputePipelineState
    private let gdnQKNormalizationPipeline: MTLComputePipelineState
    private let gdnRecurrentUpdatePipeline: MTLComputePipelineState
    private let gdnSingleTokenFusedUpdatePipeline: MTLComputePipelineState
    private let addPipeline: MTLComputePipelineState
    private let embeddingLookupPipeline: MTLComputePipelineState
    private let affineQuantizedEmbeddingLookupPipeline: MTLComputePipelineState
    private let scaledDotProductAttentionPipeline: MTLComputePipelineState
    private let copyRowsToPrefixPipeline: MTLComputePipelineState
    private let gatherRowsPipeline: MTLComputePipelineState
    private let updateAttentionScoreEMAPipeline: MTLComputePipelineState
    private let argmaxLastRowPipeline: MTLComputePipelineState
    private var affineQuantizedBufferCache: [AffineQuantizedBufferCacheKey: AffineQuantizedMetalBuffers] = [:]
    private var affineQMVArgmaxScratch: MetalAffineQMVArgmaxScratch?
    public private(set) var schedulingSnapshot: MetalCommandScheduler
    public private(set) var lastExecutionStats: MetalKernelExecutionStats?
    public private(set) var affineQuantizedBufferCacheStats = MetalQuantizedBufferCacheStats()
    private var lastScheduledByteCost: Int = 0

    public var runtimeConfiguration: MetalRuntimeConfiguration {
        runtime.configuration
    }

    public init(runtime: EdgeMetalRuntime) throws {
        self.runtime = runtime
        self.schedulingSnapshot = runtime.makeScheduler()
        do {
            self.library = try runtime.device.makeLibrary(source: Self.kernelSource, options: nil)
        } catch {
            throw MetalKernelExecutorError.libraryCompilationFailed
        }

        guard let function = library.makeFunction(name: "edge_fp32_matmul") else {
            throw MetalKernelExecutorError.missingFunction("edge_fp32_matmul")
        }

        do {
            self.fp32MatmulPipeline = try runtime.device.makeComputePipelineState(function: function)
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_fp32_matmul")
        }

        guard let q4Function = library.makeFunction(name: "edge_q4_matmul") else {
            throw MetalKernelExecutorError.missingFunction("edge_q4_matmul")
        }

        do {
            self.q4MatmulPipeline = try runtime.device.makeComputePipelineState(function: q4Function)
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_q4_matmul")
        }

        guard let affineQuantizedFunction = library.makeFunction(name: "edge_affine_quantized_matmul") else {
            throw MetalKernelExecutorError.missingFunction("edge_affine_quantized_matmul")
        }

        do {
            self.affineQuantizedMatmulPipeline = try runtime.device.makeComputePipelineState(
                function: affineQuantizedFunction
            )
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_affine_quantized_matmul")
        }

        guard let affineQuantizedQMVTransposedFunction = library.makeFunction(name: "edge_affine_qmv_transposed") else {
            throw MetalKernelExecutorError.missingFunction("edge_affine_qmv_transposed")
        }

        do {
            self.affineQuantizedQMVTransposedPipeline = try runtime.device.makeComputePipelineState(
                function: affineQuantizedQMVTransposedFunction
            )
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_affine_qmv_transposed")
        }

        guard let affineQuantizedQMVTransposedArgmaxPartialsFunction = library.makeFunction(
            name: "edge_affine_qmv_transposed_argmax_partials"
        ) else {
            throw MetalKernelExecutorError.missingFunction("edge_affine_qmv_transposed_argmax_partials")
        }

        do {
            self.affineQuantizedQMVTransposedArgmaxPartialsPipeline = try runtime.device.makeComputePipelineState(
                function: affineQuantizedQMVTransposedArgmaxPartialsFunction
            )
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_affine_qmv_transposed_argmax_partials")
        }

        guard let argmaxPartialsFunction = library.makeFunction(name: "edge_argmax_partials") else {
            throw MetalKernelExecutorError.missingFunction("edge_argmax_partials")
        }

        do {
            self.argmaxPartialsPipeline = try runtime.device.makeComputePipelineState(function: argmaxPartialsFunction)
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_argmax_partials")
        }

        guard let splitColumnsFunction = library.makeFunction(name: "edge_split_columns") else {
            throw MetalKernelExecutorError.missingFunction("edge_split_columns")
        }

        do {
            self.splitColumnsPipeline = try runtime.device.makeComputePipelineState(function: splitColumnsFunction)
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_split_columns")
        }

        guard let splitGatedQueryFunction = library.makeFunction(name: "edge_split_gated_query") else {
            throw MetalKernelExecutorError.missingFunction("edge_split_gated_query")
        }

        do {
            self.splitGatedQueryPipeline = try runtime.device.makeComputePipelineState(
                function: splitGatedQueryFunction
            )
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_split_gated_query")
        }

        guard let ropeFunction = library.makeFunction(name: "edge_rope") else {
            throw MetalKernelExecutorError.missingFunction("edge_rope")
        }

        do {
            self.ropePipeline = try runtime.device.makeComputePipelineState(function: ropeFunction)
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_rope")
        }

        guard let rmsNormFunction = library.makeFunction(name: "edge_rms_norm") else {
            throw MetalKernelExecutorError.missingFunction("edge_rms_norm")
        }

        do {
            self.rmsNormPipeline = try runtime.device.makeComputePipelineState(function: rmsNormFunction)
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_rms_norm")
        }

        guard let rmsNormByHeadFunction = library.makeFunction(name: "edge_rms_norm_by_head") else {
            throw MetalKernelExecutorError.missingFunction("edge_rms_norm_by_head")
        }

        do {
            self.rmsNormByHeadPipeline = try runtime.device.makeComputePipelineState(function: rmsNormByHeadFunction)
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_rms_norm_by_head")
        }

        guard let sigmoidMultiplyFunction = library.makeFunction(name: "edge_sigmoid_multiply") else {
            throw MetalKernelExecutorError.missingFunction("edge_sigmoid_multiply")
        }

        do {
            self.sigmoidMultiplyPipeline = try runtime.device.makeComputePipelineState(function: sigmoidMultiplyFunction)
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_sigmoid_multiply")
        }

        guard let siluMultiplyFunction = library.makeFunction(name: "edge_silu_multiply") else {
            throw MetalKernelExecutorError.missingFunction("edge_silu_multiply")
        }

        do {
            self.siluMultiplyPipeline = try runtime.device.makeComputePipelineState(function: siluMultiplyFunction)
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_silu_multiply")
        }

        guard let gdnDepthwiseConv1DFunction = library.makeFunction(name: "edge_gdn_depthwise_conv1d") else {
            throw MetalKernelExecutorError.missingFunction("edge_gdn_depthwise_conv1d")
        }

        do {
            self.gdnDepthwiseConv1DPipeline = try runtime.device.makeComputePipelineState(
                function: gdnDepthwiseConv1DFunction
            )
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_gdn_depthwise_conv1d")
        }

        guard let gdnQKNormalizationFunction = library.makeFunction(name: "edge_gdn_normalize_qk") else {
            throw MetalKernelExecutorError.missingFunction("edge_gdn_normalize_qk")
        }

        do {
            self.gdnQKNormalizationPipeline = try runtime.device.makeComputePipelineState(
                function: gdnQKNormalizationFunction
            )
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_gdn_normalize_qk")
        }

        guard let gdnRecurrentUpdateFunction = library.makeFunction(name: "edge_gdn_recurrent_update") else {
            throw MetalKernelExecutorError.missingFunction("edge_gdn_recurrent_update")
        }

        do {
            self.gdnRecurrentUpdatePipeline = try runtime.device.makeComputePipelineState(
                function: gdnRecurrentUpdateFunction
            )
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_gdn_recurrent_update")
        }

        guard let gdnSingleTokenFusedUpdateFunction = library.makeFunction(
            name: "edge_gdn_single_token_fused_update"
        ) else {
            throw MetalKernelExecutorError.missingFunction("edge_gdn_single_token_fused_update")
        }

        do {
            self.gdnSingleTokenFusedUpdatePipeline = try runtime.device.makeComputePipelineState(
                function: gdnSingleTokenFusedUpdateFunction
            )
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_gdn_single_token_fused_update")
        }

        guard let addFunction = library.makeFunction(name: "edge_add") else {
            throw MetalKernelExecutorError.missingFunction("edge_add")
        }

        do {
            self.addPipeline = try runtime.device.makeComputePipelineState(function: addFunction)
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_add")
        }

        guard let embeddingLookupFunction = library.makeFunction(name: "edge_embedding_lookup") else {
            throw MetalKernelExecutorError.missingFunction("edge_embedding_lookup")
        }

        do {
            self.embeddingLookupPipeline = try runtime.device.makeComputePipelineState(
                function: embeddingLookupFunction
            )
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_embedding_lookup")
        }

        guard let affineQuantizedEmbeddingLookupFunction = library.makeFunction(
            name: "edge_affine_quantized_embedding_lookup"
        ) else {
            throw MetalKernelExecutorError.missingFunction("edge_affine_quantized_embedding_lookup")
        }

        do {
            self.affineQuantizedEmbeddingLookupPipeline = try runtime.device.makeComputePipelineState(
                function: affineQuantizedEmbeddingLookupFunction
            )
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_affine_quantized_embedding_lookup")
        }

        guard let scaledDotProductAttentionFunction = library.makeFunction(
            name: "edge_scaled_dot_product_attention"
        ) else {
            throw MetalKernelExecutorError.missingFunction("edge_scaled_dot_product_attention")
        }

        do {
            self.scaledDotProductAttentionPipeline = try runtime.device.makeComputePipelineState(
                function: scaledDotProductAttentionFunction
            )
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_scaled_dot_product_attention")
        }

        guard let copyRowsToPrefixFunction = library.makeFunction(name: "edge_copy_rows_to_prefix") else {
            throw MetalKernelExecutorError.missingFunction("edge_copy_rows_to_prefix")
        }

        do {
            self.copyRowsToPrefixPipeline = try runtime.device.makeComputePipelineState(
                function: copyRowsToPrefixFunction
            )
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_copy_rows_to_prefix")
        }

        guard let gatherRowsFunction = library.makeFunction(name: "edge_gather_rows") else {
            throw MetalKernelExecutorError.missingFunction("edge_gather_rows")
        }

        do {
            self.gatherRowsPipeline = try runtime.device.makeComputePipelineState(
                function: gatherRowsFunction
            )
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_gather_rows")
        }

        guard let updateAttentionScoreEMAFunction = library.makeFunction(
            name: "edge_update_attention_score_ema"
        ) else {
            throw MetalKernelExecutorError.missingFunction("edge_update_attention_score_ema")
        }

        do {
            self.updateAttentionScoreEMAPipeline = try runtime.device.makeComputePipelineState(
                function: updateAttentionScoreEMAFunction
            )
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_update_attention_score_ema")
        }

        guard let argmaxLastRowFunction = library.makeFunction(name: "edge_argmax_last_row") else {
            throw MetalKernelExecutorError.missingFunction("edge_argmax_last_row")
        }

        do {
            self.argmaxLastRowPipeline = try runtime.device.makeComputePipelineState(
                function: argmaxLastRowFunction
            )
        } catch {
            throw MetalKernelExecutorError.pipelineCreationFailed("edge_argmax_last_row")
        }
    }

    public func withUnboundedCommandBufferBatch<T>(_ body: () throws -> T) rethrows -> T {
        try runtime.withUnboundedCommandBufferBatch(body)
    }

    public func makeFloat32Tensor(
        _ values: [Float],
        shape: EdgeTensorShape
    ) throws -> EdgeTensor {
        try EdgeTensor(float32: values, shape: shape, runtime: runtime)
    }

    func makeArgmaxLastRowScratch() throws -> MetalArgmaxLastRowScratch {
        guard let tokenBuffer = runtime.device.makeBuffer(
            length: MemoryLayout<UInt32>.stride,
            options: [.storageModeShared]
        ) else {
            throw EdgeTensorError.bufferAllocationFailed(byteCount: MemoryLayout<UInt32>.stride)
        }
        guard let logitBuffer = runtime.device.makeBuffer(
            length: MemoryLayout<Float>.stride,
            options: [.storageModeShared]
        ) else {
            throw EdgeTensorError.bufferAllocationFailed(byteCount: MemoryLayout<Float>.stride)
        }
        return MetalArgmaxLastRowScratch(tokenBuffer: tokenBuffer, logitBuffer: logitBuffer)
    }

    private func makeAffineQMVArgmaxScratch(columns: Int) throws -> MetalAffineQMVArgmaxScratch {
        let partialCount = affineQMVArgmaxPartialCount(columns: columns)
        if let scratch = affineQMVArgmaxScratch, scratch.partialCount == partialCount {
            return scratch
        }

        let partialTokenByteCount = partialCount * MemoryLayout<UInt32>.stride
        let partialLogitByteCount = partialCount * MemoryLayout<Float>.stride
        guard let partialTokenBuffer = runtime.device.makeBuffer(
            length: partialTokenByteCount,
            options: [.storageModeShared]
        ) else {
            throw EdgeTensorError.bufferAllocationFailed(byteCount: partialTokenByteCount)
        }
        guard let partialLogitBuffer = runtime.device.makeBuffer(
            length: partialLogitByteCount,
            options: [.storageModeShared]
        ) else {
            throw EdgeTensorError.bufferAllocationFailed(byteCount: partialLogitByteCount)
        }
        guard let tokenBuffer = runtime.device.makeBuffer(
            length: MemoryLayout<UInt32>.stride,
            options: [.storageModeShared]
        ) else {
            throw EdgeTensorError.bufferAllocationFailed(byteCount: MemoryLayout<UInt32>.stride)
        }
        guard let logitBuffer = runtime.device.makeBuffer(
            length: MemoryLayout<Float>.stride,
            options: [.storageModeShared]
        ) else {
            throw EdgeTensorError.bufferAllocationFailed(byteCount: MemoryLayout<Float>.stride)
        }
        let scratch = MetalAffineQMVArgmaxScratch(
            partialTokenBuffer: partialTokenBuffer,
            partialLogitBuffer: partialLogitBuffer,
            tokenBuffer: tokenBuffer,
            logitBuffer: logitBuffer,
            partialCount: partialCount
        )
        affineQMVArgmaxScratch = scratch
        return scratch
    }

    @discardableResult
    public func removeAffineQuantizedBuffers(
        scope: MetalQuantizedBufferCacheScope
    ) -> MetalQuantizedBufferCacheStats {
        let previousStats = affineQuantizedBufferCacheStats
        switch scope {
        case .decoder, .all:
            affineQuantizedBufferCache.removeAll(keepingCapacity: false)
            affineQuantizedBufferCacheStats.entryCount = 0
            affineQuantizedBufferCacheStats.cachedByteCount = 0
        case .vision:
            break
        }
        return previousStats
    }

    public func matmul(_ lhs: EdgeTensor, _ rhs: EdgeTensor) throws -> EdgeTensor {
        guard lhs.dataType == .float32, rhs.dataType == .float32 else {
            throw MetalKernelExecutorError.dtypeMismatch
        }
        guard lhs.shape.rank == 2, rhs.shape.rank == 2 else {
            throw MetalKernelExecutorError.invalidMatrixRank
        }

        let rows = lhs.shape.dimensions[0]
        let inner = lhs.shape.dimensions[1]
        let rhsInner = rhs.shape.dimensions[0]
        let columns = rhs.shape.dimensions[1]
        guard inner == rhsInner else {
            throw MetalKernelExecutorError.matrixDimensionMismatch
        }

        let output = try EdgeTensor(
            shape: EdgeTensorShape([rows, columns]),
            dataType: .float32,
            runtime: runtime
        )
        recordScheduledOperation(
            name: "edge_fp32_matmul",
            byteCost: lhs.byteCount + rhs.byteCount + output.byteCount
        )

        let commandBuffer = try makeCommandBufferForRecordedOperation()
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }

        encoder.setComputePipelineState(fp32MatmulPipeline)
        encoder.setBuffer(lhs.buffer, offset: 0, index: 0)
        encoder.setBuffer(rhs.buffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        retainResources(lhs.buffer, rhs.buffer, output.buffer, for: commandBuffer)

        var rows32 = UInt32(rows)
        var inner32 = UInt32(inner)
        var columns32 = UInt32(columns)
        encoder.setBytes(&rows32, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&inner32, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&columns32, length: MemoryLayout<UInt32>.stride, index: 5)

        let totalThreads = rows * columns
        let threadExecutionWidth = fp32MatmulPipeline.threadExecutionWidth
        let threadsPerGroup = MTLSize(
            width: min(max(threadExecutionWidth, 1), 256),
            height: 1,
            depth: 1
        )
        let gridSize = MTLSize(width: totalThreads, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        runtime.finishOperationCommandBuffer(commandBuffer)

        return output
    }

    public func q4Matmul(_ lhs: EdgeTensor, weights: Q4WeightMatrix) throws -> EdgeTensor {
        guard lhs.dataType == .float32 else {
            throw MetalKernelExecutorError.dtypeMismatch
        }
        guard lhs.shape.rank == 2 else {
            throw MetalKernelExecutorError.invalidMatrixRank
        }

        let rows = lhs.shape.dimensions[0]
        let inner = lhs.shape.dimensions[1]
        guard inner == weights.rows else {
            throw MetalKernelExecutorError.matrixDimensionMismatch
        }

        let output = try EdgeTensor(
            shape: EdgeTensorShape([rows, weights.columns]),
            dataType: .float32,
            runtime: runtime
        )
        let scalesByteCount = weights.scales.count * MemoryLayout<Float>.stride
        recordScheduledOperation(
            name: "edge_q4_matmul",
            byteCost: lhs.byteCount + weights.packedValues.count + scalesByteCount + output.byteCount
        )

        guard let packedBuffer = weights.packedValues.withUnsafeBytes({ bytes in
            runtime.device.makeBuffer(
                bytes: bytes.baseAddress!,
                length: weights.packedValues.count,
                options: [.storageModeShared]
            )
        }) else {
            throw EdgeTensorError.bufferAllocationFailed(byteCount: weights.packedValues.count)
        }
        guard let scalesBuffer = weights.scales.withUnsafeBytes({ bytes in
            runtime.device.makeBuffer(
                bytes: bytes.baseAddress!,
                length: scalesByteCount,
                options: [.storageModeShared]
            )
        }) else {
            throw EdgeTensorError.bufferAllocationFailed(byteCount: scalesByteCount)
        }

        let commandBuffer = try makeCommandBufferForRecordedOperation()
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }

        encoder.setComputePipelineState(q4MatmulPipeline)
        encoder.setBuffer(lhs.buffer, offset: 0, index: 0)
        encoder.setBuffer(packedBuffer, offset: 0, index: 1)
        encoder.setBuffer(scalesBuffer, offset: 0, index: 2)
        encoder.setBuffer(output.buffer, offset: 0, index: 3)
        retainResources(lhs.buffer, packedBuffer, scalesBuffer, output.buffer, for: commandBuffer)

        var rows32 = UInt32(rows)
        var inner32 = UInt32(inner)
        var columns32 = UInt32(weights.columns)
        var groupSize32 = UInt32(weights.groupSize)
        encoder.setBytes(&rows32, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&inner32, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&columns32, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&groupSize32, length: MemoryLayout<UInt32>.stride, index: 7)

        let totalThreads = rows * weights.columns
        let threadExecutionWidth = q4MatmulPipeline.threadExecutionWidth
        let threadsPerGroup = MTLSize(
            width: min(max(threadExecutionWidth, 1), 256),
            height: 1,
            depth: 1
        )
        let gridSize = MTLSize(width: totalThreads, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        runtime.finishOperationCommandBuffer(commandBuffer)

        return output
    }

    public func affineQuantizedMatmul(
        _ lhs: EdgeTensor,
        weights: EdgeQuantizedTensor,
        transpose: Bool = false,
        diagnosticSink: ((String) -> Void)? = nil,
        diagnosticName: String = "affine_quantized_matmul"
    ) throws -> EdgeTensor {
        guard lhs.dataType == .float32 else {
            throw MetalKernelExecutorError.dtypeMismatch
        }
        guard lhs.shape.rank == 2, weights.shape.count == 2 else {
            throw MetalKernelExecutorError.invalidMatrixRank
        }

        let rows = lhs.shape.dimensions[0]
        let inner = lhs.shape.dimensions[1]
        let weightRows = transpose ? weights.shape[1] : weights.shape[0]
        let columns = transpose ? weights.shape[0] : weights.shape[1]
        guard inner == weightRows else {
            throw MetalKernelExecutorError.matrixDimensionMismatch
        }

        let output = try EdgeTensor(
            shape: EdgeTensorShape([rows, columns]),
            dataType: .float32,
            runtime: runtime
        )
        let packedByteCount = weights.packedByteCount
        let scalesByteCount = weights.scalesByteCount
        let biasesByteCount = weights.affineBiasBufferByteCount
        let byteCost = lhs.byteCount + packedByteCount + scalesByteCount + biasesByteCount + output.byteCount
        let useQMVTransposed = shouldUseAffineQuantizedQMVTransposed(
            rows: rows,
            inner: inner,
            weights: weights,
            transpose: transpose
        )
        let useMLXPrefillMatmul = shouldUseMLXQuantizedPrefillMatmul(
            rows: rows,
            inner: inner,
            columns: columns,
            weights: weights,
            transpose: transpose
        )
        let useVendoredCommandBufferPrefillQMM = shouldUseVendoredCommandBufferPrefillQMM(
            rows: rows,
            inner: inner,
            columns: columns,
            weights: weights,
            transpose: transpose
        )
        let backendName: String
        if useVendoredCommandBufferPrefillQMM,
           EdgeMLXBridge.supportsAffineQMMTransposedCommandBufferFloat32MTL(
               rows: rows,
               inner: inner,
               weights: weights,
               transpose: transpose
           ) {
            backendName = "edge_cmlx_affine_qmm_t_command_buffer"
        } else if (runtime.configuration.useMLXQuantizedMatmul || useMLXPrefillMatmul),
           EdgeMLXBridge.supportsAffineQuantizedMatmulFloat32MTL(
               rows: rows,
               inner: inner,
               weights: weights,
               transpose: transpose
           ) {
            backendName = "edge_cmlx_affine_quantized_matmul"
        } else {
            backendName = useQMVTransposed ? "edge_affine_qmv_transposed" : "edge_affine_quantized_matmul"
        }
        diagnosticSink?(
            "\(diagnosticName)_begin backend=\(backendName) rows=\(rows) inner=\(inner) columns=\(columns) bits=\(weights.bits) groupSize=\(weights.groupSize) transpose=\(transpose ? 1 : 0)"
        )

        if backendName == "edge_cmlx_affine_quantized_matmul" {
            recordScheduledOperation(
                name: backendName,
                byteCost: byteCost
            )
            let cacheStatsBefore = affineQuantizedBufferCacheStats
            let weightBuffers = try affineQuantizedMetalBuffers(for: weights, preferNoCopy: false)
            emitAffineQuantizedCacheDiagnostic(
                diagnosticSink: diagnosticSink,
                diagnosticName: diagnosticName,
                before: cacheStatsBefore
            )
            diagnosticSink?("\(diagnosticName)_wait_pending_begin")
            runtime.waitForPendingWork()
            diagnosticSink?("\(diagnosticName)_wait_pending_done")
            try EdgeMLXBridge.affineQuantizedMatmulFloat32MTL(
                lhsBuffer: lhs.buffer,
                rows: rows,
                inner: inner,
                weights: weights,
                packedWeightsBuffer: weightBuffers.packedBuffer,
                scalesBuffer: weightBuffers.scalesBuffer,
                biasesBuffer: weightBuffers.biasesBuffer,
                outputBuffer: output.buffer,
                transpose: transpose
            )
            diagnosticSink?("\(diagnosticName)_done shape=\(output.shape.dimensions)")
            return output
        }

        if backendName == "edge_cmlx_affine_qmm_t_command_buffer" {
            recordScheduledOperation(
                name: backendName,
                byteCost: byteCost
            )
            let cacheStatsBefore = affineQuantizedBufferCacheStats
            let weightBuffers = try affineQuantizedMetalBuffers(for: weights, preferNoCopy: true)
            emitAffineQuantizedCacheDiagnostic(
                diagnosticSink: diagnosticSink,
                diagnosticName: diagnosticName,
                before: cacheStatsBefore
            )
            let commandBuffer = try makeCommandBufferForRecordedOperation(
                diagnosticSink: diagnosticSink,
                diagnosticName: diagnosticName
            )
            try EdgeMLXBridge.encodeAffineQMMTransposedCommandBufferFloat32MTL(
                commandBuffer: commandBuffer,
                lhsBuffer: lhs.buffer,
                rows: rows,
                inner: inner,
                weights: weights,
                packedWeightsBuffer: weightBuffers.packedBuffer,
                packedWeightsOffset: weightBuffers.packedOffset,
                scalesBuffer: weightBuffers.scalesBuffer,
                scalesOffset: weightBuffers.scalesOffset,
                biasesBuffer: weightBuffers.biasesBuffer,
                biasesOffset: weightBuffers.biasesOffset,
                outputBuffer: output.buffer,
                transpose: transpose
            )
            retainResources(
                lhs.buffer,
                weightBuffers.packedBuffer,
                weightBuffers.scalesBuffer,
                weightBuffers.biasesBuffer,
                output.buffer,
                for: commandBuffer
            )
            runtime.finishOperationCommandBuffer(commandBuffer)
            diagnosticSink?(
                "\(diagnosticName)_scheduled pendingOps=\(lastExecutionStats?.pendingOpsAfterRecord ?? -1) pendingBytes=\(lastExecutionStats?.pendingBytesAfterRecord ?? -1) commits=\(lastExecutionStats?.logicalCommitCount ?? -1)"
            )
            diagnosticSink?("\(diagnosticName)_done shape=\(output.shape.dimensions)")
            return output
        }

        recordScheduledOperation(
            name: backendName,
            byteCost: byteCost
        )

        let cacheStatsBefore = affineQuantizedBufferCacheStats
        let weightBuffers = try affineQuantizedMetalBuffers(for: weights, preferNoCopy: useQMVTransposed)
        emitAffineQuantizedCacheDiagnostic(
            diagnosticSink: diagnosticSink,
            diagnosticName: diagnosticName,
            before: cacheStatsBefore
        )

        let commandBuffer = try makeCommandBufferForRecordedOperation(
            diagnosticSink: diagnosticSink,
            diagnosticName: diagnosticName
        )
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }

        encoder.setComputePipelineState(useQMVTransposed ? affineQuantizedQMVTransposedPipeline : affineQuantizedMatmulPipeline)
        encoder.setBuffer(lhs.buffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffers.packedBuffer, offset: weightBuffers.packedOffset, index: 1)
        encoder.setBuffer(weightBuffers.scalesBuffer, offset: weightBuffers.scalesOffset, index: 2)
        encoder.setBuffer(weightBuffers.biasesBuffer, offset: weightBuffers.biasesOffset, index: 3)
        encoder.setBuffer(output.buffer, offset: 0, index: 4)
        retainResources(
            lhs.buffer,
            weightBuffers.packedBuffer,
            weightBuffers.scalesBuffer,
            weightBuffers.biasesBuffer,
            output.buffer,
            for: commandBuffer
        )

        var rows32 = UInt32(rows)
        var inner32 = UInt32(inner)
        var columns32 = UInt32(columns)
        var packedWordsPerRow32 = UInt32(weights.packedWordsPerLogicalRow)
        var scaleColumns32 = UInt32(weights.scaleShape.last!)
        var groupSize32 = UInt32(weights.groupSize)
        var bits32 = UInt32(weights.bits)
        var transpose32 = transpose ? UInt32(1) : UInt32(0)
        encoder.setBytes(&rows32, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&inner32, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&columns32, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes(&packedWordsPerRow32, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.setBytes(&scaleColumns32, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.setBytes(&groupSize32, length: MemoryLayout<UInt32>.stride, index: 10)
        encoder.setBytes(&bits32, length: MemoryLayout<UInt32>.stride, index: 11)
        encoder.setBytes(&transpose32, length: MemoryLayout<UInt32>.stride, index: 12)

        if useQMVTransposed {
            let threadsPerGroup = MTLSize(
                width: 32,
                height: 2,
                depth: 1
            )
            let gridSize = MTLSize(
                width: rows,
                height: (columns + 7) / 8,
                depth: 1
            )
            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadsPerGroup)
        } else {
            let totalThreads = rows * columns
            let threadExecutionWidth = affineQuantizedMatmulPipeline.threadExecutionWidth
            let threadsPerGroup = MTLSize(
                width: min(max(threadExecutionWidth, 1), 256),
                height: 1,
                depth: 1
            )
            let gridSize = MTLSize(width: totalThreads, height: 1, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        }
        encoder.endEncoding()
        runtime.finishOperationCommandBuffer(commandBuffer)
        diagnosticSink?(
            "\(diagnosticName)_scheduled pendingOps=\(lastExecutionStats?.pendingOpsAfterRecord ?? -1) pendingBytes=\(lastExecutionStats?.pendingBytesAfterRecord ?? -1) commits=\(lastExecutionStats?.logicalCommitCount ?? -1)"
        )
        diagnosticSink?("\(diagnosticName)_done shape=\(output.shape.dimensions)")

        return output
    }

    public func affineQuantizedMatmulArgmax(
        _ lhs: EdgeTensor,
        weights: EdgeQuantizedTensor,
        transpose: Bool = true,
        diagnosticSink: ((String) -> Void)? = nil,
        diagnosticName: String = "affine_quantized_matmul_argmax"
    ) throws -> QwenGreedyToken {
        guard lhs.dataType == .float32 else {
            throw MetalKernelExecutorError.dtypeMismatch
        }
        guard lhs.shape.rank == 2, weights.shape.count == 2 else {
            throw MetalKernelExecutorError.invalidMatrixRank
        }

        let rows = lhs.shape.dimensions[0]
        let inner = lhs.shape.dimensions[1]
        let weightRows = transpose ? weights.shape[1] : weights.shape[0]
        let columns = transpose ? weights.shape[0] : weights.shape[1]
        guard inner == weightRows else {
            throw MetalKernelExecutorError.matrixDimensionMismatch
        }
        guard rows == 1,
              shouldUseAffineQuantizedQMVTransposed(
                  rows: rows,
                  inner: inner,
                  weights: weights,
                  transpose: transpose
              )
        else {
            diagnosticSink?(
                "\(diagnosticName)_fallback reason=unsupported rows=\(rows) inner=\(inner) columns=\(columns) bits=\(weights.bits) groupSize=\(weights.groupSize) transpose=\(transpose ? 1 : 0)"
            )
            let logits = try affineQuantizedMatmul(
                lhs,
                weights: weights,
                transpose: transpose,
                diagnosticSink: diagnosticSink,
                diagnosticName: "\(diagnosticName)_fallback_logits"
            )
            return try argmaxLastRow(logits)
        }

        let scratch = try makeAffineQMVArgmaxScratch(columns: columns)
        let packedByteCount = weights.packedByteCount
        let scalesByteCount = weights.scalesByteCount
        let biasesByteCount = weights.affineBiasBufferByteCount
        let byteCost = lhs.byteCount
            + packedByteCount
            + scalesByteCount
            + biasesByteCount
            + scratch.partialCount * (MemoryLayout<UInt32>.stride + MemoryLayout<Float>.stride)
            + MemoryLayout<UInt32>.stride
            + MemoryLayout<Float>.stride
        let backendName = "edge_affine_qmv_transposed_argmax"
        diagnosticSink?(
            "\(diagnosticName)_begin backend=\(backendName) rows=\(rows) inner=\(inner) columns=\(columns) partials=\(scratch.partialCount) bits=\(weights.bits) groupSize=\(weights.groupSize) transpose=1"
        )
        recordScheduledOperation(name: backendName, byteCost: byteCost)

        let cacheStatsBefore = affineQuantizedBufferCacheStats
        let weightBuffers = try affineQuantizedMetalBuffers(for: weights, preferNoCopy: true)
        emitAffineQuantizedCacheDiagnostic(
            diagnosticSink: diagnosticSink,
            diagnosticName: diagnosticName,
            before: cacheStatsBefore
        )

        let commandBuffer = try makeCommandBufferForRecordedOperation(
            diagnosticSink: diagnosticSink,
            diagnosticName: diagnosticName
        )
        guard let partialsEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }
        partialsEncoder.setComputePipelineState(affineQuantizedQMVTransposedArgmaxPartialsPipeline)
        partialsEncoder.setBuffer(lhs.buffer, offset: 0, index: 0)
        partialsEncoder.setBuffer(weightBuffers.packedBuffer, offset: weightBuffers.packedOffset, index: 1)
        partialsEncoder.setBuffer(weightBuffers.scalesBuffer, offset: weightBuffers.scalesOffset, index: 2)
        partialsEncoder.setBuffer(weightBuffers.biasesBuffer, offset: weightBuffers.biasesOffset, index: 3)
        partialsEncoder.setBuffer(scratch.partialTokenBuffer, offset: 0, index: 4)
        partialsEncoder.setBuffer(scratch.partialLogitBuffer, offset: 0, index: 5)
        retainResources(
            lhs.buffer,
            weightBuffers.packedBuffer,
            weightBuffers.scalesBuffer,
            weightBuffers.biasesBuffer,
            scratch.partialTokenBuffer,
            scratch.partialLogitBuffer,
            scratch.tokenBuffer,
            scratch.logitBuffer,
            for: commandBuffer
        )

        var inner32 = UInt32(inner)
        var columns32 = UInt32(columns)
        var packedWordsPerRow32 = UInt32(weights.packedWordsPerLogicalRow)
        var scaleColumns32 = UInt32(weights.scaleShape.last!)
        var groupSize32 = UInt32(weights.groupSize)
        var bits32 = UInt32(weights.bits)
        partialsEncoder.setBytes(&inner32, length: MemoryLayout<UInt32>.stride, index: 6)
        partialsEncoder.setBytes(&columns32, length: MemoryLayout<UInt32>.stride, index: 7)
        partialsEncoder.setBytes(&packedWordsPerRow32, length: MemoryLayout<UInt32>.stride, index: 8)
        partialsEncoder.setBytes(&scaleColumns32, length: MemoryLayout<UInt32>.stride, index: 9)
        partialsEncoder.setBytes(&groupSize32, length: MemoryLayout<UInt32>.stride, index: 10)
        partialsEncoder.setBytes(&bits32, length: MemoryLayout<UInt32>.stride, index: 11)
        partialsEncoder.dispatchThreadgroups(
            MTLSize(width: 1, height: (columns + 7) / 8, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1)
        )
        partialsEncoder.endEncoding()

        guard let reduceEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }
        reduceEncoder.setComputePipelineState(argmaxPartialsPipeline)
        reduceEncoder.setBuffer(scratch.partialTokenBuffer, offset: 0, index: 0)
        reduceEncoder.setBuffer(scratch.partialLogitBuffer, offset: 0, index: 1)
        reduceEncoder.setBuffer(scratch.tokenBuffer, offset: 0, index: 2)
        reduceEncoder.setBuffer(scratch.logitBuffer, offset: 0, index: 3)
        var partialCount32 = UInt32(scratch.partialCount)
        reduceEncoder.setBytes(&partialCount32, length: MemoryLayout<UInt32>.stride, index: 4)
        var reduceThreadCount = min(
            256,
            max(1, argmaxPartialsPipeline.maxTotalThreadsPerThreadgroup)
        )
        while reduceThreadCount > 1 && (reduceThreadCount & (reduceThreadCount - 1)) != 0 {
            reduceThreadCount -= 1
        }
        reduceEncoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: reduceThreadCount, height: 1, depth: 1)
        )
        reduceEncoder.endEncoding()

        runtime.finishOperationCommandBuffer(commandBuffer)
        diagnosticSink?(
            "\(diagnosticName)_scheduled pendingOps=\(lastExecutionStats?.pendingOpsAfterRecord ?? -1) pendingBytes=\(lastExecutionStats?.pendingBytesAfterRecord ?? -1) commits=\(lastExecutionStats?.logicalCommitCount ?? -1)"
        )
        runtime.waitForPendingWork()

        let tokenId = scratch.tokenBuffer.contents().load(as: UInt32.self)
        let logit = scratch.logitBuffer.contents().load(as: Float.self)
        diagnosticSink?("\(diagnosticName)_done token=\(tokenId)")
        return QwenGreedyToken(tokenId: Int(tokenId), logit: logit)
    }

    public func rmsNormAffineQuantizedMatmul(
        _ lhs: EdgeTensor,
        normWeight: EdgeTensor,
        epsilon: Float,
        weights: EdgeQuantizedTensor,
        transpose: Bool = true,
        diagnosticSink: ((String) -> Void)? = nil,
        diagnosticName: String = "rms_norm_affine_quantized_matmul"
    ) throws -> EdgeTensor {
        guard lhs.dataType == .float32,
              normWeight.dataType == .float32
        else {
            throw MetalKernelExecutorError.dtypeMismatch
        }
        guard lhs.shape.rank == 2,
              normWeight.shape.rank == 1,
              weights.shape.count == 2
        else {
            throw MetalKernelExecutorError.invalidMatrixRank
        }

        let rows = lhs.shape.dimensions[0]
        let inner = lhs.shape.dimensions[1]
        guard normWeight.shape.dimensions[0] == inner else {
            throw MetalKernelExecutorError.invalidRMSNorm(
                columns: inner,
                weightCount: normWeight.shape.dimensions[0]
            )
        }
        let weightRows = transpose ? weights.shape[1] : weights.shape[0]
        let columns = transpose ? weights.shape[0] : weights.shape[1]
        guard inner == weightRows else {
            throw MetalKernelExecutorError.matrixDimensionMismatch
        }

        let output = try EdgeTensor(
            shape: EdgeTensorShape([rows, columns]),
            dataType: .float32,
            runtime: runtime
        )
        let byteCost = lhs.byteCount
            + normWeight.byteCount
            + weights.packedByteCount
            + weights.scalesByteCount
            + weights.affineBiasBufferByteCount
            + output.byteCount
        recordScheduledOperation(
            name: "edge_cmlx_lazy_rms_norm_affine_quantized_matmul",
            byteCost: byteCost
        )
        diagnosticSink?(
            "\(diagnosticName)_begin backend=edge_cmlx_lazy rows=\(rows) inner=\(inner) columns=\(columns) bits=\(weights.bits) groupSize=\(weights.groupSize) transpose=\(transpose ? 1 : 0)"
        )

        let cacheStatsBefore = affineQuantizedBufferCacheStats
        let weightBuffers = try affineQuantizedMetalBuffers(for: weights, preferNoCopy: false)
        emitAffineQuantizedCacheDiagnostic(
            diagnosticSink: diagnosticSink,
            diagnosticName: diagnosticName,
            before: cacheStatsBefore
        )
        diagnosticSink?("\(diagnosticName)_wait_pending_begin")
        runtime.waitForPendingWork()
        diagnosticSink?("\(diagnosticName)_wait_pending_done")
        try EdgeMLXBridge.rmsNormAffineQuantizedMatmulFloat32MTL(
            lhsBuffer: lhs.buffer,
            rows: rows,
            inner: inner,
            normWeightBuffer: normWeight.buffer,
            epsilon: epsilon,
            weights: weights,
            packedWeightsBuffer: weightBuffers.packedBuffer,
            scalesBuffer: weightBuffers.scalesBuffer,
            biasesBuffer: weightBuffers.biasesBuffer,
            outputBuffer: output.buffer,
            transpose: transpose
        )
        diagnosticSink?("\(diagnosticName)_done shape=\(output.shape.dimensions)")
        return output
    }

    private func emitAffineQuantizedCacheDiagnostic(
        diagnosticSink: ((String) -> Void)?,
        diagnosticName: String,
        before: MetalQuantizedBufferCacheStats
    ) {
        guard diagnosticSink != nil else {
            return
        }
        let after = affineQuantizedBufferCacheStats
        diagnosticSink?(
            "\(diagnosticName)_qcache hits=\(after.hitCount - before.hitCount) misses=\(after.missCount - before.missCount) uploaded=\(after.uploadedByteCount - before.uploadedByteCount) cached=\(after.cachedByteCount) entries=\(after.entryCount) released=\(after.releasedHostStorageByteCount - before.releasedHostStorageByteCount)"
        )
    }

    private func shouldUseAffineQuantizedQMVTransposed(
        rows: Int,
        inner: Int,
        weights: EdgeQuantizedTensor,
        transpose: Bool
    ) -> Bool {
        transpose
            && rows > 0
            && inner >= 64
            && weights.groupSize == 64
            && weights.scaleShape.last == inner / 64
            && inner.isMultiple(of: 64)
            && (weights.bits == 4 || weights.bits == 6)
    }

    private func shouldUseMLXQuantizedPrefillMatmul(
        rows: Int,
        inner: Int,
        columns: Int,
        weights: EdgeQuantizedTensor,
        transpose: Bool
    ) -> Bool {
        runtime.configuration.useMLXQuantizedPrefillMatmul
            && !runtime.configuration.useMLXQuantizedMatmul
            && transpose
            && rows >= mlxQMVBatchLimit(inner: inner, columns: columns)
            && inner >= 64
            && weights.groupSize == 64
            && weights.scaleShape.last == inner / 64
            && inner.isMultiple(of: 64)
            && (weights.bits == 4 || weights.bits == 6)
    }

    private func shouldUseVendoredCommandBufferPrefillQMM(
        rows: Int,
        inner: Int,
        columns: Int,
        weights: EdgeQuantizedTensor,
        transpose: Bool
    ) -> Bool {
        runtime.configuration.useVendoredCommandBufferPrefillQMM
            && !runtime.configuration.useMLXQuantizedMatmul
            && transpose
            && rows >= mlxQMVBatchLimit(inner: inner, columns: columns)
            && inner >= 64
            && weights.groupSize == 64
            && weights.scaleShape.last == inner / 64
            && inner.isMultiple(of: 64)
            && (weights.bits == 4 || weights.bits == 6)
    }

    private func mlxQMVBatchLimit(inner: Int, columns: Int) -> Int {
        let architectureName: String
        if #available(macOS 13.0, iOS 16.0, *) {
            architectureName = runtime.device.architecture.name
        } else {
            architectureName = ""
        }
        let architectureSuffix = architectureName.last
        let architectureGeneration = metalArchitectureGeneration(from: architectureName)

        if architectureGeneration == 13 || architectureGeneration == 14 {
            if architectureSuffix == "d" {
                if inner <= 2_048 && columns <= 2_048 { return 32 }
                if inner <= 4_096 && columns <= 4_096 { return 18 }
                return 12
            }
            if inner <= 2_048 && columns <= 2_048 { return 14 }
            if inner <= 4_096 && columns <= 4_096 { return 10 }
            return 6
        }

        if architectureSuffix == "d" {
            if inner <= 2_048 && columns <= 2_048 { return 32 }
            if inner <= 4_096 && columns <= 4_096 { return 18 }
            return 12
        }
        if inner <= 2_048 && columns <= 2_048 { return 18 }
        if inner <= 4_096 && columns <= 4_096 { return 12 }
        return 10
    }

    private func affineQMVArgmaxPartialCount(columns: Int) -> Int {
        ((columns + 7) / 8) * 2
    }

    private func metalArchitectureGeneration(from architectureName: String) -> Int? {
        guard let gIndex = architectureName.firstIndex(of: "g") else {
            return nil
        }
        let digits = architectureName[architectureName.index(after: gIndex)...].prefix { $0.isNumber }
        guard !digits.isEmpty else {
            return nil
        }
        return Int(String(digits))
    }

    public func splitColumns(
        _ tensor: EdgeTensor,
        firstColumnCount: Int
    ) throws -> (first: EdgeTensor, second: EdgeTensor) {
        guard tensor.dataType == .float32 else {
            throw MetalKernelExecutorError.dtypeMismatch
        }
        guard tensor.shape.rank == 2 else {
            throw MetalKernelExecutorError.invalidMatrixRank
        }

        let rows = tensor.shape.dimensions[0]
        let columns = tensor.shape.dimensions[1]
        guard firstColumnCount > 0, firstColumnCount < columns else {
            throw MetalKernelExecutorError.invalidColumnSplit(
                firstColumnCount: firstColumnCount,
                columns: columns
            )
        }

        let secondColumnCount = columns - firstColumnCount
        let first = try EdgeTensor(
            shape: EdgeTensorShape([rows, firstColumnCount]),
            dataType: .float32,
            runtime: runtime
        )
        let second = try EdgeTensor(
            shape: EdgeTensorShape([rows, secondColumnCount]),
            dataType: .float32,
            runtime: runtime
        )
        recordScheduledOperation(
            name: "edge_split_columns",
            byteCost: tensor.byteCount + first.byteCount + second.byteCount
        )

        let commandBuffer = try makeCommandBufferForRecordedOperation()
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }

        encoder.setComputePipelineState(splitColumnsPipeline)
        encoder.setBuffer(tensor.buffer, offset: 0, index: 0)
        encoder.setBuffer(first.buffer, offset: 0, index: 1)
        encoder.setBuffer(second.buffer, offset: 0, index: 2)
        retainResources(tensor.buffer, first.buffer, second.buffer, for: commandBuffer)

        var rows32 = UInt32(rows)
        var columns32 = UInt32(columns)
        var firstColumnCount32 = UInt32(firstColumnCount)
        encoder.setBytes(&rows32, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&columns32, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&firstColumnCount32, length: MemoryLayout<UInt32>.stride, index: 5)

        let totalThreads = rows * columns
        let threadExecutionWidth = splitColumnsPipeline.threadExecutionWidth
        let threadsPerGroup = MTLSize(
            width: min(max(threadExecutionWidth, 1), 256),
            height: 1,
            depth: 1
        )
        let gridSize = MTLSize(width: totalThreads, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        runtime.finishOperationCommandBuffer(commandBuffer)

        return (first, second)
    }

    public func splitGatedQuery(
        _ tensor: EdgeTensor,
        headCount: Int,
        headDimension: Int
    ) throws -> (query: EdgeTensor, gate: EdgeTensor) {
        guard tensor.dataType == .float32 else {
            throw MetalKernelExecutorError.dtypeMismatch
        }
        guard tensor.shape.rank == 2 else {
            throw MetalKernelExecutorError.invalidMatrixRank
        }

        let rows = tensor.shape.dimensions[0]
        let columns = tensor.shape.dimensions[1]
        let outputColumns = headCount * headDimension
        guard headCount > 0, headDimension > 0, columns == outputColumns * 2 else {
            throw MetalKernelExecutorError.invalidGatedQuerySplit(
                headCount: headCount,
                headDimension: headDimension,
                columns: columns
            )
        }

        let query = try EdgeTensor(
            shape: EdgeTensorShape([rows, outputColumns]),
            dataType: .float32,
            runtime: runtime
        )
        let gate = try EdgeTensor(
            shape: EdgeTensorShape([rows, outputColumns]),
            dataType: .float32,
            runtime: runtime
        )
        recordScheduledOperation(
            name: "edge_split_gated_query",
            byteCost: tensor.byteCount + query.byteCount + gate.byteCount
        )

        let commandBuffer = try makeCommandBufferForRecordedOperation()
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }

        encoder.setComputePipelineState(splitGatedQueryPipeline)
        encoder.setBuffer(tensor.buffer, offset: 0, index: 0)
        encoder.setBuffer(query.buffer, offset: 0, index: 1)
        encoder.setBuffer(gate.buffer, offset: 0, index: 2)
        retainResources(tensor.buffer, query.buffer, gate.buffer, for: commandBuffer)

        var rows32 = UInt32(rows)
        var headCount32 = UInt32(headCount)
        var headDimension32 = UInt32(headDimension)
        encoder.setBytes(&rows32, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&headCount32, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&headDimension32, length: MemoryLayout<UInt32>.stride, index: 5)

        let totalThreads = rows * outputColumns
        let threadExecutionWidth = splitGatedQueryPipeline.threadExecutionWidth
        let threadsPerGroup = MTLSize(
            width: min(max(threadExecutionWidth, 1), 256),
            height: 1,
            depth: 1
        )
        let gridSize = MTLSize(width: totalThreads, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        runtime.finishOperationCommandBuffer(commandBuffer)

        return (query, gate)
    }

    public func applyRoPE(
        _ tensor: EdgeTensor,
        headCount: Int,
        headDimension: Int,
        rotaryDimension: Int,
        base: Float,
        scale: Float = 1,
        offset: Int = 0
    ) throws -> EdgeTensor {
        guard tensor.dataType == .float32 else {
            throw MetalKernelExecutorError.dtypeMismatch
        }
        guard tensor.shape.rank == 2 else {
            throw MetalKernelExecutorError.invalidMatrixRank
        }
        guard headCount > 0,
              headDimension > 0,
              rotaryDimension > 0,
              rotaryDimension <= headDimension,
              base > 0
        else {
            throw MetalKernelExecutorError.invalidRoPEConfiguration(
                rotaryDimension: rotaryDimension,
                headDimension: headDimension,
                base: base
            )
        }

        let tokens = tensor.shape.dimensions[0]
        let columns = tensor.shape.dimensions[1]
        guard columns == headCount * headDimension else {
            throw MetalKernelExecutorError.matrixDimensionMismatch
        }

        let output = try EdgeTensor(
            shape: tensor.shape,
            dataType: .float32,
            runtime: runtime
        )
        recordScheduledOperation(
            name: "edge_rope",
            byteCost: tensor.byteCount + output.byteCount
        )

        let commandBuffer = try makeCommandBufferForRecordedOperation()
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }

        encoder.setComputePipelineState(ropePipeline)
        encoder.setBuffer(tensor.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)
        retainResources(tensor.buffer, output.buffer, for: commandBuffer)

        var tokens32 = UInt32(tokens)
        var headCount32 = UInt32(headCount)
        var headDimension32 = UInt32(headDimension)
        var rotaryDimension32 = UInt32(rotaryDimension)
        var base32 = base
        var scale32 = scale
        var offset32 = UInt32(offset)
        encoder.setBytes(&tokens32, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&headCount32, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&headDimension32, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&rotaryDimension32, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&base32, length: MemoryLayout<Float>.stride, index: 6)
        encoder.setBytes(&scale32, length: MemoryLayout<Float>.stride, index: 7)
        encoder.setBytes(&offset32, length: MemoryLayout<UInt32>.stride, index: 8)

        let totalThreads = tokens * headCount * headDimension
        let threadExecutionWidth = ropePipeline.threadExecutionWidth
        let threadsPerGroup = MTLSize(
            width: min(max(threadExecutionWidth, 1), 256),
            height: 1,
            depth: 1
        )
        let gridSize = MTLSize(width: totalThreads, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        runtime.finishOperationCommandBuffer(commandBuffer)

        return output
    }

    public func rmsNorm(
        _ tensor: EdgeTensor,
        weight: EdgeTensor,
        epsilon: Float = 1e-6
    ) throws -> EdgeTensor {
        guard tensor.dataType == .float32, weight.dataType == .float32 else {
            throw MetalKernelExecutorError.dtypeMismatch
        }
        guard tensor.shape.rank == 2, weight.shape.rank == 1 else {
            throw MetalKernelExecutorError.invalidMatrixRank
        }

        let rows = tensor.shape.dimensions[0]
        let columns = tensor.shape.dimensions[1]
        let weightCount = weight.shape.dimensions[0]
        guard columns == weightCount else {
            throw MetalKernelExecutorError.invalidRMSNorm(columns: columns, weightCount: weightCount)
        }

        let output = try EdgeTensor(
            shape: tensor.shape,
            dataType: .float32,
            runtime: runtime
        )
        if runtime.configuration.useCmlxFastRMSNorm,
           EdgeMLXBridge.supportsFastRMSNormFloat32MTL(rows: rows, columns: columns) {
            recordScheduledOperation(
                name: "edge_cmlx_fast_rms_norm",
                byteCost: tensor.byteCount + weight.byteCount + output.byteCount
            )
            let commandBuffer = try makeCommandBufferForRecordedOperation()
            try EdgeMLXBridge.encodeFastRMSNormFloat32MTL(
                commandBuffer: commandBuffer,
                inputBuffer: tensor.buffer,
                rows: rows,
                columns: columns,
                weightBuffer: weight.buffer,
                epsilon: epsilon,
                outputBuffer: output.buffer
            )
            retainResources(tensor.buffer, weight.buffer, output.buffer, for: commandBuffer)
            runtime.finishOperationCommandBuffer(commandBuffer)
            return output
        }
        recordScheduledOperation(
            name: "edge_rms_norm",
            byteCost: tensor.byteCount + weight.byteCount + output.byteCount
        )

        let commandBuffer = try makeCommandBufferForRecordedOperation()
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }

        encoder.setComputePipelineState(rmsNormPipeline)
        encoder.setBuffer(tensor.buffer, offset: 0, index: 0)
        encoder.setBuffer(weight.buffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        retainResources(tensor.buffer, weight.buffer, output.buffer, for: commandBuffer)

        var rows32 = UInt32(rows)
        var columns32 = UInt32(columns)
        var epsilon32 = epsilon
        encoder.setBytes(&rows32, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&columns32, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&epsilon32, length: MemoryLayout<Float>.stride, index: 5)

        let totalThreads = rows * columns
        let threadExecutionWidth = rmsNormPipeline.threadExecutionWidth
        let threadsPerGroup = MTLSize(
            width: min(max(threadExecutionWidth, 1), 256),
            height: 1,
            depth: 1
        )
        let gridSize = MTLSize(width: totalThreads, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        runtime.finishOperationCommandBuffer(commandBuffer)

        return output
    }

    public func rmsNormByHead(
        _ tensor: EdgeTensor,
        weight: EdgeTensor,
        headCount: Int,
        headDimension: Int,
        epsilon: Float = 1e-6
    ) throws -> EdgeTensor {
        guard tensor.dataType == .float32, weight.dataType == .float32 else {
            throw MetalKernelExecutorError.dtypeMismatch
        }
        guard tensor.shape.rank == 2, weight.shape.rank == 1 else {
            throw MetalKernelExecutorError.invalidMatrixRank
        }

        let rows = tensor.shape.dimensions[0]
        let columns = tensor.shape.dimensions[1]
        let weightCount = weight.shape.dimensions[0]
        guard headCount > 0,
              headDimension > 0,
              columns == headCount * headDimension,
              weightCount == headDimension
        else {
            throw MetalKernelExecutorError.invalidRMSNormByHead(
                headCount: headCount,
                headDimension: headDimension,
                columns: columns,
                weightCount: weightCount
            )
        }

        let output = try EdgeTensor(
            shape: tensor.shape,
            dataType: .float32,
            runtime: runtime
        )
        recordScheduledOperation(
            name: "edge_rms_norm_by_head",
            byteCost: tensor.byteCount + weight.byteCount + output.byteCount
        )

        let commandBuffer = try makeCommandBufferForRecordedOperation()
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }

        encoder.setComputePipelineState(rmsNormByHeadPipeline)
        encoder.setBuffer(tensor.buffer, offset: 0, index: 0)
        encoder.setBuffer(weight.buffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        retainResources(tensor.buffer, weight.buffer, output.buffer, for: commandBuffer)

        var rows32 = UInt32(rows)
        var headCount32 = UInt32(headCount)
        var headDimension32 = UInt32(headDimension)
        var epsilon32 = epsilon
        encoder.setBytes(&rows32, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&headCount32, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&headDimension32, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&epsilon32, length: MemoryLayout<Float>.stride, index: 6)

        let totalThreads = rows * columns
        let threadExecutionWidth = rmsNormByHeadPipeline.threadExecutionWidth
        let threadsPerGroup = MTLSize(
            width: min(max(threadExecutionWidth, 1), 256),
            height: 1,
            depth: 1
        )
        let gridSize = MTLSize(width: totalThreads, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        runtime.finishOperationCommandBuffer(commandBuffer)

        return output
    }

    public func sigmoidMultiply(_ tensor: EdgeTensor, gate: EdgeTensor) throws -> EdgeTensor {
        guard tensor.dataType == .float32, gate.dataType == .float32 else {
            throw MetalKernelExecutorError.dtypeMismatch
        }
        guard tensor.shape.dimensions == gate.shape.dimensions else {
            throw MetalKernelExecutorError.elementwiseShapeMismatch(
                lhs: tensor.shape.dimensions,
                rhs: gate.shape.dimensions
            )
        }

        let output = try EdgeTensor(
            shape: tensor.shape,
            dataType: .float32,
            runtime: runtime
        )
        recordScheduledOperation(
            name: "edge_sigmoid_multiply",
            byteCost: tensor.byteCount + gate.byteCount + output.byteCount
        )

        let commandBuffer = try makeCommandBufferForRecordedOperation()
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }

        encoder.setComputePipelineState(sigmoidMultiplyPipeline)
        encoder.setBuffer(tensor.buffer, offset: 0, index: 0)
        encoder.setBuffer(gate.buffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        retainResources(tensor.buffer, gate.buffer, output.buffer, for: commandBuffer)

        var elementCount32 = UInt32(tensor.shape.elementCount)
        encoder.setBytes(&elementCount32, length: MemoryLayout<UInt32>.stride, index: 3)

        let threadExecutionWidth = sigmoidMultiplyPipeline.threadExecutionWidth
        let threadsPerGroup = MTLSize(
            width: min(max(threadExecutionWidth, 1), 256),
            height: 1,
            depth: 1
        )
        let gridSize = MTLSize(width: tensor.shape.elementCount, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        runtime.finishOperationCommandBuffer(commandBuffer)

        return output
    }

    public func siluMultiply(
        gate: EdgeTensor,
        up: EdgeTensor,
        diagnosticSink: ((String) -> Void)? = nil,
        diagnosticName: String = "silu_multiply"
    ) throws -> EdgeTensor {
        guard gate.dataType == .float32, up.dataType == .float32 else {
            throw MetalKernelExecutorError.dtypeMismatch
        }
        guard gate.shape.dimensions == up.shape.dimensions else {
            throw MetalKernelExecutorError.elementwiseShapeMismatch(
                lhs: gate.shape.dimensions,
                rhs: up.shape.dimensions
            )
        }

        diagnosticSink?("\(diagnosticName)_output_alloc_begin bytes=\(gate.byteCount)")
        let output = try EdgeTensor(
            shape: gate.shape,
            dataType: .float32,
            runtime: runtime
        )
        diagnosticSink?("\(diagnosticName)_output_alloc_done")
        diagnosticSink?("\(diagnosticName)_schedule_begin")
        recordScheduledOperation(
            name: "edge_silu_multiply",
            byteCost: gate.byteCount + up.byteCount + output.byteCount
        )
        diagnosticSink?("\(diagnosticName)_schedule_done")

        diagnosticSink?("\(diagnosticName)_command_buffer_begin")
        let commandBuffer = try makeCommandBufferForRecordedOperation()
        diagnosticSink?("\(diagnosticName)_command_buffer_done")
        diagnosticSink?("\(diagnosticName)_encoder_begin")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }
        diagnosticSink?("\(diagnosticName)_encoder_done")

        encoder.setComputePipelineState(siluMultiplyPipeline)
        encoder.setBuffer(gate.buffer, offset: 0, index: 0)
        encoder.setBuffer(up.buffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        retainResources(gate.buffer, up.buffer, output.buffer, for: commandBuffer)

        var elementCount32 = UInt32(gate.shape.elementCount)
        encoder.setBytes(&elementCount32, length: MemoryLayout<UInt32>.stride, index: 3)

        let threadExecutionWidth = siluMultiplyPipeline.threadExecutionWidth
        let threadsPerGroup = MTLSize(
            width: min(max(threadExecutionWidth, 1), 256),
            height: 1,
            depth: 1
        )
        let gridSize = MTLSize(width: gate.shape.elementCount, height: 1, depth: 1)

        diagnosticSink?("\(diagnosticName)_dispatch_begin elements=\(gate.shape.elementCount)")
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        diagnosticSink?("\(diagnosticName)_dispatch_done")
        diagnosticSink?("\(diagnosticName)_finish_begin")
        runtime.finishOperationCommandBuffer(commandBuffer)
        diagnosticSink?("\(diagnosticName)_finish_done")

        return output
    }

    public func gdnDepthwiseConv1D(
        input: EdgeTensor,
        weights: EdgeTensor,
        convState: EdgeTensor,
        diagnosticSink: ((String) -> Void)? = nil,
        diagnosticName: String? = nil
    ) throws -> (activated: EdgeTensor, nextConvState: EdgeTensor) {
        guard input.dataType == .float32,
              weights.dataType == .float32,
              convState.dataType == .float32
        else {
            throw MetalKernelExecutorError.dtypeMismatch
        }
        guard input.shape.rank == 2,
              weights.shape.rank == 3,
              convState.shape.rank == 2
        else {
            throw MetalKernelExecutorError.invalidMatrixRank
        }

        let tokenCount = input.shape.dimensions[0]
        let channelCount = input.shape.dimensions[1]
        let kernelSize = weights.shape.dimensions[1]
        let stateTokenCount = kernelSize - 1
        guard tokenCount > 0,
              channelCount > 0,
              kernelSize > 1,
              weights.shape.dimensions == [channelCount, kernelSize, 1],
              convState.shape.dimensions == [stateTokenCount, channelCount]
        else {
            throw MetalKernelExecutorError.invalidGDNDepthwiseConvShape(
                input: input.shape.dimensions,
                weights: weights.shape.dimensions,
                convState: convState.shape.dimensions
            )
        }

        let activated = try EdgeTensor(
            shape: input.shape,
            dataType: .float32,
            runtime: runtime
        )
        let nextConvState = try EdgeTensor(
            shape: convState.shape,
            dataType: .float32,
            runtime: runtime
        )
        recordScheduledOperation(
            name: "edge_gdn_depthwise_conv1d",
            byteCost: input.byteCount + weights.byteCount + convState.byteCount
                + activated.byteCount + nextConvState.byteCount
        )

        let commandBuffer = try makeCommandBufferForRecordedOperation(
            diagnosticSink: diagnosticSink,
            diagnosticName: diagnosticName
        )
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }

        encoder.setComputePipelineState(gdnDepthwiseConv1DPipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(weights.buffer, offset: 0, index: 1)
        encoder.setBuffer(convState.buffer, offset: 0, index: 2)
        encoder.setBuffer(activated.buffer, offset: 0, index: 3)
        encoder.setBuffer(nextConvState.buffer, offset: 0, index: 4)
        retainResources(
            input.buffer,
            weights.buffer,
            convState.buffer,
            activated.buffer,
            nextConvState.buffer,
            for: commandBuffer
        )

        var tokenCount32 = UInt32(tokenCount)
        var channelCount32 = UInt32(channelCount)
        var kernelSize32 = UInt32(kernelSize)
        var stateTokenCount32 = UInt32(stateTokenCount)
        encoder.setBytes(&tokenCount32, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&channelCount32, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&kernelSize32, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes(&stateTokenCount32, length: MemoryLayout<UInt32>.stride, index: 8)

        let totalThreads = max(tokenCount * channelCount, stateTokenCount * channelCount)
        let threadExecutionWidth = gdnDepthwiseConv1DPipeline.threadExecutionWidth
        let threadsPerGroup = MTLSize(
            width: min(max(threadExecutionWidth, 1), 256),
            height: 1,
            depth: 1
        )
        let gridSize = MTLSize(width: totalThreads, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        runtime.finishOperationCommandBuffer(commandBuffer)

        return (activated, nextConvState)
    }

    public func gdnRecurrentUpdate(
        query: EdgeTensor,
        key: EdgeTensor,
        value: EdgeTensor,
        a: EdgeTensor,
        b: EdgeTensor,
        aLog: EdgeTensor,
        dtBias: EdgeTensor,
        state: EdgeTensor,
        keyHeadCount: Int,
        valueHeadCount: Int,
        keyHeadDimension: Int,
        valueHeadDimension: Int,
        diagnosticSink: ((String) -> Void)? = nil,
        diagnosticName: String? = nil
    ) throws -> (output: EdgeTensor, nextState: EdgeTensor) {
        guard query.dataType == .float32,
              key.dataType == .float32,
              value.dataType == .float32,
              a.dataType == .float32,
              b.dataType == .float32,
              aLog.dataType == .float32,
              dtBias.dataType == .float32,
              state.dataType == .float32
        else {
            throw MetalKernelExecutorError.dtypeMismatch
        }
        guard query.shape.rank == 2,
              key.shape.rank == 2,
              value.shape.rank == 2,
              a.shape.rank == 2,
              b.shape.rank == 2,
              aLog.shape.rank == 1,
              dtBias.shape.rank == 1,
              state.shape.rank == 3
        else {
            throw MetalKernelExecutorError.invalidMatrixRank
        }

        let tokenCount = query.shape.dimensions[0]
        let keyHiddenSize = keyHeadCount * keyHeadDimension
        let valueHiddenSize = valueHeadCount * valueHeadDimension
        guard tokenCount > 0,
              keyHeadCount > 0,
              valueHeadCount > 0,
              keyHeadDimension > 0,
              valueHeadDimension > 0,
              valueHeadCount % keyHeadCount == 0,
              query.shape.dimensions == [tokenCount, keyHiddenSize],
              key.shape.dimensions == [tokenCount, keyHiddenSize],
              value.shape.dimensions == [tokenCount, valueHiddenSize],
              a.shape.dimensions == [tokenCount, valueHeadCount],
              b.shape.dimensions == [tokenCount, valueHeadCount],
              aLog.shape.dimensions == [valueHeadCount],
              dtBias.shape.dimensions == [valueHeadCount],
              state.shape.dimensions == [valueHeadCount, valueHeadDimension, keyHeadDimension]
        else {
            throw MetalKernelExecutorError.invalidGDNRecurrentShape(
                query: query.shape.dimensions,
                key: key.shape.dimensions,
                value: value.shape.dimensions,
                a: a.shape.dimensions,
                b: b.shape.dimensions,
                aLog: aLog.shape.dimensions,
                dtBias: dtBias.shape.dimensions,
                state: state.shape.dimensions
            )
        }

        let output = try EdgeTensor(
            shape: EdgeTensorShape([tokenCount, valueHiddenSize]),
            dataType: .float32,
            runtime: runtime
        )
        let nextState = try EdgeTensor(
            shape: state.shape,
            dataType: .float32,
            runtime: runtime
        )
        recordScheduledOperation(
            name: "edge_gdn_recurrent_update",
            byteCost: query.byteCount + key.byteCount + value.byteCount + a.byteCount + b.byteCount
                + aLog.byteCount + dtBias.byteCount + state.byteCount + output.byteCount + nextState.byteCount
        )

        let commandBuffer = try makeCommandBufferForRecordedOperation(
            diagnosticSink: diagnosticSink,
            diagnosticName: diagnosticName
        )
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }

        encoder.setComputePipelineState(gdnRecurrentUpdatePipeline)
        encoder.setBuffer(query.buffer, offset: 0, index: 0)
        encoder.setBuffer(key.buffer, offset: 0, index: 1)
        encoder.setBuffer(value.buffer, offset: 0, index: 2)
        encoder.setBuffer(a.buffer, offset: 0, index: 3)
        encoder.setBuffer(b.buffer, offset: 0, index: 4)
        encoder.setBuffer(aLog.buffer, offset: 0, index: 5)
        encoder.setBuffer(dtBias.buffer, offset: 0, index: 6)
        encoder.setBuffer(state.buffer, offset: 0, index: 7)
        encoder.setBuffer(output.buffer, offset: 0, index: 8)
        encoder.setBuffer(nextState.buffer, offset: 0, index: 9)
        retainResources(
            query.buffer,
            key.buffer,
            value.buffer,
            a.buffer,
            b.buffer,
            aLog.buffer,
            dtBias.buffer,
            state.buffer,
            output.buffer,
            nextState.buffer,
            for: commandBuffer
        )

        var tokenCount32 = UInt32(tokenCount)
        var keyHeadCount32 = UInt32(keyHeadCount)
        var valueHeadCount32 = UInt32(valueHeadCount)
        var keyHeadDimension32 = UInt32(keyHeadDimension)
        var valueHeadDimension32 = UInt32(valueHeadDimension)
        encoder.setBytes(&tokenCount32, length: MemoryLayout<UInt32>.stride, index: 10)
        encoder.setBytes(&keyHeadCount32, length: MemoryLayout<UInt32>.stride, index: 11)
        encoder.setBytes(&valueHeadCount32, length: MemoryLayout<UInt32>.stride, index: 12)
        encoder.setBytes(&keyHeadDimension32, length: MemoryLayout<UInt32>.stride, index: 13)
        encoder.setBytes(&valueHeadDimension32, length: MemoryLayout<UInt32>.stride, index: 14)

        let totalThreads = valueHeadCount * valueHeadDimension
        let threadExecutionWidth = gdnRecurrentUpdatePipeline.threadExecutionWidth
        let threadsPerGroup = MTLSize(
            width: min(max(threadExecutionWidth, 1), 256),
            height: 1,
            depth: 1
        )
        let gridSize = MTLSize(width: totalThreads, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        runtime.finishOperationCommandBuffer(commandBuffer)

        return (output, nextState)
    }

    public func gdnSingleTokenFusedUpdate(
        mixedQKV: EdgeTensor,
        weights: EdgeTensor,
        convState: EdgeTensor,
        a: EdgeTensor,
        b: EdgeTensor,
        aLog: EdgeTensor,
        dtBias: EdgeTensor,
        recurrentState: EdgeTensor,
        keyHeadCount: Int,
        valueHeadCount: Int,
        keyHeadDimension: Int,
        valueHeadDimension: Int,
        epsilon: Float = 1e-6,
        diagnosticSink: ((String) -> Void)? = nil,
        diagnosticName: String? = nil
    ) throws -> (output: EdgeTensor, nextConvState: EdgeTensor, nextState: EdgeTensor) {
        guard mixedQKV.dataType == .float32,
              weights.dataType == .float32,
              convState.dataType == .float32,
              a.dataType == .float32,
              b.dataType == .float32,
              aLog.dataType == .float32,
              dtBias.dataType == .float32,
              recurrentState.dataType == .float32
        else {
            throw MetalKernelExecutorError.dtypeMismatch
        }
        guard mixedQKV.shape.rank == 2,
              weights.shape.rank == 3,
              convState.shape.rank == 2,
              a.shape.rank == 2,
              b.shape.rank == 2,
              aLog.shape.rank == 1,
              dtBias.shape.rank == 1,
              recurrentState.shape.rank == 3
        else {
            throw MetalKernelExecutorError.invalidMatrixRank
        }

        let tokenCount = mixedQKV.shape.dimensions[0]
        let keyHiddenSize = keyHeadCount * keyHeadDimension
        let valueHiddenSize = valueHeadCount * valueHeadDimension
        let channelCount = keyHiddenSize * 2 + valueHiddenSize
        let kernelSize = weights.shape.dimensions[1]
        let stateTokenCount = kernelSize - 1
        guard tokenCount == 1,
              keyHeadCount > 0,
              valueHeadCount > 0,
              keyHeadDimension > 0,
              valueHeadDimension > 0,
              valueHeadCount % keyHeadCount == 0,
              kernelSize > 1,
              mixedQKV.shape.dimensions == [1, channelCount],
              weights.shape.dimensions == [channelCount, kernelSize, 1],
              convState.shape.dimensions == [stateTokenCount, channelCount],
              a.shape.dimensions == [1, valueHeadCount],
              b.shape.dimensions == [1, valueHeadCount],
              aLog.shape.dimensions == [valueHeadCount],
              dtBias.shape.dimensions == [valueHeadCount],
              recurrentState.shape.dimensions == [valueHeadCount, valueHeadDimension, keyHeadDimension]
        else {
            throw MetalKernelExecutorError.invalidGDNSingleTokenFusedUpdateShape(
                mixedQKV: mixedQKV.shape.dimensions,
                weights: weights.shape.dimensions,
                convState: convState.shape.dimensions,
                a: a.shape.dimensions,
                b: b.shape.dimensions,
                aLog: aLog.shape.dimensions,
                dtBias: dtBias.shape.dimensions,
                state: recurrentState.shape.dimensions
            )
        }

        let output = try EdgeTensor(
            shape: EdgeTensorShape([1, valueHiddenSize]),
            dataType: .float32,
            runtime: runtime
        )
        let nextConvState = try EdgeTensor(
            shape: convState.shape,
            dataType: .float32,
            runtime: runtime
        )
        let nextState = try EdgeTensor(
            shape: recurrentState.shape,
            dataType: .float32,
            runtime: runtime
        )
        recordScheduledOperation(
            name: "edge_gdn_single_token_fused_update",
            byteCost: mixedQKV.byteCount + weights.byteCount + convState.byteCount
                + a.byteCount + b.byteCount + aLog.byteCount + dtBias.byteCount
                + recurrentState.byteCount + output.byteCount + nextConvState.byteCount
                + nextState.byteCount
        )

        let commandBuffer = try makeCommandBufferForRecordedOperation(
            diagnosticSink: diagnosticSink,
            diagnosticName: diagnosticName
        )
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }

        encoder.setComputePipelineState(gdnSingleTokenFusedUpdatePipeline)
        encoder.setBuffer(mixedQKV.buffer, offset: 0, index: 0)
        encoder.setBuffer(weights.buffer, offset: 0, index: 1)
        encoder.setBuffer(convState.buffer, offset: 0, index: 2)
        encoder.setBuffer(a.buffer, offset: 0, index: 3)
        encoder.setBuffer(b.buffer, offset: 0, index: 4)
        encoder.setBuffer(aLog.buffer, offset: 0, index: 5)
        encoder.setBuffer(dtBias.buffer, offset: 0, index: 6)
        encoder.setBuffer(recurrentState.buffer, offset: 0, index: 7)
        encoder.setBuffer(output.buffer, offset: 0, index: 8)
        encoder.setBuffer(nextConvState.buffer, offset: 0, index: 9)
        encoder.setBuffer(nextState.buffer, offset: 0, index: 10)
        retainResources(
            mixedQKV.buffer,
            weights.buffer,
            convState.buffer,
            a.buffer,
            b.buffer,
            aLog.buffer,
            dtBias.buffer,
            recurrentState.buffer,
            output.buffer,
            nextConvState.buffer,
            nextState.buffer,
            for: commandBuffer
        )

        var keyHeadCount32 = UInt32(keyHeadCount)
        var valueHeadCount32 = UInt32(valueHeadCount)
        var keyHeadDimension32 = UInt32(keyHeadDimension)
        var valueHeadDimension32 = UInt32(valueHeadDimension)
        var channelCount32 = UInt32(channelCount)
        var kernelSize32 = UInt32(kernelSize)
        var stateTokenCount32 = UInt32(stateTokenCount)
        var epsilon32 = epsilon
        encoder.setBytes(&keyHeadCount32, length: MemoryLayout<UInt32>.stride, index: 11)
        encoder.setBytes(&valueHeadCount32, length: MemoryLayout<UInt32>.stride, index: 12)
        encoder.setBytes(&keyHeadDimension32, length: MemoryLayout<UInt32>.stride, index: 13)
        encoder.setBytes(&valueHeadDimension32, length: MemoryLayout<UInt32>.stride, index: 14)
        encoder.setBytes(&channelCount32, length: MemoryLayout<UInt32>.stride, index: 15)
        encoder.setBytes(&kernelSize32, length: MemoryLayout<UInt32>.stride, index: 16)
        encoder.setBytes(&stateTokenCount32, length: MemoryLayout<UInt32>.stride, index: 17)
        encoder.setBytes(&epsilon32, length: MemoryLayout<Float>.stride, index: 18)

        let totalThreads = max(
            stateTokenCount * channelCount,
            valueHeadCount * valueHeadDimension
        )
        let threadExecutionWidth = gdnSingleTokenFusedUpdatePipeline.threadExecutionWidth
        let threadsPerGroup = MTLSize(
            width: min(max(threadExecutionWidth, 1), 256),
            height: 1,
            depth: 1
        )
        let gridSize = MTLSize(width: totalThreads, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        runtime.finishOperationCommandBuffer(commandBuffer)

        return (output, nextConvState, nextState)
    }

    public func gdnNormalizeQK(
        query: EdgeTensor,
        key: EdgeTensor,
        headCount: Int,
        headDimension: Int,
        epsilon: Float = 1e-6,
        diagnosticSink: ((String) -> Void)? = nil,
        diagnosticName: String? = nil
    ) throws -> (query: EdgeTensor, key: EdgeTensor) {
        guard query.dataType == .float32, key.dataType == .float32 else {
            throw MetalKernelExecutorError.dtypeMismatch
        }
        guard query.shape.rank == 2, key.shape.rank == 2 else {
            throw MetalKernelExecutorError.invalidMatrixRank
        }

        let tokenCount = query.shape.dimensions[0]
        let columns = headCount * headDimension
        guard tokenCount > 0,
              headCount > 0,
              headDimension > 0,
              query.shape.dimensions == [tokenCount, columns],
              key.shape.dimensions == [tokenCount, columns]
        else {
            throw MetalKernelExecutorError.invalidGDNQKNormalizationShape(
                query: query.shape.dimensions,
                key: key.shape.dimensions,
                headCount: headCount,
                headDimension: headDimension
            )
        }

        let normalizedQuery = try EdgeTensor(
            shape: query.shape,
            dataType: .float32,
            runtime: runtime
        )
        let normalizedKey = try EdgeTensor(
            shape: key.shape,
            dataType: .float32,
            runtime: runtime
        )
        recordScheduledOperation(
            name: "edge_gdn_normalize_qk",
            byteCost: query.byteCount + key.byteCount + normalizedQuery.byteCount + normalizedKey.byteCount
        )

        let commandBuffer = try makeCommandBufferForRecordedOperation(
            diagnosticSink: diagnosticSink,
            diagnosticName: diagnosticName
        )
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }

        encoder.setComputePipelineState(gdnQKNormalizationPipeline)
        encoder.setBuffer(query.buffer, offset: 0, index: 0)
        encoder.setBuffer(key.buffer, offset: 0, index: 1)
        encoder.setBuffer(normalizedQuery.buffer, offset: 0, index: 2)
        encoder.setBuffer(normalizedKey.buffer, offset: 0, index: 3)
        retainResources(
            query.buffer,
            key.buffer,
            normalizedQuery.buffer,
            normalizedKey.buffer,
            for: commandBuffer
        )

        var tokenCount32 = UInt32(tokenCount)
        var headCount32 = UInt32(headCount)
        var headDimension32 = UInt32(headDimension)
        var epsilon32 = epsilon
        encoder.setBytes(&tokenCount32, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&headCount32, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&headDimension32, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&epsilon32, length: MemoryLayout<Float>.stride, index: 7)

        let totalThreads = tokenCount * columns
        let threadExecutionWidth = gdnQKNormalizationPipeline.threadExecutionWidth
        let threadsPerGroup = MTLSize(
            width: min(max(threadExecutionWidth, 1), 256),
            height: 1,
            depth: 1
        )
        let gridSize = MTLSize(width: totalThreads, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        runtime.finishOperationCommandBuffer(commandBuffer)

        return (normalizedQuery, normalizedKey)
    }

    public func add(
        _ lhs: EdgeTensor,
        _ rhs: EdgeTensor,
        diagnosticSink: ((String) -> Void)? = nil,
        diagnosticName: String = "add"
    ) throws -> EdgeTensor {
        guard lhs.dataType == .float32, rhs.dataType == .float32 else {
            throw MetalKernelExecutorError.dtypeMismatch
        }
        guard lhs.shape.dimensions == rhs.shape.dimensions else {
            throw MetalKernelExecutorError.elementwiseShapeMismatch(
                lhs: lhs.shape.dimensions,
                rhs: rhs.shape.dimensions
            )
        }

        diagnosticSink?("\(diagnosticName)_output_alloc_begin bytes=\(lhs.byteCount)")
        let output = try EdgeTensor(
            shape: lhs.shape,
            dataType: .float32,
            runtime: runtime
        )
        diagnosticSink?("\(diagnosticName)_output_alloc_done")
        diagnosticSink?("\(diagnosticName)_schedule_begin")
        recordScheduledOperation(
            name: "edge_add",
            byteCost: lhs.byteCount + rhs.byteCount + output.byteCount
        )
        diagnosticSink?("\(diagnosticName)_schedule_done")

        diagnosticSink?("\(diagnosticName)_command_buffer_begin")
        let commandBuffer = try makeCommandBufferForRecordedOperation()
        diagnosticSink?("\(diagnosticName)_command_buffer_done")
        diagnosticSink?("\(diagnosticName)_encoder_begin")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }
        diagnosticSink?("\(diagnosticName)_encoder_done")

        encoder.setComputePipelineState(addPipeline)
        encoder.setBuffer(lhs.buffer, offset: 0, index: 0)
        encoder.setBuffer(rhs.buffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        retainResources(lhs.buffer, rhs.buffer, output.buffer, for: commandBuffer)

        var elementCount32 = UInt32(lhs.shape.elementCount)
        encoder.setBytes(&elementCount32, length: MemoryLayout<UInt32>.stride, index: 3)

        let threadExecutionWidth = addPipeline.threadExecutionWidth
        let threadsPerGroup = MTLSize(
            width: min(max(threadExecutionWidth, 1), 256),
            height: 1,
            depth: 1
        )
        let gridSize = MTLSize(width: lhs.shape.elementCount, height: 1, depth: 1)

        diagnosticSink?("\(diagnosticName)_dispatch_begin elements=\(lhs.shape.elementCount)")
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        diagnosticSink?("\(diagnosticName)_dispatch_done")
        diagnosticSink?("\(diagnosticName)_finish_begin")
        runtime.finishOperationCommandBuffer(commandBuffer)
        diagnosticSink?("\(diagnosticName)_finish_done")

        return output
    }

    public func embeddingLookup(tokenIds: [Int], embeddings: EdgeTensor) throws -> EdgeTensor {
        guard embeddings.dataType == .float32 else {
            throw MetalKernelExecutorError.dtypeMismatch
        }
        let vocabularySize = embeddings.shape.rank == 2 ? embeddings.shape.dimensions[0] : 0
        let hiddenSize = embeddings.shape.rank == 2 ? embeddings.shape.dimensions[1] : 0
        guard embeddings.shape.rank == 2,
              !tokenIds.isEmpty,
              vocabularySize > 0,
              hiddenSize > 0,
              vocabularySize <= Int(UInt32.max) / hiddenSize,
              hiddenSize <= Int(UInt32.max),
              tokenIds.count <= Int(UInt32.max) / hiddenSize,
              tokenIds.allSatisfy({ $0 >= 0 && $0 < vocabularySize })
        else {
            throw MetalKernelExecutorError.invalidEmbeddingLookup(
                tokenIds: tokenIds,
                embeddings: embeddings.shape.dimensions
            )
        }

        let tokenIdValues = tokenIds.map(UInt32.init)
        let tokenIdByteCount = tokenIdValues.count * MemoryLayout<UInt32>.stride
        guard let tokenIdBuffer = tokenIdValues.withUnsafeBytes({ bytes in
            runtime.device.makeBuffer(
                bytes: bytes.baseAddress!,
                length: tokenIdByteCount,
                options: [.storageModeShared]
            )
        }) else {
            throw EdgeTensorError.bufferAllocationFailed(byteCount: tokenIdByteCount)
        }

        let output = try EdgeTensor(
            shape: EdgeTensorShape([tokenIds.count, hiddenSize]),
            dataType: .float32,
            runtime: runtime
        )
        recordScheduledOperation(
            name: "edge_embedding_lookup",
            byteCost: embeddings.byteCount + tokenIdByteCount + output.byteCount
        )

        let commandBuffer = try makeCommandBufferForRecordedOperation()
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }

        encoder.setComputePipelineState(embeddingLookupPipeline)
        encoder.setBuffer(embeddings.buffer, offset: 0, index: 0)
        encoder.setBuffer(tokenIdBuffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        retainResources(embeddings.buffer, tokenIdBuffer, output.buffer, for: commandBuffer)

        var tokenCount32 = UInt32(tokenIds.count)
        var hiddenSize32 = UInt32(hiddenSize)
        encoder.setBytes(&tokenCount32, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&hiddenSize32, length: MemoryLayout<UInt32>.stride, index: 4)

        let totalThreads = output.shape.elementCount
        let threadExecutionWidth = embeddingLookupPipeline.threadExecutionWidth
        let threadsPerGroup = MTLSize(
            width: min(max(threadExecutionWidth, 1), 256),
            height: 1,
            depth: 1
        )
        let gridSize = MTLSize(width: totalThreads, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        runtime.finishOperationCommandBuffer(commandBuffer)

        return output
    }

    public func affineQuantizedEmbeddingLookup(
        tokenIds: [Int],
        embeddings: EdgeQuantizedTensor
    ) throws -> EdgeTensor {
        let vocabularySize = embeddings.shape.count == 2 ? embeddings.shape[0] : 0
        let hiddenSize = embeddings.shape.count == 2 ? embeddings.shape[1] : 0
        guard embeddings.shape.count == 2,
              !tokenIds.isEmpty,
              vocabularySize > 0,
              hiddenSize > 0,
              vocabularySize <= Int(UInt32.max) / hiddenSize,
              hiddenSize <= Int(UInt32.max),
              tokenIds.count <= Int(UInt32.max) / hiddenSize,
              tokenIds.allSatisfy({ $0 >= 0 && $0 < vocabularySize })
        else {
            throw MetalKernelExecutorError.invalidEmbeddingLookup(
                tokenIds: tokenIds,
                embeddings: embeddings.shape
            )
        }

        let tokenIdValues = tokenIds.map(UInt32.init)
        let tokenIdByteCount = tokenIdValues.count * MemoryLayout<UInt32>.stride
        guard let tokenIdBuffer = tokenIdValues.withUnsafeBytes({ bytes in
            runtime.device.makeBuffer(
                bytes: bytes.baseAddress!,
                length: tokenIdByteCount,
                options: [.storageModeShared]
            )
        }) else {
            throw EdgeTensorError.bufferAllocationFailed(byteCount: tokenIdByteCount)
        }

        let output = try EdgeTensor(
            shape: EdgeTensorShape([tokenIds.count, hiddenSize]),
            dataType: .float32,
            runtime: runtime
        )
        let packedByteCount = embeddings.packedByteCount
        let scalesByteCount = embeddings.scalesByteCount
        let biasesByteCount = embeddings.affineBiasBufferByteCount
        recordScheduledOperation(
            name: "edge_affine_quantized_embedding_lookup",
            byteCost: packedByteCount + scalesByteCount + biasesByteCount + tokenIdByteCount + output.byteCount
        )

        let embeddingBuffers = try affineQuantizedMetalBuffers(for: embeddings, preferNoCopy: true)
        guard embeddingBuffers.packedOffset.isMultiple(of: MemoryLayout<UInt32>.stride) else {
            throw EdgeTensorError.bufferAllocationFailed(byteCount: packedByteCount)
        }
        var packedWordOffset32 = UInt32(embeddingBuffers.packedOffset / MemoryLayout<UInt32>.stride)

        let commandBuffer = try makeCommandBufferForRecordedOperation()
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }

        encoder.setComputePipelineState(affineQuantizedEmbeddingLookupPipeline)
        encoder.setBuffer(embeddingBuffers.packedBuffer, offset: 0, index: 0)
        encoder.setBuffer(embeddingBuffers.scalesBuffer, offset: embeddingBuffers.scalesOffset, index: 1)
        encoder.setBuffer(embeddingBuffers.biasesBuffer, offset: embeddingBuffers.biasesOffset, index: 2)
        encoder.setBuffer(tokenIdBuffer, offset: 0, index: 3)
        encoder.setBuffer(output.buffer, offset: 0, index: 4)
        retainResources(
            embeddingBuffers.packedBuffer,
            embeddingBuffers.scalesBuffer,
            embeddingBuffers.biasesBuffer,
            tokenIdBuffer,
            output.buffer,
            for: commandBuffer
        )

        var tokenCount32 = UInt32(tokenIds.count)
        var hiddenSize32 = UInt32(hiddenSize)
        var packedWordsPerRow32 = UInt32(embeddings.packedWordsPerLogicalRow)
        var scaleColumns32 = UInt32(embeddings.scaleShape.last!)
        var groupSize32 = UInt32(embeddings.groupSize)
        var bits32 = UInt32(embeddings.bits)
        encoder.setBytes(&tokenCount32, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&hiddenSize32, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&packedWordsPerRow32, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes(&scaleColumns32, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.setBytes(&groupSize32, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.setBytes(&bits32, length: MemoryLayout<UInt32>.stride, index: 10)
        encoder.setBytes(&packedWordOffset32, length: MemoryLayout<UInt32>.stride, index: 11)

        let totalThreads = output.shape.elementCount
        let threadExecutionWidth = affineQuantizedEmbeddingLookupPipeline.threadExecutionWidth
        let threadsPerGroup = MTLSize(
            width: min(max(threadExecutionWidth, 1), 256),
            height: 1,
            depth: 1
        )
        let gridSize = MTLSize(width: totalThreads, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        runtime.finishOperationCommandBuffer(commandBuffer)

        return output
    }

    public func scaledDotProductAttention(
        query: EdgeTensor,
        key: EdgeTensor,
        value: EdgeTensor,
        queryHeadCount: Int,
        keyValueHeadCount: Int,
        headDimension: Int,
        causal: Bool = true,
        scale: Float? = nil
    ) throws -> EdgeTensor {
        try scaledDotProductAttention(
            query: query,
            key: key,
            value: value,
            keyValueTokenCount: nil,
            queryPositionOffset: 0,
            queryHeadCount: queryHeadCount,
            keyValueHeadCount: keyValueHeadCount,
            headDimension: headDimension,
            causal: causal,
            scale: scale
        )
    }

    public func scaledDotProductAttention(
        query: EdgeTensor,
        key: EdgeTensor,
        value: EdgeTensor,
        keyValueTokenCount: Int? = nil,
        queryPositionOffset: Int = 0,
        queryHeadCount: Int,
        keyValueHeadCount: Int,
        headDimension: Int,
        causal: Bool = true,
        scale: Float? = nil
    ) throws -> EdgeTensor {
        guard query.dataType == .float32, key.dataType == .float32, value.dataType == .float32 else {
            throw MetalKernelExecutorError.dtypeMismatch
        }
        let queryTokenCount = query.shape.rank == 2 ? query.shape.dimensions[0] : 0
        let keyCapacity = key.shape.rank == 2 ? key.shape.dimensions[0] : 0
        let valueCapacity = value.shape.rank == 2 ? value.shape.dimensions[0] : 0
        let activeKeyValueTokenCount = keyValueTokenCount ?? keyCapacity
        guard query.shape.rank == 2,
              key.shape.rank == 2,
              value.shape.rank == 2,
              queryTokenCount > 0,
              activeKeyValueTokenCount > 0,
              queryPositionOffset >= 0,
              activeKeyValueTokenCount <= keyCapacity,
              activeKeyValueTokenCount <= valueCapacity,
              queryHeadCount > 0,
              keyValueHeadCount > 0,
              headDimension > 0,
              queryHeadCount % keyValueHeadCount == 0,
              query.shape.dimensions[1] == queryHeadCount * headDimension,
              key.shape.dimensions[1] == keyValueHeadCount * headDimension,
              value.shape.dimensions[1] == keyValueHeadCount * headDimension
        else {
            throw MetalKernelExecutorError.invalidAttentionShape(
                query: query.shape.dimensions,
                key: key.shape.dimensions,
                value: value.shape.dimensions,
                queryHeadCount: queryHeadCount,
                keyValueHeadCount: keyValueHeadCount,
                headDimension: headDimension
            )
        }

        let output = try EdgeTensor(
            shape: query.shape,
            dataType: .float32,
            runtime: runtime
        )
        recordScheduledOperation(
            name: "edge_scaled_dot_product_attention",
            byteCost: query.byteCount + key.byteCount + value.byteCount + output.byteCount
        )

        let commandBuffer = try makeCommandBufferForRecordedOperation()
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }

        encoder.setComputePipelineState(scaledDotProductAttentionPipeline)
        encoder.setBuffer(query.buffer, offset: 0, index: 0)
        encoder.setBuffer(key.buffer, offset: 0, index: 1)
        encoder.setBuffer(value.buffer, offset: 0, index: 2)
        encoder.setBuffer(output.buffer, offset: 0, index: 3)
        retainResources(query.buffer, key.buffer, value.buffer, output.buffer, for: commandBuffer)

        var queryTokenCount32 = UInt32(queryTokenCount)
        var keyValueTokenCount32 = UInt32(activeKeyValueTokenCount)
        var queryPositionOffset32 = UInt32(queryPositionOffset)
        var queryHeadCount32 = UInt32(queryHeadCount)
        var keyValueHeadCount32 = UInt32(keyValueHeadCount)
        var headDimension32 = UInt32(headDimension)
        var causal32 = causal ? UInt32(1) : UInt32(0)
        var scale32 = scale ?? (1.0 / Float(Double(headDimension).squareRoot()))
        encoder.setBytes(&queryTokenCount32, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&keyValueTokenCount32, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&queryPositionOffset32, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&queryHeadCount32, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes(&keyValueHeadCount32, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.setBytes(&headDimension32, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.setBytes(&causal32, length: MemoryLayout<UInt32>.stride, index: 10)
        encoder.setBytes(&scale32, length: MemoryLayout<Float>.stride, index: 11)

        let totalThreads = query.shape.elementCount
        let threadExecutionWidth = scaledDotProductAttentionPipeline.threadExecutionWidth
        let threadsPerGroup = MTLSize(
            width: min(max(threadExecutionWidth, 1), 256),
            height: 1,
            depth: 1
        )
        let gridSize = MTLSize(width: totalThreads, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        runtime.finishOperationCommandBuffer(commandBuffer)

        return output
    }

    public func copyRowsToPrefix(
        source: EdgeTensor,
        destination: EdgeTensor,
        startRow: Int
    ) throws {
        guard source.dataType == .float32, destination.dataType == .float32 else {
            throw MetalKernelExecutorError.dtypeMismatch
        }
        let sourceRows = source.shape.rank == 2 ? source.shape.dimensions[0] : 0
        let destinationRows = destination.shape.rank == 2 ? destination.shape.dimensions[0] : 0
        let columns = source.shape.rank == 2 ? source.shape.dimensions[1] : 0
        guard source.shape.rank == 2,
              destination.shape.rank == 2,
              sourceRows > 0,
              startRow >= 0,
              source.shape.dimensions[1] == destination.shape.dimensions[1],
              startRow + sourceRows <= destinationRows
        else {
            throw MetalKernelExecutorError.invalidRowCopy(
                source: source.shape.dimensions,
                destination: destination.shape.dimensions,
                startRow: startRow
            )
        }

        recordScheduledOperation(
            name: "edge_copy_rows_to_prefix",
            byteCost: source.byteCount + destination.byteCount
        )

        let commandBuffer = try makeCommandBufferForRecordedOperation()
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }

        encoder.setComputePipelineState(copyRowsToPrefixPipeline)
        encoder.setBuffer(source.buffer, offset: 0, index: 0)
        encoder.setBuffer(destination.buffer, offset: 0, index: 1)
        retainResources(source.buffer, destination.buffer, for: commandBuffer)

        var sourceRows32 = UInt32(sourceRows)
        var columns32 = UInt32(columns)
        var startRow32 = UInt32(startRow)
        encoder.setBytes(&sourceRows32, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&columns32, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&startRow32, length: MemoryLayout<UInt32>.stride, index: 4)

        let totalThreads = source.shape.elementCount
        let threadExecutionWidth = copyRowsToPrefixPipeline.threadExecutionWidth
        let threadsPerGroup = MTLSize(
            width: min(max(threadExecutionWidth, 1), 256),
            height: 1,
            depth: 1
        )
        let gridSize = MTLSize(width: totalThreads, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        runtime.finishOperationCommandBuffer(commandBuffer)
    }

    public func gatherRows(
        source: EdgeTensor,
        rowIndices: [Int]
    ) throws -> EdgeTensor {
        guard source.dataType == .float32 else {
            throw MetalKernelExecutorError.dtypeMismatch
        }
        let sourceRows = source.shape.rank == 2 ? source.shape.dimensions[0] : 0
        let columns = source.shape.rank == 2 ? source.shape.dimensions[1] : 0
        guard source.shape.rank == 2,
              sourceRows > 0,
              !rowIndices.isEmpty,
              rowIndices.allSatisfy({ $0 >= 0 && $0 < sourceRows })
        else {
            throw MetalKernelExecutorError.invalidRowGather(
                source: source.shape.dimensions,
                rowIndices: rowIndices
            )
        }

        let output = try EdgeTensor(
            shape: EdgeTensorShape([rowIndices.count, columns]),
            dataType: .float32,
            runtime: runtime
        )
        let indexValues = rowIndices.map(UInt32.init)
        let indexByteCount = indexValues.count * MemoryLayout<UInt32>.stride
        guard let indexBuffer = indexValues.withUnsafeBytes({
            runtime.device.makeBuffer(
                bytes: $0.baseAddress!,
                length: indexByteCount,
                options: [.storageModeShared]
            )
        }) else {
            throw EdgeTensorError.bufferAllocationFailed(byteCount: indexByteCount)
        }

        recordScheduledOperation(
            name: "edge_gather_rows",
            byteCost: source.byteCount + output.byteCount + indexByteCount
        )

        let commandBuffer = try makeCommandBufferForRecordedOperation()
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }

        encoder.setComputePipelineState(gatherRowsPipeline)
        encoder.setBuffer(source.buffer, offset: 0, index: 0)
        encoder.setBuffer(indexBuffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        retainResources(source.buffer, indexBuffer, output.buffer, for: commandBuffer)

        var outputRows32 = UInt32(rowIndices.count)
        var columns32 = UInt32(columns)
        encoder.setBytes(&outputRows32, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&columns32, length: MemoryLayout<UInt32>.stride, index: 4)

        let totalThreads = output.shape.elementCount
        let threadExecutionWidth = gatherRowsPipeline.threadExecutionWidth
        let threadsPerGroup = MTLSize(
            width: min(max(threadExecutionWidth, 1), 256),
            height: 1,
            depth: 1
        )
        let gridSize = MTLSize(width: totalThreads, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        runtime.finishOperationCommandBuffer(commandBuffer)

        return output
    }

    public func updateAttentionScoreEMA(
        query: EdgeTensor,
        key: EdgeTensor,
        scores: EdgeTensor,
        keyValueTokenCount: Int,
        queryHeadCount: Int,
        keyValueHeadCount: Int,
        headDimension: Int,
        scale: Float? = nil,
        decay: Float = 0.95,
        hasExistingScores: Bool
    ) throws {
        guard query.dataType == .float32,
              key.dataType == .float32,
              scores.dataType == .float32
        else {
            throw MetalKernelExecutorError.dtypeMismatch
        }
        let queryTokenCount = query.shape.rank == 2 ? query.shape.dimensions[0] : 0
        let keyCapacity = key.shape.rank == 2 ? key.shape.dimensions[0] : 0
        let scoreCapacity = scores.shape.rank == 2 ? scores.shape.dimensions[0] : 0
        guard query.shape.rank == 2,
              key.shape.rank == 2,
              scores.shape.rank == 2,
              queryTokenCount == 1,
              keyValueTokenCount > 0,
              keyValueTokenCount <= keyCapacity,
              keyValueTokenCount <= scoreCapacity,
              queryHeadCount > 0,
              keyValueHeadCount > 0,
              headDimension > 0,
              queryHeadCount % keyValueHeadCount == 0,
              query.shape.dimensions[1] == queryHeadCount * headDimension,
              key.shape.dimensions[1] == keyValueHeadCount * headDimension,
              scores.shape.dimensions[1] == keyValueHeadCount
        else {
            throw MetalKernelExecutorError.invalidAttentionScoreShape(
                query: query.shape.dimensions,
                key: key.shape.dimensions,
                scores: scores.shape.dimensions,
                keyValueTokenCount: keyValueTokenCount,
                queryHeadCount: queryHeadCount,
                keyValueHeadCount: keyValueHeadCount,
                headDimension: headDimension
            )
        }

        recordScheduledOperation(
            name: "edge_update_attention_score_ema",
            byteCost: query.byteCount + key.byteCount + scores.byteCount
        )

        let commandBuffer = try makeCommandBufferForRecordedOperation()
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }

        encoder.setComputePipelineState(updateAttentionScoreEMAPipeline)
        encoder.setBuffer(query.buffer, offset: 0, index: 0)
        encoder.setBuffer(key.buffer, offset: 0, index: 1)
        encoder.setBuffer(scores.buffer, offset: 0, index: 2)
        retainResources(query.buffer, key.buffer, scores.buffer, for: commandBuffer)

        var keyValueTokenCount32 = UInt32(keyValueTokenCount)
        var queryHeadCount32 = UInt32(queryHeadCount)
        var keyValueHeadCount32 = UInt32(keyValueHeadCount)
        var headDimension32 = UInt32(headDimension)
        var scale32 = scale ?? (1.0 / Float(Double(headDimension).squareRoot()))
        var decay32 = decay
        var hasExistingScores32 = hasExistingScores ? UInt32(1) : UInt32(0)
        encoder.setBytes(&keyValueTokenCount32, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&queryHeadCount32, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&keyValueHeadCount32, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&headDimension32, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&scale32, length: MemoryLayout<Float>.stride, index: 7)
        encoder.setBytes(&decay32, length: MemoryLayout<Float>.stride, index: 8)
        encoder.setBytes(&hasExistingScores32, length: MemoryLayout<UInt32>.stride, index: 9)

        let totalThreads = keyValueTokenCount * keyValueHeadCount
        let threadExecutionWidth = updateAttentionScoreEMAPipeline.threadExecutionWidth
        let threadsPerGroup = MTLSize(
            width: min(max(threadExecutionWidth, 1), 256),
            height: 1,
            depth: 1
        )
        let gridSize = MTLSize(width: totalThreads, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        runtime.finishOperationCommandBuffer(commandBuffer)
    }

    public func copyTensor(source: EdgeTensor, destination: EdgeTensor) throws {
        guard source.dataType == destination.dataType else {
            throw MetalKernelExecutorError.dtypeMismatch
        }
        guard source.shape.dimensions == destination.shape.dimensions else {
            throw MetalKernelExecutorError.elementwiseShapeMismatch(
                lhs: source.shape.dimensions,
                rhs: destination.shape.dimensions
            )
        }

        recordScheduledOperation(
            name: "edge_copy_tensor",
            byteCost: source.byteCount + destination.byteCount
        )

        let commandBuffer = try makeCommandBufferForRecordedOperation()
        guard let encoder = commandBuffer.makeBlitCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }
        encoder.copy(
            from: source.buffer,
            sourceOffset: 0,
            to: destination.buffer,
            destinationOffset: 0,
            size: source.byteCount
        )
        encoder.endEncoding()
        runtime.finishOperationCommandBuffer(commandBuffer)
    }

    public func argmaxLastRow(_ logits: EdgeTensor) throws -> QwenGreedyToken {
        let scratch = try makeArgmaxLastRowScratch()
        return try argmaxLastRow(logits, scratch: scratch)
    }

    func argmaxLastRow(
        _ logits: EdgeTensor,
        scratch: MetalArgmaxLastRowScratch
    ) throws -> QwenGreedyToken {
        guard logits.dataType == .float32 else {
            throw MetalKernelExecutorError.dtypeMismatch
        }
        guard logits.shape.rank == 2,
              logits.shape.dimensions[0] > 0,
              logits.shape.dimensions[1] > 0
        else {
            throw MetalKernelExecutorError.invalidMatrixRank
        }

        let rows = logits.shape.dimensions[0]
        let columns = logits.shape.dimensions[1]
        recordScheduledOperation(
            name: "edge_argmax_last_row",
            byteCost: logits.byteCount + MemoryLayout<UInt32>.stride + MemoryLayout<Float>.stride
        )

        let commandBuffer = try makeCommandBufferForRecordedOperation()
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalKernelExecutorError.commandEncoderCreationFailed
        }

        encoder.setComputePipelineState(argmaxLastRowPipeline)
        encoder.setBuffer(logits.buffer, offset: 0, index: 0)
        encoder.setBuffer(scratch.tokenBuffer, offset: 0, index: 1)
        encoder.setBuffer(scratch.logitBuffer, offset: 0, index: 2)
        retainResources(
            logits.buffer,
            scratch.tokenBuffer,
            scratch.logitBuffer,
            for: commandBuffer
        )

        var rows32 = UInt32(rows)
        var columns32 = UInt32(columns)
        encoder.setBytes(&rows32, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&columns32, length: MemoryLayout<UInt32>.stride, index: 4)

        var threadCount = min(256, max(1, argmaxLastRowPipeline.maxTotalThreadsPerThreadgroup))
        while threadCount > 1 && (threadCount & (threadCount - 1)) != 0 {
            threadCount -= 1
        }
        let threadsPerGroup = MTLSize(width: threadCount, height: 1, depth: 1)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: threadsPerGroup
        )
        encoder.endEncoding()
        runtime.finishOperationCommandBuffer(commandBuffer)
        runtime.waitForPendingWork()

        let tokenId = scratch.tokenBuffer.contents().load(as: UInt32.self)
        let logit = scratch.logitBuffer.contents().load(as: Float.self)
        return QwenGreedyToken(tokenId: Int(tokenId), logit: logit)
    }

    private func affineQuantizedMetalBuffers(
        for weights: EdgeQuantizedTensor,
        preferNoCopy: Bool = true
    ) throws -> AffineQuantizedMetalBuffers {
        let cacheKey = AffineQuantizedBufferCacheKey(weights)
        if let cached = affineQuantizedBufferCache[cacheKey] {
            affineQuantizedBufferCacheStats.hitCount += 1
            affineQuantizedBufferCacheStats.entryCount = affineQuantizedBufferCache.count
            return cached
        }

        let packedByteCount = weights.packedByteCount
        let scalesByteCount = weights.scalesByteCount
        let biasesByteCount = weights.affineBiasBufferByteCount

        let totalByteCount = packedByteCount + scalesByteCount + biasesByteCount
        let shouldCacheEntry = runtime.configuration.allowsQuantizedBufferCacheEntry(
            byteCount: totalByteCount,
            currentCachedByteCount: affineQuantizedBufferCacheStats.cachedByteCount
        )

        let uploadedBuffers = try weights.withHostStorageByteStorage { packedData, scalesData, biasesData in
            var hostRetainers: [Data] = []
            var usesNoCopyBuffer = false

            let packedBuffer = try makeQuantizedInputBuffer(
                storage: packedData,
                byteCount: packedByteCount,
                preferNoCopy: preferNoCopy
                    && shouldCacheEntry
                    && runtime.configuration.quantizedNoCopyBuffersEnabled,
                hostRetainers: &hostRetainers,
                usesNoCopyBuffer: &usesNoCopyBuffer
            )
            let scalesBuffer = try makeQuantizedInputBuffer(
                data: scalesData,
                byteCount: scalesByteCount,
                preferNoCopy: false,
                hostRetainers: &hostRetainers,
                usesNoCopyBuffer: &usesNoCopyBuffer
            )
            let biasesBuffer: QuantizedInputMetalBuffer
            if let biasesData {
                biasesBuffer = try makeQuantizedInputBuffer(
                    data: biasesData,
                    byteCount: biasesByteCount,
                    preferNoCopy: false,
                    hostRetainers: &hostRetainers,
                    usesNoCopyBuffer: &usesNoCopyBuffer
                )
            } else {
                guard let buffer = runtime.device.makeBuffer(
                    length: biasesByteCount,
                    options: [.storageModeShared]
                ) else {
                    throw EdgeTensorError.bufferAllocationFailed(byteCount: biasesByteCount)
                }
                memset(buffer.contents(), 0, biasesByteCount)
                biasesBuffer = QuantizedInputMetalBuffer(buffer: buffer, offset: 0)
            }
            return (packedBuffer, scalesBuffer, biasesBuffer, hostRetainers, usesNoCopyBuffer)
        }

        let buffers = AffineQuantizedMetalBuffers(
            packedBuffer: uploadedBuffers.0.buffer,
            packedOffset: uploadedBuffers.0.offset,
            scalesBuffer: uploadedBuffers.1.buffer,
            scalesOffset: uploadedBuffers.1.offset,
            biasesBuffer: uploadedBuffers.2.buffer,
            biasesOffset: uploadedBuffers.2.offset,
            hostRetainers: uploadedBuffers.3
        )
        if shouldCacheEntry {
            affineQuantizedBufferCache[cacheKey] = buffers
            affineQuantizedBufferCacheStats.cachedByteCount += totalByteCount
            if runtime.configuration.releaseQuantizedHostStorageAfterUpload && !uploadedBuffers.4 {
                let releasedByteCount = weights.releaseHostStorage()
                if releasedByteCount > 0 {
                    affineQuantizedBufferCacheStats.releasedHostStorageByteCount += releasedByteCount
                    affineQuantizedBufferCacheStats.releasedHostStorageCount += 1
                }
            }
        }
        affineQuantizedBufferCacheStats.missCount += 1
        affineQuantizedBufferCacheStats.entryCount = affineQuantizedBufferCache.count
        affineQuantizedBufferCacheStats.uploadedByteCount += totalByteCount
        return buffers
    }

    func sharedAffineQuantizedMetalBuffers(
        for weights: EdgeQuantizedTensor,
        preferNoCopy: Bool = true
    ) throws -> AffineQuantizedMetalBuffers {
        try affineQuantizedMetalBuffers(for: weights, preferNoCopy: preferNoCopy)
    }

    private func makeQuantizedInputBuffer(
        storage: EdgeQuantizedByteStorage,
        byteCount: Int,
        preferNoCopy: Bool,
        hostRetainers: inout [Data],
        usesNoCopyBuffer: inout Bool
    ) throws -> QuantizedInputMetalBuffer {
        if preferNoCopy,
           let backing = storage.pageAlignedNoCopyDataRange(
               pageSize: Int(getpagesize()),
               offsetAlignment: MemoryLayout<UInt32>.stride
           ) {
            if let buffer = makeNoCopyBuffer(
                data: backing.data,
                range: backing.range,
                minimumByteCount: byteCount + backing.offset
            ) {
                hostRetainers.append(backing.data)
                usesNoCopyBuffer = true
                return QuantizedInputMetalBuffer(buffer: buffer, offset: backing.offset)
            }
        }

        guard let buffer = storage.withUnsafeBytes({ bytes in
            runtime.device.makeBuffer(
                bytes: bytes.baseAddress!,
                length: byteCount,
                options: [.storageModeShared]
            )
        }) else {
            throw EdgeTensorError.bufferAllocationFailed(byteCount: byteCount)
        }
        return QuantizedInputMetalBuffer(buffer: buffer, offset: 0)
    }

    private func makeQuantizedInputBuffer(
        data: Data,
        byteCount: Int,
        preferNoCopy: Bool,
        hostRetainers: inout [Data],
        usesNoCopyBuffer: inout Bool
    ) throws -> QuantizedInputMetalBuffer {
        if preferNoCopy,
           let buffer = makeNoCopyBuffer(
               data: data,
               range: 0..<data.count,
               minimumByteCount: byteCount
           ) {
            hostRetainers.append(data)
            usesNoCopyBuffer = true
            return QuantizedInputMetalBuffer(buffer: buffer, offset: 0)
        }

        guard let buffer = data.withUnsafeBytes({ bytes in
            runtime.device.makeBuffer(
                bytes: bytes.baseAddress!,
                length: byteCount,
                options: [.storageModeShared]
            )
        }) else {
            throw EdgeTensorError.bufferAllocationFailed(byteCount: byteCount)
        }
        return QuantizedInputMetalBuffer(buffer: buffer, offset: 0)
    }

    private func makeNoCopyBuffer(
        data: Data,
        range: Range<Int>,
        minimumByteCount: Int
    ) -> MTLBuffer? {
        guard range.count >= minimumByteCount else {
            return nil
        }
        return data.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress else {
                return nil
            }
            let pointer = UnsafeMutableRawPointer(
                mutating: baseAddress.advanced(by: range.lowerBound)
            )
            return runtime.device.makeBuffer(
                bytesNoCopy: pointer,
                length: range.count,
                options: [.storageModeShared],
                deallocator: nil
            )
        }
    }

    private func recordScheduledOperation(name: String, byteCost: Int) {
        lastScheduledByteCost = max(0, byteCost)
        schedulingSnapshot.updateConfiguration(runtime.configuration)
        let precommitted = schedulingSnapshot.recordOperation(byteCost: byteCost)
        lastExecutionStats = MetalKernelExecutionStats(
            operationName: name,
            scheduledByteCost: byteCost,
            precommittedBeforeEncoding: precommitted,
            effectiveMaxOpsPerCommandBuffer: schedulingSnapshot.effectiveMaxOps,
            pendingOpsAfterRecord: schedulingSnapshot.pendingOps,
            pendingBytesAfterRecord: schedulingSnapshot.pendingBytes,
            logicalCommitCount: schedulingSnapshot.commitCount
        )
    }

    private func makeCommandBufferForRecordedOperation(
        diagnosticSink: ((String) -> Void)? = nil,
        diagnosticName: String? = nil
    ) throws -> MTLCommandBuffer {
        guard let commandBuffer = runtime.commandBufferForOperation(byteCost: lastScheduledByteCost) else {
            throw MetalKernelExecutorError.commandBufferCreationFailed
        }
        if let diagnosticSink, let diagnosticName, let stats = runtime.lastCommandBufferAcquisitionStats {
            diagnosticSink(
                "\(diagnosticName)_command_buffer backpressure=\(stats.waitedForBackpressure ? 1 : 0) "
                    + "waitNs=\(stats.backpressureWaitNanoseconds) "
                    + "committed=\(stats.committedActiveBuffer ? 1 : 0) "
                    + "submittedBefore=\(stats.submittedBeforeWait) "
                    + "submittedAfter=\(stats.submittedAfterWait) "
                    + "maxInflight=\(stats.maxInFlightCommandBuffers) "
                    + "activeOps=\(stats.activeOperationCountAfterRecord) "
                    + "activeBytes=\(stats.activeByteCountAfterRecord)"
            )
        }
        return commandBuffer
    }

    private func retainResources(_ resources: AnyObject..., for commandBuffer: MTLCommandBuffer) {
        runtime.retainResources(resources, for: commandBuffer)
    }

    private static let kernelSource = """
    #include <metal_stdlib>
    #include <metal_simdgroup>
    using namespace metal;

    kernel void edge_fp32_matmul(
        device const float* lhs [[buffer(0)]],
        device const float* rhs [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& inner [[buffer(4)]],
        constant uint& columns [[buffer(5)]],
        uint gid [[thread_position_in_grid]]
    ) {
        uint total = rows * columns;
        if (gid >= total) {
            return;
        }

        uint row = gid / columns;
        uint column = gid - row * columns;
        float acc = 0.0f;
        for (uint index = 0; index < inner; ++index) {
            acc += lhs[row * inner + index] * rhs[index * columns + column];
        }
        output[gid] = acc;
    }

    kernel void edge_q4_matmul(
        device const float* lhs [[buffer(0)]],
        device const uchar* packedWeights [[buffer(1)]],
        device const float* scales [[buffer(2)]],
        device float* output [[buffer(3)]],
        constant uint& rows [[buffer(4)]],
        constant uint& inner [[buffer(5)]],
        constant uint& columns [[buffer(6)]],
        constant uint& groupSize [[buffer(7)]],
        uint gid [[thread_position_in_grid]]
    ) {
        uint total = rows * columns;
        if (gid >= total) {
            return;
        }

        uint row = gid / columns;
        uint column = gid - row * columns;
        float acc = 0.0f;
        for (uint index = 0; index < inner; ++index) {
            uint weightIndex = index * columns + column;
            uchar packed = packedWeights[weightIndex / 2];
            uchar nibble = (weightIndex % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
            int signedValue = (nibble >= 8) ? (int(nibble) - 16) : int(nibble);
            float weight = float(signedValue) * scales[weightIndex / groupSize];
            acc += lhs[row * inner + index] * weight;
        }
        output[gid] = acc;
    }

    kernel void edge_affine_quantized_matmul(
        device const float* lhs [[buffer(0)]],
        device const uint* packedWeights [[buffer(1)]],
        device const float* scales [[buffer(2)]],
        device const float* biases [[buffer(3)]],
        device float* output [[buffer(4)]],
        constant uint& rows [[buffer(5)]],
        constant uint& inner [[buffer(6)]],
        constant uint& columns [[buffer(7)]],
        constant uint& packedWordsPerRow [[buffer(8)]],
        constant uint& scaleColumns [[buffer(9)]],
        constant uint& groupSize [[buffer(10)]],
        constant uint& bits [[buffer(11)]],
        constant uint& transpose [[buffer(12)]],
        uint gid [[thread_position_in_grid]]
    ) {
        uint total = rows * columns;
        if (gid >= total) {
            return;
        }

        uint row = gid / columns;
        uint column = gid - row * columns;
        uint mask = (1u << bits) - 1u;
        float acc = 0.0f;
        for (uint index = 0; index < inner; ++index) {
            uint logicalRow = (transpose == 0u) ? index : column;
            uint logicalColumn = (transpose == 0u) ? column : index;
            uint bitOffset = logicalColumn * bits;
            uint packedIndex = logicalRow * packedWordsPerRow + bitOffset / 32u;
            uint shift = bitOffset % 32u;
            uint raw = packedWeights[packedIndex] >> shift;
            if (shift + bits > 32u) {
                raw |= packedWeights[packedIndex + 1u] << (32u - shift);
            }
            raw &= mask;

            uint scaleIndex = logicalRow * scaleColumns + logicalColumn / groupSize;
            float weight = float(raw) * scales[scaleIndex] + biases[scaleIndex];
            acc += lhs[row * inner + index] * weight;
        }
        output[gid] = acc;
    }

    inline float edge_qmv_load4(device const float* x, thread float* xThread) {
        float sum = 0.0f;
        for (uint i = 0u; i < 16u; i += 4u) {
            float x0 = x[i];
            float x1 = x[i + 1u];
            float x2 = x[i + 2u];
            float x3 = x[i + 3u];
            sum += x0 + x1 + x2 + x3;
            xThread[i] = x0;
            xThread[i + 1u] = x1 / 16.0f;
            xThread[i + 2u] = x2 / 256.0f;
            xThread[i + 3u] = x3 / 4096.0f;
        }
        return sum;
    }

    inline float edge_qmv_load6(device const float* x, thread float* xThread) {
        float sum = 0.0f;
        for (uint i = 0u; i < 8u; i += 4u) {
            float x0 = x[i];
            float x1 = x[i + 1u];
            float x2 = x[i + 2u];
            float x3 = x[i + 3u];
            sum += x0 + x1 + x2 + x3;
            xThread[i] = x0;
            xThread[i + 1u] = x1 / 64.0f;
            xThread[i + 2u] = x2 / 16.0f;
            xThread[i + 3u] = x3 / 4.0f;
        }
        return sum;
    }

    inline float edge_qmv_qdot4(
        device const uchar* weights,
        thread const float* xThread,
        float scale,
        float bias,
        float sum
    ) {
        device const ushort* packed = reinterpret_cast<device const ushort*>(weights);
        float accum = 0.0f;
        for (uint i = 0u; i < 4u; ++i) {
            ushort word = packed[i];
            uint base = 4u * i;
            accum += xThread[base] * float(word & 0x000fu);
            accum += xThread[base + 1u] * float(word & 0x00f0u);
            accum += xThread[base + 2u] * float(word & 0x0f00u);
            accum += xThread[base + 3u] * float(word & 0xf000u);
        }
        return scale * accum + sum * bias;
    }

    inline float edge_qmv_qdot6(
        device const uchar* weights,
        thread const float* xThread,
        float scale,
        float bias,
        float sum
    ) {
        float accum = 0.0f;
        for (uint pack = 0u; pack < 2u; ++pack) {
            device const uchar* w = weights + 3u * pack;
            thread const float* x = xThread + 4u * pack;
            accum += float(w[0] & 0x3fu) * x[0];
            accum += float(w[0] & 0xc0u) * x[1];
            accum += float(w[1] & 0x0fu) * (x[1] * 256.0f);
            accum += float(w[1] & 0xf0u) * x[2];
            accum += float(w[2] & 0x03u) * (x[2] * 256.0f);
            accum += float(w[2] & 0xfcu) * x[3];
        }
        return scale * accum + sum * bias;
    }

    inline uint edge_load_affine_packed_value_bytes(
        device const uchar* packedWeights,
        uint packedWordsPerRow,
        uint logicalRow,
        uint logicalColumn,
        uint bits
    ) {
        uint bitOffset = logicalColumn * bits;
        uint rowByteOffset = logicalRow * packedWordsPerRow * 4u;
        uint byteOffset = rowByteOffset + bitOffset / 8u;
        uint shift = bitOffset % 8u;
        uint rowByteEnd = rowByteOffset + packedWordsPerRow * 4u;
        uint raw = uint(packedWeights[byteOffset]);
        if (byteOffset + 1u < rowByteEnd) {
            raw |= uint(packedWeights[byteOffset + 1u]) << 8u;
        }
        return (raw >> shift) & ((1u << bits) - 1u);
    }

    // Adapted from MLX (Apple Inc., MIT License).
    // Modified by AtomGradient for edge-engine's fp32, transposed affine qmv path.
    kernel void edge_affine_qmv_transposed(
        device const float* lhs [[buffer(0)]],
        device const uchar* packedWeights [[buffer(1)]],
        device const float* scales [[buffer(2)]],
        device const float* biases [[buffer(3)]],
        device float* output [[buffer(4)]],
        constant uint& rows [[buffer(5)]],
        constant uint& inner [[buffer(6)]],
        constant uint& columns [[buffer(7)]],
        constant uint& packedWordsPerRow [[buffer(8)]],
        constant uint& scaleColumns [[buffer(9)]],
        constant uint& groupSize [[buffer(10)]],
        constant uint& bits [[buffer(11)]],
        constant uint& transpose [[buffer(12)]],
        uint3 tid [[threadgroup_position_in_grid]],
        uint simdGroupID [[simdgroup_index_in_threadgroup]],
        uint simdLaneID [[thread_index_in_simdgroup]]
    ) {
        if (tid.x >= rows || transpose == 0u || groupSize != 64u || (bits != 4u && bits != 6u)) {
            return;
        }

        constexpr uint resultsPerSimdgroup = 4;
        constexpr uint numSimdgroups = 2;
        uint outputBase = tid.y * 8u + simdGroupID * resultsPerSimdgroup;
        if (outputBase >= columns || simdGroupID >= numSimdgroups) {
            return;
        }

        uint fastBlockSize = (bits == 4u) ? 512u : 256u;
        if ((inner % fastBlockSize) != 0u) {
            uint valuesPerThread = (bits == 4u) ? 8u : 4u;
            uint safeBlockSize = valuesPerThread * 32u;
            float partial0 = 0.0f;
            float partial1 = 0.0f;
            float partial2 = 0.0f;
            float partial3 = 0.0f;

            for (uint block = 0u; block < inner; block += safeBlockSize) {
                uint baseIndex = block + simdLaneID * valuesPerThread;
                for (uint laneValue = 0u; laneValue < valuesPerThread; laneValue += 1u) {
                    uint inputColumn = baseIndex + laneValue;
                    if (inputColumn >= inner) {
                        continue;
                    }
                    float inputValue = lhs[tid.x * inner + inputColumn];
                    uint groupIndex = inputColumn / groupSize;

                    uint outputColumn0 = outputBase;
                    if (outputColumn0 < columns) {
                        uint scaleIndex = outputColumn0 * scaleColumns + groupIndex;
                        uint raw = edge_load_affine_packed_value_bytes(
                            packedWeights,
                            packedWordsPerRow,
                            outputColumn0,
                            inputColumn,
                            bits
                        );
                        partial0 += inputValue * (float(raw) * scales[scaleIndex] + biases[scaleIndex]);
                    }

                    uint outputColumn1 = outputBase + 1u;
                    if (outputColumn1 < columns) {
                        uint scaleIndex = outputColumn1 * scaleColumns + groupIndex;
                        uint raw = edge_load_affine_packed_value_bytes(
                            packedWeights,
                            packedWordsPerRow,
                            outputColumn1,
                            inputColumn,
                            bits
                        );
                        partial1 += inputValue * (float(raw) * scales[scaleIndex] + biases[scaleIndex]);
                    }

                    uint outputColumn2 = outputBase + 2u;
                    if (outputColumn2 < columns) {
                        uint scaleIndex = outputColumn2 * scaleColumns + groupIndex;
                        uint raw = edge_load_affine_packed_value_bytes(
                            packedWeights,
                            packedWordsPerRow,
                            outputColumn2,
                            inputColumn,
                            bits
                        );
                        partial2 += inputValue * (float(raw) * scales[scaleIndex] + biases[scaleIndex]);
                    }

                    uint outputColumn3 = outputBase + 3u;
                    if (outputColumn3 < columns) {
                        uint scaleIndex = outputColumn3 * scaleColumns + groupIndex;
                        uint raw = edge_load_affine_packed_value_bytes(
                            packedWeights,
                            packedWordsPerRow,
                            outputColumn3,
                            inputColumn,
                            bits
                        );
                        partial3 += inputValue * (float(raw) * scales[scaleIndex] + biases[scaleIndex]);
                    }
                }
            }

            partial0 = simd_sum(partial0);
            partial1 = simd_sum(partial1);
            partial2 = simd_sum(partial2);
            partial3 = simd_sum(partial3);

            if (simdLaneID == 0u) {
                uint outputRowOffset = tid.x * columns;
                uint outputColumn0 = outputBase;
                if (outputColumn0 < columns) {
                    output[outputRowOffset + outputColumn0] = partial0;
                }
                uint outputColumn1 = outputBase + 1u;
                if (outputColumn1 < columns) {
                    output[outputRowOffset + outputColumn1] = partial1;
                }
                uint outputColumn2 = outputBase + 2u;
                if (outputColumn2 < columns) {
                    output[outputRowOffset + outputColumn2] = partial2;
                }
                uint outputColumn3 = outputBase + 3u;
                if (outputColumn3 < columns) {
                    output[outputRowOffset + outputColumn3] = partial3;
                }
            }
            return;
        }

        thread float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        device const uchar* packedBase = packedWeights;

        if (bits == 4u) {
            constexpr uint valuesPerThread = 16;
            constexpr uint blockSize = valuesPerThread * 32u;
            constexpr uint bytesPerPack = 4;
            constexpr uint packFactor = 8;
            constexpr uint packsPerThread = 2;
            constexpr uint scaleStepPerThread = 4;

            if ((inner % blockSize) != 0u) {
                return;
            }

            uint inVecSizeBytes = inner * bytesPerPack / packFactor;
            device const uchar* weightsCursor = packedBase
                + outputBase * inVecSizeBytes
                + simdLaneID * packsPerThread * bytesPerPack;
            device const float* scalesCursor = scales
                + outputBase * scaleColumns
                + simdLaneID / scaleStepPerThread;
            device const float* biasesCursor = biases
                + outputBase * scaleColumns
                + simdLaneID / scaleStepPerThread;
            device const float* inputCursor = lhs + tid.x * inner + simdLaneID * valuesPerThread;
            thread float xThread[16];

            for (uint k = 0u; k < inner; k += blockSize) {
                float sum = edge_qmv_load4(inputCursor, xThread);
                for (uint row = 0u; row < resultsPerSimdgroup; ++row) {
                    uint outputColumn = outputBase + row;
                    if (outputColumn < columns) {
                        device const uchar* weightsRow = weightsCursor + row * inVecSizeBytes;
                        device const float* scalesRow = scalesCursor + row * scaleColumns;
                        device const float* biasesRow = biasesCursor + row * scaleColumns;
                        result[row] += edge_qmv_qdot4(weightsRow, xThread, scalesRow[0], biasesRow[0], sum);
                    }
                }
                weightsCursor += blockSize * bytesPerPack / packFactor;
                scalesCursor += blockSize / groupSize;
                biasesCursor += blockSize / groupSize;
                inputCursor += blockSize;
            }
        } else {
            constexpr uint valuesPerThread = 8;
            constexpr uint blockSize = valuesPerThread * 32u;
            constexpr uint bytesPerPack = 3;
            constexpr uint packFactor = 4;
            constexpr uint packsPerThread = 2;
            constexpr uint scaleStepPerThread = 8;

            if ((inner % blockSize) != 0u) {
                return;
            }

            uint inVecSizeBytes = inner * bytesPerPack / packFactor;
            device const uchar* weightsCursor = packedBase
                + outputBase * inVecSizeBytes
                + simdLaneID * packsPerThread * bytesPerPack;
            device const float* scalesCursor = scales
                + outputBase * scaleColumns
                + simdLaneID / scaleStepPerThread;
            device const float* biasesCursor = biases
                + outputBase * scaleColumns
                + simdLaneID / scaleStepPerThread;
            device const float* inputCursor = lhs + tid.x * inner + simdLaneID * valuesPerThread;
            thread float xThread[8];

            for (uint k = 0u; k < inner; k += blockSize) {
                float sum = edge_qmv_load6(inputCursor, xThread);
                for (uint row = 0u; row < resultsPerSimdgroup; ++row) {
                    uint outputColumn = outputBase + row;
                    if (outputColumn < columns) {
                        device const uchar* weightsRow = weightsCursor + row * inVecSizeBytes;
                        device const float* scalesRow = scalesCursor + row * scaleColumns;
                        device const float* biasesRow = biasesCursor + row * scaleColumns;
                        result[row] += edge_qmv_qdot6(weightsRow, xThread, scalesRow[0], biasesRow[0], sum);
                    }
                }
                weightsCursor += blockSize * bytesPerPack / packFactor;
                scalesCursor += blockSize / groupSize;
                biasesCursor += blockSize / groupSize;
                inputCursor += blockSize;
            }
        }

        for (uint row = 0u; row < resultsPerSimdgroup; ++row) {
            float reduced = simd_sum(result[row]);
            uint outputColumn = outputBase + row;
            if (simdLaneID == 0u && outputColumn < columns) {
                output[tid.x * columns + outputColumn] = reduced;
            }
        }
    }

    kernel void edge_split_columns(
        device const float* input [[buffer(0)]],
        device float* first [[buffer(1)]],
        device float* second [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& columns [[buffer(4)]],
        constant uint& firstColumns [[buffer(5)]],
        uint gid [[thread_position_in_grid]]
    ) {
        uint total = rows * columns;
        if (gid >= total) {
            return;
        }

        uint row = gid / columns;
        uint column = gid - row * columns;
        float value = input[gid];
        if (column < firstColumns) {
            first[row * firstColumns + column] = value;
        } else {
            uint secondColumns = columns - firstColumns;
            uint secondColumn = column - firstColumns;
            second[row * secondColumns + secondColumn] = value;
        }
    }

    kernel void edge_split_gated_query(
        device const float* input [[buffer(0)]],
        device float* query [[buffer(1)]],
        device float* gate [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& headCount [[buffer(4)]],
        constant uint& headDimension [[buffer(5)]],
        uint gid [[thread_position_in_grid]]
    ) {
        uint outputColumns = headCount * headDimension;
        uint total = rows * outputColumns;
        if (gid >= total) {
            return;
        }

        uint row = gid / outputColumns;
        uint flatColumn = gid - row * outputColumns;
        uint head = flatColumn / headDimension;
        uint headOffset = flatColumn - head * headDimension;
        uint inputColumns = outputColumns * 2u;
        uint inputHeadOffset = head * headDimension * 2u;
        uint inputBase = row * inputColumns + inputHeadOffset + headOffset;

        query[gid] = input[inputBase];
        gate[gid] = input[inputBase + headDimension];
    }

    kernel void edge_rope(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint& tokens [[buffer(2)]],
        constant uint& headCount [[buffer(3)]],
        constant uint& headDimension [[buffer(4)]],
        constant uint& rotaryDimension [[buffer(5)]],
        constant float& base [[buffer(6)]],
        constant float& positionScale [[buffer(7)]],
        constant uint& positionOffset [[buffer(8)]],
        uint gid [[thread_position_in_grid]]
    ) {
        uint total = tokens * headCount * headDimension;
        if (gid >= total) {
            return;
        }

        uint perToken = headCount * headDimension;
        uint token = gid / perToken;
        uint tokenRemainder = gid - token * perToken;
        uint head = tokenRemainder / headDimension;
        uint dimension = tokenRemainder - head * headDimension;
        uint rotaryHalf = rotaryDimension / 2u;

        if (rotaryHalf == 0u || dimension >= 2u * rotaryHalf) {
            output[gid] = input[gid];
            return;
        }

        uint headOffset = token * perToken + head * headDimension;
        uint pairIndex;
        uint x1Index;
        uint x2Index;
        bool firstHalf = dimension < rotaryHalf;
        if (firstHalf) {
            pairIndex = dimension;
            x1Index = headOffset + pairIndex;
            x2Index = headOffset + rotaryHalf + pairIndex;
        } else {
            pairIndex = dimension - rotaryHalf;
            x1Index = headOffset + pairIndex;
            x2Index = headOffset + rotaryHalf + pairIndex;
        }

        float position = float(positionOffset + token) * positionScale;
        float inverseFrequency = exp(-float(pairIndex) * log(base) / float(rotaryHalf));
        float theta = position * inverseFrequency;
        float cosine = cos(theta);
        float sine = sin(theta);
        float x1 = input[x1Index];
        float x2 = input[x2Index];
        output[gid] = firstHalf ? (x1 * cosine - x2 * sine) : (x1 * sine + x2 * cosine);
    }

    kernel void edge_rms_norm(
        device const float* input [[buffer(0)]],
        device const float* weight [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& columns [[buffer(4)]],
        constant float& epsilon [[buffer(5)]],
        uint gid [[thread_position_in_grid]]
    ) {
        uint total = rows * columns;
        if (gid >= total) {
            return;
        }

        uint row = gid / columns;
        uint column = gid - row * columns;
        uint rowBase = row * columns;

        float meanSquare = 0.0f;
        for (uint index = 0; index < columns; ++index) {
            float value = input[rowBase + index];
            meanSquare += value * value;
        }
        meanSquare /= float(columns);
        float scale = rsqrt(meanSquare + epsilon);
        output[gid] = input[gid] * scale * weight[column];
    }

    kernel void edge_rms_norm_by_head(
        device const float* input [[buffer(0)]],
        device const float* weight [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& headCount [[buffer(4)]],
        constant uint& headDimension [[buffer(5)]],
        constant float& epsilon [[buffer(6)]],
        uint gid [[thread_position_in_grid]]
    ) {
        uint columns = headCount * headDimension;
        uint total = rows * columns;
        if (gid >= total) {
            return;
        }

        uint row = gid / columns;
        uint flatColumn = gid - row * columns;
        uint head = flatColumn / headDimension;
        uint dimension = flatColumn - head * headDimension;
        uint headBase = row * columns + head * headDimension;

        float meanSquare = 0.0f;
        for (uint index = 0; index < headDimension; ++index) {
            float value = input[headBase + index];
            meanSquare += value * value;
        }
        meanSquare /= float(headDimension);
        float scale = rsqrt(meanSquare + epsilon);
        output[gid] = input[gid] * scale * weight[dimension];
    }

    kernel void edge_sigmoid_multiply(
        device const float* input [[buffer(0)]],
        device const float* gate [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& elementCount [[buffer(3)]],
        uint gid [[thread_position_in_grid]]
    ) {
        if (gid >= elementCount) {
            return;
        }

        float gateValue = gate[gid];
        float sigmoid = 1.0f / (1.0f + exp(-gateValue));
        output[gid] = input[gid] * sigmoid;
    }

    kernel void edge_silu_multiply(
        device const float* gate [[buffer(0)]],
        device const float* up [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& elementCount [[buffer(3)]],
        uint gid [[thread_position_in_grid]]
    ) {
        if (gid >= elementCount) {
            return;
        }

        float gateValue = gate[gid];
        float silu = gateValue / (1.0f + exp(-gateValue));
        output[gid] = silu * up[gid];
    }

    kernel void edge_gdn_depthwise_conv1d(
        device const float* input [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device const float* convState [[buffer(2)]],
        device float* activated [[buffer(3)]],
        device float* nextConvState [[buffer(4)]],
        constant uint& tokenCount [[buffer(5)]],
        constant uint& channelCount [[buffer(6)]],
        constant uint& kernelSize [[buffer(7)]],
        constant uint& stateTokenCount [[buffer(8)]],
        uint gid [[thread_position_in_grid]]
    ) {
        uint outputCount = tokenCount * channelCount;
        if (gid < outputCount) {
            uint token = gid / channelCount;
            uint channel = gid - token * channelCount;
            float acc = 0.0f;
            for (uint kernelIndex = 0; kernelIndex < kernelSize; ++kernelIndex) {
                uint concatenatedToken = token + kernelIndex;
                float value = 0.0f;
                if (concatenatedToken < stateTokenCount) {
                    value = convState[concatenatedToken * channelCount + channel];
                } else {
                    uint inputToken = concatenatedToken - stateTokenCount;
                    value = input[inputToken * channelCount + channel];
                }
                float weight = weights[channel * kernelSize + kernelIndex];
                acc += value * weight;
            }
            activated[gid] = acc / (1.0f + exp(-acc));
        }

        uint stateCount = stateTokenCount * channelCount;
        if (gid < stateCount) {
            uint stateToken = gid / channelCount;
            uint channel = gid - stateToken * channelCount;
            uint concatenatedToken = tokenCount + stateToken;
            if (concatenatedToken < stateTokenCount) {
                nextConvState[gid] = convState[concatenatedToken * channelCount + channel];
            } else {
                uint inputToken = concatenatedToken - stateTokenCount;
                nextConvState[gid] = input[inputToken * channelCount + channel];
            }
        }
    }

    float edge_gdn_softplus(float value) {
        if (value > 20.0f) {
            return value;
        }
        if (value < -20.0f) {
            return exp(value);
        }
        return log(1.0f + exp(value));
    }

    float edge_gdn_sigmoid(float value) {
        if (value >= 0.0f) {
            return 1.0f / (1.0f + exp(-value));
        }
        float expValue = exp(value);
        return expValue / (1.0f + expValue);
    }

    kernel void edge_gdn_normalize_qk(
        device const float* query [[buffer(0)]],
        device const float* key [[buffer(1)]],
        device float* normalizedQuery [[buffer(2)]],
        device float* normalizedKey [[buffer(3)]],
        constant uint& tokenCount [[buffer(4)]],
        constant uint& headCount [[buffer(5)]],
        constant uint& headDimension [[buffer(6)]],
        constant float& epsilon [[buffer(7)]],
        uint gid [[thread_position_in_grid]]
    ) {
        uint columns = headCount * headDimension;
        uint total = tokenCount * columns;
        if (gid >= total) {
            return;
        }

        uint token = gid / columns;
        uint flatColumn = gid - token * columns;
        uint head = flatColumn / headDimension;
        uint headBase = token * columns + head * headDimension;

        float queryMeanSquare = 0.0f;
        float keyMeanSquare = 0.0f;
        for (uint dimension = 0; dimension < headDimension; ++dimension) {
            float queryValue = query[headBase + dimension];
            float keyValue = key[headBase + dimension];
            queryMeanSquare += queryValue * queryValue;
            keyMeanSquare += keyValue * keyValue;
        }
        queryMeanSquare /= float(headDimension);
        keyMeanSquare /= float(headDimension);

        float invScale = rsqrt(float(headDimension));
        float queryScale = rsqrt(queryMeanSquare + epsilon) * invScale * invScale;
        float keyScale = rsqrt(keyMeanSquare + epsilon) * invScale;
        normalizedQuery[gid] = query[gid] * queryScale;
        normalizedKey[gid] = key[gid] * keyScale;
    }

    kernel void edge_gdn_recurrent_update(
        device const float* query [[buffer(0)]],
        device const float* key [[buffer(1)]],
        device const float* value [[buffer(2)]],
        device const float* a [[buffer(3)]],
        device const float* b [[buffer(4)]],
        device const float* aLog [[buffer(5)]],
        device const float* dtBias [[buffer(6)]],
        device const float* stateIn [[buffer(7)]],
        device float* output [[buffer(8)]],
        device float* stateOut [[buffer(9)]],
        constant uint& tokenCount [[buffer(10)]],
        constant uint& keyHeadCount [[buffer(11)]],
        constant uint& valueHeadCount [[buffer(12)]],
        constant uint& keyHeadDimension [[buffer(13)]],
        constant uint& valueHeadDimension [[buffer(14)]],
        uint gid [[thread_position_in_grid]]
    ) {
        uint totalRows = valueHeadCount * valueHeadDimension;
        if (gid >= totalRows) {
            return;
        }

        uint valueHead = gid / valueHeadDimension;
        uint valueDimension = gid - valueHead * valueHeadDimension;
        uint repeatFactor = valueHeadCount / keyHeadCount;
        uint keyHead = valueHead / repeatFactor;

        for (uint keyDimension = 0; keyDimension < keyHeadDimension; ++keyDimension) {
            uint stateIndex = (valueHead * valueHeadDimension + valueDimension) * keyHeadDimension
                + keyDimension;
            stateOut[stateIndex] = stateIn[stateIndex];
        }

        for (uint token = 0; token < tokenCount; ++token) {
            uint gateIndex = token * valueHeadCount + valueHead;
            float decay = exp(
                -exp(aLog[valueHead]) * edge_gdn_softplus(a[gateIndex] + dtBias[valueHead])
            );
            float beta = edge_gdn_sigmoid(b[gateIndex]);

            float kvMemory = 0.0f;
            for (uint keyDimension = 0; keyDimension < keyHeadDimension; ++keyDimension) {
                uint stateIndex = (valueHead * valueHeadDimension + valueDimension) * keyHeadDimension
                    + keyDimension;
                uint keyIndex = (token * keyHeadCount + keyHead) * keyHeadDimension + keyDimension;
                float decayedState = stateOut[stateIndex] * decay;
                stateOut[stateIndex] = decayedState;
                kvMemory += decayedState * key[keyIndex];
            }

            uint valueIndex = (token * valueHeadCount + valueHead) * valueHeadDimension + valueDimension;
            float delta = (value[valueIndex] - kvMemory) * beta;

            float projected = 0.0f;
            for (uint keyDimension = 0; keyDimension < keyHeadDimension; ++keyDimension) {
                uint stateIndex = (valueHead * valueHeadDimension + valueDimension) * keyHeadDimension
                    + keyDimension;
                uint keyIndex = (token * keyHeadCount + keyHead) * keyHeadDimension + keyDimension;
                uint queryIndex = (token * keyHeadCount + keyHead) * keyHeadDimension + keyDimension;
                float newState = stateOut[stateIndex] + key[keyIndex] * delta;
                stateOut[stateIndex] = newState;
                projected += newState * query[queryIndex];
            }
            output[valueIndex] = projected;
        }
    }

    float edge_gdn_single_token_conv_silu(
        device const float* mixedQKV,
        device const float* weights,
        device const float* convState,
        uint channel,
        uint channelCount,
        uint kernelSize,
        uint stateTokenCount
    ) {
        float acc = 0.0f;
        for (uint kernelIndex = 0; kernelIndex < kernelSize; ++kernelIndex) {
            float value = 0.0f;
            if (kernelIndex < stateTokenCount) {
                value = convState[kernelIndex * channelCount + channel];
            } else {
                value = mixedQKV[channel];
            }
            float weight = weights[channel * kernelSize + kernelIndex];
            acc += value * weight;
        }
        return acc / (1.0f + exp(-acc));
    }

    kernel void edge_gdn_single_token_fused_update(
        device const float* mixedQKV [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device const float* convState [[buffer(2)]],
        device const float* a [[buffer(3)]],
        device const float* b [[buffer(4)]],
        device const float* aLog [[buffer(5)]],
        device const float* dtBias [[buffer(6)]],
        device const float* stateIn [[buffer(7)]],
        device float* output [[buffer(8)]],
        device float* nextConvState [[buffer(9)]],
        device float* stateOut [[buffer(10)]],
        constant uint& keyHeadCount [[buffer(11)]],
        constant uint& valueHeadCount [[buffer(12)]],
        constant uint& keyHeadDimension [[buffer(13)]],
        constant uint& valueHeadDimension [[buffer(14)]],
        constant uint& channelCount [[buffer(15)]],
        constant uint& kernelSize [[buffer(16)]],
        constant uint& stateTokenCount [[buffer(17)]],
        constant float& epsilon [[buffer(18)]],
        uint gid [[thread_position_in_grid]]
    ) {
        uint stateCount = stateTokenCount * channelCount;
        if (gid < stateCount) {
            uint stateToken = gid / channelCount;
            uint channel = gid - stateToken * channelCount;
            uint concatenatedToken = 1 + stateToken;
            if (concatenatedToken < stateTokenCount) {
                nextConvState[gid] = convState[concatenatedToken * channelCount + channel];
            } else {
                nextConvState[gid] = mixedQKV[channel];
            }
        }

        uint totalRows = valueHeadCount * valueHeadDimension;
        if (gid >= totalRows) {
            return;
        }

        uint keyHiddenSize = keyHeadCount * keyHeadDimension;
        uint valueHead = gid / valueHeadDimension;
        uint valueDimension = gid - valueHead * valueHeadDimension;
        uint repeatFactor = valueHeadCount / keyHeadCount;
        uint keyHead = valueHead / repeatFactor;

        float queryMeanSquare = 0.0f;
        float keyMeanSquare = 0.0f;
        for (uint keyDimension = 0; keyDimension < keyHeadDimension; ++keyDimension) {
            uint qChannel = keyHead * keyHeadDimension + keyDimension;
            uint kChannel = keyHiddenSize + qChannel;
            float queryValue = edge_gdn_single_token_conv_silu(
                mixedQKV,
                weights,
                convState,
                qChannel,
                channelCount,
                kernelSize,
                stateTokenCount
            );
            float keyValue = edge_gdn_single_token_conv_silu(
                mixedQKV,
                weights,
                convState,
                kChannel,
                channelCount,
                kernelSize,
                stateTokenCount
            );
            queryMeanSquare += queryValue * queryValue;
            keyMeanSquare += keyValue * keyValue;
        }
        queryMeanSquare /= float(keyHeadDimension);
        keyMeanSquare /= float(keyHeadDimension);

        float invScale = rsqrt(float(keyHeadDimension));
        float queryScale = rsqrt(queryMeanSquare + epsilon) * invScale * invScale;
        float keyScale = rsqrt(keyMeanSquare + epsilon) * invScale;

        uint valueChannel = keyHiddenSize * 2 + valueHead * valueHeadDimension + valueDimension;
        float value = edge_gdn_single_token_conv_silu(
            mixedQKV,
            weights,
            convState,
            valueChannel,
            channelCount,
            kernelSize,
            stateTokenCount
        );

        float decay = exp(
            -exp(aLog[valueHead]) * edge_gdn_softplus(a[valueHead] + dtBias[valueHead])
        );
        float beta = edge_gdn_sigmoid(b[valueHead]);

        float kvMemory = 0.0f;
        for (uint keyDimension = 0; keyDimension < keyHeadDimension; ++keyDimension) {
            uint qChannel = keyHead * keyHeadDimension + keyDimension;
            uint kChannel = keyHiddenSize + qChannel;
            uint stateIndex = (valueHead * valueHeadDimension + valueDimension) * keyHeadDimension
                + keyDimension;
            float keyValue = edge_gdn_single_token_conv_silu(
                mixedQKV,
                weights,
                convState,
                kChannel,
                channelCount,
                kernelSize,
                stateTokenCount
            ) * keyScale;
            float decayedState = stateIn[stateIndex] * decay;
            stateOut[stateIndex] = decayedState;
            kvMemory += decayedState * keyValue;
        }

        float delta = (value - kvMemory) * beta;
        float projected = 0.0f;
        for (uint keyDimension = 0; keyDimension < keyHeadDimension; ++keyDimension) {
            uint qChannel = keyHead * keyHeadDimension + keyDimension;
            uint kChannel = keyHiddenSize + qChannel;
            uint stateIndex = (valueHead * valueHeadDimension + valueDimension) * keyHeadDimension
                + keyDimension;
            float queryValue = edge_gdn_single_token_conv_silu(
                mixedQKV,
                weights,
                convState,
                qChannel,
                channelCount,
                kernelSize,
                stateTokenCount
            ) * queryScale;
            float keyValue = edge_gdn_single_token_conv_silu(
                mixedQKV,
                weights,
                convState,
                kChannel,
                channelCount,
                kernelSize,
                stateTokenCount
            ) * keyScale;
            float newState = stateOut[stateIndex] + keyValue * delta;
            stateOut[stateIndex] = newState;
            projected += newState * queryValue;
        }
        output[gid] = projected;
    }

    kernel void edge_add(
        device const float* lhs [[buffer(0)]],
        device const float* rhs [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& elementCount [[buffer(3)]],
        uint gid [[thread_position_in_grid]]
    ) {
        if (gid >= elementCount) {
            return;
        }

        output[gid] = lhs[gid] + rhs[gid];
    }

    kernel void edge_embedding_lookup(
        device const float* embeddings [[buffer(0)]],
        device const uint* tokenIds [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& tokenCount [[buffer(3)]],
        constant uint& hiddenSize [[buffer(4)]],
        uint gid [[thread_position_in_grid]]
    ) {
        uint total = tokenCount * hiddenSize;
        if (gid >= total) {
            return;
        }

        uint tokenIndex = gid / hiddenSize;
        uint hiddenIndex = gid - tokenIndex * hiddenSize;
        uint tokenId = tokenIds[tokenIndex];
        output[gid] = embeddings[tokenId * hiddenSize + hiddenIndex];
    }

    kernel void edge_affine_quantized_embedding_lookup(
        device const uint* packedEmbeddings [[buffer(0)]],
        device const float* scales [[buffer(1)]],
        device const float* biases [[buffer(2)]],
        device const uint* tokenIds [[buffer(3)]],
        device float* output [[buffer(4)]],
        constant uint& tokenCount [[buffer(5)]],
        constant uint& hiddenSize [[buffer(6)]],
        constant uint& packedWordsPerRow [[buffer(7)]],
        constant uint& scaleColumns [[buffer(8)]],
        constant uint& groupSize [[buffer(9)]],
        constant uint& bits [[buffer(10)]],
        constant uint& packedWordOffset [[buffer(11)]],
        uint gid [[thread_position_in_grid]]
    ) {
        uint total = tokenCount * hiddenSize;
        if (gid >= total) {
            return;
        }

        uint tokenIndex = gid / hiddenSize;
        uint hiddenIndex = gid - tokenIndex * hiddenSize;
        uint tokenId = tokenIds[tokenIndex];
        uint bitOffset = hiddenIndex * bits;
        uint packedIndex = tokenId * packedWordsPerRow + bitOffset / 32;
        uint shift = bitOffset % 32;
        uint mask = (1u << bits) - 1u;
        uint raw = packedEmbeddings[packedWordOffset + packedIndex] >> shift;
        if (shift + bits > 32) {
            raw |= packedEmbeddings[packedWordOffset + packedIndex + 1] << (32 - shift);
        }
        raw &= mask;

        uint scaleIndex = tokenId * scaleColumns + hiddenIndex / groupSize;
        output[gid] = float(raw) * scales[scaleIndex] + biases[scaleIndex];
    }

    kernel void edge_scaled_dot_product_attention(
        device const float* query [[buffer(0)]],
        device const float* key [[buffer(1)]],
        device const float* value [[buffer(2)]],
        device float* output [[buffer(3)]],
        constant uint& queryTokenCount [[buffer(4)]],
        constant uint& keyValueTokenCount [[buffer(5)]],
        constant uint& queryPositionOffset [[buffer(6)]],
        constant uint& queryHeadCount [[buffer(7)]],
        constant uint& keyValueHeadCount [[buffer(8)]],
        constant uint& headDimension [[buffer(9)]],
        constant uint& causal [[buffer(10)]],
        constant float& scale [[buffer(11)]],
        uint gid [[thread_position_in_grid]]
    ) {
        uint total = queryTokenCount * queryHeadCount * headDimension;
        if (gid >= total) {
            return;
        }

        uint dimension = gid % headDimension;
        uint queryHead = (gid / headDimension) % queryHeadCount;
        uint token = gid / (queryHeadCount * headDimension);
        uint queryHeadsPerKeyValueHead = queryHeadCount / keyValueHeadCount;
        uint keyValueHead = queryHead / queryHeadsPerKeyValueHead;
        uint absoluteQueryPosition = queryPositionOffset + token;
        uint causalTokenCount = min(keyValueTokenCount, absoluteQueryPosition + 1u);
        uint attendableTokenCount = (causal != 0u) ? causalTokenCount : keyValueTokenCount;

        float maxScore = -3.402823466e+38f;
        for (uint keyToken = 0; keyToken < attendableTokenCount; ++keyToken) {
            float dot = 0.0f;
            for (uint index = 0; index < headDimension; ++index) {
                uint queryIndex = (token * queryHeadCount + queryHead) * headDimension + index;
                uint keyIndex = (keyToken * keyValueHeadCount + keyValueHead) * headDimension + index;
                dot += query[queryIndex] * key[keyIndex];
            }
            maxScore = max(maxScore, dot * scale);
        }

        float denominator = 0.0f;
        float accumulator = 0.0f;
        for (uint keyToken = 0; keyToken < attendableTokenCount; ++keyToken) {
            float dot = 0.0f;
            for (uint index = 0; index < headDimension; ++index) {
                uint queryIndex = (token * queryHeadCount + queryHead) * headDimension + index;
                uint keyIndex = (keyToken * keyValueHeadCount + keyValueHead) * headDimension + index;
                dot += query[queryIndex] * key[keyIndex];
            }
            float probabilityNumerator = exp(dot * scale - maxScore);
            uint valueIndex = (keyToken * keyValueHeadCount + keyValueHead) * headDimension + dimension;
            denominator += probabilityNumerator;
            accumulator += probabilityNumerator * value[valueIndex];
        }

        output[gid] = accumulator / denominator;
    }

    kernel void edge_copy_rows_to_prefix(
        device const float* source [[buffer(0)]],
        device float* destination [[buffer(1)]],
        constant uint& sourceRows [[buffer(2)]],
        constant uint& columns [[buffer(3)]],
        constant uint& startRow [[buffer(4)]],
        uint gid [[thread_position_in_grid]]
    ) {
        uint total = sourceRows * columns;
        if (gid >= total) {
            return;
        }

        uint row = gid / columns;
        uint column = gid - row * columns;
        uint destinationIndex = (startRow + row) * columns + column;
        destination[destinationIndex] = source[gid];
    }

    kernel void edge_gather_rows(
        device const float* source [[buffer(0)]],
        device const uint* rowIndices [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& outputRows [[buffer(3)]],
        constant uint& columns [[buffer(4)]],
        uint gid [[thread_position_in_grid]]
    ) {
        uint total = outputRows * columns;
        if (gid >= total) {
            return;
        }

        uint row = gid / columns;
        uint column = gid - row * columns;
        uint sourceRow = rowIndices[row];
        output[gid] = source[sourceRow * columns + column];
    }

    kernel void edge_update_attention_score_ema(
        device const float* query [[buffer(0)]],
        device const float* key [[buffer(1)]],
        device float* scores [[buffer(2)]],
        constant uint& keyValueTokenCount [[buffer(3)]],
        constant uint& queryHeadCount [[buffer(4)]],
        constant uint& keyValueHeadCount [[buffer(5)]],
        constant uint& headDimension [[buffer(6)]],
        constant float& scale [[buffer(7)]],
        constant float& decay [[buffer(8)]],
        constant uint& hasExistingScores [[buffer(9)]],
        uint gid [[thread_position_in_grid]]
    ) {
        uint total = keyValueTokenCount * keyValueHeadCount;
        if (gid >= total) {
            return;
        }

        uint keyValueHead = gid % keyValueHeadCount;
        uint keyToken = gid / keyValueHeadCount;
        uint queryHeadsPerKeyValueHead = queryHeadCount / keyValueHeadCount;
        uint firstQueryHead = keyValueHead * queryHeadsPerKeyValueHead;

        float scoreSum = 0.0f;
        for (uint localHead = 0; localHead < queryHeadsPerKeyValueHead; ++localHead) {
            uint queryHead = firstQueryHead + localHead;
            float dot = 0.0f;
            for (uint index = 0; index < headDimension; ++index) {
                uint queryIndex = queryHead * headDimension + index;
                uint keyIndex = (keyToken * keyValueHeadCount + keyValueHead) * headDimension + index;
                dot += query[queryIndex] * key[keyIndex];
            }
            scoreSum += fabs(dot * scale);
        }

        float importance = scoreSum / float(queryHeadsPerKeyValueHead);
        uint scoreIndex = keyToken * keyValueHeadCount + keyValueHead;
        if (hasExistingScores != 0u) {
            scores[scoreIndex] = scores[scoreIndex] * decay + importance * (1.0f - decay);
        } else {
            scores[scoreIndex] = importance;
        }
    }

    kernel void edge_affine_qmv_transposed_argmax_partials(
        device const float* lhs [[buffer(0)]],
        device const uchar* packedWeights [[buffer(1)]],
        device const float* scales [[buffer(2)]],
        device const float* biases [[buffer(3)]],
        device uint* partialTokens [[buffer(4)]],
        device float* partialLogits [[buffer(5)]],
        constant uint& inner [[buffer(6)]],
        constant uint& columns [[buffer(7)]],
        constant uint& packedWordsPerRow [[buffer(8)]],
        constant uint& scaleColumns [[buffer(9)]],
        constant uint& groupSize [[buffer(10)]],
        constant uint& bits [[buffer(11)]],
        uint3 tid [[threadgroup_position_in_grid]],
        uint simdGroupID [[simdgroup_index_in_threadgroup]],
        uint simdLaneID [[thread_index_in_simdgroup]]
    ) {
        constexpr uint resultsPerSimdgroup = 4;
        constexpr uint numSimdgroups = 2;
        uint outputBase = tid.y * 8u + simdGroupID * resultsPerSimdgroup;
        uint partialIndex = tid.y * numSimdgroups + simdGroupID;
        if (simdGroupID >= numSimdgroups
            || outputBase >= columns
            || groupSize != 64u
            || (bits != 4u && bits != 6u)) {
            if (simdLaneID == 0u) {
                partialTokens[partialIndex] = 0u;
                partialLogits[partialIndex] = -INFINITY;
            }
            return;
        }

        thread float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        device const uchar* packedBase = packedWeights;

        uint fastBlockSize = (bits == 4u) ? 512u : 256u;
        if ((inner % fastBlockSize) != 0u) {
            uint valuesPerThread = (bits == 4u) ? 8u : 4u;
            uint safeBlockSize = valuesPerThread * 32u;
            float partial0 = 0.0f;
            float partial1 = 0.0f;
            float partial2 = 0.0f;
            float partial3 = 0.0f;

            for (uint block = 0u; block < inner; block += safeBlockSize) {
                uint baseIndex = block + simdLaneID * valuesPerThread;
                for (uint laneValue = 0u; laneValue < valuesPerThread; laneValue += 1u) {
                    uint inputColumn = baseIndex + laneValue;
                    if (inputColumn >= inner) {
                        continue;
                    }
                    float inputValue = lhs[inputColumn];
                    uint groupIndex = inputColumn / groupSize;

                    uint outputColumn0 = outputBase;
                    if (outputColumn0 < columns) {
                        uint scaleIndex = outputColumn0 * scaleColumns + groupIndex;
                        uint raw = edge_load_affine_packed_value_bytes(
                            packedWeights,
                            packedWordsPerRow,
                            outputColumn0,
                            inputColumn,
                            bits
                        );
                        partial0 += inputValue * (float(raw) * scales[scaleIndex] + biases[scaleIndex]);
                    }

                    uint outputColumn1 = outputBase + 1u;
                    if (outputColumn1 < columns) {
                        uint scaleIndex = outputColumn1 * scaleColumns + groupIndex;
                        uint raw = edge_load_affine_packed_value_bytes(
                            packedWeights,
                            packedWordsPerRow,
                            outputColumn1,
                            inputColumn,
                            bits
                        );
                        partial1 += inputValue * (float(raw) * scales[scaleIndex] + biases[scaleIndex]);
                    }

                    uint outputColumn2 = outputBase + 2u;
                    if (outputColumn2 < columns) {
                        uint scaleIndex = outputColumn2 * scaleColumns + groupIndex;
                        uint raw = edge_load_affine_packed_value_bytes(
                            packedWeights,
                            packedWordsPerRow,
                            outputColumn2,
                            inputColumn,
                            bits
                        );
                        partial2 += inputValue * (float(raw) * scales[scaleIndex] + biases[scaleIndex]);
                    }

                    uint outputColumn3 = outputBase + 3u;
                    if (outputColumn3 < columns) {
                        uint scaleIndex = outputColumn3 * scaleColumns + groupIndex;
                        uint raw = edge_load_affine_packed_value_bytes(
                            packedWeights,
                            packedWordsPerRow,
                            outputColumn3,
                            inputColumn,
                            bits
                        );
                        partial3 += inputValue * (float(raw) * scales[scaleIndex] + biases[scaleIndex]);
                    }
                }
            }

            result[0] = simd_sum(partial0);
            result[1] = simd_sum(partial1);
            result[2] = simd_sum(partial2);
            result[3] = simd_sum(partial3);
        } else if (bits == 4u) {
            constexpr uint valuesPerThread = 16;
            constexpr uint blockSize = valuesPerThread * 32u;
            constexpr uint bytesPerPack = 4;
            constexpr uint packFactor = 8;
            constexpr uint packsPerThread = 2;
            constexpr uint scaleStepPerThread = 4;

            uint inVecSizeBytes = inner * bytesPerPack / packFactor;
            device const uchar* weightsCursor = packedBase
                + outputBase * inVecSizeBytes
                + simdLaneID * packsPerThread * bytesPerPack;
            device const float* scalesCursor = scales
                + outputBase * scaleColumns
                + simdLaneID / scaleStepPerThread;
            device const float* biasesCursor = biases
                + outputBase * scaleColumns
                + simdLaneID / scaleStepPerThread;
            device const float* inputCursor = lhs + simdLaneID * valuesPerThread;
            thread float xThread[16];

            for (uint k = 0u; k < inner; k += blockSize) {
                float sum = edge_qmv_load4(inputCursor, xThread);
                for (uint row = 0u; row < resultsPerSimdgroup; ++row) {
                    uint outputColumn = outputBase + row;
                    if (outputColumn < columns) {
                        device const uchar* weightsRow = weightsCursor + row * inVecSizeBytes;
                        device const float* scalesRow = scalesCursor + row * scaleColumns;
                        device const float* biasesRow = biasesCursor + row * scaleColumns;
                        result[row] += edge_qmv_qdot4(weightsRow, xThread, scalesRow[0], biasesRow[0], sum);
                    }
                }
                weightsCursor += blockSize * bytesPerPack / packFactor;
                scalesCursor += blockSize / groupSize;
                biasesCursor += blockSize / groupSize;
                inputCursor += blockSize;
            }
            for (uint row = 0u; row < resultsPerSimdgroup; ++row) {
                result[row] = simd_sum(result[row]);
            }
        } else {
            constexpr uint valuesPerThread = 8;
            constexpr uint blockSize = valuesPerThread * 32u;
            constexpr uint bytesPerPack = 3;
            constexpr uint packFactor = 4;
            constexpr uint packsPerThread = 2;
            constexpr uint scaleStepPerThread = 8;

            uint inVecSizeBytes = inner * bytesPerPack / packFactor;
            device const uchar* weightsCursor = packedBase
                + outputBase * inVecSizeBytes
                + simdLaneID * packsPerThread * bytesPerPack;
            device const float* scalesCursor = scales
                + outputBase * scaleColumns
                + simdLaneID / scaleStepPerThread;
            device const float* biasesCursor = biases
                + outputBase * scaleColumns
                + simdLaneID / scaleStepPerThread;
            device const float* inputCursor = lhs + simdLaneID * valuesPerThread;
            thread float xThread[8];

            for (uint k = 0u; k < inner; k += blockSize) {
                float sum = edge_qmv_load6(inputCursor, xThread);
                for (uint row = 0u; row < resultsPerSimdgroup; ++row) {
                    uint outputColumn = outputBase + row;
                    if (outputColumn < columns) {
                        device const uchar* weightsRow = weightsCursor + row * inVecSizeBytes;
                        device const float* scalesRow = scalesCursor + row * scaleColumns;
                        device const float* biasesRow = biasesCursor + row * scaleColumns;
                        result[row] += edge_qmv_qdot6(weightsRow, xThread, scalesRow[0], biasesRow[0], sum);
                    }
                }
                weightsCursor += blockSize * bytesPerPack / packFactor;
                scalesCursor += blockSize / groupSize;
                biasesCursor += blockSize / groupSize;
                inputCursor += blockSize;
            }
            for (uint row = 0u; row < resultsPerSimdgroup; ++row) {
                result[row] = simd_sum(result[row]);
            }
        }

        if (simdLaneID == 0u) {
            float bestValue = -INFINITY;
            uint bestToken = 0u;
            for (uint row = 0u; row < resultsPerSimdgroup; ++row) {
                uint outputColumn = outputBase + row;
                if (outputColumn < columns) {
                    float value = result[row];
                    if (value > bestValue || (value == bestValue && outputColumn < bestToken)) {
                        bestValue = value;
                        bestToken = outputColumn;
                    }
                }
            }
            partialTokens[partialIndex] = bestToken;
            partialLogits[partialIndex] = bestValue;
        }
    }

    kernel void edge_argmax_partials(
        device const uint* partialTokens [[buffer(0)]],
        device const float* partialLogits [[buffer(1)]],
        device uint* outputToken [[buffer(2)]],
        device float* outputLogit [[buffer(3)]],
        constant uint& partialCount [[buffer(4)]],
        uint tid [[thread_position_in_threadgroup]],
        uint threadsPerGroup [[threads_per_threadgroup]]
    ) {
        threadgroup float bestValues[256];
        threadgroup uint bestTokens[256];

        float bestValue = -INFINITY;
        uint bestToken = 0u;
        for (uint index = tid; index < partialCount; index += threadsPerGroup) {
            float value = partialLogits[index];
            uint token = partialTokens[index];
            if (value > bestValue || (value == bestValue && token < bestToken)) {
                bestValue = value;
                bestToken = token;
            }
        }

        bestValues[tid] = bestValue;
        bestTokens[tid] = bestToken;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threadsPerGroup >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float otherValue = bestValues[tid + stride];
                uint otherToken = bestTokens[tid + stride];
                if (otherValue > bestValues[tid]
                    || (otherValue == bestValues[tid] && otherToken < bestTokens[tid])) {
                    bestValues[tid] = otherValue;
                    bestTokens[tid] = otherToken;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0u) {
            outputToken[0] = bestTokens[0];
            outputLogit[0] = bestValues[0];
        }
    }

    kernel void edge_argmax_last_row(
        device const float* logits [[buffer(0)]],
        device uint* outputToken [[buffer(1)]],
        device float* outputLogit [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& columns [[buffer(4)]],
        uint tid [[thread_position_in_threadgroup]],
        uint threadsPerGroup [[threads_per_threadgroup]]
    ) {
        threadgroup float bestValues[256];
        threadgroup uint bestTokens[256];

        uint rowOffset = (rows - 1) * columns;
        float bestValue = -INFINITY;
        uint bestToken = 0;
        for (uint column = tid; column < columns; column += threadsPerGroup) {
            float value = logits[rowOffset + column];
            if (value > bestValue || (value == bestValue && column < bestToken)) {
                bestValue = value;
                bestToken = column;
            }
        }

        bestValues[tid] = bestValue;
        bestTokens[tid] = bestToken;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threadsPerGroup >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float otherValue = bestValues[tid + stride];
                uint otherToken = bestTokens[tid + stride];
                if (otherValue > bestValues[tid]
                    || (otherValue == bestValues[tid] && otherToken < bestTokens[tid])) {
                    bestValues[tid] = otherValue;
                    bestTokens[tid] = otherToken;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            outputToken[0] = bestTokens[0];
            outputLogit[0] = bestValues[0];
        }
    }
    """
}

private struct AffineQuantizedBufferCacheKey: Hashable {
    var storageIdentifier: ObjectIdentifier
    var shape: [Int]
    var packedShape: [Int]
    var scaleShape: [Int]
    var groupSize: Int
    var bits: Int
    var mode: EdgeQuantizationMode

    init(_ weights: EdgeQuantizedTensor) {
        self.storageIdentifier = weights.storageIdentifier
        self.shape = weights.shape
        self.packedShape = weights.packedShape
        self.scaleShape = weights.scaleShape
        self.groupSize = weights.groupSize
        self.bits = weights.bits
        self.mode = weights.mode
    }
}

struct AffineQuantizedMetalBuffers {
    var packedBuffer: MTLBuffer
    var packedOffset: Int = 0
    var scalesBuffer: MTLBuffer
    var scalesOffset: Int = 0
    var biasesBuffer: MTLBuffer
    var biasesOffset: Int = 0
    var hostRetainers: [Data] = []
}

private struct QuantizedInputMetalBuffer {
    var buffer: MTLBuffer
    var offset: Int
}
