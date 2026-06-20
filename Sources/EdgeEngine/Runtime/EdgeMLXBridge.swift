// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeCmlx
import Foundation
import Metal

public enum EdgeMLXBridgeError: Error, Equatable {
    case invalidShape
    case executionFailed(String)
}

public struct EdgeMLXStateProbeSnapshot: Equatable {
    public var status: Int
    public var isAvailable: Bool
    public var hasPrimitive: Bool
    public var siblingCount: Int
}

public struct EdgeMLXStateBoundaryProbeResult: Equatable {
    public var syncSteps: Int
    public var syncRecurrentAfterTokenEval: EdgeMLXStateProbeSnapshot
    public var syncConvAfterTokenEval: EdgeMLXStateProbeSnapshot
    public var syncRecurrentStopGradientAfterTokenEval: EdgeMLXStateProbeSnapshot
    public var syncConvStopGradientAfterTokenEval: EdgeMLXStateProbeSnapshot
    public var syncCustomRecurrentAfterTokenEval: EdgeMLXStateProbeSnapshot
    public var syncCustomRecurrentStopGradientAfterTokenEval: EdgeMLXStateProbeSnapshot
    public var asyncSteps: Int
    public var asyncRecurrentAfterSchedule: EdgeMLXStateProbeSnapshot
    public var asyncConvAfterSchedule: EdgeMLXStateProbeSnapshot
    public var asyncRecurrentStopGradientAfterSchedule: EdgeMLXStateProbeSnapshot
    public var asyncConvStopGradientAfterSchedule: EdgeMLXStateProbeSnapshot
    public var asyncCustomRecurrentAfterSchedule: EdgeMLXStateProbeSnapshot
    public var asyncCustomRecurrentStopGradientAfterSchedule: EdgeMLXStateProbeSnapshot
    public var asyncRecurrentAfterFinalEval: EdgeMLXStateProbeSnapshot
    public var asyncConvAfterFinalEval: EdgeMLXStateProbeSnapshot
    public var asyncCustomRecurrentAfterFinalEval: EdgeMLXStateProbeSnapshot
}

public enum EdgeMLXBridge {
    public static var vendorVersion: String {
        String(cString: edge_cmlx_vendor_version())
    }

    public static var vendorVersionNumeric: Int {
        Int(edge_cmlx_vendor_version_numeric())
    }

    public static var isVendorPresent: Bool {
        edge_cmlx_vendor_present() == 1
    }

    public static var isDefaultMetallibAvailable: Bool {
        edge_cmlx_default_metallib_available() == 1
    }

    public static func runStateBoundaryProbe(steps: Int = 20) throws -> EdgeMLXStateBoundaryProbeResult {
        guard steps > 0 else {
            throw EdgeMLXBridgeError.invalidShape
        }

        var result = EdgeCmlxStateBoundaryProbeResult()
        let status = edge_cmlx_run_state_boundary_probe(Int32(steps), &result)
        guard status == 0 else {
            let message = edge_cmlx_last_error().map { String(cString: $0) } ?? "unknown Cmlx error"
            throw EdgeMLXBridgeError.executionFailed(message)
        }
        return EdgeMLXStateBoundaryProbeResult(
            syncSteps: Int(result.sync_steps),
            syncRecurrentAfterTokenEval: makeStateProbeSnapshot(
                result.sync_recurrent_after_token_eval
            ),
            syncConvAfterTokenEval: makeStateProbeSnapshot(
                result.sync_conv_after_token_eval
            ),
            syncRecurrentStopGradientAfterTokenEval: makeStateProbeSnapshot(
                result.sync_recurrent_stop_gradient_after_token_eval
            ),
            syncConvStopGradientAfterTokenEval: makeStateProbeSnapshot(
                result.sync_conv_stop_gradient_after_token_eval
            ),
            syncCustomRecurrentAfterTokenEval: makeStateProbeSnapshot(
                result.sync_custom_recurrent_after_token_eval
            ),
            syncCustomRecurrentStopGradientAfterTokenEval: makeStateProbeSnapshot(
                result.sync_custom_recurrent_stop_gradient_after_token_eval
            ),
            asyncSteps: Int(result.async_steps),
            asyncRecurrentAfterSchedule: makeStateProbeSnapshot(
                result.async_recurrent_after_schedule
            ),
            asyncConvAfterSchedule: makeStateProbeSnapshot(
                result.async_conv_after_schedule
            ),
            asyncRecurrentStopGradientAfterSchedule: makeStateProbeSnapshot(
                result.async_recurrent_stop_gradient_after_schedule
            ),
            asyncConvStopGradientAfterSchedule: makeStateProbeSnapshot(
                result.async_conv_stop_gradient_after_schedule
            ),
            asyncCustomRecurrentAfterSchedule: makeStateProbeSnapshot(
                result.async_custom_recurrent_after_schedule
            ),
            asyncCustomRecurrentStopGradientAfterSchedule: makeStateProbeSnapshot(
                result.async_custom_recurrent_stop_gradient_after_schedule
            ),
            asyncRecurrentAfterFinalEval: makeStateProbeSnapshot(
                result.async_recurrent_after_final_eval
            ),
            asyncConvAfterFinalEval: makeStateProbeSnapshot(
                result.async_conv_after_final_eval
            ),
            asyncCustomRecurrentAfterFinalEval: makeStateProbeSnapshot(
                result.async_custom_recurrent_after_final_eval
            )
        )
    }

    public static func runCrossThreadStreamProbe() throws {
        let status = edge_cmlx_run_cross_thread_stream_probe()
        guard status == 0 else {
            let message = edge_cmlx_last_error().map { String(cString: $0) } ?? "unknown Cmlx error"
            throw EdgeMLXBridgeError.executionFailed(message)
        }
    }

    private static func makeStateProbeSnapshot(
        _ snapshot: EdgeCmlxStateProbeSnapshot
    ) -> EdgeMLXStateProbeSnapshot {
        EdgeMLXStateProbeSnapshot(
            status: Int(snapshot.status),
            isAvailable: snapshot.is_available != 0,
            hasPrimitive: snapshot.has_primitive != 0,
            siblingCount: Int(snapshot.sibling_count)
        )
    }

    public static func matmulFloat32CPU(
        lhs: [Float],
        rows: Int,
        inner: Int,
        rhs: [Float],
        columns: Int
    ) throws -> [Float] {
        try matmulFloat32(
            lhs: lhs,
            rows: rows,
            inner: inner,
            rhs: rhs,
            columns: columns,
            run: edge_cmlx_matmul_f32_cpu
        )
    }

    public static func matmulFloat32GPU(
        lhs: [Float],
        rows: Int,
        inner: Int,
        rhs: [Float],
        columns: Int
    ) throws -> [Float] {
        try matmulFloat32(
            lhs: lhs,
            rows: rows,
            inner: inner,
            rhs: rhs,
            columns: columns,
            run: edge_cmlx_matmul_f32_gpu
        )
    }

    public static func softmaxFloat32GPU(
        _ input: [Float],
        rows: Int,
        columns: Int
    ) throws -> [Float] {
        guard rows > 0, columns > 0, input.count == rows * columns else {
            throw EdgeMLXBridgeError.invalidShape
        }

        var output = Array(repeating: Float.zero, count: rows * columns)
        let status = output.withUnsafeMutableBufferPointer { outputBuffer in
            input.withUnsafeBufferPointer { inputBuffer in
                edge_cmlx_softmax_f32_gpu(
                    inputBuffer.baseAddress,
                    Int32(rows),
                    Int32(columns),
                    outputBuffer.baseAddress,
                    outputBuffer.count
                )
            }
        }

        guard status == 0 else {
            let message = edge_cmlx_last_error().map { String(cString: $0) } ?? "unknown Cmlx error"
            throw EdgeMLXBridgeError.executionFailed(message)
        }
        return output
    }

    public static func sampleTokenFloat32GPU(
        logits: [Float],
        temperature: Float,
        topK: Int?,
        topP: Float,
        minP: Float = 0,
        seed: UInt64
    ) throws -> Int {
        guard !logits.isEmpty,
              temperature >= 0,
              temperature.isFinite,
              topP > 0,
              topP <= 1,
              topP.isFinite,
              minP >= 0,
              minP.isFinite
        else {
            throw EdgeMLXBridgeError.invalidShape
        }
        if let topK, topK <= 0 {
            throw EdgeMLXBridgeError.invalidShape
        }

        var output = Int32.zero
        let status = logits.withUnsafeBufferPointer { logitsBuffer in
            edge_cmlx_sample_token_f32_gpu(
                logitsBuffer.baseAddress,
                Int32(logits.count),
                temperature,
                Int32(topK ?? 0),
                topP,
                minP,
                seed,
                &output
            )
        }
        guard status == 0 else {
            let message = edge_cmlx_last_error().map { String(cString: $0) } ?? "unknown Cmlx error"
            throw EdgeMLXBridgeError.executionFailed(message)
        }
        return Int(output)
    }

    public static func fastRMSNormFloat32GPU(
        _ input: [Float],
        rows: Int,
        columns: Int,
        weight: [Float],
        epsilon: Float
    ) throws -> [Float] {
        guard rows > 0,
              columns > 0,
              input.count == rows * columns,
              weight.count == columns,
              epsilon >= 0
        else {
            throw EdgeMLXBridgeError.invalidShape
        }

        var output = Array(repeating: Float.zero, count: rows * columns)
        let status = output.withUnsafeMutableBufferPointer { outputBuffer in
            input.withUnsafeBufferPointer { inputBuffer in
                weight.withUnsafeBufferPointer { weightBuffer in
                    edge_cmlx_fast_rms_norm_f32_gpu(
                        inputBuffer.baseAddress,
                        Int32(rows),
                        Int32(columns),
                        weightBuffer.baseAddress,
                        epsilon,
                        outputBuffer.baseAddress,
                        outputBuffer.count
                    )
                }
            }
        }

        guard status == 0 else {
            let message = edge_cmlx_last_error().map { String(cString: $0) } ?? "unknown Cmlx error"
            throw EdgeMLXBridgeError.executionFailed(message)
        }
        return output
    }

    public static func rmsNormScaleFloat32GPU(
        _ input: [Float],
        rows: Int,
        columns: Int,
        epsilon: Float,
        scale: Float
    ) throws -> [Float] {
        guard rows > 0,
              columns > 0,
              input.count == rows * columns,
              epsilon >= 0,
              scale.isFinite
        else {
            throw EdgeMLXBridgeError.invalidShape
        }

        var output = Array(repeating: Float.zero, count: rows * columns)
        let status = output.withUnsafeMutableBufferPointer { outputBuffer in
            input.withUnsafeBufferPointer { inputBuffer in
                edge_cmlx_rms_norm_scale_f32_gpu(
                    inputBuffer.baseAddress,
                    Int32(rows),
                    Int32(columns),
                    epsilon,
                    scale,
                    outputBuffer.baseAddress,
                    outputBuffer.count
                )
            }
        }

        guard status == 0 else {
            let message = edge_cmlx_last_error().map { String(cString: $0) } ?? "unknown Cmlx error"
            throw EdgeMLXBridgeError.executionFailed(message)
        }
        return output
    }

    public static func supportsFastRMSNormFloat32MTL(
        rows: Int,
        columns: Int
    ) -> Bool {
        rows > 0 && columns > 0
    }

    public static func encodeFastRMSNormFloat32MTL(
        commandBuffer: MTLCommandBuffer,
        inputBuffer: MTLBuffer,
        inputOffset: Int = 0,
        rows: Int,
        columns: Int,
        weightBuffer: MTLBuffer,
        weightOffset: Int = 0,
        epsilon: Float,
        outputBuffer: MTLBuffer,
        outputOffset: Int = 0
    ) throws {
        guard supportsFastRMSNormFloat32MTL(rows: rows, columns: columns),
              inputOffset >= 0,
              weightOffset >= 0,
              outputOffset >= 0,
              epsilon >= 0,
              inputBuffer.length >= inputOffset + rows * columns * MemoryLayout<Float>.stride,
              weightBuffer.length >= weightOffset + columns * MemoryLayout<Float>.stride,
              outputBuffer.length >= outputOffset + rows * columns * MemoryLayout<Float>.stride
        else {
            throw EdgeMLXBridgeError.invalidShape
        }

        let status = edge_cmlx_encode_fast_rms_norm_f32_mtl(
            opaqueMutablePointer(to: commandBuffer),
            opaquePointer(to: inputBuffer),
            inputOffset,
            Int32(rows),
            Int32(columns),
            opaquePointer(to: weightBuffer),
            weightOffset,
            epsilon,
            opaqueMutablePointer(to: outputBuffer),
            outputOffset,
            rows * columns
        )

        guard status == 0 else {
            let message = edge_cmlx_last_error().map { String(cString: $0) } ?? "unknown Cmlx error"
            throw EdgeMLXBridgeError.executionFailed(message)
        }
    }

    public static func affineQuantizedMatmulFloat32GPU(
        lhs: [Float],
        rows: Int,
        inner: Int,
        weights: EdgeQuantizedTensor,
        transpose: Bool = false
    ) throws -> [Float] {
        guard let outputColumns = affineQuantizedMatmulOutputColumns(
            rows: rows,
            inner: inner,
            weights: weights,
            transpose: transpose
        ), lhs.count == rows * inner else {
            throw EdgeMLXBridgeError.invalidShape
        }

        let packedWeights = weights.packedValues
        let scales = weights.scales
        let biases = weights.biases ?? Array(repeating: Float.zero, count: weights.scaleCount)
        var output = Array(repeating: Float.zero, count: rows * outputColumns)
        let status = output.withUnsafeMutableBufferPointer { outputBuffer in
            lhs.withUnsafeBufferPointer { lhsBuffer in
                packedWeights.withUnsafeBufferPointer { packedBuffer in
                    scales.withUnsafeBufferPointer { scalesBuffer in
                        biases.withUnsafeBufferPointer { biasesBuffer in
                            edge_cmlx_affine_quantized_matmul_f32_gpu(
                                lhsBuffer.baseAddress,
                                Int32(rows),
                                Int32(inner),
                                packedBuffer.baseAddress,
                                Int32(weights.packedShape[0]),
                                Int32(weights.packedShape[1]),
                                scalesBuffer.baseAddress,
                                Int32(weights.scaleShape[0]),
                                Int32(weights.scaleShape[1]),
                                biasesBuffer.baseAddress,
                                Int32(weights.groupSize),
                                Int32(weights.bits),
                                transpose ? 1 : 0,
                                outputBuffer.baseAddress,
                                outputBuffer.count
                            )
                        }
                    }
                }
            }
        }

        guard status == 0 else {
            let message = edge_cmlx_last_error().map { String(cString: $0) } ?? "unknown Cmlx error"
            throw EdgeMLXBridgeError.executionFailed(message)
        }
        return output
    }

    public static func supportsAffineQuantizedMatmulFloat32MTL(
        rows: Int,
        inner: Int,
        weights: EdgeQuantizedTensor,
        transpose: Bool = false
    ) -> Bool {
        affineQuantizedMatmulOutputColumns(
            rows: rows,
            inner: inner,
            weights: weights,
            transpose: transpose
        ) != nil
    }

    public static func supportsAffineQMMTransposedCommandBufferFloat32MTL(
        rows: Int,
        inner: Int,
        weights: EdgeQuantizedTensor,
        transpose: Bool = false
    ) -> Bool {
        guard transpose,
              rows > 1,
              weights.groupSize == 32 || weights.groupSize == 64 || weights.groupSize == 128,
              weights.bits == 4 || weights.bits == 6
        else {
            return false
        }
        return affineQuantizedMatmulOutputColumns(
            rows: rows,
            inner: inner,
            weights: weights,
            transpose: transpose
        ) != nil
    }

    public static func affineQuantizedMatmulFloat32MTL(
        lhsBuffer: MTLBuffer,
        rows: Int,
        inner: Int,
        weights: EdgeQuantizedTensor,
        packedWeightsBuffer: MTLBuffer,
        scalesBuffer: MTLBuffer,
        biasesBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        transpose: Bool = false
    ) throws {
        guard let outputColumns = affineQuantizedMatmulOutputColumns(
            rows: rows,
            inner: inner,
            weights: weights,
            transpose: transpose
        ) else {
            throw EdgeMLXBridgeError.invalidShape
        }
        guard lhsBuffer.length >= rows * inner * MemoryLayout<Float>.stride,
              packedWeightsBuffer.length >= weights.packedByteCount,
              scalesBuffer.length >= weights.scalesByteCount,
              biasesBuffer.length >= weights.affineBiasBufferByteCount,
              outputBuffer.length >= rows * outputColumns * MemoryLayout<Float>.stride
        else {
            throw EdgeMLXBridgeError.invalidShape
        }

        let status = edge_cmlx_affine_quantized_matmul_f32_mtl(
            opaquePointer(to: lhsBuffer),
            Int32(rows),
            Int32(inner),
            opaquePointer(to: packedWeightsBuffer),
            Int32(weights.packedShape[0]),
            Int32(weights.packedShape[1]),
            opaquePointer(to: scalesBuffer),
            Int32(weights.scaleShape[0]),
            Int32(weights.scaleShape[1]),
            opaquePointer(to: biasesBuffer),
            Int32(weights.groupSize),
            Int32(weights.bits),
            transpose ? 1 : 0,
            opaqueMutablePointer(to: outputBuffer),
            rows * outputColumns
        )

        guard status == 0 else {
            let message = edge_cmlx_last_error().map { String(cString: $0) } ?? "unknown Cmlx error"
            throw EdgeMLXBridgeError.executionFailed(message)
        }
    }

    public static func rmsNormAffineQuantizedMatmulFloat32MTL(
        lhs: EdgeTensor,
        normWeight: EdgeTensor,
        epsilon: Float,
        weights: EdgeQuantizedTensor,
        runtime: EdgeMetalRuntime,
        transpose: Bool = true
    ) throws -> EdgeTensor {
        guard lhs.dataType == .float32,
              normWeight.dataType == .float32,
              lhs.shape.rank == 2,
              normWeight.shape.rank == 1
        else {
            throw EdgeMLXBridgeError.invalidShape
        }
        let rows = lhs.shape.dimensions[0]
        let inner = lhs.shape.dimensions[1]
        guard normWeight.shape.dimensions[0] == inner,
              let outputColumns = affineQuantizedMatmulOutputColumns(
                  rows: rows,
                  inner: inner,
                  weights: weights,
                  transpose: transpose
              )
        else {
            throw EdgeMLXBridgeError.invalidShape
        }

        let packedWeightsBuffer = try makeBuffer(
            runtime: runtime,
            values: weights.packedValues,
            label: "edge.cmlx.rms_qmm.packed"
        )
        let scalesBuffer = try makeBuffer(
            runtime: runtime,
            values: weights.scales,
            label: "edge.cmlx.rms_qmm.scales"
        )
        let biasesBuffer = try makeBuffer(
            runtime: runtime,
            values: weights.biases ?? Array(repeating: Float.zero, count: weights.scaleCount),
            label: "edge.cmlx.rms_qmm.biases"
        )
        let output = try EdgeTensor(
            shape: EdgeTensorShape([rows, outputColumns]),
            dataType: .float32,
            runtime: runtime
        )

        try rmsNormAffineQuantizedMatmulFloat32MTL(
            lhsBuffer: lhs.buffer,
            rows: rows,
            inner: inner,
            normWeightBuffer: normWeight.buffer,
            epsilon: epsilon,
            weights: weights,
            packedWeightsBuffer: packedWeightsBuffer,
            scalesBuffer: scalesBuffer,
            biasesBuffer: biasesBuffer,
            outputBuffer: output.buffer,
            transpose: transpose
        )
        return output
    }

    public static func rmsNormAffineQuantizedMatmulFloat32MTL(
        lhsBuffer: MTLBuffer,
        rows: Int,
        inner: Int,
        normWeightBuffer: MTLBuffer,
        epsilon: Float,
        weights: EdgeQuantizedTensor,
        packedWeightsBuffer: MTLBuffer,
        scalesBuffer: MTLBuffer,
        biasesBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        transpose: Bool = true
    ) throws {
        guard let outputColumns = affineQuantizedMatmulOutputColumns(
            rows: rows,
            inner: inner,
            weights: weights,
            transpose: transpose
        ) else {
            throw EdgeMLXBridgeError.invalidShape
        }
        guard rows > 0,
              inner > 0,
              lhsBuffer.length >= rows * inner * MemoryLayout<Float>.stride,
              normWeightBuffer.length >= inner * MemoryLayout<Float>.stride,
              packedWeightsBuffer.length >= weights.packedByteCount,
              scalesBuffer.length >= weights.scalesByteCount,
              biasesBuffer.length >= weights.affineBiasBufferByteCount,
              outputBuffer.length >= rows * outputColumns * MemoryLayout<Float>.stride
        else {
            throw EdgeMLXBridgeError.invalidShape
        }

        let status = edge_cmlx_rms_norm_affine_quantized_matmul_f32_mtl(
            opaquePointer(to: lhsBuffer),
            Int32(rows),
            Int32(inner),
            opaquePointer(to: normWeightBuffer),
            epsilon,
            opaquePointer(to: packedWeightsBuffer),
            Int32(weights.packedShape[0]),
            Int32(weights.packedShape[1]),
            opaquePointer(to: scalesBuffer),
            Int32(weights.scaleShape[0]),
            Int32(weights.scaleShape[1]),
            opaquePointer(to: biasesBuffer),
            Int32(weights.groupSize),
            Int32(weights.bits),
            transpose ? 1 : 0,
            opaqueMutablePointer(to: outputBuffer),
            rows * outputColumns
        )

        guard status == 0 else {
            let message = edge_cmlx_last_error().map { String(cString: $0) } ?? "unknown Cmlx error"
            throw EdgeMLXBridgeError.executionFailed(message)
        }
    }

    public static func encodeAffineQMMTransposedCommandBufferFloat32MTL(
        commandBuffer: MTLCommandBuffer,
        lhsBuffer: MTLBuffer,
        lhsOffset: Int = 0,
        rows: Int,
        inner: Int,
        weights: EdgeQuantizedTensor,
        packedWeightsBuffer: MTLBuffer,
        packedWeightsOffset: Int = 0,
        scalesBuffer: MTLBuffer,
        scalesOffset: Int = 0,
        biasesBuffer: MTLBuffer,
        biasesOffset: Int = 0,
        outputBuffer: MTLBuffer,
        outputOffset: Int = 0,
        transpose: Bool = true
    ) throws {
        guard supportsAffineQMMTransposedCommandBufferFloat32MTL(
            rows: rows,
            inner: inner,
            weights: weights,
            transpose: transpose
        ), let outputColumns = affineQuantizedMatmulOutputColumns(
            rows: rows,
            inner: inner,
            weights: weights,
            transpose: transpose
        ) else {
            throw EdgeMLXBridgeError.invalidShape
        }
        guard lhsOffset >= 0,
              packedWeightsOffset >= 0,
              scalesOffset >= 0,
              biasesOffset >= 0,
              outputOffset >= 0,
              lhsBuffer.length >= lhsOffset + rows * inner * MemoryLayout<Float>.stride,
              packedWeightsBuffer.length >= packedWeightsOffset + weights.packedByteCount,
              scalesBuffer.length >= scalesOffset + weights.scalesByteCount,
              biasesBuffer.length >= biasesOffset + weights.affineBiasBufferByteCount,
              outputBuffer.length >= outputOffset + rows * outputColumns * MemoryLayout<Float>.stride
        else {
            throw EdgeMLXBridgeError.invalidShape
        }

        let status = edge_cmlx_encode_affine_qmm_t_f32_mtl(
            opaqueMutablePointer(to: commandBuffer),
            opaquePointer(to: lhsBuffer),
            lhsOffset,
            Int32(rows),
            Int32(inner),
            opaquePointer(to: packedWeightsBuffer),
            packedWeightsOffset,
            Int32(weights.packedShape[0]),
            Int32(weights.packedShape[1]),
            opaquePointer(to: scalesBuffer),
            scalesOffset,
            Int32(weights.scaleShape[0]),
            Int32(weights.scaleShape[1]),
            opaquePointer(to: biasesBuffer),
            biasesOffset,
            Int32(weights.groupSize),
            Int32(weights.bits),
            opaqueMutablePointer(to: outputBuffer),
            outputOffset,
            rows * outputColumns
        )

        guard status == 0 else {
            let message = edge_cmlx_last_error().map { String(cString: $0) } ?? "unknown Cmlx error"
            throw EdgeMLXBridgeError.executionFailed(message)
        }
    }

    private static func matmulFloat32(
        lhs: [Float],
        rows: Int,
        inner: Int,
        rhs: [Float],
        columns: Int,
        run: (
            UnsafePointer<Float>?,
            Int32,
            Int32,
            UnsafePointer<Float>?,
            Int32,
            Int32,
            UnsafeMutablePointer<Float>?,
            Int
        ) -> Int32
    ) throws -> [Float] {
        guard rows > 0, inner > 0, columns > 0 else {
            throw EdgeMLXBridgeError.invalidShape
        }
        guard lhs.count == rows * inner, rhs.count == inner * columns else {
            throw EdgeMLXBridgeError.invalidShape
        }

        var output = Array(repeating: Float.zero, count: rows * columns)
        let status = output.withUnsafeMutableBufferPointer { outputBuffer in
            lhs.withUnsafeBufferPointer { lhsBuffer in
                rhs.withUnsafeBufferPointer { rhsBuffer in
                    run(
                        lhsBuffer.baseAddress,
                        Int32(rows),
                        Int32(inner),
                        rhsBuffer.baseAddress,
                        Int32(inner),
                        Int32(columns),
                        outputBuffer.baseAddress,
                        outputBuffer.count
                    )
                }
            }
        }

        guard status == 0 else {
            let message = edge_cmlx_last_error().map { String(cString: $0) } ?? "unknown Cmlx error"
            throw EdgeMLXBridgeError.executionFailed(message)
        }
        return output
    }

    private static func affineQuantizedMatmulOutputColumns(
        rows: Int,
        inner: Int,
        weights: EdgeQuantizedTensor,
        transpose: Bool
    ) -> Int? {
        guard rows > 0,
              inner > 0,
              weights.shape.count == 2,
              weights.packedShape.count == 2,
              weights.scaleShape.count == 2,
              weights.mode == .affine,
              weights.bits > 0,
              weights.groupSize > 0,
              supportsVendoredMLXGroupSize(weights.groupSize)
        else {
            return nil
        }

        let expandedPackedColumns = weights.packedShape[1] * 32 / weights.bits
        guard weights.shape[1] == expandedPackedColumns,
              weights.scaleShape[1] * weights.groupSize == expandedPackedColumns
        else {
            return nil
        }

        let expectedInner = transpose ? expandedPackedColumns : weights.shape[0]
        guard inner == expectedInner else {
            return nil
        }
        return transpose ? weights.shape[0] : expandedPackedColumns
    }

    private static func supportsVendoredMLXGroupSize(_ groupSize: Int) -> Bool {
        groupSize == 32 || groupSize == 64 || groupSize == 128
    }

    private static func opaquePointer(to buffer: MTLBuffer) -> UnsafeRawPointer {
        UnsafeRawPointer(Unmanaged.passUnretained(buffer as AnyObject).toOpaque())
    }

    private static func opaqueMutablePointer(to buffer: MTLBuffer) -> UnsafeMutableRawPointer {
        Unmanaged.passUnretained(buffer as AnyObject).toOpaque()
    }

    private static func opaqueMutablePointer(to commandBuffer: MTLCommandBuffer) -> UnsafeMutableRawPointer {
        Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    }

    private static func makeBuffer<T>(
        runtime: EdgeMetalRuntime,
        values: [T],
        label: String
    ) throws -> MTLBuffer {
        let byteCount = values.count * MemoryLayout<T>.stride
        guard byteCount > 0 else {
            throw EdgeMLXBridgeError.invalidShape
        }
        guard let buffer = values.withUnsafeBytes({ bytes in
            runtime.device.makeBuffer(
                bytes: bytes.baseAddress!,
                length: byteCount,
                options: [.storageModeShared]
            )
        }) else {
            throw EdgeMLXBridgeError.executionFailed("failed to allocate \(label)")
        }
        buffer.label = label
        return buffer
    }

    static func mtlBufferRetainCountForTesting(_ buffer: MTLBuffer) -> UInt64 {
        edge_cmlx_mtl_buffer_retain_count(opaquePointer(to: buffer))
    }
}
