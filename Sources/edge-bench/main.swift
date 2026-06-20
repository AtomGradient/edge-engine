// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Dispatch
import EdgeEngine
import Foundation

@main
struct EdgeBench {
    static func main() throws {
        let arguments = CommandLine.arguments
        let rows = argumentValue("--rows", in: arguments).flatMap(Int.init) ?? 1
        let inner = argumentValue("--inner", in: arguments).flatMap(Int.init) ?? 4096
        let columns = argumentValue("--columns", in: arguments).flatMap(Int.init) ?? 2048
        let bits = argumentValue("--bits", in: arguments).flatMap(Int.init) ?? 6
        let groupSize = argumentValue("--group-size", in: arguments).flatMap(Int.init) ?? 64
        let warmups = argumentValue("--warmups", in: arguments).flatMap(Int.init) ?? 3
        let iterations = argumentValue("--iterations", in: arguments).flatMap(Int.init) ?? 20
        let backend = argumentValue("--backend", in: arguments).flatMap(BenchmarkBackend.init(rawValue:)) ?? .native

        guard inner.isMultiple(of: groupSize) else {
            throw BenchmarkError.invalidGroupSize(inner: inner, groupSize: groupSize)
        }
        guard bits == 4 || bits == 6 else {
            throw BenchmarkError.unsupportedBits(bits)
        }

        let lhsValues = (0..<(rows * inner)).map { Float(Int($0 % 31) - 15) / 15.0 }
        let weights = try makeQuantizedWeights(
            rows: columns,
            columns: inner,
            bits: bits,
            groupSize: groupSize
        )

        let results: [BenchmarkResult]
        switch backend {
        case .native:
            results = [
                try runNative(
                    lhsValues: lhsValues,
                    weights: weights,
                    rows: rows,
                    inner: inner,
                    columns: columns,
                    warmups: warmups,
                    iterations: iterations,
                    bits: bits,
                    groupSize: groupSize
                ),
            ]
        case .mlx:
            results = [
                try runMLX(
                    lhsValues: lhsValues,
                    weights: weights,
                    rows: rows,
                    inner: inner,
                    columns: columns,
                    warmups: warmups,
                    iterations: iterations,
                    bits: bits,
                    groupSize: groupSize
                ),
            ]
        case .executorMLX:
            results = [
                try runExecutorMLX(
                    lhsValues: lhsValues,
                    weights: weights,
                    rows: rows,
                    inner: inner,
                    columns: columns,
                    warmups: warmups,
                    iterations: iterations,
                    bits: bits,
                    groupSize: groupSize
                ),
            ]
        case .both:
            results = [
                try runNative(
                    lhsValues: lhsValues,
                    weights: weights,
                    rows: rows,
                    inner: inner,
                    columns: columns,
                    warmups: warmups,
                    iterations: iterations,
                    bits: bits,
                    groupSize: groupSize
                ),
                try runMLX(
                    lhsValues: lhsValues,
                    weights: weights,
                    rows: rows,
                    inner: inner,
                    columns: columns,
                    warmups: warmups,
                    iterations: iterations,
                    bits: bits,
                    groupSize: groupSize
                ),
            ]
        case .all:
            results = [
                try runNative(
                    lhsValues: lhsValues,
                    weights: weights,
                    rows: rows,
                    inner: inner,
                    columns: columns,
                    warmups: warmups,
                    iterations: iterations,
                    bits: bits,
                    groupSize: groupSize
                ),
                try runMLX(
                    lhsValues: lhsValues,
                    weights: weights,
                    rows: rows,
                    inner: inner,
                    columns: columns,
                    warmups: warmups,
                    iterations: iterations,
                    bits: bits,
                    groupSize: groupSize
                ),
                try runExecutorMLX(
                    lhsValues: lhsValues,
                    weights: weights,
                    rows: rows,
                    inner: inner,
                    columns: columns,
                    warmups: warmups,
                    iterations: iterations,
                    bits: bits,
                    groupSize: groupSize
                ),
            ]
        }

        for result in results {
            print("EdgeBench")
            print("backend=\(result.backend)")
            print("kernel=\(result.kernel)")
            print("lhs=[\(rows),\(inner)] weights=[\(columns),\(inner)] transpose=true bits=\(bits) groupSize=\(groupSize)")
            print(String(format: "avg_ms=%.3f", result.averageMilliseconds))
            print(String(format: "gops=%.2f", result.gops))
            print("iterations=\(iterations) warmups=\(warmups)")
        }
    }

    private static func argumentValue(_ name: String, in arguments: [String]) -> String? {
        guard let index = arguments.firstIndex(of: name), index + 1 < arguments.count else {
            return nil
        }
        return arguments[index + 1]
    }

    private static func runNative(
        lhsValues: [Float],
        weights: EdgeQuantizedTensor,
        rows: Int,
        inner: Int,
        columns: Int,
        warmups: Int,
        iterations: Int,
        bits: Int,
        groupSize: Int
    ) throws -> BenchmarkResult {
        let runtime = try EdgeMetalRuntime(
            configuration: MetalRuntimeConfiguration(
                useMLXQuantizedMatmul: false,
                useMLXQuantizedPrefillMatmul: false
            )
        )
        let executor = try MetalKernelExecutor(runtime: runtime)
        let lhs = try EdgeTensor(float32: lhsValues, shape: EdgeTensorShape([rows, inner]), runtime: runtime)

        for _ in 0..<warmups {
            _ = try executor.affineQuantizedMatmul(lhs, weights: weights, transpose: true)
        }
        runtime.waitForPendingWork()

        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<iterations {
            _ = try executor.affineQuantizedMatmul(lhs, weights: weights, transpose: true)
        }
        runtime.waitForPendingWork()
        let end = DispatchTime.now().uptimeNanoseconds

        return makeResult(
            backend: "native",
            kernel: executor.lastExecutionStats?.operationName ?? "unknown",
            start: start,
            end: end,
            iterations: iterations,
            rows: rows,
            inner: inner,
            columns: columns
        )
    }

    private static func runExecutorMLX(
        lhsValues: [Float],
        weights: EdgeQuantizedTensor,
        rows: Int,
        inner: Int,
        columns: Int,
        warmups: Int,
        iterations: Int,
        bits: Int,
        groupSize: Int
    ) throws -> BenchmarkResult {
        let runtime = try EdgeMetalRuntime(
            configuration: MetalRuntimeConfiguration(useMLXQuantizedMatmul: true)
        )
        let executor = try MetalKernelExecutor(runtime: runtime)
        let lhs = try EdgeTensor(float32: lhsValues, shape: EdgeTensorShape([rows, inner]), runtime: runtime)

        for _ in 0..<warmups {
            _ = try executor.affineQuantizedMatmul(lhs, weights: weights, transpose: true)
        }
        runtime.waitForPendingWork()

        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<iterations {
            _ = try executor.affineQuantizedMatmul(lhs, weights: weights, transpose: true)
        }
        runtime.waitForPendingWork()
        let end = DispatchTime.now().uptimeNanoseconds

        return makeResult(
            backend: "executor-mlx",
            kernel: executor.lastExecutionStats?.operationName ?? "unknown",
            start: start,
            end: end,
            iterations: iterations,
            rows: rows,
            inner: inner,
            columns: columns
        )
    }

    private static func runMLX(
        lhsValues: [Float],
        weights: EdgeQuantizedTensor,
        rows: Int,
        inner: Int,
        columns: Int,
        warmups: Int,
        iterations: Int,
        bits: Int,
        groupSize: Int
    ) throws -> BenchmarkResult {
        for _ in 0..<warmups {
            _ = try EdgeMLXBridge.affineQuantizedMatmulFloat32GPU(
                lhs: lhsValues,
                rows: rows,
                inner: inner,
                weights: weights,
                transpose: true
            )
        }

        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<iterations {
            _ = try EdgeMLXBridge.affineQuantizedMatmulFloat32GPU(
                lhs: lhsValues,
                rows: rows,
                inner: inner,
                weights: weights,
                transpose: true
            )
        }
        let end = DispatchTime.now().uptimeNanoseconds

        return makeResult(
            backend: "mlx",
            kernel: "edge_cmlx_affine_quantized_matmul_f32_gpu",
            start: start,
            end: end,
            iterations: iterations,
            rows: rows,
            inner: inner,
            columns: columns
        )
    }

    private static func makeResult(
        backend: String,
        kernel: String,
        start: UInt64,
        end: UInt64,
        iterations: Int,
        rows: Int,
        inner: Int,
        columns: Int
    ) -> BenchmarkResult {
        let elapsedSeconds = Double(end - start) / 1_000_000_000.0
        let averageMilliseconds = elapsedSeconds / Double(iterations) * 1000.0
        let operations = Double(rows * inner * columns * 2)
        let gops = operations / (elapsedSeconds / Double(iterations)) / 1_000_000_000.0
        return BenchmarkResult(
            backend: backend,
            kernel: kernel,
            averageMilliseconds: averageMilliseconds,
            gops: gops
        )
    }

    private static func makeQuantizedWeights(
        rows: Int,
        columns: Int,
        bits: Int,
        groupSize: Int
    ) throws -> EdgeQuantizedTensor {
        let packedWordsPerRow = (columns * bits + 31) / 32
        let scaleColumns = columns / groupSize
        let mask = UInt32((1 << bits) - 1)
        var packedValues = Array(repeating: UInt32.zero, count: rows * packedWordsPerRow)
        for row in 0..<rows {
            for column in 0..<columns {
                let raw = UInt32((row * 131 + column * 17) & Int(mask))
                pack(raw, bits: bits, column: column, rowOffset: row * packedWordsPerRow, into: &packedValues)
            }
        }
        let scaleBase = 1.0 / Float(mask + 1)
        let scales = (0..<(rows * scaleColumns)).map { index in
            scaleBase * (1.0 + Float(index % 7) / 16.0)
        }
        let biases = Array(repeating: Float.zero, count: rows * scaleColumns)
        return try EdgeQuantizedTensor(
            shape: [rows, columns],
            packedShape: [rows, packedWordsPerRow],
            scaleShape: [rows, scaleColumns],
            groupSize: groupSize,
            bits: bits,
            packedValues: packedValues,
            scales: scales,
            biases: biases
        )
    }

    private static func pack(
        _ rawValue: UInt32,
        bits: Int,
        column: Int,
        rowOffset: Int,
        into packedValues: inout [UInt32]
    ) {
        let bitOffset = column * bits
        let wordOffset = rowOffset + bitOffset / 32
        let shift = bitOffset % 32
        let mask = UInt32((1 << bits) - 1)
        packedValues[wordOffset] |= (rawValue & mask) << UInt32(shift)
        if shift + bits > 32 {
            packedValues[wordOffset + 1] |= (rawValue & mask) >> UInt32(32 - shift)
        }
    }
}

enum BenchmarkBackend: String {
    case native
    case mlx
    case executorMLX = "executor-mlx"
    case both
    case all
}

struct BenchmarkResult {
    var backend: String
    var kernel: String
    var averageMilliseconds: Double
    var gops: Double
}

enum BenchmarkError: Error, CustomStringConvertible {
    case invalidGroupSize(inner: Int, groupSize: Int)
    case unsupportedBits(Int)

    var description: String {
        switch self {
        case .invalidGroupSize(let inner, let groupSize):
            "inner=\(inner) must be divisible by groupSize=\(groupSize)"
        case .unsupportedBits(let bits):
            "benchmark currently targets qmv bits=4/6, got \(bits)"
        }
    }
}
