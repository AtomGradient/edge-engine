// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public enum Q4WeightMatrixError: Error, Equatable {
    case invalidShape
    case invalidGroupSize
    case invalidPackedStorage
}

/// Symmetric int4 weight matrix used by the native runtime spike.
///
/// The matrix is logically row-major `[rows, columns]`. Quantization groups are
/// laid out over the flattened row-major storage. Each group stores a single
/// scale and signed 4-bit values in two's-complement nibbles.
public struct Q4WeightMatrix: Equatable, Sendable {
    public let rows: Int
    public let columns: Int
    public let groupSize: Int
    public let packedValues: [UInt8]
    public let scales: [Float]

    public init(
        rows: Int,
        columns: Int,
        groupSize: Int,
        packedValues: [UInt8],
        scales: [Float]
    ) throws {
        guard rows > 0, columns > 0 else { throw Q4WeightMatrixError.invalidShape }
        guard groupSize > 0 else { throw Q4WeightMatrixError.invalidGroupSize }

        let elementCount = rows * columns
        let expectedPackedBytes = (elementCount + 1) / 2
        let expectedGroups = (elementCount + groupSize - 1) / groupSize
        guard packedValues.count == expectedPackedBytes,
              scales.count == expectedGroups
        else {
            throw Q4WeightMatrixError.invalidPackedStorage
        }

        self.rows = rows
        self.columns = columns
        self.groupSize = groupSize
        self.packedValues = packedValues
        self.scales = scales
    }

    public static func quantizeSymmetric(
        _ values: [Float],
        rows: Int,
        columns: Int,
        groupSize: Int
    ) throws -> Q4WeightMatrix {
        guard rows > 0, columns > 0, values.count == rows * columns else {
            throw Q4WeightMatrixError.invalidShape
        }
        guard groupSize > 0 else { throw Q4WeightMatrixError.invalidGroupSize }

        let groupCount = (values.count + groupSize - 1) / groupSize
        var quantized = Array(repeating: Int8.zero, count: values.count)
        var scales = Array(repeating: Float.zero, count: groupCount)

        for group in 0..<groupCount {
            let start = group * groupSize
            let end = min(values.count, start + groupSize)
            let maxAbs = values[start..<end].map { abs($0) }.max() ?? 0
            let scale = max(maxAbs / 7.0, Float.leastNonzeroMagnitude)
            scales[group] = scale

            for index in start..<end {
                let raw = Int((values[index] / scale).rounded())
                let clamped = max(-8, min(7, raw))
                quantized[index] = Int8(clamped)
            }
        }

        var packed = Array(repeating: UInt8.zero, count: (values.count + 1) / 2)
        for index in 0..<quantized.count {
            let nibble = UInt8(bitPattern: quantized[index]) & 0x0F
            let byteIndex = index / 2
            if index.isMultiple(of: 2) {
                packed[byteIndex] = (packed[byteIndex] & 0xF0) | nibble
            } else {
                packed[byteIndex] = (packed[byteIndex] & 0x0F) | (nibble << 4)
            }
        }

        return try Q4WeightMatrix(
            rows: rows,
            columns: columns,
            groupSize: groupSize,
            packedValues: packed,
            scales: scales
        )
    }

    public func dequantizedValue(at index: Int) -> Float {
        let byte = packedValues[index / 2]
        let rawNibble = index.isMultiple(of: 2) ? (byte & 0x0F) : ((byte >> 4) & 0x0F)
        let signed = rawNibble >= 8 ? Int(rawNibble) - 16 : Int(rawNibble)
        let scale = scales[index / groupSize]
        return Float(signed) * scale
    }

    public func dequantizedValues() -> [Float] {
        (0..<(rows * columns)).map { dequantizedValue(at: $0) }
    }
}
