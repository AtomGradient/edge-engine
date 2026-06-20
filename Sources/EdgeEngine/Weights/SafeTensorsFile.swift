// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Metal

public enum SafeTensorsError: Error, Equatable {
    case fileTooSmall
    case invalidHeaderLength(UInt64)
    case invalidHeaderUTF8
    case invalidHeaderJSON
    case invalidShape(tensorName: String, shape: [Int])
    case invalidDataOffsets(tensorName: String, offsets: [Int])
    case tensorNotFound(String)
    case unsupportedTensorDType(tensorName: String, dtype: String)
    case tensorByteCountMismatch(tensorName: String, expected: Int, actual: Int)
    case tensorRankMismatch(tensorName: String, expected: Int, actual: Int)
}

public struct SafeTensorMetadata: Equatable, Sendable {
    public var name: String
    public var dtype: String
    public var shape: [Int]
    public var dataOffsets: Range<Int>

    public init(name: String, dtype: String, shape: [Int], dataOffsets: Range<Int>) {
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.dataOffsets = dataOffsets
    }

    public var elementCount: Int {
        shape.reduce(1, *)
    }
}

public struct SafeTensorDataSlice: Sendable {
    public let data: Data
    public let range: Range<Int>

    public init(data: Data, range: Range<Int>? = nil) {
        self.data = data
        self.range = range ?? 0..<data.count
    }

    public var count: Int {
        range.count
    }

    public func materializedData() -> Data {
        data.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress else {
                return Data()
            }
            return Data(
                bytes: baseAddress.advanced(by: range.lowerBound),
                count: range.count
            )
        }
    }

    public func withUnsafeBytes<R>(
        _ body: (UnsafeRawBufferPointer) throws -> R
    ) rethrows -> R {
        try data.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress else {
                return try body(UnsafeRawBufferPointer(start: nil, count: 0))
            }
            return try body(
                UnsafeRawBufferPointer(
                    start: baseAddress.advanced(by: range.lowerBound),
                    count: range.count
                )
            )
        }
    }
}

public protocol SafeTensorsSource: Sendable {
    var metadata: [String: String] { get }
    var tensors: [String: SafeTensorMetadata] { get }
    func tensorSlice(named name: String) throws -> SafeTensorDataSlice
    func tensorData(named name: String) throws -> Data
}

public extension SafeTensorsSource {
    func metadata(named name: String) throws -> SafeTensorMetadata {
        guard let metadata = tensors[name] else {
            throw SafeTensorsError.tensorNotFound(name)
        }
        return metadata
    }

    func tensorSlice(named name: String) throws -> SafeTensorDataSlice {
        try SafeTensorDataSlice(data: tensorData(named: name))
    }

    func loadFloat32Tensor(named name: String, runtime: EdgeMetalRuntime) throws -> EdgeTensor {
        let metadata = try metadata(named: name)
        return try EdgeTensor(
            float32: floatingPointValues(
                from: tensorSlice(named: name),
                metadata: metadata
            ),
            shape: EdgeTensorShape(metadata.shape),
            runtime: runtime
        )
    }

    func loadFloat32TensorTransposed2D(
        named name: String,
        runtime: EdgeMetalRuntime
    ) throws -> EdgeTensor {
        let metadata = try metadata(named: name)
        guard metadata.shape.count == 2 else {
            throw SafeTensorsError.tensorRankMismatch(
                tensorName: name,
                expected: 2,
                actual: metadata.shape.count
            )
        }
        let rows = metadata.shape[0]
        let columns = metadata.shape[1]
        let sourceValues = try floatingPointValues(
            from: tensorSlice(named: name),
            metadata: metadata
        )
        var transposed = Array(repeating: Float.zero, count: sourceValues.count)
        for row in 0..<rows {
            for column in 0..<columns {
                transposed[column * rows + row] = sourceValues[row * columns + column]
            }
        }

        return try EdgeTensor(
            float32: transposed,
            shape: EdgeTensorShape([columns, rows]),
            runtime: runtime
        )
    }

    func loadQuantizedTensor(
        weightName: String,
        scalesName: String,
        biasesName: String?,
        groupSize: Int,
        bits: Int
    ) throws -> EdgeQuantizedTensor {
        let weightMetadata = try metadata(named: weightName)
        let scalesMetadata = try metadata(named: scalesName)
        let biasesMetadata = try biasesName.map { try metadata(named: $0) }

        guard weightMetadata.dtype == "U32" else {
            throw SafeTensorsError.unsupportedTensorDType(
                tensorName: weightName,
                dtype: weightMetadata.dtype
            )
        }

        let packedData = try tensorSlice(named: weightName)
        let expectedPackedByteCount = weightMetadata.elementCount * MemoryLayout<UInt32>.stride
        guard packedData.count == expectedPackedByteCount else {
            throw SafeTensorsError.tensorByteCountMismatch(
                tensorName: weightName,
                expected: expectedPackedByteCount,
                actual: packedData.count
            )
        }
        let scales = try floatingPointValues(
            from: tensorSlice(named: scalesName),
            metadata: scalesMetadata
        )
        let biases = try biasesMetadata.map { metadata in
            try floatingPointValues(
                from: tensorSlice(named: metadata.name),
                metadata: metadata
            )
        }

        var logicalShape = scalesMetadata.shape
        guard let scaleLastDimension = logicalShape.last else {
            throw SafeTensorsError.invalidShape(
                tensorName: scalesName,
                shape: scalesMetadata.shape
            )
        }
        logicalShape[logicalShape.count - 1] = scaleLastDimension * groupSize

        return try EdgeQuantizedTensor(
            shape: logicalShape,
            packedShape: weightMetadata.shape,
            scaleShape: scalesMetadata.shape,
            groupSize: groupSize,
            bits: bits,
            mode: .affine,
            packedStorage: .safeTensorSlice(packedData),
            scales: scales,
            biases: biases
        )
    }
}

public struct SafeTensorsFile: SafeTensorsSource, Sendable {
    public let data: Data
    public let headerLength: Int
    public let metadata: [String: String]
    public let tensors: [String: SafeTensorMetadata]

    public init(data: Data) throws {
        guard data.count >= 8 else {
            throw SafeTensorsError.fileTooSmall
        }

        let headerLength64 = data.prefix(8).enumerated().reduce(UInt64(0)) { result, element in
            result | (UInt64(element.element) << UInt64(element.offset * 8))
        }
        guard headerLength64 <= UInt64(Int.max) else {
            throw SafeTensorsError.invalidHeaderLength(headerLength64)
        }
        let headerLength = Int(headerLength64)
        let headerStart = 8
        guard headerLength <= data.count - headerStart else {
            throw SafeTensorsError.invalidHeaderLength(headerLength64)
        }
        let headerEnd = headerStart + headerLength

        let headerData = data[headerStart..<headerEnd]
        let header = try JSONDecoder().decode(SafeTensorsHeader.self, from: headerData)
        let dataSectionStart = headerEnd
        let dataSectionLength = data.count - dataSectionStart
        var tensors: [String: SafeTensorMetadata] = [:]
        tensors.reserveCapacity(header.tensors.count)

        for (name, tensor) in header.tensors {
            guard tensor.shape.allSatisfy({ $0 > 0 }) else {
                throw SafeTensorsError.invalidShape(tensorName: name, shape: tensor.shape)
            }
            guard tensor.dataOffsets.count == 2 else {
                throw SafeTensorsError.invalidDataOffsets(tensorName: name, offsets: tensor.dataOffsets)
            }
            let start = tensor.dataOffsets[0]
            let end = tensor.dataOffsets[1]
            guard start >= 0, end >= start, end <= dataSectionLength else {
                throw SafeTensorsError.invalidDataOffsets(tensorName: name, offsets: tensor.dataOffsets)
            }
            tensors[name] = SafeTensorMetadata(
                name: name,
                dtype: tensor.dtype,
                shape: tensor.shape,
                dataOffsets: start..<end
            )
        }

        self.data = data
        self.headerLength = headerLength
        self.metadata = header.metadata
        self.tensors = tensors
    }

    public func tensorSlice(named name: String) throws -> SafeTensorDataSlice {
        let metadata = try metadata(named: name)
        let dataSectionStart = 8 + headerLength
        return SafeTensorDataSlice(
            data: data,
            range: (dataSectionStart + metadata.dataOffsets.lowerBound)..<(dataSectionStart + metadata.dataOffsets.upperBound)
        )
    }

    public func tensorData(named name: String) throws -> Data {
        try tensorSlice(named: name).materializedData()
    }
}

public struct SafeTensorsShardFile: SafeTensorsSource, Sendable {
    public let url: URL
    public let headerLength: Int
    public let dataSectionStart: Int
    public let metadata: [String: String]
    public let tensors: [String: SafeTensorMetadata]
    private let source: SafeTensorsFile

    public init(url: URL) throws {
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        let source = try SafeTensorsFile(data: data)
        self.url = url
        self.headerLength = source.headerLength
        self.dataSectionStart = 8 + source.headerLength
        self.metadata = source.metadata
        self.tensors = source.tensors
        self.source = source
    }

    public func tensorData(named name: String) throws -> Data {
        try source.tensorData(named: name)
    }

    public func tensorSlice(named name: String) throws -> SafeTensorDataSlice {
        try source.tensorSlice(named: name)
    }
}

private func uint32Values(from data: Data, metadata: SafeTensorMetadata) throws -> [UInt32] {
    let expectedByteCount = metadata.elementCount * MemoryLayout<UInt32>.stride
    guard data.count == expectedByteCount else {
        throw SafeTensorsError.tensorByteCountMismatch(
            tensorName: metadata.name,
            expected: expectedByteCount,
            actual: data.count
        )
    }
    return data.withUnsafeBytes { bytes in
        (0..<metadata.elementCount).map { index in
            let bitPattern = bytes.loadUnaligned(
                fromByteOffset: index * MemoryLayout<UInt32>.stride,
                as: UInt32.self
            )
            return UInt32(littleEndian: bitPattern)
        }
    }
}

private func floatingPointValues(from data: SafeTensorDataSlice, metadata: SafeTensorMetadata) throws -> [Float] {
    switch metadata.dtype {
    case "F32":
        let expectedByteCount = metadata.elementCount * MemoryLayout<Float>.stride
        guard data.count == expectedByteCount else {
            throw SafeTensorsError.tensorByteCountMismatch(
                tensorName: metadata.name,
                expected: expectedByteCount,
                actual: data.count
            )
        }
        return float32Values(from: data)
    case "F16":
        let expectedByteCount = metadata.elementCount * MemoryLayout<UInt16>.stride
        guard data.count == expectedByteCount else {
            throw SafeTensorsError.tensorByteCountMismatch(
                tensorName: metadata.name,
                expected: expectedByteCount,
                actual: data.count
            )
        }
        return data.withUnsafeBytes { bytes in
            (0..<metadata.elementCount).map { index in
                let bitPattern = bytes.loadUnaligned(
                    fromByteOffset: index * MemoryLayout<UInt16>.stride,
                    as: UInt16.self
                )
                return Float(Float16(bitPattern: UInt16(littleEndian: bitPattern)))
            }
        }
    case "BF16":
        let expectedByteCount = metadata.elementCount * MemoryLayout<UInt16>.stride
        guard data.count == expectedByteCount else {
            throw SafeTensorsError.tensorByteCountMismatch(
                tensorName: metadata.name,
                expected: expectedByteCount,
                actual: data.count
            )
        }
        return data.withUnsafeBytes { bytes in
            (0..<metadata.elementCount).map { index in
                let bitPattern = bytes.loadUnaligned(
                    fromByteOffset: index * MemoryLayout<UInt16>.stride,
                    as: UInt16.self
                )
                return Float(bitPattern: UInt32(UInt16(littleEndian: bitPattern)) << 16)
            }
        }
    default:
        throw SafeTensorsError.unsupportedTensorDType(
            tensorName: metadata.name,
            dtype: metadata.dtype
        )
    }
}

private func float32Values(from data: SafeTensorDataSlice) -> [Float] {
    data.withUnsafeBytes { bytes in
        let count = data.count / MemoryLayout<Float>.stride
        return (0..<count).map { index in
            let bitPattern = bytes.loadUnaligned(
                fromByteOffset: index * MemoryLayout<Float>.stride,
                as: UInt32.self
            )
            return Float(bitPattern: UInt32(littleEndian: bitPattern))
        }
    }
}

private struct SafeTensorsHeader: Decodable {
    var metadata: [String: String]
    var tensors: [String: SafeTensorsHeaderTensor]

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: DynamicCodingKey.self)
        let metadataKey = DynamicCodingKey(stringValue: "__metadata__")!
        metadata = (try? container.decode([String: String].self, forKey: metadataKey)) ?? [:]
        var tensors: [String: SafeTensorsHeaderTensor] = [:]
        for key in container.allKeys where key.stringValue != "__metadata__" {
            tensors[key.stringValue] = try container.decode(SafeTensorsHeaderTensor.self, forKey: key)
        }
        self.tensors = tensors
    }
}

private struct SafeTensorsHeaderTensor: Decodable {
    var dtype: String
    var shape: [Int]
    var dataOffsets: [Int]

    private enum CodingKeys: String, CodingKey {
        case dtype
        case shape
        case dataOffsets = "data_offsets"
    }
}

private struct DynamicCodingKey: CodingKey {
    var stringValue: String
    var intValue: Int?

    init?(stringValue: String) {
        self.stringValue = stringValue
        self.intValue = nil
    }

    init?(intValue: Int) {
        self.stringValue = "\(intValue)"
        self.intValue = intValue
    }
}
