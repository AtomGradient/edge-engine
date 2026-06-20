// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum EdgeQuantizationMode: String, Codable, Equatable, Sendable {
    case affine
}

public enum EdgeQuantizedTensorError: Error, Equatable {
    case invalidShape(shape: [Int], packedShape: [Int], scaleShape: [Int])
    case invalidGroupSize(Int)
    case unsupportedBits(Int)
    case invalidPackedStorage(expected: Int, actual: Int)
    case invalidScaleStorage(expected: Int, actual: Int)
    case invalidBiasStorage(expected: Int, actual: Int)
    case hostStorageReleased
}

/// MLX-compatible affine quantized tensor storage.
///
/// The logical tensor shape is the dequantized shape. `packedShape` is the
/// safetensors `U32` storage shape, and `scaleShape` stores one affine
/// `(scale, bias)` pair per group along the last logical dimension.
public struct EdgeQuantizedTensor: Sendable {
    public let shape: [Int]
    public let packedShape: [Int]
    public let scaleShape: [Int]
    public let groupSize: Int
    public let bits: Int
    public let mode: EdgeQuantizationMode
    private let storage: EdgeQuantizedTensorStorage

    public var packedValues: [UInt32] {
        storage.packedValues
    }

    public var scales: [Float] {
        storage.scales
    }

    public var biases: [Float]? {
        storage.biases
    }

    public init(
        shape: [Int],
        packedShape: [Int],
        scaleShape: [Int],
        groupSize: Int,
        bits: Int,
        mode: EdgeQuantizationMode = .affine,
        packedValues: [UInt32],
        scales: [Float],
        biases: [Float]? = nil
    ) throws {
        guard groupSize > 0 else {
            throw EdgeQuantizedTensorError.invalidGroupSize(groupSize)
        }
        guard [2, 3, 4, 5, 6, 8].contains(bits) else {
            throw EdgeQuantizedTensorError.unsupportedBits(bits)
        }
        guard shape.count == packedShape.count,
              shape.count == scaleShape.count,
              !shape.isEmpty,
              shape.dropLast() == packedShape.dropLast(),
              shape.dropLast() == scaleShape.dropLast(),
              scaleShape.last.map({ $0 * groupSize }) == shape.last
        else {
            throw EdgeQuantizedTensorError.invalidShape(
                shape: shape,
                packedShape: packedShape,
                scaleShape: scaleShape
            )
        }

        let expectedPackedLastDimension = (shape.last! * bits + 31) / 32
        guard packedShape.last == expectedPackedLastDimension else {
            throw EdgeQuantizedTensorError.invalidShape(
                shape: shape,
                packedShape: packedShape,
                scaleShape: scaleShape
            )
        }

        let expectedPackedCount = packedShape.reduce(1, *)
        guard packedValues.count == expectedPackedCount else {
            throw EdgeQuantizedTensorError.invalidPackedStorage(
                expected: expectedPackedCount,
                actual: packedValues.count
            )
        }

        let expectedScaleCount = scaleShape.reduce(1, *)
        guard scales.count == expectedScaleCount else {
            throw EdgeQuantizedTensorError.invalidScaleStorage(
                expected: expectedScaleCount,
                actual: scales.count
            )
        }
        if let biases, biases.count != expectedScaleCount {
            throw EdgeQuantizedTensorError.invalidBiasStorage(
                expected: expectedScaleCount,
                actual: biases.count
            )
        }

        self.shape = shape
        self.packedShape = packedShape
        self.scaleShape = scaleShape
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.storage = EdgeQuantizedTensorStorage(
            packedValues: packedValues,
            scales: scales,
            biases: biases
        )
    }

    public init(
        shape: [Int],
        packedShape: [Int],
        scaleShape: [Int],
        groupSize: Int,
        bits: Int,
        mode: EdgeQuantizationMode = .affine,
        packedData: Data,
        scales: [Float],
        biases: [Float]? = nil
    ) throws {
        guard groupSize > 0 else {
            throw EdgeQuantizedTensorError.invalidGroupSize(groupSize)
        }
        guard [2, 3, 4, 5, 6, 8].contains(bits) else {
            throw EdgeQuantizedTensorError.unsupportedBits(bits)
        }
        guard shape.count == packedShape.count,
              shape.count == scaleShape.count,
              !shape.isEmpty,
              shape.dropLast() == packedShape.dropLast(),
              shape.dropLast() == scaleShape.dropLast(),
              scaleShape.last.map({ $0 * groupSize }) == shape.last
        else {
            throw EdgeQuantizedTensorError.invalidShape(
                shape: shape,
                packedShape: packedShape,
                scaleShape: scaleShape
            )
        }

        let expectedPackedLastDimension = (shape.last! * bits + 31) / 32
        guard packedShape.last == expectedPackedLastDimension else {
            throw EdgeQuantizedTensorError.invalidShape(
                shape: shape,
                packedShape: packedShape,
                scaleShape: scaleShape
            )
        }

        let expectedPackedCount = packedShape.reduce(1, *)
        let expectedPackedByteCount = expectedPackedCount * MemoryLayout<UInt32>.stride
        guard packedData.count == expectedPackedByteCount else {
            throw EdgeQuantizedTensorError.invalidPackedStorage(
                expected: expectedPackedCount,
                actual: packedData.count / MemoryLayout<UInt32>.stride
            )
        }

        let expectedScaleCount = scaleShape.reduce(1, *)
        guard scales.count == expectedScaleCount else {
            throw EdgeQuantizedTensorError.invalidScaleStorage(
                expected: expectedScaleCount,
                actual: scales.count
            )
        }
        if let biases, biases.count != expectedScaleCount {
            throw EdgeQuantizedTensorError.invalidBiasStorage(
                expected: expectedScaleCount,
                actual: biases.count
            )
        }

        self.shape = shape
        self.packedShape = packedShape
        self.scaleShape = scaleShape
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.storage = EdgeQuantizedTensorStorage(
            packedData: packedData,
            scales: scales,
            biases: biases
        )
    }

    init(
        shape: [Int],
        packedShape: [Int],
        scaleShape: [Int],
        groupSize: Int,
        bits: Int,
        mode: EdgeQuantizationMode = .affine,
        packedStorage: EdgeQuantizedByteStorage,
        scales: [Float],
        biases: [Float]? = nil
    ) throws {
        guard groupSize > 0 else {
            throw EdgeQuantizedTensorError.invalidGroupSize(groupSize)
        }
        guard [2, 3, 4, 5, 6, 8].contains(bits) else {
            throw EdgeQuantizedTensorError.unsupportedBits(bits)
        }
        guard shape.count == packedShape.count,
              shape.count == scaleShape.count,
              !shape.isEmpty,
              shape.dropLast() == packedShape.dropLast(),
              shape.dropLast() == scaleShape.dropLast(),
              scaleShape.last.map({ $0 * groupSize }) == shape.last
        else {
            throw EdgeQuantizedTensorError.invalidShape(
                shape: shape,
                packedShape: packedShape,
                scaleShape: scaleShape
            )
        }

        let expectedPackedLastDimension = (shape.last! * bits + 31) / 32
        guard packedShape.last == expectedPackedLastDimension else {
            throw EdgeQuantizedTensorError.invalidShape(
                shape: shape,
                packedShape: packedShape,
                scaleShape: scaleShape
            )
        }

        let expectedPackedCount = packedShape.reduce(1, *)
        let expectedPackedByteCount = expectedPackedCount * MemoryLayout<UInt32>.stride
        guard packedStorage.count == expectedPackedByteCount else {
            throw EdgeQuantizedTensorError.invalidPackedStorage(
                expected: expectedPackedCount,
                actual: packedStorage.count / MemoryLayout<UInt32>.stride
            )
        }

        let expectedScaleCount = scaleShape.reduce(1, *)
        guard scales.count == expectedScaleCount else {
            throw EdgeQuantizedTensorError.invalidScaleStorage(
                expected: expectedScaleCount,
                actual: scales.count
            )
        }
        if let biases, biases.count != expectedScaleCount {
            throw EdgeQuantizedTensorError.invalidBiasStorage(
                expected: expectedScaleCount,
                actual: biases.count
            )
        }

        self.shape = shape
        self.packedShape = packedShape
        self.scaleShape = scaleShape
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.storage = EdgeQuantizedTensorStorage(
            packedStorage: packedStorage,
            scales: scales,
            biases: biases
        )
    }

    public var elementCount: Int {
        shape.reduce(1, *)
    }

    public var packedValueCount: Int {
        packedShape.reduce(1, *)
    }

    public var scaleCount: Int {
        scaleShape.reduce(1, *)
    }

    public var packedByteCount: Int {
        packedValueCount * MemoryLayout<UInt32>.stride
    }

    public var scalesByteCount: Int {
        scaleCount * MemoryLayout<Float>.stride
    }

    public var storedBiasByteCount: Int {
        storage.hasBiases ? scaleCount * MemoryLayout<Float>.stride : 0
    }

    public var affineBiasBufferByteCount: Int {
        scaleCount * MemoryLayout<Float>.stride
    }

    public var hostStorageByteCount: Int {
        storage.hostStorageByteCount
    }

    public var hostStorageReleased: Bool {
        storage.isReleased
    }

    var storageIdentifier: ObjectIdentifier {
        ObjectIdentifier(storage)
    }

    public var packedWordsPerLogicalRow: Int {
        packedShape.last!
    }

    @discardableResult
    public func releaseHostStorage() -> Int {
        storage.releaseHostStorage()
    }

    public func withHostStorage<R>(
        _ body: (
            _ packedValues: [UInt32],
            _ scales: [Float],
            _ biases: [Float]?
        ) throws -> R
    ) throws -> R {
        try storage.withHostStorage(body)
    }

    public func withHostStorageBytes<R>(
        _ body: (
            _ packedData: Data,
            _ scalesData: Data,
            _ biasesData: Data?
        ) throws -> R
    ) throws -> R {
        try storage.withHostStorageBytes(body)
    }

    func withHostStorageByteStorage<R>(
        _ body: (
            _ packedData: EdgeQuantizedByteStorage,
            _ scalesData: Data,
            _ biasesData: Data?
        ) throws -> R
    ) throws -> R {
        try storage.withHostStorageByteStorage(body)
    }

    public func dequantizedValue(at linearIndex: Int) -> Float {
        let columns = shape.last!
        let scaleColumns = scaleShape.last!
        let rowMajorPrefix = linearIndex / columns
        let column = linearIndex - rowMajorPrefix * columns
        let bitOffset = column * bits
        let packedIndex = rowMajorPrefix * packedWordsPerLogicalRow + bitOffset / 32
        let shift = bitOffset % 32
        let scaleIndex = rowMajorPrefix * scaleColumns + column / groupSize
        let mask = UInt32((1 << bits) - 1)
        var raw = packedValues[packedIndex] >> UInt32(shift)
        if shift + bits > 32 {
            raw |= packedValues[packedIndex + 1] << UInt32(32 - shift)
        }
        raw &= mask
        return Float(raw) * scales[scaleIndex] + (biases?[scaleIndex] ?? 0)
    }

    public func dequantizedValues() -> [Float] {
        (0..<elementCount).map { dequantizedValue(at: $0) }
    }
}

private func uint32Data(from values: [UInt32]) -> Data {
    let littleEndianValues = values.map { $0.littleEndian }
    return littleEndianValues.withUnsafeBytes { Data($0) }
}

private func float32Data(from values: [Float]) -> Data {
    let bitPatterns = values.map { $0.bitPattern.littleEndian }
    return bitPatterns.withUnsafeBytes { Data($0) }
}

enum EdgeQuantizedByteStorage: Sendable {
    case data(Data)
    case safeTensorSlice(SafeTensorDataSlice)

    var count: Int {
        switch self {
        case .data(let data):
            data.count
        case .safeTensorSlice(let slice):
            slice.count
        }
    }

    func materializedData() -> Data {
        switch self {
        case .data(let data):
            data
        case .safeTensorSlice(let slice):
            slice.materializedData()
        }
    }

    func withUnsafeBytes<R>(
        _ body: (UnsafeRawBufferPointer) throws -> R
    ) rethrows -> R {
        switch self {
        case .data(let data):
            return try data.withUnsafeBytes(body)
        case .safeTensorSlice(let slice):
            return try slice.withUnsafeBytes(body)
        }
    }

    func pageAlignedNoCopyDataRange(
        pageSize: Int,
        offsetAlignment: Int = 1
    ) -> (
        data: Data,
        range: Range<Int>,
        offset: Int
    )? {
        guard pageSize > 0, offsetAlignment > 0 else {
            return nil
        }
        switch self {
        case .data:
            return nil
        case .safeTensorSlice(let slice):
            let lower = slice.range.lowerBound / pageSize * pageSize
            let upper = min(
                slice.data.count,
                ((slice.range.upperBound + pageSize - 1) / pageSize) * pageSize
            )
            guard lower < upper else {
                return nil
            }
            let offset = slice.range.lowerBound - lower
            guard offset.isMultiple(of: offsetAlignment) else {
                return nil
            }
            return (
                data: slice.data,
                range: lower..<upper,
                offset: offset
            )
        }
    }
}

private func uint32Values(fromLittleEndianBytes storage: EdgeQuantizedByteStorage) -> [UInt32] {
    storage.withUnsafeBytes { bytes in
        let count = storage.count / MemoryLayout<UInt32>.stride
        return (0..<count).map { index in
            let bitPattern = bytes.loadUnaligned(
                fromByteOffset: index * MemoryLayout<UInt32>.stride,
                as: UInt32.self
            )
            return UInt32(littleEndian: bitPattern)
        }
    }
}

private func uint32Values(fromLittleEndianData data: Data) -> [UInt32] {
    data.withUnsafeBytes { bytes in
        let count = data.count / MemoryLayout<UInt32>.stride
        return (0..<count).map { index in
            let bitPattern = bytes.loadUnaligned(
                fromByteOffset: index * MemoryLayout<UInt32>.stride,
                as: UInt32.self
            )
            return UInt32(littleEndian: bitPattern)
        }
    }
}

private func float32Values(fromLittleEndianData data: Data) -> [Float] {
    data.withUnsafeBytes { bytes in
        let count = data.count / MemoryLayout<UInt32>.stride
        return (0..<count).map { index in
            let bitPattern = bytes.loadUnaligned(
                fromByteOffset: index * MemoryLayout<UInt32>.stride,
                as: UInt32.self
            )
            return Float(bitPattern: UInt32(littleEndian: bitPattern))
        }
    }
}

extension EdgeQuantizedTensor: Equatable {
    public static func == (lhs: EdgeQuantizedTensor, rhs: EdgeQuantizedTensor) -> Bool {
        lhs.shape == rhs.shape
            && lhs.packedShape == rhs.packedShape
            && lhs.scaleShape == rhs.scaleShape
            && lhs.groupSize == rhs.groupSize
            && lhs.bits == rhs.bits
            && lhs.mode == rhs.mode
            && lhs.storage == rhs.storage
    }
}

private final class EdgeQuantizedTensorStorage: @unchecked Sendable, Equatable {
    let hasBiases: Bool

    private let lock = NSLock()
    private var packedValuesStorage: [UInt32]?
    private var packedByteStorage: EdgeQuantizedByteStorage?
    private var scalesStorage: [Float]?
    private var scalesDataStorage: Data?
    private var biasesStorage: [Float]?
    private var biasesDataStorage: Data?

    init(
        packedValues: [UInt32],
        scales: [Float],
        biases: [Float]?
    ) {
        self.hasBiases = biases != nil
        self.packedValuesStorage = packedValues
        self.packedByteStorage = nil
        self.scalesStorage = scales
        self.scalesDataStorage = nil
        self.biasesStorage = biases
        self.biasesDataStorage = nil
    }

    init(
        packedData: Data,
        scales: [Float],
        biases: [Float]?
    ) {
        self.hasBiases = biases != nil
        self.packedValuesStorage = nil
        self.packedByteStorage = .data(packedData)
        self.scalesStorage = nil
        self.scalesDataStorage = float32Data(from: scales)
        self.biasesStorage = nil
        self.biasesDataStorage = biases.map { float32Data(from: $0) }
    }

    init(
        packedStorage: EdgeQuantizedByteStorage,
        scales: [Float],
        biases: [Float]?
    ) {
        self.hasBiases = biases != nil
        self.packedValuesStorage = nil
        self.packedByteStorage = packedStorage
        self.scalesStorage = nil
        self.scalesDataStorage = float32Data(from: scales)
        self.biasesStorage = nil
        self.biasesDataStorage = biases.map { float32Data(from: $0) }
    }

    var packedValues: [UInt32] {
        lock.lock()
        defer { lock.unlock() }
        if let packedValuesStorage {
            return packedValuesStorage
        }
        if let packedByteStorage {
            return uint32Values(fromLittleEndianBytes: packedByteStorage)
        }
        preconditionFailure("EdgeQuantizedTensor host packed storage was released")
    }

    var scales: [Float] {
        lock.lock()
        defer { lock.unlock() }
        if let scalesStorage {
            return scalesStorage
        }
        if let scalesDataStorage {
            return float32Values(fromLittleEndianData: scalesDataStorage)
        }
        preconditionFailure("EdgeQuantizedTensor host scale storage was released")
    }

    var biases: [Float]? {
        lock.lock()
        defer { lock.unlock() }
        if let biasesStorage {
            return biasesStorage
        }
        if let biasesDataStorage {
            return float32Values(fromLittleEndianData: biasesDataStorage)
        }
        guard hasBiases else { return nil }
        preconditionFailure("EdgeQuantizedTensor host bias storage was released")
    }

    var isReleased: Bool {
        lock.lock()
        defer { lock.unlock() }
        return isReleasedUnlocked
    }

    var hostStorageByteCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return (packedValuesStorage?.count ?? 0) * MemoryLayout<UInt32>.stride
            + (packedByteStorage?.count ?? 0)
            + (scalesStorage?.count ?? 0) * MemoryLayout<Float>.stride
            + (scalesDataStorage?.count ?? 0)
            + (biasesStorage?.count ?? 0) * MemoryLayout<Float>.stride
            + (biasesDataStorage?.count ?? 0)
    }

    private var isReleasedUnlocked: Bool {
        packedValuesStorage == nil
            && packedByteStorage == nil
            && scalesStorage == nil
            && scalesDataStorage == nil
            && biasesStorage == nil
            && biasesDataStorage == nil
    }

    private var packedBytes: EdgeQuantizedByteStorage {
        get throws {
            if let packedByteStorage {
                return packedByteStorage
            }
            if let packedValuesStorage {
                return .data(uint32Data(from: packedValuesStorage))
            }
            throw EdgeQuantizedTensorError.hostStorageReleased
        }
    }

    private var scalesData: Data {
        get throws {
            if let scalesDataStorage {
                return scalesDataStorage
            }
            if let scalesStorage {
                return float32Data(from: scalesStorage)
            }
            throw EdgeQuantizedTensorError.hostStorageReleased
        }
    }

    private var biasesData: Data? {
        get throws {
            if let biasesDataStorage {
                return biasesDataStorage
            }
            if let biasesStorage {
                return float32Data(from: biasesStorage)
            }
            guard hasBiases else { return nil }
            throw EdgeQuantizedTensorError.hostStorageReleased
        }
    }

    func withHostStorage<R>(
        _ body: (
            _ packedValues: [UInt32],
            _ scales: [Float],
            _ biases: [Float]?
        ) throws -> R
    ) throws -> R {
        lock.lock()
        guard !isReleasedUnlocked else {
            lock.unlock()
            throw EdgeQuantizedTensorError.hostStorageReleased
        }
        let packedValues: [UInt32]
        if let packedValuesStorage {
            packedValues = packedValuesStorage
        } else if let packedByteStorage {
            packedValues = uint32Values(fromLittleEndianBytes: packedByteStorage)
        } else {
            lock.unlock()
            throw EdgeQuantizedTensorError.hostStorageReleased
        }
        let scales: [Float]
        if let scalesStorage {
            scales = scalesStorage
        } else if let scalesDataStorage {
            scales = float32Values(fromLittleEndianData: scalesDataStorage)
        } else {
            lock.unlock()
            throw EdgeQuantizedTensorError.hostStorageReleased
        }
        let biases: [Float]?
        if let biasesStorage {
            biases = biasesStorage
        } else if let biasesDataStorage {
            biases = float32Values(fromLittleEndianData: biasesDataStorage)
        } else {
            biases = nil
        }
        lock.unlock()

        return try body(packedValues, scales, biases)
    }

    func withHostStorageBytes<R>(
        _ body: (
            _ packedData: Data,
            _ scalesData: Data,
            _ biasesData: Data?
        ) throws -> R
    ) throws -> R {
        lock.lock()
        guard !isReleasedUnlocked else {
            lock.unlock()
            throw EdgeQuantizedTensorError.hostStorageReleased
        }
        let packedData: Data
        let scalesData: Data
        let biasesData: Data?
        do {
            packedData = try self.packedBytes.materializedData()
            scalesData = try self.scalesData
            biasesData = try self.biasesData
        } catch {
            lock.unlock()
            throw error
        }
        lock.unlock()

        return try body(packedData, scalesData, biasesData)
    }

    func withHostStorageByteStorage<R>(
        _ body: (
            _ packedData: EdgeQuantizedByteStorage,
            _ scalesData: Data,
            _ biasesData: Data?
        ) throws -> R
    ) throws -> R {
        lock.lock()
        guard !isReleasedUnlocked else {
            lock.unlock()
            throw EdgeQuantizedTensorError.hostStorageReleased
        }
        let packedData: EdgeQuantizedByteStorage
        let scalesData: Data
        let biasesData: Data?
        do {
            packedData = try self.packedBytes
            scalesData = try self.scalesData
            biasesData = try self.biasesData
        } catch {
            lock.unlock()
            throw error
        }
        lock.unlock()

        return try body(packedData, scalesData, biasesData)
    }

    @discardableResult
    func releaseHostStorage() -> Int {
        lock.lock()
        defer { lock.unlock() }
        let byteCount = (packedValuesStorage?.count ?? 0) * MemoryLayout<UInt32>.stride
            + (packedByteStorage?.count ?? 0)
            + (scalesStorage?.count ?? 0) * MemoryLayout<Float>.stride
            + (scalesDataStorage?.count ?? 0)
            + (biasesStorage?.count ?? 0) * MemoryLayout<Float>.stride
            + (biasesDataStorage?.count ?? 0)
        packedValuesStorage = nil
        packedByteStorage = nil
        scalesStorage = nil
        scalesDataStorage = nil
        biasesStorage = nil
        biasesDataStorage = nil
        return byteCount
    }

    static func == (lhs: EdgeQuantizedTensorStorage, rhs: EdgeQuantizedTensorStorage) -> Bool {
        if lhs === rhs {
            return true
        }
        do {
            return try lhs.withHostStorage { lhsPacked, lhsScales, lhsBiases in
                try rhs.withHostStorage { rhsPacked, rhsScales, rhsBiases in
                    lhsPacked == rhsPacked
                        && lhsScales == rhsScales
                        && lhsBiases == rhsBiases
                }
            }
        } catch {
            return false
        }
    }
}
