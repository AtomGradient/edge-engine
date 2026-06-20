// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Metal

public enum EdgeTensorError: Error, Equatable {
    case elementCountMismatch(expected: Int, actual: Int)
    case bufferAllocationFailed(byteCount: Int)
    case unsupportedDataType
    case unsupportedStorageMode
}

/// Minimal tensor storage for native inference.
///
/// This is deliberately small: it owns shape, dtype, and an `MTLBuffer`. Higher
/// level tensor views, strides, quantized storage, and mmap-backed weights are
/// added as the Qwen path requires them.
public final class EdgeTensor {
    public let shape: EdgeTensorShape
    public let dataType: EdgeDataType
    public let buffer: MTLBuffer
    private let runtime: EdgeMetalRuntime?

    public var byteCount: Int {
        shape.elementCount * dataType.byteSize
    }

    public init(
        shape: EdgeTensorShape,
        dataType: EdgeDataType,
        buffer: MTLBuffer,
        runtime: EdgeMetalRuntime? = nil
    ) {
        self.shape = shape
        self.dataType = dataType
        self.buffer = buffer
        self.runtime = runtime
    }

    public convenience init(
        float32 values: [Float],
        shape: EdgeTensorShape,
        runtime: EdgeMetalRuntime
    ) throws {
        guard values.count == shape.elementCount else {
            throw EdgeTensorError.elementCountMismatch(
                expected: shape.elementCount,
                actual: values.count
            )
        }

        let byteCount = values.count * MemoryLayout<Float>.stride
        guard let buffer = values.withUnsafeBytes({ bytes in
            runtime.device.makeBuffer(
                bytes: bytes.baseAddress!,
                length: byteCount,
                options: [.storageModeShared]
            )
        }) else {
            throw EdgeTensorError.bufferAllocationFailed(byteCount: byteCount)
        }

        self.init(shape: shape, dataType: .float32, buffer: buffer, runtime: runtime)
    }

    public convenience init(
        shape: EdgeTensorShape,
        dataType: EdgeDataType,
        runtime: EdgeMetalRuntime,
        storageMode: MTLResourceOptions = [.storageModeShared]
    ) throws {
        let byteCount = shape.elementCount * dataType.byteSize
        guard let buffer = runtime.device.makeBuffer(length: byteCount, options: storageMode) else {
            throw EdgeTensorError.bufferAllocationFailed(byteCount: byteCount)
        }
        self.init(shape: shape, dataType: dataType, buffer: buffer, runtime: runtime)
    }

    public func waitForPendingWork() {
        runtime?.waitForPendingWork()
    }

    public func readFloat32() throws -> [Float] {
        guard dataType == .float32 else {
            throw EdgeTensorError.unsupportedDataType
        }
        waitForPendingWork()
        let count = shape.elementCount
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
}
