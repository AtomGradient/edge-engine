// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum QwenVisionPatchEmbedError: Error, Equatable {
    case missingTensor(String)
    case unsupportedDType(tensorName: String, dtype: String)
    case invalidWeightShape([Int])
    case invalidBiasShape([Int])
    case invalidInputShape([Int])
}

public struct QwenVisionPatchEmbedOutput: Equatable, Sendable {
    public var values: [Float]
    public var shape: [Int]

    public init(values: [Float], shape: [Int]) {
        self.values = values
        self.shape = shape
    }
}

public struct QwenVisionPatchEmbedWeights: Equatable, Sendable {
    public var outputChannels: Int
    public var temporalPatchSize: Int
    public var patchSize: Int
    public var inputChannels: Int
    public var projectionMatrix: [Float]
    public var bias: [Float]

    public init(
        outputChannels: Int,
        temporalPatchSize: Int,
        patchSize: Int,
        inputChannels: Int,
        projectionMatrix: [Float],
        bias: [Float]
    ) {
        self.outputChannels = outputChannels
        self.temporalPatchSize = temporalPatchSize
        self.patchSize = patchSize
        self.inputChannels = inputChannels
        self.projectionMatrix = projectionMatrix
        self.bias = bias
    }

    public static func load(from store: QwenVisionWeightStore) throws -> QwenVisionPatchEmbedWeights {
        let prefix = store.index.visionManifest.prefix
        let weightName = "\(prefix).patch_embed.proj.weight"
        let biasName = "\(prefix).patch_embed.proj.bias"
        guard store.tensorNames.contains(weightName) else {
            throw QwenVisionPatchEmbedError.missingTensor(weightName)
        }
        guard store.tensorNames.contains(biasName) else {
            throw QwenVisionPatchEmbedError.missingTensor(biasName)
        }

        let weightMetadata = try store.metadata(named: weightName)
        let biasMetadata = try store.metadata(named: biasName)
        guard weightMetadata.shape.count == 5 else {
            throw QwenVisionPatchEmbedError.invalidWeightShape(weightMetadata.shape)
        }
        guard biasMetadata.shape == [weightMetadata.shape[0]] else {
            throw QwenVisionPatchEmbedError.invalidBiasShape(biasMetadata.shape)
        }

        let outputChannels = weightMetadata.shape[0]
        let temporalPatchSize = weightMetadata.shape[1]
        let patchSize = weightMetadata.shape[2]
        let inputChannels = weightMetadata.shape[4]
        let weights = try floatValues(
            tensorName: weightName,
            metadata: weightMetadata,
            data: store.tensorData(named: weightName)
        )
        let bias = try floatValues(
            tensorName: biasName,
            metadata: biasMetadata,
            data: store.tensorData(named: biasName)
        )

        let inner = temporalPatchSize * patchSize * patchSize * inputChannels
        var projectionMatrix = Array(repeating: Float.zero, count: inner * outputChannels)
        for output in 0..<outputChannels {
            for temporal in 0..<temporalPatchSize {
                for y in 0..<patchSize {
                    for x in 0..<patchSize {
                        for channel in 0..<inputChannels {
                            let sourceIndex = (
                                (((output * temporalPatchSize + temporal) * patchSize + y)
                                    * patchSize + x) * inputChannels + channel
                            )
                            let innerIndex = (
                                ((temporal * patchSize + y) * patchSize + x)
                                    * inputChannels + channel
                            )
                            projectionMatrix[innerIndex * outputChannels + output] = weights[sourceIndex]
                        }
                    }
                }
            }
        }

        return QwenVisionPatchEmbedWeights(
            outputChannels: outputChannels,
            temporalPatchSize: temporalPatchSize,
            patchSize: patchSize,
            inputChannels: inputChannels,
            projectionMatrix: projectionMatrix,
            bias: bias
        )
    }

    public func project(
        pixelValues: [Float],
        shape: [Int]
    ) throws -> QwenVisionPatchEmbedOutput {
        guard shape.count == 2,
              shape[0] > 0,
              shape[1] == inputChannels * temporalPatchSize * patchSize * patchSize,
              pixelValues.count == shape[0] * shape[1]
        else {
            throw QwenVisionPatchEmbedError.invalidInputShape(shape)
        }

        let rows = shape[0]
        let inner = shape[1]
        guard projectionMatrix.count == inner * outputChannels else {
            throw QwenVisionPatchEmbedError.invalidWeightShape([
                inner,
                outputChannels,
                projectionMatrix.count,
            ])
        }
        guard bias.count == outputChannels else {
            throw QwenVisionPatchEmbedError.invalidBiasShape([bias.count])
        }

        let channelFirstRows = try channelLastRows(
            pixelValues,
            rows: rows,
            inner: inner
        )
        var projected = try EdgeMLXBridge.matmulFloat32GPU(
            lhs: channelFirstRows,
            rows: rows,
            inner: inner,
            rhs: projectionMatrix,
            columns: outputChannels
        )
        for row in 0..<rows {
            let offset = row * outputChannels
            for column in 0..<outputChannels {
                projected[offset + column] += bias[column]
            }
        }
        return QwenVisionPatchEmbedOutput(
            values: projected,
            shape: [rows, outputChannels]
        )
    }

    private func channelLastRows(
        _ values: [Float],
        rows: Int,
        inner: Int
    ) throws -> [Float] {
        guard values.count == rows * inner else {
            throw QwenVisionPatchEmbedError.invalidInputShape([rows, inner])
        }
        var output = Array(repeating: Float.zero, count: values.count)
        for row in 0..<rows {
            let rowOffset = row * inner
            for channel in 0..<inputChannels {
                for temporal in 0..<temporalPatchSize {
                    for y in 0..<patchSize {
                        for x in 0..<patchSize {
                            let sourceInner = (
                                ((channel * temporalPatchSize + temporal) * patchSize + y)
                                    * patchSize + x
                            )
                            let destinationInner = (
                                ((temporal * patchSize + y) * patchSize + x)
                                    * inputChannels + channel
                            )
                            output[rowOffset + destinationInner] = values[rowOffset + sourceInner]
                        }
                    }
                }
            }
        }
        return output
    }

    private static func floatValues(
        tensorName: String,
        metadata: SafeTensorMetadata,
        data: Data
    ) throws -> [Float] {
        switch metadata.dtype {
        case "F32":
            let expectedByteCount = metadata.elementCount * MemoryLayout<Float>.stride
            guard data.count == expectedByteCount else {
                throw SafeTensorsError.tensorByteCountMismatch(
                    tensorName: tensorName,
                    expected: expectedByteCount,
                    actual: data.count
                )
            }
            return data.withUnsafeBytes { bytes in
                (0..<metadata.elementCount).map { index in
                    let bitPattern = bytes.loadUnaligned(
                        fromByteOffset: index * MemoryLayout<Float>.stride,
                        as: UInt32.self
                    )
                    return Float(bitPattern: UInt32(littleEndian: bitPattern))
                }
            }
        case "F16":
            let expectedByteCount = metadata.elementCount * MemoryLayout<UInt16>.stride
            guard data.count == expectedByteCount else {
                throw SafeTensorsError.tensorByteCountMismatch(
                    tensorName: tensorName,
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
                    tensorName: tensorName,
                    expected: expectedByteCount,
                    actual: data.count
                )
            }
            return data.withUnsafeBytes { bytes in
                (0..<metadata.elementCount).map { index in
                    let bf16 = bytes.loadUnaligned(
                        fromByteOffset: index * MemoryLayout<UInt16>.stride,
                        as: UInt16.self
                    )
                    return Float(
                        bitPattern: UInt32(UInt16(littleEndian: bf16)) << 16
                    )
                }
            }
        default:
            throw QwenVisionPatchEmbedError.unsupportedDType(
                tensorName: tensorName,
                dtype: metadata.dtype
            )
        }
    }
}
