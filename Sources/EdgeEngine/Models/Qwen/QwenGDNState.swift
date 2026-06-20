// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public enum QwenGDNStateError: Error, Equatable {
    case invalidLayerIndex(Int)
    case invalidHeadCount(Int)
    case invalidHeadDimension(Int)
    case invalidKeyHeadDimension(Int)
    case nonGDNLayersUnavailable
}

public struct QwenGDNStateShape: Equatable, Sendable {
    public var layerIndex: Int
    public var valueHeadCount: Int
    public var valueHeadDimension: Int
    public var keyHeadDimension: Int

    public init(layerIndex: Int, headCount: Int, headDimension: Int) throws {
        try self.init(
            layerIndex: layerIndex,
            valueHeadCount: headCount,
            valueHeadDimension: headDimension,
            keyHeadDimension: headDimension
        )
    }

    public init(
        layerIndex: Int,
        valueHeadCount: Int,
        valueHeadDimension: Int,
        keyHeadDimension: Int
    ) throws {
        guard layerIndex >= 0 else {
            throw QwenGDNStateError.invalidLayerIndex(layerIndex)
        }
        guard valueHeadCount > 0 else {
            throw QwenGDNStateError.invalidHeadCount(valueHeadCount)
        }
        guard valueHeadDimension > 0 else {
            throw QwenGDNStateError.invalidHeadDimension(valueHeadDimension)
        }
        guard keyHeadDimension > 0 else {
            throw QwenGDNStateError.invalidKeyHeadDimension(keyHeadDimension)
        }
        self.layerIndex = layerIndex
        self.valueHeadCount = valueHeadCount
        self.valueHeadDimension = valueHeadDimension
        self.keyHeadDimension = keyHeadDimension
    }

    public var headCount: Int {
        valueHeadCount
    }

    public var headDimension: Int {
        valueHeadDimension
    }

    public var tensorShape: EdgeTensorShape {
        EdgeTensorShape([valueHeadCount, valueHeadDimension, keyHeadDimension])
    }

    public var elementCount: Int {
        valueHeadCount * valueHeadDimension * keyHeadDimension
    }

    public static func shapes(for architecture: QwenHybridArchitecture) throws -> [QwenGDNStateShape] {
        try architecture.validate()
        guard !architecture.gdnLayerIndices.isEmpty else {
            throw QwenGDNStateError.nonGDNLayersUnavailable
        }
        return try architecture.gdnLayerIndices.map { layerIndex in
            try QwenGDNStateShape(
                layerIndex: layerIndex,
                valueHeadCount: architecture.linearValueHeadCount,
                valueHeadDimension: architecture.linearValueHeadDimension,
                keyHeadDimension: architecture.linearKeyHeadDimension
            )
        }
    }
}

public final class QwenGDNState {
    public let shape: QwenGDNStateShape
    public let tensor: EdgeTensor
    public private(set) var tokenPosition: Int

    public init(shape: QwenGDNStateShape, runtime: EdgeMetalRuntime) throws {
        self.shape = shape
        self.tensor = try EdgeTensor(
            float32: Array(repeating: 0, count: shape.elementCount),
            shape: shape.tensorShape,
            runtime: runtime
        )
        self.tokenPosition = 0
    }

    public func advanceToken() {
        tokenPosition += 1
    }

    public func reset() {
        tensor.waitForPendingWork()
        tensor.buffer.contents().initializeMemory(
            as: UInt8.self,
            repeating: 0,
            count: tensor.byteCount
        )
        tokenPosition = 0
    }
}
