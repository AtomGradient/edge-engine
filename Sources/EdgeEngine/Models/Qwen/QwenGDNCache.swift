// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public enum QwenGDNCacheError: Error, Equatable {
    case invalidLayerIndex(Int)
    case layerIsNotGDN(layerIndex: Int, kind: QwenHybridLayerKind)
    case invalidConvStateLength(Int)
    case invalidNextConvStateShape(expected: [Int], actual: [Int])
    case invalidNextRecurrentStateShape(expected: [Int], actual: [Int])
    case invalidTokenCount(Int)
    case nonGDNLayersUnavailable
}

public struct QwenGDNCacheShape: Equatable, Sendable {
    public var layerIndex: Int
    public var convStateTokenCount: Int
    public var convHiddenSize: Int
    public var recurrentStateShape: QwenGDNStateShape

    public init(
        layerIndex: Int,
        convStateTokenCount: Int,
        convHiddenSize: Int,
        recurrentStateShape: QwenGDNStateShape
    ) throws {
        guard layerIndex >= 0 else {
            throw QwenGDNCacheError.invalidLayerIndex(layerIndex)
        }
        guard convStateTokenCount > 0 else {
            throw QwenGDNCacheError.invalidConvStateLength(convStateTokenCount)
        }
        guard convHiddenSize > 0 else {
            throw QwenArchitectureError.invalidHiddenSize(convHiddenSize)
        }
        self.layerIndex = layerIndex
        self.convStateTokenCount = convStateTokenCount
        self.convHiddenSize = convHiddenSize
        self.recurrentStateShape = recurrentStateShape
    }

    public var convStateTensorShape: EdgeTensorShape {
        EdgeTensorShape([convStateTokenCount, convHiddenSize])
    }

    public var convStateElementCount: Int {
        convStateTokenCount * convHiddenSize
    }

    public var recurrentStateTensorShape: EdgeTensorShape {
        recurrentStateShape.tensorShape
    }

    public var recurrentStateElementCount: Int {
        recurrentStateShape.elementCount
    }

    public static func shape(
        for architecture: QwenHybridArchitecture,
        layerIndex: Int
    ) throws -> QwenGDNCacheShape {
        try architecture.validate()
        guard let layer = architecture.layerPlan.first(where: { $0.index == layerIndex }) else {
            throw QwenGDNCacheError.invalidLayerIndex(layerIndex)
        }
        guard layer.kind == .gdn else {
            throw QwenGDNCacheError.layerIsNotGDN(layerIndex: layerIndex, kind: layer.kind)
        }
        return try QwenGDNCacheShape(
            layerIndex: layerIndex,
            convStateTokenCount: architecture.linearConvKernelSize - 1,
            convHiddenSize: architecture.linearConvHiddenSize,
            recurrentStateShape: QwenGDNStateShape(
                layerIndex: layerIndex,
                valueHeadCount: architecture.linearValueHeadCount,
                valueHeadDimension: architecture.linearValueHeadDimension,
                keyHeadDimension: architecture.linearKeyHeadDimension
            )
        )
    }

    public static func shapes(for architecture: QwenHybridArchitecture) throws -> [QwenGDNCacheShape] {
        try architecture.validate()
        guard !architecture.gdnLayerIndices.isEmpty else {
            throw QwenGDNCacheError.nonGDNLayersUnavailable
        }
        return try architecture.gdnLayerIndices.map { layerIndex in
            try shape(for: architecture, layerIndex: layerIndex)
        }
    }
}

public final class QwenGDNCache {
    public let shape: QwenGDNCacheShape
    private var convStates: [EdgeTensor]
    private var recurrentStates: [EdgeTensor]
    private var activeStateIndex: Int
    public private(set) var tokenPosition: Int

    public var convState: EdgeTensor {
        convStates[activeStateIndex]
    }

    public var recurrentState: EdgeTensor {
        recurrentStates[activeStateIndex]
    }

    public init(shape: QwenGDNCacheShape, runtime: EdgeMetalRuntime) throws {
        self.shape = shape
        self.convStates = [
            try QwenGDNCache.makeZeroTensor(
                shape: shape.convStateTensorShape,
                elementCount: shape.convStateElementCount,
                runtime: runtime
            ),
            try QwenGDNCache.makeZeroTensor(
                shape: shape.convStateTensorShape,
                elementCount: shape.convStateElementCount,
                runtime: runtime
            ),
        ]
        self.recurrentStates = [
            try QwenGDNCache.makeZeroTensor(
                shape: shape.recurrentStateTensorShape,
                elementCount: shape.recurrentStateElementCount,
                runtime: runtime
            ),
            try QwenGDNCache.makeZeroTensor(
                shape: shape.recurrentStateTensorShape,
                elementCount: shape.recurrentStateElementCount,
                runtime: runtime
            ),
        ]
        self.activeStateIndex = 0
        self.tokenPosition = 0
    }

    public func advanceToken() {
        tokenPosition += 1
    }

    public func update(
        nextConvState: EdgeTensor,
        nextRecurrentState: EdgeTensor,
        tokenCount: Int = 1,
        executor: MetalKernelExecutor? = nil
    ) throws {
        guard tokenCount > 0 else {
            throw QwenGDNCacheError.invalidTokenCount(tokenCount)
        }
        guard nextConvState.shape.dimensions == shape.convStateTensorShape.dimensions else {
            throw QwenGDNCacheError.invalidNextConvStateShape(
                expected: shape.convStateTensorShape.dimensions,
                actual: nextConvState.shape.dimensions
            )
        }
        guard nextConvState.dataType == .float32 else {
            throw QwenGDNCacheError.invalidNextConvStateShape(
                expected: shape.convStateTensorShape.dimensions,
                actual: nextConvState.shape.dimensions
            )
        }
        guard nextRecurrentState.shape.dimensions == shape.recurrentStateTensorShape.dimensions else {
            throw QwenGDNCacheError.invalidNextRecurrentStateShape(
                expected: shape.recurrentStateTensorShape.dimensions,
                actual: nextRecurrentState.shape.dimensions
            )
        }
        guard nextRecurrentState.dataType == .float32 else {
            throw QwenGDNCacheError.invalidNextRecurrentStateShape(
                expected: shape.recurrentStateTensorShape.dimensions,
                actual: nextRecurrentState.shape.dimensions
            )
        }

        let nextStateIndex = 1 - activeStateIndex
        if executor != nil {
            convStates[nextStateIndex] = nextConvState
            recurrentStates[nextStateIndex] = nextRecurrentState
        } else {
            nextConvState.waitForPendingWork()
            nextRecurrentState.waitForPendingWork()
            convStates[nextStateIndex].waitForPendingWork()
            recurrentStates[nextStateIndex].waitForPendingWork()
            convStates[nextStateIndex].buffer.contents().copyMemory(
                from: nextConvState.buffer.contents(),
                byteCount: convStates[nextStateIndex].byteCount
            )
            recurrentStates[nextStateIndex].buffer.contents().copyMemory(
                from: nextRecurrentState.buffer.contents(),
                byteCount: recurrentStates[nextStateIndex].byteCount
            )
        }
        activeStateIndex = nextStateIndex
        tokenPosition += tokenCount
    }

    public func reset() {
        for state in convStates {
            state.waitForPendingWork()
            state.buffer.contents().initializeMemory(
                as: UInt8.self,
                repeating: 0,
                count: state.byteCount
            )
        }
        for state in recurrentStates {
            state.waitForPendingWork()
            state.buffer.contents().initializeMemory(
                as: UInt8.self,
                repeating: 0,
                count: state.byteCount
            )
        }
        activeStateIndex = 0
        tokenPosition = 0
    }

    private static func makeZeroTensor(
        shape: EdgeTensorShape,
        elementCount: Int,
        runtime: EdgeMetalRuntime
    ) throws -> EdgeTensor {
        try EdgeTensor(
            float32: Array(repeating: 0, count: elementCount),
            shape: shape,
            runtime: runtime
        )
    }
}
