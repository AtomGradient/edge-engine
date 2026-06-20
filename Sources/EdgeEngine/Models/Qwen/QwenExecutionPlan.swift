// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public enum QwenExecutionPlanError: Error, Equatable {
    case layerNotFound(Int)
}

public enum QwenLayerExecutionKind: Equatable, Sendable {
    case fullAttention
    case gdn
}

public struct QwenLayerExecutionStep: Equatable, Sendable {
    public var layerIndex: Int
    public var kind: QwenLayerExecutionKind
    public var requiresKVCache: Bool
    public var requiresGDNState: Bool

    public init(layerIndex: Int, kind: QwenLayerExecutionKind) {
        self.layerIndex = layerIndex
        self.kind = kind
        switch kind {
        case .fullAttention:
            self.requiresKVCache = true
            self.requiresGDNState = false
        case .gdn:
            self.requiresKVCache = false
            self.requiresGDNState = true
        }
    }
}

public struct QwenExecutionPlan: Equatable, Sendable {
    public var architecture: QwenHybridArchitecture
    public var steps: [QwenLayerExecutionStep]
    public var gdnStateShapes: [QwenGDNStateShape]
    public var gdnCacheShapes: [QwenGDNCacheShape]

    public init(architecture: QwenHybridArchitecture) throws {
        try architecture.validate()
        self.architecture = architecture
        self.steps = architecture.layerPlan.map { layer in
            QwenLayerExecutionStep(
                layerIndex: layer.index,
                kind: layer.kind.executionKind
            )
        }
        self.gdnStateShapes = try QwenGDNStateShape.shapes(for: architecture)
        self.gdnCacheShapes = try QwenGDNCacheShape.shapes(for: architecture)
    }

    public var fullAttentionSteps: [QwenLayerExecutionStep] {
        steps.filter { $0.kind == .fullAttention }
    }

    public var gdnSteps: [QwenLayerExecutionStep] {
        steps.filter { $0.kind == .gdn }
    }

    public func step(layerIndex: Int) throws -> QwenLayerExecutionStep {
        guard let step = steps.first(where: { $0.layerIndex == layerIndex }) else {
            throw QwenExecutionPlanError.layerNotFound(layerIndex)
        }
        return step
    }
}

private extension QwenHybridLayerKind {
    var executionKind: QwenLayerExecutionKind {
        switch self {
        case .fullAttention:
            return .fullAttention
        case .gdn:
            return .gdn
        }
    }
}
