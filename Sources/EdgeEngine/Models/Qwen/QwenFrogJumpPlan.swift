// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public struct QwenFrogJumpPlan: Equatable, Sendable {
    public var enabled: Bool
    public var skipLayers: [Int]

    public init(enabled: Bool, skipLayers: [Int]) {
        self.enabled = enabled && !skipLayers.isEmpty
        self.skipLayers = skipLayers.sorted()
    }

    public var layerMask: UInt64 {
        skipLayers.reduce(UInt64.zero) { mask, layer in
            guard layer >= 0, layer < 64 else { return mask }
            return mask | (UInt64(1) << UInt64(layer))
        }
    }

    public static func disabled() -> QwenFrogJumpPlan {
        QwenFrogJumpPlan(enabled: false, skipLayers: [])
    }

    public static func compute(
        architecture: QwenHybridArchitecture,
        requestedEnabled: Bool,
        thinkingEnabled: Bool
    ) -> QwenFrogJumpPlan {
        guard requestedEnabled, !thinkingEnabled else {
            return disabled()
        }
        guard architecture.family == .qwen35 || architecture.family == .qwen36 else {
            return disabled()
        }
        return compute(layerKinds: architecture.layerPlan.map(\.kind))
    }

    public static func compute(layerKinds: [QwenHybridLayerKind]) -> QwenFrogJumpPlan {
        let interval = 4
        guard !layerKinds.isEmpty,
              layerKinds.count <= 64,
              layerKinds.count.isMultiple(of: interval)
        else {
            return disabled()
        }

        let groupCount = layerKinds.count / interval
        guard groupCount > 2 else {
            return disabled()
        }

        for group in 0..<groupCount {
            let base = group * interval
            guard layerKinds[base] == .gdn,
                  layerKinds[base + 1] == .gdn,
                  layerKinds[base + 2] == .gdn,
                  layerKinds[base + 3] == .fullAttention
            else {
                return disabled()
            }
        }

        let startGroup = 2
        let endGroup = groupCount / 2 + 1
        guard startGroup < endGroup else {
            return disabled()
        }

        var skipLayers: [Int] = []
        for group in startGroup..<endGroup {
            let base = group * interval
            skipLayers.append(base)
            skipLayers.append(base + 1)
        }
        return QwenFrogJumpPlan(enabled: true, skipLayers: skipLayers)
    }
}
