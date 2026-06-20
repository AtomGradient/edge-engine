// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/// Context-aware command buffer scheduling policy.
///
/// The effective operation budget stays at the configured maximum for short
/// contexts, then linearly tapers to `floor` between `contextLow` and
/// `contextHigh`.
public struct DynamicOpsSchedule: Equatable, Sendable {
    public var floor: Int
    public var contextLow: Int
    public var contextHigh: Int

    public init(floor: Int, contextLow: Int, contextHigh: Int) {
        self.floor = max(1, floor)
        self.contextLow = max(0, contextLow)
        self.contextHigh = max(self.contextLow + 1, contextHigh)
    }

    public func effectiveMaxOps(configuredMaxOps: Int, contextLengthHint: Int) -> Int {
        let configuredMaxOps = max(1, configuredMaxOps)
        guard configuredMaxOps > floor else { return configuredMaxOps }
        guard contextLengthHint > contextLow else { return configuredMaxOps }
        guard contextLengthHint < contextHigh else { return floor }

        let span = Double(contextHigh - contextLow)
        let progress = Double(contextLengthHint - contextLow) / span
        let tapered = Double(configuredMaxOps) - progress * Double(configuredMaxOps - floor)
        return max(floor, Int(tapered.rounded(.down)))
    }
}
