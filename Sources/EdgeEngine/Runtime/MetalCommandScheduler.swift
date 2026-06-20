// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/// Pure command-budget accounting used by `EdgeMetalRuntime`.
///
/// This type has no Metal dependency so the scheduling invariants can be tested
/// deterministically before real command encoders exist.
public struct MetalCommandScheduler: Equatable, Sendable {
    public private(set) var configuration: MetalRuntimeConfiguration
    public private(set) var pendingOps: Int
    public private(set) var pendingBytes: Int
    public private(set) var commitCount: Int

    public init(configuration: MetalRuntimeConfiguration) {
        self.configuration = configuration
        self.pendingOps = 0
        self.pendingBytes = 0
        self.commitCount = 0
    }

    public var effectiveMaxOps: Int {
        configuration.effectiveMaxOpsPerCommandBuffer
    }

    public mutating func updateConfiguration(_ configuration: MetalRuntimeConfiguration) {
        self.configuration = configuration
    }

    /// Records a logical GPU operation and commits the current buffer if adding
    /// the operation would exceed either configured budget.
    ///
    /// The operation is always retained as pending work after any pre-commit.
    @discardableResult
    public mutating func recordOperation(byteCost: Int = 0) -> Bool {
        let byteCost = max(0, byteCost)
        let wouldExceedOps = pendingOps > 0 && pendingOps + 1 > effectiveMaxOps
        let wouldExceedBytes = pendingBytes > 0
            && pendingBytes + byteCost > configuration.maxBytesPerCommandBuffer

        let committedBeforeRecording = wouldExceedOps || wouldExceedBytes
        if committedBeforeRecording {
            commit()
        }

        pendingOps += 1
        pendingBytes += byteCost
        return committedBeforeRecording
    }

    @discardableResult
    public mutating func commit() -> Bool {
        guard pendingOps > 0 || pendingBytes > 0 else { return false }
        commitCount += 1
        pendingOps = 0
        pendingBytes = 0
        return true
    }
}
