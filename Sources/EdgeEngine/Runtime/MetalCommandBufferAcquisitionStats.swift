// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public struct MetalCommandBufferAcquisitionStats: Equatable, Sendable {
    public var byteCost: Int
    public var committedActiveBuffer: Bool
    public var waitedForBackpressure: Bool
    public var backpressureWaitNanoseconds: UInt64
    public var submittedBeforeWait: Int
    public var submittedAfterWait: Int
    public var maxInFlightCommandBuffers: Int
    public var activeOperationCountAfterRecord: Int
    public var activeByteCountAfterRecord: Int

    public init(
        byteCost: Int,
        committedActiveBuffer: Bool,
        waitedForBackpressure: Bool,
        backpressureWaitNanoseconds: UInt64,
        submittedBeforeWait: Int,
        submittedAfterWait: Int,
        maxInFlightCommandBuffers: Int,
        activeOperationCountAfterRecord: Int,
        activeByteCountAfterRecord: Int
    ) {
        self.byteCost = byteCost
        self.committedActiveBuffer = committedActiveBuffer
        self.waitedForBackpressure = waitedForBackpressure
        self.backpressureWaitNanoseconds = backpressureWaitNanoseconds
        self.submittedBeforeWait = submittedBeforeWait
        self.submittedAfterWait = submittedAfterWait
        self.maxInFlightCommandBuffers = maxInFlightCommandBuffers
        self.activeOperationCountAfterRecord = activeOperationCountAfterRecord
        self.activeByteCountAfterRecord = activeByteCountAfterRecord
    }
}
