// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public struct MetalKernelExecutionStats: Equatable, Sendable {
    public var operationName: String
    public var scheduledByteCost: Int
    public var precommittedBeforeEncoding: Bool
    public var effectiveMaxOpsPerCommandBuffer: Int
    public var pendingOpsAfterRecord: Int
    public var pendingBytesAfterRecord: Int
    public var logicalCommitCount: Int

    public init(
        operationName: String,
        scheduledByteCost: Int,
        precommittedBeforeEncoding: Bool,
        effectiveMaxOpsPerCommandBuffer: Int,
        pendingOpsAfterRecord: Int,
        pendingBytesAfterRecord: Int,
        logicalCommitCount: Int
    ) {
        self.operationName = operationName
        self.scheduledByteCost = scheduledByteCost
        self.precommittedBeforeEncoding = precommittedBeforeEncoding
        self.effectiveMaxOpsPerCommandBuffer = effectiveMaxOpsPerCommandBuffer
        self.pendingOpsAfterRecord = pendingOpsAfterRecord
        self.pendingBytesAfterRecord = pendingBytesAfterRecord
        self.logicalCommitCount = logicalCommitCount
    }
}

public struct MetalQuantizedBufferCacheStats: Equatable, Sendable {
    public var entryCount: Int
    public var hitCount: Int
    public var missCount: Int
    public var uploadedByteCount: Int
    public var cachedByteCount: Int
    public var releasedHostStorageByteCount: Int
    public var releasedHostStorageCount: Int

    public init(
        entryCount: Int = 0,
        hitCount: Int = 0,
        missCount: Int = 0,
        uploadedByteCount: Int = 0,
        cachedByteCount: Int = 0,
        releasedHostStorageByteCount: Int = 0,
        releasedHostStorageCount: Int = 0
    ) {
        self.entryCount = entryCount
        self.hitCount = hitCount
        self.missCount = missCount
        self.uploadedByteCount = uploadedByteCount
        self.cachedByteCount = cachedByteCount
        self.releasedHostStorageByteCount = releasedHostStorageByteCount
        self.releasedHostStorageCount = releasedHostStorageCount
    }
}

public enum MetalQuantizedBufferCacheScope: String, Codable, Equatable, Sendable {
    case decoder
    case vision
    case all
}
