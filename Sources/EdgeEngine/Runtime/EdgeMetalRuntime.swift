// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Metal

public enum EdgeMetalRuntimeError: Error, Equatable {
    case noSystemMetalDevice
    case cannotCreateCommandQueue
}

/// Minimal native Metal runtime entry point.
///
/// Kernel recording is added in later phases; this type owns the process-wide
/// device, queue, and scheduling configuration used by those encoders.
public final class EdgeMetalRuntime {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public private(set) var configuration: MetalRuntimeConfiguration
    public private(set) var lastCommandBufferAcquisitionStats: MetalCommandBufferAcquisitionStats?
    private var activeCommandBuffer: MTLCommandBuffer?
    private var activeOperationCount: Int
    private var activeByteCount: Int
    private var activeCommandBufferRetainedResources: [AnyObject]
    private var submittedCommandBuffers: [MTLCommandBuffer]
    private var submittedCommandBufferRetainedResources: [ObjectIdentifier: [AnyObject]]
    private var unboundedCommandBufferBatchDepth: Int
    private let commandBufferLock = NSLock()

    public init(
        device: MTLDevice? = MTLCreateSystemDefaultDevice(),
        configuration: MetalRuntimeConfiguration? = nil
    ) throws {
        guard let device else {
            throw EdgeMetalRuntimeError.noSystemMetalDevice
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw EdgeMetalRuntimeError.cannotCreateCommandQueue
        }
        self.device = device
        self.commandQueue = commandQueue
        self.configuration = configuration?.applyingEnvironmentOverrides()
            ?? MetalRuntimeConfiguration.defaultConfiguration(for: device)
        self.activeOperationCount = 0
        self.activeByteCount = 0
        self.activeCommandBufferRetainedResources = []
        self.submittedCommandBuffers = []
        self.submittedCommandBufferRetainedResources = [:]
        self.unboundedCommandBufferBatchDepth = 0
        logEffectiveConfiguration()
    }

    public func makeScheduler() -> MetalCommandScheduler {
        MetalCommandScheduler(configuration: configuration)
    }

    public func updatingConfiguration(
        _ update: (inout MetalRuntimeConfiguration) -> Void
    ) {
        waitForPendingWork()
        update(&configuration)
        configuration = configuration.applyingEnvironmentOverrides()
        logEffectiveConfiguration()
    }

    public func commandBufferForOperation(byteCost: Int = 0) -> MTLCommandBuffer? {
        guard configuration.commandBufferBatchingEnabled else {
            guard let commandBuffer = makeCommandBuffer() else {
                return nil
            }
            lastCommandBufferAcquisitionStats = MetalCommandBufferAcquisitionStats(
                byteCost: max(0, byteCost),
                committedActiveBuffer: true,
                waitedForBackpressure: false,
                backpressureWaitNanoseconds: 0,
                submittedBeforeWait: 0,
                submittedAfterWait: 0,
                maxInFlightCommandBuffers: configuration.maxInFlightCommandBuffers,
                activeOperationCountAfterRecord: 1,
                activeByteCountAfterRecord: max(0, byteCost)
            )
            return commandBuffer
        }

        var commandBufferToWait: MTLCommandBuffer?
        var committedActiveBuffer = false
        var submittedBeforeWait = 0
        let maxInFlightCommandBuffers: Int
        commandBufferLock.lock()

        let byteCost = max(0, byteCost)
        let isUnboundedBatch = unboundedCommandBufferBatchDepth > 0
        if !isUnboundedBatch {
            let wouldExceedOps = activeOperationCount > 0
                && activeOperationCount + 1 > configuration.effectiveMaxOpsPerCommandBuffer
            let wouldExceedBytes = activeByteCount > 0
                && activeByteCount + byteCost > configuration.maxBytesPerCommandBuffer
            if wouldExceedOps || wouldExceedBytes {
                commitActiveCommandBufferLocked()
                committedActiveBuffer = true
            }
        }
        submittedBeforeWait = submittedCommandBuffers.count
        maxInFlightCommandBuffers = configuration.maxInFlightCommandBuffers
        if !isUnboundedBatch {
            commandBufferToWait = dequeueBackpressureWaitBufferLocked()
        }
        commandBufferLock.unlock()

        var backpressureWaitNanoseconds: UInt64 = 0
        if let commandBufferToWait {
            let start = DispatchTime.now().uptimeNanoseconds
            commandBufferToWait.waitUntilCompleted()
            backpressureWaitNanoseconds = DispatchTime.now().uptimeNanoseconds - start
            releaseRetainedResources(for: commandBufferToWait)
        }

        commandBufferLock.lock()
        defer { commandBufferLock.unlock() }

        if activeCommandBuffer == nil {
            activeCommandBuffer = makeCommandBuffer()
        }
        guard let commandBuffer = activeCommandBuffer else {
            return nil
        }
        activeOperationCount += 1
        activeByteCount += byteCost
        lastCommandBufferAcquisitionStats = MetalCommandBufferAcquisitionStats(
            byteCost: byteCost,
            committedActiveBuffer: committedActiveBuffer,
            waitedForBackpressure: commandBufferToWait != nil,
            backpressureWaitNanoseconds: backpressureWaitNanoseconds,
            submittedBeforeWait: submittedBeforeWait,
            submittedAfterWait: submittedCommandBuffers.count,
            maxInFlightCommandBuffers: maxInFlightCommandBuffers,
            activeOperationCountAfterRecord: activeOperationCount,
            activeByteCountAfterRecord: activeByteCount
        )
        return commandBuffer
    }

    public func finishOperationCommandBuffer(_ commandBuffer: MTLCommandBuffer) {
        guard !configuration.commandBufferBatchingEnabled else {
            return
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        releaseRetainedResources(for: commandBuffer)
    }

    public func retainResources(_ resources: [AnyObject], for commandBuffer: MTLCommandBuffer) {
        guard !resources.isEmpty else { return }
        commandBufferLock.lock()
        if configuration.commandBufferBatchingEnabled,
           let activeCommandBuffer,
           activeCommandBuffer === commandBuffer {
            activeCommandBufferRetainedResources.append(contentsOf: resources)
        } else {
            let key = ObjectIdentifier(commandBuffer)
            submittedCommandBufferRetainedResources[key, default: []].append(contentsOf: resources)
        }
        commandBufferLock.unlock()
    }

    public func commitPendingWork(waitUntilCompleted: Bool = false) {
        let commandBuffers: [MTLCommandBuffer]
        let shouldWait = waitUntilCompleted || configuration.syncEval
        commandBufferLock.lock()
        commitActiveCommandBufferLocked()
        if shouldWait {
            commandBuffers = submittedCommandBuffers
            submittedCommandBuffers.removeAll(keepingCapacity: true)
        } else {
            commandBuffers = []
        }
        commandBufferLock.unlock()

        if shouldWait {
            for commandBuffer in commandBuffers {
                commandBuffer.waitUntilCompleted()
                releaseRetainedResources(for: commandBuffer)
            }
        }
    }

    public func waitForPendingWork() {
        commitPendingWork(waitUntilCompleted: true)
    }

    public func withUnboundedCommandBufferBatch<T>(_ body: () throws -> T) rethrows -> T {
        guard configuration.commandBufferBatchingEnabled else {
            return try body()
        }

        commandBufferLock.lock()
        let isNestedBatch = unboundedCommandBufferBatchDepth > 0
        if isNestedBatch {
            unboundedCommandBufferBatchDepth += 1
            commandBufferLock.unlock()
            do {
                let result = try body()
                commandBufferLock.lock()
                unboundedCommandBufferBatchDepth -= 1
                commandBufferLock.unlock()
                return result
            } catch {
                commandBufferLock.lock()
                unboundedCommandBufferBatchDepth -= 1
                commandBufferLock.unlock()
                throw error
            }
        }
        commandBufferLock.unlock()

        waitForPendingWork()
        commandBufferLock.lock()
        unboundedCommandBufferBatchDepth += 1
        commandBufferLock.unlock()

        do {
            let result = try body()
            commandBufferLock.lock()
            unboundedCommandBufferBatchDepth -= 1
            commandBufferLock.unlock()
            commitPendingWork(waitUntilCompleted: true)
            return result
        } catch {
            commandBufferLock.lock()
            unboundedCommandBufferBatchDepth -= 1
            commandBufferLock.unlock()
            commitPendingWork(waitUntilCompleted: true)
            throw error
        }
    }

    private func commitActiveCommandBufferLocked() {
        guard let commandBuffer = activeCommandBuffer else {
            return
        }
        commandBuffer.addCompletedHandler { [weak self] completedBuffer in
            self?.removeSubmittedCommandBuffer(completedBuffer)
        }
        commandBuffer.commit()
        submittedCommandBuffers.append(commandBuffer)
        submittedCommandBufferRetainedResources[ObjectIdentifier(commandBuffer), default: []]
            .append(contentsOf: activeCommandBufferRetainedResources)
        activeCommandBufferRetainedResources.removeAll(keepingCapacity: true)
        activeCommandBuffer = nil
        activeOperationCount = 0
        activeByteCount = 0
    }

    private func dequeueBackpressureWaitBufferLocked() -> MTLCommandBuffer? {
        if configuration.syncEval {
            return submittedCommandBuffers.isEmpty ? nil : submittedCommandBuffers.removeFirst()
        }
        guard submittedCommandBuffers.count >= configuration.maxInFlightCommandBuffers else {
            return nil
        }
        return submittedCommandBuffers.removeFirst()
    }

    private func removeSubmittedCommandBuffer(_ commandBuffer: MTLCommandBuffer) {
        commandBufferLock.lock()
        submittedCommandBuffers.removeAll { $0 === commandBuffer }
        submittedCommandBufferRetainedResources.removeValue(forKey: ObjectIdentifier(commandBuffer))
        commandBufferLock.unlock()
    }

    private func releaseRetainedResources(for commandBuffer: MTLCommandBuffer) {
        commandBufferLock.lock()
        submittedCommandBufferRetainedResources.removeValue(forKey: ObjectIdentifier(commandBuffer))
        commandBufferLock.unlock()
    }

    private func makeCommandBuffer() -> MTLCommandBuffer? {
        commandQueue.makeCommandBufferWithUnretainedReferences()
    }

    private func logEffectiveConfiguration() {
        let archName: String
        if #available(macOS 13.0, iOS 16.0, *) {
            archName = device.architecture.name
        } else {
            archName = "unknown"
        }
        let family = archName.last.map(String.init) ?? "unknown"
        let memoryLimitMB = configuration.memoryLimitBytes.map { $0 / 1_048_576 } ?? -1
        let cacheLimitMB = configuration.quantizedBufferCacheLimitBytes.map { $0 / 1_048_576 } ?? -1
        debugPrint(
            "[EdgeEngine] Metal runtime: arch=\(archName) family=\(family) "
                + "batching=\(configuration.commandBufferBatchingEnabled ? "on" : "off") "
                + "maxInflight=\(configuration.maxInFlightCommandBuffers) "
                + "mlxQMM=\(configuration.useMLXQuantizedMatmul ? "on" : "off") "
                + "mlxPrefillQMM=\(configuration.useMLXQuantizedPrefillMatmul ? "on" : "off") "
                + "vendoredCBPrefillQMM=\(configuration.useVendoredCommandBufferPrefillQMM ? "on" : "off") "
                + "singleCBPrefill=\(configuration.useSingleCommandBufferPrefill ? "on" : "off") "
                + "singleCBDecode=\(configuration.useSingleCommandBufferDecode ? "on" : "off") "
                + "prefillLayerCB=\(configuration.usePrefillLayerCommandBufferBatching ? "on" : "off") "
                + "fusedGDNDecode=\(configuration.useFusedGDNDecode ? "on" : "off") "
                + "cmlxFastRMSNorm=\(configuration.useCmlxFastRMSNorm ? "on" : "off") "
                + "cmlxLazyOutputHead=\(configuration.useCmlxLazyOutputHead ? "on" : "off") "
                + "greedyOutputHeadArgmax=\(configuration.useGreedyOutputHeadArgmax ? "on" : "off") "
                + "syncEval=\(configuration.syncEval ? "on" : "off") "
                + "qNoCopy=\(configuration.quantizedNoCopyBuffersEnabled ? "on" : "off") "
                + "maxOps=\(configuration.effectiveMaxOpsPerCommandBuffer) "
                + "maxMB=\(configuration.maxMBPerCommandBuffer) "
                + "memoryLimitMB=\(memoryLimitMB) "
                + "cacheLimitMB=\(cacheLimitMB)"
        )
    }
}

extension EdgeMetalRuntime: @unchecked Sendable {}
