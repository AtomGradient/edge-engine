// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Testing
@testable import EdgeEngine

@Test func dynamicScheduleKeepsConfiguredBudgetForShortContext() {
    let schedule = DynamicOpsSchedule(floor: 5, contextLow: 4_096, contextHigh: 12_288)

    #expect(schedule.effectiveMaxOps(configuredMaxOps: 15, contextLengthHint: 2_048) == 15)
    #expect(schedule.effectiveMaxOps(configuredMaxOps: 15, contextLengthHint: 4_096) == 15)
}

@Test func dynamicScheduleTapersToFloorForLongContext() {
    let schedule = DynamicOpsSchedule(floor: 5, contextLow: 4_096, contextHigh: 12_288)

    #expect(schedule.effectiveMaxOps(configuredMaxOps: 15, contextLengthHint: 8_192) == 10)
    #expect(schedule.effectiveMaxOps(configuredMaxOps: 15, contextLengthHint: 12_288) == 5)
    #expect(schedule.effectiveMaxOps(configuredMaxOps: 15, contextLengthHint: 24_576) == 5)
}

@Test func schedulerCommitsWhenOperationBudgetWouldOverflow() {
    var scheduler = MetalCommandScheduler(
        configuration: .init(maxOpsPerCommandBuffer: 3, maxMBPerCommandBuffer: 40)
    )

    #expect(scheduler.recordOperation() == false)
    #expect(scheduler.recordOperation() == false)
    #expect(scheduler.recordOperation() == false)
    #expect(scheduler.commitCount == 0)

    #expect(scheduler.recordOperation() == true)
    #expect(scheduler.commitCount == 1)
    #expect(scheduler.pendingOps == 1)
}

@Test func schedulerCommitsWhenByteBudgetWouldOverflow() {
    var scheduler = MetalCommandScheduler(
        configuration: .init(maxOpsPerCommandBuffer: 100, maxMBPerCommandBuffer: 1)
    )

    #expect(scheduler.recordOperation(byteCost: 700_000) == false)
    #expect(scheduler.recordOperation(byteCost: 700_000) == true)
    #expect(scheduler.commitCount == 1)
    #expect(scheduler.pendingOps == 1)
    #expect(scheduler.pendingBytes == 700_000)
}

@Test func schedulerUsesDynamicEffectiveBudget() {
    let config = MetalRuntimeConfiguration(
        maxOpsPerCommandBuffer: 15,
        maxMBPerCommandBuffer: 40,
        contextLengthHint: 12_288,
        dynamicOpsSchedule: .init(floor: 5, contextLow: 4_096, contextHigh: 12_288)
    )
    var scheduler = MetalCommandScheduler(configuration: config)

    for _ in 0..<5 {
        #expect(scheduler.recordOperation() == false)
    }
    #expect(scheduler.recordOperation() == true)
    #expect(scheduler.commitCount == 1)
    #expect(scheduler.pendingOps == 1)
}
