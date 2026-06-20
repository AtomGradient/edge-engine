// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Testing
@testable import EdgeEngine

@Test func metalConfigurationStoreAppliesCurrentSchedulingConfig() throws {
    let store = EdgeEngineMetalConfigurationStore()
    let applied = store.apply(
        MetalRuntimeConfiguration(
            maxOpsPerCommandBuffer: 15,
            maxMBPerCommandBuffer: 7,
            contextLengthHint: 12_288,
            dynamicOpsSchedule: DynamicOpsSchedule(
                floor: 5,
                contextLow: 4_096,
                contextHigh: 12_288
            ),
            quantizedBufferCacheLimitBytes: 12_582_912
        )
    )

    #expect(applied.effectiveMaxOpsPerCommandBuffer == 5)
    #expect(store.currentConfiguration == applied)

    let scheduler = MetalCommandScheduler(configuration: store.currentConfiguration)
    #expect(scheduler.effectiveMaxOps == 5)
    #expect(scheduler.configuration.maxMBPerCommandBuffer == 7)
    #expect(scheduler.configuration.quantizedBufferCacheLimitBytes == 12_582_912)
}
