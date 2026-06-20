// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Metal

/// Process-wide native Metal configuration state.
///
/// EdgeKit applies device/context-specific scheduling decisions here before
/// constructing native Qwen/ASR/TTS executors. Existing executors keep the
/// configuration they were created with; new runtimes pick up the latest value.
public final class EdgeEngineMetalConfigurationStore: @unchecked Sendable {
    public static let shared = EdgeEngineMetalConfigurationStore()

    private let lock = NSLock()
    private var configuration: MetalRuntimeConfiguration

    public init(configuration: MetalRuntimeConfiguration = .init()) {
        self.configuration = configuration
    }

    @discardableResult
    public func apply(_ configuration: MetalRuntimeConfiguration) -> MetalRuntimeConfiguration {
        lock.lock()
        defer { lock.unlock() }
        self.configuration = configuration
        return configuration
    }

    public var currentConfiguration: MetalRuntimeConfiguration {
        lock.lock()
        defer { lock.unlock() }
        return configuration
    }

    public func makeRuntime(device: MTLDevice? = MTLCreateSystemDefaultDevice()) throws -> EdgeMetalRuntime {
        try EdgeMetalRuntime(device: device, configuration: currentConfiguration)
    }
}
