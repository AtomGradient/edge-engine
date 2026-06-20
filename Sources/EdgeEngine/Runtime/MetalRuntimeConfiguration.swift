// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Darwin
import Metal

/// Runtime limits for EdgeEngine's native Metal command scheduling.
public struct MetalRuntimeConfiguration: Equatable, Sendable {
    public var maxOpsPerCommandBuffer: Int
    public var maxMBPerCommandBuffer: Int
    public var contextLengthHint: Int
    public var dynamicOpsSchedule: DynamicOpsSchedule?
    public var commandBufferBatchingEnabled: Bool
    public var maxInFlightCommandBuffers: Int
    public var useMLXQuantizedMatmul: Bool
    public var useMLXQuantizedPrefillMatmul: Bool
    public var useVendoredCommandBufferPrefillQMM: Bool
    public var useSingleCommandBufferPrefill: Bool
    public var useSingleCommandBufferDecode: Bool
    public var usePrefillLayerCommandBufferBatching: Bool
    public var useFusedGDNDecode: Bool
    public var useCmlxFastRMSNorm: Bool
    public var useCmlxLazyOutputHead: Bool
    public var useGreedyOutputHeadArgmax: Bool
    public var syncEval: Bool

    /// Optional MLX allocator memory limit. This controls when MLX starts
    /// aggressively freeing graph buffers during eval.
    public var memoryLimitBytes: Int? {
        get {
            memoryLimitByteStorage < 0 ? nil : memoryLimitByteStorage
        }
        set {
            memoryLimitByteStorage = newValue.map { max(1, $0) } ?? -1
        }
    }

    /// Releases quantized Swift arrays after their Metal buffers enter this
    /// executor's cache. Enable only when that executor owns the model lifetime.
    public var releaseQuantizedHostStorageAfterUpload: Bool

    /// Allows cached quantized packed weights to be backed directly by mmap
    /// slices through `MTLBuffer(bytesNoCopy:)`. This is a performance knob:
    /// disabling it keeps the same kernels but forces copy-backed Metal buffers.
    public var quantizedNoCopyBuffersEnabled: Bool

    private var memoryLimitByteStorage: Int
    private var quantizedBufferCacheLimitByteStorage: Int

    public var quantizedBufferCacheLimitBytes: Int? {
        get {
            quantizedBufferCacheLimitByteStorage < 0 ? nil : quantizedBufferCacheLimitByteStorage
        }
        set {
            quantizedBufferCacheLimitByteStorage = newValue.map { max(0, $0) } ?? -1
        }
    }

    public init(
        maxOpsPerCommandBuffer: Int = 20,
        maxMBPerCommandBuffer: Int = 40,
        contextLengthHint: Int = 0,
        dynamicOpsSchedule: DynamicOpsSchedule? = nil,
        memoryLimitBytes: Int? = nil,
        commandBufferBatchingEnabled: Bool = false,
        maxInFlightCommandBuffers: Int = 16,
        quantizedBufferCacheLimitBytes: Int? = nil,
        releaseQuantizedHostStorageAfterUpload: Bool = false,
        quantizedNoCopyBuffersEnabled: Bool = true,
        useMLXQuantizedMatmul: Bool = false,
        useMLXQuantizedPrefillMatmul: Bool = false,
        useVendoredCommandBufferPrefillQMM: Bool = false,
        useSingleCommandBufferPrefill: Bool = false,
        useSingleCommandBufferDecode: Bool = false,
        usePrefillLayerCommandBufferBatching: Bool = false,
        useFusedGDNDecode: Bool = false,
        useCmlxFastRMSNorm: Bool = false,
        useCmlxLazyOutputHead: Bool = false,
        useGreedyOutputHeadArgmax: Bool = true,
        syncEval: Bool = false
    ) {
        self.maxOpsPerCommandBuffer = max(1, maxOpsPerCommandBuffer)
        self.maxMBPerCommandBuffer = max(1, maxMBPerCommandBuffer)
        self.contextLengthHint = max(0, contextLengthHint)
        self.dynamicOpsSchedule = dynamicOpsSchedule
        self.commandBufferBatchingEnabled = commandBufferBatchingEnabled
        self.maxInFlightCommandBuffers = max(1, maxInFlightCommandBuffers)
        self.useMLXQuantizedMatmul = useMLXQuantizedMatmul
        self.useMLXQuantizedPrefillMatmul = useMLXQuantizedPrefillMatmul
        self.useVendoredCommandBufferPrefillQMM = useVendoredCommandBufferPrefillQMM
        self.useSingleCommandBufferPrefill = useSingleCommandBufferPrefill
        self.useSingleCommandBufferDecode = useSingleCommandBufferDecode
        self.usePrefillLayerCommandBufferBatching = usePrefillLayerCommandBufferBatching
        self.useFusedGDNDecode = useFusedGDNDecode
        self.useCmlxFastRMSNorm = useCmlxFastRMSNorm
        self.useCmlxLazyOutputHead = useCmlxLazyOutputHead
        self.useGreedyOutputHeadArgmax = useGreedyOutputHeadArgmax
        self.syncEval = syncEval
        self.memoryLimitByteStorage = memoryLimitBytes.map { max(1, $0) } ?? -1
        self.releaseQuantizedHostStorageAfterUpload = releaseQuantizedHostStorageAfterUpload
        self.quantizedNoCopyBuffersEnabled = quantizedNoCopyBuffersEnabled
        self.quantizedBufferCacheLimitByteStorage = quantizedBufferCacheLimitBytes.map { max(0, $0) } ?? -1
    }

    public static func defaultConfiguration(for device: MTLDevice) -> MetalRuntimeConfiguration {
        let archSuffix: Character?
        if #available(macOS 13.0, iOS 16.0, *) {
            archSuffix = device.architecture.name.last
        } else {
            archSuffix = nil
        }

        let budgets: (ops: Int, mb: Int)
        switch archSuffix {
        case "p":
            budgets = (20, 40)
        case "g":
            budgets = (40, 40)
        case "s", "d":
            budgets = (50, 50)
        default:
            budgets = (40, 40)
        }

        return MetalRuntimeConfiguration(
            maxOpsPerCommandBuffer: budgets.ops,
            maxMBPerCommandBuffer: budgets.mb
        ).applyingEnvironmentOverrides()
    }

    public func applyingEnvironmentOverrides() -> MetalRuntimeConfiguration {
        var configuration = self
        configuration.maxOpsPerCommandBuffer = Self.environmentPositiveInt(
            names: ["EDGE_MAX_OPS_PER_BUFFER", "MLX_MAX_OPS_PER_BUFFER"],
            defaultValue: configuration.maxOpsPerCommandBuffer
        )
        configuration.maxMBPerCommandBuffer = Self.environmentPositiveInt(
            names: ["EDGE_MAX_MB_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER"],
            defaultValue: configuration.maxMBPerCommandBuffer
        )
        let memoryLimitBytes = Self.environmentPositiveInt(
            names: ["EDGE_MEMORY_LIMIT_BYTES", "MLX_MEMORY_LIMIT_BYTES"],
            defaultValue: configuration.memoryLimitBytes ?? 0
        )
        configuration.memoryLimitBytes = memoryLimitBytes > 0 ? memoryLimitBytes : nil
        configuration.commandBufferBatchingEnabled = Self.environmentBool(
            names: ["EDGE_BATCHING", "EDGE_COMMAND_BUFFER_BATCHING"],
            defaultValue: configuration.commandBufferBatchingEnabled
        )
        configuration.maxInFlightCommandBuffers = Self.environmentPositiveInt(
            names: ["EDGE_MAX_INFLIGHT_COMMAND_BUFFERS", "EDGE_MAX_IN_FLIGHT_COMMAND_BUFFERS"],
            defaultValue: configuration.maxInFlightCommandBuffers
        )
        configuration.useMLXQuantizedMatmul = Self.environmentBool(
            names: ["EDGE_USE_MLX_QMM"],
            defaultValue: configuration.useMLXQuantizedMatmul
        )
        configuration.useMLXQuantizedPrefillMatmul = Self.environmentBool(
            names: ["EDGE_USE_MLX_PREFILL_QMM", "EDGE_USE_VENDORED_PREFILL_QMM"],
            defaultValue: configuration.useMLXQuantizedPrefillMatmul
        )
        configuration.useVendoredCommandBufferPrefillQMM = Self.environmentBool(
            names: [
                "EDGE_USE_VENDORED_COMMAND_BUFFER_PREFILL_QMM",
                "EDGE_USE_VENDORED_PREFILL_QMM_ENCODER",
            ],
            defaultValue: configuration.useVendoredCommandBufferPrefillQMM
        )
        configuration.useSingleCommandBufferPrefill = Self.environmentBool(
            names: [
                "EDGE_USE_SINGLE_COMMAND_BUFFER_PREFILL",
                "EDGE_USE_SINGLE_CB_PREFILL",
            ],
            defaultValue: configuration.useSingleCommandBufferPrefill
        )
        configuration.useSingleCommandBufferDecode = Self.environmentBool(
            names: [
                "EDGE_USE_SINGLE_COMMAND_BUFFER_DECODE",
                "EDGE_USE_SINGLE_CB_DECODE",
            ],
            defaultValue: configuration.useSingleCommandBufferDecode
        )
        configuration.usePrefillLayerCommandBufferBatching = Self.environmentBool(
            names: [
                "EDGE_USE_PREFILL_LAYER_COMMAND_BUFFER_BATCHING",
                "EDGE_USE_PREFILL_LAYER_CB_BATCHING",
            ],
            defaultValue: configuration.usePrefillLayerCommandBufferBatching
        )
        configuration.useFusedGDNDecode = Self.environmentBool(
            names: [
                "EDGE_USE_FUSED_GDN_DECODE",
                "EDGE_FUSED_GDN_DECODE",
            ],
            defaultValue: configuration.useFusedGDNDecode
        )
        configuration.useCmlxFastRMSNorm = Self.environmentBool(
            names: [
                "EDGE_USE_CMLX_FAST_RMS_NORM",
                "EDGE_CMLX_FAST_RMS_NORM",
            ],
            defaultValue: configuration.useCmlxFastRMSNorm
        )
        configuration.useCmlxLazyOutputHead = Self.environmentBool(
            names: [
                "EDGE_USE_CMLX_LAZY_OUTPUT_HEAD",
                "EDGE_CMLX_LAZY_OUTPUT_HEAD",
            ],
            defaultValue: configuration.useCmlxLazyOutputHead
        )
        configuration.useGreedyOutputHeadArgmax = Self.environmentBool(
            names: [
                "EDGE_USE_GREEDY_OUTPUT_HEAD_ARGMAX",
                "EDGE_GREEDY_OUTPUT_HEAD_ARGMAX",
            ],
            defaultValue: configuration.useGreedyOutputHeadArgmax
        )
        configuration.syncEval = Self.environmentBool(
            names: ["EDGE_SYNC_EVAL", "EDGE_SYNCHRONOUS_EVAL"],
            defaultValue: configuration.syncEval
        )
        configuration.quantizedNoCopyBuffersEnabled = Self.environmentBool(
            names: ["EDGE_QUANTIZED_NOCOPY", "EDGE_QUANTIZED_NO_COPY"],
            defaultValue: configuration.quantizedNoCopyBuffersEnabled
        )
        return configuration
    }

    private static func environmentPositiveInt(names: [String], defaultValue: Int) -> Int {
        for name in names {
            guard let rawValue = getenv(name),
                  let value = Int(String(cString: rawValue)),
                  value > 0
            else {
                continue
            }
            return value
        }
        return defaultValue
    }

    private static func environmentBool(names: [String], defaultValue: Bool) -> Bool {
        for name in names {
            guard let rawValue = getenv(name) else {
                continue
            }
            switch String(cString: rawValue).lowercased() {
            case "1", "true", "yes", "on", "enabled":
                return true
            case "0", "false", "no", "off", "disabled":
                return false
            default:
                continue
            }
        }
        return defaultValue
    }

    public var effectiveMaxOpsPerCommandBuffer: Int {
        guard let dynamicOpsSchedule else {
            return maxOpsPerCommandBuffer
        }
        return dynamicOpsSchedule.effectiveMaxOps(
            configuredMaxOps: maxOpsPerCommandBuffer,
            contextLengthHint: contextLengthHint
        )
    }

    public var maxBytesPerCommandBuffer: Int {
        maxMBPerCommandBuffer * 1_048_576
    }

    public func allowsQuantizedBufferCacheEntry(
        byteCount: Int,
        currentCachedByteCount: Int
    ) -> Bool {
        guard let quantizedBufferCacheLimitBytes else {
            return true
        }
        guard byteCount >= 0, currentCachedByteCount >= 0 else {
            return false
        }
        return byteCount <= quantizedBufferCacheLimitBytes - currentCachedByteCount
    }
}
