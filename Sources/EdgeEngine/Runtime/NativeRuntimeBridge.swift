// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public typealias NativeQwenModelFamily = QwenModelFamily

/// Public Swift bridge for the native inference engine.
///
/// This type intentionally stays inside `edge-engine` so no-MLX
/// consumers can configure Metal scheduling, inspect Qwen plans, and build
/// speech runtime contracts without importing EdgeKit's MLX compat layer.
public enum NativeRuntimeBridge {
    public static var runtimeVersion: String {
        EdgeEngine.version
    }

    public static func defaultMetalConfiguration(
        contextLengthHint: Int = 0
    ) -> MetalRuntimeConfiguration {
        metalConfiguration(
            maxOpsPerCommandBuffer: 20,
            maxMBPerCommandBuffer: 40,
            contextLengthHint: contextLengthHint,
            dynamicOpsSchedule: DynamicOpsSchedule(
                floor: 5,
                contextLow: 4_096,
                contextHigh: 12_288
            )
        )
    }

    public static func metalConfiguration(
        maxOpsPerCommandBuffer: Int,
        maxMBPerCommandBuffer: Int,
        contextLengthHint: Int = 0,
        dynamicOpsSchedule: DynamicOpsSchedule? = nil,
        memoryLimitBytes: Int? = nil,
        quantizedBufferCacheLimitBytes: Int? = nil,
        releaseQuantizedHostStorageAfterUpload: Bool = false,
        commandBufferBatchingEnabled: Bool = true,
        maxInFlightCommandBuffers: Int = 16,
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
    ) -> MetalRuntimeConfiguration {
        MetalRuntimeConfiguration(
            maxOpsPerCommandBuffer: maxOpsPerCommandBuffer,
            maxMBPerCommandBuffer: maxMBPerCommandBuffer,
            contextLengthHint: contextLengthHint,
            dynamicOpsSchedule: dynamicOpsSchedule,
            memoryLimitBytes: memoryLimitBytes,
            commandBufferBatchingEnabled: commandBufferBatchingEnabled,
            maxInFlightCommandBuffers: maxInFlightCommandBuffers,
            quantizedBufferCacheLimitBytes: quantizedBufferCacheLimitBytes,
            releaseQuantizedHostStorageAfterUpload: releaseQuantizedHostStorageAfterUpload,
            quantizedNoCopyBuffersEnabled: quantizedNoCopyBuffersEnabled,
            useMLXQuantizedMatmul: useMLXQuantizedMatmul,
            useMLXQuantizedPrefillMatmul: useMLXQuantizedPrefillMatmul,
            useVendoredCommandBufferPrefillQMM: useVendoredCommandBufferPrefillQMM,
            useSingleCommandBufferPrefill: useSingleCommandBufferPrefill,
            useSingleCommandBufferDecode: useSingleCommandBufferDecode,
            usePrefillLayerCommandBufferBatching: usePrefillLayerCommandBufferBatching,
            useFusedGDNDecode: useFusedGDNDecode,
            useCmlxFastRMSNorm: useCmlxFastRMSNorm,
            useCmlxLazyOutputHead: useCmlxLazyOutputHead,
            useGreedyOutputHeadArgmax: useGreedyOutputHeadArgmax,
            syncEval: syncEval
        )
    }

    @discardableResult
    public static func applyMetalConfiguration(
        _ configuration: MetalRuntimeConfiguration
    ) -> MetalRuntimeConfiguration {
        EdgeEngineMetalConfigurationStore.shared.apply(configuration)
    }

    @discardableResult
    public static func applyContextLengthHint(_ contextLengthHint: Int) -> MetalRuntimeConfiguration {
        var configuration = EdgeEngineMetalConfigurationStore.shared.currentConfiguration
        configuration.contextLengthHint = max(0, contextLengthHint)
        return applyMetalConfiguration(configuration)
    }

    public static var currentMetalConfiguration: MetalRuntimeConfiguration {
        EdgeEngineMetalConfigurationStore.shared.currentConfiguration
    }

    public static func makeNativeMetalRuntime() throws -> EdgeMetalRuntime {
        try EdgeEngineMetalConfigurationStore.shared.makeRuntime()
    }

    public static func makeQwenExecutionPlan(
        configData: Data,
        family: NativeQwenModelFamily? = nil
    ) throws -> QwenExecutionPlan {
        try QwenExecutionPlan(
            architecture: QwenConfigDecoder.decodeArchitecture(from: configData, family: family)
        )
    }

    /// Builds the Qwen-specific VLM bundle index.
    ///
    /// This is intentionally a per-family bridge, not a generic VLM abstraction.
    /// The current implementation depends on Qwen3.5/3.6 FA+GDN decoder manifests,
    /// Qwen tokenizer special tokens, and Qwen VLM bundle layout. Future non-Qwen
    /// VLMs should add their own family-specific bridge until a shared contract is
    /// proven by implementation.
    public static func makeQwenVLMModelBundleIndex(
        modelRootURL: URL,
        family: QwenVLMModelFamily? = nil
    ) throws -> QwenVLMModelBundleIndex {
        try QwenVLMModelBundleIndex.load(
            from: modelRootURL,
            family: family
        )
    }

    /// Builds the Qwen-specific VLM phased loading plan.
    ///
    /// Keep this API family-scoped: the peak memory formula and loading stages are
    /// validated for the Qwen VLM native path, not for arbitrary VLM families.
    public static func makeQwenVLMPhasedLoadingPlan(
        modelRootURL: URL,
        family: QwenVLMModelFamily? = nil,
        jetsamLimitMB: Int,
        appReserveMB: Int = 1_024,
        structureBaselineMB: Int = 256,
        activationReserveMB: Int = 800
    ) throws -> QwenVLMPhasedLoadingPlan {
        try makeQwenVLMModelBundleIndex(
            modelRootURL: modelRootURL,
            family: family
        )
        .makePhasedLoadingPlan(
            jetsamLimitMB: jetsamLimitMB,
            appReserveMB: appReserveMB,
            structureBaselineMB: structureBaselineMB,
            activationReserveMB: activationReserveMB
        )
    }

    /// Loads only the Qwen VLM native structure and runtime metadata.
    ///
    /// Decoder and vision weights remain family-specific staged operations. This
    /// method deliberately avoids a generic `loadVLMStructureOnly` name until a
    /// non-Qwen VLM is implemented on the same runtime contract.
    public static func loadQwenVLMStructureOnly(
        modelRootURL: URL,
        family: QwenVLMModelFamily? = nil,
        runtimeConfiguration: MetalRuntimeConfiguration =
            EdgeEngineMetalConfigurationStore.shared.currentConfiguration
    ) throws -> QwenVLMNativeContainer {
        try QwenVLMNativeContainer.loadStructureOnly(
            from: modelRootURL,
            family: family,
            runtimeConfiguration: runtimeConfiguration
        )
    }

    public static func makeQwen3ASRPlan() throws -> EdgeSpeechRuntimePlan {
        try .qwen3ASR
    }

    public static func makeQwen3TTSPlan() throws -> EdgeSpeechRuntimePlan {
        try .qwen3TTS
    }

    public static func qwenASRFeatureConfiguration() throws -> EdgeLogMelSpectrogramConfiguration {
        try .qwenASRDefault
    }
}
