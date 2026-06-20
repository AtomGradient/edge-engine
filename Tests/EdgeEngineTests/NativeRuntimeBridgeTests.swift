// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Testing
@testable import EdgeEngine

@Test func nativeRuntimeBridgeExposesRuntimeVersionAndSchedulingDefaults() {
    #expect(NativeRuntimeBridge.runtimeVersion == EdgeEngine.version)

    let configuration = NativeRuntimeBridge.defaultMetalConfiguration(contextLengthHint: 12_288)

    #expect(configuration.maxOpsPerCommandBuffer == 20)
    #expect(configuration.effectiveMaxOpsPerCommandBuffer == 5)
    #expect(configuration.maxMBPerCommandBuffer == 40)
    #expect(configuration.useMLXQuantizedMatmul == false)
    #expect(configuration.useMLXQuantizedPrefillMatmul == false)
    #expect(configuration.useSingleCommandBufferPrefill == false)
    #expect(configuration.useSingleCommandBufferDecode == false)
    #expect(configuration.usePrefillLayerCommandBufferBatching == false)
    #expect(configuration.useFusedGDNDecode == false)
    #expect(configuration.syncEval == false)
}

@Test func nativeRuntimeBridgeAppliesMetalConfiguration() {
    let configuration = NativeRuntimeBridge.applyMetalConfiguration(
        NativeRuntimeBridge.metalConfiguration(
            maxOpsPerCommandBuffer: 15,
            maxMBPerCommandBuffer: 7,
            contextLengthHint: 12_288,
            dynamicOpsSchedule: DynamicOpsSchedule(
                floor: 5,
                contextLow: 4_096,
                contextHigh: 12_288
            ),
            memoryLimitBytes: 6 * 1_073_741_824,
            quantizedBufferCacheLimitBytes: 512 * 1_048_576,
            releaseQuantizedHostStorageAfterUpload: true,
            useSingleCommandBufferPrefill: true,
            useSingleCommandBufferDecode: true,
            usePrefillLayerCommandBufferBatching: true,
            useFusedGDNDecode: true,
            syncEval: true
        )
    )

    let current = NativeRuntimeBridge.currentMetalConfiguration
    #expect(current.maxOpsPerCommandBuffer == configuration.maxOpsPerCommandBuffer)
    #expect(current.maxMBPerCommandBuffer == configuration.maxMBPerCommandBuffer)
    #expect(current.contextLengthHint == configuration.contextLengthHint)
    #expect(current.effectiveMaxOpsPerCommandBuffer == configuration.effectiveMaxOpsPerCommandBuffer)
    #expect(current.dynamicOpsSchedule?.floor == configuration.dynamicOpsSchedule?.floor)
    #expect(current.dynamicOpsSchedule?.contextLow == configuration.dynamicOpsSchedule?.contextLow)
    #expect(current.dynamicOpsSchedule?.contextHigh == configuration.dynamicOpsSchedule?.contextHigh)
    #expect(current.memoryLimitBytes == configuration.memoryLimitBytes)
    #expect(current.quantizedBufferCacheLimitBytes == configuration.quantizedBufferCacheLimitBytes)
    #expect(
        current.releaseQuantizedHostStorageAfterUpload ==
            configuration.releaseQuantizedHostStorageAfterUpload
    )
    #expect(current.useSingleCommandBufferPrefill == true)
    #expect(current.useSingleCommandBufferDecode == true)
    #expect(current.usePrefillLayerCommandBufferBatching == true)
    #expect(current.useFusedGDNDecode == true)
    #expect(current.syncEval == true)
}

@Test func nativeRuntimeBridgeBuildsQwenPlanFromConfigJSON() throws {
    let json = """
    {
      "model_type": "qwen3_5",
      "vocab_size": 1024,
      "hidden_size": 8,
      "intermediate_size": 16,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "head_dim": 4,
      "max_position_embeddings": 4096,
      "rms_norm_eps": 1e-6,
      "rope_theta": 10000,
      "layer_types": ["full_attention", "linear_attention", "linear_attention"]
    }
    """

    let plan = try NativeRuntimeBridge.makeQwenExecutionPlan(
        configData: #require(json.data(using: .utf8))
    )

    #expect(plan.architecture.family == .qwen35)
    #expect(plan.fullAttentionSteps.map(\.layerIndex) == [0])
    #expect(plan.gdnSteps.map(\.layerIndex) == [1, 2])
}

@Test func nativeRuntimeBridgeExposesNativeSpeechPlans() throws {
    let asr = try NativeRuntimeBridge.makeQwen3ASRPlan()
    let tts = try NativeRuntimeBridge.makeQwen3TTSPlan()
    let features = try NativeRuntimeBridge.qwenASRFeatureConfiguration()

    #expect(asr.modelFamily == .qwen3ASR)
    #expect(asr.modality == .asr)
    #expect(asr.preferredSampleRate == 16_000)
    #expect(tts.modelFamily == .qwen3TTS)
    #expect(tts.modality == .tts)
    #expect(tts.preferredSampleRate == 24_000)
    #expect(features.sampleRate == 16_000)
    #expect(features.melBinCount == 128)
}
