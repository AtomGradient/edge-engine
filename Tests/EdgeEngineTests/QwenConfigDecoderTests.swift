// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Testing
@testable import EdgeEngine

@Test func qwenConfigDecoderBuildsHybridArchitectureFromLocalJSON() throws {
    let json = """
    {
      "model_type": "qwen3.5",
      "vocab_size": 128,
      "hidden_size": 8,
      "intermediate_size": 32,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "linear_num_value_heads": 2,
      "linear_num_key_heads": 1,
      "linear_key_head_dim": 3,
      "linear_value_head_dim": 2,
      "linear_conv_kernel_dim": 4,
      "max_position_embeddings": 64,
      "rms_norm_eps": 1e-6,
      "rope_theta": 10000,
      "edgeruntime_layer_plan": ["full_attention", "gdn", "fa", "gated_delta_net"]
    }
    """
    let architecture = try QwenConfigDecoder.decodeArchitecture(
        from: try #require(json.data(using: .utf8))
    )

    #expect(architecture.family == .qwen35)
    #expect(architecture.fullAttentionLayerIndices == [0, 2])
    #expect(architecture.gdnLayerIndices == [1, 3])
    #expect(architecture.attentionHeadDimension == 4)
    #expect(architecture.linearValueHeadCount == 2)
    #expect(architecture.linearKeyHeadCount == 1)
    #expect(architecture.linearKeyHeadDimension == 3)
    #expect(architecture.linearValueHeadDimension == 2)
    #expect(architecture.linearConvKernelSize == 4)
}

@Test func qwenConfigDecoderBuildsHybridArchitectureFromPublicTextConfig() throws {
    let json = """
    {
      "model_type": "qwen3_5",
      "quantization": {
        "group_size": 64,
        "bits": 4
      },
      "text_config": {
        "model_type": "qwen3_5_text",
        "vocab_size": 248320,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_num_value_heads": 64,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 192,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
          "rope_theta": 10000000,
          "partial_rotary_factor": 0.25
        },
        "layer_types": [
          "linear_attention",
          "linear_attention",
          "linear_attention",
          "full_attention",
          "linear_attention",
          "linear_attention",
          "linear_attention",
          "full_attention"
        ]
      }
    }
    """
    let architecture = try QwenConfigDecoder.decodeArchitecture(
        from: try #require(json.data(using: .utf8))
    )

    #expect(architecture.family == .qwen35)
    #expect(architecture.layerCount == 8)
    #expect(architecture.fullAttentionLayerIndices == [3, 7])
    #expect(architecture.gdnLayerIndices == [0, 1, 2, 4, 5, 6])
    #expect(architecture.contextLength == 262144)
    #expect(architecture.ropeTheta == 10000000)
    #expect(architecture.partialRotaryFactor == 0.25)
    #expect(architecture.rotaryDimension == 64)
    #expect(architecture.linearKeyHiddenSize == 3_072)
    #expect(architecture.linearValueHiddenSize == 8_192)
    #expect(architecture.linearQKVHiddenSize == 14_336)
    #expect(architecture.quantization == QwenQuantizationProfile(groupSize: 64, bits: 4))
}

@Test func qwenConfigDecoderPreservesExplicitQwen36FamilyFromPublicTextConfig() throws {
    let json = """
    {
      "model_type": "qwen3_5",
      "text_config": {
        "model_type": "qwen3_5_text",
        "vocab_size": 248320,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
          "rope_theta": 10000000
        },
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
      }
    }
    """
    let architecture = try QwenConfigDecoder.decodeArchitecture(
        from: try #require(json.data(using: .utf8)),
        family: .qwen36
    )

    #expect(architecture.family == .qwen36)
    #expect(architecture.fullAttentionLayerIndices == [3])
    #expect(architecture.gdnLayerIndices == [0, 1, 2])
    #expect(architecture.attentionHeadDimension == 256)
}

@Test func qwenConfigDecoderBuildsSparseMoEArchitectureWithoutDenseIntermediateSize() throws {
    let json = """
    {
      "model_type": "qwen3_5_moe",
      "quantization": {
        "group_size": 64,
        "bits": 8
      },
      "text_config": {
        "model_type": "qwen3_5_moe_text",
        "vocab_size": 248320,
        "hidden_size": 2048,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
          "rope_theta": 10000000,
          "partial_rotary_factor": 0.25
        },
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
      }
    }
    """
    let architecture = try QwenConfigDecoder.decodeArchitecture(
        from: try #require(json.data(using: .utf8))
    )

    #expect(architecture.family == .qwen35)
    #expect(architecture.intermediateSize == 4_096)
    #expect(architecture.usesSparseMoEMLP)
    #expect(architecture.moeMLP?.expertCount == 256)
    #expect(architecture.moeMLP?.expertsPerToken == 8)
    #expect(architecture.moeMLP?.intermediateSize == 512)
    #expect(architecture.moeMLP?.sharedExpertIntermediateSize == 512)
    #expect(architecture.moeMLP?.normalizeTopKProbabilities == true)
    #expect(architecture.attentionHeadDimension == 256)
    #expect(architecture.rotaryDimension == 64)
    #expect(architecture.quantization == QwenQuantizationProfile(groupSize: 64, bits: 8))
}

@Test func qwenConfigDecoderRejectsConfigsWithoutPublicOrExplicitHybridLayerPlan() throws {
    let json = """
    {
      "model_type": "qwen3.6",
      "text_config": {
        "vocab_size": 128,
        "hidden_size": 8,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "context_length": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "num_hidden_layers": 4
      }
    }
    """

    var rejectedMissingLayerPlan = false
    do {
        _ = try QwenConfigDecoder.decodeArchitecture(from: try #require(json.data(using: .utf8)))
        Issue.record("Qwen config decoder must not infer a layer plan from num_hidden_layers alone.")
    } catch QwenConfigDecoderError.missingHybridLayerPlan {
        rejectedMissingLayerPlan = true
    }
    #expect(rejectedMissingLayerPlan)
}
