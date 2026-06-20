// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Testing
@testable import EdgeEngine

@Test func edgeSpeechBundlePreflightLoadsQwen3ASRMetadata() throws {
    let root = try makeTemporarySpeechBundle()
    try writeJSON(
        """
        {
          "model_type": "qwen3_asr",
          "support_languages": ["Chinese", "English"],
          "thinker_config": {
            "audio_start_token_id": 151669,
            "audio_end_token_id": 151670,
            "audio_token_id": 151676
          }
        }
        """,
        to: root.appendingPathComponent("config.json")
    )
    try writeJSON(
        """
        {
          "feature_size": 128,
          "hop_length": 160,
          "n_fft": 400,
          "n_samples": 480000,
          "chunk_length": 30,
          "processor_class": "Qwen3ASRProcessor"
        }
        """,
        to: root.appendingPathComponent("preprocessor_config.json")
    )
    try writeJSON(makeASRTokenizerJSON(), to: root.appendingPathComponent("tokenizer.json"))
    try Data().write(to: root.appendingPathComponent("model.safetensors"))

    let result = try EdgeSpeechBundlePreflightRunner.run(
        configuration: EdgeSpeechBundlePreflightConfiguration(modelRootURL: root)
    )

    #expect(result.modelType == "qwen3_asr")
    #expect(result.plan.modelFamily == .qwen3ASR)
    #expect(result.plan.modality == .asr)
    #expect(result.passesPreflight)
    #expect(result.failureReasons.isEmpty)
    #expect(result.supportedLanguages == ["Chinese", "English"])
    #expect(result.audioTokenID == 151676)
    #expect(result.missingRequiredResourceNames == [])
    #expect(result.tokenizerSpecialTokenChecks.map(\.matches) == [true, true, true])
    #expect(result.tokenizerSpecialTokenChecks.map(\.actualContent) == [
        "<|audio_start|>",
        "<|audio_end|>",
        "<|audio_pad|>",
    ])

    let features = try #require(result.asrFeatureConfiguration)
    #expect(features.sampleRate == 16_000)
    #expect(features.fftSize == 400)
    #expect(features.hopLength == 160)
    #expect(features.melBinCount == 128)
    #expect(result.sampleRateChecks == [
        EdgeSpeechSampleRateCheck(
            source: "preprocessor_config.json",
            expectedSampleRate: 16_000,
            actualSampleRate: 16_000
        )
    ])

    let encoded = try JSONEncoder().encode(result)
    let encodedJSON = try #require(String(data: encoded, encoding: .utf8))
    #expect(encodedJSON.contains(#""passesPreflight":true"#))
    let decoded = try JSONDecoder().decode(EdgeSpeechBundlePreflightResult.self, from: encoded)
    #expect(decoded == result)

    var legacyObject = try #require(
        JSONSerialization.jsonObject(with: encoded) as? [String: Any]
    )
    legacyObject.removeValue(forKey: "failureReasons")
    legacyObject.removeValue(forKey: "passesPreflight")
    let legacyData = try JSONSerialization.data(withJSONObject: legacyObject)
    let legacyDecoded = try JSONDecoder().decode(EdgeSpeechBundlePreflightResult.self, from: legacyData)
    #expect(legacyDecoded.passesPreflight)
    #expect(legacyDecoded.failureReasons.isEmpty)
}

@Test func edgeSpeechBundlePreflightMarksASRNotPassingPreflightWhenTokenizerAudioTokenIsMissing() throws {
    let root = try makeTemporarySpeechBundle()
    try writeJSON(
        """
        {
          "model_type": "qwen3_asr",
          "thinker_config": {
            "audio_start_token_id": 151669,
            "audio_end_token_id": 151670,
            "audio_token_id": 151676
          }
        }
        """,
        to: root.appendingPathComponent("config.json")
    )
    try writeJSON(
        """
        {
          "sampling_rate": 16000,
          "feature_size": 128,
          "hop_length": 160,
          "n_fft": 400
        }
        """,
        to: root.appendingPathComponent("preprocessor_config.json")
    )
    try writeJSON(
        """
        {
          "added_tokens": [
            {
              "id": 151669,
              "content": "<|audio_start|>",
              "special": true
            },
            {
              "id": 151670,
              "content": "<|audio_end|>",
              "special": true
            }
          ]
        }
        """,
        to: root.appendingPathComponent("tokenizer.json")
    )
    try Data().write(to: root.appendingPathComponent("model.safetensors"))

    let result = try EdgeSpeechBundlePreflightRunner.run(
        configuration: EdgeSpeechBundlePreflightConfiguration(modelRootURL: root)
    )

    #expect(!result.passesPreflight)
    #expect(result.failureReasons == [.tokenizerSpecialTokenMismatch])
    #expect(result.missingRequiredResourceNames == [])
    let audioPad = try #require(result.tokenizerSpecialTokenChecks.last)
    #expect(audioPad.name == "audio_token_id")
    #expect(audioPad.actualContent == nil)
    #expect(!audioPad.matches)
}

@Test func edgeSpeechBundlePreflightLoadsQwen3TTSMetadata() throws {
    let root = try makeTemporarySpeechBundle()
    let speechTokenizerRoot = root.appendingPathComponent("speech_tokenizer")
    try FileManager.default.createDirectory(at: speechTokenizerRoot, withIntermediateDirectories: true)

    try writeJSON(
        """
        {
          "model_type": "qwen3_tts",
          "tts_model_type": "custom_voice",
          "tts_model_size": "0b6"
        }
        """,
        to: root.appendingPathComponent("config.json")
    )
    try writeJSON(makeASRTokenizerJSON(), to: root.appendingPathComponent("tokenizer.json"))
    try Data().write(to: root.appendingPathComponent("model.safetensors"))
    try writeJSON(
        """
        {
          "model_type": "qwen3_tts_tokenizer_12hz",
          "input_sample_rate": 24000,
          "output_sample_rate": 24000,
          "decode_upsample_rate": 1920,
          "encoder_valid_num_quantizers": 16
        }
        """,
        to: speechTokenizerRoot.appendingPathComponent("config.json")
    )
    try writeJSON(
        """
        {
          "feature_extractor_type": "EncodecFeatureExtractor",
          "feature_size": 1,
          "sampling_rate": 24000
        }
        """,
        to: speechTokenizerRoot.appendingPathComponent("preprocessor_config.json")
    )
    try Data().write(to: speechTokenizerRoot.appendingPathComponent("model.safetensors"))

    let result = try EdgeSpeechBundlePreflightRunner.run(
        configuration: EdgeSpeechBundlePreflightConfiguration(modelRootURL: root, modelFamily: .qwen3TTS)
    )

    #expect(result.modelType == "qwen3_tts")
    #expect(result.plan.modelFamily == .qwen3TTS)
    #expect(result.plan.modality == .tts)
    #expect(result.passesPreflight)
    #expect(result.failureReasons.isEmpty)
    #expect(result.ttsModelType == "custom_voice")
    #expect(result.ttsModelSize == "0b6")
    #expect(result.speechTokenizerModelType == "qwen3_tts_tokenizer_12hz")
    #expect(result.speechTokenizerInputSampleRate == 24_000)
    #expect(result.speechTokenizerOutputSampleRate == 24_000)
    #expect(result.speechTokenizerCodecSamplesPerFrame == 1_920)
    #expect(result.speechTokenizerCodecFrameRate == 12.5)
    #expect(result.speechTokenizerEncoderValidNumQuantizers == 16)
    #expect(result.sampleRateChecks.map(\.matches) == [true, true, true])
}

@Test func edgeSpeechBundlePreflightReportsMissingWeightResources() throws {
    let root = try makeTemporarySpeechBundle()
    try writeJSON(
        """
        {
          "model_type": "qwen3_asr",
          "thinker_config": {
            "audio_token_id": 151676
          }
        }
        """,
        to: root.appendingPathComponent("config.json")
    )
    try writeJSON(
        """
        {
          "sampling_rate": 16000,
          "feature_size": 128,
          "hop_length": 160,
          "n_fft": 400
        }
        """,
        to: root.appendingPathComponent("preprocessor_config.json")
    )
    try writeJSON(makeASRTokenizerJSON(), to: root.appendingPathComponent("tokenizer.json"))

    let result = try EdgeSpeechBundlePreflightRunner.run(
        configuration: EdgeSpeechBundlePreflightConfiguration(modelRootURL: root)
    )

    #expect(!result.passesPreflight)
    #expect(result.failureReasons == [.missingRequiredResources])
    #expect(result.missingRequiredResourceNames == [
        "model.safetensors or model.safetensors.index.json"
    ])
}

@Test func edgeSpeechBundlePreflightAcceptsShardedWeightIndexAsPresent() throws {
    let root = try makeTemporarySpeechBundle()
    try writeJSON(
        """
        {
          "model_type": "qwen3_asr",
          "thinker_config": {
            "audio_start_token_id": 151669,
            "audio_end_token_id": 151670,
            "audio_token_id": 151676
          }
        }
        """,
        to: root.appendingPathComponent("config.json")
    )
    try writeJSON(
        """
        {
          "sampling_rate": 16000,
          "feature_size": 128,
          "hop_length": 160,
          "n_fft": 400
        }
        """,
        to: root.appendingPathComponent("preprocessor_config.json")
    )
    try writeJSON(makeASRTokenizerJSON(), to: root.appendingPathComponent("tokenizer.json"))
    try writeJSON("{}", to: root.appendingPathComponent("model.safetensors.index.json"))

    let result = try EdgeSpeechBundlePreflightRunner.run(
        configuration: EdgeSpeechBundlePreflightConfiguration(modelRootURL: root)
    )

    #expect(result.passesPreflight)
    #expect(result.failureReasons.isEmpty)
    #expect(result.missingRequiredResourceNames == [])
}

@Test func edgeSpeechBundlePreflightMarksSampleRateMismatchAsNotPassingPreflight() throws {
    let root = try makeTemporarySpeechBundle()
    try writeJSON(
        """
        {
          "model_type": "qwen3_asr",
          "thinker_config": {
            "audio_start_token_id": 151669,
            "audio_end_token_id": 151670,
            "audio_token_id": 151676
          }
        }
        """,
        to: root.appendingPathComponent("config.json")
    )
    try writeJSON(
        """
        {
          "sampling_rate": 8000,
          "feature_size": 128,
          "hop_length": 160,
          "n_fft": 400
        }
        """,
        to: root.appendingPathComponent("preprocessor_config.json")
    )
    try writeJSON(makeASRTokenizerJSON(), to: root.appendingPathComponent("tokenizer.json"))
    try Data().write(to: root.appendingPathComponent("model.safetensors"))

    let result = try EdgeSpeechBundlePreflightRunner.run(
        configuration: EdgeSpeechBundlePreflightConfiguration(modelRootURL: root)
    )

    #expect(!result.passesPreflight)
    #expect(result.failureReasons == [.sampleRateMismatch])
    #expect(result.missingRequiredResourceNames == [])
    #expect(result.sampleRateChecks == [
        EdgeSpeechSampleRateCheck(
            source: "preprocessor_config.json",
            expectedSampleRate: 16_000,
            actualSampleRate: 8_000
        )
    ])
}

@Test func edgeSpeechBundlePreflightReportsCombinedFailureReasons() throws {
    let root = try makeTemporarySpeechBundle()
    try writeJSON(
        """
        {
          "model_type": "qwen3_asr",
          "thinker_config": {
            "audio_start_token_id": 151669,
            "audio_end_token_id": 151670,
            "audio_token_id": 151676
          }
        }
        """,
        to: root.appendingPathComponent("config.json")
    )
    try writeJSON(
        """
        {
          "sampling_rate": 8000,
          "feature_size": 128,
          "hop_length": 160,
          "n_fft": 400
        }
        """,
        to: root.appendingPathComponent("preprocessor_config.json")
    )
    try writeJSON(
        """
        {
          "added_tokens": [
            {
              "id": 151669,
              "content": "<|audio_start|>",
              "special": true
            },
            {
              "id": 151670,
              "content": "<|audio_end|>",
              "special": true
            }
          ]
        }
        """,
        to: root.appendingPathComponent("tokenizer.json")
    )

    let result = try EdgeSpeechBundlePreflightRunner.run(
        configuration: EdgeSpeechBundlePreflightConfiguration(modelRootURL: root)
    )

    #expect(!result.passesPreflight)
    #expect(result.failureReasons == [
        .missingRequiredResources,
        .sampleRateMismatch,
        .tokenizerSpecialTokenMismatch,
    ])
    #expect(result.missingRequiredResourceNames == [
        "model.safetensors or model.safetensors.index.json"
    ])
    #expect(result.sampleRateChecks.map(\.matches) == [false])
    #expect(result.tokenizerSpecialTokenChecks.map(\.matches) == [true, true, false])
}

@Test func edgeSpeechBundlePreflightRejectsUnsupportedSpeechModelType() throws {
    let root = try makeTemporarySpeechBundle()
    try writeJSON(
        """
        {
          "model_type": "qwen3_vlm"
        }
        """,
        to: root.appendingPathComponent("config.json")
    )

    do {
        _ = try EdgeSpeechBundlePreflightRunner.run(
            configuration: EdgeSpeechBundlePreflightConfiguration(modelRootURL: root)
        )
        Issue.record("Speech preflight must reject non-speech model types.")
    } catch EdgeSpeechBundlePreflightError.unsupportedModelType("qwen3_vlm") {
        return
    }
    Issue.record("Speech preflight threw the wrong error for unsupported model type.")
}

@Test func edgeSpeechBundlePreflightRejectsFamilyMismatch() throws {
    let root = try makeTemporarySpeechBundle()
    try writeJSON(
        """
        {
          "model_type": "qwen3_asr"
        }
        """,
        to: root.appendingPathComponent("config.json")
    )

    do {
        _ = try EdgeSpeechBundlePreflightRunner.run(
            configuration: EdgeSpeechBundlePreflightConfiguration(modelRootURL: root, modelFamily: .qwen3TTS)
        )
        Issue.record("Speech preflight must reject explicit family mismatches.")
    } catch EdgeSpeechBundlePreflightError.modelFamilyMismatch(expected: .qwen3TTS, actual: .qwen3ASR) {
        return
    }
    Issue.record("Speech preflight threw the wrong error for family mismatch.")
}

private func makeTemporarySpeechBundle() throws -> URL {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-speech-bundle-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
}

private func writeJSON(_ json: String, to url: URL) throws {
    try json.write(to: url, atomically: true, encoding: .utf8)
}

private func makeASRTokenizerJSON() -> String {
    """
    {
      "added_tokens": [
        {
          "id": 151669,
          "content": "<|audio_start|>",
          "special": true
        },
        {
          "id": 151670,
          "content": "<|audio_end|>",
          "special": true
        },
        {
          "id": 151676,
          "content": "<|audio_pad|>",
          "special": true
        }
      ]
    }
    """
}
