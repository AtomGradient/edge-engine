// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public enum EdgeSpeechModelFamily: String, Codable, Equatable, Sendable {
    case qwen3ASR
    case qwen3TTS
}

public enum EdgeSpeechModality: String, Codable, Equatable, Sendable {
    case asr
    case tts
}

public struct EdgeSpeechRuntimePlan: Codable, Equatable, Sendable {
    public var modelFamily: EdgeSpeechModelFamily
    public var modality: EdgeSpeechModality
    public var preferredSampleRate: Int
    public var supportsStreaming: Bool

    public init(
        modelFamily: EdgeSpeechModelFamily,
        modality: EdgeSpeechModality,
        preferredSampleRate: Int,
        supportsStreaming: Bool
    ) throws {
        guard preferredSampleRate > 0 else {
            throw EdgeAudioError.invalidSampleRate(preferredSampleRate)
        }
        self.modelFamily = modelFamily
        self.modality = modality
        self.preferredSampleRate = preferredSampleRate
        self.supportsStreaming = supportsStreaming
    }

    public static var qwen3ASR: EdgeSpeechRuntimePlan {
        get throws {
            try EdgeSpeechRuntimePlan(
                modelFamily: .qwen3ASR,
                modality: .asr,
                preferredSampleRate: 16_000,
                supportsStreaming: false
            )
        }
    }

    public static var qwen3TTS: EdgeSpeechRuntimePlan {
        get throws {
            try EdgeSpeechRuntimePlan(
                modelFamily: .qwen3TTS,
                modality: .tts,
                preferredSampleRate: 24_000,
                supportsStreaming: false
            )
        }
    }
}

public struct EdgeASRRequest: Codable, Equatable, Sendable {
    public var audio: EdgeAudioBuffer
    public var languageHint: String?
    public var maxTokens: Int
    public var featureConfiguration: EdgeLogMelSpectrogramConfiguration

    public init(
        audio: EdgeAudioBuffer,
        languageHint: String? = nil,
        maxTokens: Int = 8_192,
        featureConfiguration: EdgeLogMelSpectrogramConfiguration
    ) {
        self.audio = audio
        self.languageHint = languageHint
        self.maxTokens = max(1, maxTokens)
        self.featureConfiguration = featureConfiguration
    }

    public func logMelFeatures() throws -> EdgeLogMelSpectrogram {
        try EdgeLogMelSpectrogram.extract(from: audio, configuration: featureConfiguration)
    }
}

public struct EdgeASRSegment: Codable, Equatable, Sendable {
    public var text: String
    public var startTimeSeconds: Double
    public var endTimeSeconds: Double

    public init(text: String, startTimeSeconds: Double, endTimeSeconds: Double) {
        self.text = text
        self.startTimeSeconds = startTimeSeconds
        self.endTimeSeconds = endTimeSeconds
    }
}

public struct EdgeTTSRequest: Codable, Equatable, Sendable {
    public var text: String
    public var voiceIdentifier: String?
    public var targetSampleRate: Int
    public var maxCodecTokens: Int

    public init(
        text: String,
        voiceIdentifier: String? = nil,
        targetSampleRate: Int = 24_000,
        maxCodecTokens: Int = 2_048
    ) throws {
        guard targetSampleRate > 0 else {
            throw EdgeAudioError.invalidSampleRate(targetSampleRate)
        }
        self.text = text
        self.voiceIdentifier = voiceIdentifier
        self.targetSampleRate = targetSampleRate
        self.maxCodecTokens = max(1, maxCodecTokens)
    }
}

public struct EdgeTTSAudioChunk: Codable, Equatable, Sendable {
    public var audio: EdgeAudioBuffer
    public var codecTokenCount: Int
    public var isFinal: Bool

    public init(audio: EdgeAudioBuffer, codecTokenCount: Int, isFinal: Bool) {
        self.audio = audio
        self.codecTokenCount = max(0, codecTokenCount)
        self.isFinal = isFinal
    }
}
