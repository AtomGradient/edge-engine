// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Tokenizers

public enum Qwen3SpeechNativeSessionError: Error, Equatable, CustomStringConvertible {
    case invalidTTSModelConfiguration
    case invalidASRModelConfiguration
    case missingSafeTensors
    case missingSpeechTokenizer
    case missingTokenizerConfig
    case invalidTokenizerJSON
    case missingSpeaker(String)
    case missingLanguage(String)
    case tokenizerMissingToken(String)
    case audioFeatureMismatch
    case transcriptionModeRequired
    case emptyTokenIDs

    public var description: String {
        switch self {
        case .invalidTTSModelConfiguration:
            "model config is not a Qwen3-TTS config"
        case .invalidASRModelConfiguration:
            "model config is not a Qwen3-ASR config"
        case .missingSafeTensors:
            "model does not contain safetensors weights"
        case .missingSpeechTokenizer:
            "model does not contain speech tokenizer weights"
        case .missingTokenizerConfig:
            "model does not contain tokenizer_config.json"
        case .invalidTokenizerJSON:
            "model tokenizer.json could not be patched"
        case .missingSpeaker(let speaker):
            "speaker not found: \(speaker)"
        case .missingLanguage(let language):
            "language not found: \(language)"
        case .tokenizerMissingToken(let token):
            "tokenizer missing required token \(token)"
        case .audioFeatureMismatch:
            "audio feature/token count mismatch"
        case .transcriptionModeRequired:
            "Qwen3 ASR session was created in encoder-only mode"
        case .emptyTokenIDs:
            "tokenizer returned empty token ids"
        }
    }
}

public struct Qwen3TTSSynthesisRequest: Sendable {
    public var text: String
    public var speaker: String?
    public var language: String
    public var maxTokens: Int
    public var temperature: Float
    public var topK: Int
    public var seed: UInt64
    public var decodeAudio: Bool

    public init(
        text: String,
        speaker: String? = nil,
        language: String = "auto",
        maxTokens: Int = 2_048,
        temperature: Float = 0.9,
        topK: Int = 50,
        seed: UInt64 = 0,
        decodeAudio: Bool = true
    ) {
        self.text = text
        self.speaker = speaker
        self.language = language
        self.maxTokens = max(1, maxTokens)
        self.temperature = max(0, temperature)
        self.topK = max(0, topK)
        self.seed = seed
        self.decodeAudio = decodeAudio
    }
}

public struct Qwen3TTSSynthesisResult: Sendable {
    public var prompt: String
    public var targetTokenCount: Int
    public var speakerID: Int?
    public var codecTokens: EdgeMLXTTSCodes
    public var audio: EdgeAudioBuffer?
    public var generationSeconds: TimeInterval
    public var decodeSeconds: TimeInterval?

    public init(
        prompt: String,
        targetTokenCount: Int,
        speakerID: Int?,
        codecTokens: EdgeMLXTTSCodes,
        audio: EdgeAudioBuffer?,
        generationSeconds: TimeInterval,
        decodeSeconds: TimeInterval?
    ) {
        self.prompt = prompt
        self.targetTokenCount = targetTokenCount
        self.speakerID = speakerID
        self.codecTokens = codecTokens
        self.audio = audio
        self.generationSeconds = generationSeconds
        self.decodeSeconds = decodeSeconds
    }
}

public struct Qwen3TTSNativeMetadata: Sendable {
    public var supportedLanguages: [String]
    public var availableSpeakers: [String]
    public var outputSampleRate: Int
    public var decodeUpsampleRate: Int

    public init(
        supportedLanguages: [String],
        availableSpeakers: [String],
        outputSampleRate: Int,
        decodeUpsampleRate: Int
    ) {
        self.supportedLanguages = supportedLanguages
        self.availableSpeakers = availableSpeakers
        self.outputSampleRate = outputSampleRate
        self.decodeUpsampleRate = decodeUpsampleRate
    }
}

public final class Qwen3TTSNativeSession {
    public let modelURL: URL
    public let metadata: Qwen3TTSNativeMetadata

    private let config: Qwen3TTSRawConfig
    private let speechConfig: Qwen3TTSSpeechTokenizerRawConfig
    private let tokenizer: Tokenizer
    private let session: EdgeMLXQwen35Session
    private let speechTokenizerURL: URL
    private var didLoadSpeechTokenizer = false

    public init(
        modelURL: URL,
        runtimeConfiguration: MetalRuntimeConfiguration =
            EdgeEngineMetalConfigurationStore.shared.currentConfiguration
    ) async throws {
        self.modelURL = modelURL
        let config = try Qwen3TTSRawConfig.load(from: modelURL)
        let speechConfig = try Qwen3TTSSpeechTokenizerRawConfig.load(from: modelURL)
        let tokenizer = try await Qwen3TokenizerLoader.loadPatchedTokenizer(
            modelURL: modelURL,
            specialTokens: qwen3TTSSpecialTokens
        )
        let runtime = try EdgeMetalRuntime(configuration: runtimeConfiguration)
        let session = try EdgeMLXQwen35Session(
            architecture: try config.makeTalkerArchitecture(),
            runtime: runtime
        )
        let quantization = config.quantizationOrDefault
        try session.loadTTSSafetensors(
            shardURLs: try Qwen3SpeechSafetensors.shardURLs(
                modelURL: modelURL,
                include: { $0.hasPrefix("talker.") }
            ),
            groupSize: quantization.groupSize,
            bits: quantization.bits,
            codePredictorArchitecture: try config.makeCodePredictorArchitecture()
        )
        let speechTokenizerURL = modelURL
            .appendingPathComponent("speech_tokenizer")
            .appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: speechTokenizerURL.path) else {
            throw Qwen3SpeechNativeSessionError.missingSpeechTokenizer
        }

        self.config = config
        self.speechConfig = speechConfig
        self.tokenizer = tokenizer
        self.session = session
        self.speechTokenizerURL = speechTokenizerURL
        self.metadata = Qwen3TTSNativeMetadata(
            supportedLanguages: config.supportedLanguages,
            availableSpeakers: config.talkerConfig.spkID.keys.sorted(),
            outputSampleRate: speechConfig.outputSampleRate,
            decodeUpsampleRate: speechConfig.decodeUpsampleRate
        )
    }

    public static func targetPrompt(for text: String) -> String {
        "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
    }

    public static func targetTokenCount(modelURL: URL, text: String) async throws -> Int {
        let tokenizer = try await Qwen3TokenizerLoader.loadPatchedTokenizer(
            modelURL: modelURL,
            specialTokens: qwen3TTSSpecialTokens
        )
        let tokens = tokenizer.encode(text: targetPrompt(for: text))
        guard !tokens.isEmpty else {
            throw Qwen3SpeechNativeSessionError.emptyTokenIDs
        }
        return tokens.count
    }

    public func synthesize(_ request: Qwen3TTSSynthesisRequest) throws -> Qwen3TTSSynthesisResult {
        let prompt = Self.targetPrompt(for: request.text)
        let targetTokens = tokenizer.encode(text: prompt)
        guard !targetTokens.isEmpty else {
            throw Qwen3SpeechNativeSessionError.emptyTokenIDs
        }
        let speakerID = try config.speakerID(request.speaker)
        let codecPrefix = try config.codecPrefix(language: request.language)
        let started = Date()
        let codes = try session.generateTTSCodes(
            targetTokenIDs: targetTokens,
            ttsBOSTokenID: config.ttsBOSTokenID,
            ttsEOSTokenID: config.ttsEOSTokenID,
            ttsPADTokenID: config.ttsPADTokenID,
            codecPrefixTokenIDs: codecPrefix,
            codecPADTokenID: config.talkerConfig.codecPadID,
            codecBOSTokenID: config.talkerConfig.codecBosID,
            codecEOSTokenID: config.talkerConfig.codecEOSTokenID,
            speakerID: speakerID,
            maxTokens: request.maxTokens,
            temperature: request.temperature,
            topK: request.topK,
            seed: request.seed
        )
        let generationSeconds = Date().timeIntervalSince(started)

        var audio: EdgeAudioBuffer?
        var decodeSeconds: TimeInterval?
        if request.decodeAudio {
            let decodeStarted = Date()
            try ensureSpeechTokenizerLoaded()
            audio = try session.decodeTTSAudio(
                codes: codes,
                sampleRate: speechConfig.outputSampleRate,
                decodeUpsampleRate: speechConfig.decodeUpsampleRate
            )
            decodeSeconds = Date().timeIntervalSince(decodeStarted)
        }

        return Qwen3TTSSynthesisResult(
            prompt: prompt,
            targetTokenCount: targetTokens.count,
            speakerID: speakerID,
            codecTokens: codes,
            audio: audio,
            generationSeconds: generationSeconds,
            decodeSeconds: decodeSeconds
        )
    }

    private func ensureSpeechTokenizerLoaded() throws {
        guard !didLoadSpeechTokenizer else { return }
        try session.loadTTSSpeechTokenizerSafetensors(safetensorsURL: speechTokenizerURL)
        didLoadSpeechTokenizer = true
    }
}

public enum Qwen3ASRNativeSessionMode: Sendable {
    case encoderOnly
    case transcription
}

public struct Qwen3ASRAudioEncodingResult: Sendable {
    public var inputFrameCount: Int
    public var encoding: EdgeMLXQwen35AudioEncoding

    public init(inputFrameCount: Int, encoding: EdgeMLXQwen35AudioEncoding) {
        self.inputFrameCount = inputFrameCount
        self.encoding = encoding
    }
}

public struct Qwen3ASRTranscriptionRequest: Sendable {
    public var audio: EdgeAudioBuffer
    public var language: String
    public var maxTokens: Int
    public var maxAudioSeconds: Double?

    public init(
        audio: EdgeAudioBuffer,
        language: String = "English",
        maxTokens: Int = 8_192,
        maxAudioSeconds: Double? = nil
    ) {
        self.audio = audio
        self.language = language
        self.maxTokens = max(1, maxTokens)
        self.maxAudioSeconds = maxAudioSeconds
    }
}

public struct Qwen3ASRTranscriptionResult: Sendable {
    public var text: String
    public var language: String
    public var promptTokenCount: Int
    public var audioTokenCount: Int
    public var inputFrameCount: Int
    public var generatedTokenIDs: [Int]
    public var prefillSeconds: TimeInterval
    public var decodeSeconds: TimeInterval

    public init(
        text: String,
        language: String,
        promptTokenCount: Int,
        audioTokenCount: Int,
        inputFrameCount: Int,
        generatedTokenIDs: [Int],
        prefillSeconds: TimeInterval,
        decodeSeconds: TimeInterval
    ) {
        self.text = text
        self.language = language
        self.promptTokenCount = promptTokenCount
        self.audioTokenCount = audioTokenCount
        self.inputFrameCount = inputFrameCount
        self.generatedTokenIDs = generatedTokenIDs
        self.prefillSeconds = prefillSeconds
        self.decodeSeconds = decodeSeconds
    }
}

public struct Qwen3ASRNativeMetadata: Sendable {
    public var supportedLanguages: [String]
    public var audioMelBinCount: Int
    public var preferredSampleRate: Int

    public init(
        supportedLanguages: [String],
        audioMelBinCount: Int,
        preferredSampleRate: Int
    ) {
        self.supportedLanguages = supportedLanguages
        self.audioMelBinCount = audioMelBinCount
        self.preferredSampleRate = preferredSampleRate
    }
}

public final class Qwen3ASRNativeSession {
    public let modelURL: URL
    public let mode: Qwen3ASRNativeSessionMode
    public let metadata: Qwen3ASRNativeMetadata

    private let config: Qwen3ASRRawConfig
    private let tokenizer: Tokenizer?
    private let audioTokenID: Int?
    private let session: EdgeMLXQwen35Session

    public init(
        modelURL: URL,
        audioPrefix: String = "audio_tower",
        mode: Qwen3ASRNativeSessionMode = .transcription,
        runtimeConfiguration: MetalRuntimeConfiguration =
            EdgeEngineMetalConfigurationStore.shared.currentConfiguration
    ) async throws {
        self.modelURL = modelURL
        self.mode = mode
        let config = try Qwen3ASRRawConfig.load(from: modelURL)
        let runtime = try EdgeMetalRuntime(configuration: runtimeConfiguration)
        let session = try EdgeMLXQwen35Session(
            architecture: try mode == .transcription
                ? config.makeTextArchitecture()
                : config.makeEncoderSmokeArchitecture(),
            runtime: runtime
        )
        try session.setAudioConfig(config.makeAudioTowerConfiguration())
        try session.loadAudioSafetensors(
            shardURLs: try Qwen3SpeechSafetensors.shardURLs(
                modelURL: modelURL,
                include: { $0.hasPrefix(audioPrefix + ".") }
            ),
            audioPrefix: audioPrefix
        )

        let tokenizer: Tokenizer?
        let audioTokenID: Int?
        if mode == .transcription {
            try session.loadSafetensors(
                shardURLs: try Qwen3SpeechSafetensors.shardURLs(
                    modelURL: modelURL,
                    include: { $0.hasPrefix("model.") || $0.hasPrefix("lm_head.") }
                ),
                modelPrefix: "model",
                groupSize: config.quantization.groupSize,
                bits: config.quantization.bits
            )
            let loadedTokenizer = try await Qwen3TokenizerLoader.loadPatchedTokenizer(
                modelURL: modelURL,
                specialTokens: qwen3ASRSpecialTokens
            )
            tokenizer = loadedTokenizer
            audioTokenID = try Self.requiredTokenID("<|audio_pad|>", tokenizer: loadedTokenizer)
        } else {
            tokenizer = nil
            audioTokenID = nil
        }

        self.config = config
        self.session = session
        self.tokenizer = tokenizer
        self.audioTokenID = audioTokenID
        self.metadata = Qwen3ASRNativeMetadata(
            supportedLanguages: config.supportLanguages,
            audioMelBinCount: config.thinkerConfig.audioConfig.numMelBins,
            preferredSampleRate: 16_000
        )
    }

    public func encode(
        audio: EdgeAudioBuffer,
        maxAudioSeconds: Double? = nil
    ) throws -> Qwen3ASRAudioEncodingResult {
        let normalized = try audio.resampled(to: metadata.preferredSampleRate)
            .truncated(to: maxAudioSeconds)
        let logMel = try EdgeLogMelSpectrogram.extract(
            from: normalized,
            configuration: .qwenASRDefault
        )
        return try encode(
            logMelFeatures: logMel.frames.flatMap { $0 },
            featureShape: [logMel.frames.count, metadata.audioMelBinCount]
        )
    }

    public func encode(
        logMelFeatures: [Float],
        featureShape: [Int]
    ) throws -> Qwen3ASRAudioEncodingResult {
        let encoding = try session.audioEncode(
            logMelFeatures: logMelFeatures,
            featureShape: featureShape
        )
        return Qwen3ASRAudioEncodingResult(
            inputFrameCount: featureShape[0],
            encoding: encoding
        )
    }

    public func decodedPrompt(
        audioTokenCount: Int,
        language: String
    ) throws -> String {
        guard let tokenizer else {
            throw Qwen3SpeechNativeSessionError.transcriptionModeRequired
        }
        let promptTokens = try Self.makePromptTokens(
            audioTokenCount: audioTokenCount,
            language: canonicalLanguageName(language),
            tokenizer: tokenizer
        )
        return tokenizer.decode(tokens: promptTokens, skipSpecialTokens: false)
    }

    public func transcribe(
        _ request: Qwen3ASRTranscriptionRequest,
        onTextDelta: ((String) -> Void)? = nil
    ) throws -> Qwen3ASRTranscriptionResult {
        guard let tokenizer, let audioTokenID else {
            throw Qwen3SpeechNativeSessionError.transcriptionModeRequired
        }
        try session.resetDecodeCache()
        let encodingResult = try encode(
            audio: request.audio,
            maxAudioSeconds: request.maxAudioSeconds
        )
        let language = canonicalLanguageName(request.language)
        let promptTokens = try Self.makePromptTokens(
            audioTokenCount: encodingResult.encoding.shape[0],
            language: language,
            tokenizer: tokenizer
        )
        guard promptTokens.filter({ $0 == audioTokenID }).count == encodingResult.encoding.shape[0] else {
            throw Qwen3SpeechNativeSessionError.audioFeatureMismatch
        }

        let prefillStartedAt = Date()
        var nextTokenID: Int? = try session.prefillMediaFeatures(
            tokenIDs: promptTokens,
            mediaFeatures: encodingResult.encoding.values,
            mediaFeatureShape: encodingResult.encoding.shape,
            mediaTokenID: audioTokenID
        )
        let prefillSeconds = Date().timeIntervalSince(prefillStartedAt)

        let endTokenIDs = Self.defaultEndTokenIDs(tokenizer: tokenizer)
        var generatedTokenIDs: [Int] = []
        generatedTokenIDs.reserveCapacity(request.maxTokens)
        var emittedText = ""
        let decodeStartedAt = Date()
        while generatedTokenIDs.count < request.maxTokens, let tokenID = nextTokenID {
            if endTokenIDs.contains(tokenID) {
                break
            }
            generatedTokenIDs.append(tokenID)
            let decodedText = tokenizer.decode(tokens: generatedTokenIDs, skipSpecialTokens: true)
            let delta: String
            if decodedText.hasPrefix(emittedText) {
                let start = decodedText.index(decodedText.startIndex, offsetBy: emittedText.count)
                delta = String(decodedText[start...])
            } else {
                delta = decodedText
            }
            emittedText = decodedText
            if !delta.isEmpty {
                onTextDelta?(delta)
            }
            if generatedTokenIDs.count < request.maxTokens {
                nextTokenID = try session.decodeStep(tokenID: tokenID)
            } else {
                nextTokenID = nil
            }
        }
        let decodeSeconds = Date().timeIntervalSince(decodeStartedAt)

        return Qwen3ASRTranscriptionResult(
            text: emittedText,
            language: language,
            promptTokenCount: promptTokens.count,
            audioTokenCount: encodingResult.encoding.shape[0],
            inputFrameCount: encodingResult.inputFrameCount,
            generatedTokenIDs: generatedTokenIDs,
            prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds
        )
    }

    private func canonicalLanguageName(_ language: String) -> String {
        config.canonicalLanguageName(language)
    }

    private static func makePromptTokens(
        audioTokenCount: Int,
        language: String,
        tokenizer: Tokenizer
    ) throws -> [Int] {
        let audioPads = String(repeating: "<|audio_pad|>", count: audioTokenCount)
        let prompt = "<|im_start|>system\n<|im_end|>\n"
            + "<|im_start|>user\n<|audio_start|>"
            + audioPads
            + "<|audio_end|><|im_end|>\n"
            + "<|im_start|>assistant\nlanguage \(language)<asr_text>"
        let tokens = tokenizer.encode(text: prompt)
        guard !tokens.isEmpty else {
            throw Qwen3SpeechNativeSessionError.emptyTokenIDs
        }
        return tokens
    }

    private static func requiredTokenID(_ token: String, tokenizer: Tokenizer) throws -> Int {
        guard let id = tokenizer.convertTokenToId(token) else {
            throw Qwen3SpeechNativeSessionError.tokenizerMissingToken(token)
        }
        return id
    }

    private static func defaultEndTokenIDs(tokenizer: Tokenizer) -> Set<Int> {
        var ids = Set<Int>()
        if let eos = tokenizer.eosTokenId {
            ids.insert(eos)
        }
        if let imEnd = tokenizer.convertTokenToId("<|im_end|>") {
            ids.insert(imEnd)
        }
        if let endOfText = tokenizer.convertTokenToId("<|endoftext|>") {
            ids.insert(endOfText)
        }
        return ids
    }
}

private struct Qwen3TTSRawConfig: Decodable {
    var modelType: String
    var quantization: Qwen3SpeechQuantizationConfig?
    var ttsBOSTokenID: Int
    var ttsEOSTokenID: Int
    var ttsPADTokenID: Int
    var talkerConfig: Qwen3TTSTalkerConfig
    var supportedLanguages: [String]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case quantization
        case ttsBOSTokenID = "tts_bos_token_id"
        case ttsEOSTokenID = "tts_eos_token_id"
        case ttsPADTokenID = "tts_pad_token_id"
        case talkerConfig = "talker_config"
        case supportedLanguages = "support_languages"
    }

    init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        quantization = try container.decodeIfPresent(Qwen3SpeechQuantizationConfig.self, forKey: .quantization)
        ttsBOSTokenID = try container.decode(Int.self, forKey: .ttsBOSTokenID)
        ttsEOSTokenID = try container.decode(Int.self, forKey: .ttsEOSTokenID)
        ttsPADTokenID = try container.decode(Int.self, forKey: .ttsPADTokenID)
        talkerConfig = try container.decode(Qwen3TTSTalkerConfig.self, forKey: .talkerConfig)
        supportedLanguages = try container.decodeIfPresent([String].self, forKey: .supportedLanguages) ?? []
    }

    static func load(from modelURL: URL) throws -> Qwen3TTSRawConfig {
        let data = try Data(contentsOf: modelURL.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Qwen3TTSRawConfig.self, from: data)
        guard config.modelType == "qwen3_tts" else {
            throw Qwen3SpeechNativeSessionError.invalidTTSModelConfiguration
        }
        return config
    }

    var quantizationOrDefault: Qwen3SpeechQuantizationConfig {
        quantization ?? Qwen3SpeechQuantizationConfig(groupSize: 64, bits: 4)
    }

    func makeTalkerArchitecture() throws -> QwenHybridArchitecture {
        try talkerConfig.makeArchitecture(
            vocabularySize: talkerConfig.vocabSize,
            hiddenSize: talkerConfig.hiddenSize,
            intermediateSize: talkerConfig.intermediateSize,
            layerCount: talkerConfig.numHiddenLayers,
            attentionHeadCount: talkerConfig.numAttentionHeads,
            keyValueHeadCount: talkerConfig.numKeyValueHeads,
            headDimension: talkerConfig.headDim,
            contextLength: talkerConfig.maxPositionEmbeddings,
            rmsNormEpsilon: talkerConfig.rmsNormEps,
            ropeTheta: talkerConfig.ropeTheta,
            quantization: quantizationOrDefault.profile
        )
    }

    func makeCodePredictorArchitecture() throws -> QwenHybridArchitecture {
        let codePredictor = talkerConfig.codePredictorConfig
        return try talkerConfig.makeArchitecture(
            vocabularySize: codePredictor.vocabSize,
            hiddenSize: codePredictor.hiddenSize,
            intermediateSize: codePredictor.intermediateSize,
            layerCount: codePredictor.numHiddenLayers,
            attentionHeadCount: codePredictor.numAttentionHeads,
            keyValueHeadCount: codePredictor.numKeyValueHeads,
            headDimension: codePredictor.headDim,
            contextLength: codePredictor.maxPositionEmbeddings,
            rmsNormEpsilon: codePredictor.rmsNormEps,
            ropeTheta: codePredictor.ropeTheta,
            quantization: quantizationOrDefault.profile
        )
    }

    func speakerID(_ requested: String?) throws -> Int? {
        guard !talkerConfig.spkID.isEmpty else { return nil }
        let speaker = requested?.lowercased()
            ?? (talkerConfig.spkID["serena"] == nil ? talkerConfig.spkID.keys.sorted().first : "serena")
        guard let speaker, let id = talkerConfig.spkID[speaker] else {
            throw Qwen3SpeechNativeSessionError.missingSpeaker(requested ?? "default")
        }
        return id
    }

    func codecPrefix(language: String) throws -> [Int] {
        let languageName = language.lowercased()
        if languageName == "auto" {
            return [
                talkerConfig.codecNothinkID,
                talkerConfig.codecThinkBosID,
                talkerConfig.codecThinkEosID,
            ]
        }
        guard let languageID = talkerConfig.codecLanguageID[languageName] else {
            throw Qwen3SpeechNativeSessionError.missingLanguage(language)
        }
        return [
            talkerConfig.codecThinkID,
            talkerConfig.codecThinkBosID,
            languageID,
            talkerConfig.codecThinkEosID,
        ]
    }
}

private struct Qwen3TTSSpeechTokenizerRawConfig: Decodable {
    var modelType: String
    var outputSampleRate: Int
    var decodeUpsampleRate: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case outputSampleRate = "output_sample_rate"
        case decodeUpsampleRate = "decode_upsample_rate"
    }

    static func load(from modelURL: URL) throws -> Qwen3TTSSpeechTokenizerRawConfig {
        let configURL = modelURL
            .appendingPathComponent("speech_tokenizer")
            .appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw Qwen3SpeechNativeSessionError.missingSpeechTokenizer
        }
        let data = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(Qwen3TTSSpeechTokenizerRawConfig.self, from: data)
        guard config.modelType == "qwen3_tts_tokenizer_12hz" else {
            throw Qwen3SpeechNativeSessionError.invalidTTSModelConfiguration
        }
        return config
    }
}

private struct Qwen3SpeechQuantizationConfig: Decodable {
    var groupSize: Int
    var bits: Int

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
    }

    var profile: QwenQuantizationProfile {
        QwenQuantizationProfile(groupSize: groupSize, bits: bits)
    }
}

private struct Qwen3TTSTalkerConfig: Decodable {
    var vocabSize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var headDim: Int
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var codecEOSTokenID: Int
    var codecThinkID: Int
    var codecNothinkID: Int
    var codecThinkBosID: Int
    var codecThinkEosID: Int
    var codecPadID: Int
    var codecBosID: Int
    var codecLanguageID: [String: Int]
    var spkID: [String: Int]
    var codePredictorConfig: Qwen3TTSCodePredictorConfig

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case codecEOSTokenID = "codec_eos_token_id"
        case codecThinkID = "codec_think_id"
        case codecNothinkID = "codec_nothink_id"
        case codecThinkBosID = "codec_think_bos_id"
        case codecThinkEosID = "codec_think_eos_id"
        case codecPadID = "codec_pad_id"
        case codecBosID = "codec_bos_id"
        case codecLanguageID = "codec_language_id"
        case spkID = "spk_id"
        case codePredictorConfig = "code_predictor_config"
    }

    func makeArchitecture(
        vocabularySize: Int,
        hiddenSize: Int,
        intermediateSize: Int,
        layerCount: Int,
        attentionHeadCount: Int,
        keyValueHeadCount: Int,
        headDimension: Int,
        contextLength: Int,
        rmsNormEpsilon: Float,
        ropeTheta: Float,
        quantization: QwenQuantizationProfile
    ) throws -> QwenHybridArchitecture {
        try QwenHybridArchitecture(
            family: .qwen35,
            vocabularySize: vocabularySize,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            attentionHeadCount: attentionHeadCount,
            keyValueHeadCount: keyValueHeadCount,
            headDimension: headDimension,
            contextLength: contextLength,
            rmsNormEpsilon: rmsNormEpsilon,
            ropeTheta: ropeTheta,
            partialRotaryFactor: 1.0,
            quantization: quantization,
            layerKinds: Array(repeating: .fullAttention, count: layerCount),
            allowFullAttentionOnly: true
        )
    }
}

private struct Qwen3TTSCodePredictorConfig: Decodable {
    var vocabSize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var headDim: Int
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeTheta: Float

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
    }
}

private struct Qwen3ASRRawConfig: Decodable {
    var modelType: String
    var quantization: Qwen3SpeechQuantizationConfig
    var thinkerConfig: Qwen3ASRThinkerConfig
    var supportLanguages: [String]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case quantization
        case thinkerConfig = "thinker_config"
        case supportLanguages = "support_languages"
    }

    static func load(from modelURL: URL) throws -> Qwen3ASRRawConfig {
        let data = try Data(contentsOf: modelURL.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Qwen3ASRRawConfig.self, from: data)
        guard config.modelType == "qwen3_asr" else {
            throw Qwen3SpeechNativeSessionError.invalidASRModelConfiguration
        }
        return config
    }

    func makeAudioTowerConfiguration() -> EdgeMLXQwen35AudioTowerConfiguration {
        let audio = thinkerConfig.audioConfig
        return EdgeMLXQwen35AudioTowerConfiguration(
            numMelBins: audio.numMelBins,
            encoderLayers: audio.encoderLayers,
            encoderAttentionHeads: audio.encoderAttentionHeads,
            encoderFFNDim: audio.encoderFFNDim,
            dModel: audio.dModel,
            maxSourcePositions: audio.maxSourcePositions,
            nWindow: audio.nWindow,
            outputDim: audio.outputDim,
            nWindowInfer: audio.nWindowInfer,
            downsampleHiddenSize: audio.downsampleHiddenSize
        )
    }

    func makeEncoderSmokeArchitecture() throws -> QwenHybridArchitecture {
        let text = thinkerConfig.textConfig
        return try QwenHybridArchitecture(
            family: .qwen35,
            vocabularySize: text.vocabSize,
            hiddenSize: text.hiddenSize,
            intermediateSize: text.intermediateSize,
            attentionHeadCount: text.numAttentionHeads,
            keyValueHeadCount: text.numKeyValueHeads,
            headDimension: text.headDim,
            contextLength: text.maxPositionEmbeddings,
            rmsNormEpsilon: text.rmsNormEps,
            ropeTheta: text.ropeTheta,
            quantization: nil,
            layerKinds: [.fullAttention, .gdn]
        )
    }

    func makeTextArchitecture() throws -> QwenHybridArchitecture {
        let text = thinkerConfig.textConfig
        return try QwenHybridArchitecture(
            family: .qwen35,
            vocabularySize: text.vocabSize,
            hiddenSize: text.hiddenSize,
            intermediateSize: text.intermediateSize,
            attentionHeadCount: text.numAttentionHeads,
            keyValueHeadCount: text.numKeyValueHeads,
            headDimension: text.headDim,
            contextLength: text.maxPositionEmbeddings,
            rmsNormEpsilon: text.rmsNormEps,
            ropeTheta: text.ropeTheta,
            partialRotaryFactor: 1.0,
            quantization: quantization.profile,
            layerKinds: Array(repeating: .fullAttention, count: text.numHiddenLayers),
            allowFullAttentionOnly: true
        )
    }

    func canonicalLanguageName(_ language: String) -> String {
        let lower = Dictionary(uniqueKeysWithValues: supportLanguages.map { ($0.lowercased(), $0) })
        return lower[language.lowercased()] ?? language
    }
}

private struct Qwen3ASRThinkerConfig: Decodable {
    var audioConfig: Qwen3ASRAudioConfig
    var textConfig: Qwen3ASRTextConfig

    enum CodingKeys: String, CodingKey {
        case audioConfig = "audio_config"
        case textConfig = "text_config"
    }
}

private struct Qwen3ASRAudioConfig: Decodable {
    var numMelBins: Int
    var encoderLayers: Int
    var encoderAttentionHeads: Int
    var encoderFFNDim: Int
    var dModel: Int
    var maxSourcePositions: Int
    var nWindow: Int
    var outputDim: Int
    var nWindowInfer: Int
    var downsampleHiddenSize: Int

    enum CodingKeys: String, CodingKey {
        case numMelBins = "num_mel_bins"
        case encoderLayers = "encoder_layers"
        case encoderAttentionHeads = "encoder_attention_heads"
        case encoderFFNDim = "encoder_ffn_dim"
        case dModel = "d_model"
        case maxSourcePositions = "max_source_positions"
        case nWindow = "n_window"
        case outputDim = "output_dim"
        case nWindowInfer = "n_window_infer"
        case downsampleHiddenSize = "downsample_hidden_size"
    }
}

private struct Qwen3ASRTextConfig: Decodable {
    var vocabSize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var headDim: Int
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeTheta: Float

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
    }
}

private struct Qwen3SpeechSafetensors {
    private struct SafeTensorsIndex: Decodable {
        var weightMap: [String: String]

        enum CodingKeys: String, CodingKey {
            case weightMap = "weight_map"
        }
    }

    static func shardURLs(modelURL: URL, include: (String) -> Bool) throws -> [URL] {
        let indexURL = modelURL.appendingPathComponent("model.safetensors.index.json")
        if let data = try? Data(contentsOf: indexURL),
           let index = try? JSONDecoder().decode(SafeTensorsIndex.self, from: data) {
            let fileNames = Set(index.weightMap.compactMap { key, value in
                include(key) ? value : nil
            })
            let urls = fileNames.sorted().map { modelURL.appendingPathComponent($0) }
            if !urls.isEmpty {
                return urls
            }
        }
        let single = modelURL.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: single.path) {
            return [single]
        }
        throw Qwen3SpeechNativeSessionError.missingSafeTensors
    }
}

private struct Qwen3TokenizerLoader {
    static func loadPatchedTokenizer(
        modelURL: URL,
        specialTokens: [(id: Int, content: String)]
    ) async throws -> Tokenizer {
        let tokenizerConfigURL = modelURL.appendingPathComponent("tokenizer_config.json")
        guard FileManager.default.fileExists(atPath: tokenizerConfigURL.path) else {
            throw Qwen3SpeechNativeSessionError.missingTokenizerConfig
        }
        let tokenizerData = try patchedTokenizerData(
            tokenizerURL: modelURL.appendingPathComponent("tokenizer.json"),
            specialTokens: specialTokens
        )
        let tempRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("edge-engine-tokenizer-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: tempRoot)
        }
        try tokenizerData.write(to: tempRoot.appendingPathComponent("tokenizer.json"))
        try FileManager.default.copyItem(
            at: tokenizerConfigURL,
            to: tempRoot.appendingPathComponent("tokenizer_config.json")
        )
        for optionalName in ["chat_template.jinja", "chat_template.json", "config.json"] {
            let source = modelURL.appendingPathComponent(optionalName)
            guard FileManager.default.fileExists(atPath: source.path) else { continue }
            try FileManager.default.copyItem(
                at: source,
                to: tempRoot.appendingPathComponent(optionalName)
            )
        }
        return try await AutoTokenizer.from(modelFolder: tempRoot, strict: false)
    }

    private static func patchedTokenizerData(
        tokenizerURL: URL,
        specialTokens: [(id: Int, content: String)]
    ) throws -> Data {
        let data = try Data(contentsOf: tokenizerURL)
        guard var root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw Qwen3SpeechNativeSessionError.invalidTokenizerJSON
        }
        let existingTokens = (root["added_tokens"] as? [[String: Any]]) ?? []
        let existingIDs = Set(existingTokens.compactMap { $0["id"] as? Int })
        var newTokens = existingTokens
        for (id, content) in specialTokens where !existingIDs.contains(id) {
            newTokens.append([
                "id": id,
                "content": content,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            ])
        }
        guard newTokens.count != existingTokens.count else { return data }
        newTokens.sort { ($0["id"] as? Int ?? 0) < ($1["id"] as? Int ?? 0) }
        root["added_tokens"] = newTokens
        return try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
    }
}

private let qwen3TTSSpecialTokens: [(id: Int, content: String)] = [
    (151643, "<|endoftext|>"), (151644, "<|im_start|>"), (151645, "<|im_end|>"),
    (151671, "<tts_pad>"), (151672, "<tts_text_bos>"), (151673, "<tts_text_eod>"),
]

private let qwen3ASRSpecialTokens: [(id: Int, content: String)] = [
    (151643, "<|endoftext|>"), (151644, "<|im_start|>"), (151645, "<|im_end|>"),
    (151646, "<|object_ref_start|>"), (151647, "<|object_ref_end|>"),
    (151648, "<|box_start|>"), (151649, "<|box_end|>"),
    (151650, "<|quad_start|>"), (151651, "<|quad_end|>"),
    (151652, "<|vision_start|>"), (151653, "<|vision_end|>"),
    (151654, "<|vision_pad|>"), (151655, "<|image_pad|>"), (151656, "<|video_pad|>"),
    (151657, "<tool_call>"), (151658, "</tool_call>"),
    (151659, "<|fim_prefix|>"), (151660, "<|fim_middle|>"),
    (151661, "<|fim_suffix|>"), (151662, "<|fim_pad|>"),
    (151663, "<|repo_name|>"), (151664, "<|file_sep|>"),
    (151665, "<tool_response>"), (151666, "</tool_response>"),
    (151667, "<think>"), (151668, "</think>"),
    (151669, "<|audio_start|>"), (151670, "<|audio_end|>"),
    (151671, "<tts_pad>"), (151672, "<tts_text_bos>"), (151673, "<tts_text_eod>"),
    (151674, "<tts_text_bos_single>"), (151675, "<non_speech>"),
    (151676, "<|audio_pad|>"),
    (151677, "<blank1>"), (151678, "<blank2>"), (151679, "<blank3>"),
    (151680, "<blank4>"), (151681, "<blank5>"), (151682, "<blank6>"),
    (151683, "<blank7>"), (151684, "<blank8>"), (151685, "<blank9>"),
    (151686, "<blank10>"), (151687, "<blank11>"), (151688, "<blank12>"),
    (151689, "<blank13>"), (151690, "<blank14>"), (151691, "<blank15>"),
    (151692, "<blank16>"), (151693, "<blank17>"), (151694, "<blank18>"),
    (151695, "<blank19>"), (151696, "<blank20>"), (151697, "<blank21>"),
    (151698, "<blank22>"), (151699, "<blank23>"), (151700, "<blank24>"),
    (151701, "<blank25>"), (151702, "<blank26>"), (151703, "<blank27>"),
    (151704, "<asr_text>"),
]

private extension EdgeAudioBuffer {
    func truncated(to maxSeconds: Double?) throws -> EdgeAudioBuffer {
        guard let maxSeconds else { return self }
        let maxFrames = max(1, Int((maxSeconds * Double(sampleRate)).rounded()))
        guard frameCount > maxFrames else { return self }
        return try EdgeAudioBuffer(
            sampleRate: sampleRate,
            channelCount: channelCount,
            interleavedSamples: Array(interleavedSamples.prefix(maxFrames * channelCount))
        )
    }
}
