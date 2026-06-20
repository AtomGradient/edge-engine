// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum EdgeSpeechBundlePreflightError: Error, Equatable {
    case missingRequiredFile(String)
    case unsupportedModelType(String)
    case modelFamilyMismatch(expected: EdgeSpeechModelFamily, actual: EdgeSpeechModelFamily)
    case invalidASRFeatureConfiguration
}

public enum EdgeSpeechBundlePreflightFailureReason: String, Codable, Equatable, Sendable {
    case missingRequiredResources = "missing_required_resources"
    case sampleRateMismatch = "sample_rate_mismatch"
    case tokenizerSpecialTokenMismatch = "tokenizer_special_token_mismatch"
}

public struct EdgeSpeechBundlePreflightConfiguration {
    public var modelRootURL: URL
    public var modelFamily: EdgeSpeechModelFamily?

    public init(modelRootURL: URL, modelFamily: EdgeSpeechModelFamily? = nil) {
        self.modelRootURL = modelRootURL
        self.modelFamily = modelFamily
    }
}

public struct EdgeSpeechSampleRateCheck: Codable, Equatable, Sendable {
    public var source: String
    public var expectedSampleRate: Int
    public var actualSampleRate: Int

    public init(source: String, expectedSampleRate: Int, actualSampleRate: Int) {
        self.source = source
        self.expectedSampleRate = expectedSampleRate
        self.actualSampleRate = actualSampleRate
    }

    public var matches: Bool {
        expectedSampleRate == actualSampleRate
    }
}

public struct EdgeSpeechTokenizerSpecialTokenCheck: Codable, Equatable, Sendable {
    public var name: String
    public var tokenID: Int
    public var expectedContent: String
    public var actualContent: String?
    public var isSpecial: Bool

    public init(
        name: String,
        tokenID: Int,
        expectedContent: String,
        actualContent: String?,
        isSpecial: Bool
    ) {
        self.name = name
        self.tokenID = tokenID
        self.expectedContent = expectedContent
        self.actualContent = actualContent
        self.isSpecial = isSpecial
    }

    public var matches: Bool {
        actualContent == expectedContent && isSpecial
    }
}

public struct EdgeSpeechBundlePreflightResult: Codable, Equatable, Sendable {
    public var modelRootPath: String
    public var modelType: String
    public var plan: EdgeSpeechRuntimePlan
    public var requiredResourceNames: [String]
    public var missingRequiredResourceNames: [String]
    public var sampleRateChecks: [EdgeSpeechSampleRateCheck]
    public var tokenizerSpecialTokenChecks: [EdgeSpeechTokenizerSpecialTokenCheck]
    public var passesPreflight: Bool
    public var failureReasons: [EdgeSpeechBundlePreflightFailureReason]
    public var asrFeatureConfiguration: EdgeLogMelSpectrogramConfiguration?
    public var supportedLanguages: [String]
    public var audioTokenID: Int?
    public var ttsModelType: String?
    public var ttsModelSize: String?
    public var speechTokenizerModelType: String?
    public var speechTokenizerInputSampleRate: Int?
    public var speechTokenizerOutputSampleRate: Int?
    public var speechTokenizerCodecSamplesPerFrame: Int?
    public var speechTokenizerCodecFrameRate: Double?
    public var speechTokenizerEncoderValidNumQuantizers: Int?

    public init(
        modelRootPath: String,
        modelType: String,
        plan: EdgeSpeechRuntimePlan,
        requiredResourceNames: [String],
        missingRequiredResourceNames: [String],
        sampleRateChecks: [EdgeSpeechSampleRateCheck],
        tokenizerSpecialTokenChecks: [EdgeSpeechTokenizerSpecialTokenCheck] = [],
        passesPreflight: Bool? = nil,
        failureReasons: [EdgeSpeechBundlePreflightFailureReason]? = nil,
        asrFeatureConfiguration: EdgeLogMelSpectrogramConfiguration? = nil,
        supportedLanguages: [String] = [],
        audioTokenID: Int? = nil,
        ttsModelType: String? = nil,
        ttsModelSize: String? = nil,
        speechTokenizerModelType: String? = nil,
        speechTokenizerInputSampleRate: Int? = nil,
        speechTokenizerOutputSampleRate: Int? = nil,
        speechTokenizerCodecSamplesPerFrame: Int? = nil,
        speechTokenizerCodecFrameRate: Double? = nil,
        speechTokenizerEncoderValidNumQuantizers: Int? = nil
    ) {
        self.modelRootPath = modelRootPath
        self.modelType = modelType
        self.plan = plan
        self.requiredResourceNames = requiredResourceNames
        self.missingRequiredResourceNames = missingRequiredResourceNames
        self.sampleRateChecks = sampleRateChecks
        self.tokenizerSpecialTokenChecks = tokenizerSpecialTokenChecks
        let resolvedFailureReasons = failureReasons ?? Self.makeFailureReasons(
            missingRequiredResourceNames: missingRequiredResourceNames,
            sampleRateChecks: sampleRateChecks,
            tokenizerSpecialTokenChecks: tokenizerSpecialTokenChecks
        )
        self.failureReasons = resolvedFailureReasons
        self.passesPreflight = passesPreflight ?? resolvedFailureReasons.isEmpty
        self.asrFeatureConfiguration = asrFeatureConfiguration
        self.supportedLanguages = supportedLanguages
        self.audioTokenID = audioTokenID
        self.ttsModelType = ttsModelType
        self.ttsModelSize = ttsModelSize
        self.speechTokenizerModelType = speechTokenizerModelType
        self.speechTokenizerInputSampleRate = speechTokenizerInputSampleRate
        self.speechTokenizerOutputSampleRate = speechTokenizerOutputSampleRate
        self.speechTokenizerCodecSamplesPerFrame = speechTokenizerCodecSamplesPerFrame
        self.speechTokenizerCodecFrameRate = speechTokenizerCodecFrameRate
        self.speechTokenizerEncoderValidNumQuantizers = speechTokenizerEncoderValidNumQuantizers
    }

    private enum CodingKeys: String, CodingKey {
        case modelRootPath
        case modelType
        case plan
        case requiredResourceNames
        case missingRequiredResourceNames
        case sampleRateChecks
        case tokenizerSpecialTokenChecks
        case passesPreflight
        case failureReasons
        case asrFeatureConfiguration
        case supportedLanguages
        case audioTokenID
        case ttsModelType
        case ttsModelSize
        case speechTokenizerModelType
        case speechTokenizerInputSampleRate
        case speechTokenizerOutputSampleRate
        case speechTokenizerCodecSamplesPerFrame
        case speechTokenizerCodecFrameRate
        case speechTokenizerEncoderValidNumQuantizers
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelRootPath = try container.decode(String.self, forKey: .modelRootPath)
        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.plan = try container.decode(EdgeSpeechRuntimePlan.self, forKey: .plan)
        self.requiredResourceNames = try container.decode([String].self, forKey: .requiredResourceNames)
        self.missingRequiredResourceNames = try container.decode(
            [String].self,
            forKey: .missingRequiredResourceNames
        )
        self.sampleRateChecks = try container.decode(
            [EdgeSpeechSampleRateCheck].self,
            forKey: .sampleRateChecks
        )
        self.tokenizerSpecialTokenChecks = try container.decode(
            [EdgeSpeechTokenizerSpecialTokenCheck].self,
            forKey: .tokenizerSpecialTokenChecks
        )
        self.asrFeatureConfiguration = try container.decodeIfPresent(
            EdgeLogMelSpectrogramConfiguration.self,
            forKey: .asrFeatureConfiguration
        )
        self.supportedLanguages = try container.decode([String].self, forKey: .supportedLanguages)
        self.audioTokenID = try container.decodeIfPresent(Int.self, forKey: .audioTokenID)
        self.ttsModelType = try container.decodeIfPresent(String.self, forKey: .ttsModelType)
        self.ttsModelSize = try container.decodeIfPresent(String.self, forKey: .ttsModelSize)
        self.speechTokenizerModelType = try container.decodeIfPresent(
            String.self,
            forKey: .speechTokenizerModelType
        )
        self.speechTokenizerInputSampleRate = try container.decodeIfPresent(
            Int.self,
            forKey: .speechTokenizerInputSampleRate
        )
        self.speechTokenizerOutputSampleRate = try container.decodeIfPresent(
            Int.self,
            forKey: .speechTokenizerOutputSampleRate
        )
        self.speechTokenizerCodecSamplesPerFrame = try container.decodeIfPresent(
            Int.self,
            forKey: .speechTokenizerCodecSamplesPerFrame
        )
        self.speechTokenizerCodecFrameRate = try container.decodeIfPresent(
            Double.self,
            forKey: .speechTokenizerCodecFrameRate
        )
        self.speechTokenizerEncoderValidNumQuantizers = try container.decodeIfPresent(
            Int.self,
            forKey: .speechTokenizerEncoderValidNumQuantizers
        )

        let defaultReasons = Self.makeFailureReasons(
            missingRequiredResourceNames: missingRequiredResourceNames,
            sampleRateChecks: sampleRateChecks,
            tokenizerSpecialTokenChecks: tokenizerSpecialTokenChecks
        )
        self.failureReasons = try container.decodeIfPresent(
            [EdgeSpeechBundlePreflightFailureReason].self,
            forKey: .failureReasons
        ) ?? defaultReasons
        self.passesPreflight = try container.decodeIfPresent(
            Bool.self,
            forKey: .passesPreflight
        ) ?? failureReasons.isEmpty
    }

    public static func makeFailureReasons(
        missingRequiredResourceNames: [String],
        sampleRateChecks: [EdgeSpeechSampleRateCheck],
        tokenizerSpecialTokenChecks: [EdgeSpeechTokenizerSpecialTokenCheck]
    ) -> [EdgeSpeechBundlePreflightFailureReason] {
        var reasons: [EdgeSpeechBundlePreflightFailureReason] = []
        if !missingRequiredResourceNames.isEmpty {
            reasons.append(.missingRequiredResources)
        }
        if sampleRateChecks.contains(where: { !$0.matches }) {
            reasons.append(.sampleRateMismatch)
        }
        if tokenizerSpecialTokenChecks.contains(where: { !$0.matches }) {
            reasons.append(.tokenizerSpecialTokenMismatch)
        }
        return reasons
    }
}

public enum EdgeSpeechBundlePreflightRunner {
    public static func run(
        configuration: EdgeSpeechBundlePreflightConfiguration
    ) throws -> EdgeSpeechBundlePreflightResult {
        let root = configuration.modelRootURL
        let rootConfig = try loadJSON(
            RawSpeechRootConfig.self,
            from: root.appendingPathComponent("config.json"),
            resourceName: "config.json"
        )
        let actualFamily = try decodeFamily(modelType: rootConfig.modelType)
        if let expectedFamily = configuration.modelFamily, expectedFamily != actualFamily {
            throw EdgeSpeechBundlePreflightError.modelFamilyMismatch(
                expected: expectedFamily,
                actual: actualFamily
            )
        }

        switch actualFamily {
        case .qwen3ASR:
            return try runASR(root: root, rootConfig: rootConfig)
        case .qwen3TTS:
            return try runTTS(root: root, rootConfig: rootConfig)
        }
    }

    private static func runASR(
        root: URL,
        rootConfig: RawSpeechRootConfig
    ) throws -> EdgeSpeechBundlePreflightResult {
        let preprocessor = try loadJSON(
            RawASRPreprocessorConfig.self,
            from: root.appendingPathComponent("preprocessor_config.json"),
            resourceName: "preprocessor_config.json"
        )
        let sampleRate = preprocessor.sampleRate ?? inferredSampleRate(
            nSamples: preprocessor.nSamples,
            chunkLength: preprocessor.chunkLength
        )
        guard let sampleRate else {
            throw EdgeSpeechBundlePreflightError.invalidASRFeatureConfiguration
        }
        let featureConfiguration = try EdgeLogMelSpectrogramConfiguration(
            sampleRate: sampleRate,
            fftSize: preprocessor.fftSize ?? 400,
            hopLength: preprocessor.hopLength ?? 160,
            melBinCount: preprocessor.featureSize ?? 128,
            minimumFrequency: 0,
            maximumFrequency: Float(sampleRate) / 2
        )
        let plan = try EdgeSpeechRuntimePlan.qwen3ASR
        let requiredResources = [
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            rootModelWeightsResourceName,
        ]
        let missingResources = missing(resources: requiredResources, under: root)
        let checks = [
            EdgeSpeechSampleRateCheck(
                source: "preprocessor_config.json",
                expectedSampleRate: plan.preferredSampleRate,
                actualSampleRate: featureConfiguration.sampleRate
            )
        ]
        let specialTokenChecks = try asrSpecialTokenChecks(
            root: root,
            thinkerConfig: rootConfig.thinkerConfig
        )

        return EdgeSpeechBundlePreflightResult(
            modelRootPath: root.path,
            modelType: rootConfig.modelType,
            plan: plan,
            requiredResourceNames: requiredResources,
            missingRequiredResourceNames: missingResources,
            sampleRateChecks: checks,
            tokenizerSpecialTokenChecks: specialTokenChecks,
            asrFeatureConfiguration: featureConfiguration,
            supportedLanguages: rootConfig.supportedLanguages,
            audioTokenID: rootConfig.thinkerConfig?.audioTokenID
        )
    }

    private static func runTTS(
        root: URL,
        rootConfig: RawSpeechRootConfig
    ) throws -> EdgeSpeechBundlePreflightResult {
        let tokenizerRoot = root.appendingPathComponent("speech_tokenizer")
        let tokenizerConfig = try loadJSON(
            RawTTSSpeechTokenizerConfig.self,
            from: tokenizerRoot.appendingPathComponent("config.json"),
            resourceName: "speech_tokenizer/config.json"
        )
        let tokenizerPreprocessor = try loadJSON(
            RawTTSSpeechTokenizerPreprocessorConfig.self,
            from: tokenizerRoot.appendingPathComponent("preprocessor_config.json"),
            resourceName: "speech_tokenizer/preprocessor_config.json"
        )
        let plan = try EdgeSpeechRuntimePlan.qwen3TTS
        let requiredResources = [
            "config.json",
            "tokenizer.json",
            rootModelWeightsResourceName,
            "speech_tokenizer/config.json",
            "speech_tokenizer/preprocessor_config.json",
            "speech_tokenizer/\(rootModelWeightsResourceName)",
        ]
        let missingResources = missing(resources: requiredResources, under: root)
        let checks = [
            EdgeSpeechSampleRateCheck(
                source: "speech_tokenizer/config.json:input_sample_rate",
                expectedSampleRate: plan.preferredSampleRate,
                actualSampleRate: tokenizerConfig.inputSampleRate
            ),
            EdgeSpeechSampleRateCheck(
                source: "speech_tokenizer/config.json:output_sample_rate",
                expectedSampleRate: plan.preferredSampleRate,
                actualSampleRate: tokenizerConfig.outputSampleRate
            ),
            EdgeSpeechSampleRateCheck(
                source: "speech_tokenizer/preprocessor_config.json:sampling_rate",
                expectedSampleRate: plan.preferredSampleRate,
                actualSampleRate: tokenizerPreprocessor.samplingRate
            ),
        ]

        return EdgeSpeechBundlePreflightResult(
            modelRootPath: root.path,
            modelType: rootConfig.modelType,
            plan: plan,
            requiredResourceNames: requiredResources,
            missingRequiredResourceNames: missingResources,
            sampleRateChecks: checks,
            supportedLanguages: rootConfig.supportedLanguages,
            ttsModelType: rootConfig.ttsModelType,
            ttsModelSize: rootConfig.ttsModelSize,
            speechTokenizerModelType: tokenizerConfig.modelType,
            speechTokenizerInputSampleRate: tokenizerConfig.inputSampleRate,
            speechTokenizerOutputSampleRate: tokenizerConfig.outputSampleRate,
            speechTokenizerCodecSamplesPerFrame: tokenizerConfig.decodeUpsampleRate,
            speechTokenizerCodecFrameRate: codecFrameRate(
                outputSampleRate: tokenizerConfig.outputSampleRate,
                decodeUpsampleRate: tokenizerConfig.decodeUpsampleRate
            ),
            speechTokenizerEncoderValidNumQuantizers: tokenizerConfig.encoderValidNumQuantizers
        )
    }

    private static let rootModelWeightsResourceName = "model.safetensors or model.safetensors.index.json"

    private static func decodeFamily(modelType: String) throws -> EdgeSpeechModelFamily {
        switch normalized(modelType) {
        case "qwen3asr":
            return .qwen3ASR
        case "qwen3tts":
            return .qwen3TTS
        default:
            throw EdgeSpeechBundlePreflightError.unsupportedModelType(modelType)
        }
    }

    private static func missing(resources: [String], under root: URL) -> [String] {
        resources.filter { resource in
            if resource == rootModelWeightsResourceName {
                return !hasModelWeights(at: root)
            }
            if resource == "speech_tokenizer/\(rootModelWeightsResourceName)" {
                return !hasModelWeights(at: root.appendingPathComponent("speech_tokenizer"))
            }
            return !FileManager.default.fileExists(atPath: root.appendingPathComponent(resource).path)
        }
    }

    private static func hasModelWeights(at root: URL) -> Bool {
        FileManager.default.fileExists(atPath: root.appendingPathComponent("model.safetensors").path)
            || FileManager.default.fileExists(
                atPath: root.appendingPathComponent("model.safetensors.index.json").path
            )
    }

    private static func asrSpecialTokenChecks(
        root: URL,
        thinkerConfig: RawASRThinkerConfig?
    ) throws -> [EdgeSpeechTokenizerSpecialTokenCheck] {
        let checks = [
            (name: "audio_start_token_id", tokenID: thinkerConfig?.audioStartTokenID, expectedContent: "<|audio_start|>"),
            (name: "audio_end_token_id", tokenID: thinkerConfig?.audioEndTokenID, expectedContent: "<|audio_end|>"),
            (name: "audio_token_id", tokenID: thinkerConfig?.audioTokenID, expectedContent: "<|audio_pad|>"),
        ].compactMap { item -> (name: String, tokenID: Int, expectedContent: String)? in
            guard let tokenID = item.tokenID else { return nil }
            return (item.name, tokenID, item.expectedContent)
        }
        guard !checks.isEmpty else { return [] }

        let tokenizer = try loadJSON(
            RawTokenizerConfig.self,
            from: root.appendingPathComponent("tokenizer.json"),
            resourceName: "tokenizer.json"
        )
        let addedTokensByID = Dictionary(
            uniqueKeysWithValues: tokenizer.addedTokens.map { token in
                (token.id, token)
            }
        )
        return checks.map { check in
            let token = addedTokensByID[check.tokenID]
            return EdgeSpeechTokenizerSpecialTokenCheck(
                name: check.name,
                tokenID: check.tokenID,
                expectedContent: check.expectedContent,
                actualContent: token?.content,
                isSpecial: token?.special ?? false
            )
        }
    }

    private static func loadJSON<T: Decodable>(
        _ type: T.Type,
        from url: URL,
        resourceName: String
    ) throws -> T {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw EdgeSpeechBundlePreflightError.missingRequiredFile(resourceName)
        }
        return try JSONDecoder().decode(type, from: Data(contentsOf: url))
    }

    private static func inferredSampleRate(nSamples: Int?, chunkLength: Double?) -> Int? {
        guard let nSamples, let chunkLength, chunkLength > 0 else { return nil }
        return Int((Double(nSamples) / chunkLength).rounded())
    }

    private static func codecFrameRate(outputSampleRate: Int, decodeUpsampleRate: Int?) -> Double? {
        guard let decodeUpsampleRate, decodeUpsampleRate > 0 else { return nil }
        return Double(outputSampleRate) / Double(decodeUpsampleRate)
    }

    private static func normalized(_ value: String) -> String {
        value
            .lowercased()
            .replacingOccurrences(of: "-", with: "")
            .replacingOccurrences(of: "_", with: "")
            .replacingOccurrences(of: ".", with: "")
    }
}

private struct RawSpeechRootConfig: Decodable {
    var modelType: String
    var supportedLanguages: [String]
    var thinkerConfig: RawASRThinkerConfig?
    var ttsModelType: String?
    var ttsModelSize: String?

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case supportedLanguages = "support_languages"
        case thinkerConfig = "thinker_config"
        case ttsModelType = "tts_model_type"
        case ttsModelSize = "tts_model_size"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        supportedLanguages = try container.decodeIfPresent([String].self, forKey: .supportedLanguages) ?? []
        thinkerConfig = try container.decodeIfPresent(RawASRThinkerConfig.self, forKey: .thinkerConfig)
        ttsModelType = try container.decodeIfPresent(String.self, forKey: .ttsModelType)
        ttsModelSize = try container.decodeIfPresent(String.self, forKey: .ttsModelSize)
    }
}

private struct RawASRThinkerConfig: Decodable {
    var audioTokenID: Int?
    var audioStartTokenID: Int?
    var audioEndTokenID: Int?

    private enum CodingKeys: String, CodingKey {
        case audioTokenID = "audio_token_id"
        case audioStartTokenID = "audio_start_token_id"
        case audioEndTokenID = "audio_end_token_id"
    }
}

private struct RawTokenizerConfig: Decodable {
    var addedTokens: [RawTokenizerAddedToken]

    private enum CodingKeys: String, CodingKey {
        case addedTokens = "added_tokens"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        addedTokens = try container.decodeIfPresent([RawTokenizerAddedToken].self, forKey: .addedTokens) ?? []
    }
}

private struct RawTokenizerAddedToken: Decodable {
    var id: Int
    var content: String
    var special: Bool
}

private struct RawASRPreprocessorConfig: Decodable {
    var sampleRate: Int?
    var featureSize: Int?
    var fftSize: Int?
    var hopLength: Int?
    var nSamples: Int?
    var chunkLength: Double?

    private enum CodingKeys: String, CodingKey {
        case sampleRate = "sampling_rate"
        case alternateSampleRate = "sample_rate"
        case featureSize = "feature_size"
        case fftSize = "n_fft"
        case hopLength = "hop_length"
        case nSamples = "n_samples"
        case chunkLength = "chunk_length"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        sampleRate = try container.decodeIfPresent(Int.self, forKey: .sampleRate)
            ?? container.decodeIfPresent(Int.self, forKey: .alternateSampleRate)
        featureSize = try container.decodeIfPresent(Int.self, forKey: .featureSize)
        fftSize = try container.decodeIfPresent(Int.self, forKey: .fftSize)
        hopLength = try container.decodeIfPresent(Int.self, forKey: .hopLength)
        nSamples = try container.decodeIfPresent(Int.self, forKey: .nSamples)
        chunkLength = try container.decodeIfPresent(Double.self, forKey: .chunkLength)
    }
}

private struct RawTTSSpeechTokenizerConfig: Decodable {
    var modelType: String
    var inputSampleRate: Int
    var outputSampleRate: Int
    var decodeUpsampleRate: Int?
    var encoderValidNumQuantizers: Int?

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case inputSampleRate = "input_sample_rate"
        case outputSampleRate = "output_sample_rate"
        case decodeUpsampleRate = "decode_upsample_rate"
        case encoderValidNumQuantizers = "encoder_valid_num_quantizers"
    }
}

private struct RawTTSSpeechTokenizerPreprocessorConfig: Decodable {
    var samplingRate: Int

    private enum CodingKeys: String, CodingKey {
        case samplingRate = "sampling_rate"
    }
}
