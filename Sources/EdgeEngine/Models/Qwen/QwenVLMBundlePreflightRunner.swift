// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum QwenVLMModelFamily: String, Codable, Equatable, Sendable {
    case qwen35VLM
    case qwen36VLM
}

public enum QwenVLMBundlePreflightError: Error, Equatable {
    case missingRequiredFile(String)
    case unsupportedModelType(String)
    case modelFamilyMismatch(expected: QwenVLMModelFamily, actual: QwenVLMModelFamily)
    case invalidWeightIndex
    case invalidVisionConfiguration
}

public enum QwenVLMBundlePreflightFailureReason: String, Codable, Equatable, Sendable {
    case missingRequiredResources = "missing_required_resources"
    case missingLanguageModelTensors = "missing_language_model_tensors"
    case missingVisionTensors = "missing_vision_tensors"
    case tokenizerSpecialTokenMismatch = "tokenizer_special_token_mismatch"
}

public struct QwenVLMBundlePreflightConfiguration {
    public var modelRootURL: URL
    public var modelFamily: QwenVLMModelFamily?

    public init(modelRootURL: URL, modelFamily: QwenVLMModelFamily? = nil) {
        self.modelRootURL = modelRootURL
        self.modelFamily = modelFamily
    }
}

public struct QwenVisionConfiguration: Codable, Equatable, Sendable {
    public var hiddenSize: Int
    public var intermediateSize: Int?
    public var layerCount: Int
    public var attentionHeadCount: Int
    public var patchSize: Int?
    public var spatialMergeSize: Int?
    public var imageSize: Int?

    public init(
        hiddenSize: Int,
        intermediateSize: Int? = nil,
        layerCount: Int,
        attentionHeadCount: Int,
        patchSize: Int? = nil,
        spatialMergeSize: Int? = nil,
        imageSize: Int? = nil
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.layerCount = layerCount
        self.attentionHeadCount = attentionHeadCount
        self.patchSize = patchSize
        self.spatialMergeSize = spatialMergeSize
        self.imageSize = imageSize
    }
}

public struct QwenImageProcessorConfiguration: Codable, Equatable, Sendable {
    public var imageProcessorType: String?
    public var minPixels: Int?
    public var maxPixels: Int?
    public var patchSize: Int?
    public var mergeSize: Int?
    public var temporalPatchSize: Int?
    public var imageMean: [Float]
    public var imageStd: [Float]

    public init(
        imageProcessorType: String? = nil,
        minPixels: Int? = nil,
        maxPixels: Int? = nil,
        patchSize: Int? = nil,
        mergeSize: Int? = nil,
        temporalPatchSize: Int? = nil,
        imageMean: [Float] = [],
        imageStd: [Float] = []
    ) {
        self.imageProcessorType = imageProcessorType
        self.minPixels = minPixels
        self.maxPixels = maxPixels
        self.patchSize = patchSize
        self.mergeSize = mergeSize
        self.temporalPatchSize = temporalPatchSize
        self.imageMean = imageMean
        self.imageStd = imageStd
    }
}

public struct QwenVLMRuntimePlan: Codable, Equatable, Sendable {
    public var modelFamily: QwenVLMModelFamily
    public var languageArchitecture: QwenHybridArchitecture
    public var visionConfiguration: QwenVisionConfiguration
    public var imageProcessorConfiguration: QwenImageProcessorConfiguration
    public var supportsMultipleImages: Bool

    public init(
        modelFamily: QwenVLMModelFamily,
        languageArchitecture: QwenHybridArchitecture,
        visionConfiguration: QwenVisionConfiguration,
        imageProcessorConfiguration: QwenImageProcessorConfiguration,
        supportsMultipleImages: Bool = true
    ) {
        self.modelFamily = modelFamily
        self.languageArchitecture = languageArchitecture
        self.visionConfiguration = visionConfiguration
        self.imageProcessorConfiguration = imageProcessorConfiguration
        self.supportsMultipleImages = supportsMultipleImages
    }
}

public struct QwenVLMTokenizerSpecialTokenCheck: Codable, Equatable, Sendable {
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

public struct QwenVLMBundlePreflightResult: Codable, Equatable, Sendable {
    public var modelRootPath: String
    public var modelType: String
    public var plan: QwenVLMRuntimePlan
    public var requiredResourceNames: [String]
    public var missingRequiredResourceNames: [String]
    public var missingLanguageTensorNames: [String]
    public var visionTensorPrefixes: [String]
    public var tokenizerSpecialTokenChecks: [QwenVLMTokenizerSpecialTokenCheck]
    public var passesPreflight: Bool
    public var failureReasons: [QwenVLMBundlePreflightFailureReason]

    public init(
        modelRootPath: String,
        modelType: String,
        plan: QwenVLMRuntimePlan,
        requiredResourceNames: [String],
        missingRequiredResourceNames: [String],
        missingLanguageTensorNames: [String],
        visionTensorPrefixes: [String],
        tokenizerSpecialTokenChecks: [QwenVLMTokenizerSpecialTokenCheck],
        passesPreflight: Bool? = nil,
        failureReasons: [QwenVLMBundlePreflightFailureReason]? = nil
    ) {
        self.modelRootPath = modelRootPath
        self.modelType = modelType
        self.plan = plan
        self.requiredResourceNames = requiredResourceNames
        self.missingRequiredResourceNames = missingRequiredResourceNames
        self.missingLanguageTensorNames = missingLanguageTensorNames
        self.visionTensorPrefixes = visionTensorPrefixes
        self.tokenizerSpecialTokenChecks = tokenizerSpecialTokenChecks
        let resolvedFailureReasons = failureReasons ?? Self.makeFailureReasons(
            missingRequiredResourceNames: missingRequiredResourceNames,
            missingLanguageTensorNames: missingLanguageTensorNames,
            visionTensorPrefixes: visionTensorPrefixes,
            tokenizerSpecialTokenChecks: tokenizerSpecialTokenChecks
        )
        self.failureReasons = resolvedFailureReasons
        self.passesPreflight = passesPreflight ?? resolvedFailureReasons.isEmpty
    }

    public static func makeFailureReasons(
        missingRequiredResourceNames: [String],
        missingLanguageTensorNames: [String],
        visionTensorPrefixes: [String],
        tokenizerSpecialTokenChecks: [QwenVLMTokenizerSpecialTokenCheck]
    ) -> [QwenVLMBundlePreflightFailureReason] {
        var reasons: [QwenVLMBundlePreflightFailureReason] = []
        if !missingRequiredResourceNames.isEmpty {
            reasons.append(.missingRequiredResources)
        }
        if !missingLanguageTensorNames.isEmpty {
            reasons.append(.missingLanguageModelTensors)
        }
        if visionTensorPrefixes.isEmpty {
            reasons.append(.missingVisionTensors)
        }
        if tokenizerSpecialTokenChecks.contains(where: { !$0.matches }) {
            reasons.append(.tokenizerSpecialTokenMismatch)
        }
        return reasons
    }
}

public enum QwenVLMBundlePreflightRunner {
    public static func run(
        configuration: QwenVLMBundlePreflightConfiguration
    ) throws -> QwenVLMBundlePreflightResult {
        let root = configuration.modelRootURL
        let configURL = root.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw QwenVLMBundlePreflightError.missingRequiredFile("config.json")
        }

        let configData = try Data(contentsOf: configURL)
        let rootConfig = try JSONDecoder().decode(RawQwenVLMRootConfig.self, from: configData)
        let actualFamily = try decodeFamily(rootConfig: rootConfig)
        if let expectedFamily = configuration.modelFamily, expectedFamily != actualFamily {
            throw QwenVLMBundlePreflightError.modelFamilyMismatch(
                expected: expectedFamily,
                actual: actualFamily
            )
        }

        let languageFamily: QwenModelFamily = actualFamily == .qwen36VLM ? .qwen36 : .qwen35
        let languageArchitecture = try QwenConfigDecoder.decodeArchitecture(
            from: configData,
            family: languageFamily
        )
        let preprocessor = try loadJSON(
            RawQwenVLMPreprocessorConfig.self,
            from: root.appendingPathComponent("preprocessor_config.json"),
            resourceName: "preprocessor_config.json"
        )
        let visionConfiguration = try makeVisionConfiguration(from: rootConfig.visionConfig)
        let imageProcessorConfiguration = makeImageProcessorConfiguration(from: preprocessor)
        let weightMap = try loadWeightMap(from: root)
        let languageIndex = try QwenModelBundleIndex(
            rootURL: root,
            architecture: languageArchitecture,
            weightMap: weightMap
        )
        let specialTokenChecks = try tokenizerSpecialTokenChecks(
            root: root,
            rootConfig: rootConfig
        )

        let requiredResources = [
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            rootModelWeightsResourceName,
        ]
        let visionPrefixes = detectedVisionTensorPrefixes(weightMap: weightMap)
        let plan = QwenVLMRuntimePlan(
            modelFamily: actualFamily,
            languageArchitecture: languageArchitecture,
            visionConfiguration: visionConfiguration,
            imageProcessorConfiguration: imageProcessorConfiguration,
            supportsMultipleImages: true
        )

        return QwenVLMBundlePreflightResult(
            modelRootPath: root.path,
            modelType: rootConfig.modelType,
            plan: plan,
            requiredResourceNames: requiredResources,
            missingRequiredResourceNames: missing(resources: requiredResources, under: root),
            missingLanguageTensorNames: languageIndex.missingRequiredTensorNames,
            visionTensorPrefixes: visionPrefixes,
            tokenizerSpecialTokenChecks: specialTokenChecks
        )
    }

    private static let rootModelWeightsResourceName = "model.safetensors.index.json"

    private static func decodeFamily(rootConfig: RawQwenVLMRootConfig) throws -> QwenVLMModelFamily {
        let normalizedModelType = rootConfig.modelType
            .lowercased()
            .replacingOccurrences(of: "-", with: "")
            .replacingOccurrences(of: "_", with: "")
            .replacingOccurrences(of: ".", with: "")
        guard normalizedModelType.contains("vl")
            || normalizedModelType.contains("vision")
            || rootConfig.visionConfig != nil else {
            throw QwenVLMBundlePreflightError.unsupportedModelType(rootConfig.modelType)
        }

        let normalizedFamilyHint = rootConfig.familyHint
            .lowercased()
            .replacingOccurrences(of: "-", with: "")
            .replacingOccurrences(of: "_", with: "")
            .replacingOccurrences(of: ".", with: "")
        if normalizedFamilyHint.contains("qwen36") || normalizedFamilyHint.contains("qwen3p6") {
            return .qwen36VLM
        }
        if normalizedFamilyHint.contains("qwen35") || normalizedFamilyHint.contains("qwen3p5") {
            return .qwen35VLM
        }
        if normalizedModelType.contains("qwen36") || normalizedModelType.contains("qwen3p6") {
            return .qwen36VLM
        }
        if normalizedModelType.contains("qwen35") || normalizedModelType.contains("qwen3p5") {
            return .qwen35VLM
        }
        throw QwenVLMBundlePreflightError.unsupportedModelType(rootConfig.modelType)
    }

    private static func makeVisionConfiguration(
        from raw: RawQwenVisionConfig?
    ) throws -> QwenVisionConfiguration {
        guard let raw, let hiddenSize = raw.hiddenSize,
              let layerCount = raw.layerCount ?? raw.depth,
              let attentionHeadCount = raw.attentionHeadCount ?? raw.headCount else {
            throw QwenVLMBundlePreflightError.invalidVisionConfiguration
        }
        return QwenVisionConfiguration(
            hiddenSize: hiddenSize,
            intermediateSize: raw.intermediateSize,
            layerCount: layerCount,
            attentionHeadCount: attentionHeadCount,
            patchSize: raw.patchSize,
            spatialMergeSize: raw.spatialMergeSize,
            imageSize: raw.imageSize
        )
    }

    private static func makeImageProcessorConfiguration(
        from raw: RawQwenVLMPreprocessorConfig
    ) -> QwenImageProcessorConfiguration {
        QwenImageProcessorConfiguration(
            imageProcessorType: raw.imageProcessorType,
            minPixels: raw.minPixels,
            maxPixels: raw.maxPixels,
            patchSize: raw.patchSize,
            mergeSize: raw.mergeSize,
            temporalPatchSize: raw.temporalPatchSize,
            imageMean: raw.imageMean,
            imageStd: raw.imageStd
        )
    }

    private static func loadWeightMap(from root: URL) throws -> [String: String] {
        let indexURL = root.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexURL.path) {
            let index = try JSONDecoder().decode(
                RawQwenVLMSafeTensorsIndex.self,
                from: try Data(contentsOf: indexURL)
            )
            guard !index.weightMap.isEmpty else {
                throw QwenVLMBundlePreflightError.invalidWeightIndex
            }
            return index.weightMap
        }
        throw QwenVLMBundlePreflightError.missingRequiredFile(rootModelWeightsResourceName)
    }

    private static func detectedVisionTensorPrefixes(weightMap: [String: String]) -> [String] {
        let candidatePrefixes = [
            "visual.",
            "vision_tower.",
            "vision_model.",
            "model.visual.",
            "language_model.visual.",
        ]
        return candidatePrefixes
            .filter { prefix in weightMap.keys.contains { $0.hasPrefix(prefix) } }
            .map { String($0.dropLast()) }
    }

    private static func tokenizerSpecialTokenChecks(
        root: URL,
        rootConfig: RawQwenVLMRootConfig
    ) throws -> [QwenVLMTokenizerSpecialTokenCheck] {
        let checks = [
            (name: "vision_start_token_id", tokenID: rootConfig.visionStartTokenID, expectedContent: "<|vision_start|>"),
            (name: "vision_end_token_id", tokenID: rootConfig.visionEndTokenID, expectedContent: "<|vision_end|>"),
            (name: "image_token_id", tokenID: rootConfig.imageTokenID, expectedContent: "<|image_pad|>"),
        ].compactMap { item -> (name: String, tokenID: Int, expectedContent: String)? in
            guard let tokenID = item.tokenID else { return nil }
            return (item.name, tokenID, item.expectedContent)
        }
        guard !checks.isEmpty else { return [] }

        let tokenizer = try loadJSON(
            RawQwenVLMTokenizerConfig.self,
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
            return QwenVLMTokenizerSpecialTokenCheck(
                name: check.name,
                tokenID: check.tokenID,
                expectedContent: check.expectedContent,
                actualContent: token?.content,
                isSpecial: token?.special ?? false
            )
        }
    }

    private static func missing(resources: [String], under root: URL) -> [String] {
        resources.filter { resource in
            if resource == rootModelWeightsResourceName {
                return !hasModelWeights(at: root)
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

    private static func loadJSON<T: Decodable>(
        _ type: T.Type,
        from url: URL,
        resourceName: String
    ) throws -> T {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw QwenVLMBundlePreflightError.missingRequiredFile(resourceName)
        }
        return try JSONDecoder().decode(type, from: Data(contentsOf: url))
    }
}

private struct RawQwenVLMRootConfig: Decodable {
    var modelType: String
    var familyHint: String
    var visionConfig: RawQwenVisionConfig?
    var imageTokenID: Int?
    var visionStartTokenID: Int?
    var visionEndTokenID: Int?

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case family
        case visionConfig = "vision_config"
        case textConfig = "text_config"
        case imageTokenID = "image_token_id"
        case visionStartTokenID = "vision_start_token_id"
        case visionEndTokenID = "vision_end_token_id"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        visionConfig = try container.decodeIfPresent(RawQwenVisionConfig.self, forKey: .visionConfig)
        imageTokenID = try container.decodeIfPresent(Int.self, forKey: .imageTokenID)
        visionStartTokenID = try container.decodeIfPresent(Int.self, forKey: .visionStartTokenID)
        visionEndTokenID = try container.decodeIfPresent(Int.self, forKey: .visionEndTokenID)
        let textConfig = try container.decodeIfPresent(RawQwenVLMTextConfig.self, forKey: .textConfig)
        familyHint = try container.decodeIfPresent(String.self, forKey: .family)
            ?? textConfig?.familyHint
            ?? textConfig?.modelType
            ?? modelType
    }
}

private struct RawQwenVLMTextConfig: Decodable {
    var familyHint: String?
    var modelType: String?

    private enum CodingKeys: String, CodingKey {
        case familyHint = "family"
        case modelType = "model_type"
    }
}

private struct RawQwenVisionConfig: Decodable {
    var hiddenSize: Int?
    var intermediateSize: Int?
    var layerCount: Int?
    var depth: Int?
    var attentionHeadCount: Int?
    var headCount: Int?
    var patchSize: Int?
    var spatialMergeSize: Int?
    var imageSize: Int?

    private enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case layerCount = "num_hidden_layers"
        case depth
        case attentionHeadCount = "num_attention_heads"
        case headCount = "num_heads"
        case patchSize = "patch_size"
        case spatialMergeSize = "spatial_merge_size"
        case imageSize = "image_size"
    }
}

private struct RawQwenVLMPreprocessorConfig: Decodable {
    struct Size: Decodable {
        var shortestEdge: Int?
        var longestEdge: Int?

        private enum CodingKeys: String, CodingKey {
            case shortestEdge = "shortest_edge"
            case longestEdge = "longest_edge"
        }
    }

    var imageProcessorType: String?
    var minPixels: Int?
    var maxPixels: Int?
    var size: Size?
    var patchSize: Int?
    var mergeSize: Int?
    var temporalPatchSize: Int?
    var imageMean: [Float]
    var imageStd: [Float]

    private enum CodingKeys: String, CodingKey {
        case imageProcessorType = "image_processor_type"
        case minPixels = "min_pixels"
        case maxPixels = "max_pixels"
        case size
        case patchSize = "patch_size"
        case mergeSize = "merge_size"
        case temporalPatchSize = "temporal_patch_size"
        case imageMean = "image_mean"
        case imageStd = "image_std"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        imageProcessorType = try container.decodeIfPresent(String.self, forKey: .imageProcessorType)
        size = try container.decodeIfPresent(Size.self, forKey: .size)
        minPixels = try container.decodeIfPresent(Int.self, forKey: .minPixels)
            ?? size?.shortestEdge
        maxPixels = try container.decodeIfPresent(Int.self, forKey: .maxPixels)
            ?? size?.longestEdge
        patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize)
        mergeSize = try container.decodeIfPresent(Int.self, forKey: .mergeSize)
        temporalPatchSize = try container.decodeIfPresent(Int.self, forKey: .temporalPatchSize)
        imageMean = try container.decodeIfPresent([Float].self, forKey: .imageMean) ?? []
        imageStd = try container.decodeIfPresent([Float].self, forKey: .imageStd) ?? []
    }
}

private struct RawQwenVLMTokenizerConfig: Decodable {
    var addedTokens: [RawQwenVLMAddedToken]

    private enum CodingKeys: String, CodingKey {
        case addedTokens = "added_tokens"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        addedTokens = try container.decodeIfPresent([RawQwenVLMAddedToken].self, forKey: .addedTokens) ?? []
    }
}

private struct RawQwenVLMAddedToken: Decodable {
    var id: Int
    var content: String
    var special: Bool
}

private struct RawQwenVLMSafeTensorsIndex: Decodable {
    var weightMap: [String: String]

    private enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}
