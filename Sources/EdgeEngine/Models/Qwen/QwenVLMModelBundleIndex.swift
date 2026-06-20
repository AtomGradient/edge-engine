// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum QwenVLMModelBundleIndexError: Error, Equatable {
    case invalidWeightIndex
    case preflightFailed([QwenVLMBundlePreflightFailureReason])
    case missingLanguageTensors
    case missingVisionTensorPrefix
    case unclassifiedTensors([String])
    case missingTensor(String)
}

public enum QwenVLMTensorRole: String, Codable, Equatable, Sendable {
    case language
    case vision
}

public struct QwenVLMTensorGroupManifest: Codable, Equatable, Sendable {
    public var role: QwenVLMTensorRole
    public var prefix: String
    public var tensorNames: [String]

    public init(role: QwenVLMTensorRole, prefix: String, tensorNames: [String]) {
        self.role = role
        self.prefix = prefix
        self.tensorNames = tensorNames
    }
}

public struct QwenVLMWeightFootprint: Codable, Equatable, Sendable {
    public var languageWeightBytes: Int
    public var visionWeightBytes: Int

    public init(languageWeightBytes: Int, visionWeightBytes: Int) {
        self.languageWeightBytes = languageWeightBytes
        self.visionWeightBytes = visionWeightBytes
    }

    public var totalWeightBytes: Int {
        languageWeightBytes + visionWeightBytes
    }
}

public enum QwenVLMNativeLoadingStage: String, Codable, Equatable, Sendable {
    case validateBundle
    case loadStructureOnly
    case loadDecoderWeights
    case unloadDecoderWeights
    case loadVisionWeights
    case precomputeVisionFeatures
    case unloadVisionWeights
    case decode
}

public struct QwenVLMPhasedLoadingPlan: Codable, Equatable, Sendable {
    public var footprint: QwenVLMWeightFootprint
    public var jetsamLimitMB: Int
    public var appReserveMB: Int
    public var structureBaselineMB: Int
    public var activationReserveMB: Int
    public var requiresPhasedLoading: Bool
    public var isPhasedFeasible: Bool
    public var textTurnStages: [QwenVLMNativeLoadingStage]
    public var imageTurnStages: [QwenVLMNativeLoadingStage]

    public init(
        footprint: QwenVLMWeightFootprint,
        jetsamLimitMB: Int,
        appReserveMB: Int,
        structureBaselineMB: Int,
        activationReserveMB: Int,
        requiresPhasedLoading: Bool,
        isPhasedFeasible: Bool,
        textTurnStages: [QwenVLMNativeLoadingStage],
        imageTurnStages: [QwenVLMNativeLoadingStage]
    ) {
        self.footprint = footprint
        self.jetsamLimitMB = jetsamLimitMB
        self.appReserveMB = appReserveMB
        self.structureBaselineMB = structureBaselineMB
        self.activationReserveMB = activationReserveMB
        self.requiresPhasedLoading = requiresPhasedLoading
        self.isPhasedFeasible = isPhasedFeasible
        self.textTurnStages = textTurnStages
        self.imageTurnStages = imageTurnStages
    }

    public var fullLoadPeakMB: Int {
        bytesToMegabytes(footprint.totalWeightBytes) + structureBaselineMB + activationReserveMB
    }

    public var phasedImagePeakMB: Int {
        max(
            bytesToMegabytes(footprint.languageWeightBytes),
            bytesToMegabytes(footprint.visionWeightBytes)
        ) + structureBaselineMB + activationReserveMB
    }

    public var usableJetsamMB: Int {
        max(0, jetsamLimitMB - appReserveMB)
    }
}

public struct QwenVLMModelBundleIndex: Equatable, Sendable {
    public var rootURL: URL
    public var preflightResult: QwenVLMBundlePreflightResult
    public var weightMap: [String: String]
    public var languageIndex: QwenModelBundleIndex
    public var languageManifest: QwenVLMTensorGroupManifest
    public var visionManifest: QwenVLMTensorGroupManifest

    public init(
        rootURL: URL,
        preflightResult: QwenVLMBundlePreflightResult,
        weightMap: [String: String],
        languageIndex: QwenModelBundleIndex,
        languageManifest: QwenVLMTensorGroupManifest,
        visionManifest: QwenVLMTensorGroupManifest
    ) {
        self.rootURL = rootURL
        self.preflightResult = preflightResult
        self.weightMap = weightMap
        self.languageIndex = languageIndex
        self.languageManifest = languageManifest
        self.visionManifest = visionManifest
    }

    public static func load(
        from rootURL: URL,
        family explicitFamily: QwenVLMModelFamily? = nil
    ) throws -> QwenVLMModelBundleIndex {
        let preflight = try QwenVLMBundlePreflightRunner.run(
            configuration: QwenVLMBundlePreflightConfiguration(
                modelRootURL: rootURL,
                modelFamily: explicitFamily
            )
        )
        guard preflight.passesPreflight else {
            throw QwenVLMModelBundleIndexError.preflightFailed(preflight.failureReasons)
        }

        let weightMap = try loadWeightMap(from: rootURL)
        let languageIndex = try QwenModelBundleIndex(
            rootURL: rootURL,
            architecture: preflight.plan.languageArchitecture,
            weightMap: weightMap
        )
        guard let visionPrefix = preflight.visionTensorPrefixes.first else {
            throw QwenVLMModelBundleIndexError.missingVisionTensorPrefix
        }
        let languagePrefix = languageRootPrefix(modelPrefix: languageIndex.modelPrefix)
        let visionPrefixes = preflight.visionTensorPrefixes
        let languageNames = languageTensorNames(
            from: languageIndex,
            excludingVisionPrefixes: visionPrefixes
        )
        let visionNames = weightMap.keys
            .filter { $0.hasPrefix("\(visionPrefix).") }
            .sorted()
        guard !languageNames.isEmpty else {
            throw QwenVLMModelBundleIndexError.missingLanguageTensors
        }
        guard !visionNames.isEmpty else {
            throw QwenVLMModelBundleIndexError.missingVisionTensorPrefix
        }
        let classifiedNames = Set(languageNames).union(visionNames)
        let unclassifiedNames = Set(weightMap.keys)
            .subtracting(classifiedNames)
            .sorted()
        guard unclassifiedNames.isEmpty else {
            throw QwenVLMModelBundleIndexError.unclassifiedTensors(unclassifiedNames)
        }

        return QwenVLMModelBundleIndex(
            rootURL: rootURL,
            preflightResult: preflight,
            weightMap: weightMap,
            languageIndex: languageIndex,
            languageManifest: QwenVLMTensorGroupManifest(
                role: .language,
                prefix: languagePrefix,
                tensorNames: languageNames
            ),
            visionManifest: QwenVLMTensorGroupManifest(
                role: .vision,
                prefix: visionPrefix,
                tensorNames: visionNames
            )
        )
    }

    public func makeWeightFootprint() throws -> QwenVLMWeightFootprint {
        try QwenVLMWeightFootprint(
            languageWeightBytes: byteCount(for: languageManifest.tensorNames),
            visionWeightBytes: byteCount(for: visionManifest.tensorNames)
        )
    }

    public func shardFileName(containing tensorName: String) throws -> String {
        guard let fileName = weightMap[tensorName] else {
            throw QwenVLMModelBundleIndexError.missingTensor(tensorName)
        }
        return fileName
    }

    public func shardFileURL(containing tensorName: String) throws -> URL {
        rootURL.appendingPathComponent(try shardFileName(containing: tensorName))
    }

    public func makePhasedLoadingPlan(
        jetsamLimitMB: Int,
        appReserveMB: Int = 1_024,
        structureBaselineMB: Int = 256,
        activationReserveMB: Int = 800
    ) throws -> QwenVLMPhasedLoadingPlan {
        let footprint = try makeWeightFootprint()
        let usableJetsamMB = max(0, jetsamLimitMB - appReserveMB)
        let fullLoadPeakMB = bytesToMegabytes(footprint.totalWeightBytes)
            + structureBaselineMB
            + activationReserveMB
        let phasedImagePeakMB = max(
            bytesToMegabytes(footprint.languageWeightBytes),
            bytesToMegabytes(footprint.visionWeightBytes)
        ) + structureBaselineMB + activationReserveMB
        let requiresPhasedLoading = fullLoadPeakMB > usableJetsamMB
        let isPhasedFeasible = phasedImagePeakMB <= usableJetsamMB
        let textStages: [QwenVLMNativeLoadingStage] = [
            .validateBundle,
            .loadStructureOnly,
            .loadDecoderWeights,
            .decode,
        ]
        let imageStages: [QwenVLMNativeLoadingStage] = requiresPhasedLoading
            ? [
                .validateBundle,
                .loadStructureOnly,
                .loadVisionWeights,
                .precomputeVisionFeatures,
                .unloadVisionWeights,
                .loadDecoderWeights,
                .decode,
            ]
            : [
                .validateBundle,
                .loadStructureOnly,
                .loadDecoderWeights,
                .loadVisionWeights,
                .precomputeVisionFeatures,
                .decode,
            ]
        return QwenVLMPhasedLoadingPlan(
            footprint: footprint,
            jetsamLimitMB: jetsamLimitMB,
            appReserveMB: appReserveMB,
            structureBaselineMB: structureBaselineMB,
            activationReserveMB: activationReserveMB,
            requiresPhasedLoading: requiresPhasedLoading,
            isPhasedFeasible: isPhasedFeasible,
            textTurnStages: textStages,
            imageTurnStages: imageStages
        )
    }

    private func byteCount(for tensorNames: [String]) throws -> Int {
        try tensorNames.reduce(0) { partial, tensorName in
            guard weightMap[tensorName] != nil else {
                throw QwenVLMModelBundleIndexError.missingTensor(tensorName)
            }
            let shard = try SafeTensorsShardFile(url: shardFileURL(containing: tensorName))
            let metadata = try shard.metadata(named: tensorName)
            return partial + metadata.dataOffsets.count
        }
    }

    private static func loadWeightMap(from rootURL: URL) throws -> [String: String] {
        let indexURL = rootURL.appendingPathComponent("model.safetensors.index.json")
        let index = try JSONDecoder().decode(
            RawQwenVLMModelBundleSafeTensorsIndex.self,
            from: try Data(contentsOf: indexURL)
        )
        guard !index.weightMap.isEmpty else {
            throw QwenVLMModelBundleIndexError.invalidWeightIndex
        }
        return index.weightMap
    }
}

private func languageRootPrefix(modelPrefix: String) -> String {
    modelPrefix.hasSuffix(".model")
        ? String(modelPrefix.dropLast(".model".count))
        : modelPrefix
}

private func languageTensorNames(
    from languageIndex: QwenModelBundleIndex,
    excludingVisionPrefixes visionPrefixes: [String]
) -> [String] {
    var names = Set<String>()
    names.formUnion(languageIndex.modelLevelManifest.requiredTensorNames)
    names.formUnion(languageIndex.modelLevelManifest.optionalTensorNames)
    for layerManifest in languageIndex.layerManifests {
        names.formUnion(layerManifest.requiredTensorNames)
        names.formUnion(layerManifest.optionalTensorNames)
    }
    return names
        .filter { languageIndex.weightMap[$0] != nil }
        .filter { tensorName in
            !visionPrefixes.contains { visionPrefix in
                tensorName.hasPrefix("\(visionPrefix).")
            }
        }
        .sorted()
}

private struct RawQwenVLMModelBundleSafeTensorsIndex: Decodable {
    var weightMap: [String: String]

    private enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}

private func bytesToMegabytes(_ bytes: Int) -> Int {
    Int((Double(bytes) / 1_048_576.0).rounded(.up))
}
