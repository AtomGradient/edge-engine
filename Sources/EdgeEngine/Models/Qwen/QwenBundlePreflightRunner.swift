// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum QwenBundlePreflightOutputKind: String, Codable, Equatable, Sendable {
    case float
    case quantized
}

public enum QwenBundlePreflightFailureReason: String, Codable, Equatable, Sendable {
    case missingRequiredTensors = "missing_required_tensors"
    case indexedTensorsMissingFromShardHeaders = "indexed_tensors_missing_from_shard_headers"
}

public struct QwenBundlePreflightConfiguration {
    public var modelRootURL: URL
    public var family: QwenModelFamily?
    public var readShardHeaders: Bool

    public init(
        modelRootURL: URL,
        family: QwenModelFamily? = nil,
        readShardHeaders: Bool = true
    ) {
        self.modelRootURL = modelRootURL
        self.family = family
        self.readShardHeaders = readShardHeaders
    }
}

public struct QwenBundlePreflightShard: Codable, Equatable, Sendable {
    public var fileName: String
    public var tensorCount: Int
    public var dtypeCounts: [String: Int]
    public var fileByteCount: Int?

    public init(
        fileName: String,
        tensorCount: Int,
        dtypeCounts: [String: Int],
        fileByteCount: Int?
    ) {
        self.fileName = fileName
        self.tensorCount = tensorCount
        self.dtypeCounts = dtypeCounts
        self.fileByteCount = fileByteCount
    }
}

public struct QwenBundlePreflightResult: Codable, Equatable, Sendable {
    public var modelRootPath: String
    public var modelPrefix: String
    public var family: QwenModelFamily
    public var vocabularySize: Int
    public var hiddenSize: Int
    public var layerCount: Int
    public var fullAttentionLayerCount: Int
    public var gdnLayerCount: Int
    public var quantization: QwenQuantizationProfile?
    public var outputWeightName: String
    public var outputKind: QwenBundlePreflightOutputKind
    public var usesTiedEmbeddings: Bool
    public var readShardHeaders: Bool
    public var passesPreflight: Bool
    public var failureReasons: [QwenBundlePreflightFailureReason]
    public var requiredTensorCount: Int
    public var optionalTensorCount: Int
    public var missingRequiredTensorNames: [String]
    public var weightMapTensorCount: Int
    public var shardCount: Int
    public var shardFileNames: [String]
    public var shardHeaderTensorCount: Int
    public var dtypeCounts: [String: Int]
    public var modelLevelTensorDTypes: [String: String]
    public var quantizedGroupCount: Int
    public var quantizedGroupsWithScales: Int
    public var quantizedGroupsWithBiases: Int
    public var tensorsMissingFromShardHeaders: [String]
    public var shardHeaderTensorNamesNotInIndex: [String]
    public var shards: [QwenBundlePreflightShard]

    public init(
        modelRootPath: String,
        modelPrefix: String,
        family: QwenModelFamily,
        vocabularySize: Int,
        hiddenSize: Int,
        layerCount: Int,
        fullAttentionLayerCount: Int,
        gdnLayerCount: Int,
        quantization: QwenQuantizationProfile?,
        outputWeightName: String,
        outputKind: QwenBundlePreflightOutputKind,
        usesTiedEmbeddings: Bool,
        readShardHeaders: Bool = true,
        passesPreflight: Bool,
        failureReasons: [QwenBundlePreflightFailureReason],
        requiredTensorCount: Int,
        optionalTensorCount: Int,
        missingRequiredTensorNames: [String],
        weightMapTensorCount: Int,
        shardCount: Int,
        shardFileNames: [String],
        shardHeaderTensorCount: Int,
        dtypeCounts: [String: Int],
        modelLevelTensorDTypes: [String: String],
        quantizedGroupCount: Int,
        quantizedGroupsWithScales: Int,
        quantizedGroupsWithBiases: Int,
        tensorsMissingFromShardHeaders: [String],
        shardHeaderTensorNamesNotInIndex: [String],
        shards: [QwenBundlePreflightShard]
    ) {
        self.modelRootPath = modelRootPath
        self.modelPrefix = modelPrefix
        self.family = family
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.fullAttentionLayerCount = fullAttentionLayerCount
        self.gdnLayerCount = gdnLayerCount
        self.quantization = quantization
        self.outputWeightName = outputWeightName
        self.outputKind = outputKind
        self.usesTiedEmbeddings = usesTiedEmbeddings
        self.readShardHeaders = readShardHeaders
        self.passesPreflight = passesPreflight
        self.failureReasons = failureReasons
        self.requiredTensorCount = requiredTensorCount
        self.optionalTensorCount = optionalTensorCount
        self.missingRequiredTensorNames = missingRequiredTensorNames
        self.weightMapTensorCount = weightMapTensorCount
        self.shardCount = shardCount
        self.shardFileNames = shardFileNames
        self.shardHeaderTensorCount = shardHeaderTensorCount
        self.dtypeCounts = dtypeCounts
        self.modelLevelTensorDTypes = modelLevelTensorDTypes
        self.quantizedGroupCount = quantizedGroupCount
        self.quantizedGroupsWithScales = quantizedGroupsWithScales
        self.quantizedGroupsWithBiases = quantizedGroupsWithBiases
        self.tensorsMissingFromShardHeaders = tensorsMissingFromShardHeaders
        self.shardHeaderTensorNamesNotInIndex = shardHeaderTensorNamesNotInIndex
        self.shards = shards
    }

    private enum CodingKeys: String, CodingKey {
        case modelRootPath
        case modelPrefix
        case family
        case vocabularySize
        case hiddenSize
        case layerCount
        case fullAttentionLayerCount
        case gdnLayerCount
        case quantization
        case outputWeightName
        case outputKind
        case usesTiedEmbeddings
        case readShardHeaders
        case passesPreflight
        case failureReasons
        case requiredTensorCount
        case optionalTensorCount
        case missingRequiredTensorNames
        case weightMapTensorCount
        case shardCount
        case shardFileNames
        case shardHeaderTensorCount
        case dtypeCounts
        case modelLevelTensorDTypes
        case quantizedGroupCount
        case quantizedGroupsWithScales
        case quantizedGroupsWithBiases
        case tensorsMissingFromShardHeaders
        case shardHeaderTensorNamesNotInIndex
        case shards
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelRootPath = try container.decode(String.self, forKey: .modelRootPath)
        self.modelPrefix = try container.decode(String.self, forKey: .modelPrefix)
        self.family = try container.decode(QwenModelFamily.self, forKey: .family)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.layerCount = try container.decode(Int.self, forKey: .layerCount)
        self.fullAttentionLayerCount = try container.decode(Int.self, forKey: .fullAttentionLayerCount)
        self.gdnLayerCount = try container.decode(Int.self, forKey: .gdnLayerCount)
        self.quantization = try container.decodeIfPresent(QwenQuantizationProfile.self, forKey: .quantization)
        self.outputWeightName = try container.decode(String.self, forKey: .outputWeightName)
        self.outputKind = try container.decode(QwenBundlePreflightOutputKind.self, forKey: .outputKind)
        self.usesTiedEmbeddings = try container.decode(Bool.self, forKey: .usesTiedEmbeddings)
        self.requiredTensorCount = try container.decode(Int.self, forKey: .requiredTensorCount)
        self.optionalTensorCount = try container.decode(Int.self, forKey: .optionalTensorCount)
        self.missingRequiredTensorNames = try container.decode(
            [String].self,
            forKey: .missingRequiredTensorNames
        )
        self.weightMapTensorCount = try container.decode(Int.self, forKey: .weightMapTensorCount)
        self.shardCount = try container.decode(Int.self, forKey: .shardCount)
        self.shardFileNames = try container.decode([String].self, forKey: .shardFileNames)
        self.shardHeaderTensorCount = try container.decode(Int.self, forKey: .shardHeaderTensorCount)
        self.dtypeCounts = try container.decode([String: Int].self, forKey: .dtypeCounts)
        self.modelLevelTensorDTypes = try container.decode([String: String].self, forKey: .modelLevelTensorDTypes)
        self.quantizedGroupCount = try container.decode(Int.self, forKey: .quantizedGroupCount)
        self.quantizedGroupsWithScales = try container.decode(Int.self, forKey: .quantizedGroupsWithScales)
        self.quantizedGroupsWithBiases = try container.decode(Int.self, forKey: .quantizedGroupsWithBiases)
        self.tensorsMissingFromShardHeaders = try container.decode(
            [String].self,
            forKey: .tensorsMissingFromShardHeaders
        )
        self.shardHeaderTensorNamesNotInIndex = try container.decode(
            [String].self,
            forKey: .shardHeaderTensorNamesNotInIndex
        )
        self.shards = try container.decode([QwenBundlePreflightShard].self, forKey: .shards)

        self.readShardHeaders = try container.decodeIfPresent(
            Bool.self,
            forKey: .readShardHeaders
        ) ?? true
        let defaultReasons = Self.makeFailureReasons(
            missingRequiredTensorNames: missingRequiredTensorNames,
            tensorsMissingFromShardHeaders: tensorsMissingFromShardHeaders
        )
        self.failureReasons = try container.decodeIfPresent(
            [QwenBundlePreflightFailureReason].self,
            forKey: .failureReasons
        ) ?? defaultReasons
        self.passesPreflight = try container.decodeIfPresent(
            Bool.self,
            forKey: .passesPreflight
        ) ?? failureReasons.isEmpty
    }

    public static func makeFailureReasons(
        missingRequiredTensorNames: [String],
        tensorsMissingFromShardHeaders: [String]
    ) -> [QwenBundlePreflightFailureReason] {
        var reasons: [QwenBundlePreflightFailureReason] = []
        if !missingRequiredTensorNames.isEmpty {
            reasons.append(.missingRequiredTensors)
        }
        if !tensorsMissingFromShardHeaders.isEmpty {
            reasons.append(.indexedTensorsMissingFromShardHeaders)
        }
        return reasons
    }
}

public enum QwenBundlePreflightRunner {
    public static func run(configuration: QwenBundlePreflightConfiguration) throws -> QwenBundlePreflightResult {
        let bundleIndex = try QwenModelBundleIndex.load(
            from: configuration.modelRootURL,
            family: configuration.family
        )
        let shardFileNames = Array(Set(bundleIndex.weightMap.values)).sorted()
        let quantizedGroups = allQuantizedGroups(in: bundleIndex)
        let outputWeightName = bundleIndex.modelLevelManifest.lmHeadName
            ?? bundleIndex.modelLevelManifest.embedTokensName
        let missingRequiredTensorNames = bundleIndex.missingRequiredTensorNames

        let headerSummary = try configuration.readShardHeaders
            ? loadShardHeaderSummary(
                rootURL: configuration.modelRootURL,
                shardFileNames: shardFileNames
            )
            : ShardHeaderSummary()
        let tensorsMissingFromShardHeaders = configuration.readShardHeaders
            ? Array(Set(bundleIndex.weightMap.keys).subtracting(headerSummary.tensorNames)).sorted()
            : []
        let shardHeaderTensorNamesNotInIndex = configuration.readShardHeaders
            ? Array(headerSummary.tensorNames.subtracting(bundleIndex.weightMap.keys)).sorted()
            : []
        let failureReasons = QwenBundlePreflightResult.makeFailureReasons(
            missingRequiredTensorNames: missingRequiredTensorNames,
            tensorsMissingFromShardHeaders: tensorsMissingFromShardHeaders
        )

        return QwenBundlePreflightResult(
            modelRootPath: configuration.modelRootURL.path,
            modelPrefix: bundleIndex.modelPrefix,
            family: bundleIndex.architecture.family,
            vocabularySize: bundleIndex.architecture.vocabularySize,
            hiddenSize: bundleIndex.architecture.hiddenSize,
            layerCount: bundleIndex.architecture.layerCount,
            fullAttentionLayerCount: bundleIndex.architecture.fullAttentionLayerIndices.count,
            gdnLayerCount: bundleIndex.architecture.gdnLayerIndices.count,
            quantization: bundleIndex.architecture.quantization,
            outputWeightName: outputWeightName,
            outputKind: outputKind(
                outputWeightName: outputWeightName,
                bundleIndex: bundleIndex
            ),
            usesTiedEmbeddings: bundleIndex.modelLevelManifest.isWeightTied,
            readShardHeaders: configuration.readShardHeaders,
            passesPreflight: failureReasons.isEmpty,
            failureReasons: failureReasons,
            requiredTensorCount: requiredTensorNames(in: bundleIndex).count,
            optionalTensorCount: optionalTensorNames(in: bundleIndex).count,
            missingRequiredTensorNames: missingRequiredTensorNames,
            weightMapTensorCount: bundleIndex.weightMap.count,
            shardCount: shardFileNames.count,
            shardFileNames: shardFileNames,
            shardHeaderTensorCount: headerSummary.tensorNames.count,
            dtypeCounts: headerSummary.dtypeCounts,
            modelLevelTensorDTypes: modelLevelTensorDTypes(
                bundleIndex: bundleIndex,
                tensorMetadata: headerSummary.tensorMetadata
            ),
            quantizedGroupCount: quantizedGroups.count,
            quantizedGroupsWithScales: quantizedGroups.filter { $0.scalesName != nil }.count,
            quantizedGroupsWithBiases: quantizedGroups.filter { $0.biasesName != nil }.count,
            tensorsMissingFromShardHeaders: tensorsMissingFromShardHeaders,
            shardHeaderTensorNamesNotInIndex: shardHeaderTensorNamesNotInIndex,
            shards: headerSummary.shards
        )
    }

    private static func outputKind(
        outputWeightName: String,
        bundleIndex: QwenModelBundleIndex
    ) -> QwenBundlePreflightOutputKind {
        let hasQuantizedOutputCompanion = bundleIndex.modelLevelManifest.quantizedWeightGroups.contains { group in
            group.weightName == outputWeightName && group.scalesName != nil
        }
        if bundleIndex.architecture.quantization != nil && hasQuantizedOutputCompanion {
            return .quantized
        }
        return .float
    }

    private static func requiredTensorNames(in bundleIndex: QwenModelBundleIndex) -> [String] {
        bundleIndex.modelLevelManifest.requiredTensorNames
            + bundleIndex.layerManifests.flatMap(\.requiredTensorNames)
    }

    private static func optionalTensorNames(in bundleIndex: QwenModelBundleIndex) -> [String] {
        bundleIndex.modelLevelManifest.optionalTensorNames
            + bundleIndex.layerManifests.flatMap(\.optionalTensorNames)
    }

    private static func allQuantizedGroups(in bundleIndex: QwenModelBundleIndex) -> [QwenQuantizedWeightGroup] {
        bundleIndex.modelLevelManifest.quantizedWeightGroups
            + bundleIndex.layerManifests.flatMap(\.quantizedWeightGroups)
    }

    private static func modelLevelTensorDTypes(
        bundleIndex: QwenModelBundleIndex,
        tensorMetadata: [String: SafeTensorMetadata]
    ) -> [String: String] {
        var result: [String: String] = [:]
        for name in bundleIndex.modelLevelManifest.requiredTensorNames
            + bundleIndex.modelLevelManifest.optionalTensorNames
        {
            if let metadata = tensorMetadata[name] {
                result[name] = metadata.dtype
            }
        }
        return result
    }

    private static func loadShardHeaderSummary(
        rootURL: URL,
        shardFileNames: [String]
    ) throws -> ShardHeaderSummary {
        var summary = ShardHeaderSummary()
        for fileName in shardFileNames {
            let url = rootURL.appendingPathComponent(fileName)
            let shard = try SafeTensorsShardFile(url: url)
            var dtypeCounts: [String: Int] = [:]
            for (name, metadata) in shard.tensors {
                summary.tensorNames.insert(name)
                summary.tensorMetadata[name] = metadata
                summary.dtypeCounts[metadata.dtype, default: 0] += 1
                dtypeCounts[metadata.dtype, default: 0] += 1
            }
            summary.shards.append(
                QwenBundlePreflightShard(
                    fileName: fileName,
                    tensorCount: shard.tensors.count,
                    dtypeCounts: dtypeCounts,
                    fileByteCount: fileByteCount(url)
                )
            )
        }
        return summary
    }

    private static func fileByteCount(_ url: URL) -> Int? {
        guard let size = try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber else {
            return nil
        }
        return size.intValue
    }
}

private struct ShardHeaderSummary {
    var tensorNames = Set<String>()
    var tensorMetadata: [String: SafeTensorMetadata] = [:]
    var dtypeCounts: [String: Int] = [:]
    var shards: [QwenBundlePreflightShard] = []
}
