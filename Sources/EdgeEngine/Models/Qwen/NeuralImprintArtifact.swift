// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import CryptoKit
import Foundation

public enum NeuralImprintArtifactError: Error, Equatable {
    case missingHeaderField(String)
    case emptyHeaderField(String)
    case headerMismatch(field: String, expected: String, actual: String?)
    case unsupportedSidecarSchema(String)
    case artifactHashMismatch(expected: String, actual: String)
    case sidecarHeaderMismatch(field: String, header: String, sidecar: String)
    case cacheLayerCountMismatch(expected: Int, actual: Int)
    case cacheLayerIndexMismatch(expected: Int, actual: Int)
    case cacheClassMismatch(layerIndex: Int, expected: String, actual: String)
    case cacheStateContainerMismatch(layerIndex: Int, expected: String, actual: String)
    case cacheStateCountMismatch(layerIndex: Int, expected: Int, actual: Int)
    case tensorMissing(String)
    case tensorShapeMismatch(name: String, expected: [Int], actual: [Int])
    case tensorDTypeMismatch(name: String, expected: String, actual: String)
}

public struct NeuralImprintCompatibilityRequirements: Equatable, Sendable {
    public var modelArchitecture: String
    public var modelConfigSHA256: String
    public var modelWeightsFingerprint: String
    public var tokenizerJSONSHA256: String
    public var tokenizerConfigSHA256: String
    public var chatTemplateSHA256: String
    public var renderedPrefixSHA256: String
    public var prefixTokenIDsSHA256: String
    public var enableThinking: String
    public var cacheBackend: String
    public var cacheBackendVersion: String
    public var cacheTopology: NeuralImprintCacheTopology

    public init(
        modelArchitecture: String,
        modelConfigSHA256: String,
        modelWeightsFingerprint: String,
        tokenizerJSONSHA256: String,
        tokenizerConfigSHA256: String,
        chatTemplateSHA256: String,
        renderedPrefixSHA256: String,
        prefixTokenIDsSHA256: String,
        enableThinking: String,
        cacheBackend: String,
        cacheBackendVersion: String,
        cacheTopology: NeuralImprintCacheTopology
    ) {
        self.modelArchitecture = modelArchitecture
        self.modelConfigSHA256 = modelConfigSHA256
        self.modelWeightsFingerprint = modelWeightsFingerprint
        self.tokenizerJSONSHA256 = tokenizerJSONSHA256
        self.tokenizerConfigSHA256 = tokenizerConfigSHA256
        self.chatTemplateSHA256 = chatTemplateSHA256
        self.renderedPrefixSHA256 = renderedPrefixSHA256
        self.prefixTokenIDsSHA256 = prefixTokenIDsSHA256
        self.enableThinking = enableThinking
        self.cacheBackend = cacheBackend
        self.cacheBackendVersion = cacheBackendVersion
        self.cacheTopology = cacheTopology
    }

    public var headerFields: [String: String] {
        [
            "model_architecture": modelArchitecture,
            "model_config_sha256": modelConfigSHA256,
            "model_weights_fingerprint": modelWeightsFingerprint,
            "tokenizer_json_sha256": tokenizerJSONSHA256,
            "tokenizer_config_sha256": tokenizerConfigSHA256,
            "chat_template_sha256": chatTemplateSHA256,
            "rendered_prefix_sha256": renderedPrefixSHA256,
            "prefix_token_ids_sha256": prefixTokenIDsSHA256,
            "enable_thinking": enableThinking,
            "cache_backend": cacheBackend,
            "cache_backend_version": cacheBackendVersion,
        ]
    }
}

public struct NeuralImprintCacheTopology: Equatable, Sendable {
    public var layers: [NeuralImprintCacheTopologyLayer]

    public init(layers: [NeuralImprintCacheTopologyLayer]) {
        self.layers = layers
    }

    public static func qwen35(architecture: QwenHybridArchitecture) throws -> NeuralImprintCacheTopology {
        try architecture.validate()
        return NeuralImprintCacheTopology(
            layers: architecture.layerPlan.map { layer in
                switch layer.kind {
                case .fullAttention:
                    return NeuralImprintCacheTopologyLayer(
                        layer: layer.index,
                        cacheClass: "KVCache",
                        stateContainer: "tuple",
                        stateCount: 2
                    )
                case .gdn:
                    return NeuralImprintCacheTopologyLayer(
                        layer: layer.index,
                        cacheClass: "ArraysCache",
                        stateContainer: "list",
                        stateCount: 2
                    )
                }
            }
        )
    }
}

public struct NeuralImprintCacheTopologyLayer: Equatable, Sendable {
    public var layer: Int
    public var cacheClass: String
    public var stateContainer: String
    public var stateCount: Int

    public init(layer: Int, cacheClass: String, stateContainer: String, stateCount: Int) {
        self.layer = layer
        self.cacheClass = cacheClass
        self.stateContainer = stateContainer
        self.stateCount = stateCount
    }
}

public struct NeuralImprintSidecar: Decodable, Equatable, Sendable {
    public var schema: String
    public var artifact: String
    public var artifactSHA256: String
    public var source: Source
    public var model: Model
    public var tokenizer: Tokenizer
    public var prefix: Prefix
    public var cacheManifest: CacheManifest

    private enum CodingKeys: String, CodingKey {
        case schema
        case artifact
        case artifactSHA256 = "artifact_sha256"
        case source
        case model
        case tokenizer
        case prefix
        case cacheManifest = "cache_manifest"
    }

    public struct Source: Decodable, Equatable, Sendable {
        public var profileBodyPath: String
        public var profileBodySHA256: String
        public var toolSpecsPath: String
        public var toolSchemaSHA256: String

        private enum CodingKeys: String, CodingKey {
            case profileBodyPath = "profile_body_path"
            case profileBodySHA256 = "profile_body_sha256"
            case toolSpecsPath = "tool_specs_path"
            case toolSchemaSHA256 = "tool_schema_sha256"
        }
    }

    public struct Model: Decodable, Equatable, Sendable {
        public var id: String
        public var architecture: String
        public var hiddenSize: Int
        public var numLayers: Int
        public var layerTypes: [String]

        private enum CodingKeys: String, CodingKey {
            case id
            case architecture
            case hiddenSize = "hidden_size"
            case numLayers = "num_layers"
            case layerTypes = "layer_types"
        }
    }

    public struct Tokenizer: Decodable, Equatable, Sendable {
        public var tokenizerJSONSHA256: String
        public var tokenizerConfigSHA256: String
        public var chatTemplateSHA256: String
        public var enableThinking: Bool

        private enum CodingKeys: String, CodingKey {
            case tokenizerJSONSHA256 = "tokenizer_json_sha256"
            case tokenizerConfigSHA256 = "tokenizer_config_sha256"
            case chatTemplateSHA256 = "chat_template_sha256"
            case enableThinking = "enable_thinking"
        }
    }

    public struct Prefix: Decodable, Equatable, Sendable {
        public var tokenCount: Int
        public var renderedPrefixSHA256: String
        public var tokenIDsSHA256: String

        private enum CodingKeys: String, CodingKey {
            case tokenCount = "token_count"
            case renderedPrefixSHA256 = "rendered_prefix_sha256"
            case tokenIDsSHA256 = "token_ids_sha256"
        }
    }

    public struct CacheManifest: Decodable, Equatable, Sendable {
        public var layerCount: Int
        public var layers: [Layer]

        private enum CodingKeys: String, CodingKey {
            case layerCount = "layer_count"
            case layers
        }
    }

    public struct Layer: Decodable, Equatable, Sendable {
        public var layer: Int
        public var cacheClass: String
        public var stateContainer: String
        public var stateCount: Int
        public var states: [State]
        public var offset: Int?
        public var metaState: String

        private enum CodingKeys: String, CodingKey {
            case layer
            case cacheClass = "cache_class"
            case stateContainer = "state_container"
            case stateCount = "state_count"
            case states
            case offset
            case metaState = "meta_state"
        }
    }

    public struct State: Decodable, Equatable, Sendable {
        public var name: String
        public var shape: [Int]
        public var dtype: String
    }

    public static func load(from url: URL) throws -> NeuralImprintSidecar {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(NeuralImprintSidecar.self, from: data)
    }
}

public enum NeuralImprintArtifactValidator {
    public static let artifactType = "neural_imprint"
    public static let legacyArtifactType = "persona_kv"
    public static let artifactVersion = "2"
    public static let legacyArtifactVersion = "1"
    public static let cacheSchema = "edgestudio.neural_imprint.full_cache.v2"
    public static let legacyCacheSchema = "edgestudio.persona_kv.full_cache.v1"
    public static let sidecarSchema = "edgestudio.neural_imprint.full_cache_metadata.v2"
    public static let legacySidecarSchema = "edgestudio.persona_kv.full_cache_metadata.v1"
    public static let prefixRendererVersion = "edgestudio.neural_imprint.renderer.v1"
    public static let legacyPrefixRendererVersion = "edgestudio.persona_kv.renderer.v2"

    public static let supportedArtifactTypes = [artifactType, legacyArtifactType]
    public static let supportedArtifactVersions = [artifactVersion, legacyArtifactVersion]
    public static let supportedCacheSchemas = [cacheSchema, legacyCacheSchema]
    public static let supportedSidecarSchemas = [sidecarSchema, legacySidecarSchema]
    public static let supportedPrefixRendererVersions = [prefixRendererVersion, legacyPrefixRendererVersion]

    public static let requiredHeaderFields: [String] = [
        "format",
        "artifact_type",
        "artifact_version",
        "cache_schema",
        "model_id",
        "model_architecture",
        "model_config_sha256",
        "model_weights_fingerprint",
        "tokenizer_json_sha256",
        "tokenizer_config_sha256",
        "chat_template_sha256",
        "system_prompt_sha256",
        "rendered_prefix_sha256",
        "prefix_token_ids_sha256",
        "prefix_token_count",
        "prefix_renderer_version",
        "tool_schema_sha256",
        "profile_body_sha256",
        "enable_thinking",
        "cache_backend",
        "cache_backend_version",
        "created_at",
        "created_by",
        "writer_version",
        "min_reader_version",
    ]

    public static func validate(
        artifact: SafeTensorsShardFile,
        sidecar: NeuralImprintSidecar,
        requirements: NeuralImprintCompatibilityRequirements
    ) throws {
        let header = artifact.metadata
        try validateHeader(header, requirements: requirements)
        try validateSidecar(sidecar, header: header)

        let actualSHA256 = try sha256File(artifact.url)
        guard sidecar.artifactSHA256 == actualSHA256 else {
            throw NeuralImprintArtifactError.artifactHashMismatch(
                expected: sidecar.artifactSHA256,
                actual: actualSHA256
            )
        }

        try validateCacheManifest(
            sidecar.cacheManifest,
            tensors: artifact.tensors,
            expectedTopology: requirements.cacheTopology
        )
    }

    public static func validateHeader(
        _ header: [String: String],
        requirements: NeuralImprintCompatibilityRequirements
    ) throws {
        for field in requiredHeaderFields {
            guard let value = header[field] else {
                throw NeuralImprintArtifactError.missingHeaderField(field)
            }
            guard !value.isEmpty else {
                throw NeuralImprintArtifactError.emptyHeaderField(field)
            }
        }

        try require(header, field: "format", equals: "mlx")
        try require(header, field: "artifact_type", oneOf: supportedArtifactTypes)
        try require(header, field: "artifact_version", oneOf: supportedArtifactVersions)
        try require(header, field: "cache_schema", oneOf: supportedCacheSchemas)
        try require(header, field: "prefix_renderer_version", oneOf: supportedPrefixRendererVersions)
        for (field, expected) in requirements.headerFields {
            try require(header, field: field, equals: expected)
        }
    }

    public static func validateCacheManifest(
        _ manifest: NeuralImprintSidecar.CacheManifest,
        tensors: [String: SafeTensorMetadata],
        expectedTopology: NeuralImprintCacheTopology
    ) throws {
        guard manifest.layerCount == expectedTopology.layers.count,
              manifest.layers.count == expectedTopology.layers.count
        else {
            throw NeuralImprintArtifactError.cacheLayerCountMismatch(
                expected: expectedTopology.layers.count,
                actual: manifest.layerCount
            )
        }

        for (index, layer) in manifest.layers.enumerated() {
            let expectedLayer = expectedTopology.layers[index]
            guard layer.layer == expectedLayer.layer else {
                throw NeuralImprintArtifactError.cacheLayerIndexMismatch(
                    expected: expectedLayer.layer,
                    actual: layer.layer
                )
            }
            guard layer.cacheClass == expectedLayer.cacheClass else {
                throw NeuralImprintArtifactError.cacheClassMismatch(
                    layerIndex: layer.layer,
                    expected: expectedLayer.cacheClass,
                    actual: layer.cacheClass
                )
            }
            guard layer.stateContainer == expectedLayer.stateContainer else {
                throw NeuralImprintArtifactError.cacheStateContainerMismatch(
                    layerIndex: layer.layer,
                    expected: expectedLayer.stateContainer,
                    actual: layer.stateContainer
                )
            }
            guard layer.stateCount == expectedLayer.stateCount,
                  layer.states.count == expectedLayer.stateCount
            else {
                throw NeuralImprintArtifactError.cacheStateCountMismatch(
                    layerIndex: layer.layer,
                    expected: expectedLayer.stateCount,
                    actual: layer.stateCount
                )
            }
            for (stateIndex, state) in layer.states.enumerated() {
                let expectedName = String(format: "layer_%02d.state_%d", layer.layer, stateIndex)
                guard state.name == expectedName else {
                    throw NeuralImprintArtifactError.headerMismatch(
                        field: "tensor_name",
                        expected: expectedName,
                        actual: state.name
                    )
                }
                guard let tensor = tensors[state.name] else {
                    throw NeuralImprintArtifactError.tensorMissing(state.name)
                }
                guard tensor.shape == state.shape else {
                    throw NeuralImprintArtifactError.tensorShapeMismatch(
                        name: state.name,
                        expected: state.shape,
                        actual: tensor.shape
                    )
                }
                let expectedDType = normalizedDType(state.dtype)
                let actualDType = normalizedDType(tensor.dtype)
                guard expectedDType == actualDType else {
                    throw NeuralImprintArtifactError.tensorDTypeMismatch(
                        name: state.name,
                        expected: state.dtype,
                        actual: tensor.dtype
                    )
                }
            }
        }
    }

    private static func validateSidecar(
        _ sidecar: NeuralImprintSidecar,
        header: [String: String]
    ) throws {
        guard supportedSidecarSchemas.contains(sidecar.schema) else {
            throw NeuralImprintArtifactError.unsupportedSidecarSchema(sidecar.schema)
        }
        try requireSidecarHeader(field: "model_id", header: header, sidecar: sidecar.model.id)
        try requireSidecarHeader(field: "model_architecture", header: header, sidecar: sidecar.model.architecture)
        try requireSidecarHeader(
            field: "tokenizer_json_sha256",
            header: header,
            sidecar: sidecar.tokenizer.tokenizerJSONSHA256
        )
        try requireSidecarHeader(
            field: "tokenizer_config_sha256",
            header: header,
            sidecar: sidecar.tokenizer.tokenizerConfigSHA256
        )
        try requireSidecarHeader(
            field: "chat_template_sha256",
            header: header,
            sidecar: sidecar.tokenizer.chatTemplateSHA256
        )
        try requireSidecarHeader(
            field: "enable_thinking",
            header: header,
            sidecar: sidecar.tokenizer.enableThinking ? "true" : "false"
        )
        try requireSidecarHeader(
            field: "profile_body_sha256",
            header: header,
            sidecar: sidecar.source.profileBodySHA256
        )
        try requireSidecarHeader(
            field: "tool_schema_sha256",
            header: header,
            sidecar: sidecar.source.toolSchemaSHA256
        )
        try requireSidecarHeader(
            field: "rendered_prefix_sha256",
            header: header,
            sidecar: sidecar.prefix.renderedPrefixSHA256
        )
        try requireSidecarHeader(
            field: "prefix_token_ids_sha256",
            header: header,
            sidecar: sidecar.prefix.tokenIDsSHA256
        )
        try requireSidecarHeader(
            field: "prefix_token_count",
            header: header,
            sidecar: String(sidecar.prefix.tokenCount)
        )
    }

    private static func require(
        _ header: [String: String],
        field: String,
        equals expected: String
    ) throws {
        let actual = header[field]
        guard actual == expected else {
            throw NeuralImprintArtifactError.headerMismatch(
                field: field,
                expected: expected,
                actual: actual
            )
        }
    }

    private static func require(
        _ header: [String: String],
        field: String,
        oneOf expectedValues: [String]
    ) throws {
        let actual = header[field]
        guard let actual, expectedValues.contains(actual) else {
            throw NeuralImprintArtifactError.headerMismatch(
                field: field,
                expected: expectedValues.joined(separator: " | "),
                actual: actual
            )
        }
    }

    private static func requireSidecarHeader(
        field: String,
        header: [String: String],
        sidecar: String
    ) throws {
        let headerValue = header[field] ?? ""
        guard headerValue == sidecar else {
            throw NeuralImprintArtifactError.sidecarHeaderMismatch(
                field: field,
                header: headerValue,
                sidecar: sidecar
            )
        }
    }

    private static func normalizedDType(_ value: String) -> String {
        switch value.lowercased() {
        case "bf16", "bfloat16", "mlx.core.bfloat16":
            return "bf16"
        case "f16", "float16", "mlx.core.float16":
            return "f16"
        case "f32", "float32", "mlx.core.float32":
            return "f32"
        default:
            return value.lowercased()
        }
    }

    private static func sha256File(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let data = try autoreleasepool {
                try handle.read(upToCount: 1024 * 1024) ?? Data()
            }
            if data.isEmpty {
                break
            }
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
