// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import CryptoKit
import Foundation
import Testing
@testable import EdgeEngine

@Test func neuralImprintArtifactValidatorAcceptsMatchingHeaderSidecarAndTopology() throws {
    let fixture = try makeNeuralImprintFixture()

    try NeuralImprintArtifactValidator.validate(
        artifact: fixture.artifact,
        sidecar: fixture.sidecar,
        requirements: fixture.requirements
    )
}

@Test func neuralImprintArtifactValidatorAcceptsLegacyV1ArtifactSchema() throws {
    let fixture = try makeNeuralImprintFixture(
        artifactName: "persona_kv.safetensors",
        headerOverrides: [
            "artifact_type": NeuralImprintArtifactValidator.legacyArtifactType,
            "artifact_version": NeuralImprintArtifactValidator.legacyArtifactVersion,
            "cache_schema": NeuralImprintArtifactValidator.legacyCacheSchema,
            "prefix_renderer_version": NeuralImprintArtifactValidator.legacyPrefixRendererVersion,
        ],
        sidecarSchema: NeuralImprintArtifactValidator.legacySidecarSchema
    )

    try NeuralImprintArtifactValidator.validate(
        artifact: fixture.artifact,
        sidecar: fixture.sidecar,
        requirements: fixture.requirements
    )
}

@Test func neuralImprintArtifactValidatorRejectsMissingRequiredHeaderField() throws {
    let fixture = try makeNeuralImprintFixture(headerOverrides: ["cache_backend_version": nil])

    do {
        try NeuralImprintArtifactValidator.validate(
            artifact: fixture.artifact,
            sidecar: fixture.sidecar,
            requirements: fixture.requirements
        )
        Issue.record("neural_imprint artifacts without required header fields must fail closed.")
    } catch NeuralImprintArtifactError.missingHeaderField("cache_backend_version") {
        return
    }
    Issue.record("neural_imprint validator rejected with the wrong error.")
}

@Test func neuralImprintArtifactValidatorRejectsSidecarHeaderMismatch() throws {
    let fixture = try makeNeuralImprintFixture(sidecarProfileSHA256: "different-profile")

    do {
        try NeuralImprintArtifactValidator.validate(
            artifact: fixture.artifact,
            sidecar: fixture.sidecar,
            requirements: fixture.requirements
        )
        Issue.record("neural_imprint sidecar/header mismatches must fail closed.")
    } catch NeuralImprintArtifactError.sidecarHeaderMismatch(
        field: "profile_body_sha256",
        header: "profile-sha",
        sidecar: "different-profile"
    ) {
        return
    }
    Issue.record("neural_imprint validator rejected with the wrong error.")
}

@Test func neuralImprintArtifactValidatorRejectsRuntimeTopologyMismatch() throws {
    let fixture = try makeNeuralImprintFixture(sidecarLayer0CacheClass: "KVCache")

    do {
        try NeuralImprintArtifactValidator.validate(
            artifact: fixture.artifact,
            sidecar: fixture.sidecar,
            requirements: fixture.requirements
        )
        Issue.record("neural_imprint cache topology mismatches must fail closed.")
    } catch NeuralImprintArtifactError.cacheClassMismatch(
        layerIndex: 0,
        expected: "ArraysCache",
        actual: "KVCache"
    ) {
        return
    }
    Issue.record("neural_imprint validator rejected with the wrong error.")
}

@Test func neuralImprintArtifactValidatorRejectsTensorDTypeMismatch() throws {
    let fixture = try makeNeuralImprintFixture(sidecarLayer0State0DType: "mlx.core.float32")

    do {
        try NeuralImprintArtifactValidator.validate(
            artifact: fixture.artifact,
            sidecar: fixture.sidecar,
            requirements: fixture.requirements
        )
        Issue.record("neural_imprint tensor dtype mismatches must fail closed.")
    } catch NeuralImprintArtifactError.tensorDTypeMismatch(
        name: "layer_00.state_0",
        expected: "mlx.core.float32",
        actual: "BF16"
    ) {
        return
    }
    Issue.record("neural_imprint validator rejected with the wrong error.")
}

@Test func neuralImprintArtifactRestoresIntoCmlxDecodeCache() throws {
    let fixture = try makeNeuralImprintCmlxRestoreFixture()
    let runtime = try EdgeMetalRuntime(
        configuration: MetalRuntimeConfiguration(maxOpsPerCommandBuffer: 50, maxMBPerCommandBuffer: 50)
    )
    let session = try EdgeMLXQwen35Session(
        architecture: fixture.architecture,
        runtime: runtime
    )

    #expect(session.decodedTokenCount == 0)
    try session.restoreNeuralImprintCache(
        artifactURL: fixture.artifactURL,
        prefixTokenCount: fixture.prefixTokenCount
    )
    #expect(session.decodedTokenCount == fixture.prefixTokenCount)
}

@Test func neuralImprintArtifactSavesCurrentCmlxDecodeCacheAsSafeTensors() throws {
    let fixture = try makeNeuralImprintCmlxRestoreFixture()
    let runtime = try EdgeMetalRuntime(
        configuration: MetalRuntimeConfiguration(maxOpsPerCommandBuffer: 50, maxMBPerCommandBuffer: 50)
    )
    let session = try EdgeMLXQwen35Session(
        architecture: fixture.architecture,
        runtime: runtime
    )
    try session.restoreNeuralImprintCache(
        artifactURL: fixture.artifactURL,
        prefixTokenCount: fixture.prefixTokenCount
    )

    let exportURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("neural-imprint-cmlx-export-\(UUID().uuidString)")
        .appendingPathComponent("neural_imprint.safetensors")
    try FileManager.default.createDirectory(
        at: exportURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try session.saveNeuralImprintCache(
        artifactURL: exportURL,
        metadata: neuralImprintHeader(overrides: [:])
    )

    let exported = try SafeTensorsShardFile(url: exportURL)
    #expect(exported.metadata["artifact_type"] == "neural_imprint")
    #expect(exported.metadata["prefix_token_count"] == String(fixture.prefixTokenCount))
    #expect(exported.tensors["layer_00.state_0"]?.dtype == "BF16")
    #expect(exported.tensors["layer_00.state_0"]?.shape == [1, 3, 10])
    #expect(exported.tensors["layer_00.state_1"]?.dtype == "F32")
    #expect(exported.tensors["layer_00.state_1"]?.shape == [1, 2, 2, 3])
    #expect(exported.tensors["layer_01.state_0"]?.dtype == "BF16")
    #expect(exported.tensors["layer_01.state_0"]?.shape == [1, 1, 3, 4])
    #expect(exported.tensors["layer_01.state_1"]?.dtype == "BF16")
    #expect(exported.tensors["layer_01.state_1"]?.shape == [1, 1, 3, 4])

    let restoredExport = try EdgeMLXQwen35Session(
        architecture: fixture.architecture,
        runtime: runtime
    )
    try restoredExport.restoreNeuralImprintCache(
        artifactURL: exportURL,
        prefixTokenCount: fixture.prefixTokenCount
    )
    #expect(restoredExport.decodedTokenCount == fixture.prefixTokenCount)
}

private struct NeuralImprintFixture {
    var artifact: SafeTensorsShardFile
    var sidecar: NeuralImprintSidecar
    var requirements: NeuralImprintCompatibilityRequirements
}

private struct NeuralImprintCmlxRestoreFixture {
    var artifactURL: URL
    var architecture: QwenHybridArchitecture
    var prefixTokenCount: Int
}

private func makeNeuralImprintFixture(
    artifactName: String = "neural_imprint.safetensors",
    headerOverrides: [String: String?] = [:],
    sidecarSchema: String = NeuralImprintArtifactValidator.sidecarSchema,
    sidecarProfileSHA256: String = "profile-sha",
    sidecarLayer0CacheClass: String = "ArraysCache",
    sidecarLayer0State0DType: String = "mlx.core.bfloat16"
) throws -> NeuralImprintFixture {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("neural-imprint-artifact-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let artifactURL = directory.appendingPathComponent(artifactName)
    let sidecarURL = directory.appendingPathComponent("neural_imprint_metadata.json")
    let header = neuralImprintHeader(overrides: headerOverrides)
    let fileData = try neuralImprintSafeTensorsData(headerMetadata: header)
    try fileData.write(to: artifactURL)
    let artifactSHA256 = sha256(fileData)
    let sidecarJSON = neuralImprintSidecarJSON(
        schema: sidecarSchema,
        artifactName: artifactName,
        artifactSHA256: artifactSHA256,
        profileSHA256: sidecarProfileSHA256,
        layer0CacheClass: sidecarLayer0CacheClass,
        layer0State0DType: sidecarLayer0State0DType
    )
    try sidecarJSON.data(using: .utf8)!.write(to: sidecarURL)

    let artifact = try SafeTensorsShardFile(url: artifactURL)
    let sidecar = try NeuralImprintSidecar.load(from: sidecarURL)
    let requirements = try NeuralImprintCompatibilityRequirements(
        modelArchitecture: "qwen3_5",
        modelConfigSHA256: "config-sha",
        modelWeightsFingerprint: "weights-fingerprint",
        tokenizerJSONSHA256: "tokenizer-json-sha",
        tokenizerConfigSHA256: "tokenizer-config-sha",
        chatTemplateSHA256: "chat-template-sha",
        renderedPrefixSHA256: "rendered-prefix-sha",
        prefixTokenIDsSHA256: "token-ids-sha",
        enableThinking: "false",
        cacheBackend: "mlx-lm",
        cacheBackendVersion: "0.31.3",
        cacheTopology: .qwen35(architecture: makeNeuralImprintArchitecture())
    )
    return NeuralImprintFixture(
        artifact: artifact,
        sidecar: sidecar,
        requirements: requirements
    )
}

private func makeNeuralImprintCmlxRestoreFixture() throws -> NeuralImprintCmlxRestoreFixture {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("neural-imprint-cmlx-restore-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let artifactURL = directory.appendingPathComponent("neural_imprint.safetensors")
    let architecture = try makeNeuralImprintArchitecture()
    try neuralImprintCmlxRestoreSafeTensorsData().write(to: artifactURL)
    return NeuralImprintCmlxRestoreFixture(
        artifactURL: artifactURL,
        architecture: architecture,
        prefixTokenCount: 3
    )
}

private func makeNeuralImprintArchitecture() throws -> QwenHybridArchitecture {
    try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 128,
        hiddenSize: 8,
        intermediateSize: 32,
        attentionHeadCount: 2,
        keyValueHeadCount: 1,
        linearValueHeadCount: 2,
        linearKeyHeadCount: 1,
        linearKeyHeadDimension: 3,
        linearValueHeadDimension: 2,
        linearConvKernelSize: 4,
        contextLength: 64,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000,
        layerKinds: [.gdn, .fullAttention]
    )
}

private func neuralImprintHeader(overrides: [String: String?]) -> [String: String] {
    var header = [
        "format": "mlx",
        "artifact_type": NeuralImprintArtifactValidator.artifactType,
        "artifact_version": NeuralImprintArtifactValidator.artifactVersion,
        "cache_schema": NeuralImprintArtifactValidator.cacheSchema,
        "model_id": "Qwen3.5-4B-4bit",
        "model_architecture": "qwen3_5",
        "model_config_sha256": "config-sha",
        "model_weights_fingerprint": "weights-fingerprint",
        "tokenizer_json_sha256": "tokenizer-json-sha",
        "tokenizer_config_sha256": "tokenizer-config-sha",
        "chat_template_sha256": "chat-template-sha",
        "system_prompt_sha256": "system-prompt-sha",
        "rendered_prefix_sha256": "rendered-prefix-sha",
        "prefix_token_ids_sha256": "token-ids-sha",
        "prefix_token_count": "3",
        "prefix_renderer_version": NeuralImprintArtifactValidator.prefixRendererVersion,
        "tool_schema_sha256": "tool-sha",
        "profile_body_sha256": "profile-sha",
        "enable_thinking": "false",
        "cache_backend": "mlx-lm",
        "cache_backend_version": "0.31.3",
        "created_at": "2026-05-22T00:00:00Z",
        "created_by": "edgestudio-core",
        "writer_version": "edgestudio-core 0.0.1",
        "min_reader_version": "edgestudio-core 0.0.1",
    ]
    for (key, value) in overrides {
        header[key] = value
    }
    return header
}

private func neuralImprintCmlxRestoreSafeTensorsData() throws -> Data {
    try safeTensorsData(entries: [
        ("layer_00.state_0", "BF16", [1, 3, 10]),
        ("layer_00.state_1", "F32", [1, 2, 2, 3]),
        ("layer_01.state_0", "BF16", [1, 1, 3, 4]),
        ("layer_01.state_1", "BF16", [1, 1, 3, 4]),
    ])
}

private func neuralImprintSafeTensorsData(headerMetadata: [String: String]) throws -> Data {
    try safeTensorsData(
        entries: [
            ("layer_00.state_0", "BF16", [1]),
            ("layer_00.state_1", "F32", [1]),
            ("layer_01.state_0", "BF16", [1, 1, 1, 1]),
            ("layer_01.state_1", "BF16", [1, 1, 1, 1]),
        ],
        headerMetadata: headerMetadata
    )
}

private func safeTensorsData(
    entries: [(String, String, [Int])],
    headerMetadata: [String: String]? = nil
) throws -> Data {
    var payload = Data()
    var offset = 0
    var tensors: [String: Any] = [:]
    if let headerMetadata {
        tensors["__metadata__"] = headerMetadata
    }
    for entry in entries {
        let data = zeroTensorData(dtype: entry.1, shape: entry.2)
        payload.append(data)
        tensors[entry.0] = [
            "dtype": entry.1,
            "shape": entry.2,
            "data_offsets": [offset, offset + data.count],
        ]
        offset += data.count
    }
    let headerData = try JSONSerialization.data(withJSONObject: tensors, options: [.sortedKeys])
    var headerLength = UInt64(headerData.count).littleEndian
    var data = withUnsafeBytes(of: &headerLength) { Data($0) }
    data.append(headerData)
    data.append(payload)
    return data
}

private func zeroTensorData(dtype: String, shape: [Int]) -> Data {
    let elementCount = shape.reduce(1, *)
    let bytesPerElement = dtype == "F32" ? 4 : 2
    return Data(repeating: 0, count: elementCount * bytesPerElement)
}

private func neuralImprintSidecarJSON(
    schema: String = NeuralImprintArtifactValidator.sidecarSchema,
    artifactName: String = "neural_imprint.safetensors",
    artifactSHA256: String,
    profileSHA256: String,
    layer0CacheClass: String,
    layer0State0DType: String
) -> String {
    """
    {
      "schema": "\(schema)",
      "artifact": "\(artifactName)",
      "artifact_sha256": "\(artifactSHA256)",
      "source": {
        "profile_body_path": "profile_body.txt",
        "profile_body_sha256": "\(profileSHA256)",
        "tool_specs_path": "tool_specs.json",
        "tool_schema_sha256": "tool-sha"
      },
      "model": {
        "id": "Qwen3.5-4B-4bit",
        "architecture": "qwen3_5",
        "hidden_size": 8,
        "num_layers": 2,
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
        "layer_types": ["linear_attention", "full_attention"]
      },
      "tokenizer": {
        "tokenizer_json_sha256": "tokenizer-json-sha",
        "tokenizer_config_sha256": "tokenizer-config-sha",
        "chat_template_sha256": "chat-template-sha",
        "enable_thinking": false
      },
      "prefix": {
        "token_count": 3,
        "rendered_prefix_sha256": "rendered-prefix-sha",
        "token_ids_sha256": "token-ids-sha"
      },
      "cache_manifest": {
        "layer_count": 2,
        "layers": [
          {
            "layer": 0,
            "cache_class": "\(layer0CacheClass)",
            "state_container": "list",
            "state_count": 2,
            "states": [
              {"name": "layer_00.state_0", "shape": [1], "dtype": "\(layer0State0DType)"},
              {"name": "layer_00.state_1", "shape": [1], "dtype": "mlx.core.float32"}
            ],
            "offset": null,
            "meta_state": ""
          },
          {
            "layer": 1,
            "cache_class": "KVCache",
            "state_container": "tuple",
            "state_count": 2,
            "states": [
              {"name": "layer_01.state_0", "shape": [1, 1, 1, 1], "dtype": "mlx.core.bfloat16"},
              {"name": "layer_01.state_1", "shape": [1, 1, 1, 1], "dtype": "mlx.core.bfloat16"}
            ],
            "offset": 3,
            "meta_state": ""
          }
        ]
      }
    }
    """
}

private func sha256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}
