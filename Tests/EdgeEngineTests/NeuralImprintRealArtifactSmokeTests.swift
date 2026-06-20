// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import CryptoKit
import Foundation
import XCTest
@testable import EdgeEngine

final class NeuralImprintRealArtifactSmokeTests: XCTestCase {
    func testRealPhase1NeuralImprintArtifactValidatesWhenEnabled() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["EDGE_RUN_NEURAL_IMPRINT_REAL_ARTIFACT"] == "1" else {
            throw XCTSkip("Set EDGE_RUN_NEURAL_IMPRINT_REAL_ARTIFACT=1 to validate the real Phase 1 Neural Imprint artifact.")
        }

        let artifactDirectory = environment["EDGE_NEURAL_IMPRINT_ARTIFACT_DIR"]
            .map(expandedFileURL)
            ?? defaultPhase1ArtifactDirectory()
        let modelDirectory = try edgeTestDirectoryURL(
            fromEnvironment: "EDGE_NEURAL_IMPRINT_MODEL_PATH",
            purpose: "Neural Imprint real artifact smoke validation",
            requiredFileName: "config.json"
        )

        let artifactURL = firstExistingFile(
            in: artifactDirectory,
            names: ["neural_imprint.safetensors", "persona_kv.safetensors"]
        )
        let sidecarURL = firstExistingFile(
            in: artifactDirectory,
            names: ["neural_imprint_metadata.json", "persona_kv_metadata.json"]
        )
        let configURL = modelDirectory.appendingPathComponent("config.json")
        let tokenizerJSONURL = modelDirectory.appendingPathComponent("tokenizer.json")
        let tokenizerConfigURL = modelDirectory.appendingPathComponent("tokenizer_config.json")
        for url in [artifactURL, sidecarURL, configURL, tokenizerJSONURL, tokenizerConfigURL] {
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw XCTSkip("Required Neural Imprint smoke fixture is missing: \(url.path)")
            }
        }

        let artifact = try SafeTensorsShardFile(url: artifactURL)
        let sidecar = try NeuralImprintSidecar.load(from: sidecarURL)
        let header = artifact.metadata
        let architecture = try QwenConfigDecoder.decodeArchitecture(
            from: try Data(contentsOf: configURL),
            family: .qwen35
        )
        let configSHA256 = try sha256File(configURL)
        let tokenizerJSONSHA256 = try sha256File(tokenizerJSONURL)
        let tokenizerConfigSHA256 = try sha256File(tokenizerConfigURL)

        XCTAssertEqual(header["model_config_sha256"], configSHA256)
        XCTAssertEqual(header["tokenizer_json_sha256"], tokenizerJSONSHA256)
        XCTAssertEqual(header["tokenizer_config_sha256"], tokenizerConfigSHA256)

        let requirements = try NeuralImprintCompatibilityRequirements(
            modelArchitecture: requiredHeader("model_architecture", in: header),
            modelConfigSHA256: configSHA256,
            modelWeightsFingerprint: requiredHeader("model_weights_fingerprint", in: header),
            tokenizerJSONSHA256: tokenizerJSONSHA256,
            tokenizerConfigSHA256: tokenizerConfigSHA256,
            chatTemplateSHA256: requiredHeader("chat_template_sha256", in: header),
            renderedPrefixSHA256: requiredHeader("rendered_prefix_sha256", in: header),
            prefixTokenIDsSHA256: requiredHeader("prefix_token_ids_sha256", in: header),
            enableThinking: requiredHeader("enable_thinking", in: header),
            cacheBackend: requiredHeader("cache_backend", in: header),
            cacheBackendVersion: requiredHeader("cache_backend_version", in: header),
            cacheTopology: .qwen35(architecture: architecture)
        )

        try NeuralImprintArtifactValidator.validate(
            artifact: artifact,
            sidecar: sidecar,
            requirements: requirements
        )

        let runtime = try EdgeMetalRuntime(
            configuration: MetalRuntimeConfiguration(maxOpsPerCommandBuffer: 50, maxMBPerCommandBuffer: 50)
        )
        let session = try EdgeMLXQwen35Session(
            architecture: architecture,
            runtime: runtime
        )
        try session.restoreNeuralImprintCache(
            artifactURL: artifactURL,
            prefixTokenCount: sidecar.prefix.tokenCount
        )

        XCTAssertEqual(sidecar.cacheManifest.layerCount, architecture.layerPlan.count)
        XCTAssertEqual(sidecar.cacheManifest.layers.count, architecture.layerPlan.count)
        XCTAssertEqual(
            artifact.tensors.count,
            sidecar.cacheManifest.layers.reduce(0) { $0 + $1.states.count }
        )
        XCTAssertEqual(String(sidecar.prefix.tokenCount), header["prefix_token_count"])
        XCTAssertEqual(session.decodedTokenCount, sidecar.prefix.tokenCount)
    }
}

private func defaultPhase1ArtifactDirectory() -> URL {
    let thisFile = URL(fileURLWithPath: #filePath)
    let packageRoot = thisFile
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    return packageRoot
        .deletingLastPathComponent()
        .appendingPathComponent("RealData/persona_kv_multiturn_20260520/edgestudio_core_host_smoke")
}

private func expandedFileURL(_ path: String) -> URL {
    URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
}

private func firstExistingFile(in directory: URL, names: [String]) -> URL {
    let fileManager = FileManager.default
    for name in names {
        let url = directory.appendingPathComponent(name)
        if fileManager.fileExists(atPath: url.path) {
            return url
        }
    }
    return directory.appendingPathComponent(names[0])
}

private func requiredHeader(_ key: String, in header: [String: String]) throws -> String {
    guard let value = header[key], !value.isEmpty else {
        throw NeuralImprintArtifactError.missingHeaderField(key)
    }
    return value
}

private func sha256File(_ url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var hasher = SHA256()
    while true {
        let data = try handle.read(upToCount: 1024 * 1024) ?? Data()
        if data.isEmpty {
            break
        }
        hasher.update(data: data)
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
}
