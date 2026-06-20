// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import XCTest
@testable import EdgeEngine

final class QwenVisionCmlxEncoderTests: XCTestCase {
    func testCmlxVisionEncoderFixtureParityWhenEnabled() throws {
        guard ProcessInfo.processInfo.environment["EDGE_RUN_VLM_FULL_VISION"] == "1" else {
            throw XCTSkip("Set EDGE_RUN_VLM_FULL_VISION=1 to run the full Cmlx vision encoder fixture.")
        }
        guard EdgeMLXBridge.isVendorPresent else {
            throw XCTSkip("Cmlx vendor bridge is unavailable")
        }

        let root = try edgeTestDirectoryURL(
            fromEnvironment: "EDGE_VLM_MODEL_PATH",
            purpose: "full Cmlx vision encoder fixture parity",
            requiredFileName: "config.json"
        )
        let imageURL = try edgeTestFileURL(
            fromEnvironment: "EDGE_VLM_FIXTURE_IMAGE_PATH",
            purpose: "full Cmlx vision encoder fixture parity"
        )

        let fixture = try JSONDecoder().decode(
            VLMVisionEncoderFixture.self,
            from: Data(contentsOf: try fixtureFileURL("vlm_vision_encoder_fixture.json"))
        )
        let index = try QwenVLMModelBundleIndex.load(from: root)
        let preprocessing = try QwenImagePreprocessor.preprocessImage(
            at: imageURL,
            configuration: index.preflightResult.plan.imageProcessorConfiguration
        )
        let runtime = try EdgeMetalRuntime()
        let session = try EdgeMLXQwen35Session(
            architecture: index.languageIndex.architecture,
            runtime: runtime
        )
        try session.setVisionConfig(plan: index.preflightResult.plan)
        try session.loadVisionSafetensors(index: index)

        let output = try session.visionEncode(
            pixelValues: preprocessing.pixelValues,
            pixelValuesShape: preprocessing.pixelValuesShape,
            gridTHW: [preprocessing.imageGridTHW],
            spatialMergeSize: index.preflightResult.plan.visionConfiguration.spatialMergeSize
                ?? index.preflightResult.plan.imageProcessorConfiguration.mergeSize
                ?? 2,
            outputHiddenSize: index.preflightResult.plan.languageArchitecture.hiddenSize
        )

        XCTAssertEqual(output.shape, fixture.fullVisionOutput.shape)
        let tolerance = Float(5e-2)
        for (actual, expected) in zip(output.values.prefix(10), fixture.fullVisionOutput.first10) {
            XCTAssertEqual(actual, expected, accuracy: tolerance)
        }
        for (actual, expected) in zip(output.values.suffix(10), fixture.fullVisionOutput.last10) {
            XCTAssertEqual(actual, expected, accuracy: tolerance)
        }
        let stats = meanAndStandardDeviation(output.values)
        XCTAssertEqual(stats.mean, fixture.fullVisionOutput.mean, accuracy: 5e-2)
        XCTAssertEqual(stats.standardDeviation, fixture.fullVisionOutput.std, accuracy: 5e-2)
    }

    private func fixtureFileURL(_ name: String) throws -> URL {
        let testFile = URL(fileURLWithPath: #filePath)
        let fixture = testFile
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures")
            .appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: fixture.path) else {
            throw XCTSkip("Fixture JSON not available at \(fixture.path)")
        }
        return fixture
    }
}

private struct VLMVisionEncoderFixture: Decodable {
    var fullVisionOutput: Checkpoint

    private enum CodingKeys: String, CodingKey {
        case fullVisionOutput = "full_vision_output"
    }

    struct Checkpoint: Decodable {
        var shape: [Int]
        var first10: [Float]
        var last10: [Float]
        var mean: Double
        var std: Double

        private enum CodingKeys: String, CodingKey {
            case shape
            case first10 = "first_10"
            case last10 = "last_10"
            case mean
            case std
        }
    }
}

private func meanAndStandardDeviation(_ values: [Float]) -> (mean: Double, standardDeviation: Double) {
    let count = Double(values.count)
    let mean = values.reduce(0.0) { $0 + Double($1) } / count
    let variance = values.reduce(0.0) { partial, value in
        let delta = Double(value) - mean
        return partial + delta * delta
    } / count
    return (mean, sqrt(variance))
}
