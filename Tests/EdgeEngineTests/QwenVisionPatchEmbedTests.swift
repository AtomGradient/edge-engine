// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import XCTest
@testable import EdgeEngine

final class QwenVisionPatchEmbedTests: XCTestCase {
    func testProjectReordersChannelFirstPatchRowsBeforeProjection() throws {
        guard EdgeMLXBridge.isVendorPresent else {
            throw XCTSkip("Cmlx vendor bridge is unavailable")
        }

        let weights = QwenVisionPatchEmbedWeights(
            outputChannels: 3,
            temporalPatchSize: 1,
            patchSize: 2,
            inputChannels: 2,
            projectionMatrix: [
                1, 0, 0,
                0, 2, 0,
                0, 0, 3,
                4, 0, 0,
                0, 5, 0,
                0, 0, 6,
                7, 0, 0,
                0, 8, 0,
            ],
            bias: [0.5, -1, 2]
        )
        let output = try weights.project(
            pixelValues: [
                1, 2,
                3, 4,
                10, 20,
                30, 40,
            ],
            shape: [1, 8]
        )

        XCTAssertEqual(output.shape, [1, 3])
        XCTAssertEqual(output.values[0], 109.5, accuracy: 1e-5)
        XCTAssertEqual(output.values[1], 354, accuracy: 1e-5)
        XCTAssertEqual(output.values[2], 188, accuracy: 1e-5)
    }

    func testPatchEmbedMatchesVisionFixtureWhenRealModelAvailable() throws {
        guard EdgeMLXBridge.isVendorPresent else {
            throw XCTSkip("Cmlx vendor bridge is unavailable")
        }
        let root = try edgeTestDirectoryURL(
            fromEnvironment: "EDGE_VLM_MODEL_PATH",
            purpose: "Qwen VLM patch embedding fixture parity",
            requiredFileName: "config.json"
        )
        let imageURL = try edgeTestFileURL(
            fromEnvironment: "EDGE_VLM_FIXTURE_IMAGE_PATH",
            purpose: "Qwen VLM patch embedding fixture parity"
        )

        let visionFixtureURL = try fixtureFileURL("vlm_vision_encoder_fixture.json")
        let preprocessingFixtureURL = try fixtureFileURL("vlm_preprocessing_fixture.json")
        let visionFixture = try JSONDecoder().decode(
            VLMVisionEncoderFixture.self,
            from: Data(contentsOf: visionFixtureURL)
        )
        let preprocessingFixture = try JSONDecoder().decode(
            VLMPreprocessingFixture.self,
            from: Data(contentsOf: preprocessingFixtureURL)
        )

        let index = try QwenVLMModelBundleIndex.load(from: root)
        let preprocessing = try QwenImagePreprocessor.preprocessImage(
            at: imageURL,
            configuration: index.preflightResult.plan.imageProcessorConfiguration
        )
        XCTAssertEqual(preprocessing.pixelValuesShape, preprocessingFixture.pixelValuesShape)
        XCTAssertEqual(preprocessing.pixelValuesShape, visionFixture.inputShape)

        let weights = try QwenVisionPatchEmbedWeights.load(
            from: QwenVisionWeightStore(index: index)
        )
        let output = try weights.project(
            pixelValues: preprocessing.pixelValues,
            shape: preprocessing.pixelValuesShape
        )

        XCTAssertEqual(output.shape, visionFixture.patchEmbedOutput.shape)
        let imageDecodeTolerance = Float(4e-3)
        for (actual, expected) in zip(output.values.prefix(10), visionFixture.patchEmbedOutput.first10) {
            XCTAssertEqual(actual, expected, accuracy: imageDecodeTolerance)
        }
        for (actual, expected) in zip(output.values.suffix(10), visionFixture.patchEmbedOutput.last10) {
            XCTAssertEqual(actual, expected, accuracy: imageDecodeTolerance)
        }

        let stats = meanAndStandardDeviation(output.values)
        XCTAssertEqual(stats.mean, visionFixture.patchEmbedOutput.mean, accuracy: 1e-3)
        XCTAssertEqual(stats.standardDeviation, visionFixture.patchEmbedOutput.std, accuracy: 1e-3)
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
    var inputShape: [Int]
    var patchEmbedOutput: Checkpoint

    private enum CodingKeys: String, CodingKey {
        case inputShape = "input_shape"
        case patchEmbedOutput = "patch_embed_output"
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

private struct VLMPreprocessingFixture: Decodable {
    var pixelValuesShape: [Int]

    private enum CodingKeys: String, CodingKey {
        case pixelValuesShape = "pixel_values_shape"
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
