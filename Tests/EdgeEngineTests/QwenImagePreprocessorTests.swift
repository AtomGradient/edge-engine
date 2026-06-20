// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import XCTest
@testable import EdgeEngine

final class QwenImagePreprocessorTests: XCTestCase {
    func testTargetSizeMatchesQwenFixtureGrid() throws {
        let target = try QwenImagePreprocessor.targetSize(
            height: 1_024,
            width: 768,
            factor: 32,
            minPixels: 65_536,
            maxPixels: 16_777_216
        )

        XCTAssertEqual(target, QwenImageTargetSize(height: 1_024, width: 768))
        XCTAssertEqual(target.height / 16, 64)
        XCTAssertEqual(target.width / 16, 48)
    }

    func testNonAlignedTargetSizeMatchesQwenFixtureGrid() throws {
        let fixtureURL = try fixtureFileURL(named: "vlm_preprocessing_fixture_nonaligned.json")
        let fixture = try JSONDecoder().decode(
            VLMNonAlignedPreprocessingFixture.self,
            from: Data(contentsOf: fixtureURL)
        )
        let originalWidth = fixture.originalSize[0]
        let originalHeight = fixture.originalSize[1]

        let target = try QwenImagePreprocessor.targetSize(
            height: originalHeight,
            width: originalWidth,
            factor: 32,
            minPixels: 65_536,
            maxPixels: 16_777_216
        )
        let expectedGrid = QwenImageGridTHW(
            temporal: fixture.imageGridTHW[0][0],
            height: fixture.imageGridTHW[0][1],
            width: fixture.imageGridTHW[0][2]
        )

        XCTAssertEqual(target, QwenImageTargetSize(height: 4_704, width: 3_520))
        XCTAssertEqual(target.height / 16, expectedGrid.height)
        XCTAssertEqual(target.width / 16, expectedGrid.width)
        XCTAssertEqual(
            fixture.pixelValuesShape,
            [expectedGrid.product, 3 * 2 * 16 * 16]
        )
    }

    func testPatchifyDuplicatesSingleFrameAcrossTemporalPatch() throws {
        let result = try QwenImagePreprocessor.patchify(
            planarRGB: [
                1, 2,
                3, 4,
            ],
            height: 2,
            width: 2,
            channelCount: 1,
            patchSize: 2,
            mergeSize: 1,
            temporalPatchSize: 2
        )

        XCTAssertEqual(result.pixelValuesShape, [1, 8])
        XCTAssertEqual(result.imageGridTHW, QwenImageGridTHW(temporal: 1, height: 1, width: 1))
        XCTAssertEqual(result.pixelValues, [1, 2, 3, 4, 1, 2, 3, 4])
    }

    func testFixtureImagePreprocessingMatchesPythonFixture() throws {
        let fixtureURL = try fixtureFileURL()
        let imageURL = try edgeTestFileURL(
            fromEnvironment: "EDGE_VLM_FIXTURE_IMAGE_PATH",
            purpose: "Qwen VLM image preprocessing fixture parity"
        )

        let fixture = try JSONDecoder().decode(
            VLMPreprocessingFixture.self,
            from: Data(contentsOf: fixtureURL)
        )
        let configuration = QwenImageProcessorConfiguration(
            imageProcessorType: "Qwen2VLImageProcessorFast",
            minPixels: 65_536,
            maxPixels: 16_777_216,
            patchSize: 16,
            mergeSize: 2,
            temporalPatchSize: 2,
            imageMean: [0.5, 0.5, 0.5],
            imageStd: [0.5, 0.5, 0.5]
        )

        let result = try QwenImagePreprocessor.preprocessImage(
            at: imageURL,
            configuration: configuration
        )

        XCTAssertEqual(result.pixelValuesShape, fixture.pixelValuesShape)
        XCTAssertEqual(result.imageGridTHW, QwenImageGridTHW(temporal: 1, height: 64, width: 48))
        XCTAssertEqual(
            result.imageGridTHW,
            QwenImageGridTHW(
                temporal: fixture.imageGridTHW[0][0],
                height: fixture.imageGridTHW[0][1],
                width: fixture.imageGridTHW[0][2]
            )
        )
        XCTAssertEqual(result.pixelValues.count, fixture.pixelValuesShape.reduce(1, *))
        XCTAssertEqual(result.pixelValues.min()!, fixture.pixelValuesMin, accuracy: 1e-6)
        XCTAssertEqual(result.pixelValues.max()!, fixture.pixelValuesMax, accuracy: 1e-6)

        let oneByteNormalizedTolerance = Float(2.0 / 255.0 + 1e-6)
        for (actual, expected) in zip(result.pixelValues.prefix(20), fixture.pixelValuesFirst20) {
            XCTAssertEqual(actual, expected, accuracy: oneByteNormalizedTolerance)
        }
        for (actual, expected) in zip(result.pixelValues.suffix(10), fixture.pixelValuesLast10) {
            XCTAssertEqual(actual, expected, accuracy: oneByteNormalizedTolerance)
        }

        let stats = meanAndStandardDeviation(result.pixelValues)
        XCTAssertEqual(stats.mean, fixture.pixelValuesMean, accuracy: 2e-4)
        XCTAssertEqual(stats.standardDeviation, fixture.pixelValuesStd, accuracy: 2e-4)
    }

    func testNonAlignedFixtureImagePreprocessingMatchesPythonFixtureWhenEnabled() throws {
        guard ProcessInfo.processInfo.environment["EDGE_RUN_VLM_NONALIGNED_FIXTURE"] == "1" else {
            throw XCTSkip("Set EDGE_RUN_VLM_NONALIGNED_FIXTURE=1 to run the large resize fixture.")
        }

        let fixtureURL = try fixtureFileURL(named: "vlm_preprocessing_fixture_nonaligned.json")
        let imageURL = try edgeTestFileURL(
            fromEnvironment: "EDGE_VLM_NONALIGNED_FIXTURE_IMAGE_PATH",
            purpose: "Qwen VLM non-aligned image preprocessing fixture parity"
        )

        let fixture = try JSONDecoder().decode(
            VLMNonAlignedPreprocessingFixture.self,
            from: Data(contentsOf: fixtureURL)
        )
        let configuration = QwenImageProcessorConfiguration(
            imageProcessorType: "Qwen2VLImageProcessorFast",
            minPixels: 65_536,
            maxPixels: 16_777_216,
            patchSize: 16,
            mergeSize: 2,
            temporalPatchSize: 2,
            imageMean: [0.5, 0.5, 0.5],
            imageStd: [0.5, 0.5, 0.5]
        )

        let result = try QwenImagePreprocessor.preprocessImage(
            at: imageURL,
            configuration: configuration
        )

        XCTAssertEqual(result.pixelValuesShape, fixture.pixelValuesShape)
        XCTAssertEqual(
            result.imageGridTHW,
            QwenImageGridTHW(
                temporal: fixture.imageGridTHW[0][0],
                height: fixture.imageGridTHW[0][1],
                width: fixture.imageGridTHW[0][2]
            )
        )
        XCTAssertEqual(result.pixelValues.count, fixture.pixelValuesShape.reduce(1, *))
        XCTAssertEqual(result.pixelValues.min()!, fixture.pixelValuesMin, accuracy: 1e-6)
        XCTAssertEqual(result.pixelValues.max()!, fixture.pixelValuesMax, accuracy: 1e-6)

        let oneByteNormalizedTolerance = Float(2.0 / 255.0 + 1e-6)
        for (actual, expected) in zip(result.pixelValues.prefix(10), fixture.pixelValuesFirst10) {
            XCTAssertEqual(actual, expected, accuracy: oneByteNormalizedTolerance)
        }
        for (actual, expected) in zip(result.pixelValues.suffix(10), fixture.pixelValuesLast10) {
            XCTAssertEqual(actual, expected, accuracy: oneByteNormalizedTolerance)
        }

        let stats = meanAndStandardDeviation(result.pixelValues)
        XCTAssertEqual(stats.mean, fixture.pixelValuesMean, accuracy: 5e-4)
        XCTAssertEqual(stats.standardDeviation, fixture.pixelValuesStd, accuracy: 5e-4)
    }

    func testPreflightParsesNestedProcessorSize() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("edge-engine-qwen-vlm-size-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try writeText(
            """
            {
              "model_type": "qwen3_5",
              "image_token_id": 248056,
              "vision_start_token_id": 248053,
              "vision_end_token_id": 248054,
              "text_config": {
                "model_type": "qwen3_5",
                "vocab_size": 3,
                "hidden_size": 2,
                "intermediate_size": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 1,
                "linear_num_value_heads": 1,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 1,
                "linear_value_head_dim": 1,
                "linear_conv_kernel_dim": 4,
                "context_length": 8,
                "rms_norm_eps": 0.000001,
                "rope_theta": 10000,
                "partial_rotary_factor": 0.25,
                "layer_types": ["full_attention", "linear_attention"]
              },
              "vision_config": {
                "hidden_size": 4,
                "intermediate_size": 8,
                "depth": 2,
                "num_heads": 2,
                "patch_size": 16,
                "spatial_merge_size": 2
              }
            }
            """,
            to: root.appendingPathComponent("config.json")
        )
        try writeText(
            """
            {
              "image_processor_type": "Qwen2VLImageProcessorFast",
              "size": {
                "shortest_edge": 65536,
                "longest_edge": 16777216
              },
              "patch_size": 16,
              "merge_size": 2,
              "temporal_patch_size": 2,
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5]
            }
            """,
            to: root.appendingPathComponent("preprocessor_config.json")
        )
        try writeText(
            """
            {
              "added_tokens": [
                {"id": 248053, "content": "<|vision_start|>", "special": true},
                {"id": 248054, "content": "<|vision_end|>", "special": true},
                {"id": 248056, "content": "<|image_pad|>", "special": true}
              ]
            }
            """,
            to: root.appendingPathComponent("tokenizer.json")
        )
        try writeVLMIndex(
            weightMap: Dictionary(uniqueKeysWithValues: (
                minimalLanguageTensorNames() + ["vision_tower.patch_embed.proj.weight"]
            ).map { ($0, "model.safetensors") }),
            to: root
        )
        try writeSafeTensorsFile(
            tensorNames: minimalLanguageTensorNames() + ["vision_tower.patch_embed.proj.weight"],
            to: root.appendingPathComponent("model.safetensors")
        )

        let result = try QwenVLMBundlePreflightRunner.run(
            configuration: QwenVLMBundlePreflightConfiguration(modelRootURL: root)
        )

        XCTAssertEqual(result.plan.imageProcessorConfiguration.minPixels, 65_536)
        XCTAssertEqual(result.plan.imageProcessorConfiguration.maxPixels, 16_777_216)
        XCTAssertEqual(result.plan.imageProcessorConfiguration.patchSize, 16)
    }

    private func fixtureFileURL(named name: String = "vlm_preprocessing_fixture.json") throws -> URL {
        let testFile = URL(fileURLWithPath: #filePath)
        let testsRoot = testFile
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let fixture = testsRoot
            .appendingPathComponent("Fixtures")
            .appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: fixture.path) else {
            throw XCTSkip("Fixture JSON not available at \(fixture.path)")
        }
        return fixture
    }
}

private struct VLMPreprocessingFixture: Decodable {
    var pixelValuesShape: [Int]
    var pixelValuesFirst20: [Float]
    var pixelValuesLast10: [Float]
    var pixelValuesMean: Double
    var pixelValuesStd: Double
    var pixelValuesMin: Float
    var pixelValuesMax: Float
    var imageGridTHW: [[Int]]

    private enum CodingKeys: String, CodingKey {
        case pixelValuesShape = "pixel_values_shape"
        case pixelValuesFirst20 = "pixel_values_first_20"
        case pixelValuesLast10 = "pixel_values_last_10"
        case pixelValuesMean = "pixel_values_mean"
        case pixelValuesStd = "pixel_values_std"
        case pixelValuesMin = "pixel_values_min"
        case pixelValuesMax = "pixel_values_max"
        case imageGridTHW = "image_grid_thw"
    }
}

private struct VLMNonAlignedPreprocessingFixture: Decodable {
    var originalSize: [Int]
    var pixelValuesShape: [Int]
    var pixelValuesFirst10: [Float]
    var pixelValuesLast10: [Float]
    var pixelValuesMean: Double
    var pixelValuesStd: Double
    var pixelValuesMin: Float
    var pixelValuesMax: Float
    var imageGridTHW: [[Int]]

    private enum CodingKeys: String, CodingKey {
        case originalSize = "original_size"
        case pixelValuesShape = "pixel_values_shape"
        case pixelValuesFirst10 = "pixel_values_first_10"
        case pixelValuesLast10 = "pixel_values_last_10"
        case pixelValuesMean = "pixel_values_mean"
        case pixelValuesStd = "pixel_values_std"
        case pixelValuesMin = "pixel_values_min"
        case pixelValuesMax = "pixel_values_max"
        case imageGridTHW = "image_grid_thw"
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

private func writeText(_ text: String, to url: URL) throws {
    try text.data(using: .utf8)!.write(to: url)
}

private func writeVLMIndex(weightMap: [String: String], to root: URL) throws {
    let payload: [String: Any] = ["weight_map": weightMap]
    let data = try JSONSerialization.data(
        withJSONObject: payload,
        options: [.prettyPrinted, .sortedKeys]
    )
    try data.write(to: root.appendingPathComponent("model.safetensors.index.json"))
}

private func writeSafeTensorsFile(tensorNames: [String], to url: URL) throws {
    var offset = 0
    var header: [String: Any] = [:]
    for name in tensorNames {
        header[name] = [
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [offset, offset + 4],
        ]
        offset += 4
    }
    let headerData = try JSONSerialization.data(
        withJSONObject: header,
        options: [.sortedKeys]
    )
    var data = Data()
    var headerLength = UInt64(headerData.count).littleEndian
    withUnsafeBytes(of: &headerLength) { data.append(contentsOf: $0) }
    data.append(headerData)
    data.append(Data(repeating: 0, count: offset))
    try data.write(to: url)
}

private func minimalLanguageTensorNames() -> [String] {
    [
        "language_model.model.embed_tokens.weight",
        "language_model.model.norm.weight",
        "language_model.lm_head.weight",
        "language_model.model.layers.0.input_layernorm.weight",
        "language_model.model.layers.0.post_attention_layernorm.weight",
        "language_model.model.layers.0.mlp.gate_proj.weight",
        "language_model.model.layers.0.mlp.up_proj.weight",
        "language_model.model.layers.0.mlp.down_proj.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.0.self_attn.k_proj.weight",
        "language_model.model.layers.0.self_attn.v_proj.weight",
        "language_model.model.layers.0.self_attn.o_proj.weight",
        "language_model.model.layers.0.self_attn.q_norm.weight",
        "language_model.model.layers.0.self_attn.k_norm.weight",
        "language_model.model.layers.1.input_layernorm.weight",
        "language_model.model.layers.1.post_attention_layernorm.weight",
        "language_model.model.layers.1.mlp.gate_proj.weight",
        "language_model.model.layers.1.mlp.up_proj.weight",
        "language_model.model.layers.1.mlp.down_proj.weight",
        "language_model.model.layers.1.linear_attn.A_log",
        "language_model.model.layers.1.linear_attn.conv1d.weight",
        "language_model.model.layers.1.linear_attn.dt_bias",
        "language_model.model.layers.1.linear_attn.in_proj_a.weight",
        "language_model.model.layers.1.linear_attn.in_proj_b.weight",
        "language_model.model.layers.1.linear_attn.in_proj_qkv.weight",
        "language_model.model.layers.1.linear_attn.in_proj_z.weight",
        "language_model.model.layers.1.linear_attn.norm.weight",
        "language_model.model.layers.1.linear_attn.out_proj.weight",
    ]
}
