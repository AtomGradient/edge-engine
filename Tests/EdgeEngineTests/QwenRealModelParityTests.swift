// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import XCTest
@testable import EdgeEngine

final class QwenRealModelParityTests: XCTestCase {
    func testRealQwen35FirstGDNLayerMatchesCmlxWhenEnabled() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["EDGE_RUN_REAL_QWEN35_PARITY"] == "1" else {
            throw XCTSkip("Set EDGE_RUN_REAL_QWEN35_PARITY=1 to run real Qwen3.5 parity diagnostics.")
        }

        let modelPath = try edgeTestDirectoryURL(
            fromEnvironment: "EDGE_QWEN35_MODEL_PATH",
            purpose: "real Qwen3.5 parity diagnostics",
            requiredFileName: "config.json"
        ).path
        let runtime = try EdgeMetalRuntime(
            configuration: MetalRuntimeConfiguration(maxOpsPerCommandBuffer: 50, maxMBPerCommandBuffer: 50)
        )
        let executor = try MetalKernelExecutor(runtime: runtime)
        let bundleIndex = try QwenModelBundleIndex.load(
            from: URL(fileURLWithPath: modelPath),
            family: .qwen35
        )
        let model = try QwenHybridModelReference.loadHuggingFaceLayout(
            weightStore: QwenModelWeightStore(bundleIndex: bundleIndex),
            runtime: runtime
        )
        guard case .quantizedGDN(let layer) = model.decoderLayers[0] else {
            XCTFail("Expected layer 0 to be a quantized GDN layer.")
            return
        }

        let hiddenStates = try model.embeddings.hiddenStates(tokenIds: [0], executor: executor)
        let cacheShape = try QwenGDNCacheShape.shape(for: bundleIndex.architecture, layerIndex: 0)
        let convState = try EdgeTensor(
            float32: Array(repeating: 0, count: cacheShape.convStateElementCount),
            shape: cacheShape.convStateTensorShape,
            runtime: runtime
        )
        let recurrentState = try EdgeTensor(
            float32: Array(repeating: 0, count: cacheShape.recurrentStateElementCount),
            shape: cacheShape.recurrentStateTensorShape,
            runtime: runtime
        )

        let expected = try layer.outputTensor(
            hiddenStates: hiddenStates,
            convState: convState,
            recurrentState: recurrentState,
            executor: executor
        )
        let cmlx = try QwenCmlxLazyDecodeSession(model: model, runtime: runtime)
        let actual = try cmlx.session.evalQuantizedGDNDecodeLayer(
            input: hiddenStates,
            layerIndex: 0,
            convState: convState,
            recurrentState: recurrentState
        )

        let hiddenError = try NumericComparison.maxAbsoluteError(
            actual.hiddenStates.readFloat32(),
            expected.hiddenStates.readFloat32()
        )
        let convError = try NumericComparison.maxAbsoluteError(
            actual.nextConvState.readFloat32(),
            expected.nextConvState.readFloat32()
        )
        let recurrentError = try NumericComparison.maxAbsoluteError(
            actual.nextRecurrentState.readFloat32(),
            expected.nextRecurrentState.readFloat32()
        )

        XCTAssertLessThan(hiddenError, 0.2)
        XCTAssertLessThan(convError, 0.05)
        XCTAssertLessThan(recurrentError, 0.05)

        var hiddenBeforeFirstAttention = expected.hiddenStates
        for layerIndex in 1...2 {
            guard case .quantizedGDN(let gdnLayer) = model.decoderLayers[layerIndex] else {
                XCTFail("Expected layer \(layerIndex) to be a quantized GDN layer.")
                return
            }
            let shape = try QwenGDNCacheShape.shape(
                for: bundleIndex.architecture,
                layerIndex: layerIndex
            )
            let zeroConv = try EdgeTensor(
                float32: Array(repeating: 0, count: shape.convStateElementCount),
                shape: shape.convStateTensorShape,
                runtime: runtime
            )
            let zeroRecurrent = try EdgeTensor(
                float32: Array(repeating: 0, count: shape.recurrentStateElementCount),
                shape: shape.recurrentStateTensorShape,
                runtime: runtime
            )
            hiddenBeforeFirstAttention = try gdnLayer.outputTensor(
                hiddenStates: hiddenBeforeFirstAttention,
                convState: zeroConv,
                recurrentState: zeroRecurrent,
                executor: executor
            ).hiddenStates
        }

        guard case .quantizedFullAttention(let attentionLayer) = model.decoderLayers[3] else {
            XCTFail("Expected layer 3 to be a quantized full-attention layer.")
            return
        }
        let kvCache = try QwenKVCache(
            shape: QwenKVCacheShape.shape(for: bundleIndex.architecture, layerIndex: 3, capacity: 1),
            runtime: runtime
        )
        let expectedAttention = try attentionLayer.outputTensor(
            hiddenStates: hiddenBeforeFirstAttention,
            executor: executor,
            positionOffset: 0,
            kvCache: kvCache
        )
        let actualAttention = try cmlx.session.evalQuantizedFullAttentionDecodeLayer(
            input: hiddenBeforeFirstAttention,
            layerIndex: 3,
            positionOffset: 0
        )
        let attentionError = try NumericComparison.maxAbsoluteError(
            actualAttention.readFloat32(),
            expectedAttention.readFloat32()
        )
        XCTAssertLessThan(attentionError, 0.3)
    }
}
