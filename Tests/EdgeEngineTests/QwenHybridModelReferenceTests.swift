// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Testing
@testable import EdgeEngine

@Test func qwenHybridModelReferenceRunsForwardWithHybridCaches() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeHybridModelArchitecture()
    let caches = try QwenHybridDecoderCaches(
        architecture: architecture,
        runtime: runtime,
        kvCapacity: 4
    )
    let model = try makeHybridModelReference(
        architecture: architecture,
        runtime: runtime
    )

    let logits = try model.logits(
        tokenIds: [0],
        caches: caches,
        executor: executor
    )

    #expect(logits.shape == EdgeTensorShape([1, 3]))
    #expect(try caches.kvCache(layerIndex: 0).tokenCount == 1)
    #expect(try caches.gdnCache(layerIndex: 1).tokenPosition == 1)
    #expect(try caches.tokenPosition() == 1)
}

@Test func qwenHybridModelReferenceRejectsInvalidLayerPlan() throws {
    let runtime = try EdgeMetalRuntime()
    let architecture = try makeHybridModelArchitecture()
    let model = try makeHybridModelReference(
        architecture: architecture,
        runtime: runtime
    )
    var rejectedCount = false
    var rejectedKind = false

    do {
        _ = try QwenHybridModelReference(
            architecture: architecture,
            embeddings: model.embeddings,
            decoderLayers: [model.decoderLayers[0]],
            outputWeights: model.outputWeights
        )
    } catch QwenHybridModelReferenceError.layerCountMismatch(expected: 2, actual: 1) {
        rejectedCount = true
    }

    do {
        _ = try QwenHybridModelReference(
            architecture: architecture,
            embeddings: model.embeddings,
            decoderLayers: [model.decoderLayers[0], model.decoderLayers[0]],
            outputWeights: model.outputWeights
        )
    } catch QwenHybridModelReferenceError.layerIndexMismatch(expected: 1, actual: 0) {
        rejectedKind = true
    }

    #expect(rejectedCount)
    #expect(rejectedKind)
}

@Test func qwenHybridModelReferenceRejectsEmptyTokenInput() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeHybridModelArchitecture()
    let model = try makeHybridModelReference(
        architecture: architecture,
        runtime: runtime
    )
    let caches = try QwenHybridDecoderCaches(architecture: architecture, runtime: runtime)
    var rejected = false

    do {
        _ = try model.logits(tokenIds: [], caches: caches, executor: executor)
    } catch QwenHybridModelReferenceError.emptyTokenIds {
        rejected = true
    }

    #expect(rejected)
}

@Test func qwenHybridModelReferenceCanProjectOnlyLastTokenLogits() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeHybridModelArchitecture()
    let model = try makeHybridModelReference(
        architecture: architecture,
        runtime: runtime
    )
    let fullCaches = try QwenHybridDecoderCaches(
        architecture: architecture,
        runtime: runtime,
        kvCapacity: 4
    )
    let lastOnlyCaches = try QwenHybridDecoderCaches(
        architecture: architecture,
        runtime: runtime,
        kvCapacity: 4
    )

    let fullLogits = try model.logits(
        tokenIds: [0, 1],
        caches: fullCaches,
        executor: executor
    )
    let lastOnlyLogits = try model.lastTokenLogits(
        tokenIds: [0, 1],
        caches: lastOnlyCaches,
        executor: executor
    )

    let vocabularySize = architecture.vocabularySize
    let fullValues = try fullLogits.readFloat32()
    let lastOnlyValues = try lastOnlyLogits.readFloat32()
    let fullLastRow = Array(fullValues[vocabularySize..<(2 * vocabularySize)])
    let error = try NumericComparison.maxAbsoluteError(fullLastRow, lastOnlyValues)

    #expect(fullLogits.shape == EdgeTensorShape([2, 3]))
    #expect(lastOnlyLogits.shape == EdgeTensorShape([1, 3]))
    #expect(try fullCaches.tokenPosition() == 2)
    #expect(try lastOnlyCaches.tokenPosition() == 2)
    #expect(error < 1e-4)
}

@Test func qwenHybridModelReferenceCapturesLastTokenHiddenStateAtLayer() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeHybridModelArchitecture()
    let model = try makeHybridModelReference(
        architecture: architecture,
        runtime: runtime
    )
    let caches = try QwenHybridDecoderCaches(
        architecture: architecture,
        runtime: runtime,
        kvCapacity: 4
    )

    let hidden = try model.lastTokenHiddenState(
        tokenIds: [0, 1],
        targetLayer: 0,
        caches: caches,
        executor: executor
    )

    #expect(hidden.shape == EdgeTensorShape([1, architecture.hiddenSize]))
    #expect(try caches.kvCache(layerIndex: 0).tokenCount == 2)
    #expect(try caches.gdnCache(layerIndex: 1).tokenPosition == 0)
}

@Test func qwenHybridModelReferenceLoadsHuggingFaceLayoutAndRunsForward() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-hybrid-model-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let architecture = try makeHybridModelArchitecture()
    let index = try QwenModelBundleIndex(
        rootURL: root,
        architecture: architecture,
        weightMap: hybridModelWeightMap()
    )
    try writeHybridModelSafeTensorsShard(
        tensors: hybridModelTensorEntries(),
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let model = try QwenHybridModelReference.loadHuggingFaceLayout(
        weightStore: QwenModelWeightStore(bundleIndex: index),
        runtime: runtime
    )
    let caches = try QwenHybridDecoderCaches(
        architecture: architecture,
        runtime: runtime,
        kvCapacity: 4
    )

    let logits = try model.logits(
        tokenIds: [0],
        caches: caches,
        executor: executor
    )

    #expect(model.decoderLayers.map(\.kind) == [.fullAttention, .gdn])
    guard case .float = model.outputWeights else {
        Issue.record("Model output should load float lm_head weights without quantization.")
        return
    }
    #expect(logits.shape == EdgeTensorShape([1, 3]))
    #expect(try caches.kvCache(layerIndex: 0).tokenCount == 1)
    #expect(try caches.gdnCache(layerIndex: 1).tokenPosition == 1)
    #expect(try caches.tokenPosition() == 1)
}

@Test func qwenBundleSmokeRunnerLoadsBundleAndGeneratesGreedyToken() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-smoke-bundle-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try writeHybridModelConfig(to: root)
    try writeHybridModelIndex(weightMap: hybridModelWeightMap(), to: root)
    try writeHybridModelSafeTensorsShard(
        tensors: hybridModelTensorEntries(),
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )

    let result = try QwenBundleSmokeRunner.run(
        configuration: QwenBundleSmokeConfiguration(
            modelRootURL: root,
            promptTokenIds: [0],
            maxNewTokenCount: 1,
            runtimeConfiguration: MetalRuntimeConfiguration(
                maxOpsPerCommandBuffer: 3,
                maxMBPerCommandBuffer: 1
            )
        )
    )

    #expect(result.modelPrefix == "model")
    #expect(result.family == .qwen35)
    #expect(result.vocabularySize == 3)
    #expect(result.hiddenSize == 2)
    #expect(result.layerCount == 2)
    #expect(result.fullAttentionLayerCount == 1)
    #expect(result.gdnLayerCount == 1)
    #expect(result.outputKind == .float)
    #expect(!result.usesTiedEmbeddings)
    #expect(result.promptTokenIds == [0])
    #expect(result.maxNewTokenCount == 1)
    #expect(result.generatedTokenIds.count == 1)
    #expect(result.steps.count == 1)
    #expect(result.prefillLogitsShape == [1, 3])
    #expect(result.prefillTopLogits.count == 3)
    for (lhs, rhs) in zip(result.prefillTopLogits, result.prefillTopLogits.dropFirst()) {
        #expect(lhs.logit >= rhs.logit)
        if lhs.logit == rhs.logit {
            #expect(lhs.tokenId < rhs.tokenId)
        }
    }
    if let topLogit = result.prefillTopLogits.first,
       let firstStep = result.steps.first {
        #expect(topLogit.tokenId == firstStep.tokenId)
        #expect(abs(topLogit.logit - firstStep.logit) < 0.0001)
        #expect(firstStep.topLogits == result.prefillTopLogits)
    } else {
        Issue.record("Smoke result should include prefill logits and one generated step.")
    }
    #expect(result.cacheTokenPosition == 2)
    #expect(result.runtime.maxOpsPerCommandBuffer == 3)
    #expect(result.runtime.maxMBPerCommandBuffer == 1)
    #expect(result.runtime.contextLengthHint == 2)
    #expect(result.runtime.effectiveMaxOpsPerCommandBuffer == 3)
    #expect(result.runtime.quantizedBufferCacheLimitBytes == nil)
    #expect(!result.runtime.releaseQuantizedHostStorageAfterUpload)
    #expect(result.runtime.quantizedBufferCachedByteCount == result.runtime.quantizedBufferUploadedByteCount)
    #expect(result.runtime.quantizedHostStorageReleasedByteCount == 0)
    #expect(result.runtime.quantizedHostStorageReleaseCount == 0)
    #expect((result.memory.residentAtStartBytes ?? 0) > 0)
    #expect((result.memory.residentAfterModelLoadBytes ?? 0) > 0)
    #expect((result.memory.residentAfterPrefillBytes ?? 0) > 0)
    #expect((result.memory.peakResidentBytes ?? 0) > 0)

    let encoded = try JSONEncoder().encode(result)
    let decoded = try JSONDecoder().decode(QwenBundleSmokeResult.self, from: encoded)
    #expect(decoded.prefillTopLogits == result.prefillTopLogits)
    #expect(decoded.steps.first?.topLogits == result.steps.first?.topLogits)

    var legacyObject = try #require(
        JSONSerialization.jsonObject(with: encoded) as? [String: Any]
    )
    legacyObject.removeValue(forKey: "prefillTopLogits")
    legacyObject.removeValue(forKey: "memory")
    if var legacySteps = legacyObject["steps"] as? [[String: Any]] {
        for index in legacySteps.indices {
            legacySteps[index].removeValue(forKey: "topLogits")
        }
        legacyObject["steps"] = legacySteps
    }
    let legacyData = try JSONSerialization.data(withJSONObject: legacyObject)
    let legacyDecoded = try JSONDecoder().decode(QwenBundleSmokeResult.self, from: legacyData)
    #expect(legacyDecoded.prefillTopLogits.isEmpty)
    #expect(legacyDecoded.steps.first?.topLogits.isEmpty == true)
    #expect(legacyDecoded.memory == .empty)
}

@Test func qwenBundleSmokeRunnerCanUseCmlxDecodeStepBackend() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-cmlx-smoke-bundle-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try writeCmlxSmokeModelConfig(to: root)
    try writeHybridModelIndex(weightMap: cmlxSmokeWeightMap(), to: root)
    try writeCmlxSmokeSafeTensorsShard(
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )

    let result = try QwenBundleSmokeRunner.run(
        configuration: QwenBundleSmokeConfiguration(
            modelRootURL: root,
            promptTokenIds: [0],
            maxNewTokenCount: 1,
            runtimeConfiguration: MetalRuntimeConfiguration(
                maxOpsPerCommandBuffer: 3,
                maxMBPerCommandBuffer: 1
            ),
            prefillTopLogitCount: 0,
            decodeBackend: .cmlxDecodeStep
        )
    )

    #expect(result.decodeBackend == .cmlxDecodeStep)
    #expect(result.outputKind == .quantized)
    #expect(result.vocabularySize == 16)
    #expect(result.hiddenSize == 32)
    #expect(result.generatedTokenIds.count == 1)
    #expect(result.steps.count == 1)
    #expect(result.prefillLogitsShape.isEmpty)
    #expect(result.prefillTopLogits.isEmpty)
    #expect(result.steps.first?.topLogits.isEmpty == true)
    #expect(result.cacheTokenPosition == 2)
    #expect(result.prefillDurationSeconds != nil)
    #expect(result.decodeDurationSeconds != nil)
    #expect((result.decodeTokensPerSecond ?? 0) > 0)
    #expect((result.memory.residentAfterPrefillBytes ?? 0) > 0)
    #expect((result.memory.residentAfterDecodeBytes ?? 0) > 0)
}

@Test func qwenBundlePreflightRunnerSummarizesBundleWithoutLoadingWeights() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-preflight-bundle-\(UUID().uuidString)")
    let weightMap = hybridModelWeightMap()
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try writeHybridModelConfig(to: root)
    try writeHybridModelIndex(weightMap: weightMap, to: root)
    try writeHybridModelSafeTensorsShard(
        tensors: hybridModelTensorEntries(),
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )

    let result = try QwenBundlePreflightRunner.run(
        configuration: QwenBundlePreflightConfiguration(modelRootURL: root)
    )

    #expect(result.modelPrefix == "model")
    #expect(result.family == .qwen35)
    #expect(result.vocabularySize == 3)
    #expect(result.hiddenSize == 2)
    #expect(result.layerCount == 2)
    #expect(result.fullAttentionLayerCount == 1)
    #expect(result.gdnLayerCount == 1)
    #expect(result.outputWeightName == "model.lm_head.weight")
    #expect(result.outputKind == .float)
    #expect(!result.usesTiedEmbeddings)
    #expect(result.readShardHeaders)
    #expect(result.passesPreflight)
    #expect(result.failureReasons.isEmpty)
    #expect(result.missingRequiredTensorNames.isEmpty)
    #expect(result.weightMapTensorCount == weightMap.count)
    #expect(result.shardCount == 1)
    #expect(result.shardHeaderTensorCount == weightMap.count)
    #expect(result.dtypeCounts == ["F32": weightMap.count])
    #expect(result.modelLevelTensorDTypes["model.embed_tokens.weight"] == "F32")
    #expect(result.modelLevelTensorDTypes["model.norm.weight"] == "F32")
    #expect(result.modelLevelTensorDTypes["model.lm_head.weight"] == "F32")
    #expect(result.quantizedGroupCount > 0)
    #expect(result.quantizedGroupsWithScales == 0)
    #expect(result.quantizedGroupsWithBiases == 0)
    #expect(result.tensorsMissingFromShardHeaders.isEmpty)
    #expect(result.shardHeaderTensorNamesNotInIndex.isEmpty)
    #expect(result.shards.map(\.fileName) == ["model-00001-of-00001.safetensors"])

    let encoded = try JSONEncoder().encode(result)
    let decoded = try JSONDecoder().decode(QwenBundlePreflightResult.self, from: encoded)
    #expect(decoded == result)

    var legacyObject = try #require(
        JSONSerialization.jsonObject(with: encoded) as? [String: Any]
    )
    legacyObject.removeValue(forKey: "readShardHeaders")
    legacyObject.removeValue(forKey: "passesPreflight")
    legacyObject.removeValue(forKey: "failureReasons")
    let legacyData = try JSONSerialization.data(withJSONObject: legacyObject)
    let legacyDecoded = try JSONDecoder().decode(QwenBundlePreflightResult.self, from: legacyData)
    #expect(legacyDecoded.readShardHeaders)
    #expect(legacyDecoded.passesPreflight)
    #expect(legacyDecoded.failureReasons.isEmpty)
}

@Test func qwenBundlePreflightRunnerReportsFailureReasons() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-preflight-failures-\(UUID().uuidString)")
    var weightMap = hybridModelWeightMap()
    weightMap.removeValue(forKey: "model.layers.0.self_attn.k_norm.weight")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try writeHybridModelConfig(to: root)
    try writeHybridModelIndex(weightMap: weightMap, to: root)
    var tensors = hybridModelTensorEntries()
    tensors["model.layers.0.self_attn.k_norm.weight"] = nil
    tensors["model.layers.0.self_attn.q_proj.weight"] = nil
    tensors["model.extra.unused.weight"] = ([1], [42])
    try writeHybridModelSafeTensorsShard(
        tensors: tensors,
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )

    let result = try QwenBundlePreflightRunner.run(
        configuration: QwenBundlePreflightConfiguration(modelRootURL: root)
    )

    #expect(!result.passesPreflight)
    #expect(result.failureReasons == [
        .missingRequiredTensors,
        .indexedTensorsMissingFromShardHeaders,
    ])
    #expect(result.missingRequiredTensorNames == ["model.layers.0.self_attn.k_norm.weight"])
    #expect(result.tensorsMissingFromShardHeaders == ["model.layers.0.self_attn.q_proj.weight"])
    #expect(result.shardHeaderTensorNamesNotInIndex == ["model.extra.unused.weight"])
}

@Test func qwenBundlePreflightRunnerReportsMissingRequiredTensorReasonOnly() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-preflight-missing-required-\(UUID().uuidString)")
    var weightMap = hybridModelWeightMap()
    weightMap.removeValue(forKey: "model.layers.0.self_attn.k_norm.weight")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try writeHybridModelConfig(to: root)
    try writeHybridModelIndex(weightMap: weightMap, to: root)
    var tensors = hybridModelTensorEntries()
    tensors["model.layers.0.self_attn.k_norm.weight"] = nil
    try writeHybridModelSafeTensorsShard(
        tensors: tensors,
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )

    let result = try QwenBundlePreflightRunner.run(
        configuration: QwenBundlePreflightConfiguration(modelRootURL: root)
    )

    #expect(!result.passesPreflight)
    #expect(result.failureReasons == [.missingRequiredTensors])
    #expect(result.missingRequiredTensorNames == ["model.layers.0.self_attn.k_norm.weight"])
    #expect(result.tensorsMissingFromShardHeaders.isEmpty)
    #expect(result.shardHeaderTensorNamesNotInIndex.isEmpty)
}

@Test func qwenBundlePreflightRunnerReportsShardHeaderReasonOnly() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-preflight-missing-header-\(UUID().uuidString)")
    let weightMap = hybridModelWeightMap()
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try writeHybridModelConfig(to: root)
    try writeHybridModelIndex(weightMap: weightMap, to: root)
    var tensors = hybridModelTensorEntries()
    tensors["model.layers.0.self_attn.q_proj.weight"] = nil
    try writeHybridModelSafeTensorsShard(
        tensors: tensors,
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )

    let result = try QwenBundlePreflightRunner.run(
        configuration: QwenBundlePreflightConfiguration(modelRootURL: root)
    )

    #expect(!result.passesPreflight)
    #expect(result.failureReasons == [.indexedTensorsMissingFromShardHeaders])
    #expect(result.missingRequiredTensorNames.isEmpty)
    #expect(result.tensorsMissingFromShardHeaders == ["model.layers.0.self_attn.q_proj.weight"])
    #expect(result.shardHeaderTensorNamesNotInIndex.isEmpty)
}

@Test func qwenBundlePreflightRunnerCanSkipShardHeaderValidation() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-preflight-no-headers-\(UUID().uuidString)")
    let weightMap = hybridModelWeightMap()
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try writeHybridModelConfig(to: root)
    try writeHybridModelIndex(weightMap: weightMap, to: root)

    let result = try QwenBundlePreflightRunner.run(
        configuration: QwenBundlePreflightConfiguration(
            modelRootURL: root,
            readShardHeaders: false
        )
    )

    #expect(!result.readShardHeaders)
    #expect(result.passesPreflight)
    #expect(result.failureReasons.isEmpty)
    #expect(result.shardHeaderTensorCount == 0)
    #expect(result.tensorsMissingFromShardHeaders.isEmpty)
    #expect(result.shardHeaderTensorNamesNotInIndex.isEmpty)
    #expect(result.shards.isEmpty)
}

@Test func qwenHybridModelReferenceLoadsQuantizedDecoderLayersAndRunsForward() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-quantized-hybrid-model-\(UUID().uuidString)")
    let expectedRoot = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-float-hybrid-model-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: expectedRoot, withIntermediateDirectories: true)
    let architecture = try makeQuantizedHybridModelArchitecture()
    let index = try QwenModelBundleIndex(
        rootURL: root,
        architecture: architecture,
        weightMap: quantizedHybridModelWeightMap()
    )
    let expectedIndex = try QwenModelBundleIndex(
        rootURL: expectedRoot,
        architecture: try makeHybridModelArchitecture(),
        weightMap: hybridModelWeightMap()
    )
    try writeQuantizedHybridModelSafeTensorsShard(
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )
    try writeHybridModelSafeTensorsShard(
        tensors: hybridModelTensorEntries(),
        to: expectedRoot.appendingPathComponent("model-00001-of-00001.safetensors")
    )
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let expectedModel = try QwenHybridModelReference.loadHuggingFaceLayout(
        weightStore: QwenModelWeightStore(bundleIndex: expectedIndex),
        runtime: runtime
    )
    let model = try QwenHybridModelReference.loadHuggingFaceLayout(
        weightStore: QwenModelWeightStore(bundleIndex: index),
        runtime: runtime
    )
    let expectedCaches = try QwenHybridDecoderCaches(
        architecture: expectedModel.architecture,
        runtime: runtime,
        kvCapacity: 4
    )
    let caches = try QwenHybridDecoderCaches(
        architecture: architecture,
        runtime: runtime,
        kvCapacity: 4
    )

    let expected = try expectedModel.logits(
        tokenIds: [0, 1],
        caches: expectedCaches,
        executor: executor
    )
    let actual = try model.logits(
        tokenIds: [0, 1],
        caches: caches,
        executor: executor
    )
    let error = try NumericComparison.maxAbsoluteError(
        try actual.readFloat32(),
        try expected.readFloat32()
    )

    guard case .quantizedFullAttention = model.decoderLayers[0] else {
        Issue.record("Layer 0 should load as quantized full attention.")
        return
    }
    guard case .quantizedGDN = model.decoderLayers[1] else {
        Issue.record("Layer 1 should load as quantized GDN.")
        return
    }
    guard case .quantized = model.outputWeights else {
        Issue.record("Model output should load quantized lm_head weights.")
        return
    }
    #expect(actual.shape == EdgeTensorShape([2, 3]))
    #expect(try caches.kvCache(layerIndex: 0).tokenCount == 2)
    #expect(try caches.gdnCache(layerIndex: 1).tokenPosition == 2)
    #expect(try caches.tokenPosition() == 2)
    #expect(error < 1e-4)
}

@Test func qwenCmlxLazyDecodeSessionCapturesLastHiddenWithoutAdvancingDecodeState() throws {
    guard ProcessInfo.processInfo.environment["EDGE_RUN_CMLX_CAPTURE_UNIT"] == "1" else {
        return
    }
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-cmlx-capture-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let architecture = try makeQuantizedHybridModelArchitecture()
    let index = try QwenModelBundleIndex(
        rootURL: root,
        architecture: architecture,
        weightMap: quantizedHybridModelWeightMap()
    )
    try writeQuantizedHybridModelSafeTensorsShard(
        to: root.appendingPathComponent("model-00001-of-00001.safetensors")
    )
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let model = try QwenHybridModelReference.loadHuggingFaceLayout(
        weightStore: QwenModelWeightStore(bundleIndex: index),
        runtime: runtime
    )
    let session = try QwenCmlxLazyDecodeSession(
        model: model,
        runtime: runtime,
        executor: executor,
        frogJumpLayerMask: UInt64(1) << 1
    )

    let captured = try session.captureLastHidden(tokenIDs: [0, 1], targetLayer: 0)

    #expect(captured.count == architecture.hiddenSize)
    #expect(captured.allSatisfy { $0.isFinite })
    #expect(session.tokenPosition == 0)
    #expect(session.decodedTokenCount == 0)

    _ = try session.prefill(tokenIDs: [0])
    let decodedBefore = session.decodedTokenCount
    let tokenPositionBefore = session.tokenPosition
    let capturedAfterPrefill = try session.captureLastHidden(
        tokenIDs: [1],
        targetLayer: 1
    )

    #expect(capturedAfterPrefill.count == architecture.hiddenSize)
    #expect(capturedAfterPrefill.allSatisfy { $0.isFinite })
    #expect(session.tokenPosition == tokenPositionBefore)
    #expect(session.decodedTokenCount == decodedBefore)
}

@Test func qwenHybridModelReferenceUsesFloatOutputWhenQuantizedCompanionIsMissing() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-quantized-hybrid-float-output-\(UUID().uuidString)")
    let expectedRoot = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-float-hybrid-output-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: expectedRoot, withIntermediateDirectories: true)
    let architecture = try makeQuantizedHybridModelArchitecture()
    let index = try QwenModelBundleIndex(
        rootURL: root,
        architecture: architecture,
        weightMap: quantizedHybridModelWeightMap(quantizedOutput: false)
    )
    let expectedIndex = try QwenModelBundleIndex(
        rootURL: expectedRoot,
        architecture: try makeHybridModelArchitecture(),
        weightMap: hybridModelWeightMap()
    )
    try writeQuantizedHybridModelSafeTensorsShard(
        to: root.appendingPathComponent("model-00001-of-00001.safetensors"),
        quantizedOutput: false
    )
    try writeHybridModelSafeTensorsShard(
        tensors: hybridModelTensorEntries(),
        to: expectedRoot.appendingPathComponent("model-00001-of-00001.safetensors")
    )
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let expectedModel = try QwenHybridModelReference.loadHuggingFaceLayout(
        weightStore: QwenModelWeightStore(bundleIndex: expectedIndex),
        runtime: runtime
    )
    let model = try QwenHybridModelReference.loadHuggingFaceLayout(
        weightStore: QwenModelWeightStore(bundleIndex: index),
        runtime: runtime
    )
    let expectedCaches = try QwenHybridDecoderCaches(
        architecture: expectedModel.architecture,
        runtime: runtime,
        kvCapacity: 4
    )
    let caches = try QwenHybridDecoderCaches(
        architecture: architecture,
        runtime: runtime,
        kvCapacity: 4
    )

    let expected = try expectedModel.logits(
        tokenIds: [0, 1],
        caches: expectedCaches,
        executor: executor
    )
    let actual = try model.logits(
        tokenIds: [0, 1],
        caches: caches,
        executor: executor
    )
    let error = try NumericComparison.maxAbsoluteError(
        try actual.readFloat32(),
        try expected.readFloat32()
    )

    guard case .float = model.outputWeights else {
        Issue.record("Model output should stay float when lm_head has no quantized companion.")
        return
    }
    #expect(actual.shape == EdgeTensorShape([2, 3]))
    #expect(error < 1e-4)
}

@Test func qwenHybridModelReferenceUsesFloatTiedOutputForQuantizedDecoderBundles() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-quantized-hybrid-tied-output-\(UUID().uuidString)")
    let expectedRoot = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-float-hybrid-tied-output-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: expectedRoot, withIntermediateDirectories: true)
    let architecture = try makeQuantizedHybridModelArchitecture()
    let index = try QwenModelBundleIndex(
        rootURL: root,
        architecture: architecture,
        weightMap: quantizedHybridModelWeightMap(includeLMHead: false, quantizedOutput: false)
    )
    let expectedIndex = try QwenModelBundleIndex(
        rootURL: expectedRoot,
        architecture: try makeHybridModelArchitecture(),
        weightMap: hybridModelWeightMap(includeLMHead: false)
    )
    try writeQuantizedHybridModelSafeTensorsShard(
        to: root.appendingPathComponent("model-00001-of-00001.safetensors"),
        quantizedOutput: false
    )
    try writeHybridModelSafeTensorsShard(
        tensors: hybridModelTensorEntries(),
        to: expectedRoot.appendingPathComponent("model-00001-of-00001.safetensors")
    )
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let expectedModel = try QwenHybridModelReference.loadHuggingFaceLayout(
        weightStore: QwenModelWeightStore(bundleIndex: expectedIndex),
        runtime: runtime
    )
    let model = try QwenHybridModelReference.loadHuggingFaceLayout(
        weightStore: QwenModelWeightStore(bundleIndex: index),
        runtime: runtime
    )
    let expectedCaches = try QwenHybridDecoderCaches(
        architecture: expectedModel.architecture,
        runtime: runtime,
        kvCapacity: 4
    )
    let caches = try QwenHybridDecoderCaches(
        architecture: architecture,
        runtime: runtime,
        kvCapacity: 4
    )

    let expected = try expectedModel.logits(
        tokenIds: [0, 1],
        caches: expectedCaches,
        executor: executor
    )
    let actual = try model.logits(
        tokenIds: [0, 1],
        caches: caches,
        executor: executor
    )
    let error = try NumericComparison.maxAbsoluteError(
        try actual.readFloat32(),
        try expected.readFloat32()
    )

    guard case .float = model.outputWeights else {
        Issue.record("Tied embedding output should use float output weights.")
        return
    }
    #expect(model.outputWeights.usesTiedEmbeddings)
    #expect(actual.shape == EdgeTensorShape([2, 3]))
    #expect(error < 1e-4)
}

@Test func qwenHybridModelReferenceUsesQuantizedTiedEmbeddingOutput() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-quantized-hybrid-quantized-tied-output-\(UUID().uuidString)")
    let expectedRoot = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-qwen-float-hybrid-quantized-tied-output-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: expectedRoot, withIntermediateDirectories: true)
    let architecture = try makeQuantizedHybridModelArchitecture()
    let index = try QwenModelBundleIndex(
        rootURL: root,
        architecture: architecture,
        weightMap: quantizedHybridModelWeightMap(
            includeLMHead: false,
            quantizedOutput: true,
            quantizedEmbeddings: true
        )
    )
    let expectedIndex = try QwenModelBundleIndex(
        rootURL: expectedRoot,
        architecture: try makeHybridModelArchitecture(),
        weightMap: hybridModelWeightMap(includeLMHead: false)
    )
    try writeQuantizedHybridModelSafeTensorsShard(
        to: root.appendingPathComponent("model-00001-of-00001.safetensors"),
        quantizedOutput: true,
        quantizedEmbeddings: true
    )
    try writeHybridModelSafeTensorsShard(
        tensors: hybridModelTensorEntries(),
        to: expectedRoot.appendingPathComponent("model-00001-of-00001.safetensors")
    )
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let expectedModel = try QwenHybridModelReference.loadHuggingFaceLayout(
        weightStore: QwenModelWeightStore(bundleIndex: expectedIndex),
        runtime: runtime
    )
    let model = try QwenHybridModelReference.loadHuggingFaceLayout(
        weightStore: QwenModelWeightStore(bundleIndex: index),
        runtime: runtime
    )
    let expectedCaches = try QwenHybridDecoderCaches(
        architecture: expectedModel.architecture,
        runtime: runtime,
        kvCapacity: 4
    )
    let caches = try QwenHybridDecoderCaches(
        architecture: architecture,
        runtime: runtime,
        kvCapacity: 4
    )

    let expected = try expectedModel.logits(
        tokenIds: [0, 1],
        caches: expectedCaches,
        executor: executor
    )
    let actual = try model.logits(
        tokenIds: [0, 1],
        caches: caches,
        executor: executor
    )
    let error = try NumericComparison.maxAbsoluteError(
        try actual.readFloat32(),
        try expected.readFloat32()
    )

    guard case .quantized(let outputWeights) = model.outputWeights else {
        Issue.record("Quantized tied embeddings should use quantized output weights.")
        return
    }
    guard case .quantized(let embeddingWeights) = model.embeddings.embeddings else {
        Issue.record("Quantized tied embeddings should keep quantized token storage.")
        return
    }
    #expect(model.outputWeights.usesTiedEmbeddings)
    #expect(embeddingWeights.storageIdentifier == outputWeights.lmHead.storageIdentifier)
    #expect(try model.embeddings.embeddings.readFloat32() == [
        1, 0,
        0, 1,
        1, 1,
    ])
    #expect(actual.shape == EdgeTensorShape([2, 3]))
    #expect(error < 1e-4)
}

@Test func qwenGreedyDecoderSelectsMaximumFromLastLogitsRow() throws {
    let runtime = try EdgeMetalRuntime()
    let logits = try EdgeTensor(
        float32: [
            9, 8, 7,
            -1, 4, 3,
        ],
        shape: EdgeTensorShape([2, 3]),
        runtime: runtime
    )

    let token = try QwenGreedyDecoder.nextToken(logits: logits)

    #expect(token == QwenGreedyToken(tokenId: 1, logit: 4))
}

@Test func qwenGreedyDecoderRejectsInvalidInputs() throws {
    let runtime = try EdgeMetalRuntime()
    let logits = try EdgeTensor(
        float32: [1, 2, 3],
        shape: EdgeTensorShape([3]),
        runtime: runtime
    )
    let architecture = try makeHybridModelArchitecture()
    let model = try makeHybridModelReference(architecture: architecture, runtime: runtime)
    let caches = try QwenHybridDecoderCaches(architecture: architecture, runtime: runtime)
    let executor = try MetalKernelExecutor(runtime: runtime)
    var rejectedLogits = false
    var rejectedCount = false

    do {
        _ = try QwenGreedyDecoder.nextToken(logits: logits)
    } catch QwenGreedyDecoderError.invalidLogitsShape(expected: [-1, -1], actual: [3]) {
        rejectedLogits = true
    }

    do {
        _ = try QwenGreedyDecoder.generateNextTokens(
            promptTokenIds: [0],
            model: model,
            caches: caches,
            executor: executor,
            maxTokenCount: -1
        )
    } catch QwenGreedyDecoderError.invalidMaxTokenCount(-1) {
        rejectedCount = true
    }

    #expect(rejectedLogits)
    #expect(rejectedCount)
}

@Test func qwenGreedyDecoderGeneratesWithHybridModelCaches() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeHybridModelArchitecture()
    let model = try makeHybridModelReference(architecture: architecture, runtime: runtime)
    let caches = try QwenHybridDecoderCaches(
        architecture: architecture,
        runtime: runtime,
        kvCapacity: 4
    )

    let generated = try QwenGreedyDecoder.generateNextTokens(
        promptTokenIds: [0],
        model: model,
        caches: caches,
        executor: executor,
        maxTokenCount: 2
    )

    #expect(generated.count == 2)
    #expect(generated.allSatisfy { $0 >= 0 && $0 < architecture.vocabularySize })
    #expect(try caches.kvCache(layerIndex: 0).tokenCount == 2)
    #expect(try caches.gdnCache(layerIndex: 1).tokenPosition == 2)
    #expect(try caches.tokenPosition() == 2)
}

@Test func qwenGreedyDecodeSessionPrefillsAndResumesAcrossCalls() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeHybridModelArchitecture()
    let model = try makeHybridModelReference(architecture: architecture, runtime: runtime)
    let caches = try QwenHybridDecoderCaches(
        architecture: architecture,
        runtime: runtime,
        kvCapacity: 5
    )
    let session = QwenGreedyDecodeSession(
        model: model,
        caches: caches,
        executor: executor
    )

    let promptLogits = try session.prefill(promptTokenIds: [0])
    let firstStep = try session.generateNextToken()
    let secondStep = try session.generateNextToken()

    #expect(promptLogits.shape == EdgeTensorShape([1, 3]))
    #expect(firstStep.token.tokenId >= 0 && firstStep.token.tokenId < architecture.vocabularySize)
    #expect(secondStep.token.tokenId >= 0 && secondStep.token.tokenId < architecture.vocabularySize)
    #expect(firstStep.cacheTokenPosition == 2)
    #expect(secondStep.cacheTokenPosition == 3)
    #expect(try session.tokenPosition() == 3)

    session.reset()
    #expect(try session.tokenPosition() == 0)
}

@Test func qwenGreedyDecodeSessionMatchesFullPrefillForSplitToken() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeHybridModelArchitecture()
    let model = try makeHybridModelReference(architecture: architecture, runtime: runtime)
    let fullCaches = try QwenHybridDecoderCaches(
        architecture: architecture,
        runtime: runtime,
        kvCapacity: 5
    )
    let splitCaches = try QwenHybridDecoderCaches(
        architecture: architecture,
        runtime: runtime,
        kvCapacity: 5
    )
    let session = QwenGreedyDecodeSession(
        model: model,
        caches: splitCaches,
        executor: executor
    )

    let fullLogits = try model.logits(
        tokenIds: [0, 1],
        caches: fullCaches,
        executor: executor
    )
    try session.prefill(promptTokenIds: [0])
    let splitLogits = try session.advance(with: 1)

    let vocabularySize = architecture.vocabularySize
    let fullValues = try fullLogits.readFloat32()
    let splitValues = try splitLogits.readFloat32()
    let fullLastRow = Array(fullValues[vocabularySize..<(2 * vocabularySize)])
    let error = try NumericComparison.maxAbsoluteError(fullLastRow, splitValues)

    #expect(fullLogits.shape == EdgeTensorShape([2, 3]))
    #expect(splitLogits.shape == EdgeTensorShape([1, 3]))
    #expect(try fullCaches.tokenPosition() == 2)
    #expect(try splitCaches.tokenPosition() == 2)
    #expect(error < 1e-4)
}

@Test func qwenGreedyDecodeSessionPrefillReturnsOnlyLastPromptLogits() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeHybridModelArchitecture()
    let model = try makeHybridModelReference(architecture: architecture, runtime: runtime)
    let fullCaches = try QwenHybridDecoderCaches(
        architecture: architecture,
        runtime: runtime,
        kvCapacity: 5
    )
    let sessionCaches = try QwenHybridDecoderCaches(
        architecture: architecture,
        runtime: runtime,
        kvCapacity: 5
    )
    let session = QwenGreedyDecodeSession(
        model: model,
        caches: sessionCaches,
        executor: executor
    )

    let fullLogits = try model.logits(
        tokenIds: [0, 1],
        caches: fullCaches,
        executor: executor
    )
    let sessionLogits = try session.prefill(promptTokenIds: [0, 1])

    let vocabularySize = architecture.vocabularySize
    let fullValues = try fullLogits.readFloat32()
    let sessionValues = try sessionLogits.readFloat32()
    let fullLastRow = Array(fullValues[vocabularySize..<(2 * vocabularySize)])
    let error = try NumericComparison.maxAbsoluteError(fullLastRow, sessionValues)

    #expect(fullLogits.shape == EdgeTensorShape([2, 3]))
    #expect(sessionLogits.shape == EdgeTensorShape([1, 3]))
    #expect(try fullCaches.tokenPosition() == 2)
    #expect(try sessionCaches.tokenPosition() == 2)
    #expect(error < 1e-4)
}

@Test func qwenGreedyDecodeSessionSupportsPrefillOnlyAndRejectsMissingLogits() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeHybridModelArchitecture()
    let model = try makeHybridModelReference(architecture: architecture, runtime: runtime)
    let caches = try QwenHybridDecoderCaches(
        architecture: architecture,
        runtime: runtime,
        kvCapacity: 5
    )
    let session = QwenGreedyDecodeSession(
        model: model,
        caches: caches,
        executor: executor
    )
    var rejectedMissingLogits = false
    var rejectedNegativeCount = false

    do {
        _ = try session.generateNextToken()
    } catch QwenGreedyDecodeSessionError.missingLogits {
        rejectedMissingLogits = true
    }

    do {
        _ = try session.generateNextTokens(
            promptTokenIds: [0, 1],
            maxTokenCount: -1
        )
    } catch QwenGreedyDecoderError.invalidMaxTokenCount(-1) {
        rejectedNegativeCount = true
    }

    let generated = try session.generateNextTokens(
        promptTokenIds: [0, 1],
        maxTokenCount: 0
    )
    session.invalidateCurrentLogits()
    var rejectedAfterInvalidate = false
    do {
        _ = try session.generateNextToken()
    } catch QwenGreedyDecodeSessionError.missingLogits {
        rejectedAfterInvalidate = true
    }

    #expect(rejectedMissingLogits)
    #expect(rejectedNegativeCount)
    #expect(generated.isEmpty)
    #expect(try session.tokenPosition() == 2)
    #expect(rejectedAfterInvalidate)
}

@Test func qwenGreedyDecodeSessionSampledGreedyMatchesGreedyStep() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeHybridModelArchitecture()
    let model = try makeHybridModelReference(architecture: architecture, runtime: runtime)
    let greedySession = QwenGreedyDecodeSession(
        model: model,
        caches: try QwenHybridDecoderCaches(
            architecture: architecture,
            runtime: runtime,
            kvCapacity: 5
        ),
        executor: executor
    )
    let sampledSession = QwenGreedyDecodeSession(
        model: model,
        caches: try QwenHybridDecoderCaches(
            architecture: architecture,
            runtime: runtime,
            kvCapacity: 5
        ),
        executor: executor
    )
    var rng = EdgeSeededRandomNumberGenerator(seed: 7)

    try greedySession.prefill(promptTokenIds: [0])
    try sampledSession.prefill(promptTokenIds: [0])
    let greedyStep = try greedySession.generateNextToken()
    let sampledStep = try sampledSession.sampleNextToken(
        configuration: .greedy,
        rng: &rng
    )

    #expect(sampledStep.token.tokenId == greedyStep.token.tokenId)
    #expect(sampledStep.token.logit == greedyStep.token.logit)
    #expect(sampledStep.token.probability == 1)
    #expect(sampledStep.cacheTokenPosition == greedyStep.cacheTokenPosition)
    #expect(try sampledSession.tokenPosition() == 2)
}

@Test func qwenGreedyDecodeSessionSamplesAndResumesAcrossCalls() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeHybridModelArchitecture()
    let model = try makeHybridModelReference(architecture: architecture, runtime: runtime)
    let session = QwenGreedyDecodeSession(
        model: model,
        caches: try QwenHybridDecoderCaches(
            architecture: architecture,
            runtime: runtime,
            kvCapacity: 5
        ),
        executor: executor
    )
    var rng = EdgeSeededRandomNumberGenerator(seed: 11)

    try session.prefill(promptTokenIds: [0])
    let firstStep = try session.sampleNextToken(
        configuration: QwenSamplingConfiguration(temperature: 1, topK: 2),
        rng: &rng
    )
    let secondStep = try session.sampleNextToken(
        configuration: QwenSamplingConfiguration(temperature: 1, topP: 0.9),
        rng: &rng
    )

    #expect(firstStep.token.tokenId >= 0 && firstStep.token.tokenId < architecture.vocabularySize)
    #expect(secondStep.token.tokenId >= 0 && secondStep.token.tokenId < architecture.vocabularySize)
    #expect(firstStep.token.probability > 0 && firstStep.token.probability <= 1)
    #expect(secondStep.token.probability > 0 && secondStep.token.probability <= 1)
    #expect(firstStep.cacheTokenPosition == 2)
    #expect(secondStep.cacheTokenPosition == 3)
    #expect(try session.tokenPosition() == 3)
}

@Test func qwenGreedyDecodeSessionSampledPrefillOnlyWarmsCaches() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeHybridModelArchitecture()
    let model = try makeHybridModelReference(architecture: architecture, runtime: runtime)
    let session = QwenGreedyDecodeSession(
        model: model,
        caches: try QwenHybridDecoderCaches(
            architecture: architecture,
            runtime: runtime,
            kvCapacity: 5
        ),
        executor: executor
    )
    var rng = EdgeSeededRandomNumberGenerator(seed: 13)

    let generated = try session.sampleNextTokens(
        promptTokenIds: [0, 1],
        maxTokenCount: 0,
        configuration: QwenSamplingConfiguration(temperature: 1),
        rng: &rng
    )

    #expect(generated.isEmpty)
    #expect(try session.tokenPosition() == 2)
}

@Test func qwenGreedyDecodeSessionSampledEndTokenStopsWithoutAdvancingCache() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeHybridModelArchitecture()
    let model = try makeHybridModelReference(architecture: architecture, runtime: runtime)
    let session = QwenGreedyDecodeSession(
        model: model,
        caches: try QwenHybridDecoderCaches(
            architecture: architecture,
            runtime: runtime,
            kvCapacity: 5
        ),
        executor: executor
    )
    var rng = EdgeSeededRandomNumberGenerator(seed: 17)

    try session.prefill(promptTokenIds: [0])
    let selectedToken = try session.selectSampledToken(
        configuration: .greedy,
        rng: &rng
    )
    let step = try session.sampleNextToken(
        configuration: .greedy,
        endTokenIds: [selectedToken.tokenId],
        rng: &rng
    )
    var rejectedAfterStop = false

    do {
        _ = try session.sampleNextToken(
            configuration: .greedy,
            rng: &rng
        )
    } catch QwenGreedyDecodeSessionError.missingLogits {
        rejectedAfterStop = true
    }

    #expect(step.token.tokenId == selectedToken.tokenId)
    #expect(step.reachedEndToken)
    #expect(step.cacheTokenPosition == 1)
    #expect(try session.tokenPosition() == 1)
    #expect(rejectedAfterStop)
}

@Test func qwenGreedyDecodeSessionGreedyEndTokenStopsWithoutAdvancingCache() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeHybridModelArchitecture()
    let model = try makeHybridModelReference(architecture: architecture, runtime: runtime)
    let session = QwenGreedyDecodeSession(
        model: model,
        caches: try QwenHybridDecoderCaches(
            architecture: architecture,
            runtime: runtime,
            kvCapacity: 5
        ),
        executor: executor
    )

    try session.prefillGreedy(promptTokenIds: [0])
    let selectedToken = try session.selectNextToken()
    let step = try session.generateNextToken(
        endTokenIds: [selectedToken.tokenId]
    )
    var rejectedAfterStop = false

    do {
        _ = try session.generateNextToken()
    } catch QwenGreedyDecodeSessionError.missingLogits {
        rejectedAfterStop = true
    }

    #expect(step.token.tokenId == selectedToken.tokenId)
    #expect(step.reachedEndToken)
    #expect(step.cacheTokenPosition == 1)
    #expect(try session.tokenPosition() == 1)
    #expect(rejectedAfterStop)
}

@Test func qwenGreedyDecodeSessionStopsOnMultiTokenStopSequence() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeHybridModelArchitecture()
    let model = try makeHybridModelReference(architecture: architecture, runtime: runtime)
    let discoverySession = QwenGreedyDecodeSession(
        model: model,
        caches: try QwenHybridDecoderCaches(
            architecture: architecture,
            runtime: runtime,
            kvCapacity: 5
        ),
        executor: executor
    )
    let session = QwenGreedyDecodeSession(
        model: model,
        caches: try QwenHybridDecoderCaches(
            architecture: architecture,
            runtime: runtime,
            kvCapacity: 5
        ),
        executor: executor
    )

    try discoverySession.prefill(promptTokenIds: [0])
    let firstToken = try discoverySession.selectNextToken()
    try discoverySession.advance(with: firstToken.tokenId)
    let secondToken = try discoverySession.selectNextToken()
    let generated = try session.generateNextTokens(
        promptTokenIds: [0],
        maxTokenCount: 5,
        stopSequences: [[firstToken.tokenId, secondToken.tokenId]]
    )
    var rejectedAfterStop = false

    do {
        _ = try session.generateNextToken()
    } catch QwenGreedyDecodeSessionError.missingLogits {
        rejectedAfterStop = true
    }

    #expect(generated == [firstToken.tokenId, secondToken.tokenId])
    #expect(try session.tokenPosition() == 2)
    #expect(rejectedAfterStop)
}

@Test func qwenGreedyDecodeSessionSampledStopsOnMultiTokenStopSequence() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeHybridModelArchitecture()
    let model = try makeHybridModelReference(architecture: architecture, runtime: runtime)
    let discoverySession = QwenGreedyDecodeSession(
        model: model,
        caches: try QwenHybridDecoderCaches(
            architecture: architecture,
            runtime: runtime,
            kvCapacity: 5
        ),
        executor: executor
    )
    let session = QwenGreedyDecodeSession(
        model: model,
        caches: try QwenHybridDecoderCaches(
            architecture: architecture,
            runtime: runtime,
            kvCapacity: 5
        ),
        executor: executor
    )
    var rng = EdgeSeededRandomNumberGenerator(seed: 23)

    try discoverySession.prefill(promptTokenIds: [0])
    let firstToken = try discoverySession.selectNextToken()
    try discoverySession.advance(with: firstToken.tokenId)
    let secondToken = try discoverySession.selectNextToken()
    let generated = try session.sampleNextTokens(
        promptTokenIds: [0],
        maxTokenCount: 5,
        configuration: .greedy,
        stopSequences: [[firstToken.tokenId, secondToken.tokenId]],
        rng: &rng
    )
    var rejectedAfterStop = false

    do {
        _ = try session.sampleNextToken(
            configuration: .greedy,
            rng: &rng
        )
    } catch QwenGreedyDecodeSessionError.missingLogits {
        rejectedAfterStop = true
    }

    #expect(generated == [firstToken.tokenId, secondToken.tokenId])
    #expect(try session.tokenPosition() == 2)
    #expect(rejectedAfterStop)
}

@Test func qwenGreedyDecodeSessionValidatesStopSequencesBeforePrefill() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeHybridModelArchitecture()
    let model = try makeHybridModelReference(architecture: architecture, runtime: runtime)
    let session = QwenGreedyDecodeSession(
        model: model,
        caches: try QwenHybridDecoderCaches(
            architecture: architecture,
            runtime: runtime,
            kvCapacity: 5
        ),
        executor: executor
    )

    try session.prefill(promptTokenIds: [0])
    var rejectedEmptyStopSequence = false

    do {
        _ = try session.generateNextTokens(
            promptTokenIds: [0, 1],
            maxTokenCount: 1,
            stopSequences: [[]]
        )
    } catch QwenStopTokenMatcherError.emptyStopSequence(index: 0) {
        rejectedEmptyStopSequence = true
    }

    #expect(rejectedEmptyStopSequence)
    #expect(try session.tokenPosition() == 1)
}

@Test func qwenGreedyDecodeSessionValidatesSampledStopSequencesBeforePrefill() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeHybridModelArchitecture()
    let model = try makeHybridModelReference(architecture: architecture, runtime: runtime)
    let session = QwenGreedyDecodeSession(
        model: model,
        caches: try QwenHybridDecoderCaches(
            architecture: architecture,
            runtime: runtime,
            kvCapacity: 5
        ),
        executor: executor
    )
    var rng = EdgeSeededRandomNumberGenerator(seed: 29)

    try session.prefill(promptTokenIds: [0])
    var rejectedEmptyStopSequence = false

    do {
        _ = try session.sampleNextTokens(
            promptTokenIds: [0, 1],
            maxTokenCount: 1,
            configuration: .greedy,
            stopSequences: [[]],
            rng: &rng
        )
    } catch QwenStopTokenMatcherError.emptyStopSequence(index: 0) {
        rejectedEmptyStopSequence = true
    }

    #expect(rejectedEmptyStopSequence)
    #expect(try session.tokenPosition() == 1)
}

@Test func qwenGreedyDecodeSessionSelectSampledTokenDoesNotAdvanceCache() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = try MetalKernelExecutor(runtime: runtime)
    let architecture = try makeHybridModelArchitecture()
    let model = try makeHybridModelReference(architecture: architecture, runtime: runtime)
    let session = QwenGreedyDecodeSession(
        model: model,
        caches: try QwenHybridDecoderCaches(
            architecture: architecture,
            runtime: runtime,
            kvCapacity: 5
        ),
        executor: executor
    )
    var rng = EdgeSeededRandomNumberGenerator(seed: 19)

    try session.prefill(promptTokenIds: [0])
    let token = try session.selectSampledToken(
        configuration: QwenSamplingConfiguration(temperature: 1, topK: 2),
        rng: &rng
    )

    #expect(token.tokenId >= 0 && token.tokenId < architecture.vocabularySize)
    #expect(token.probability > 0 && token.probability <= 1)
    #expect(try session.tokenPosition() == 1)
}

private func makeHybridModelReference(
    architecture: QwenHybridArchitecture,
    runtime: EdgeMetalRuntime
) throws -> QwenHybridModelReference {
    try QwenHybridModelReference(
        architecture: architecture,
        embeddings: QwenTokenEmbeddingWeights(
            embeddings: try EdgeTensor(
                float32: [
                    1, 0,
                    0, 1,
                    1, 1,
                ],
                shape: EdgeTensorShape([3, 2]),
                runtime: runtime
            )
        ),
        decoderLayers: [
            .fullAttention(try makeHybridModelFullAttentionLayer(architecture: architecture, runtime: runtime)),
            .gdn(try makeHybridModelGDNLayer(runtime: runtime)),
        ],
        outputWeights: QwenModelOutputWeights(
            finalNorm: try EdgeTensor(float32: [1, 1], shape: EdgeTensorShape([2]), runtime: runtime),
            lmHead: try EdgeTensor(
                float32: [
                    1, 0, 1,
                    0, 1, -1,
                ],
                shape: EdgeTensorShape([2, 3]),
                runtime: runtime
            ),
            rmsNormEpsilon: 1e-6
        )
    )
}

private func makeHybridModelFullAttentionLayer(
    architecture: QwenHybridArchitecture,
    runtime: EdgeMetalRuntime
) throws -> QwenFullAttentionDecoderLayerReference {
    let attention = try QwenFullAttentionReference(
        architecture: architecture,
        layerIndex: 0,
        projectionWeights: QwenAttentionProjectionWeights(
            layerIndex: 0,
            query: try EdgeTensor(
                float32: [
                    1, 0, 1, 0,
                    0, 1, 0, 1,
                ],
                shape: EdgeTensorShape([2, 4]),
                runtime: runtime
            ),
            key: try EdgeTensor(float32: [1, 0], shape: EdgeTensorShape([2, 1]), runtime: runtime),
            value: try EdgeTensor(float32: [0, 1], shape: EdgeTensorShape([2, 1]), runtime: runtime)
        ),
        normalizationWeights: QwenAttentionNormWeights(
            layerIndex: 0,
            query: try EdgeTensor(float32: [1], shape: EdgeTensorShape([1]), runtime: runtime),
            key: try EdgeTensor(float32: [1], shape: EdgeTensorShape([1]), runtime: runtime)
        ),
        outputProjectionWeights: QwenAttentionOutputProjectionWeights(
            layerIndex: 0,
            weight: try EdgeTensor(
                float32: [
                    1, 0,
                    0, 1,
                ],
                shape: EdgeTensorShape([2, 2]),
                runtime: runtime
            )
        )
    )
    return try QwenFullAttentionDecoderLayerReference(
        attention: attention,
        mlp: makeHybridModelMLP(layerIndex: 0, runtime: runtime),
        inputLayerNorm: try EdgeTensor(float32: [1, 1], shape: EdgeTensorShape([2]), runtime: runtime),
        postAttentionLayerNorm: try EdgeTensor(float32: [1, 1], shape: EdgeTensorShape([2]), runtime: runtime),
        rmsNormEpsilon: 1e-6
    )
}

private func makeHybridModelGDNLayer(runtime: EdgeMetalRuntime) throws -> QwenGDNDecoderLayerReference {
    let linearAttention = QwenGDNWeights(
        layerIndex: 1,
        inProjQKV: try EdgeTensor(
            float32: [
                1, 0, 1,
                0, 1, 0,
            ],
            shape: EdgeTensorShape([2, 3]),
            runtime: runtime
        ),
        inProjZ: try EdgeTensor(float32: [1, 0], shape: EdgeTensorShape([2, 1]), runtime: runtime),
        inProjB: try EdgeTensor(float32: [1, 0], shape: EdgeTensorShape([2, 1]), runtime: runtime),
        inProjA: try EdgeTensor(float32: [1, 0], shape: EdgeTensorShape([2, 1]), runtime: runtime),
        conv1D: try EdgeTensor(
            float32: Array(repeating: Float(1), count: 12),
            shape: EdgeTensorShape([3, 4, 1]),
            runtime: runtime
        ),
        convWeightLayout: .mlxSanitizedDepthwise,
        aLog: try EdgeTensor(float32: [0.25], shape: EdgeTensorShape([1]), runtime: runtime),
        dtBias: try EdgeTensor(float32: [1], shape: EdgeTensorShape([1]), runtime: runtime),
        norm: try EdgeTensor(float32: [1], shape: EdgeTensorShape([1]), runtime: runtime),
        outProj: try EdgeTensor(float32: [1, 1], shape: EdgeTensorShape([1, 2]), runtime: runtime),
        linearKeyHeadCount: 1,
        linearValueHeadCount: 1,
        linearKeyHeadDimension: 1,
        linearValueHeadDimension: 1,
        linearKeyHiddenSize: 1,
        linearValueHiddenSize: 1,
        rmsNormEpsilon: 1e-6
    )
    return try QwenGDNDecoderLayerReference(
        linearAttention: linearAttention,
        mlp: makeHybridModelMLP(layerIndex: 1, runtime: runtime),
        inputLayerNorm: try EdgeTensor(float32: [1, 1], shape: EdgeTensorShape([2]), runtime: runtime),
        postAttentionLayerNorm: try EdgeTensor(float32: [1, 1], shape: EdgeTensorShape([2]), runtime: runtime),
        rmsNormEpsilon: 1e-6
    )
}

private func makeHybridModelMLP(layerIndex: Int, runtime: EdgeMetalRuntime) throws -> QwenMLPWeights {
    QwenMLPWeights(
        layerIndex: layerIndex,
        gate: try EdgeTensor(float32: Array(repeating: 0, count: 4), shape: EdgeTensorShape([2, 2]), runtime: runtime),
        up: try EdgeTensor(float32: Array(repeating: 0, count: 4), shape: EdgeTensorShape([2, 2]), runtime: runtime),
        down: try EdgeTensor(float32: Array(repeating: 0, count: 4), shape: EdgeTensorShape([2, 2]), runtime: runtime)
    )
}

private func makeHybridModelArchitecture() throws -> QwenHybridArchitecture {
    try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 3,
        hiddenSize: 2,
        intermediateSize: 2,
        attentionHeadCount: 2,
        keyValueHeadCount: 1,
        headDimension: 1,
        linearValueHeadCount: 1,
        linearKeyHeadCount: 1,
        linearKeyHeadDimension: 1,
        linearValueHeadDimension: 1,
        linearConvKernelSize: 4,
        contextLength: 8,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000,
        layerKinds: [.fullAttention, .gdn]
    )
}

private func makeQuantizedHybridModelArchitecture() throws -> QwenHybridArchitecture {
    try QwenHybridArchitecture(
        family: .qwen35,
        vocabularySize: 3,
        hiddenSize: 2,
        intermediateSize: 2,
        attentionHeadCount: 2,
        keyValueHeadCount: 1,
        headDimension: 1,
        linearValueHeadCount: 1,
        linearKeyHeadCount: 1,
        linearKeyHeadDimension: 1,
        linearValueHeadDimension: 1,
        linearConvKernelSize: 4,
        contextLength: 8,
        rmsNormEpsilon: 1e-6,
        ropeTheta: 10_000,
        quantization: QwenQuantizationProfile(groupSize: 1, bits: 8),
        layerKinds: [.fullAttention, .gdn]
    )
}

private func hybridModelWeightMap(includeLMHead: Bool = true) -> [String: String] {
    Dictionary(
        uniqueKeysWithValues: hybridModelTensorEntries().keys.filter {
            includeLMHead || $0 != "model.lm_head.weight"
        }.map {
            ($0, "model-00001-of-00001.safetensors")
        }
    )
}

private func writeHybridModelConfig(to root: URL) throws {
    let json = """
    {
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
    }
    """
    try json.data(using: .utf8)!.write(to: root.appendingPathComponent("config.json"))
}

private func writeQuantizedHybridModelConfig(to root: URL) throws {
    let json = """
    {
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
      "quantization": {
        "group_size": 1,
        "bits": 8
      },
      "layer_types": ["full_attention", "linear_attention"]
    }
    """
    try json.data(using: .utf8)!.write(to: root.appendingPathComponent("config.json"))
}

private func writeCmlxSmokeModelConfig(to root: URL) throws {
    let json = """
    {
      "model_type": "qwen3_5",
      "vocab_size": 16,
      "hidden_size": 32,
      "intermediate_size": 32,
      "num_attention_heads": 4,
      "num_key_value_heads": 2,
      "head_dim": 8,
      "linear_num_value_heads": 4,
      "linear_num_key_heads": 2,
      "linear_key_head_dim": 8,
      "linear_value_head_dim": 8,
      "linear_conv_kernel_dim": 4,
      "context_length": 64,
      "rms_norm_eps": 0.000001,
      "rope_theta": 1000000,
      "partial_rotary_factor": 0.5,
      "quantization": {
        "group_size": 32,
        "bits": 4
      },
      "layer_types": ["full_attention", "linear_attention"]
    }
    """
    try json.data(using: .utf8)!.write(to: root.appendingPathComponent("config.json"))
}

private func writeHybridModelIndex(weightMap: [String: String], to root: URL) throws {
    let payload: [String: Any] = [
        "weight_map": weightMap,
    ]
    let data = try JSONSerialization.data(
        withJSONObject: payload,
        options: [.prettyPrinted, .sortedKeys]
    )
    try data.write(to: root.appendingPathComponent("model.safetensors.index.json"))
}

private func quantizedHybridModelWeightMap(
    includeLMHead: Bool = true,
    quantizedOutput: Bool = true,
    quantizedEmbeddings: Bool = false
) -> [String: String] {
    var weightMap = hybridModelWeightMap(includeLMHead: includeLMHead)
    for name in quantizedHybridWeightNames(
        quantizedOutput: includeLMHead && quantizedOutput,
        quantizedEmbeddings: quantizedEmbeddings || (!includeLMHead && quantizedOutput)
    ) {
        let base = String(name.dropLast(".weight".count))
        weightMap["\(base).scales"] = "model-00001-of-00001.safetensors"
    }
    return weightMap
}

private func quantizedHybridWeightNames(
    quantizedOutput: Bool = true,
    quantizedEmbeddings: Bool = false
) -> [String] {
    var names = quantizedHybridDecoderWeightNames()
    if quantizedEmbeddings {
        names.append("model.embed_tokens.weight")
    }
    if quantizedOutput {
        names.append("model.lm_head.weight")
    }
    return names
}

private func quantizedHybridDecoderWeightNames() -> [String] {
    [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.linear_attn.in_proj_qkv.weight",
        "model.layers.1.linear_attn.in_proj_z.weight",
        "model.layers.1.linear_attn.in_proj_b.weight",
        "model.layers.1.linear_attn.in_proj_a.weight",
        "model.layers.1.linear_attn.out_proj.weight",
        "model.layers.1.mlp.gate_proj.weight",
        "model.layers.1.mlp.up_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
    ]
}

private func hybridModelTensorEntries() -> [String: (shape: [Int], values: [Float])] {
    [
        "model.embed_tokens.weight": ([3, 2], [
            1, 0,
            0, 1,
            1, 1,
        ]),
        "model.norm.weight": ([2], [1, 1]),
        "model.lm_head.weight": ([3, 2], [
            1, 0,
            0, 1,
            1, 2,
        ]),
        "model.layers.0.input_layernorm.weight": ([2], [1, 1]),
        "model.layers.0.post_attention_layernorm.weight": ([2], [1, 1]),
        "model.layers.0.mlp.gate_proj.weight": ([2, 2], Array(repeating: Float.zero, count: 4)),
        "model.layers.0.mlp.up_proj.weight": ([2, 2], Array(repeating: Float.zero, count: 4)),
        "model.layers.0.mlp.down_proj.weight": ([2, 2], Array(repeating: Float.zero, count: 4)),
        "model.layers.0.self_attn.q_proj.weight": ([4, 2], [
            1, 0,
            0, 1,
            1, 0,
            0, 1,
        ]),
        "model.layers.0.self_attn.k_proj.weight": ([1, 2], [1, 0]),
        "model.layers.0.self_attn.v_proj.weight": ([1, 2], [0, 1]),
        "model.layers.0.self_attn.o_proj.weight": ([2, 2], [
            1, 0,
            0, 1,
        ]),
        "model.layers.0.self_attn.q_norm.weight": ([1], [1]),
        "model.layers.0.self_attn.k_norm.weight": ([1], [1]),
        "model.layers.1.input_layernorm.weight": ([2], [1, 1]),
        "model.layers.1.post_attention_layernorm.weight": ([2], [1, 1]),
        "model.layers.1.mlp.gate_proj.weight": ([2, 2], Array(repeating: Float.zero, count: 4)),
        "model.layers.1.mlp.up_proj.weight": ([2, 2], Array(repeating: Float.zero, count: 4)),
        "model.layers.1.mlp.down_proj.weight": ([2, 2], Array(repeating: Float.zero, count: 4)),
        "model.layers.1.linear_attn.in_proj_qkv.weight": ([3, 2], [
            1, 0,
            0, 1,
            1, 0,
        ]),
        "model.layers.1.linear_attn.in_proj_z.weight": ([1, 2], [1, 0]),
        "model.layers.1.linear_attn.in_proj_b.weight": ([1, 2], [1, 0]),
        "model.layers.1.linear_attn.in_proj_a.weight": ([1, 2], [1, 0]),
        "model.layers.1.linear_attn.conv1d.weight": ([3, 4, 1], Array(repeating: Float(1), count: 12)),
        "model.layers.1.linear_attn.A_log": ([1], [0.25]),
        "model.layers.1.linear_attn.dt_bias": ([1], [1]),
        "model.layers.1.linear_attn.norm.weight": ([1], [1]),
        "model.layers.1.linear_attn.out_proj.weight": ([2, 1], [1, 1]),
    ]
}

private func writeHybridModelSafeTensorsShard(
    tensors: [String: (shape: [Int], values: [Float])],
    to url: URL
) throws {
    var payload = Data()
    var fields: [String] = []
    var offset = 0
    for name in tensors.keys.sorted() {
        let tensor = tensors[name]!
        let data = tensor.values.withUnsafeBufferPointer { pointer in
            Data(buffer: pointer)
        }
        let end = offset + data.count
        fields.append(
            """
            "\(name)": {
              "dtype": "F32",
              "shape": \(hybridModelShapeJSON(tensor.shape)),
              "data_offsets": [\(offset), \(end)]
            }
            """
        )
        payload.append(data)
        offset = end
    }
    let headerJSON = "{\(fields.joined(separator: ","))}"
    let headerData = headerJSON.data(using: .utf8)!
    var headerLength = UInt64(headerData.count).littleEndian
    var fileData = withUnsafeBytes(of: &headerLength) { Data($0) }
    fileData.append(headerData)
    fileData.append(payload)
    try fileData.write(to: url)
}

private func writeQuantizedHybridModelSafeTensorsShard(
    to url: URL,
    quantizedOutput: Bool = true,
    quantizedEmbeddings: Bool = false
) throws {
    let baseEntries = hybridModelTensorEntries()
    let quantizedNames = Set(
        quantizedHybridWeightNames(
            quantizedOutput: quantizedOutput,
            quantizedEmbeddings: quantizedEmbeddings
        )
    )
    var entries: [(name: String, dtype: String, shape: [Int], data: Data)] = []
    for name in baseEntries.keys.sorted() {
        let tensor = baseEntries[name]!
        if quantizedNames.contains(name) {
            entries += quantizedHybridEntries(
                name: name,
                shape: tensor.shape,
                values: tensor.values
            )
        } else {
            entries.append((
                name,
                "F32",
                tensor.shape,
                floatDataForHybridModel(tensor.values)
            ))
        }
    }
    try writeHybridModelSafeTensorsEntries(entries, to: url)
}

private func cmlxSmokeWeightMap() -> [String: String] {
    Dictionary(
        uniqueKeysWithValues: cmlxSmokeSafeTensorEntries().flatMap { entry -> [(String, String)] in
            [(entry.name, "model-00001-of-00001.safetensors")]
        }
    )
}

private func writeCmlxSmokeSafeTensorsShard(to url: URL) throws {
    try writeHybridModelSafeTensorsEntries(cmlxSmokeSafeTensorEntries(), to: url)
}

private func cmlxSmokeSafeTensorEntries() -> [(name: String, dtype: String, shape: [Int], data: Data)] {
    let hidden = 32
    let vocabulary = 16
    let intermediate = 32
    let headDimension = 8
    let attentionHidden = 32
    let keyValueHidden = 16
    let linearValueHeadCount = 4
    let linearValueHidden = 32
    let convHidden = 64
    let convKernel = 4

    func floatEntry(_ name: String, _ shape: [Int], _ values: [Float]) -> (String, String, [Int], Data) {
        (name, "F32", shape, floatDataForHybridModel(values))
    }

    func quantizedEntries(_ name: String, _ rows: Int, _ columns: Int, _ seed: Int) -> [(String, String, [Int], Data)] {
        let values = (0..<(rows * columns)).map { Float(($0 + seed) % 16) }
        return quantizedHybridEntries(
            name: name,
            shape: [rows, columns],
            values: values,
            groupSize: 32,
            bits: 4
        )
    }

    var entries: [(name: String, dtype: String, shape: [Int], data: Data)] = [
        floatEntry(
            "model.embed_tokens.weight",
            [vocabulary, hidden],
            (0..<(vocabulary * hidden)).map { 0.25 + Float($0 % hidden) / 128.0 }
        ),
        floatEntry("model.norm.weight", [hidden], Array(repeating: Float(1), count: hidden)),
        floatEntry("model.layers.0.input_layernorm.weight", [hidden], Array(repeating: Float(1), count: hidden)),
        floatEntry("model.layers.0.post_attention_layernorm.weight", [hidden], Array(repeating: Float(1), count: hidden)),
        floatEntry("model.layers.0.self_attn.q_norm.weight", [headDimension], Array(repeating: Float(1), count: headDimension)),
        floatEntry("model.layers.0.self_attn.k_norm.weight", [headDimension], Array(repeating: Float(1), count: headDimension)),
        floatEntry("model.layers.1.input_layernorm.weight", [hidden], Array(repeating: Float(1), count: hidden)),
        floatEntry("model.layers.1.post_attention_layernorm.weight", [hidden], Array(repeating: Float(1), count: hidden)),
        floatEntry(
            "model.layers.1.linear_attn.conv1d.weight",
            [convHidden, convKernel, 1],
            (0..<(convHidden * convKernel)).map { index in
                index % convKernel == convKernel - 1 ? 0.55 : 0
            }
        ),
        floatEntry("model.layers.1.linear_attn.A_log", [linearValueHeadCount], Array(repeating: Float(-1), count: linearValueHeadCount)),
        floatEntry("model.layers.1.linear_attn.dt_bias", [linearValueHeadCount], Array(repeating: Float.zero, count: linearValueHeadCount)),
        floatEntry("model.layers.1.linear_attn.norm.weight", [headDimension], Array(repeating: Float(1), count: headDimension)),
    ]

    entries += quantizedEntries("model.lm_head.weight", vocabulary, hidden, 1)
    entries += quantizedEntries("model.layers.0.self_attn.q_proj.weight", attentionHidden * 2, hidden, 2)
    entries += quantizedEntries("model.layers.0.self_attn.k_proj.weight", keyValueHidden, hidden, 3)
    entries += quantizedEntries("model.layers.0.self_attn.v_proj.weight", keyValueHidden, hidden, 4)
    entries += quantizedEntries("model.layers.0.self_attn.o_proj.weight", hidden, attentionHidden, 5)
    entries += quantizedEntries("model.layers.0.mlp.gate_proj.weight", intermediate, hidden, 6)
    entries += quantizedEntries("model.layers.0.mlp.up_proj.weight", intermediate, hidden, 7)
    entries += quantizedEntries("model.layers.0.mlp.down_proj.weight", hidden, intermediate, 8)
    entries += quantizedEntries("model.layers.1.linear_attn.in_proj_qkv.weight", convHidden, hidden, 9)
    entries += quantizedEntries("model.layers.1.linear_attn.in_proj_z.weight", linearValueHidden, hidden, 10)
    entries += quantizedEntries("model.layers.1.linear_attn.in_proj_b.weight", linearValueHeadCount, hidden, 11)
    entries += quantizedEntries("model.layers.1.linear_attn.in_proj_a.weight", linearValueHeadCount, hidden, 12)
    entries += quantizedEntries("model.layers.1.linear_attn.out_proj.weight", hidden, linearValueHidden, 13)
    entries += quantizedEntries("model.layers.1.mlp.gate_proj.weight", intermediate, hidden, 14)
    entries += quantizedEntries("model.layers.1.mlp.up_proj.weight", intermediate, hidden, 15)
    entries += quantizedEntries("model.layers.1.mlp.down_proj.weight", hidden, intermediate, 1)
    return entries
}

private func quantizedHybridEntries(
    name: String,
    shape: [Int],
    values: [Float],
    groupSize: Int = 1,
    bits: Int = 8
) -> [(name: String, dtype: String, shape: [Int], data: Data)] {
    let rows = hybridRows(shape: shape, values: values)
    let columns = rows.first?.count ?? 0
    let scaleColumns = columns / groupSize
    let base = String(name.dropLast(".weight".count))
    return [
        (
            name,
            "U32",
            [rows.count, (columns * bits + 31) / 32],
            uint32DataForHybridModel(packHybridQuantizedRows(rows, bits: bits))
        ),
        (
            "\(base).scales",
            "F32",
            [rows.count, scaleColumns],
            floatDataForHybridModel(Array(repeating: Float(1), count: rows.count * scaleColumns))
        ),
    ]
}

private func hybridRows(shape: [Int], values: [Float]) -> [[UInt32]] {
    let rows = shape[0]
    let columns = shape[1]
    return (0..<rows).map { row in
        (0..<columns).map { column in
            UInt32(Int(values[row * columns + column]))
        }
    }
}

private func packHybridQuantizedRows(_ rows: [[UInt32]], bits: Int) -> [UInt32] {
    rows.flatMap { packHybridQuantizedWords($0, bits: bits) }
}

private func packHybridQuantizedWords(_ values: [UInt32], bits: Int) -> [UInt32] {
    var words = Array(repeating: UInt32.zero, count: (values.count * bits + 31) / 32)
    let mask = UInt32((1 << bits) - 1)
    for (index, value) in values.enumerated() {
        let bitOffset = index * bits
        let wordIndex = bitOffset / 32
        let shift = bitOffset % 32
        words[wordIndex] |= (value & mask) << UInt32(shift)
        if shift + bits > 32 {
            words[wordIndex + 1] |= (value & mask) >> UInt32(32 - shift)
        }
    }
    return words
}

private func writeHybridModelSafeTensorsEntries(
    _ entries: [(name: String, dtype: String, shape: [Int], data: Data)],
    to url: URL
) throws {
    var payload = Data()
    var fields: [String] = []
    var offset = 0
    for entry in entries {
        let end = offset + entry.data.count
        fields.append(
            """
            "\(entry.name)": {
              "dtype": "\(entry.dtype)",
              "shape": \(hybridModelShapeJSON(entry.shape)),
              "data_offsets": [\(offset), \(end)]
            }
            """
        )
        payload.append(entry.data)
        offset = end
    }
    let headerJSON = "{\(fields.joined(separator: ","))}"
    let headerData = headerJSON.data(using: .utf8)!
    var headerLength = UInt64(headerData.count).littleEndian
    var fileData = withUnsafeBytes(of: &headerLength) { Data($0) }
    fileData.append(headerData)
    fileData.append(payload)
    try fileData.write(to: url)
}

private func floatDataForHybridModel(_ values: [Float]) -> Data {
    var data = Data()
    for value in values {
        var littleEndianValue = value.bitPattern.littleEndian
        data.append(withUnsafeBytes(of: &littleEndianValue) { Data($0) })
    }
    return data
}

private func uint32DataForHybridModel(_ values: [UInt32]) -> Data {
    var data = Data()
    for value in values {
        var littleEndianValue = value.littleEndian
        data.append(withUnsafeBytes(of: &littleEndianValue) { Data($0) })
    }
    return data
}

private func hybridModelShapeJSON(_ shape: [Int]) -> String {
    "[\(shape.map(String.init).joined(separator: ","))]"
}
