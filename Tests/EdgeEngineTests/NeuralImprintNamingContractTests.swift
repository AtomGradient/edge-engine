// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Testing

@Test func neuralImprintCurrentNamingContractHasNoOldLiveSymbols() throws {
    let root = packageRoot()
    let liveFiles = [
        "Sources/EdgeEngine/Models/Qwen/NeuralImprintArtifact.swift",
        "Sources/EdgeEngine/Runtime/EdgeMLXQwen35Session.swift",
        "Sources/EdgeEngine/Models/Qwen/QwenCmlxLazyDecodeSession.swift",
        "Sources/EdgeEngine/Models/Qwen/QwenVLMNativeContainer.swift",
        "Sources/edge-llm/main.swift",
        "Sources/Cmlx/shim/include/CmlxShim.h",
        "Sources/Cmlx/shim/shim.cpp",
    ]
    let oldSnake = "persona" + "_kv"
    let oldKebab = "persona" + "-kv"
    let oldCamel = "persona" + "KV"
    let oldPascal = "Persona" + "KV"
    let oldEnv = "PERSONA" + "_KV"
    let disallowed = [
        oldPascal,
        oldCamel,
        oldEnv,
        "restore_" + oldSnake + "_cache",
        "save_" + oldSnake + "_cache",
        "restore" + oldPascal + "Cache",
        "save" + oldPascal + "Cache",
        "--" + oldKebab + "-dir",
        oldSnake + "_restore_configured",
        "cmlx_" + oldSnake + "_restore",
    ]

    for relativePath in liveFiles {
        let text = try String(
            contentsOf: root.appendingPathComponent(relativePath),
            encoding: .utf8
        )
        for needle in disallowed {
            #expect(!text.contains(needle))
        }
    }

    #expect(
        FileManager.default.fileExists(
            atPath: root
                .appendingPathComponent("Sources/EdgeEngine/Models/Qwen/NeuralImprintArtifact.swift")
                .path
        )
    )
    #expect(
        !FileManager.default.fileExists(
            atPath: root
                .appendingPathComponent("Sources/EdgeEngine/Models/Qwen/" + oldPascal + "Artifact.swift")
                .path
        )
    )
}

@Test func neuralImprintLegacyArtifactValuesStayExplicit() throws {
    let root = packageRoot()
    let oldSnake = "persona" + "_kv"
    let text = try String(
        contentsOf: root.appendingPathComponent("Sources/EdgeEngine/Models/Qwen/NeuralImprintArtifact.swift"),
        encoding: .utf8
    )

    #expect(text.contains("legacyArtifactType = \"\(oldSnake)\""))
    #expect(text.contains("legacyCacheSchema = \"edgestudio.\(oldSnake).full_cache.v1\""))
    #expect(text.contains("legacySidecarSchema = \"edgestudio.\(oldSnake).full_cache_metadata.v1\""))
    #expect(text.contains("legacyPrefixRendererVersion = \"edgestudio.\(oldSnake).renderer.v2\""))
}

private func packageRoot() -> URL {
    URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
}
