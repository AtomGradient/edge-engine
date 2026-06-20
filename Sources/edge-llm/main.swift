// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Darwin
import EdgeEngine
import Foundation
import Tokenizers

enum LLMGenerateCLIError: Error, CustomStringConvertible {
    case missingRequiredOption(String)
    case missingValue(String)
    case invalidInteger(option: String, value: String)
    case invalidBoolean(option: String, value: String)
    case invalidFamily(String)
    case invalidHistoryJSON(String)
    case invalidHistoryRole(String)
    case unknownOption(String)

    var description: String {
        switch self {
        case .missingRequiredOption(let option):
            "missing required option \(option)"
        case .missingValue(let option):
            "missing value for \(option)"
        case .invalidInteger(let option, let value):
            "invalid integer for \(option): \(value)"
        case .invalidBoolean(let option, let value):
            "invalid boolean for \(option): \(value)"
        case .invalidFamily(let value):
            "invalid family: \(value) (expected qwen35 or qwen36)"
        case .invalidHistoryJSON(let reason):
            "invalid JSON for --history: \(reason)"
        case .invalidHistoryRole(let role):
            "invalid --history role: \(role) (expected system, user, assistant, or tool)"
        case .unknownOption(let option):
            "unknown option \(option)"
        }
    }
}

struct LLMGenerateHistoryMessage: Decodable {
    let role: String
    let content: String

    var tokenizerMessage: Message {
        [
            "role": role,
            "content": content,
        ]
    }
}

struct LLMGenerateCLIOptions {
    var modelPath: String?
    var prompt: String?
    var history: [LLMGenerateHistoryMessage] = []
    var maxTokens = 200
    var maxTokensWasExplicit = false
    var thinkEnabled = false
    var preserveThinking = false
    var dumpPrompt = false
    var modelFamily: QwenModelFamily?
    var maxOpsPerCommandBuffer = 700
    var maxMBPerCommandBuffer = 256
    var quantizedCacheMB = 4_096
    var neuralImprintDirectory: String?

    func modelRootURL() throws -> URL {
        guard let modelPath else {
            throw LLMGenerateCLIError.missingRequiredOption("--model")
        }
        return URL(fileURLWithPath: expandedPath(modelPath))
    }

    func promptText() throws -> String {
        guard let prompt else {
            throw LLMGenerateCLIError.missingRequiredOption("--prompt")
        }
        return prompt
    }

    func chatMessages() throws -> [Message] {
        var messages = history.map(\.tokenizerMessage)
        messages.append([
            "role": "user",
            "content": try promptText(),
        ])
        return messages
    }

    func neuralImprintDirectoryURL() -> URL? {
        neuralImprintDirectory.map { URL(fileURLWithPath: expandedPath($0)) }
    }
}

@main
enum EdgeLLMCommand {
    static func main() async {
        do {
            let arguments = Array(CommandLine.arguments.dropFirst())
            if arguments.contains("--help") || arguments.contains("-h") {
                print(Self.usage)
                return
            }
            let options = try parse(arguments: arguments)
            try await run(options: options)
        } catch let error as LLMGenerateCLIError {
            fputs("error: \(error.description)\n\n\(Self.usage)\n", stderr)
            exit(2)
        } catch {
            fputs("error: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func run(options: LLMGenerateCLIOptions) async throws {
        var memorySampler = CLIMemorySampler()
        let modelURL = try options.modelRootURL()
        let messages = try options.chatMessages()
        memorySampler.sample()
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: modelURL,
            strict: false
        )
        let promptTokens = try makePromptTokens(
            messages: messages,
            tokenizer: tokenizer,
            thinkEnabled: options.thinkEnabled,
            preserveThinking: options.preserveThinking
        )
        memorySampler.sample()

        if options.dumpPrompt {
            let decodedPrompt = tokenizer.decode(tokens: promptTokens, skipSpecialTokens: false)
            FileHandle.standardOutput.write(Data(decodedPrompt.utf8))
            if !decodedPrompt.hasSuffix("\n") {
                FileHandle.standardOutput.write(Data("\n".utf8))
            }
            return
        }

        let bundleIndex = try QwenModelBundleIndex.load(
            from: modelURL,
            family: options.modelFamily
        )

        let runtimeConfiguration = NativeRuntimeBridge.metalConfiguration(
            maxOpsPerCommandBuffer: options.maxOpsPerCommandBuffer,
            maxMBPerCommandBuffer: options.maxMBPerCommandBuffer,
            contextLengthHint: promptTokens.count + options.maxTokens,
            dynamicOpsSchedule: DynamicOpsSchedule(
                floor: 5,
                contextLow: 4_096,
                contextHigh: 12_288
            ),
            quantizedBufferCacheLimitBytes: options.quantizedCacheMB * 1_048_576,
            commandBufferBatchingEnabled: true,
            useVendoredCommandBufferPrefillQMM: true,
            useFusedGDNDecode: true
        ).applyingEnvironmentOverrides()
        _ = NativeRuntimeBridge.applyMetalConfiguration(runtimeConfiguration)

        let runtime = try EdgeMetalRuntime(configuration: runtimeConfiguration)
        let session = try QwenCmlxLazyDecodeSession(
            bundleIndex: bundleIndex,
            runtime: runtime
        )
        try session.reset()
        memorySampler.sample()

        if let neuralImprintDirectory = options.neuralImprintDirectoryURL() {
            let sidecar = try loadNeuralImprintSidecar(from: neuralImprintDirectory)
            let artifactURL = try neuralImprintArtifactURL(
                sidecarArtifactPath: sidecar.artifact,
                directory: neuralImprintDirectory
            )
            try session.restoreNeuralImprintCache(
                artifactURL: artifactURL,
                prefixTokenCount: sidecar.prefix.tokenCount
            )
            fputs(
                "neural_imprint_restore_configured prefix=\(sidecar.prefix.tokenCount) artifactSHA256=\(sidecar.artifactSHA256) modelID=\(sidecar.model.id)\n",
                stderr
            )
            memorySampler.sample()
        }

        let prefillStartedAt = Date()
        var nextTokenID: Int? = try session.prefill(tokenIDs: promptTokens)
        let prefillSeconds = Date().timeIntervalSince(prefillStartedAt)
        memorySampler.sample()

        let endTokenIds = defaultEndTokenIds(tokenizer: tokenizer)
        var generatedTokenIds: [Int] = []
        generatedTokenIds.reserveCapacity(options.maxTokens)
        var emittedText = ""
        let decodeStartedAt = Date()

        while generatedTokenIds.count < options.maxTokens, let tokenID = nextTokenID {
            if endTokenIds.contains(tokenID) {
                break
            }
            generatedTokenIds.append(tokenID)
            let decodedText = tokenizer.decode(
                tokens: generatedTokenIds,
                skipSpecialTokens: true
            )
            let delta: String
            if decodedText.hasPrefix(emittedText) {
                let start = decodedText.index(decodedText.startIndex, offsetBy: emittedText.count)
                delta = String(decodedText[start...])
            } else {
                delta = decodedText
            }
            emittedText = decodedText
            if !delta.isEmpty {
                FileHandle.standardOutput.write(Data(delta.utf8))
                fflush(stdout)
            }
            if generatedTokenIds.count < options.maxTokens {
                nextTokenID = try session.decodeStep(tokenID: tokenID)
            } else {
                nextTokenID = nil
            }
            memorySampler.sample()
        }

        let decodeSeconds = Date().timeIntervalSince(decodeStartedAt)
        if !emittedText.hasSuffix("\n") {
            FileHandle.standardOutput.write(Data("\n".utf8))
        }
        writeGenerationFooter(
            promptTokenCount: promptTokens.count,
            prefillSeconds: prefillSeconds,
            generatedTokenCount: generatedTokenIds.count,
            decodeSeconds: decodeSeconds,
            peakPhysicalFootprintBytes: memorySampler.peakPhysicalFootprintBytes
        )
    }

    private static func parse(arguments: [String]) throws -> LLMGenerateCLIOptions {
        var options = LLMGenerateCLIOptions()
        var index = 0
        while index < arguments.count {
            let raw = arguments[index]
            if raw == "--think" {
                options.thinkEnabled = true
                index += 1
                continue
            }
            if raw == "--preserve-thinking" {
                options.preserveThinking = true
                index += 1
                continue
            }
            if raw == "--dump-prompt" {
                options.dumpPrompt = true
                index += 1
                continue
            }
            let option: String
            let value: String
            if let equals = raw.firstIndex(of: "=") {
                option = String(raw[..<equals])
                value = String(raw[raw.index(after: equals)...])
            } else {
                option = raw
                guard index + 1 < arguments.count else {
                    throw LLMGenerateCLIError.missingValue(option)
                }
                index += 1
                value = arguments[index]
            }

            switch option {
            case "--model":
                options.modelPath = value
            case "--prompt":
                options.prompt = value
            case "--history":
                options.history = try parseHistory(value)
            case "--max-tokens":
                options.maxTokens = try parsePositiveInteger(option: option, value: value)
                options.maxTokensWasExplicit = true
            case "--think":
                options.thinkEnabled = try parseBoolean(option: option, value: value)
            case "--preserve-thinking":
                options.preserveThinking = try parseBoolean(option: option, value: value)
            case "--dump-prompt":
                options.dumpPrompt = try parseBoolean(option: option, value: value)
            case "--family":
                options.modelFamily = try parseFamily(value)
            case "--max-ops-per-buffer":
                options.maxOpsPerCommandBuffer = try parsePositiveInteger(option: option, value: value)
            case "--max-mb-per-buffer":
                options.maxMBPerCommandBuffer = try parsePositiveInteger(option: option, value: value)
            case "--quantized-cache-mb":
                options.quantizedCacheMB = try parseNonNegativeInteger(option: option, value: value)
            case "--neural-imprint-dir":
                options.neuralImprintDirectory = value
            default:
                throw LLMGenerateCLIError.unknownOption(option)
            }
            index += 1
        }
        if options.thinkEnabled && !options.maxTokensWasExplicit {
            options.maxTokens = 2_048
        }
        return options
    }

    private static func makePromptTokens(
        messages: [Message],
        tokenizer: Tokenizer,
        thinkEnabled: Bool,
        preserveThinking: Bool
    ) throws -> [Int] {
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: nil,
            additionalContext: [
                "enable_thinking": thinkEnabled,
                "preserve_thinking": preserveThinking,
            ]
        )
        guard !tokens.isEmpty else {
            throw QwenHybridModelReferenceError.emptyTokenIds
        }
        return tokens
    }

    private static func parseHistory(_ value: String) throws -> [LLMGenerateHistoryMessage] {
        do {
            let messages = try JSONDecoder().decode(
                [LLMGenerateHistoryMessage].self,
                from: Data(value.utf8)
            )
            for message in messages {
                switch message.role {
                case "system", "user", "assistant", "tool":
                    continue
                default:
                    throw LLMGenerateCLIError.invalidHistoryRole(message.role)
                }
            }
            return messages
        } catch let error as LLMGenerateCLIError {
            throw error
        } catch {
            throw LLMGenerateCLIError.invalidHistoryJSON(String(describing: error))
        }
    }

    private static func defaultEndTokenIds(tokenizer: Tokenizer) -> Set<Int> {
        var ids = Set<Int>()
        if let eos = tokenizer.eosTokenId {
            ids.insert(eos)
        }
        if let imEnd = tokenizer.convertTokenToId("<|im_end|>") {
            ids.insert(imEnd)
        }
        if let endOfText = tokenizer.convertTokenToId("<|endoftext|>") {
            ids.insert(endOfText)
        }
        return ids
    }

    private static func parseFamily(_ value: String) throws -> QwenModelFamily {
        switch value.lowercased().replacingOccurrences(of: "_", with: "") {
        case "qwen35", "qwen3.5", "qwen3-5", "qwen3p5":
            return .qwen35
        case "qwen36", "qwen3.6", "qwen3-6", "qwen3p6":
            return .qwen36
        default:
            throw LLMGenerateCLIError.invalidFamily(value)
        }
    }

    private static func parsePositiveInteger(option: String, value: String) throws -> Int {
        guard let parsed = Int(value), parsed > 0 else {
            throw LLMGenerateCLIError.invalidInteger(option: option, value: value)
        }
        return parsed
    }

    private static func parseBoolean(option: String, value: String) throws -> Bool {
        switch value.lowercased() {
        case "true", "1", "yes", "on":
            return true
        case "false", "0", "no", "off":
            return false
        default:
            throw LLMGenerateCLIError.invalidBoolean(option: option, value: value)
        }
    }

    private static func parseNonNegativeInteger(option: String, value: String) throws -> Int {
        guard let parsed = Int(value), parsed >= 0 else {
            throw LLMGenerateCLIError.invalidInteger(option: option, value: value)
        }
        return parsed
    }

    private static var usage: String {
        """
        Usage:
          edge-llm --model <model-dir> --prompt <text> [options]

        Options:
          --history <json>            Prior chat messages as JSON array with role/content fields.
          --max-tokens <n>            Maximum generated tokens. Default: 200
          --think                     Enable Qwen thinking mode. Default max tokens becomes 2048 unless set.
          --preserve-thinking         Preserve historical assistant thinking in chat template.
          --dump-prompt               Print decoded chat-template prompt and exit before loading weights.
          --family <qwen35|qwen36>    Override family detection from config.json.
          --max-ops-per-buffer <n>    Native Metal op budget. Default: 700
          --max-mb-per-buffer <n>     Native Metal byte budget. Default: 256
          --quantized-cache-mb <n>    Cap native quantized Metal buffer cache. Default: 4096
          --neural-imprint-dir <dir>      Restore a Neural Imprint full-cache artifact before generation.

        Example:
          swift run edge-llm \\
            --model ~/Models/Qwen3.5-4B-4bit \\
            --prompt "What is the capital of France?" \\
            --max-tokens 200

          swift run edge-llm \\
            --model ~/Models/Qwen3.6-35B-A3B-8bit \\
            --history '[{"role":"user","content":"What is 2+2?"},{"role":"assistant","content":"<think>Simple math.</think>\\n\\nThe answer is 4."}]' \\
            --prompt "And what is 3+3?" \\
            --think --preserve-thinking --dump-prompt
        """
    }
}

private func loadNeuralImprintSidecar(from directory: URL) throws -> NeuralImprintSidecar {
    let fileManager = FileManager.default
    let candidateNames = [
        "neural_imprint_metadata.json",
        "persona_kv_metadata.json",
    ]
    for name in candidateNames {
        let url = directory.appendingPathComponent(name)
        if fileManager.fileExists(atPath: url.path) {
            let sidecar = try NeuralImprintSidecar.load(from: url)
            _ = try neuralImprintArtifactURL(
                sidecarArtifactPath: sidecar.artifact,
                directory: directory
            )
            return sidecar
        }
    }
    throw CocoaError(
        .fileNoSuchFile,
        userInfo: [
            NSFilePathErrorKey: candidateNames
                .map { directory.appendingPathComponent($0).path }
                .joined(separator: ", ")
        ]
    )
}

private func neuralImprintArtifactURL(
    sidecarArtifactPath: String,
    directory: URL
) throws -> URL {
    let fileManager = FileManager.default
    let sidecarPath = sidecarArtifactPath as NSString
    if sidecarPath.isAbsolutePath {
        let absoluteURL = URL(fileURLWithPath: sidecarArtifactPath)
        if fileManager.fileExists(atPath: absoluteURL.path) {
            return absoluteURL
        }
        let fallbackURL = directory.appendingPathComponent(sidecarPath.lastPathComponent)
        if fileManager.fileExists(atPath: fallbackURL.path) {
            return fallbackURL
        }
        throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: absoluteURL.path])
    }

    let relativeURL = directory.appendingPathComponent(sidecarArtifactPath)
    if fileManager.fileExists(atPath: relativeURL.path) {
        return relativeURL
    }

    let defaultURLs = [
        directory.appendingPathComponent("neural_imprint.safetensors"),
        directory.appendingPathComponent("persona_kv.safetensors"),
        directory.appendingPathComponent(sidecarPath.lastPathComponent),
    ]
    if let defaultURL = defaultURLs.first(where: { fileManager.fileExists(atPath: $0.path) }) {
        return defaultURL
    }
    throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: relativeURL.path])
}

private func expandedPath(_ path: String) -> String {
    guard path == "~" || path.hasPrefix("~/") else {
        return path
    }
    let home = FileManager.default.homeDirectoryForCurrentUser.path
    if path == "~" {
        return home
    }
    return home + String(path.dropFirst())
}

private struct CLIMemorySampler {
    private(set) var peakPhysicalFootprintBytes: UInt64? = currentPhysicalFootprintBytes()

    mutating func sample() {
        guard let bytes = currentPhysicalFootprintBytes() else { return }
        peakPhysicalFootprintBytes = max(peakPhysicalFootprintBytes ?? 0, bytes)
    }
}

private func writeGenerationFooter(
    promptTokenCount: Int,
    prefillSeconds: TimeInterval,
    generatedTokenCount: Int,
    decodeSeconds: TimeInterval,
    peakPhysicalFootprintBytes: UInt64?
) {
    let promptTPS = tokensPerSecond(tokenCount: promptTokenCount, seconds: prefillSeconds)
    let generationTPS = tokensPerSecond(tokenCount: generatedTokenCount, seconds: decodeSeconds)
    let peakGB = Double(peakPhysicalFootprintBytes ?? 0) / 1_073_741_824.0
    let footer = String(
        format: """
        ==========
        Prompt: %d tokens, %.1f tokens-per-sec
        Generation: %d tokens, %.1f tokens-per-sec
        Peak memory: %.3f GB
        ==========

        """,
        promptTokenCount,
        promptTPS,
        generatedTokenCount,
        generationTPS,
        peakGB
    )
    FileHandle.standardOutput.write(Data(footer.utf8))
}

private func tokensPerSecond(tokenCount: Int, seconds: TimeInterval) -> Double {
    guard tokenCount > 0 else { return 0 }
    return Double(tokenCount) / max(seconds, 0.001)
}

private func currentPhysicalFootprintBytes() -> UInt64? {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.stride / MemoryLayout<natural_t>.stride
    )
    let result = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
            task_info(
                mach_task_self_,
                task_flavor_t(TASK_VM_INFO),
                rebound,
                &count
            )
        }
    }
    guard result == KERN_SUCCESS else {
        return nil
    }
    return UInt64(info.phys_footprint)
}
