// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Darwin
import EdgeEngine
import Foundation
import Tokenizers

enum VLMGenerateCLIError: Error, CustomStringConvertible {
    case missingRequiredOption(String)
    case missingValue(String)
    case invalidInteger(option: String, value: String)
    case invalidBoolean(option: String, value: String)
    case invalidFamily(String)
    case invalidImageShape([Int])
    case invalidVisionTokenCount
    case tokenizerMissingToken(String)
    case promptImageTokenMismatch(prompt: Int, features: Int)
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
            "invalid family: \(value) (expected qwen35-vlm or qwen36-vlm)"
        case .invalidImageShape(let shape):
            "invalid image preprocessing shape \(shape)"
        case .invalidVisionTokenCount:
            "vision encoder returned an invalid image token count"
        case .tokenizerMissingToken(let token):
            "tokenizer is missing required token \(token)"
        case .promptImageTokenMismatch(let prompt, let features):
            "prompt image token count mismatch: prompt=\(prompt), features=\(features)"
        case .unknownOption(let option):
            "unknown option \(option)"
        }
    }
}

struct VLMGenerateCLIOptions {
    var modelPath: String?
    var imagePath: String?
    var prompt = "Describe this image."
    var maxTokens = 200
    var maxTokensWasExplicit = false
    var thinkEnabled = false
    var preserveThinking = false
    var modelFamily: QwenVLMModelFamily?
    var maxOpsPerCommandBuffer = 700
    var maxMBPerCommandBuffer = 256
    var quantizedCacheMB = 4_096

    func modelRootURL() throws -> URL {
        guard let modelPath else {
            throw VLMGenerateCLIError.missingRequiredOption("--model")
        }
        return URL(fileURLWithPath: expandedPath(modelPath))
    }

    func imageURL() throws -> URL {
        guard let imagePath else {
            throw VLMGenerateCLIError.missingRequiredOption("--image")
        }
        return URL(fileURLWithPath: expandedPath(imagePath))
    }
}

@main
enum EdgeVLMCommand {
    static func main() async {
        do {
            let arguments = Array(CommandLine.arguments.dropFirst())
            if arguments.contains("--help") || arguments.contains("-h") {
                print(Self.usage)
                return
            }
            let options = try parse(arguments: arguments)
            try await run(options: options)
        } catch let error as VLMGenerateCLIError {
            fputs("error: \(error.description)\n\n\(Self.usage)\n", stderr)
            exit(2)
        } catch {
            fputs("error: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func run(options: VLMGenerateCLIOptions) async throws {
        var memorySampler = CLIMemorySampler()
        let modelURL = try options.modelRootURL()
        let imageURL = try options.imageURL()
        let index = try QwenVLMModelBundleIndex.load(
            from: modelURL,
            family: options.modelFamily
        )
        memorySampler.sample()
        let plan = index.preflightResult.plan
        let preprocessing = try QwenImagePreprocessor.preprocessImage(
            at: imageURL,
            configuration: plan.imageProcessorConfiguration
        )
        memorySampler.sample()
        guard preprocessing.pixelValuesShape.count == 2,
              preprocessing.pixelValuesShape[0] > 0,
              preprocessing.pixelValuesShape[1] > 0
        else {
            throw VLMGenerateCLIError.invalidImageShape(preprocessing.pixelValuesShape)
        }
        let imageTokenCount = try imageTokenCount(
            for: preprocessing.imageGridTHW,
            plan: plan
        )
        let contextLengthHint = imageTokenCount + options.maxTokens + 128
        let runtimeConfiguration = NativeRuntimeBridge.metalConfiguration(
            maxOpsPerCommandBuffer: options.maxOpsPerCommandBuffer,
            maxMBPerCommandBuffer: options.maxMBPerCommandBuffer,
            contextLengthHint: contextLengthHint,
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

        let tokenizer = try await AutoTokenizer.from(
            modelFolder: modelURL,
            strict: false
        )
        memorySampler.sample()
        let runtime = try EdgeMetalRuntime(configuration: runtimeConfiguration)
        let executor = try MetalKernelExecutor(runtime: runtime)
        let container = QwenVLMNativeContainer(
            index: index,
            runtime: runtime,
            executor: executor
        )
        memorySampler.sample()

        try container.loadCmlxVisionWeights()
        memorySampler.sample()
        let visionEncoding = try container.visionEncode(
            pixelValues: preprocessing.pixelValues,
            pixelValuesShape: preprocessing.pixelValuesShape,
            gridTHW: [preprocessing.imageGridTHW]
        )
        memorySampler.sample()
        container.unloadCmlxVisionWeights()
        guard visionEncoding.shape.first == imageTokenCount else {
            throw VLMGenerateCLIError.invalidVisionTokenCount
        }

        let imageTokenID = try tokenID("<|image_pad|>", tokenizer: tokenizer)
        let promptTokens = try makePromptTokens(
            prompt: options.prompt,
            imageTokenCount: imageTokenCount,
            tokenizer: tokenizer,
            thinkEnabled: options.thinkEnabled,
            preserveThinking: options.preserveThinking
        )
        let placeholderCount = promptTokens.reduce(0) { count, token in
            count + (token == imageTokenID ? 1 : 0)
        }
        guard placeholderCount == imageTokenCount else {
            throw VLMGenerateCLIError.promptImageTokenMismatch(
                prompt: placeholderCount,
                features: imageTokenCount
            )
        }

        try container.loadCmlxDecoderWeights()
        memorySampler.sample()
        _ = try container.resetCmlxDecoder()
        let prefillStartedAt = Date()
        var nextTokenID: Int? = try container.prefillImageFeatures(
            tokenIDs: promptTokens,
            imageFeatures: visionEncoding.values,
            imageFeatureShape: visionEncoding.shape,
            imageTokenID: imageTokenID
        )
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
                nextTokenID = try container.decodeCmlxStep(tokenID: tokenID)
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

    private static func parse(arguments: [String]) throws -> VLMGenerateCLIOptions {
        var options = VLMGenerateCLIOptions()
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
            let option: String
            let value: String
            if let equals = raw.firstIndex(of: "=") {
                option = String(raw[..<equals])
                value = String(raw[raw.index(after: equals)...])
            } else {
                option = raw
                guard index + 1 < arguments.count else {
                    throw VLMGenerateCLIError.missingValue(option)
                }
                index += 1
                value = arguments[index]
            }

            switch option {
            case "--model":
                options.modelPath = value
            case "--image":
                options.imagePath = value
            case "--prompt":
                options.prompt = value
            case "--max-tokens":
                options.maxTokens = try parsePositiveInteger(option: option, value: value)
                options.maxTokensWasExplicit = true
            case "--think":
                options.thinkEnabled = try parseBoolean(option: option, value: value)
            case "--preserve-thinking":
                options.preserveThinking = try parseBoolean(option: option, value: value)
            case "--family":
                options.modelFamily = try parseFamily(value)
            case "--max-ops-per-buffer":
                options.maxOpsPerCommandBuffer = try parsePositiveInteger(option: option, value: value)
            case "--max-mb-per-buffer":
                options.maxMBPerCommandBuffer = try parsePositiveInteger(option: option, value: value)
            case "--quantized-cache-mb":
                options.quantizedCacheMB = try parseNonNegativeInteger(option: option, value: value)
            default:
                throw VLMGenerateCLIError.unknownOption(option)
            }
            index += 1
        }
        if options.thinkEnabled && !options.maxTokensWasExplicit {
            options.maxTokens = 2_048
        }
        return options
    }

    private static func imageTokenCount(
        for grid: QwenImageGridTHW,
        plan: QwenVLMRuntimePlan
    ) throws -> Int {
        let mergeSize = plan.visionConfiguration.spatialMergeSize
            ?? plan.imageProcessorConfiguration.mergeSize
            ?? 2
        let mergeArea = mergeSize * mergeSize
        guard mergeArea > 0, grid.product % mergeArea == 0 else {
            throw VLMGenerateCLIError.invalidVisionTokenCount
        }
        let count = grid.product / mergeArea
        guard count > 0 else {
            throw VLMGenerateCLIError.invalidVisionTokenCount
        }
        return count
    }

    private static func tokenID(_ token: String, tokenizer: Tokenizer) throws -> Int {
        guard let id = tokenizer.convertTokenToId(token) else {
            throw VLMGenerateCLIError.tokenizerMissingToken(token)
        }
        return id
    }

    private static func makePromptTokens(
        prompt: String,
        imageTokenCount: Int,
        tokenizer: Tokenizer,
        thinkEnabled: Bool,
        preserveThinking: Bool
    ) throws -> [Int] {
        let imagePadding = String(repeating: "<|image_pad|>", count: imageTokenCount)
        let visionBlock = "<|vision_start|>\(imagePadding)<|vision_end|>"
        let messages: [Message] = [
            [
                "role": "user",
                "content": visionBlock + prompt,
            ],
        ]
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

    private static func parseFamily(_ value: String) throws -> QwenVLMModelFamily {
        switch value.lowercased().replacingOccurrences(of: "_", with: "-") {
        case "qwen35-vlm", "qwen3.5-vlm", "qwen3-5-vlm", "qwen3p5-vlm", "qwen35", "qwen3.5":
            return .qwen35VLM
        case "qwen36-vlm", "qwen3.6-vlm", "qwen3-6-vlm", "qwen3p6-vlm", "qwen36", "qwen3.6":
            return .qwen36VLM
        default:
            throw VLMGenerateCLIError.invalidFamily(value)
        }
    }

    private static func parsePositiveInteger(option: String, value: String) throws -> Int {
        guard let parsed = Int(value), parsed > 0 else {
            throw VLMGenerateCLIError.invalidInteger(option: option, value: value)
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
            throw VLMGenerateCLIError.invalidBoolean(option: option, value: value)
        }
    }

    private static func parseNonNegativeInteger(option: String, value: String) throws -> Int {
        guard let parsed = Int(value), parsed >= 0 else {
            throw VLMGenerateCLIError.invalidInteger(option: option, value: value)
        }
        return parsed
    }

    private static var usage: String {
        """
        Usage:
          edge-vlm --model <model-dir> --image <image-path> [options]

        Options:
          --prompt <text>             Prompt text. Default: "Describe this image."
          --max-tokens <n>            Maximum generated tokens. Default: 200
          --think                     Enable Qwen thinking mode. Default max tokens becomes 2048 unless set.
          --preserve-thinking         Preserve historical assistant thinking in chat template.
          --family <qwen35-vlm|qwen36-vlm>
                                      Override family detection from config.json.
          --max-ops-per-buffer <n>    Native Metal op budget. Default: 700
          --max-mb-per-buffer <n>     Native Metal byte budget. Default: 256
          --quantized-cache-mb <n>    Cap native quantized Metal buffer cache. Default: 4096

        Example:
          swift run edge-vlm \\
            --model ~/Models/Qwen3.5-4B-4bit \\
            --image ~/Pictures/example.jpg \\
            --prompt "Describe this image." \\
            --max-tokens 200
        """
    }
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
