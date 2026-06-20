// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Darwin
import EdgeEngine
import Foundation

enum TTSGenerateCLIError: Error, CustomStringConvertible {
    case missingRequiredOption(String)
    case missingValue(String)
    case invalidInteger(option: String, value: String)
    case invalidFloat(option: String, value: String)
    case unknownOption(String)
    case invalidModelConfiguration
    case missingSafeTensors
    case missingSpeechTokenizer
    case missingSpeaker(String)
    case missingLanguage(String)

    var description: String {
        switch self {
        case .missingRequiredOption(let option):
            "missing required option \(option)"
        case .missingValue(let option):
            "missing value for \(option)"
        case .invalidInteger(let option, let value):
            "invalid integer for \(option): \(value)"
        case .invalidFloat(let option, let value):
            "invalid decimal for \(option): \(value)"
        case .unknownOption(let option):
            "unknown option \(option)"
        case .invalidModelConfiguration:
            "model config is not a Qwen3-TTS config"
        case .missingSafeTensors:
            "model does not contain safetensors weights"
        case .missingSpeechTokenizer:
            "model does not contain speech tokenizer weights"
        case .missingSpeaker(let speaker):
            "speaker not found: \(speaker)"
        case .missingLanguage(let language):
            "language not found: \(language)"
        }
    }
}

struct TTSGenerateCLIOptions {
    var modelPath: String?
    var text = "Hello world."
    var speaker: String?
    var language = "auto"
    var maxTokens = 16
    var temperature: Float = 0.9
    var topK = 50
    var seed: UInt64 = 0
    var outputPath: String?
    var dumpPrompt = false
    var maxOpsPerCommandBuffer = 700
    var maxMBPerCommandBuffer = 256
    var quantizedCacheMB = 4_096

    var modelURL: URL {
        get throws {
            guard let modelPath else {
                throw TTSGenerateCLIError.missingRequiredOption("--model")
            }
            return URL(fileURLWithPath: expandedPath(modelPath))
        }
    }
}

@main
enum EdgeTTSCommand {
    static func main() async {
        do {
            let arguments = Array(CommandLine.arguments.dropFirst())
            if arguments.contains("--help") || arguments.contains("-h") {
                print(Self.usage)
                return
            }
            let options = try parse(arguments: arguments)
            try await run(options: options)
        } catch let error as TTSGenerateCLIError {
            fputs("error: \(error.description)\n\n\(Self.usage)\n", stderr)
            exit(2)
        } catch {
            fputs("error: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func run(options: TTSGenerateCLIOptions) async throws {
        var memorySampler = CLIMemorySampler()
        let modelURL = try options.modelURL
        if options.dumpPrompt {
            FileHandle.standardOutput.write(Data(Qwen3TTSNativeSession.targetPrompt(for: options.text).utf8))
            return
        }

        let targetTokenCount = try await Qwen3TTSNativeSession.targetTokenCount(
            modelURL: modelURL,
            text: options.text
        )
        let runtimeConfiguration = NativeRuntimeBridge.metalConfiguration(
            maxOpsPerCommandBuffer: options.maxOpsPerCommandBuffer,
            maxMBPerCommandBuffer: options.maxMBPerCommandBuffer,
            contextLengthHint: targetTokenCount + options.maxTokens + 16,
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
        let session = try await Qwen3TTSNativeSession(
            modelURL: modelURL,
            runtimeConfiguration: runtimeConfiguration
        )
        memorySampler.sample()

        let result = try session.synthesize(
            Qwen3TTSSynthesisRequest(
                text: options.text,
                speaker: options.speaker,
                language: options.language,
                maxTokens: options.maxTokens,
                temperature: options.temperature,
                topK: options.topK,
                seed: options.seed,
                decodeAudio: options.outputPath != nil
            )
        )
        memorySampler.sample()

        let codes = result.codecTokens
        let preview = codes.values.prefix(min(codes.values.count, 32)).map(String.init).joined(separator: ", ")
        print("Codec tokens shape: \(codes.shape)")
        print("Preview: [\(preview)]")
        print("Generation: \(codes.shape.first ?? 0) steps, \(String(format: "%.1f", Double(codes.shape.first ?? 0) / max(result.generationSeconds, 0.001))) steps-per-sec")
        if let outputPath = options.outputPath {
            guard let audio = result.audio else {
                throw TTSGenerateCLIError.missingSpeechTokenizer
            }
            let wavData = try EdgeWAVFile.encodePCM16(audio)
            let outputURL = URL(fileURLWithPath: expandedPath(outputPath))
            try FileManager.default.createDirectory(
                at: outputURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try wavData.write(to: outputURL, options: [.atomic])
            print("Audio: \(audio.frameCount) samples, \(String(format: "%.3f", audio.durationSeconds))s, \(String(format: "%.2f", result.decodeSeconds ?? 0))s decode")
            print("WAV: \(outputURL.path)")
        }
        print("Peak memory: \(formatBytes(memorySampler.peakPhysicalFootprintBytes))")
    }

    private static func parse(arguments: [String]) throws -> TTSGenerateCLIOptions {
        var options = TTSGenerateCLIOptions()
        var index = 0
        while index < arguments.count {
            let raw = arguments[index]
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
                    throw TTSGenerateCLIError.missingValue(option)
                }
                index += 1
                value = arguments[index]
            }
            switch option {
            case "--model":
                options.modelPath = value
            case "--text":
                options.text = value
            case "--speaker":
                options.speaker = value
            case "--language":
                options.language = value
            case "--max-tokens":
                options.maxTokens = try parsePositiveInteger(option: option, value: value)
            case "--temperature":
                options.temperature = try parseNonNegativeFloat(option: option, value: value)
            case "--top-k":
                options.topK = try parseNonNegativeInteger(option: option, value: value)
            case "--seed":
                options.seed = try parseNonNegativeUInt64(option: option, value: value)
            case "--output", "--wav":
                options.outputPath = value
            case "--dump-prompt":
                options.dumpPrompt = try parseBoolean(option: option, value: value)
            case "--max-ops-per-buffer":
                options.maxOpsPerCommandBuffer = try parsePositiveInteger(option: option, value: value)
            case "--max-mb-per-buffer":
                options.maxMBPerCommandBuffer = try parsePositiveInteger(option: option, value: value)
            case "--quantized-cache-mb":
                options.quantizedCacheMB = try parseNonNegativeInteger(option: option, value: value)
            default:
                throw TTSGenerateCLIError.unknownOption(option)
            }
            index += 1
        }
        return options
    }

    private static func parsePositiveInteger(option: String, value: String) throws -> Int {
        guard let parsed = Int(value), parsed > 0 else {
            throw TTSGenerateCLIError.invalidInteger(option: option, value: value)
        }
        return parsed
    }

    private static func parseNonNegativeInteger(option: String, value: String) throws -> Int {
        guard let parsed = Int(value), parsed >= 0 else {
            throw TTSGenerateCLIError.invalidInteger(option: option, value: value)
        }
        return parsed
    }

    private static func parseNonNegativeUInt64(option: String, value: String) throws -> UInt64 {
        guard let parsed = UInt64(value) else {
            throw TTSGenerateCLIError.invalidInteger(option: option, value: value)
        }
        return parsed
    }

    private static func parseNonNegativeFloat(option: String, value: String) throws -> Float {
        guard let parsed = Float(value), parsed >= 0, parsed.isFinite else {
            throw TTSGenerateCLIError.invalidFloat(option: option, value: value)
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
            throw TTSGenerateCLIError.invalidInteger(option: option, value: value)
        }
    }

    private static var usage: String {
        """
        Usage:
          edge-tts --model <model-dir> [options]

        Options:
          --text <text>               Text to synthesize. Default: Hello world.
          --speaker <name>            CustomVoice speaker. Default: serena if available.
          --language <name|auto>      Codec language prefix. Default: auto.
          --max-tokens <n>            Maximum codec steps. Default: 16
          --temperature <value>       Codec sampling temperature, 0 = greedy. Default: 0.9
          --top-k <n>                 Codec top-k sampling, 0 = disabled. Default: 50
          --seed <n>                  Codec sampling seed. Default: 0
          --output <wav-file>         Decode codec tokens and write PCM16 WAV.
          --dump-prompt               Print the target TTS prompt and exit before loading weights.
          --max-ops-per-buffer <n>    Native Metal op budget. Default: 700
          --max-mb-per-buffer <n>     Native Metal byte budget. Default: 256
          --quantized-cache-mb <n>    Cap native quantized Metal buffer cache. Default: 4096
        """
    }
}

private struct CLIMemorySampler {
    private(set) var peakPhysicalFootprintBytes: UInt64 = 0

    mutating func sample() {
        peakPhysicalFootprintBytes = max(peakPhysicalFootprintBytes, currentPhysicalFootprintBytes())
    }
}

private func currentPhysicalFootprintBytes() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    let result = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), rebound, &count)
        }
    }
    guard result == KERN_SUCCESS else { return 0 }
    return UInt64(info.resident_size)
}

private func formatBytes(_ bytes: UInt64) -> String {
    String(format: "%.3f GB", Double(bytes) / 1_073_741_824.0)
}

private func expandedPath(_ path: String) -> String {
    guard path.hasPrefix("~") else { return path }
    let home = FileManager.default.homeDirectoryForCurrentUser.path
    if path == "~" { return home }
    if path.hasPrefix("~/") {
        return home + String(path.dropFirst())
    }
    return path
}
