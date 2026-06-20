// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import AVFoundation
import Darwin
import EdgeEngine
import Foundation

enum ASRGenerateCLIError: Error, CustomStringConvertible {
    case missingRequiredOption(String)
    case missingValue(String)
    case invalidInteger(option: String, value: String)
    case invalidDouble(option: String, value: String)
    case invalidBoolean(option: String, value: String)
    case unknownOption(String)
    case invalidModelConfiguration
    case missingSafeTensors
    case tokenizerMissingToken(String)
    case audioFeatureMismatch

    var description: String {
        switch self {
        case .missingRequiredOption(let option):
            "missing required option \(option)"
        case .missingValue(let option):
            "missing value for \(option)"
        case .invalidInteger(let option, let value):
            "invalid integer for \(option): \(value)"
        case .invalidDouble(let option, let value):
            "invalid decimal for \(option): \(value)"
        case .invalidBoolean(let option, let value):
            "invalid boolean for \(option): \(value)"
        case .unknownOption(let option):
            "unknown option \(option)"
        case .invalidModelConfiguration:
            "model config is not a Qwen3-ASR config"
        case .missingSafeTensors:
            "model does not contain safetensors weights"
        case .tokenizerMissingToken(let token):
            "tokenizer missing required token \(token)"
        case .audioFeatureMismatch:
            "audio feature/token count mismatch"
        }
    }
}

struct ASRGenerateCLIOptions {
    var modelPath: String?
    var audioPath: String?
    var frameCount = 100
    var audioPrefix = "audio_tower"
    var language = "English"
    var maxTokens = 200
    var maxAudioSeconds: Double? = 30
    var encoderOnly = false
    var dumpPrompt = false
    var maxOpsPerCommandBuffer = 700
    var maxMBPerCommandBuffer = 256
    var quantizedCacheMB = 4_096

    var modelURL: URL {
        get throws {
            guard let modelPath else {
                throw ASRGenerateCLIError.missingRequiredOption("--model")
            }
            return URL(fileURLWithPath: expandedPath(modelPath))
        }
    }

    var audioURL: URL? {
        audioPath.map { URL(fileURLWithPath: expandedPath($0)) }
    }
}

struct ASRGenerateSmokeReport: Codable {
    var modelPath: String
    var audioPrefix: String
    var inputShape: [Int]
    var outputShape: [Int]
    var outputIsFinite: Bool
    var preview: [Float]
}

@main
enum EdgeASRCommand {
    static func main() async {
        do {
            let arguments = Array(CommandLine.arguments.dropFirst())
            if arguments.contains("--help") || arguments.contains("-h") {
                print(Self.usage)
                return
            }
            let options = try parse(arguments: arguments)
            try await run(options: options)
        } catch let error as ASRGenerateCLIError {
            fputs("error: \(error.description)\n\n\(Self.usage)\n", stderr)
            exit(2)
        } catch {
            fputs("error: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func run(options: ASRGenerateCLIOptions) async throws {
        let modelURL = try options.modelURL
        let shouldDecodeText = options.audioURL != nil && !options.encoderOnly
        let runtimeConfiguration = NativeRuntimeBridge.metalConfiguration(
            maxOpsPerCommandBuffer: options.maxOpsPerCommandBuffer,
            maxMBPerCommandBuffer: options.maxMBPerCommandBuffer,
            contextLengthHint: options.maxTokens + 1_024,
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
        let session = try await Qwen3ASRNativeSession(
            modelURL: modelURL,
            audioPrefix: options.audioPrefix,
            mode: shouldDecodeText ? .transcription : .encoderOnly,
            runtimeConfiguration: runtimeConfiguration
        )

        if !shouldDecodeText {
            let encoding: Qwen3ASRAudioEncodingResult
            if let audioURL = options.audioURL {
                let audio = try loadAudioBuffer(from: audioURL, targetSampleRate: 16_000)
                encoding = try session.encode(
                    audio: audio,
                    maxAudioSeconds: options.maxAudioSeconds
                )
            } else {
                let features = makeDeterministicMelFeatures(
                    frameCount: options.frameCount,
                    melBinCount: session.metadata.audioMelBinCount
                )
                encoding = try session.encode(
                    logMelFeatures: features,
                    featureShape: [options.frameCount, session.metadata.audioMelBinCount]
                )
            }
            let report = ASRGenerateSmokeReport(
                modelPath: modelURL.path,
                audioPrefix: options.audioPrefix,
                inputShape: [
                    encoding.inputFrameCount,
                    session.metadata.audioMelBinCount,
                ],
                outputShape: encoding.encoding.shape,
                outputIsFinite: encoding.encoding.values.allSatisfy(\.isFinite),
                preview: Array(encoding.encoding.values.prefix(8))
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(report)
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write(Data("\n".utf8))
            return
        }

        guard let audioURL = options.audioURL else { return }
        let audio = try loadAudioBuffer(from: audioURL, targetSampleRate: 16_000)
        if options.dumpPrompt {
            let encoding = try session.encode(
                audio: audio,
                maxAudioSeconds: options.maxAudioSeconds
            )
            let decodedPrompt = try session.decodedPrompt(
                audioTokenCount: encoding.encoding.shape[0],
                language: options.language
            )
            FileHandle.standardOutput.write(Data(decodedPrompt.utf8))
            if !decodedPrompt.hasSuffix("\n") {
                FileHandle.standardOutput.write(Data("\n".utf8))
            }
            return
        }

        var memorySampler = CLIMemorySampler()
        memorySampler.sample()
        let result = try session.transcribe(
            Qwen3ASRTranscriptionRequest(
                audio: audio,
                language: options.language,
                maxTokens: options.maxTokens,
                maxAudioSeconds: options.maxAudioSeconds
            ),
            onTextDelta: { delta in
                if !delta.isEmpty {
                    FileHandle.standardOutput.write(Data(delta.utf8))
                    fflush(stdout)
                }
            }
        )
        memorySampler.sample()
        if !result.text.hasSuffix("\n") {
            FileHandle.standardOutput.write(Data("\n".utf8))
        }
        writeGenerationFooter(
            promptTokenCount: result.promptTokenCount,
            audioTokenCount: result.audioTokenCount,
            inputFrameCount: result.inputFrameCount,
            prefillSeconds: result.prefillSeconds,
            generatedTokenCount: result.generatedTokenIDs.count,
            decodeSeconds: result.decodeSeconds,
            peakPhysicalFootprintBytes: memorySampler.peakPhysicalFootprintBytes
        )
    }

    private static func parse(arguments: [String]) throws -> ASRGenerateCLIOptions {
        var options = ASRGenerateCLIOptions()
        var index = 0
        while index < arguments.count {
            let raw = arguments[index]
            if raw == "--encoder-only" {
                options.encoderOnly = true
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
                    throw ASRGenerateCLIError.missingValue(option)
                }
                index += 1
                value = arguments[index]
            }
            switch option {
            case "--model":
                options.modelPath = value
            case "--audio":
                options.audioPath = value
            case "--frames":
                options.frameCount = try parsePositiveInt(option: option, value: value)
            case "--audio-prefix":
                options.audioPrefix = value
            case "--language":
                options.language = value
            case "--max-tokens":
                options.maxTokens = try parsePositiveInt(option: option, value: value)
            case "--max-audio-seconds":
                let seconds = try parseNonNegativeDouble(option: option, value: value)
                options.maxAudioSeconds = seconds > 0 ? seconds : nil
            case "--encoder-only":
                options.encoderOnly = try parseBoolean(option: option, value: value)
            case "--dump-prompt":
                options.dumpPrompt = try parseBoolean(option: option, value: value)
            case "--max-ops-per-buffer":
                options.maxOpsPerCommandBuffer = try parsePositiveInt(option: option, value: value)
            case "--max-mb-per-buffer":
                options.maxMBPerCommandBuffer = try parsePositiveInt(option: option, value: value)
            case "--quantized-cache-mb":
                options.quantizedCacheMB = try parseNonNegativeInt(option: option, value: value)
            default:
                throw ASRGenerateCLIError.unknownOption(option)
            }
            index += 1
        }
        return options
    }

    private static func parsePositiveInt(option: String, value: String) throws -> Int {
        guard let parsed = Int(value), parsed > 0 else {
            throw ASRGenerateCLIError.invalidInteger(option: option, value: value)
        }
        return parsed
    }

    private static func parseNonNegativeInt(option: String, value: String) throws -> Int {
        guard let parsed = Int(value), parsed >= 0 else {
            throw ASRGenerateCLIError.invalidInteger(option: option, value: value)
        }
        return parsed
    }

    private static func parsePositiveDouble(option: String, value: String) throws -> Double {
        guard let parsed = Double(value), parsed > 0 else {
            throw ASRGenerateCLIError.invalidDouble(option: option, value: value)
        }
        return parsed
    }

    private static func parseNonNegativeDouble(option: String, value: String) throws -> Double {
        guard let parsed = Double(value), parsed >= 0 else {
            throw ASRGenerateCLIError.invalidDouble(option: option, value: value)
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
            throw ASRGenerateCLIError.invalidBoolean(option: option, value: value)
        }
    }

    private static var usage: String {
        """
        Usage:
          edge-asr --model <model-dir> [--frames <count>]
          edge-asr --model <model-dir> --audio <audio-file> [options]

        Modes:
          Without --audio, runs Qwen3-ASR audio_tower encoder smoke with
          deterministic log-mel features and prints JSON shape diagnostics.
          With --audio, runs native ASR text smoke: audio_tower -> media prefill
          -> Qwen text decoder.

        Options:
          --model <model-dir>         Local Qwen3-ASR bundle directory.
          --audio <audio-file>        MP3/WAV/M4A readable by AVFoundation.
          --frames <count>            Encoder-only log-mel frame count. Default: 100.
          --language <name>           Prompt language. Default: English.
          --max-tokens <n>            Maximum generated tokens. Default: 200.
          --max-audio-seconds <sec>   Truncate audio for smoke tests. Default: 30, 0 = full audio.
          --encoder-only              Force encoder-only mode even with --audio.
          --dump-prompt               Print decoded ASR prompt and exit.
          --audio-prefix <prefix>     Safetensors prefix. Default: audio_tower.
          --max-ops-per-buffer <n>    Native Metal op budget. Default: 700.
          --max-mb-per-buffer <n>     Native Metal byte budget. Default: 256.
          --quantized-cache-mb <n>    Cap native quantized Metal buffer cache. Default: 4096.
        """
    }
}

private func loadAudioBuffer(from url: URL, targetSampleRate: Int) throws -> EdgeAudioBuffer {
    let file = try AVAudioFile(forReading: url)
    let format = file.processingFormat
    let frameCount = AVAudioFrameCount(file.length)
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
        throw EdgeAudioError.emptySamples
    }
    try file.read(into: buffer)
    guard let channelData = buffer.floatChannelData else {
        throw EdgeAudioError.emptySamples
    }
    let channels = Int(format.channelCount)
    let frames = Int(buffer.frameLength)
    var samples = [Float](repeating: 0, count: frames)
    for frame in 0..<frames {
        var sum = Float.zero
        for channel in 0..<channels {
            sum += channelData[channel][frame]
        }
        samples[frame] = sum / Float(channels)
    }
    let audio = try EdgeAudioBuffer(
        sampleRate: Int(format.sampleRate),
        channelCount: 1,
        interleavedSamples: samples
    )
    return try audio.resampled(to: targetSampleRate)
}

private func makeDeterministicMelFeatures(frameCount: Int, melBinCount: Int) -> [Float] {
    var values = [Float]()
    values.reserveCapacity(frameCount * melBinCount)
    for frame in 0..<frameCount {
        for mel in 0..<melBinCount {
            let wave = Foundation.sin(Double(frame + 1) * 0.013) * Foundation.cos(Double(mel + 3) * 0.017)
            values.append(Float(wave) * 0.1)
        }
    }
    return values
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
    audioTokenCount: Int,
    inputFrameCount: Int,
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
        Audio frames: %d
        Audio tokens: %d
        Prompt: %d tokens, %.1f tokens-per-sec
        Generation: %d tokens, %.1f tokens-per-sec
        Peak memory: %.3f GB
        ==========

        """,
        inputFrameCount,
        audioTokenCount,
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
