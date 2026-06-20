// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Darwin
import EdgeEngine
import Foundation

enum QwenSmokeCLIError: Error, CustomStringConvertible {
    case missingRequiredOption(String)
    case missingValue(String)
    case invalidInteger(option: String, value: String)
    case invalidTokenList(option: String, value: String)
    case invalidFamily(String)
    case unknownOption(String)
    case incompleteDynamicSchedule

    var description: String {
        switch self {
        case .missingRequiredOption(let option):
            "missing required option \(option)"
        case .missingValue(let option):
            "missing value for \(option)"
        case .invalidInteger(let option, let value):
            "invalid integer for \(option): \(value)"
        case .invalidTokenList(let option, let value):
            "invalid comma-separated token list for \(option): \(value)"
        case .invalidFamily(let value):
            "invalid family: \(value) (expected qwen35 or qwen36)"
        case .unknownOption(let option):
            "unknown option \(option)"
        case .incompleteDynamicSchedule:
            "dynamic schedule requires --dynamic-ops-floor, --dynamic-ops-low, and --dynamic-ops-high"
        }
    }
}

struct QwenSmokeCLIOptions {
    var modelPath: String?
    var tokenIds: [Int]?
    var preflight = false
    var readShardHeaders = true
    var requirePass = false
    var maxNewTokenCount = 1
    var prefillTopLogitCount = 5
    var stepTopLogitCount: Int?
    var endTokenIds = Set<Int>()
    var kvCapacity: Int?
    var family: QwenModelFamily?
    var maxOpsPerCommandBuffer: Int
    var maxMBPerCommandBuffer: Int
    var contextLengthHint: Int
    var quantizedBufferCacheLimitMB: Int?
    var releaseQuantizedHostStorageAfterUpload: Bool
    var dynamicOpsFloor: Int?
    var dynamicOpsLow: Int?
    var dynamicOpsHigh: Int?
    var decodeBackend: QwenBundleSmokeDecodeBackend = .swiftReference

    init(environment: [String: String]) {
        let defaults = MetalRuntimeConfiguration()
        self.maxOpsPerCommandBuffer = Self.environmentInt(
            named: "MLX_MAX_OPS_PER_BUFFER",
            environment: environment
        ) ?? defaults.maxOpsPerCommandBuffer
        self.maxMBPerCommandBuffer = Self.environmentInt(
            named: "MLX_MAX_MB_PER_BUFFER",
            environment: environment
        ) ?? defaults.maxMBPerCommandBuffer
        self.contextLengthHint = defaults.contextLengthHint
        self.quantizedBufferCacheLimitMB = Self.environmentInt(
            named: "EDGE_QUANTIZED_BUFFER_CACHE_MB",
            environment: environment
        )
        self.releaseQuantizedHostStorageAfterUpload = Self.environmentBool(
            named: "EDGE_RELEASE_QUANTIZED_HOST_STORAGE",
            environment: environment
        ) ?? false
    }

    func configuration() throws -> QwenBundleSmokeConfiguration {
        guard let modelPath else {
            throw QwenSmokeCLIError.missingRequiredOption("--model")
        }
        guard let tokenIds else {
            throw QwenSmokeCLIError.missingRequiredOption("--tokens")
        }
        let dynamicOpsSchedule = try makeDynamicOpsSchedule()
        return QwenBundleSmokeConfiguration(
            modelRootURL: URL(fileURLWithPath: expandedPath(modelPath)),
            promptTokenIds: tokenIds,
            maxNewTokenCount: maxNewTokenCount,
            endTokenIds: endTokenIds,
            kvCapacity: kvCapacity,
            family: family,
            runtimeConfiguration: MetalRuntimeConfiguration(
                maxOpsPerCommandBuffer: maxOpsPerCommandBuffer,
                maxMBPerCommandBuffer: maxMBPerCommandBuffer,
                contextLengthHint: contextLengthHint,
                dynamicOpsSchedule: dynamicOpsSchedule,
                quantizedBufferCacheLimitBytes: quantizedBufferCacheLimitMB.map { max(0, $0) * 1_048_576 },
                releaseQuantizedHostStorageAfterUpload: releaseQuantizedHostStorageAfterUpload
            ),
            prefillTopLogitCount: prefillTopLogitCount,
            stepTopLogitCount: stepTopLogitCount,
            decodeBackend: decodeBackend
        )
    }

    func preflightConfiguration() throws -> QwenBundlePreflightConfiguration {
        guard let modelPath else {
            throw QwenSmokeCLIError.missingRequiredOption("--model")
        }
        return QwenBundlePreflightConfiguration(
            modelRootURL: URL(fileURLWithPath: expandedPath(modelPath)),
            family: family,
            readShardHeaders: readShardHeaders
        )
    }

    private func makeDynamicOpsSchedule() throws -> DynamicOpsSchedule? {
        let values = [dynamicOpsFloor, dynamicOpsLow, dynamicOpsHigh]
        guard values.contains(where: { $0 != nil }) else {
            return nil
        }
        guard let floor = dynamicOpsFloor,
              let low = dynamicOpsLow,
              let high = dynamicOpsHigh
        else {
            throw QwenSmokeCLIError.incompleteDynamicSchedule
        }
        return DynamicOpsSchedule(
            floor: floor,
            contextLow: low,
            contextHigh: high
        )
    }

    private static func environmentInt(
        named name: String,
        environment: [String: String]
    ) -> Int? {
        guard let value = environment[name], !value.isEmpty else {
            return nil
        }
        return Int(value)
    }

    private static func environmentBool(
        named name: String,
        environment: [String: String]
    ) -> Bool? {
        guard let value = environment[name], !value.isEmpty else {
            return nil
        }
        switch value.lowercased() {
        case "1", "true", "yes":
            return true
        case "0", "false", "no":
            return false
        default:
            return nil
        }
    }
}

@main
enum EdgeSmokeCommand {
    static func main() {
        do {
            let arguments = Array(CommandLine.arguments.dropFirst())
            if arguments.contains("--help") || arguments.contains("-h") {
                print(Self.usage)
                return
            }
            let options = try parse(
                arguments: arguments,
                environment: ProcessInfo.processInfo.environment
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            if options.preflight {
                let result = try QwenBundlePreflightRunner.run(
                    configuration: try options.preflightConfiguration()
                )
                let data = try encoder.encode(result)
                FileHandle.standardOutput.write(data)
                FileHandle.standardOutput.write(Data("\n".utf8))
                if options.requirePass && !result.passesPreflight {
                    exit(3)
                }
            } else {
                let data = try encoder.encode(
                    QwenBundleSmokeRunner.run(
                        configuration: try options.configuration()
                    )
                )
                FileHandle.standardOutput.write(data)
                FileHandle.standardOutput.write(Data("\n".utf8))
            }
        } catch let error as QwenSmokeCLIError {
            fputs("error: \(error.description)\n\n\(Self.usage)\n", stderr)
            exit(2)
        } catch {
            fputs("error: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func parse(
        arguments: [String],
        environment: [String: String]
    ) throws -> QwenSmokeCLIOptions {
        var options = QwenSmokeCLIOptions(environment: environment)
        var index = 0
        while index < arguments.count {
            let raw = arguments[index]
            if raw == "--preflight" {
                options.preflight = true
                index += 1
                continue
            }
            if raw == "--no-read-shard-headers" {
                options.readShardHeaders = false
                index += 1
                continue
            }
            if raw == "--require-pass" {
                options.requirePass = true
                index += 1
                continue
            }
            if raw == "--cmlx-decode-step" {
                options.decodeBackend = .cmlxDecodeStep
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
                    throw QwenSmokeCLIError.missingValue(option)
                }
                index += 1
                value = arguments[index]
            }

            switch option {
            case "--model":
                options.modelPath = value
            case "--preflight":
                options.preflight = try parseBool(option: option, value: value)
            case "--read-shard-headers":
                options.readShardHeaders = try parseBool(option: option, value: value)
            case "--no-read-shard-headers":
                options.readShardHeaders = !(try parseBool(option: option, value: value))
            case "--require-pass":
                options.requirePass = try parseBool(option: option, value: value)
            case "--tokens", "--prompt-tokens":
                options.tokenIds = try parseTokenList(option: option, value: value)
            case "--max-new-tokens":
                options.maxNewTokenCount = try parseInt(option: option, value: value)
            case "--prefill-top-logits":
                options.prefillTopLogitCount = try parseInt(option: option, value: value)
            case "--step-top-logits":
                options.stepTopLogitCount = try parseInt(option: option, value: value)
            case "--end-tokens":
                options.endTokenIds = Set(try parseTokenList(option: option, value: value))
            case "--kv-capacity":
                options.kvCapacity = try parseInt(option: option, value: value)
            case "--family":
                options.family = try parseFamily(value)
            case "--max-ops-per-buffer":
                options.maxOpsPerCommandBuffer = try parseInt(option: option, value: value)
            case "--max-mb-per-buffer":
                options.maxMBPerCommandBuffer = try parseInt(option: option, value: value)
            case "--context-length-hint":
                options.contextLengthHint = try parseInt(option: option, value: value)
            case "--quantized-cache-mb":
                options.quantizedBufferCacheLimitMB = try parseInt(option: option, value: value)
            case "--release-quantized-host-storage":
                options.releaseQuantizedHostStorageAfterUpload = try parseBool(option: option, value: value)
            case "--decode-backend":
                options.decodeBackend = try parseDecodeBackend(value)
            case "--cmlx-decode-step":
                options.decodeBackend = try parseBool(option: option, value: value) ? .cmlxDecodeStep : .swiftReference
            case "--dynamic-ops-floor":
                options.dynamicOpsFloor = try parseInt(option: option, value: value)
            case "--dynamic-ops-low":
                options.dynamicOpsLow = try parseInt(option: option, value: value)
            case "--dynamic-ops-high":
                options.dynamicOpsHigh = try parseInt(option: option, value: value)
            default:
                throw QwenSmokeCLIError.unknownOption(option)
            }
            index += 1
        }
        return options
    }

    private static func parseInt(option: String, value: String) throws -> Int {
        guard let parsed = Int(value) else {
            throw QwenSmokeCLIError.invalidInteger(option: option, value: value)
        }
        return parsed
    }

    private static func parseBool(option: String, value: String) throws -> Bool {
        switch value.lowercased() {
        case "1", "true", "yes":
            return true
        case "0", "false", "no":
            return false
        default:
            throw QwenSmokeCLIError.invalidInteger(option: option, value: value)
        }
    }

    private static func parseTokenList(option: String, value: String) throws -> [Int] {
        let parts = value.split(separator: ",", omittingEmptySubsequences: false)
        guard !parts.isEmpty else {
            throw QwenSmokeCLIError.invalidTokenList(option: option, value: value)
        }
        return try parts.map { part in
            guard let tokenId = Int(part.trimmingCharacters(in: .whitespacesAndNewlines)) else {
                throw QwenSmokeCLIError.invalidTokenList(option: option, value: value)
            }
            return tokenId
        }
    }

    private static func parseFamily(_ value: String) throws -> QwenModelFamily {
        switch value.lowercased().replacingOccurrences(of: "_", with: "") {
        case "qwen35":
            return .qwen35
        case "qwen36":
            return .qwen36
        default:
            throw QwenSmokeCLIError.invalidFamily(value)
        }
    }

    private static func parseDecodeBackend(_ value: String) throws -> QwenBundleSmokeDecodeBackend {
        switch value.lowercased().replacingOccurrences(of: "_", with: "-") {
        case "swift", "swift-reference", "reference":
            return .swiftReference
        case "cmlx", "cmlx-decode-step", "decode-step":
            return .cmlxDecodeStep
        default:
            throw QwenSmokeCLIError.unknownOption("--decode-backend=\(value)")
        }
    }

    private static var usage: String {
        """
        Usage:
          edge-smoke --model <model-dir> --tokens <id,id,...> [options]
          edge-smoke --model <model-dir> --preflight [options]

        Options:
          --preflight                 Validate config/index/shard headers without loading tensor payloads.
          --read-shard-headers <bool> Enable or disable safetensors header reads during preflight. Default: true
          --no-read-shard-headers     Skip safetensors header reads during preflight.
          --require-pass              Exit with code 3 when preflight does not pass.
          --max-new-tokens <n>        Greedy tokens to generate after prefill. Default: 1
          --prefill-top-logits <n>    Top logits to include from the final prefill row. Default: 5
          --step-top-logits <n>       Top logits to include for each generated step. Default: prefill top-k
          --end-tokens <id,id,...>    Stop when any generated token matches.
          --kv-capacity <n>           Override KV cache capacity.
          --family <qwen35|qwen36>    Override family detection from config.json.
          --max-ops-per-buffer <n>    Native Metal op budget. Defaults to MLX_MAX_OPS_PER_BUFFER or 20.
          --max-mb-per-buffer <n>     Native Metal byte budget. Defaults to MLX_MAX_MB_PER_BUFFER or 40.
          --context-length-hint <n>   Scheduling context hint. Defaults to prompt + max-new.
          --quantized-cache-mb <n>    Cap native quantized Metal buffer cache; 0 disables long-lived cache.
          --release-quantized-host-storage <bool>
                                      Release quantized host arrays after cached Metal upload.
          --decode-backend <swift|cmlx>
                                      Select smoke decode backend. Default: swift.
          --cmlx-decode-step [bool]   Shortcut for --decode-backend cmlx.
          --dynamic-ops-floor <n>     Enable dynamic op taper floor.
          --dynamic-ops-low <n>       Dynamic taper starts above this context length.
          --dynamic-ops-high <n>      Dynamic taper reaches floor at this context length.
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
