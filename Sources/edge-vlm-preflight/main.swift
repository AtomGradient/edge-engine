// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Darwin
import EdgeEngine
import Foundation

enum VLMPreflightCLIError: Error, CustomStringConvertible {
    case missingRequiredOption(String)
    case missingValue(String)
    case invalidBoolean(option: String, value: String)
    case invalidInteger(option: String, value: String)
    case invalidFamily(String)
    case unknownOption(String)

    var description: String {
        switch self {
        case .missingRequiredOption(let option):
            "missing required option \(option)"
        case .missingValue(let option):
            "missing value for \(option)"
        case .invalidBoolean(let option, let value):
            "invalid boolean for \(option): \(value)"
        case .invalidInteger(let option, let value):
            "invalid integer for \(option): \(value)"
        case .invalidFamily(let value):
            "invalid family: \(value) (expected qwen35-vlm or qwen36-vlm)"
        case .unknownOption(let option):
            "unknown option \(option)"
        }
    }
}

struct VLMPreflightCLIOptions {
    var modelPath: String?
    var modelFamily: QwenVLMModelFamily?
    var requirePass = false
    var loadIndex = false
    var jetsamLimitMB: Int?
    var appReserveMB = 1_024
    var structureBaselineMB = 256
    var activationReserveMB = 800

    func configuration() throws -> QwenVLMBundlePreflightConfiguration {
        guard let modelPath else {
            throw VLMPreflightCLIError.missingRequiredOption("--model")
        }
        return QwenVLMBundlePreflightConfiguration(
            modelRootURL: URL(fileURLWithPath: expandedPath(modelPath)),
            modelFamily: modelFamily
        )
    }

    func modelRootURL() throws -> URL {
        guard let modelPath else {
            throw VLMPreflightCLIError.missingRequiredOption("--model")
        }
        return URL(fileURLWithPath: expandedPath(modelPath))
    }
}

struct VLMIndexSummary: Codable, Equatable {
    var family: QwenVLMModelFamily
    var weightMapTensorCount: Int
    var languagePrefix: String
    var languageTensorCount: Int
    var visionPrefix: String
    var visionTensorCount: Int
    var footprint: QwenVLMWeightFootprint
}

struct VLMIndexPreflightReport: Codable, Equatable {
    var preflight: QwenVLMBundlePreflightResult
    var index: VLMIndexSummary
    var phasedLoadingPlan: QwenVLMPhasedLoadingPlan?
}

@main
enum EdgeVLMPreflightCommand {
    static func main() {
        do {
            let arguments = Array(CommandLine.arguments.dropFirst())
            if arguments.contains("--help") || arguments.contains("-h") {
                print(Self.usage)
                return
            }
            let options = try parse(arguments: arguments)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let result = try QwenVLMBundlePreflightRunner.run(
                configuration: try options.configuration()
            )
            let data: Data
            if options.loadIndex {
                let index = try QwenVLMModelBundleIndex.load(
                    from: options.modelRootURL(),
                    family: options.modelFamily
                )
                let footprint = try index.makeWeightFootprint()
                let plan = try options.jetsamLimitMB.map {
                    try index.makePhasedLoadingPlan(
                        jetsamLimitMB: $0,
                        appReserveMB: options.appReserveMB,
                        structureBaselineMB: options.structureBaselineMB,
                        activationReserveMB: options.activationReserveMB
                    )
                }
                data = try encoder.encode(
                    VLMIndexPreflightReport(
                        preflight: result,
                        index: VLMIndexSummary(
                            family: index.preflightResult.plan.modelFamily,
                            weightMapTensorCount: index.weightMap.count,
                            languagePrefix: index.languageManifest.prefix,
                            languageTensorCount: index.languageManifest.tensorNames.count,
                            visionPrefix: index.visionManifest.prefix,
                            visionTensorCount: index.visionManifest.tensorNames.count,
                            footprint: footprint
                        ),
                        phasedLoadingPlan: plan
                    )
                )
            } else {
                data = try encoder.encode(result)
            }
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write(Data("\n".utf8))
            if options.requirePass && !result.passesPreflight {
                exit(3)
            }
        } catch let error as VLMPreflightCLIError {
            fputs("error: \(error.description)\n\n\(Self.usage)\n", stderr)
            exit(2)
        } catch {
            fputs("error: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func parse(arguments: [String]) throws -> VLMPreflightCLIOptions {
        var options = VLMPreflightCLIOptions()
        var index = 0
        while index < arguments.count {
            let raw = arguments[index]
            if raw == "--require-pass" {
                options.requirePass = true
                index += 1
                continue
            }
            if raw == "--load-index" {
                options.loadIndex = true
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
                    throw VLMPreflightCLIError.missingValue(option)
                }
                index += 1
                value = arguments[index]
            }

            switch option {
            case "--model":
                options.modelPath = value
            case "--family":
                options.modelFamily = try parseFamily(value)
            case "--require-pass":
                options.requirePass = try parseBool(option: option, value: value)
            case "--load-index":
                options.loadIndex = try parseBool(option: option, value: value)
            case "--jetsam-limit-mb":
                options.jetsamLimitMB = try parseInteger(option: option, value: value)
                options.loadIndex = true
            case "--app-reserve-mb":
                options.appReserveMB = try parseInteger(option: option, value: value)
            case "--structure-baseline-mb":
                options.structureBaselineMB = try parseInteger(option: option, value: value)
            case "--activation-reserve-mb":
                options.activationReserveMB = try parseInteger(option: option, value: value)
            default:
                throw VLMPreflightCLIError.unknownOption(option)
            }
            index += 1
        }
        return options
    }

    private static func parseFamily(_ value: String) throws -> QwenVLMModelFamily {
        switch value.lowercased().replacingOccurrences(of: "_", with: "-") {
        case "qwen35-vlm", "qwen3.5-vlm", "qwen3-5-vlm", "qwen3p5-vlm", "qwen35", "qwen3.5":
            return .qwen35VLM
        case "qwen36-vlm", "qwen3.6-vlm", "qwen3-6-vlm", "qwen3p6-vlm", "qwen36", "qwen3.6":
            return .qwen36VLM
        default:
            throw VLMPreflightCLIError.invalidFamily(value)
        }
    }

    private static func parseBool(option: String, value: String) throws -> Bool {
        switch value.lowercased() {
        case "1", "true", "yes":
            return true
        case "0", "false", "no":
            return false
        default:
            throw VLMPreflightCLIError.invalidBoolean(option: option, value: value)
        }
    }

    private static func parseInteger(option: String, value: String) throws -> Int {
        guard let parsed = Int(value) else {
            throw VLMPreflightCLIError.invalidInteger(option: option, value: value)
        }
        return parsed
    }

    private static var usage: String {
        """
        Usage:
          edge-vlm-preflight --model <model-dir> [--family <qwen35-vlm|qwen36-vlm>] [--load-index]

        Options:
          --model <model-dir>         Local Qwen3.5/3.6 VLM bundle directory.
          --family <family>           Optional explicit family guard.
                                      Accepted values: qwen35-vlm, qwen3.5-vlm,
                                      qwen36-vlm, qwen3.6-vlm.
          --require-pass              Exit with code 3 when preflight does not pass.
          --load-index                Also load the native model bundle index and
                                      emit tensor group counts plus footprint.
          --jetsam-limit-mb <mb>      Also emit the phased loading plan for this
                                      device jetsam budget. Implies --load-index.
          --app-reserve-mb <mb>       App-side memory reserve for plan calculation.
          --structure-baseline-mb <mb>
                                      Structure/tokenizer/runtime baseline for plan.
          --activation-reserve-mb <mb>
                                      Activation scratch reserve for plan calculation.

        Output:
          JSON VLM preflight report by default. With --load-index, the command also
          verifies QwenVLMModelBundleIndex.load, safetensors metadata footprint, and
          optional phased loading plan without importing MLX or loading tensor data.
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
