// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Darwin
import EdgeEngine
import Foundation

enum SpeechPreflightCLIError: Error, CustomStringConvertible {
    case missingRequiredOption(String)
    case missingValue(String)
    case invalidBoolean(option: String, value: String)
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
        case .invalidFamily(let value):
            "invalid family: \(value) (expected qwen3-asr or qwen3-tts)"
        case .unknownOption(let option):
            "unknown option \(option)"
        }
    }
}

struct SpeechPreflightCLIOptions {
    var modelPath: String?
    var modelFamily: EdgeSpeechModelFamily?
    var requirePass = false

    func configuration() throws -> EdgeSpeechBundlePreflightConfiguration {
        guard let modelPath else {
            throw SpeechPreflightCLIError.missingRequiredOption("--model")
        }
        return EdgeSpeechBundlePreflightConfiguration(
            modelRootURL: URL(fileURLWithPath: expandedPath(modelPath)),
            modelFamily: modelFamily
        )
    }
}

@main
enum EdgeSpeechPreflightCommand {
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
            let result = try EdgeSpeechBundlePreflightRunner.run(
                configuration: try options.configuration()
            )
            let data = try encoder.encode(result)
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write(Data("\n".utf8))
            if options.requirePass && !result.passesPreflight {
                exit(3)
            }
        } catch let error as SpeechPreflightCLIError {
            fputs("error: \(error.description)\n\n\(Self.usage)\n", stderr)
            exit(2)
        } catch {
            fputs("error: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func parse(arguments: [String]) throws -> SpeechPreflightCLIOptions {
        var options = SpeechPreflightCLIOptions()
        var index = 0
        while index < arguments.count {
            let raw = arguments[index]
            if raw == "--require-pass" {
                options.requirePass = true
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
                    throw SpeechPreflightCLIError.missingValue(option)
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
            default:
                throw SpeechPreflightCLIError.unknownOption(option)
            }
            index += 1
        }
        return options
    }

    private static func parseFamily(_ value: String) throws -> EdgeSpeechModelFamily {
        switch value.lowercased().replacingOccurrences(of: "_", with: "-") {
        case "qwen3-asr", "asr":
            return .qwen3ASR
        case "qwen3-tts", "tts":
            return .qwen3TTS
        default:
            throw SpeechPreflightCLIError.invalidFamily(value)
        }
    }

    private static func parseBool(option: String, value: String) throws -> Bool {
        switch value.lowercased() {
        case "1", "true", "yes":
            return true
        case "0", "false", "no":
            return false
        default:
            throw SpeechPreflightCLIError.invalidBoolean(option: option, value: value)
        }
    }

    private static var usage: String {
        """
        Usage:
          edge-speech-preflight --model <model-dir> [--family <qwen3-asr|qwen3-tts>]

        Options:
          --model <model-dir>         Local Qwen3-ASR or Qwen3-TTS bundle directory.
          --family <family>           Optional explicit family guard.
                                      Accepted values: qwen3-asr, asr, qwen3-tts, tts.
          --require-pass              Exit with code 3 when preflight does not pass.

        Output:
          JSON speech preflight report. This validates bundle metadata and
          required resources without importing MLXAudio or loading model weights.
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
