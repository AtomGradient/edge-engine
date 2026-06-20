// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import XCTest

func edgeTestFileURL(fromEnvironment name: String, purpose: String) throws -> URL {
    guard let value = ProcessInfo.processInfo.environment[name], !value.isEmpty else {
        throw XCTSkip("Set \(name) to run \(purpose).")
    }
    let url = URL(fileURLWithPath: (value as NSString).expandingTildeInPath)
    guard FileManager.default.fileExists(atPath: url.path) else {
        throw XCTSkip("Required fixture for \(purpose) is missing: \(url.path)")
    }
    return url
}

func edgeTestDirectoryURL(
    fromEnvironment name: String,
    purpose: String,
    requiredFileName: String
) throws -> URL {
    let url = try edgeTestFileURL(fromEnvironment: name, purpose: purpose)
    guard FileManager.default.fileExists(atPath: url.appendingPathComponent(requiredFileName).path) else {
        throw XCTSkip("Required fixture for \(purpose) is missing: \(url.appendingPathComponent(requiredFileName).path)")
    }
    return url
}
