// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public struct EdgeTensorShape: Equatable, Sendable {
    public var dimensions: [Int]

    public init(_ dimensions: [Int]) {
        precondition(!dimensions.isEmpty, "EdgeTensorShape requires at least one dimension")
        precondition(dimensions.allSatisfy { $0 > 0 }, "EdgeTensorShape dimensions must be positive")
        self.dimensions = dimensions
    }

    public var rank: Int {
        dimensions.count
    }

    public var elementCount: Int {
        dimensions.reduce(1, *)
    }

    var mpsShape: [NSNumber] {
        dimensions.map { NSNumber(value: $0) }
    }
}
