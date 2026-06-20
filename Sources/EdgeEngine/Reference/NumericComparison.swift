// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public enum NumericComparisonError: Error, Equatable {
    case lengthMismatch
    case emptyInput
}

public enum NumericComparison {
    public static func cosineSimilarity(_ lhs: [Float], _ rhs: [Float]) throws -> Float {
        guard lhs.count == rhs.count else { throw NumericComparisonError.lengthMismatch }
        guard !lhs.isEmpty else { throw NumericComparisonError.emptyInput }

        var dot = Float.zero
        var lhsNorm = Float.zero
        var rhsNorm = Float.zero
        for (left, right) in zip(lhs, rhs) {
            dot += left * right
            lhsNorm += left * left
            rhsNorm += right * right
        }

        guard lhsNorm > 0, rhsNorm > 0 else { return 0 }
        return dot / (lhsNorm.squareRoot() * rhsNorm.squareRoot())
    }

    public static func maxAbsoluteError(_ lhs: [Float], _ rhs: [Float]) throws -> Float {
        guard lhs.count == rhs.count else { throw NumericComparisonError.lengthMismatch }
        guard !lhs.isEmpty else { throw NumericComparisonError.emptyInput }

        return zip(lhs, rhs).map { abs($0 - $1) }.max()!
    }
}
