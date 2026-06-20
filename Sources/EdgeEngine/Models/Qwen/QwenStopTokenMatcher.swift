// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public enum QwenStopTokenMatcherError: Error, Equatable {
    case emptyStopSequence(index: Int)
}

public struct QwenStopTokenMatch: Equatable, Sendable {
    public var tokenIds: [Int]

    public init(tokenIds: [Int]) {
        self.tokenIds = tokenIds
    }

    public var tokenCount: Int {
        tokenIds.count
    }
}

public struct QwenStopTokenMatcher: Equatable, Sendable {
    public let stopSequences: [[Int]]
    public private(set) var bufferedTokenIds: [Int]

    private let maxStopSequenceLength: Int

    public init(stopSequences: [[Int]]) throws {
        for (index, sequence) in stopSequences.enumerated() where sequence.isEmpty {
            throw QwenStopTokenMatcherError.emptyStopSequence(index: index)
        }

        self.stopSequences = stopSequences
        self.bufferedTokenIds = []
        self.maxStopSequenceLength = stopSequences.map(\.count).max() ?? 0
    }

    public mutating func append(tokenId: Int) -> QwenStopTokenMatch? {
        guard maxStopSequenceLength > 0 else {
            return nil
        }

        bufferedTokenIds.append(tokenId)
        if bufferedTokenIds.count > maxStopSequenceLength {
            bufferedTokenIds.removeFirst(bufferedTokenIds.count - maxStopSequenceLength)
        }

        return currentMatch()
    }

    public mutating func append(contentsOf tokenIds: [Int]) -> QwenStopTokenMatch? {
        for tokenId in tokenIds {
            if let match = append(tokenId: tokenId) {
                return match
            }
        }
        return nil
    }

    public func currentMatch() -> QwenStopTokenMatch? {
        var bestMatch: [Int]?
        for sequence in stopSequences {
            guard sequence.count <= bufferedTokenIds.count else {
                continue
            }

            let suffix = bufferedTokenIds.suffix(sequence.count)
            if Array(suffix) != sequence {
                continue
            }

            if let currentBestMatch = bestMatch {
                if sequence.count > currentBestMatch.count {
                    bestMatch = sequence
                }
            } else {
                bestMatch = sequence
            }
        }

        guard let bestMatch else {
            return nil
        }
        return QwenStopTokenMatch(tokenIds: bestMatch)
    }

    public mutating func reset() {
        bufferedTokenIds.removeAll(keepingCapacity: true)
    }
}
