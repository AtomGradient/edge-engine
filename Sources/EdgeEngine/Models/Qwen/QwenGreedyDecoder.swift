// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public enum QwenGreedyDecoderError: Error, Equatable {
    case invalidLogitsShape(expected: [Int], actual: [Int])
    case invalidMaxTokenCount(Int)
}

public struct QwenGreedyToken: Equatable, Sendable {
    public var tokenId: Int
    public var logit: Float

    public init(tokenId: Int, logit: Float) {
        self.tokenId = tokenId
        self.logit = logit
    }
}

public enum QwenGreedyDecoder {
    public static func nextToken(logits: EdgeTensor) throws -> QwenGreedyToken {
        guard logits.shape.rank == 2,
              logits.shape.dimensions[0] > 0,
              logits.shape.dimensions[1] > 0
        else {
            throw QwenGreedyDecoderError.invalidLogitsShape(
                expected: [-1, -1],
                actual: logits.shape.dimensions
            )
        }

        let values = try logits.readFloat32()
        let vocabularySize = logits.shape.dimensions[1]
        let lastRowOffset = (logits.shape.dimensions[0] - 1) * vocabularySize
        var bestTokenId = 0
        var bestLogit = values[lastRowOffset]
        for tokenId in 1..<vocabularySize {
            let value = values[lastRowOffset + tokenId]
            if value > bestLogit {
                bestTokenId = tokenId
                bestLogit = value
            }
        }
        return QwenGreedyToken(tokenId: bestTokenId, logit: bestLogit)
    }

    public static func generateNextTokens(
        promptTokenIds: [Int],
        model: QwenHybridModelReference,
        caches: QwenHybridDecoderCaches,
        executor: MetalKernelExecutor,
        maxTokenCount: Int,
        endTokenIds: Set<Int> = []
    ) throws -> [Int] {
        guard maxTokenCount >= 0 else {
            throw QwenGreedyDecoderError.invalidMaxTokenCount(maxTokenCount)
        }
        guard maxTokenCount > 0 else {
            return []
        }

        var generated: [Int] = []
        var logits = try model.logits(
            tokenIds: promptTokenIds,
            caches: caches,
            executor: executor
        )
        while generated.count < maxTokenCount {
            let token = try nextToken(logits: logits)
            generated.append(token.tokenId)
            if endTokenIds.contains(token.tokenId) {
                break
            }
            if generated.count < maxTokenCount {
                logits = try model.logits(
                    tokenIds: [token.tokenId],
                    caches: caches,
                    executor: executor
                )
            }
        }
        return generated
    }
}
