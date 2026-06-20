// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public enum QwenGreedyDecodeSessionError: Error, Equatable {
    case missingLogits
}

public struct QwenGreedyDecodeStep: Equatable, Sendable {
    public var token: QwenGreedyToken
    public var cacheTokenPosition: Int
    public var reachedEndToken: Bool

    public init(
        token: QwenGreedyToken,
        cacheTokenPosition: Int,
        reachedEndToken: Bool
    ) {
        self.token = token
        self.cacheTokenPosition = cacheTokenPosition
        self.reachedEndToken = reachedEndToken
    }
}

public struct QwenSampledDecodeStep: Equatable, Sendable {
    public var token: QwenSampledToken
    public var cacheTokenPosition: Int
    public var reachedEndToken: Bool

    public init(
        token: QwenSampledToken,
        cacheTokenPosition: Int,
        reachedEndToken: Bool
    ) {
        self.token = token
        self.cacheTokenPosition = cacheTokenPosition
        self.reachedEndToken = reachedEndToken
    }
}

public final class QwenGreedyDecodeSession {
    public let model: QwenHybridModelReference
    public let caches: QwenHybridDecoderCaches
    public let executor: MetalKernelExecutor
    public var diagnosticSink: ((String) -> Void)?

    private var currentLogits: EdgeTensor?
    private var currentGreedyToken: QwenGreedyToken?
    private var argmaxScratch: MetalArgmaxLastRowScratch?

    public init(
        model: QwenHybridModelReference,
        caches: QwenHybridDecoderCaches,
        executor: MetalKernelExecutor,
        diagnosticSink: ((String) -> Void)? = nil
    ) {
        self.model = model
        self.caches = caches
        self.executor = executor
        self.diagnosticSink = diagnosticSink
    }

    public func tokenPosition() throws -> Int {
        try caches.tokenPosition()
    }

    @discardableResult
    public func prefill(promptTokenIds: [Int]) throws -> EdgeTensor {
        guard !promptTokenIds.isEmpty else {
            throw QwenHybridModelReferenceError.emptyTokenIds
        }
        reset()
        return try advance(tokenIds: promptTokenIds)
    }

    @discardableResult
    public func prefillGreedy(promptTokenIds: [Int]) throws -> QwenGreedyToken {
        guard !promptTokenIds.isEmpty else {
            throw QwenHybridModelReferenceError.emptyTokenIds
        }
        reset()
        return try advanceGreedy(tokenIds: promptTokenIds)
    }

    @discardableResult
    public func advance(tokenIds: [Int]) throws -> EdgeTensor {
        let logits: EdgeTensor
        if shouldUseSingleCommandBuffer(forTokenCount: tokenIds.count) {
            let phase = tokenIds.count == 1 ? "decode" : "prefill"
            diagnosticSink?("\(phase)_single_command_buffer_begin")
            logits = try executor.withUnboundedCommandBufferBatch {
                try model.lastTokenLogits(
                    tokenIds: tokenIds,
                    caches: caches,
                    executor: executor,
                    diagnosticSink: diagnosticSink
                )
            }
            diagnosticSink?("\(phase)_single_command_buffer_done shape=\(logits.shape.dimensions)")
        } else {
            logits = try model.lastTokenLogits(
                tokenIds: tokenIds,
                caches: caches,
                executor: executor,
                diagnosticSink: diagnosticSink
            )
        }
        currentLogits = logits
        currentGreedyToken = nil
        return logits
    }

    @discardableResult
    public func advanceGreedy(tokenIds: [Int]) throws -> QwenGreedyToken {
        let token: QwenGreedyToken
        if shouldUseSingleCommandBuffer(forTokenCount: tokenIds.count) {
            let phase = tokenIds.count == 1 ? "decode" : "prefill"
            diagnosticSink?("\(phase)_single_command_buffer_begin")
            token = try executor.withUnboundedCommandBufferBatch {
                try model.lastTokenGreedyToken(
                    tokenIds: tokenIds,
                    caches: caches,
                    executor: executor,
                    diagnosticSink: diagnosticSink
                )
            }
            diagnosticSink?("\(phase)_single_command_buffer_done token=\(token.tokenId)")
        } else {
            token = try model.lastTokenGreedyToken(
                tokenIds: tokenIds,
                caches: caches,
                executor: executor,
                diagnosticSink: diagnosticSink
            )
        }
        currentLogits = nil
        currentGreedyToken = token
        return token
    }

    private func shouldUseSingleCommandBuffer(forTokenCount tokenCount: Int) -> Bool {
        if tokenCount == 1 {
            return executor.runtimeConfiguration.useSingleCommandBufferDecode
        }
        return executor.runtimeConfiguration.useSingleCommandBufferPrefill
    }

    @discardableResult
    public func advance(with tokenId: Int) throws -> EdgeTensor {
        try advance(tokenIds: [tokenId])
    }

    @discardableResult
    public func advanceGreedy(with tokenId: Int) throws -> QwenGreedyToken {
        try advanceGreedy(tokenIds: [tokenId])
    }

    public func selectNextToken() throws -> QwenGreedyToken {
        if let currentGreedyToken {
            return currentGreedyToken
        }
        guard let currentLogits else {
            throw QwenGreedyDecodeSessionError.missingLogits
        }
        let scratch: MetalArgmaxLastRowScratch
        if let existingScratch = argmaxScratch {
            scratch = existingScratch
        } else {
            let newScratch = try executor.makeArgmaxLastRowScratch()
            argmaxScratch = newScratch
            scratch = newScratch
        }
        return try executor.argmaxLastRow(currentLogits, scratch: scratch)
    }

    public func selectSampledToken<R: RandomNumberGenerator>(
        configuration: QwenSamplingConfiguration,
        rng: inout R
    ) throws -> QwenSampledToken {
        guard let currentLogits else {
            throw QwenGreedyDecodeSessionError.missingLogits
        }
        return try QwenTokenSampler.sampleToken(
            logits: currentLogits,
            configuration: configuration,
            rng: &rng
        )
    }

    public func generateNextToken(endTokenIds: Set<Int> = []) throws -> QwenGreedyDecodeStep {
        let token = try selectNextToken()
        let reachedEndToken = endTokenIds.contains(token.tokenId)
        if reachedEndToken {
            invalidateCurrentLogits()
        } else {
            try advance(with: token.tokenId)
        }
        return QwenGreedyDecodeStep(
            token: token,
            cacheTokenPosition: try tokenPosition(),
            reachedEndToken: reachedEndToken
        )
    }

    public func sampleNextToken<R: RandomNumberGenerator>(
        configuration: QwenSamplingConfiguration,
        endTokenIds: Set<Int> = [],
        rng: inout R
    ) throws -> QwenSampledDecodeStep {
        let token = try selectSampledToken(
            configuration: configuration,
            rng: &rng
        )
        let reachedEndToken = endTokenIds.contains(token.tokenId)
        if reachedEndToken {
            invalidateCurrentLogits()
        } else {
            try advance(with: token.tokenId)
        }
        return QwenSampledDecodeStep(
            token: token,
            cacheTokenPosition: try tokenPosition(),
            reachedEndToken: reachedEndToken
        )
    }

    public func generateNextTokens(
        maxTokenCount: Int,
        endTokenIds: Set<Int> = [],
        stopSequences: [[Int]] = []
    ) throws -> [Int] {
        guard maxTokenCount >= 0 else {
            throw QwenGreedyDecoderError.invalidMaxTokenCount(maxTokenCount)
        }

        var stopMatcher = try QwenStopTokenMatcher(stopSequences: stopSequences)
        var generated: [Int] = []
        while generated.count < maxTokenCount {
            let token = try selectNextToken()
            let reachedStop = endTokenIds.contains(token.tokenId)
                || stopMatcher.append(tokenId: token.tokenId) != nil
            generated.append(token.tokenId)
            if reachedStop {
                invalidateCurrentLogits()
                break
            }
            try advance(with: token.tokenId)
        }
        return generated
    }

    public func generateNextTokens(
        promptTokenIds: [Int],
        maxTokenCount: Int,
        endTokenIds: Set<Int> = [],
        stopSequences: [[Int]] = []
    ) throws -> [Int] {
        guard maxTokenCount >= 0 else {
            throw QwenGreedyDecoderError.invalidMaxTokenCount(maxTokenCount)
        }
        try validate(stopSequences: stopSequences)
        try prefill(promptTokenIds: promptTokenIds)
        return try generateNextTokens(
            maxTokenCount: maxTokenCount,
            endTokenIds: endTokenIds,
            stopSequences: stopSequences
        )
    }

    public func sampleNextTokens<R: RandomNumberGenerator>(
        maxTokenCount: Int,
        configuration: QwenSamplingConfiguration,
        endTokenIds: Set<Int> = [],
        stopSequences: [[Int]] = [],
        rng: inout R
    ) throws -> [Int] {
        guard maxTokenCount >= 0 else {
            throw QwenGreedyDecoderError.invalidMaxTokenCount(maxTokenCount)
        }

        var stopMatcher = try QwenStopTokenMatcher(stopSequences: stopSequences)
        var generated: [Int] = []
        while generated.count < maxTokenCount {
            let token = try selectSampledToken(
                configuration: configuration,
                rng: &rng
            )
            let reachedStop = endTokenIds.contains(token.tokenId)
                || stopMatcher.append(tokenId: token.tokenId) != nil
            generated.append(token.tokenId)
            if reachedStop {
                invalidateCurrentLogits()
                break
            }
            try advance(with: token.tokenId)
        }
        return generated
    }

    public func sampleNextTokens<R: RandomNumberGenerator>(
        promptTokenIds: [Int],
        maxTokenCount: Int,
        configuration: QwenSamplingConfiguration,
        endTokenIds: Set<Int> = [],
        stopSequences: [[Int]] = [],
        rng: inout R
    ) throws -> [Int] {
        guard maxTokenCount >= 0 else {
            throw QwenGreedyDecoderError.invalidMaxTokenCount(maxTokenCount)
        }
        try validate(stopSequences: stopSequences)
        try prefill(promptTokenIds: promptTokenIds)
        return try sampleNextTokens(
            maxTokenCount: maxTokenCount,
            configuration: configuration,
            endTokenIds: endTokenIds,
            stopSequences: stopSequences,
            rng: &rng
        )
    }

    public func reset() {
        caches.reset()
        currentLogits = nil
        currentGreedyToken = nil
    }

    /// Clears stale logits while preserving KV/GDN cache state for inspection.
    public func invalidateCurrentLogits() {
        currentLogits = nil
        currentGreedyToken = nil
    }

    private func validate(stopSequences: [[Int]]) throws {
        _ = try QwenStopTokenMatcher(stopSequences: stopSequences)
    }
}
