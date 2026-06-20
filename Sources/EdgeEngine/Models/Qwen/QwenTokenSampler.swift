// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum QwenTokenSamplerError: Error, Equatable {
    case invalidLogitsShape(expected: [Int], actual: [Int])
    case invalidTemperature(Float)
    case invalidTopK(Int)
    case invalidTopP(Float)
    case invalidMinP(Float)
    case invalidRepetitionPenalty(Float)
    case invalidPresencePenalty(Float)
    case invalidFrequencyPenalty(Float)
    case invalidGeneratedTokenCount(Int)
    case invalidMinimumGeneratedTokens(Int)
    case invalidEOSPenaltyUntilToken(Int)
    case invalidEOSLogitPenalty(Float)
    case nonFiniteLogit(tokenId: Int, value: Float)
}

public struct QwenSamplingConfiguration: Equatable, Sendable {
    public var temperature: Float
    public var topK: Int?
    public var topP: Float?
    public var minP: Float
    public var repetitionPenalty: Float
    public var repetitionTokenIds: [Int]
    public var presencePenalty: Float
    public var presenceTokenIds: [Int]
    public var frequencyPenalty: Float
    public var frequencyTokenIds: [Int]
    public var endTokenIds: [Int]
    public var generatedTokenCount: Int
    public var minimumGeneratedTokens: Int
    public var eosPenaltyUntilToken: Int
    public var eosLogitPenalty: Float

    public init(
        temperature: Float = 1,
        topK: Int? = nil,
        topP: Float? = nil,
        minP: Float = 0,
        repetitionPenalty: Float = 1,
        repetitionTokenIds: [Int] = [],
        presencePenalty: Float = 0,
        presenceTokenIds: [Int] = [],
        frequencyPenalty: Float = 0,
        frequencyTokenIds: [Int] = [],
        endTokenIds: [Int] = [],
        generatedTokenCount: Int = 0,
        minimumGeneratedTokens: Int = 0,
        eosPenaltyUntilToken: Int = 0,
        eosLogitPenalty: Float = 20
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionTokenIds = repetitionTokenIds
        self.presencePenalty = presencePenalty
        self.presenceTokenIds = presenceTokenIds
        self.frequencyPenalty = frequencyPenalty
        self.frequencyTokenIds = frequencyTokenIds
        self.endTokenIds = endTokenIds
        self.generatedTokenCount = generatedTokenCount
        self.minimumGeneratedTokens = minimumGeneratedTokens
        self.eosPenaltyUntilToken = eosPenaltyUntilToken
        self.eosLogitPenalty = eosLogitPenalty
    }

    public static let greedy = QwenSamplingConfiguration(temperature: 0)
}

public struct QwenSampledToken: Equatable, Sendable {
    public var tokenId: Int
    public var logit: Float
    public var probability: Float

    public init(tokenId: Int, logit: Float, probability: Float) {
        self.tokenId = tokenId
        self.logit = logit
        self.probability = probability
    }
}

public struct EdgeSeededRandomNumberGenerator: RandomNumberGenerator, Equatable, Sendable {
    public private(set) var state: UInt64

    public init(seed: UInt64) {
        self.state = seed == 0 ? 0x9e37_79b9_7f4a_7c15 : seed
    }

    public mutating func next() -> UInt64 {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return state
    }
}

public enum QwenTokenSampler {
    public static func sampleToken(
        logits: EdgeTensor,
        configuration: QwenSamplingConfiguration = .greedy,
        seed: UInt64
    ) throws -> QwenSampledToken {
        var rng = EdgeSeededRandomNumberGenerator(seed: seed)
        return try sampleToken(
            logits: logits,
            configuration: configuration,
            rng: &rng
        )
    }

    public static func sampleToken<R: RandomNumberGenerator>(
        logits: EdgeTensor,
        configuration: QwenSamplingConfiguration = .greedy,
        rng: inout R
    ) throws -> QwenSampledToken {
        try validate(configuration)
        let candidates = applyPenalties(
            to: try lastRowCandidates(logits: logits),
            configuration: configuration
        )
        if configuration.temperature == 0 {
            let token = sortedByLogit(candidates)[0]
            return QwenSampledToken(
                tokenId: token.tokenId,
                logit: token.logit,
                probability: 1
            )
        }

        let distribution: [CandidateProbability]
        if (configuration.topP.map { $0 < 1 } ?? false) || configuration.minP > 0 {
            distribution = filteredDistribution(
                candidates,
                topP: configuration.topP,
                topK: configuration.topK,
                minP: configuration.minP,
                temperature: configuration.temperature
            )
        } else {
            var filtered = sortedByLogit(candidates)
            if let topK = configuration.topK {
                filtered = Array(filtered.prefix(min(topK, filtered.count)))
            }
            distribution = probabilities(
                for: filtered,
                temperature: configuration.temperature
            )
        }
        let draw = uniform01(rng: &rng)
        var cumulative = Float.zero
        for candidate in distribution {
            cumulative += candidate.probability
            if draw < cumulative {
                return QwenSampledToken(
                    tokenId: candidate.tokenId,
                    logit: candidate.logit,
                    probability: candidate.probability
                )
            }
        }

        let fallback = distribution[distribution.count - 1]
        return QwenSampledToken(
            tokenId: fallback.tokenId,
            logit: fallback.logit,
            probability: fallback.probability
        )
    }

    private struct Candidate {
        var tokenId: Int
        var logit: Float
    }

    private struct CandidateProbability {
        var tokenId: Int
        var logit: Float
        var probability: Float
    }

    private static func validate(_ configuration: QwenSamplingConfiguration) throws {
        guard configuration.temperature >= 0, configuration.temperature.isFinite else {
            throw QwenTokenSamplerError.invalidTemperature(configuration.temperature)
        }
        if let topK = configuration.topK, topK <= 0 {
            throw QwenTokenSamplerError.invalidTopK(topK)
        }
        if let topP = configuration.topP,
           !(topP > 0 && topP <= 1 && topP.isFinite) {
            throw QwenTokenSamplerError.invalidTopP(topP)
        }
        guard configuration.minP >= 0, configuration.minP.isFinite else {
            throw QwenTokenSamplerError.invalidMinP(configuration.minP)
        }
        guard configuration.repetitionPenalty > 0,
              configuration.repetitionPenalty.isFinite
        else {
            throw QwenTokenSamplerError.invalidRepetitionPenalty(configuration.repetitionPenalty)
        }
        guard configuration.presencePenalty.isFinite else {
            throw QwenTokenSamplerError.invalidPresencePenalty(configuration.presencePenalty)
        }
        guard configuration.frequencyPenalty.isFinite else {
            throw QwenTokenSamplerError.invalidFrequencyPenalty(configuration.frequencyPenalty)
        }
        guard configuration.generatedTokenCount >= 0 else {
            throw QwenTokenSamplerError.invalidGeneratedTokenCount(configuration.generatedTokenCount)
        }
        guard configuration.minimumGeneratedTokens >= 0 else {
            throw QwenTokenSamplerError.invalidMinimumGeneratedTokens(configuration.minimumGeneratedTokens)
        }
        guard configuration.eosPenaltyUntilToken >= 0 else {
            throw QwenTokenSamplerError.invalidEOSPenaltyUntilToken(configuration.eosPenaltyUntilToken)
        }
        guard configuration.eosLogitPenalty >= 0,
              configuration.eosLogitPenalty.isFinite
        else {
            throw QwenTokenSamplerError.invalidEOSLogitPenalty(configuration.eosLogitPenalty)
        }
    }

    private static func lastRowCandidates(logits: EdgeTensor) throws -> [Candidate] {
        guard logits.shape.rank == 2,
              logits.shape.dimensions[0] > 0,
              logits.shape.dimensions[1] > 0
        else {
            throw QwenTokenSamplerError.invalidLogitsShape(
                expected: [-1, -1],
                actual: logits.shape.dimensions
            )
        }

        let values = try logits.readFloat32()
        let vocabularySize = logits.shape.dimensions[1]
        let lastRowOffset = (logits.shape.dimensions[0] - 1) * vocabularySize
        return try (0..<vocabularySize).map { tokenId in
            let logit = values[lastRowOffset + tokenId]
            guard logit.isFinite else {
                throw QwenTokenSamplerError.nonFiniteLogit(tokenId: tokenId, value: logit)
            }
            return Candidate(tokenId: tokenId, logit: logit)
        }
    }

    private static func sortedByLogit(_ candidates: [Candidate]) -> [Candidate] {
        candidates.sorted { lhs, rhs in
            if lhs.logit == rhs.logit {
                return lhs.tokenId < rhs.tokenId
            }
            return lhs.logit > rhs.logit
        }
    }

    private static func sortedByLogitAscending(_ candidates: [Candidate]) -> [Candidate] {
        candidates.sorted { lhs, rhs in
            if lhs.logit == rhs.logit {
                return lhs.tokenId < rhs.tokenId
            }
            return lhs.logit < rhs.logit
        }
    }

    private static func applyPenalties(
        to candidates: [Candidate],
        configuration: QwenSamplingConfiguration
    ) -> [Candidate] {
        let repeatedTokenIds: Set<Int> = configuration.repetitionPenalty != 1
            ? Set(configuration.repetitionTokenIds)
            : Set()
        let presenceTokenIds: Set<Int> = configuration.presencePenalty != 0
            ? Set(configuration.presenceTokenIds)
            : Set()
        let frequencyCounts: [Int: Int] = configuration.frequencyPenalty != 0
            ? tokenCounts(configuration.frequencyTokenIds)
            : [:]
        let endTokenIds = Set(configuration.endTokenIds)
        return candidates.map { candidate in
            var logit = candidate.logit
            if repeatedTokenIds.contains(candidate.tokenId) {
                if logit > 0 {
                    logit /= configuration.repetitionPenalty
                } else {
                    logit *= configuration.repetitionPenalty
                }
            }
            if presenceTokenIds.contains(candidate.tokenId) {
                logit -= configuration.presencePenalty
            }
            if let count = frequencyCounts[candidate.tokenId] {
                logit -= configuration.frequencyPenalty * Float(count)
            }
            if endTokenIds.contains(candidate.tokenId) {
                if configuration.generatedTokenCount < configuration.minimumGeneratedTokens {
                    logit = -Float.greatestFiniteMagnitude
                } else if configuration.generatedTokenCount < configuration.eosPenaltyUntilToken {
                    logit -= configuration.eosLogitPenalty
                }
            }
            return Candidate(tokenId: candidate.tokenId, logit: logit)
        }
    }

    private static func tokenCounts(_ tokenIds: [Int]) -> [Int: Int] {
        var counts: [Int: Int] = [:]
        for tokenId in tokenIds {
            counts[tokenId, default: 0] += 1
        }
        return counts
    }

    private static func filteredDistribution(
        _ candidates: [Candidate],
        topP: Float?,
        topK: Int?,
        minP: Float,
        temperature: Float
    ) -> [CandidateProbability] {
        let sortedAscendingDistribution = probabilities(
            for: sortedByLogitAscending(candidates),
            temperature: 1
        )

        var allowedTokenIds = Set(sortedAscendingDistribution.map(\.tokenId))
        if let topP, topP < 1 {
            var cumulative = Float.zero
            var nucleusTokenIds = Set<Int>()
            let threshold = 1 - topP
            for candidate in sortedAscendingDistribution {
                cumulative += candidate.probability
                if cumulative > threshold {
                    nucleusTokenIds.insert(candidate.tokenId)
                }
            }
            allowedTokenIds.formIntersection(nucleusTokenIds)
        }

        if minP > 0 {
            let maxProbability = sortedAscendingDistribution.last?.probability ?? 0
            let threshold = maxProbability * minP
            allowedTokenIds.formIntersection(
                Set(sortedAscendingDistribution
                    .filter { $0.probability >= threshold }
                    .map(\.tokenId))
            )
        }

        let sortedDescending = sortedByLogit(candidates)
        if let topK, topK < sortedDescending.count {
            allowedTokenIds.formIntersection(Set(sortedDescending.prefix(topK).map(\.tokenId)))
        }

        if let maxCandidate = sortedDescending.first {
            allowedTokenIds.insert(maxCandidate.tokenId)
        }
        let filtered = sortedAscendingDistribution
            .filter { allowedTokenIds.contains($0.tokenId) }
            .map { Candidate(tokenId: $0.tokenId, logit: $0.logit) }

        return probabilities(for: filtered, temperature: temperature)
    }

    private static func probabilities(
        for candidates: [Candidate],
        temperature: Float
    ) -> [CandidateProbability] {
        let maxLogit = candidates.map(\.logit).max() ?? 0
        let scaled = candidates.map { candidate -> (candidate: Candidate, weight: Float) in
            let value = Foundation.exp(Double((candidate.logit - maxLogit) / temperature))
            return (candidate, Float(value))
        }
        let denominator = scaled.reduce(Float.zero) { $0 + $1.weight }
        return scaled.map { entry in
            CandidateProbability(
                tokenId: entry.candidate.tokenId,
                logit: entry.candidate.logit,
                probability: entry.weight / denominator
            )
        }
    }

    private static func uniform01<R: RandomNumberGenerator>(rng: inout R) -> Float {
        Float(Double(rng.next()) / (Double(UInt64.max) + 1.0))
    }
}
