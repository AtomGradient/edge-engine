// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Testing
@testable import EdgeEngine

@Test func qwenTokenSamplerGreedyUsesLastLogitsRow() throws {
    let runtime = try EdgeMetalRuntime()
    let logits = try EdgeTensor(
        float32: [
            9, 8, 7,
            1, 4, 3,
        ],
        shape: EdgeTensorShape([2, 3]),
        runtime: runtime
    )

    let token = try QwenTokenSampler.sampleToken(
        logits: logits,
        configuration: .greedy,
        seed: 42
    )

    #expect(token == QwenSampledToken(tokenId: 1, logit: 4, probability: 1))
}

@Test func qwenTokenSamplerSamplesFromTemperatureDistribution() throws {
    let runtime = try EdgeMetalRuntime()
    let logits = try EdgeTensor(
        float32: [0, 0],
        shape: EdgeTensorShape([1, 2]),
        runtime: runtime
    )
    var rng = FixedSamplerRNG(values: [(UInt64.max / 4) * 3])

    let token = try QwenTokenSampler.sampleToken(
        logits: logits,
        configuration: QwenSamplingConfiguration(temperature: 1),
        rng: &rng
    )

    #expect(token.tokenId == 1)
    #expect(token.logit == 0)
    #expect(abs(token.probability - 0.5) < 1e-6)
}

@Test func qwenTokenSamplerAppliesTopKBeforeSampling() throws {
    let runtime = try EdgeMetalRuntime()
    let logits = try EdgeTensor(
        float32: [1, 10, 9, 8],
        shape: EdgeTensorShape([1, 4]),
        runtime: runtime
    )
    var rng = FixedSamplerRNG(values: [UInt64.max - 1])

    let token = try QwenTokenSampler.sampleToken(
        logits: logits,
        configuration: QwenSamplingConfiguration(
            temperature: 1,
            topK: 2
        ),
        rng: &rng
    )

    #expect(token.tokenId == 2)
    #expect(token.logit == 9)
}

@Test func qwenTokenSamplerAppliesTopPBeforeSampling() throws {
    let runtime = try EdgeMetalRuntime()
    let logits = try EdgeTensor(
        float32: [10, 9, 1],
        shape: EdgeTensorShape([1, 3]),
        runtime: runtime
    )
    var rng = FixedSamplerRNG(values: [UInt64.max - 1])

    let token = try QwenTokenSampler.sampleToken(
        logits: logits,
        configuration: QwenSamplingConfiguration(
            temperature: 1,
            topP: 0.7
        ),
        rng: &rng
    )

    #expect(token.tokenId == 0)
    #expect(token.logit == 10)
    #expect(abs(token.probability - 1) < 1e-6)
}

@Test func qwenTokenSamplerAppliesTopPBeforeTopKWithoutRenormalizingTopK() throws {
    let runtime = try EdgeMetalRuntime()
    let logits = try EdgeTensor(
        float32: [Float(10), Float(7.6)] + Array(repeating: Float(7.5), count: 62),
        shape: EdgeTensorShape([1, 64]),
        runtime: runtime
    )
    var rng = FixedSamplerRNG(values: [0])

    let token = try QwenTokenSampler.sampleToken(
        logits: logits,
        configuration: QwenSamplingConfiguration(
            temperature: 1,
            topK: 2,
            topP: 0.9
        ),
        rng: &rng
    )

    #expect(token.tokenId == 1)
    #expect(token.logit == Float(7.6))
}

@Test func qwenTokenSamplerAppliesMinPBeforeSampling() throws {
    let runtime = try EdgeMetalRuntime()
    let logits = try EdgeTensor(
        float32: [0, 0, 4],
        shape: EdgeTensorShape([1, 3]),
        runtime: runtime
    )
    var rng = FixedSamplerRNG(values: [UInt64.max - 1])

    let token = try QwenTokenSampler.sampleToken(
        logits: logits,
        configuration: QwenSamplingConfiguration(
            temperature: 1,
            minP: 0.95
        ),
        rng: &rng
    )

    #expect(token.tokenId == 2)
    #expect(token.logit == 4)
}

@Test func qwenTokenSamplerAppliesRepetitionPenaltyBeforeSelection() throws {
    let runtime = try EdgeMetalRuntime()
    let logits = try EdgeTensor(
        float32: [2.0, 1.9],
        shape: EdgeTensorShape([1, 2]),
        runtime: runtime
    )

    let token = try QwenTokenSampler.sampleToken(
        logits: logits,
        configuration: QwenSamplingConfiguration(
            temperature: 0,
            repetitionPenalty: 2.0,
            repetitionTokenIds: [0, 0]
        ),
        seed: 1
    )

    #expect(token.tokenId == 1)
    #expect(token.logit == 1.9)
}

@Test func qwenTokenSamplerAppliesPresencePenaltyBeforeSelection() throws {
    let runtime = try EdgeMetalRuntime()
    let logits = try EdgeTensor(
        float32: [1.0, 2.0, 3.0, 4.0],
        shape: EdgeTensorShape([1, 4]),
        runtime: runtime
    )

    let token = try QwenTokenSampler.sampleToken(
        logits: logits,
        configuration: QwenSamplingConfiguration(
            temperature: 0,
            presencePenalty: 2.0,
            presenceTokenIds: [3, 3]
        ),
        seed: 1
    )

    #expect(token.tokenId == 2)
    #expect(token.logit == 3.0)
}

@Test func qwenTokenSamplerAppliesFrequencyPenaltyByCountBeforeSelection() throws {
    let runtime = try EdgeMetalRuntime()
    let logits = try EdgeTensor(
        float32: [1.0, 3.0, 2.0],
        shape: EdgeTensorShape([1, 3]),
        runtime: runtime
    )

    let token = try QwenTokenSampler.sampleToken(
        logits: logits,
        configuration: QwenSamplingConfiguration(
            temperature: 0,
            frequencyPenalty: 1.0,
            frequencyTokenIds: [1, 1]
        ),
        seed: 1
    )

    #expect(token.tokenId == 2)
    #expect(token.logit == 2.0)
}

@Test func qwenTokenSamplerSuppressesEOSBeforeMinimumGeneratedTokens() throws {
    let runtime = try EdgeMetalRuntime()
    let logits = try EdgeTensor(
        float32: [1.0, 5.0, 4.0],
        shape: EdgeTensorShape([1, 3]),
        runtime: runtime
    )

    let earlyToken = try QwenTokenSampler.sampleToken(
        logits: logits,
        configuration: QwenSamplingConfiguration(
            temperature: 0,
            endTokenIds: [1],
            generatedTokenCount: 4,
            minimumGeneratedTokens: 8
        ),
        seed: 1
    )
    let matureToken = try QwenTokenSampler.sampleToken(
        logits: logits,
        configuration: QwenSamplingConfiguration(
            temperature: 0,
            endTokenIds: [1],
            generatedTokenCount: 8,
            minimumGeneratedTokens: 8
        ),
        seed: 1
    )

    #expect(earlyToken.tokenId == 2)
    #expect(matureToken.tokenId == 1)
}

@Test func qwenTokenSamplerAppliesEOSPenaltyUntilConfiguredTokenCount() throws {
    let runtime = try EdgeMetalRuntime()
    let logits = try EdgeTensor(
        float32: [1.0, 5.0, 4.0],
        shape: EdgeTensorShape([1, 3]),
        runtime: runtime
    )

    let penalizedToken = try QwenTokenSampler.sampleToken(
        logits: logits,
        configuration: QwenSamplingConfiguration(
            temperature: 0,
            endTokenIds: [1],
            generatedTokenCount: 4,
            eosPenaltyUntilToken: 8,
            eosLogitPenalty: 2
        ),
        seed: 1
    )
    let matureToken = try QwenTokenSampler.sampleToken(
        logits: logits,
        configuration: QwenSamplingConfiguration(
            temperature: 0,
            endTokenIds: [1],
            generatedTokenCount: 8,
            eosPenaltyUntilToken: 8,
            eosLogitPenalty: 2
        ),
        seed: 1
    )

    #expect(penalizedToken.tokenId == 2)
    #expect(matureToken.tokenId == 1)
}

@Test func qwenTokenSamplerRejectsInvalidInputs() throws {
    let runtime = try EdgeMetalRuntime()
    let logits = try EdgeTensor(
        float32: [1, 2, 3],
        shape: EdgeTensorShape([3]),
        runtime: runtime
    )
    let validLogits = try EdgeTensor(
        float32: [1, 2, 3],
        shape: EdgeTensorShape([1, 3]),
        runtime: runtime
    )
    var invalidShape = false
    var invalidTemperature = false
    var invalidTopK = false
    var invalidTopP = false
    var invalidMinP = false
    var invalidRepetitionPenalty = false
    var invalidPresencePenalty = false
    var invalidFrequencyPenalty = false

    do {
        _ = try QwenTokenSampler.sampleToken(logits: logits, seed: 1)
    } catch QwenTokenSamplerError.invalidLogitsShape(expected: [-1, -1], actual: [3]) {
        invalidShape = true
    }

    do {
        _ = try QwenTokenSampler.sampleToken(
            logits: validLogits,
            configuration: QwenSamplingConfiguration(temperature: -0.1),
            seed: 1
        )
    } catch QwenTokenSamplerError.invalidTemperature(-0.1) {
        invalidTemperature = true
    }

    do {
        _ = try QwenTokenSampler.sampleToken(
            logits: validLogits,
            configuration: QwenSamplingConfiguration(topK: 0),
            seed: 1
        )
    } catch QwenTokenSamplerError.invalidTopK(0) {
        invalidTopK = true
    }

    do {
        _ = try QwenTokenSampler.sampleToken(
            logits: validLogits,
            configuration: QwenSamplingConfiguration(topP: 1.1),
            seed: 1
        )
    } catch QwenTokenSamplerError.invalidTopP(1.1) {
        invalidTopP = true
    }

    do {
        _ = try QwenTokenSampler.sampleToken(
            logits: validLogits,
            configuration: QwenSamplingConfiguration(minP: -0.1),
            seed: 1
        )
    } catch QwenTokenSamplerError.invalidMinP(-0.1) {
        invalidMinP = true
    }

    do {
        _ = try QwenTokenSampler.sampleToken(
            logits: validLogits,
            configuration: QwenSamplingConfiguration(repetitionPenalty: 0),
            seed: 1
        )
    } catch QwenTokenSamplerError.invalidRepetitionPenalty(0) {
        invalidRepetitionPenalty = true
    }

    do {
        _ = try QwenTokenSampler.sampleToken(
            logits: validLogits,
            configuration: QwenSamplingConfiguration(presencePenalty: .nan),
            seed: 1
        )
    } catch QwenTokenSamplerError.invalidPresencePenalty(_) {
        invalidPresencePenalty = true
    }

    do {
        _ = try QwenTokenSampler.sampleToken(
            logits: validLogits,
            configuration: QwenSamplingConfiguration(frequencyPenalty: .infinity),
            seed: 1
        )
    } catch QwenTokenSamplerError.invalidFrequencyPenalty(_) {
        invalidFrequencyPenalty = true
    }

    #expect(invalidShape)
    #expect(invalidTemperature)
    #expect(invalidTopK)
    #expect(invalidTopP)
    #expect(invalidMinP)
    #expect(invalidRepetitionPenalty)
    #expect(invalidPresencePenalty)
    #expect(invalidFrequencyPenalty)
}

private struct FixedSamplerRNG: RandomNumberGenerator {
    var values: [UInt64]

    mutating func next() -> UInt64 {
        values.removeFirst()
    }
}
