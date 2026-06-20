// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum QwenGDNReferenceError: Error, Equatable {
    case invalidShape
    case invalidHeadMapping(keyHeadCount: Int, valueHeadCount: Int)
}

public struct QwenGDNReferenceStep: Equatable, Sendable {
    public var updatedState: [Float]
    public var output: [Float]

    public init(updatedState: [Float], output: [Float]) {
        self.updatedState = updatedState
        self.output = output
    }
}

public struct QwenGatedDeltaReferenceUpdate: Equatable, Sendable {
    public var output: [Float]
    public var updatedState: [Float]

    public init(output: [Float], updatedState: [Float]) {
        self.output = output
        self.updatedState = updatedState
    }
}

public enum QwenGDNReference {
    public static func computeDecay(
        aLog: [Float],
        a: [Float],
        dtBias: [Float],
        batchSize: Int,
        tokenCount: Int,
        valueHeadCount: Int
    ) throws -> [Float] {
        guard batchSize > 0,
              tokenCount > 0,
              valueHeadCount > 0,
              aLog.count == valueHeadCount,
              dtBias.count == valueHeadCount,
              a.count == batchSize * tokenCount * valueHeadCount
        else {
            throw QwenGDNReferenceError.invalidShape
        }

        var decay = Array(repeating: Float.zero, count: a.count)
        for batch in 0..<batchSize {
            for token in 0..<tokenCount {
                for head in 0..<valueHeadCount {
                    let index = gateIndex(
                        batch: batch,
                        token: token,
                        head: head,
                        tokenCount: tokenCount,
                        headCount: valueHeadCount
                    )
                    let dt = softplus(a[index] + dtBias[head])
                    decay[index] = expFloat(-expFloat(aLog[head]) * dt)
                }
            }
        }
        return decay
    }

    public static func gatedDeltaUpdate(
        query: [Float],
        key: [Float],
        value: [Float],
        a: [Float],
        b: [Float],
        aLog: [Float],
        dtBias: [Float],
        initialState: [Float]? = nil,
        batchSize: Int,
        tokenCount: Int,
        keyHeadCount: Int,
        valueHeadCount: Int,
        keyHeadDimension: Int,
        valueHeadDimension: Int
    ) throws -> QwenGatedDeltaReferenceUpdate {
        guard batchSize > 0,
              tokenCount > 0,
              keyHeadCount > 0,
              valueHeadCount > 0,
              keyHeadDimension > 0,
              valueHeadDimension > 0
        else {
            throw QwenGDNReferenceError.invalidShape
        }
        guard valueHeadCount % keyHeadCount == 0 else {
            throw QwenGDNReferenceError.invalidHeadMapping(
                keyHeadCount: keyHeadCount,
                valueHeadCount: valueHeadCount
            )
        }

        let queryElementCount = batchSize * tokenCount * keyHeadCount * keyHeadDimension
        let keyElementCount = queryElementCount
        let valueElementCount = batchSize * tokenCount * valueHeadCount * valueHeadDimension
        let gateElementCount = batchSize * tokenCount * valueHeadCount
        let stateElementCount = batchSize * valueHeadCount * valueHeadDimension * keyHeadDimension
        guard query.count == queryElementCount,
              key.count == keyElementCount,
              value.count == valueElementCount,
              a.count == gateElementCount,
              b.count == gateElementCount,
              aLog.count == valueHeadCount,
              dtBias.count == valueHeadCount
        else {
            throw QwenGDNReferenceError.invalidShape
        }
        if let initialState {
            guard initialState.count == stateElementCount else {
                throw QwenGDNReferenceError.invalidShape
            }
        }

        let repeatFactor = valueHeadCount / keyHeadCount
        var updatedState = initialState ?? Array(repeating: Float.zero, count: stateElementCount)
        var output = Array(repeating: Float.zero, count: valueElementCount)

        for batch in 0..<batchSize {
            for token in 0..<tokenCount {
                for valueHead in 0..<valueHeadCount {
                    let keyHead = valueHead / repeatFactor
                    let gate = gateIndex(
                        batch: batch,
                        token: token,
                        head: valueHead,
                        tokenCount: tokenCount,
                        headCount: valueHeadCount
                    )
                    let decay = expFloat(-expFloat(aLog[valueHead]) * softplus(a[gate] + dtBias[valueHead]))
                    let beta = sigmoid(b[gate])

                    for valueDimension in 0..<valueHeadDimension {
                        var kvMemory = Float.zero
                        for keyDimension in 0..<keyHeadDimension {
                            let state = stateIndex(
                                batch: batch,
                                valueHead: valueHead,
                                valueDimension: valueDimension,
                                keyDimension: keyDimension,
                                valueHeadCount: valueHeadCount,
                                valueHeadDimension: valueHeadDimension,
                                keyHeadDimension: keyHeadDimension
                            )
                            let keyValue = key[sequenceHeadIndex(
                                batch: batch,
                                token: token,
                                head: keyHead,
                                dimension: keyDimension,
                                tokenCount: tokenCount,
                                headCount: keyHeadCount,
                                headDimension: keyHeadDimension
                            )]
                            let decayedState = updatedState[state] * decay
                            updatedState[state] = decayedState
                            kvMemory += decayedState * keyValue
                        }

                        let valueIndex = sequenceHeadIndex(
                            batch: batch,
                            token: token,
                            head: valueHead,
                            dimension: valueDimension,
                            tokenCount: tokenCount,
                            headCount: valueHeadCount,
                            headDimension: valueHeadDimension
                        )
                        let delta = (value[valueIndex] - kvMemory) * beta
                        var projected = Float.zero
                        for keyDimension in 0..<keyHeadDimension {
                            let state = stateIndex(
                                batch: batch,
                                valueHead: valueHead,
                                valueDimension: valueDimension,
                                keyDimension: keyDimension,
                                valueHeadCount: valueHeadCount,
                                valueHeadDimension: valueHeadDimension,
                                keyHeadDimension: keyHeadDimension
                            )
                            let keyValue = key[sequenceHeadIndex(
                                batch: batch,
                                token: token,
                                head: keyHead,
                                dimension: keyDimension,
                                tokenCount: tokenCount,
                                headCount: keyHeadCount,
                                headDimension: keyHeadDimension
                            )]
                            let queryValue = query[sequenceHeadIndex(
                                batch: batch,
                                token: token,
                                head: keyHead,
                                dimension: keyDimension,
                                tokenCount: tokenCount,
                                headCount: keyHeadCount,
                                headDimension: keyHeadDimension
                            )]
                            let newState = updatedState[state] + keyValue * delta
                            updatedState[state] = newState
                            projected += newState * queryValue
                        }
                        output[valueIndex] = projected
                    }
                }
            }
        }

        return QwenGatedDeltaReferenceUpdate(output: output, updatedState: updatedState)
    }

    public static func step(
        state: [Float],
        query: [Float],
        key: [Float],
        value: [Float],
        decay: [Float],
        headCount: Int,
        headDimension: Int
    ) throws -> QwenGDNReferenceStep {
        let updatedState = try updateState(
            state: state,
            key: key,
            value: value,
            decay: decay,
            headCount: headCount,
            headDimension: headDimension
        )
        let output = try readout(
            query: query,
            state: updatedState,
            headCount: headCount,
            headDimension: headDimension
        )
        return QwenGDNReferenceStep(updatedState: updatedState, output: output)
    }

    public static func updateState(
        state: [Float],
        key: [Float],
        value: [Float],
        decay: [Float],
        headCount: Int,
        headDimension: Int
    ) throws -> [Float] {
        guard headCount > 0,
              headDimension > 0,
              state.count == headCount * headDimension * headDimension,
              key.count == headCount * headDimension,
              value.count == headCount * headDimension,
              decay.count == headCount
        else {
            throw QwenGDNReferenceError.invalidShape
        }

        var updated = state
        for head in 0..<headCount {
            for row in 0..<headDimension {
                for column in 0..<headDimension {
                    let stateIndex = ((head * headDimension + row) * headDimension) + column
                    let keyIndex = head * headDimension + row
                    let valueIndex = head * headDimension + column
                    updated[stateIndex] = state[stateIndex] * decay[head]
                        + key[keyIndex] * value[valueIndex]
                }
            }
        }
        return updated
    }

    public static func readout(
        query: [Float],
        state: [Float],
        headCount: Int,
        headDimension: Int
    ) throws -> [Float] {
        guard headCount > 0,
              headDimension > 0,
              query.count == headCount * headDimension,
              state.count == headCount * headDimension * headDimension
        else {
            throw QwenGDNReferenceError.invalidShape
        }

        var output = Array(repeating: Float.zero, count: headCount * headDimension)
        for head in 0..<headCount {
            for column in 0..<headDimension {
                var acc = Float.zero
                for row in 0..<headDimension {
                    let queryIndex = head * headDimension + row
                    let stateIndex = ((head * headDimension + row) * headDimension) + column
                    acc += query[queryIndex] * state[stateIndex]
                }
                output[head * headDimension + column] = acc
            }
        }
        return output
    }

    private static func gateIndex(
        batch: Int,
        token: Int,
        head: Int,
        tokenCount: Int,
        headCount: Int
    ) -> Int {
        ((batch * tokenCount + token) * headCount) + head
    }

    private static func sequenceHeadIndex(
        batch: Int,
        token: Int,
        head: Int,
        dimension: Int,
        tokenCount: Int,
        headCount: Int,
        headDimension: Int
    ) -> Int {
        (((batch * tokenCount + token) * headCount + head) * headDimension) + dimension
    }

    private static func stateIndex(
        batch: Int,
        valueHead: Int,
        valueDimension: Int,
        keyDimension: Int,
        valueHeadCount: Int,
        valueHeadDimension: Int,
        keyHeadDimension: Int
    ) -> Int {
        (((batch * valueHeadCount + valueHead) * valueHeadDimension + valueDimension) * keyHeadDimension)
            + keyDimension
    }

    private static func softplus(_ value: Float) -> Float {
        if value > 20 {
            return value
        }
        if value < -20 {
            return expFloat(value)
        }
        return Float(Foundation.log1p(Foundation.exp(Double(value))))
    }

    private static func sigmoid(_ value: Float) -> Float {
        if value >= 0 {
            return 1 / (1 + expFloat(-value))
        }
        let expValue = expFloat(value)
        return expValue / (1 + expValue)
    }

    private static func expFloat(_ value: Float) -> Float {
        Float(Foundation.exp(Double(value)))
    }
}
