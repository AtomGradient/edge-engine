// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum CPUReferenceOpsError: Error, Equatable {
    case invalidMatrixShape
    case invalidAttentionShape
    case invalidRoPEShape
    case invalidVectorLength
    case invalidRMSNormWeight
    case invalidEmbeddingShape
    case invalidDepthwiseConvShape
    case emptyInput
}

/// Small CPU reference implementation for deterministic parity tests.
///
/// These functions are not the production runtime. They are intentionally
/// straightforward oracles for validating Metal/MPSGraph kernels.
public enum CPUReferenceOps {
    public static func matmul(
        _ lhs: [Float],
        rows: Int,
        inner: Int,
        _ rhs: [Float],
        columns: Int
    ) throws -> [Float] {
        guard rows > 0, inner > 0, columns > 0,
              lhs.count == rows * inner,
              rhs.count == inner * columns
        else {
            throw CPUReferenceOpsError.invalidMatrixShape
        }

        var output = Array(repeating: Float.zero, count: rows * columns)
        for row in 0..<rows {
            for column in 0..<columns {
                var acc = Float.zero
                for index in 0..<inner {
                    acc += lhs[row * inner + index] * rhs[index * columns + column]
                }
                output[row * columns + column] = acc
            }
        }
        return output
    }

    public static func softmax(_ values: [Float]) throws -> [Float] {
        guard !values.isEmpty else { throw CPUReferenceOpsError.emptyInput }
        let maxValue = values.max()!
        let exponentials = values.map { Foundation.exp(Double($0 - maxValue)) }
        let sum = exponentials.reduce(0.0, +)
        return exponentials.map { Float($0 / sum) }
    }

    public static func rmsNorm(
        _ values: [Float],
        weight: [Float],
        epsilon: Float = 1e-6
    ) throws -> [Float] {
        guard !values.isEmpty else { throw CPUReferenceOpsError.emptyInput }
        guard values.count == weight.count else {
            throw CPUReferenceOpsError.invalidRMSNormWeight
        }

        let meanSquare = values.reduce(Float.zero) { partial, value in
            partial + value * value
        } / Float(values.count)
        let scale = 1.0 / Foundation.sqrt(meanSquare + epsilon)
        return zip(values, weight).map { value, weight in
            value * scale * weight
        }
    }

    public static func rmsNormByHead(
        _ values: [Float],
        weight: [Float],
        tokenCount: Int,
        headCount: Int,
        headDimension: Int,
        epsilon: Float = 1e-6
    ) throws -> [Float] {
        guard tokenCount > 0,
              headCount > 0,
              headDimension > 0,
              values.count == tokenCount * headCount * headDimension,
              weight.count == headDimension
        else {
            throw CPUReferenceOpsError.invalidRMSNormWeight
        }

        var output = values
        for tokenIndex in 0..<tokenCount {
            for headIndex in 0..<headCount {
                let baseIndex = (tokenIndex * headCount + headIndex) * headDimension
                var meanSquare = Float.zero
                for dimension in 0..<headDimension {
                    let value = values[baseIndex + dimension]
                    meanSquare += value * value
                }
                meanSquare /= Float(headDimension)
                let scale = 1.0 / Foundation.sqrt(meanSquare + epsilon)
                for dimension in 0..<headDimension {
                    output[baseIndex + dimension] = values[baseIndex + dimension] * scale * weight[dimension]
                }
            }
        }
        return output
    }

    public static func gdnNormalizeQK(
        query: [Float],
        key: [Float],
        tokenCount: Int,
        headCount: Int,
        headDimension: Int,
        epsilon: Float = 1e-6
    ) throws -> (query: [Float], key: [Float]) {
        guard tokenCount > 0,
              headCount > 0,
              headDimension > 0,
              query.count == key.count
        else {
            throw CPUReferenceOpsError.invalidVectorLength
        }

        let unitWeight = Array(repeating: Float(1), count: headDimension)
        let normalizedQuery = try rmsNormByHead(
            query,
            weight: unitWeight,
            tokenCount: tokenCount,
            headCount: headCount,
            headDimension: headDimension,
            epsilon: epsilon
        )
        let normalizedKey = try rmsNormByHead(
            key,
            weight: unitWeight,
            tokenCount: tokenCount,
            headCount: headCount,
            headDimension: headDimension,
            epsilon: epsilon
        )
        let invScale = Float(1) / Foundation.sqrt(Float(headDimension))
        return (
            normalizedQuery.map { $0 * invScale * invScale },
            normalizedKey.map { $0 * invScale }
        )
    }

    public static func silu(_ values: [Float]) -> [Float] {
        values.map { value in
            value / (1.0 + Foundation.exp(-value))
        }
    }

    public static func swiglu(gate: [Float], up: [Float]) throws -> [Float] {
        guard gate.count == up.count else {
            throw CPUReferenceOpsError.invalidVectorLength
        }
        return zip(silu(gate), up).map(*)
    }

    public static func sigmoidMultiply(_ values: [Float], gate: [Float]) throws -> [Float] {
        guard values.count == gate.count else {
            throw CPUReferenceOpsError.invalidVectorLength
        }
        return zip(values, gate).map { value, gateValue in
            value / (1.0 + Foundation.exp(-gateValue))
        }
    }

    public static func gdnDepthwiseConv1D(
        _ input: [Float],
        convState: [Float],
        weights: [Float],
        tokenCount: Int,
        channelCount: Int,
        kernelSize: Int
    ) throws -> (activated: [Float], nextConvState: [Float]) {
        let stateTokenCount = kernelSize - 1
        guard tokenCount > 0,
              channelCount > 0,
              kernelSize > 1,
              input.count == tokenCount * channelCount,
              convState.count == stateTokenCount * channelCount,
              weights.count == channelCount * kernelSize
        else {
            throw CPUReferenceOpsError.invalidDepthwiseConvShape
        }

        func valueAt(concatenatedTokenIndex: Int, channel: Int) -> Float {
            if concatenatedTokenIndex < stateTokenCount {
                return convState[concatenatedTokenIndex * channelCount + channel]
            }
            let inputTokenIndex = concatenatedTokenIndex - stateTokenCount
            return input[inputTokenIndex * channelCount + channel]
        }

        var activated = Array(repeating: Float.zero, count: tokenCount * channelCount)
        for tokenIndex in 0..<tokenCount {
            for channel in 0..<channelCount {
                var acc = Float.zero
                for kernelIndex in 0..<kernelSize {
                    let value = valueAt(
                        concatenatedTokenIndex: tokenIndex + kernelIndex,
                        channel: channel
                    )
                    acc += value * weights[channel * kernelSize + kernelIndex]
                }
                activated[tokenIndex * channelCount + channel] = acc / (1.0 + Foundation.exp(-acc))
            }
        }

        var nextConvState = Array(repeating: Float.zero, count: stateTokenCount * channelCount)
        for stateTokenIndex in 0..<stateTokenCount {
            for channel in 0..<channelCount {
                nextConvState[stateTokenIndex * channelCount + channel] = valueAt(
                    concatenatedTokenIndex: tokenCount + stateTokenIndex,
                    channel: channel
                )
            }
        }

        return (activated, nextConvState)
    }

    public static func add(_ lhs: [Float], _ rhs: [Float]) throws -> [Float] {
        guard lhs.count == rhs.count else {
            throw CPUReferenceOpsError.invalidVectorLength
        }
        return zip(lhs, rhs).map(+)
    }

    public static func embeddingLookup(
        tokenIds: [Int],
        embeddings: [Float],
        vocabularySize: Int,
        hiddenSize: Int
    ) throws -> [Float] {
        guard !tokenIds.isEmpty,
              vocabularySize > 0,
              hiddenSize > 0,
              embeddings.count == vocabularySize * hiddenSize,
              tokenIds.allSatisfy({ $0 >= 0 && $0 < vocabularySize })
        else {
            throw CPUReferenceOpsError.invalidEmbeddingShape
        }

        var output = Array(repeating: Float.zero, count: tokenIds.count * hiddenSize)
        for tokenIndex in 0..<tokenIds.count {
            let tokenId = tokenIds[tokenIndex]
            let sourceBase = tokenId * hiddenSize
            let outputBase = tokenIndex * hiddenSize
            for hiddenIndex in 0..<hiddenSize {
                output[outputBase + hiddenIndex] = embeddings[sourceBase + hiddenIndex]
            }
        }
        return output
    }

    public static func rotaryEmbedding(
        _ values: [Float],
        tokenCount: Int,
        headCount: Int,
        headDimension: Int,
        rotaryDimension: Int,
        base: Float,
        scale: Float = 1,
        offset: Int = 0
    ) throws -> [Float] {
        guard tokenCount > 0,
              headCount > 0,
              headDimension > 0,
              rotaryDimension > 0,
              rotaryDimension <= headDimension,
              base > 0,
              values.count == tokenCount * headCount * headDimension
        else {
            throw CPUReferenceOpsError.invalidRoPEShape
        }

        let rotaryHalf = rotaryDimension / 2
        guard rotaryHalf > 0 else {
            return values
        }

        var output = values
        for tokenIndex in 0..<tokenCount {
            let position = Float(offset + tokenIndex) * scale
            for headIndex in 0..<headCount {
                let headOffset = (tokenIndex * headCount + headIndex) * headDimension
                for pairIndex in 0..<rotaryHalf {
                    let x1Index = headOffset + pairIndex
                    let x2Index = headOffset + rotaryHalf + pairIndex
                    let inverseFrequency = Foundation.exp(
                        -Float(pairIndex) * Foundation.log(base) / Float(rotaryHalf)
                    )
                    let theta = position * inverseFrequency
                    let cosine = Foundation.cos(theta)
                    let sine = Foundation.sin(theta)
                    let x1 = values[x1Index]
                    let x2 = values[x2Index]
                    output[x1Index] = x1 * cosine - x2 * sine
                    output[x2Index] = x1 * sine + x2 * cosine
                }
            }
        }
        return output
    }

    public static func q4Matmul(
        _ lhs: [Float],
        rows: Int,
        inner: Int,
        weights: Q4WeightMatrix
    ) throws -> [Float] {
        guard rows > 0, inner > 0,
              lhs.count == rows * inner,
              weights.rows == inner
        else {
            throw CPUReferenceOpsError.invalidMatrixShape
        }

        var output = Array(repeating: Float.zero, count: rows * weights.columns)
        for row in 0..<rows {
            for column in 0..<weights.columns {
                var acc = Float.zero
                for index in 0..<inner {
                    let lhsValue = lhs[row * inner + index]
                    let weightIndex = index * weights.columns + column
                    acc += lhsValue * weights.dequantizedValue(at: weightIndex)
                }
                output[row * weights.columns + column] = acc
            }
        }
        return output
    }

    public static func scaledDotProductAttention(
        query: [Float],
        key: [Float],
        value: [Float],
        tokenCount: Int,
        queryHeadCount: Int,
        keyValueHeadCount: Int,
        headDimension: Int,
        causal: Bool = true,
        scale: Float? = nil
    ) throws -> [Float] {
        try scaledDotProductAttention(
            query: query,
            key: key,
            value: value,
            queryTokenCount: tokenCount,
            keyValueTokenCount: tokenCount,
            queryPositionOffset: 0,
            queryHeadCount: queryHeadCount,
            keyValueHeadCount: keyValueHeadCount,
            headDimension: headDimension,
            causal: causal,
            scale: scale
        )
    }

    public static func scaledDotProductAttention(
        query: [Float],
        key: [Float],
        value: [Float],
        queryTokenCount: Int,
        keyValueTokenCount: Int,
        queryPositionOffset: Int = 0,
        queryHeadCount: Int,
        keyValueHeadCount: Int,
        headDimension: Int,
        causal: Bool = true,
        scale: Float? = nil
    ) throws -> [Float] {
        guard queryTokenCount > 0,
              keyValueTokenCount > 0,
              queryPositionOffset >= 0,
              queryHeadCount > 0,
              keyValueHeadCount > 0,
              headDimension > 0,
              queryHeadCount % keyValueHeadCount == 0,
              query.count == queryTokenCount * queryHeadCount * headDimension,
              key.count == keyValueTokenCount * keyValueHeadCount * headDimension,
              value.count == keyValueTokenCount * keyValueHeadCount * headDimension
        else {
            throw CPUReferenceOpsError.invalidAttentionShape
        }

        let scale = scale ?? (1.0 / Float(Foundation.sqrt(Double(headDimension))))
        let queryHeadsPerKeyValueHead = queryHeadCount / keyValueHeadCount
        var output = Array(repeating: Float.zero, count: queryTokenCount * queryHeadCount * headDimension)

        for tokenIndex in 0..<queryTokenCount {
            let absoluteQueryPosition = queryPositionOffset + tokenIndex
            let attendableTokenCount = causal
                ? min(keyValueTokenCount, absoluteQueryPosition + 1)
                : keyValueTokenCount
            for queryHead in 0..<queryHeadCount {
                let keyValueHead = queryHead / queryHeadsPerKeyValueHead
                var scores = Array(repeating: Float.zero, count: attendableTokenCount)

                for keyTokenIndex in 0..<attendableTokenCount {
                    var dot = Float.zero
                    for dimension in 0..<headDimension {
                        dot += query[
                            ((tokenIndex * queryHeadCount + queryHead) * headDimension) + dimension
                        ] * key[
                            ((keyTokenIndex * keyValueHeadCount + keyValueHead) * headDimension) + dimension
                        ]
                    }
                    scores[keyTokenIndex] = dot * scale
                }

                let probabilities = try softmax(scores)
                for dimension in 0..<headDimension {
                    var acc = Float.zero
                    for keyTokenIndex in 0..<attendableTokenCount {
                        acc += probabilities[keyTokenIndex] * value[
                            ((keyTokenIndex * keyValueHeadCount + keyValueHead) * headDimension) + dimension
                        ]
                    }
                    output[((tokenIndex * queryHeadCount + queryHead) * headDimension) + dimension] = acc
                }
            }
        }
        return output
    }
}
