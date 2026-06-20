// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public enum EdgeAudioError: Error, Equatable {
    case invalidSampleRate(Int)
    case invalidChannelCount(Int)
    case emptySamples
    case channelFrameMismatch(sampleCount: Int, channelCount: Int)
}

public struct EdgeAudioBuffer: Codable, Equatable, Sendable {
    public var sampleRate: Int
    public var channelCount: Int
    public var interleavedSamples: [Float]

    public init(
        sampleRate: Int,
        channelCount: Int = 1,
        interleavedSamples: [Float]
    ) throws {
        guard sampleRate > 0 else {
            throw EdgeAudioError.invalidSampleRate(sampleRate)
        }
        guard channelCount > 0 else {
            throw EdgeAudioError.invalidChannelCount(channelCount)
        }
        guard !interleavedSamples.isEmpty else {
            throw EdgeAudioError.emptySamples
        }
        guard interleavedSamples.count.isMultiple(of: channelCount) else {
            throw EdgeAudioError.channelFrameMismatch(
                sampleCount: interleavedSamples.count,
                channelCount: channelCount
            )
        }

        self.sampleRate = sampleRate
        self.channelCount = channelCount
        self.interleavedSamples = interleavedSamples
    }

    public var frameCount: Int {
        interleavedSamples.count / channelCount
    }

    public var durationSeconds: Double {
        Double(frameCount) / Double(sampleRate)
    }

    public func monoSamples() -> [Float] {
        guard channelCount > 1 else { return interleavedSamples }

        var mono = [Float](repeating: 0, count: frameCount)
        for frame in 0..<frameCount {
            var sum = Float.zero
            let base = frame * channelCount
            for channel in 0..<channelCount {
                sum += interleavedSamples[base + channel]
            }
            mono[frame] = sum / Float(channelCount)
        }
        return mono
    }

    public func resampled(to targetSampleRate: Int) throws -> EdgeAudioBuffer {
        guard targetSampleRate > 0 else {
            throw EdgeAudioError.invalidSampleRate(targetSampleRate)
        }
        guard targetSampleRate != sampleRate else {
            return try EdgeAudioBuffer(
                sampleRate: sampleRate,
                channelCount: 1,
                interleavedSamples: monoSamples()
            )
        }

        return try EdgeAudioBuffer(
            sampleRate: targetSampleRate,
            channelCount: 1,
            interleavedSamples: EdgeAudioResampler.linear(
                monoSamples(),
                from: sampleRate,
                to: targetSampleRate
            )
        )
    }
}

public enum EdgeAudioResampler {
    public static func linear(_ samples: [Float], from sourceRate: Int, to targetRate: Int) -> [Float] {
        guard sourceRate > 0, targetRate > 0, !samples.isEmpty else { return samples }
        guard sourceRate != targetRate else { return samples }

        let ratio = Double(targetRate) / Double(sourceRate)
        let outputCount = max(1, Int((Double(samples.count) * ratio).rounded()))
        guard outputCount > 1, samples.count > 1 else {
            return [Float](repeating: samples[0], count: outputCount)
        }

        let step = Double(samples.count - 1) / Double(outputCount - 1)
        var output = [Float](repeating: 0, count: outputCount)
        for index in 0..<outputCount {
            let position = Double(index) * step
            let lower = min(Int(position), samples.count - 1)
            let upper = min(lower + 1, samples.count - 1)
            let fraction = Float(position - Double(lower))
            output[index] = samples[lower] + (samples[upper] - samples[lower]) * fraction
        }
        return output
    }
}
