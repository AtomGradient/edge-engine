// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Accelerate
import Foundation

public enum EdgeLogMelSpectrogramError: Error, Equatable {
    case invalidFFTSize(Int)
    case invalidHopLength(Int)
    case invalidMelBinCount(Int)
    case invalidFrequencyRange(min: Float, max: Float)
}

public struct EdgeLogMelSpectrogramConfiguration: Codable, Equatable, Sendable {
    public var sampleRate: Int
    public var fftSize: Int
    public var hopLength: Int
    public var melBinCount: Int
    public var minimumFrequency: Float
    public var maximumFrequency: Float
    public var logFloor: Float

    public init(
        sampleRate: Int = 16_000,
        fftSize: Int = 400,
        hopLength: Int = 160,
        melBinCount: Int = 128,
        minimumFrequency: Float = 0,
        maximumFrequency: Float = 8_000,
        logFloor: Float = 1e-10
    ) throws {
        guard sampleRate > 0 else {
            throw EdgeAudioError.invalidSampleRate(sampleRate)
        }
        guard fftSize > 1 else {
            throw EdgeLogMelSpectrogramError.invalidFFTSize(fftSize)
        }
        guard hopLength > 0 else {
            throw EdgeLogMelSpectrogramError.invalidHopLength(hopLength)
        }
        guard melBinCount > 0 else {
            throw EdgeLogMelSpectrogramError.invalidMelBinCount(melBinCount)
        }
        guard minimumFrequency >= 0,
              maximumFrequency > minimumFrequency,
              maximumFrequency <= Float(sampleRate) / 2
        else {
            throw EdgeLogMelSpectrogramError.invalidFrequencyRange(
                min: minimumFrequency,
                max: maximumFrequency
            )
        }

        self.sampleRate = sampleRate
        self.fftSize = fftSize
        self.hopLength = hopLength
        self.melBinCount = melBinCount
        self.minimumFrequency = minimumFrequency
        self.maximumFrequency = maximumFrequency
        self.logFloor = max(logFloor, Float.leastNonzeroMagnitude)
    }

    public static var qwenASRDefault: EdgeLogMelSpectrogramConfiguration {
        get throws {
            try EdgeLogMelSpectrogramConfiguration()
        }
    }
}

public struct EdgeLogMelSpectrogram: Equatable, Sendable {
    public var configuration: EdgeLogMelSpectrogramConfiguration
    public var frames: [[Float]]

    public init(configuration: EdgeLogMelSpectrogramConfiguration, frames: [[Float]]) {
        self.configuration = configuration
        self.frames = frames
    }

    public static func extract(
        from audio: EdgeAudioBuffer,
        configuration: EdgeLogMelSpectrogramConfiguration
    ) throws -> EdgeLogMelSpectrogram {
        let source = audio.sampleRate == configuration.sampleRate
            ? audio
            : try audio.resampled(to: configuration.sampleRate)
        let samples = reflectPadded(source.monoSamples(), padding: configuration.fftSize / 2)
        let fftLength = nextPowerOfTwo(configuration.fftSize)
        let fftLog2Length = vDSP_Length(log2(Double(fftLength)))
        guard let fftSetup = vDSP_create_fftsetup(fftLog2Length, FFTRadix(kFFTRadix2)) else {
            throw EdgeLogMelSpectrogramError.invalidFFTSize(configuration.fftSize)
        }
        defer {
            vDSP_destroy_fftsetup(fftSetup)
        }
        let filterBank = melFilterBank(configuration: configuration, fftLength: fftLength)
        let frameCount = max(
            1,
            samples.count <= configuration.fftSize
                ? 1
                : 1 + (samples.count - configuration.fftSize) / configuration.hopLength
        )

        var frames: [[Float]] = []
        frames.reserveCapacity(frameCount)

        for frameIndex in 0..<frameCount {
            let start = frameIndex * configuration.hopLength
            let windowed = windowedFrame(
                samples: samples,
                start: start,
                fftSize: configuration.fftSize
            )
            let power = powerSpectrum(
                windowed,
                fftLength: fftLength,
                fftLog2Length: fftLog2Length,
                fftSetup: fftSetup
            )
            frames.append(apply(filterBank: filterBank, to: power, logFloor: configuration.logFloor))
        }

        return EdgeLogMelSpectrogram(
            configuration: configuration,
            frames: normalizeWhisperLogMel(frames, logFloor: configuration.logFloor)
        )
    }

    private static func reflectPadded(_ samples: [Float], padding: Int) -> [Float] {
        guard padding > 0, samples.count > 1 else { return samples }

        let prefixEnd = min(padding + 1, samples.count)
        let prefix = samples[1..<prefixEnd].reversed()

        let suffixStart = max(0, samples.count - padding - 1)
        let suffixEnd = max(1, samples.count - 1)
        let suffix = suffixStart < suffixEnd
            ? Array(samples[suffixStart..<suffixEnd].reversed())
            : []

        return Array(prefix) + samples + suffix
    }

    private static func windowedFrame(samples: [Float], start: Int, fftSize: Int) -> [Float] {
        var frame = [Float](repeating: 0, count: fftSize)
        guard fftSize > 1 else { return frame }

        for index in 0..<fftSize {
            let sampleIndex = start + index
            let sample = sampleIndex < samples.count ? samples[sampleIndex] : 0
            let window = 0.5 - 0.5 * Foundation.cos((2.0 * Double.pi * Double(index)) / Double(fftSize - 1))
            frame[index] = sample * Float(window)
        }
        return frame
    }

    private static func powerSpectrum(
        _ frame: [Float],
        fftLength: Int,
        fftLog2Length: vDSP_Length,
        fftSetup: FFTSetup
    ) -> [Float] {
        let halfLength = fftLength / 2
        let binCount = halfLength + 1
        var spectrum = [Float](repeating: 0, count: binCount)
        var padded = [Float](repeating: 0, count: fftLength)
        padded.replaceSubrange(0..<min(frame.count, fftLength), with: frame.prefix(fftLength))

        var real = [Float](repeating: 0, count: halfLength)
        var imaginary = [Float](repeating: 0, count: halfLength)
        real.withUnsafeMutableBufferPointer { realBuffer in
            imaginary.withUnsafeMutableBufferPointer { imaginaryBuffer in
                var split = DSPSplitComplex(
                    realp: realBuffer.baseAddress!,
                    imagp: imaginaryBuffer.baseAddress!
                )
                padded.withUnsafeBufferPointer { paddedBuffer in
                    paddedBuffer.baseAddress!.withMemoryRebound(
                        to: DSPComplex.self,
                        capacity: halfLength
                    ) { complexBuffer in
                        vDSP_ctoz(
                            complexBuffer,
                            2,
                            &split,
                            1,
                            vDSP_Length(halfLength)
                        )
                    }
                }
                vDSP_fft_zrip(fftSetup, &split, 1, fftLog2Length, FFTDirection(FFT_FORWARD))
                spectrum[0] = realBuffer[0] * realBuffer[0]
                spectrum[halfLength] = imaginaryBuffer[0] * imaginaryBuffer[0]
                if halfLength > 1 {
                    for bin in 1..<halfLength {
                        spectrum[bin] = realBuffer[bin] * realBuffer[bin]
                            + imaginaryBuffer[bin] * imaginaryBuffer[bin]
                    }
                }
            }
        }
        return spectrum
    }

    private static func melFilterBank(
        configuration: EdgeLogMelSpectrogramConfiguration,
        fftLength: Int
    ) -> [[Float]] {
        let melMin = hzToSlaneyMel(configuration.minimumFrequency)
        let melMax = hzToSlaneyMel(configuration.maximumFrequency)
        let points = (0..<(configuration.melBinCount + 2)).map { index in
            let t = Float(index) / Float(configuration.melBinCount + 1)
            return slaneyMelToHz(melMin + t * (melMax - melMin), fMin: configuration.minimumFrequency)
        }

        let spectrumBinCount = fftLength / 2 + 1
        return (0..<configuration.melBinCount).map { melIndex in
            let left = points[melIndex]
            let center = points[melIndex + 1]
            let right = points[melIndex + 2]
            return (0..<spectrumBinCount).map { fftBin in
                let frequency = Float(fftBin) * Float(configuration.sampleRate) / Float(fftLength)
                if frequency <= left || frequency >= right {
                    return 0
                }
                if frequency <= center {
                    return (frequency - left) / max(center - left, Float.leastNonzeroMagnitude)
                }
                return (right - frequency) / max(right - center, Float.leastNonzeroMagnitude)
            }
        }
    }

    private static func apply(filterBank: [[Float]], to power: [Float], logFloor: Float) -> [Float] {
        filterBank.map { filter in
            let energy = zip(filter, power).reduce(Float.zero) { partial, pair in
                partial + pair.0 * pair.1
            }
            return Foundation.log10(max(energy, logFloor))
        }
    }

    private static func normalizeWhisperLogMel(_ frames: [[Float]], logFloor: Float) -> [[Float]] {
        let maximum = frames.flatMap { $0 }.max() ?? Foundation.log10(logFloor)
        let floor = maximum - 8.0
        return frames.map { frame in
            frame.map { value in
                (max(value, floor) + 4.0) / 4.0
            }
        }
    }

    private static func hzToSlaneyMel(_ hz: Float, fMin: Float = 0) -> Float {
        let minLogHz: Float = 1_000
        let fSp: Float = 200.0 / 3.0
        if hz < minLogHz {
            return (hz - fMin) / fSp
        }
        let minLogMel = (minLogHz - fMin) / fSp
        let logStep = Foundation.log(Float(6.4)) / 27.0
        return minLogMel + Foundation.log(hz / minLogHz) / logStep
    }

    private static func slaneyMelToHz(_ mel: Float, fMin: Float = 0) -> Float {
        let minLogHz: Float = 1_000
        let fSp: Float = 200.0 / 3.0
        let minLogMel = (minLogHz - fMin) / fSp
        if mel < minLogMel {
            return fMin + fSp * mel
        }
        let logStep = Foundation.log(Float(6.4)) / 27.0
        return minLogHz * Foundation.exp(logStep * (mel - minLogMel))
    }

    private static func nextPowerOfTwo(_ value: Int) -> Int {
        var power = 1
        while power < value {
            power <<= 1
        }
        return power
    }
}
