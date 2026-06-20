// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Testing
@testable import EdgeEngine

@Test func edgeAudioBufferMixesStereoToMonoAndResamples() throws {
    let buffer = try EdgeAudioBuffer(
        sampleRate: 2,
        channelCount: 2,
        interleavedSamples: [
            0, 0,
            1, 1,
        ]
    )

    #expect(buffer.frameCount == 2)
    #expect(buffer.monoSamples() == [0, 1])

    let resampled = try buffer.resampled(to: 4)
    #expect(resampled.sampleRate == 4)
    #expect(resampled.channelCount == 1)
    #expect(resampled.interleavedSamples.count == 4)
    #expect(abs(resampled.interleavedSamples[1] - 0.33333334) < 1e-6)
    #expect(abs(resampled.interleavedSamples[2] - 0.6666667) < 1e-6)
}

@Test func edgeLogMelSpectrogramProducesFiniteFeatureFrames() throws {
    let configuration = try EdgeLogMelSpectrogramConfiguration(
        sampleRate: 8,
        fftSize: 8,
        hopLength: 4,
        melBinCount: 4,
        minimumFrequency: 0,
        maximumFrequency: 4
    )
    let buffer = try EdgeAudioBuffer(
        sampleRate: 8,
        interleavedSamples: [0, 1, 0, -1, 0, 1, 0, -1]
    )

    let features = try EdgeLogMelSpectrogram.extract(from: buffer, configuration: configuration)

    #expect(features.frames.count == 3)
    #expect(features.frames[0].count == 4)
    #expect(features.frames[0].allSatisfy { value in value.isFinite })
    #expect(features.frames.flatMap { $0 }.contains { $0 > -1 })
}

@Test func speechRuntimeContractsBuildASRAndTTSRequests() throws {
    let asrPlan = try EdgeSpeechRuntimePlan.qwen3ASR
    let ttsPlan = try EdgeSpeechRuntimePlan.qwen3TTS
    let audio = try EdgeAudioBuffer(sampleRate: 16_000, interleavedSamples: [0, 0.1, -0.1, 0])
    let asr = try EdgeASRRequest(
        audio: audio,
        languageHint: "zh",
        maxTokens: 0,
        featureConfiguration: .qwenASRDefault
    )
    let tts = try EdgeTTSRequest(text: "edge runtime", maxCodecTokens: 0)

    #expect(asrPlan.modality == .asr)
    #expect(ttsPlan.modality == .tts)
    #expect(asr.maxTokens == 1)
    #expect(tts.maxCodecTokens == 1)
    #expect(try asr.logMelFeatures().configuration.sampleRate == 16_000)
}
