// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Testing
@testable import EdgeEngine

@Test func wavFileRoundTripsPCM16AudioBuffer() throws {
    let source = try EdgeAudioBuffer(
        sampleRate: 16_000,
        channelCount: 1,
        interleavedSamples: [-1, -0.5, 0, 0.5, 1]
    )

    let wav = try EdgeWAVFile.encodePCM16(source)
    let decoded = try EdgeWAVFile.decode(wav)

    #expect(decoded.sampleRate == 16_000)
    #expect(decoded.channelCount == 1)
    #expect(decoded.interleavedSamples.count == source.interleavedSamples.count)
    #expect(abs(decoded.interleavedSamples[0] + 0.9999695) < 1e-6)
    #expect(abs(decoded.interleavedSamples[2]) < 1e-6)
    #expect(abs(decoded.interleavedSamples[4] - 0.9999695) < 1e-6)
}

@Test func wavFileDecodesFloat32AudioData() throws {
    var data = Data()
    data.append(contentsOf: "RIFF".utf8)
    appendUInt32LE(4 + 8 + 16 + 8 + 8, to: &data)
    data.append(contentsOf: "WAVE".utf8)
    data.append(contentsOf: "fmt ".utf8)
    appendUInt32LE(16, to: &data)
    appendUInt16LE(3, to: &data)
    appendUInt16LE(1, to: &data)
    appendUInt32LE(8_000, to: &data)
    appendUInt32LE(8_000 * 4, to: &data)
    appendUInt16LE(4, to: &data)
    appendUInt16LE(32, to: &data)
    data.append(contentsOf: "data".utf8)
    appendUInt32LE(8, to: &data)
    appendUInt32LE(Float(0.25).bitPattern, to: &data)
    appendUInt32LE(Float(-0.5).bitPattern, to: &data)

    let decoded = try EdgeWAVFile.decode(data)

    #expect(decoded.sampleRate == 8_000)
    #expect(decoded.interleavedSamples == [0.25, -0.5])
}

@Test func wavFileRejectsUnsupportedFormat() throws {
    var data = Data()
    data.append(contentsOf: "RIFF".utf8)
    appendUInt32LE(4 + 8 + 16 + 8, to: &data)
    data.append(contentsOf: "WAVE".utf8)
    data.append(contentsOf: "fmt ".utf8)
    appendUInt32LE(16, to: &data)
    appendUInt16LE(6, to: &data)
    appendUInt16LE(1, to: &data)
    appendUInt32LE(8_000, to: &data)
    appendUInt32LE(8_000, to: &data)
    appendUInt16LE(1, to: &data)
    appendUInt16LE(8, to: &data)
    data.append(contentsOf: "data".utf8)
    appendUInt32LE(0, to: &data)

    var rejected = false
    do {
        _ = try EdgeWAVFile.decode(data)
        Issue.record("WAV decoder must reject unsupported codecs.")
    } catch EdgeWAVFileError.unsupportedAudioFormat(6) {
        rejected = true
    }
    #expect(rejected)
}

private func appendUInt16LE(_ value: UInt16, to data: inout Data) {
    data.append(UInt8(value & 0x00FF))
    data.append(UInt8((value >> 8) & 0x00FF))
}

private func appendUInt32LE(_ value: UInt32, to data: inout Data) {
    data.append(UInt8(value & 0x000000FF))
    data.append(UInt8((value >> 8) & 0x000000FF))
    data.append(UInt8((value >> 16) & 0x000000FF))
    data.append(UInt8((value >> 24) & 0x000000FF))
}
