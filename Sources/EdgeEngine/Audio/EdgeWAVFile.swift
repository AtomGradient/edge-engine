// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum EdgeWAVFileError: Error, Equatable {
    case invalidHeader
    case missingFormatChunk
    case missingDataChunk
    case unsupportedAudioFormat(UInt16)
    case unsupportedBitsPerSample(UInt16)
    case truncatedChunk(String)
    case invalidFrameByteCount
}

public enum EdgeWAVFile {
    public static func decode(_ data: Data) throws -> EdgeAudioBuffer {
        guard data.count >= 12,
              ascii(data, at: 0, count: 4) == "RIFF",
              ascii(data, at: 8, count: 4) == "WAVE"
        else {
            throw EdgeWAVFileError.invalidHeader
        }

        var format: WAVFormat?
        var audioData: Data?
        var offset = 12
        while offset + 8 <= data.count {
            let chunkID = ascii(data, at: offset, count: 4)
            let chunkSize = Int(readUInt32LE(data, at: offset + 4))
            let payloadStart = offset + 8
            let payloadEnd = payloadStart + chunkSize
            guard payloadEnd <= data.count else {
                throw EdgeWAVFileError.truncatedChunk(chunkID)
            }

            let payload = data[payloadStart..<payloadEnd]
            if chunkID == "fmt " {
                format = try WAVFormat(payload: payload)
            } else if chunkID == "data" {
                audioData = Data(payload)
            }

            offset = payloadEnd + (chunkSize.isMultiple(of: 2) ? 0 : 1)
        }

        guard let format else { throw EdgeWAVFileError.missingFormatChunk }
        guard let audioData else { throw EdgeWAVFileError.missingDataChunk }

        let bytesPerSample = Int(format.bitsPerSample / 8)
        let frameByteCount = bytesPerSample * Int(format.channelCount)
        guard frameByteCount > 0, audioData.count.isMultiple(of: frameByteCount) else {
            throw EdgeWAVFileError.invalidFrameByteCount
        }

        let samples: [Float]
        switch (format.audioFormat, format.bitsPerSample) {
        case (1, 16):
            samples = stride(from: 0, to: audioData.count, by: 2).map { byteOffset in
                Float(readInt16LE(audioData, at: byteOffset)) / 32_768.0
            }
        case (3, 32):
            samples = stride(from: 0, to: audioData.count, by: 4).map { byteOffset in
                Float(bitPattern: readUInt32LE(audioData, at: byteOffset))
            }
        case (let audioFormat, _):
            guard audioFormat == 1 || audioFormat == 3 else {
                throw EdgeWAVFileError.unsupportedAudioFormat(audioFormat)
            }
            throw EdgeWAVFileError.unsupportedBitsPerSample(format.bitsPerSample)
        }

        return try EdgeAudioBuffer(
            sampleRate: Int(format.sampleRate),
            channelCount: Int(format.channelCount),
            interleavedSamples: samples
        )
    }

    public static func encodePCM16(_ buffer: EdgeAudioBuffer) throws -> Data {
        var data = Data()
        var sampleBytes = Data()
        for sample in buffer.interleavedSamples {
            let clamped = max(-1.0, min(1.0, sample))
            let quantized = Int16((clamped * 32_767.0).rounded())
            appendInt16LE(quantized, to: &sampleBytes)
        }

        let fmtChunkSize = UInt32(16)
        let audioFormat = UInt16(1)
        let channelCount = UInt16(buffer.channelCount)
        let sampleRate = UInt32(buffer.sampleRate)
        let bitsPerSample = UInt16(16)
        let blockAlign = channelCount * (bitsPerSample / 8)
        let byteRate = sampleRate * UInt32(blockAlign)
        let dataChunkSize = UInt32(sampleBytes.count)
        let riffSize = UInt32(4 + 8 + fmtChunkSize + 8 + dataChunkSize)

        data.append(contentsOf: "RIFF".utf8)
        appendUInt32LE(riffSize, to: &data)
        data.append(contentsOf: "WAVE".utf8)

        data.append(contentsOf: "fmt ".utf8)
        appendUInt32LE(fmtChunkSize, to: &data)
        appendUInt16LE(audioFormat, to: &data)
        appendUInt16LE(channelCount, to: &data)
        appendUInt32LE(sampleRate, to: &data)
        appendUInt32LE(byteRate, to: &data)
        appendUInt16LE(blockAlign, to: &data)
        appendUInt16LE(bitsPerSample, to: &data)

        data.append(contentsOf: "data".utf8)
        appendUInt32LE(dataChunkSize, to: &data)
        data.append(sampleBytes)
        if !sampleBytes.count.isMultiple(of: 2) {
            data.append(0)
        }
        return data
    }

    private struct WAVFormat {
        var audioFormat: UInt16
        var channelCount: UInt16
        var sampleRate: UInt32
        var bitsPerSample: UInt16

        init(payload: Data.SubSequence) throws {
            guard payload.count >= 16 else {
                throw EdgeWAVFileError.truncatedChunk("fmt ")
            }
            let data = Data(payload)
            self.audioFormat = readUInt16LE(data, at: 0)
            self.channelCount = readUInt16LE(data, at: 2)
            self.sampleRate = readUInt32LE(data, at: 4)
            self.bitsPerSample = readUInt16LE(data, at: 14)
        }
    }

    private static func ascii(_ data: Data, at offset: Int, count: Int) -> String {
        guard offset >= 0, offset + count <= data.count else { return "" }
        return String(decoding: data[offset..<(offset + count)], as: UTF8.self)
    }

    private static func readUInt16LE(_ data: Data, at offset: Int) -> UInt16 {
        UInt16(data[offset]) | (UInt16(data[offset + 1]) << 8)
    }

    private static func readInt16LE(_ data: Data, at offset: Int) -> Int16 {
        Int16(bitPattern: readUInt16LE(data, at: offset))
    }

    private static func readUInt32LE(_ data: Data, at offset: Int) -> UInt32 {
        UInt32(data[offset])
            | (UInt32(data[offset + 1]) << 8)
            | (UInt32(data[offset + 2]) << 16)
            | (UInt32(data[offset + 3]) << 24)
    }

    private static func appendUInt16LE(_ value: UInt16, to data: inout Data) {
        data.append(UInt8(value & 0x00FF))
        data.append(UInt8((value >> 8) & 0x00FF))
    }

    private static func appendInt16LE(_ value: Int16, to data: inout Data) {
        appendUInt16LE(UInt16(bitPattern: value), to: &data)
    }

    private static func appendUInt32LE(_ value: UInt32, to data: inout Data) {
        data.append(UInt8(value & 0x000000FF))
        data.append(UInt8((value >> 8) & 0x000000FF))
        data.append(UInt8((value >> 16) & 0x000000FF))
        data.append(UInt8((value >> 24) & 0x000000FF))
    }
}
