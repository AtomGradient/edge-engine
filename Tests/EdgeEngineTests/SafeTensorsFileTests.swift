// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Testing
@testable import EdgeEngine

@Test func safeTensorsFileParsesHeaderAndReturnsTensorDataSlices() throws {
    let payload = floatData([1, 2, 3, 4])
    let fileData = makeSafeTensorsData(
        headerJSON: """
        {
          "__metadata__": {"format": "pt"},
          "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [0, 16]
          }
        }
        """,
        payload: payload
    )

    let file = try SafeTensorsFile(data: fileData)
    let metadata = try file.metadata(named: "model.layers.0.self_attn.q_proj.weight")

    #expect(file.metadata == ["format": "pt"])
    #expect(metadata.dtype == "F32")
    #expect(metadata.shape == [2, 2])
    #expect(metadata.elementCount == 4)
    #expect(try file.tensorData(named: metadata.name) == payload)
}

@Test func safeTensorsFileRejectsOutOfRangeOffsets() throws {
    let fileData = makeSafeTensorsData(
        headerJSON: """
        {
          "broken.weight": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [0, 128]
          }
        }
        """,
        payload: Data(repeating: 0, count: 16)
    )

    var rejectedOffsets = false
    do {
        _ = try SafeTensorsFile(data: fileData)
        Issue.record("Out-of-range tensor data offsets must be rejected.")
    } catch SafeTensorsError.invalidDataOffsets {
        rejectedOffsets = true
    }
    #expect(rejectedOffsets)
}

@Test func safeTensorsFileRejectsHeaderLengthThatExceedsFileSizeWithoutOverflow() throws {
    var headerLength = UInt64(Int.max).littleEndian
    var fileData = Data()
    withUnsafeBytes(of: &headerLength) { bytes in
        fileData.append(contentsOf: bytes)
    }

    var rejectedLength = false
    do {
        _ = try SafeTensorsFile(data: fileData)
        Issue.record("Oversized header length must be rejected before headerEnd arithmetic.")
    } catch SafeTensorsError.invalidHeaderLength(UInt64(Int.max)) {
        rejectedLength = true
    }
    #expect(rejectedLength)
}

@Test func safeTensorsFileLoadsFloat32TensorIntoEdgeTensor() throws {
    let fileData = makeSafeTensorsData(
        headerJSON: """
        {
          "linear.weight": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [0, 16]
          }
        }
        """,
        payload: floatData([1, 2, 3, 4])
    )
    let runtime = try EdgeMetalRuntime()
    let file = try SafeTensorsFile(data: fileData)

    let tensor = try file.loadFloat32Tensor(named: "linear.weight", runtime: runtime)

    #expect(tensor.shape == EdgeTensorShape([2, 2]))
    #expect(try tensor.readFloat32() == [1, 2, 3, 4])
}

@Test func safeTensorsFileLoadsBF16TensorIntoFloat32EdgeTensor() throws {
    let fileData = makeSafeTensorsData(
        headerJSON: """
        {
          "linear.weight": {
            "dtype": "BF16",
            "shape": [2, 2],
            "data_offsets": [0, 8]
          }
        }
        """,
        payload: bf16Data([1, 2, 3, 4])
    )
    let file = try SafeTensorsFile(data: fileData)
    let runtime = try EdgeMetalRuntime()

    let tensor = try file.loadFloat32Tensor(named: "linear.weight", runtime: runtime)

    #expect(tensor.shape == EdgeTensorShape([2, 2]))
    #expect(try tensor.readFloat32() == [1, 2, 3, 4])
}

@Test func safeTensorsFileLoadsTransposedFloat32Tensor() throws {
    let fileData = makeSafeTensorsData(
        headerJSON: """
        {
          "linear.weight": {
            "dtype": "F32",
            "shape": [2, 3],
            "data_offsets": [0, 24]
          }
        }
        """,
        payload: floatData([
            1, 2, 3,
            4, 5, 6,
        ])
    )
    let runtime = try EdgeMetalRuntime()
    let file = try SafeTensorsFile(data: fileData)

    let tensor = try file.loadFloat32TensorTransposed2D(named: "linear.weight", runtime: runtime)

    #expect(tensor.shape == EdgeTensorShape([3, 2]))
    #expect(try tensor.readFloat32() == [
        1, 4,
        2, 5,
        3, 6,
    ])
}

@Test func safeTensorsFileLoadsTransposedBF16Tensor() throws {
    let fileData = makeSafeTensorsData(
        headerJSON: """
        {
          "linear.weight": {
            "dtype": "BF16",
            "shape": [2, 3],
            "data_offsets": [0, 12]
          }
        }
        """,
        payload: bf16Data([
            1, 2, 3,
            4, 5, 6,
        ])
    )
    let runtime = try EdgeMetalRuntime()
    let file = try SafeTensorsFile(data: fileData)

    let tensor = try file.loadFloat32TensorTransposed2D(named: "linear.weight", runtime: runtime)

    #expect(tensor.shape == EdgeTensorShape([3, 2]))
    #expect(try tensor.readFloat32() == [
        1, 4,
        2, 5,
        3, 6,
    ])
}

@Test func safeTensorsShardFileReadsHeaderAndTensorSlicesFromDisk() throws {
    let fileURL = try writeTemporarySafeTensorsFile(
        headerJSON: """
        {
          "first.weight": {
            "dtype": "F32",
            "shape": [2],
            "data_offsets": [0, 8]
          },
          "second.weight": {
            "dtype": "F32",
            "shape": [2],
            "data_offsets": [8, 16]
          }
        }
        """,
        payload: floatData([1, 2, 3, 4])
    )

    let shard = try SafeTensorsShardFile(url: fileURL)

    #expect(try shard.metadata(named: "second.weight").shape == [2])
    #expect(try shard.tensorData(named: "first.weight") == floatData([1, 2]))
    #expect(try shard.tensorData(named: "second.weight") == floatData([3, 4]))
}

@Test func safeTensorsShardFileLoadsFloat32TensorIntoEdgeTensor() throws {
    let fileURL = try writeTemporarySafeTensorsFile(
        headerJSON: """
        {
          "linear.weight": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [0, 16]
          }
        }
        """,
        payload: floatData([1, 2, 3, 4])
    )
    let runtime = try EdgeMetalRuntime()
    let shard = try SafeTensorsShardFile(url: fileURL)

    let tensor = try shard.loadFloat32Tensor(named: "linear.weight", runtime: runtime)

    #expect(tensor.shape == EdgeTensorShape([2, 2]))
    #expect(try tensor.readFloat32() == [1, 2, 3, 4])
}

@Test func safeTensorsShardFileLoadsTransposedFloat32Tensor() throws {
    let fileURL = try writeTemporarySafeTensorsFile(
        headerJSON: """
        {
          "linear.weight": {
            "dtype": "F32",
            "shape": [2, 3],
            "data_offsets": [0, 24]
          }
        }
        """,
        payload: floatData([
            1, 2, 3,
            4, 5, 6,
        ])
    )
    let runtime = try EdgeMetalRuntime()
    let shard = try SafeTensorsShardFile(url: fileURL)

    let tensor = try shard.loadFloat32TensorTransposed2D(named: "linear.weight", runtime: runtime)

    #expect(tensor.shape == EdgeTensorShape([3, 2]))
    #expect(try tensor.readFloat32() == [
        1, 4,
        2, 5,
        3, 6,
    ])
}

@Test func safeTensorsShardFileRejectsOutOfRangeOffsetsWithoutLoadingPayload() throws {
    let fileURL = try writeTemporarySafeTensorsFile(
        headerJSON: """
        {
          "broken.weight": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [0, 128]
          }
        }
        """,
        payload: Data(repeating: 0, count: 16)
    )

    var rejectedOffsets = false
    do {
        _ = try SafeTensorsShardFile(url: fileURL)
        Issue.record("Out-of-range file-backed tensor data offsets must be rejected.")
    } catch SafeTensorsError.invalidDataOffsets {
        rejectedOffsets = true
    }
    #expect(rejectedOffsets)
}

@Test func safeTensorsFileLoadsAffineQuantizedTensor() throws {
    let payload = uint32Data([
        packQuantizedValues([0, 1, 2, 3], bits: 4),
        packQuantizedValues([4, 5, 6, 7], bits: 4),
    ])
    + floatData([1, 10, 100, 1_000])
    + floatData([0, -1, -10, -100])
    let fileData = makeSafeTensorsData(
        headerJSON: """
        {
          "linear.weight": {
            "dtype": "U32",
            "shape": [2, 1],
            "data_offsets": [0, 8]
          },
          "linear.scales": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [8, 24]
          },
          "linear.biases": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [24, 40]
          }
        }
        """,
        payload: payload
    )
    let file = try SafeTensorsFile(data: fileData)

    let quantized = try file.loadQuantizedTensor(
        weightName: "linear.weight",
        scalesName: "linear.scales",
        biasesName: "linear.biases",
        groupSize: 2,
        bits: 4
    )

    #expect(quantized.shape == [2, 4])
    #expect(quantized.packedShape == [2, 1])
    #expect(quantized.scaleShape == [2, 2])
    #expect(quantized.dequantizedValues() == [
        0, 1, 19, 29,
        390, 490, 5_900, 6_900,
    ])
}

@Test func safeTensorsFileLoadsBF16AffineQuantizedTensorCompanions() throws {
    let payload = uint32Data([packQuantizedValues([1, 2, 3, 4], bits: 4)])
        + bf16Data([1, 2])
        + bf16Data([0, 10])
    let fileData = makeSafeTensorsData(
        headerJSON: """
        {
          "linear.weight": {
            "dtype": "U32",
            "shape": [1, 1],
            "data_offsets": [0, 4]
          },
          "linear.scales": {
            "dtype": "BF16",
            "shape": [1, 2],
            "data_offsets": [4, 8]
          },
          "linear.biases": {
            "dtype": "BF16",
            "shape": [1, 2],
            "data_offsets": [8, 12]
          }
        }
        """,
        payload: payload
    )
    let file = try SafeTensorsFile(data: fileData)

    let quantized = try file.loadQuantizedTensor(
        weightName: "linear.weight",
        scalesName: "linear.scales",
        biasesName: "linear.biases",
        groupSize: 2,
        bits: 4
    )

    #expect(quantized.dequantizedValues() == [1, 2, 16, 18])
}

@Test func safeTensorsShardFileLoadsAffineQuantizedTensorFromDisk() throws {
    let payload = uint32Data([packQuantizedValues([1, 2, 3, 4], bits: 4)])
        + floatData([0.5, 2])
    let fileURL = try writeTemporarySafeTensorsFile(
        headerJSON: """
        {
          "linear.weight": {
            "dtype": "U32",
            "shape": [1, 1],
            "data_offsets": [0, 4]
          },
          "linear.scales": {
            "dtype": "F32",
            "shape": [1, 2],
            "data_offsets": [4, 12]
          }
        }
        """,
        payload: payload
    )
    let shard = try SafeTensorsShardFile(url: fileURL)

    let quantized = try shard.loadQuantizedTensor(
        weightName: "linear.weight",
        scalesName: "linear.scales",
        biasesName: nil,
        groupSize: 2,
        bits: 4
    )

    #expect(quantized.dequantizedValues() == [0.5, 1, 6, 8])
}

@Test func safeTensorsFileLoadsEightBitAffineQuantizedTensor() throws {
    let payload = uint32Data([packQuantizedValues([1, 2, 3, 4], bits: 8)])
        + floatData([0.25])
    let fileData = makeSafeTensorsData(
        headerJSON: """
        {
          "linear.weight": {
            "dtype": "U32",
            "shape": [1, 1],
            "data_offsets": [0, 4]
          },
          "linear.scales": {
            "dtype": "F32",
            "shape": [1, 1],
            "data_offsets": [4, 8]
          }
        }
        """,
        payload: payload
    )
    let file = try SafeTensorsFile(data: fileData)

    let quantized = try file.loadQuantizedTensor(
        weightName: "linear.weight",
        scalesName: "linear.scales",
        biasesName: nil,
        groupSize: 4,
        bits: 8
    )

    #expect(quantized.shape == [1, 4])
    #expect(quantized.dequantizedValues() == [0.25, 0.5, 0.75, 1])
}

@Test func safeTensorsFileLoadsSixBitAffineQuantizedTensorAcrossWordBoundary() throws {
    let payload = uint32Data(packQuantizedWords([0, 1, 2, 3, 4, 5, 6, 7], bits: 6))
        + floatData([1])
    let fileData = makeSafeTensorsData(
        headerJSON: """
        {
          "linear.weight": {
            "dtype": "U32",
            "shape": [1, 2],
            "data_offsets": [0, 8]
          },
          "linear.scales": {
            "dtype": "F32",
            "shape": [1, 1],
            "data_offsets": [8, 12]
          }
        }
        """,
        payload: payload
    )
    let file = try SafeTensorsFile(data: fileData)

    let quantized = try file.loadQuantizedTensor(
        weightName: "linear.weight",
        scalesName: "linear.scales",
        biasesName: nil,
        groupSize: 8,
        bits: 6
    )

    #expect(quantized.shape == [1, 8])
    #expect(quantized.dequantizedValues() == [0, 1, 2, 3, 4, 5, 6, 7])
}

private func makeSafeTensorsData(headerJSON: String, payload: Data) -> Data {
    let headerData = headerJSON.data(using: .utf8)!
    var headerLength = UInt64(headerData.count).littleEndian
    var data = withUnsafeBytes(of: &headerLength) { Data($0) }
    data.append(headerData)
    data.append(payload)
    return data
}

private func writeTemporarySafeTensorsFile(headerJSON: String, payload: Data) throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("edgeruntime-safetensors-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let fileURL = directory.appendingPathComponent("model.safetensors")
    try makeSafeTensorsData(headerJSON: headerJSON, payload: payload).write(to: fileURL)
    return fileURL
}

private func floatData(_ values: [Float]) -> Data {
    values.withUnsafeBufferPointer { buffer in
        Data(buffer: buffer)
    }
}

private func uint32Data(_ values: [UInt32]) -> Data {
    var data = Data()
    for value in values {
        var littleEndianValue = value.littleEndian
        data.append(withUnsafeBytes(of: &littleEndianValue) { Data($0) })
    }
    return data
}

private func bf16Data(_ values: [Float]) -> Data {
    var data = Data()
    for value in values {
        var bitPattern = UInt16(value.bitPattern >> 16).littleEndian
        data.append(withUnsafeBytes(of: &bitPattern) { Data($0) })
    }
    return data
}

private func packQuantizedValues(_ values: [UInt32], bits: Int) -> UInt32 {
    packQuantizedWords(values, bits: bits)[0]
}

private func packQuantizedWords(_ values: [UInt32], bits: Int) -> [UInt32] {
    var words = Array(repeating: UInt32.zero, count: (values.count * bits + 31) / 32)
    let mask = UInt32((1 << bits) - 1)
    for (index, value) in values.enumerated() {
        let bitOffset = index * bits
        let wordIndex = bitOffset / 32
        let shift = bitOffset % 32
        words[wordIndex] |= (value & mask) << UInt32(shift)
        if shift + bits > 32 {
            words[wordIndex + 1] |= (value & mask) >> UInt32(32 - shift)
        }
    }
    return words
}
