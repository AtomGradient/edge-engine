// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum QwenVisionWeightStoreError: Error, Equatable {
    case tensorNotInVisionManifest(String)
}

public struct QwenVisionTensorData: Equatable, Sendable {
    public var name: String
    public var metadata: SafeTensorMetadata
    public var data: Data

    public init(name: String, metadata: SafeTensorMetadata, data: Data) {
        self.name = name
        self.metadata = metadata
        self.data = data
    }

    public var byteCount: Int {
        data.count
    }
}

public struct QwenVisionWeightSnapshot: Equatable, Sendable {
    public var tensors: [QwenVisionTensorData]

    public init(tensors: [QwenVisionTensorData]) {
        self.tensors = tensors
    }

    public var tensorNames: [String] {
        tensors.map(\.name)
    }

    public var totalByteCount: Int {
        tensors.reduce(0) { $0 + $1.byteCount }
    }
}

public final class QwenVisionWeightStore {
    public let index: QwenVLMModelBundleIndex
    private var shardsByFileName: [String: SafeTensorsShardFile]
    private let visionTensorNameSet: Set<String>

    public init(index: QwenVLMModelBundleIndex) {
        self.index = index
        self.shardsByFileName = [:]
        self.visionTensorNameSet = Set(index.visionManifest.tensorNames)
    }

    public var tensorNames: [String] {
        index.visionManifest.tensorNames
    }

    public func shard(containing tensorName: String) throws -> SafeTensorsShardFile {
        try validateVisionTensorName(tensorName)
        let fileName = try index.shardFileName(containing: tensorName)
        if let shard = shardsByFileName[fileName] {
            return shard
        }
        let shard = try SafeTensorsShardFile(
            url: index.rootURL.appendingPathComponent(fileName)
        )
        shardsByFileName[fileName] = shard
        return shard
    }

    public func metadata(named name: String) throws -> SafeTensorMetadata {
        try shard(containing: name).metadata(named: name)
    }

    public func tensorData(named name: String) throws -> Data {
        try shard(containing: name).tensorData(named: name)
    }

    public func materializeAllTensors() throws -> QwenVisionWeightSnapshot {
        let tensors = try tensorNames.map { name in
            try QwenVisionTensorData(
                name: name,
                metadata: metadata(named: name),
                data: tensorData(named: name)
            )
        }
        return QwenVisionWeightSnapshot(tensors: tensors)
    }

    private func validateVisionTensorName(_ name: String) throws {
        guard visionTensorNameSet.contains(name) else {
            throw QwenVisionWeightStoreError.tensorNotInVisionManifest(name)
        }
    }
}
