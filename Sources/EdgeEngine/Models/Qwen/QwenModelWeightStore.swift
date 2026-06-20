// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public final class QwenModelWeightStore {
    public let bundleIndex: QwenModelBundleIndex
    private var shardsByFileName: [String: SafeTensorsShardFile]

    public init(bundleIndex: QwenModelBundleIndex) {
        self.bundleIndex = bundleIndex
        self.shardsByFileName = [:]
    }

    public func shard(containing tensorName: String) throws -> SafeTensorsShardFile {
        let fileName = try bundleIndex.shardFileName(containing: tensorName)
        if let shard = shardsByFileName[fileName] {
            return shard
        }
        let shard = try SafeTensorsShardFile(url: bundleIndex.rootURL.appendingPathComponent(fileName))
        shardsByFileName[fileName] = shard
        return shard
    }

    public func metadata(named name: String) throws -> SafeTensorMetadata {
        try shard(containing: name).metadata(named: name)
    }

    public func tensorData(named name: String) throws -> Data {
        try shard(containing: name).tensorData(named: name)
    }

    public func loadFloat32Tensor(named name: String, runtime: EdgeMetalRuntime) throws -> EdgeTensor {
        try shard(containing: name).loadFloat32Tensor(named: name, runtime: runtime)
    }

    public func loadFloat32TensorTransposed2D(
        named name: String,
        runtime: EdgeMetalRuntime
    ) throws -> EdgeTensor {
        try shard(containing: name).loadFloat32TensorTransposed2D(named: name, runtime: runtime)
    }

    public func loadQuantizedTensor(
        weightName: String,
        scalesName: String,
        biasesName: String?,
        groupSize: Int,
        bits: Int
    ) throws -> EdgeQuantizedTensor {
        try shard(containing: weightName).loadQuantizedTensor(
            weightName: weightName,
            scalesName: scalesName,
            biasesName: biasesName,
            groupSize: groupSize,
            bits: bits
        )
    }
}
