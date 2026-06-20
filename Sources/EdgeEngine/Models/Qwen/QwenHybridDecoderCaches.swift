// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public enum QwenHybridDecoderCachesError: Error, Equatable {
    case missingKVCache(layerIndex: Int)
    case missingGDNCache(layerIndex: Int)
    case tokenPositionMismatch(expected: Int, actual: Int, layerIndex: Int)
}

public final class QwenHybridDecoderCaches {
    public let kvCaches: [Int: QwenKVCache]
    public let gdnCaches: [Int: QwenGDNCache]

    public init(
        architecture: QwenHybridArchitecture,
        runtime: EdgeMetalRuntime,
        kvCapacity: Int? = nil,
        dsrPolicies: [Int: QwenDSRKVCachePolicy] = [:]
    ) throws {
        let kvPairs = try QwenKVCacheShape.shapes(
            for: architecture,
            capacity: kvCapacity
        ).map { shape in
            let dsrPolicy = dsrPolicies[shape.layerIndex]
            let effectiveShape: QwenKVCacheShape
            if let dsrPolicy, shape.capacity < dsrPolicy.minimumTransientCapacity {
                effectiveShape = try QwenKVCacheShape(
                    layerIndex: shape.layerIndex,
                    capacity: dsrPolicy.minimumTransientCapacity,
                    keyValueHeadCount: shape.keyValueHeadCount,
                    headDimension: shape.headDimension
                )
            } else {
                effectiveShape = shape
            }
            return (
                effectiveShape.layerIndex,
                try QwenKVCache(
                    shape: effectiveShape,
                    runtime: runtime,
                    dsrPolicy: dsrPolicy
                )
            )
        }
        let gdnPairs = try QwenGDNCacheShape.shapes(for: architecture).map { shape in
            (shape.layerIndex, try QwenGDNCache(shape: shape, runtime: runtime))
        }
        self.kvCaches = Dictionary(uniqueKeysWithValues: kvPairs)
        self.gdnCaches = Dictionary(uniqueKeysWithValues: gdnPairs)
    }

    public var kvLayerIndices: [Int] {
        kvCaches.keys.sorted()
    }

    public var gdnLayerIndices: [Int] {
        gdnCaches.keys.sorted()
    }

    public func kvCache(layerIndex: Int) throws -> QwenKVCache {
        guard let cache = kvCaches[layerIndex] else {
            throw QwenHybridDecoderCachesError.missingKVCache(layerIndex: layerIndex)
        }
        return cache
    }

    public func gdnCache(layerIndex: Int) throws -> QwenGDNCache {
        guard let cache = gdnCaches[layerIndex] else {
            throw QwenHybridDecoderCachesError.missingGDNCache(layerIndex: layerIndex)
        }
        return cache
    }

    public func tokenPosition() throws -> Int {
        var expectedPosition: Int?
        for layerIndex in kvLayerIndices {
            let position = try kvCache(layerIndex: layerIndex).tokenCount
            if let expectedPosition, expectedPosition != position {
                throw QwenHybridDecoderCachesError.tokenPositionMismatch(
                    expected: expectedPosition,
                    actual: position,
                    layerIndex: layerIndex
                )
            }
            expectedPosition = position
        }
        for layerIndex in gdnLayerIndices {
            let position = try gdnCache(layerIndex: layerIndex).tokenPosition
            if let expectedPosition, expectedPosition != position {
                throw QwenHybridDecoderCachesError.tokenPositionMismatch(
                    expected: expectedPosition,
                    actual: position,
                    layerIndex: layerIndex
                )
            }
            expectedPosition = position
        }
        return expectedPosition ?? 0
    }

    public func reset() {
        for cache in kvCaches.values {
            cache.reset()
        }
        for cache in gdnCaches.values {
            cache.reset()
        }
    }
}
