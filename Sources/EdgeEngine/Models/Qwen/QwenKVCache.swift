// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

public enum QwenKVCacheError: Error, Equatable {
    case invalidLayerIndex(Int)
    case layerIsNotFullAttention(layerIndex: Int, kind: QwenHybridLayerKind)
    case invalidCapacity(Int)
    case invalidHeadCount(Int)
    case invalidHeadDimension(Int)
    case invalidTensorShape(expected: [Int], actual: [Int])
    case cacheOverflow(capacity: Int, requestedTokenCount: Int)
    case invalidDSRBudget(maxSize: Int, sinkSize: Int, heavyBudget: Int, recentBudget: Int)
    case invalidDSRCapacity(capacity: Int, required: Int)
}

public struct QwenDSRKVCachePolicy: Equatable, Sendable {
    public var maxSize: Int
    public var heavyBudget: Int
    public var recentBudget: Int
    public var sinkSize: Int
    public var evictionInterval: Int
    public var scoreActivationRatio: Float
    public var scoreDecay: Float

    public init(
        maxSize: Int,
        heavyBudget: Int,
        recentBudget: Int,
        sinkSize: Int = 4,
        evictionInterval: Int = 32,
        scoreActivationRatio: Float = 0.75,
        scoreDecay: Float = 0.95
    ) {
        self.maxSize = maxSize
        self.heavyBudget = heavyBudget
        self.recentBudget = recentBudget
        self.sinkSize = sinkSize
        self.evictionInterval = max(1, evictionInterval)
        self.scoreActivationRatio = max(0, min(1, scoreActivationRatio))
        self.scoreDecay = max(0, min(1, scoreDecay))
    }

    public var minimumTransientCapacity: Int {
        maxSize + evictionInterval + 1
    }
}

public enum QwenDSRKVCacheScene: String, Codable, CaseIterable, Sendable {
    case chat
    case code
    case image
    case translate
    case summary
    case creative

    public var heavyRatio: Double {
        switch self {
        case .chat: return 0.50
        case .code: return 0.65
        case .image: return 0.60
        case .translate: return 0.30
        case .summary: return 0.70
        case .creative: return 0.55
        }
    }
}

public extension QwenDSRKVCachePolicy {
    static func layerAwarePolicies(
        for architecture: QwenHybridArchitecture,
        maxSize: Int,
        heavyBudget explicitHeavyBudget: Int? = nil,
        recentBudget explicitRecentBudget: Int? = nil,
        sinkSize: Int = 4,
        scene: QwenDSRKVCacheScene = .chat,
        evictionInterval: Int = 32,
        scoreActivationRatio: Float = 0.75,
        scoreDecay: Float = 0.95
    ) throws -> [Int: QwenDSRKVCachePolicy] {
        let usableBudget = maxSize - sinkSize
        guard usableBudget >= 2 else {
            throw QwenKVCacheError.invalidDSRBudget(
                maxSize: maxSize,
                sinkSize: sinkSize,
                heavyBudget: explicitHeavyBudget ?? 0,
                recentBudget: explicitRecentBudget ?? 0
            )
        }

        let baseHeavyBudget: Int
        let baseRecentBudget: Int
        if let explicitHeavyBudget {
            baseHeavyBudget = explicitHeavyBudget
            baseRecentBudget = explicitRecentBudget ?? (usableBudget - explicitHeavyBudget)
        } else {
            baseHeavyBudget = Int(Double(usableBudget) * scene.heavyRatio)
            baseRecentBudget = usableBudget - baseHeavyBudget
        }

        let fullAttentionLayerIndices = architecture.fullAttentionLayerIndices
        let shallowEnd = fullAttentionLayerIndices.count / 3
        let deepStart = fullAttentionLayerIndices.count * 2 / 3

        func scaledBudgets(position: Int) throws -> (heavy: Int, recent: Int) {
            let scale: Double
            if position < shallowEnd {
                scale = 0.7
            } else if position >= deepStart {
                scale = 1.3
            } else {
                scale = 1.0
            }

            var heavy = max(1, Int(Double(baseHeavyBudget) * scale))
            var recent = max(1, Int(Double(baseRecentBudget) * scale))
            let maxUsableBudget = maxSize - sinkSize

            if heavy + recent > maxUsableBudget {
                let excess = heavy + recent - maxUsableBudget
                heavy = max(1, heavy - ((excess + 1) / 2))
                let remainingExcess = max(0, heavy + recent - maxUsableBudget)
                recent = max(1, recent - remainingExcess)
            }
            guard heavy + recent <= maxUsableBudget else {
                throw QwenKVCacheError.invalidDSRBudget(
                    maxSize: maxSize,
                    sinkSize: sinkSize,
                    heavyBudget: heavy,
                    recentBudget: recent
                )
            }
            return (heavy, recent)
        }

        var policies: [Int: QwenDSRKVCachePolicy] = [:]
        for (position, layerIndex) in fullAttentionLayerIndices.enumerated() {
            let budgets = try scaledBudgets(position: position)
            policies[layerIndex] = QwenDSRKVCachePolicy(
                maxSize: maxSize,
                heavyBudget: budgets.heavy,
                recentBudget: budgets.recent,
                sinkSize: sinkSize,
                evictionInterval: evictionInterval,
                scoreActivationRatio: scoreActivationRatio,
                scoreDecay: scoreDecay
            )
        }
        return policies
    }
}

public struct QwenKVCacheShape: Equatable, Sendable {
    public var layerIndex: Int
    public var capacity: Int
    public var keyValueHeadCount: Int
    public var headDimension: Int

    public init(
        layerIndex: Int,
        capacity: Int,
        keyValueHeadCount: Int,
        headDimension: Int
    ) throws {
        guard layerIndex >= 0 else {
            throw QwenKVCacheError.invalidLayerIndex(layerIndex)
        }
        guard capacity > 0 else {
            throw QwenKVCacheError.invalidCapacity(capacity)
        }
        guard keyValueHeadCount > 0 else {
            throw QwenKVCacheError.invalidHeadCount(keyValueHeadCount)
        }
        guard headDimension > 0 else {
            throw QwenKVCacheError.invalidHeadDimension(headDimension)
        }
        self.layerIndex = layerIndex
        self.capacity = capacity
        self.keyValueHeadCount = keyValueHeadCount
        self.headDimension = headDimension
    }

    public var tensorShape: EdgeTensorShape {
        EdgeTensorShape([capacity, keyValueHeadCount * headDimension])
    }

    public var rowShape: [Int] {
        [keyValueHeadCount * headDimension]
    }

    public var elementCount: Int {
        capacity * keyValueHeadCount * headDimension
    }

    public static func shape(
        for architecture: QwenHybridArchitecture,
        layerIndex: Int,
        capacity: Int? = nil
    ) throws -> QwenKVCacheShape {
        try architecture.validate()
        guard layerIndex >= 0, layerIndex < architecture.layerPlan.count else {
            throw QwenKVCacheError.invalidLayerIndex(layerIndex)
        }
        let layerKind = architecture.layerPlan[layerIndex].kind
        guard layerKind == .fullAttention else {
            throw QwenKVCacheError.layerIsNotFullAttention(
                layerIndex: layerIndex,
                kind: layerKind
            )
        }
        return try QwenKVCacheShape(
            layerIndex: layerIndex,
            capacity: capacity ?? architecture.contextLength,
            keyValueHeadCount: architecture.keyValueHeadCount,
            headDimension: architecture.attentionHeadDimension
        )
    }

    public static func shapes(
        for architecture: QwenHybridArchitecture,
        capacity: Int? = nil
    ) throws -> [QwenKVCacheShape] {
        try architecture.fullAttentionLayerIndices.map { layerIndex in
            try shape(for: architecture, layerIndex: layerIndex, capacity: capacity)
        }
    }
}

public final class QwenKVCache {
    public let shape: QwenKVCacheShape
    public let keys: EdgeTensor
    public let values: EdgeTensor
    public let dsrPolicy: QwenDSRKVCachePolicy?
    public private(set) var tokenCount: Int
    public private(set) var activeTokenCount: Int

    private let attentionScores: EdgeTensor?
    private var hasAttentionScores: Bool
    private var tokensSinceEviction: Int

    public init(
        shape: QwenKVCacheShape,
        runtime: EdgeMetalRuntime,
        dsrPolicy: QwenDSRKVCachePolicy? = nil
    ) throws {
        if let dsrPolicy {
            guard dsrPolicy.maxSize > 0,
                  dsrPolicy.sinkSize >= 0,
                  dsrPolicy.heavyBudget > 0,
                  dsrPolicy.recentBudget > 0,
                  dsrPolicy.sinkSize + dsrPolicy.heavyBudget + dsrPolicy.recentBudget <= dsrPolicy.maxSize
            else {
                throw QwenKVCacheError.invalidDSRBudget(
                    maxSize: dsrPolicy.maxSize,
                    sinkSize: dsrPolicy.sinkSize,
                    heavyBudget: dsrPolicy.heavyBudget,
                    recentBudget: dsrPolicy.recentBudget
                )
            }
            guard shape.capacity >= dsrPolicy.minimumTransientCapacity else {
                throw QwenKVCacheError.invalidDSRCapacity(
                    capacity: shape.capacity,
                    required: dsrPolicy.minimumTransientCapacity
                )
            }
        }
        self.shape = shape
        self.keys = try EdgeTensor(
            float32: Array(repeating: 0, count: shape.elementCount),
            shape: shape.tensorShape,
            runtime: runtime
        )
        self.values = try EdgeTensor(
            float32: Array(repeating: 0, count: shape.elementCount),
            shape: shape.tensorShape,
            runtime: runtime
        )
        if dsrPolicy != nil {
            self.attentionScores = try EdgeTensor(
                float32: Array(repeating: 0, count: shape.capacity * shape.keyValueHeadCount),
                shape: EdgeTensorShape([shape.capacity, shape.keyValueHeadCount]),
                runtime: runtime
            )
        } else {
            self.attentionScores = nil
        }
        self.dsrPolicy = dsrPolicy
        self.tokenCount = 0
        self.activeTokenCount = 0
        self.hasAttentionScores = false
        self.tokensSinceEviction = 0
    }

    public var remainingCapacity: Int {
        shape.capacity - activeTokenCount
    }

    public func append(
        keys newKeys: EdgeTensor,
        values newValues: EdgeTensor,
        executor: MetalKernelExecutor
    ) throws {
        let expectedColumns = shape.keyValueHeadCount * shape.headDimension
        guard newKeys.shape.rank == 2,
              newKeys.shape.dimensions[1] == expectedColumns
        else {
            throw QwenKVCacheError.invalidTensorShape(
                expected: [-1, expectedColumns],
                actual: newKeys.shape.dimensions
            )
        }
        guard newValues.shape.rank == 2,
              newValues.shape.dimensions == newKeys.shape.dimensions
        else {
            throw QwenKVCacheError.invalidTensorShape(
                expected: newKeys.shape.dimensions,
                actual: newValues.shape.dimensions
            )
        }

        let newTokenCount = newKeys.shape.dimensions[0]
        guard newTokenCount > 0 else {
            throw QwenKVCacheError.invalidTensorShape(
                expected: [-1, expectedColumns],
                actual: newKeys.shape.dimensions
            )
        }
        if dsrPolicy != nil, activeTokenCount + newTokenCount > shape.capacity {
            try evictIfNeeded(executor: executor, force: true)
        }
        let requestedTokenCount = activeTokenCount + newTokenCount
        guard requestedTokenCount <= shape.capacity else {
            throw QwenKVCacheError.cacheOverflow(
                capacity: shape.capacity,
                requestedTokenCount: requestedTokenCount
            )
        }

        try executor.copyRowsToPrefix(source: newKeys, destination: keys, startRow: activeTokenCount)
        try executor.copyRowsToPrefix(source: newValues, destination: values, startRow: activeTokenCount)
        activeTokenCount = requestedTokenCount
        tokenCount += newTokenCount
    }

    public func updateDSRScores(
        query: EdgeTensor,
        executor: MetalKernelExecutor,
        queryHeadCount: Int,
        headDimension: Int,
        scale: Float? = nil
    ) throws {
        guard let dsrPolicy, let attentionScores else {
            return
        }
        guard query.shape.rank == 2, query.shape.dimensions[0] == 1 else {
            return
        }
        guard activeTokenCount >= Int(Float(dsrPolicy.maxSize) * dsrPolicy.scoreActivationRatio) else {
            return
        }

        try executor.updateAttentionScoreEMA(
            query: query,
            key: keys,
            scores: attentionScores,
            keyValueTokenCount: activeTokenCount,
            queryHeadCount: queryHeadCount,
            keyValueHeadCount: shape.keyValueHeadCount,
            headDimension: headDimension,
            scale: scale,
            decay: dsrPolicy.scoreDecay,
            hasExistingScores: hasAttentionScores
        )
        hasAttentionScores = true
        tokensSinceEviction += 1
        guard tokensSinceEviction >= dsrPolicy.evictionInterval else {
            return
        }
        tokensSinceEviction = 0
        try evictIfNeeded(executor: executor)
    }

    private func evictIfNeeded(executor: MetalKernelExecutor, force: Bool = false) throws {
        guard let dsrPolicy else {
            return
        }
        guard activeTokenCount > dsrPolicy.maxSize || (force && activeTokenCount >= dsrPolicy.maxSize) else {
            return
        }
        let keepIndices = try makeDSRKeepIndices(policy: dsrPolicy)
        guard keepIndices.count < activeTokenCount else {
            return
        }

        let gatheredKeys = try executor.gatherRows(source: keys, rowIndices: keepIndices)
        let gatheredValues = try executor.gatherRows(source: values, rowIndices: keepIndices)
        try executor.copyRowsToPrefix(source: gatheredKeys, destination: keys, startRow: 0)
        try executor.copyRowsToPrefix(source: gatheredValues, destination: values, startRow: 0)

        if let attentionScores {
            let gatheredScores = try executor.gatherRows(source: attentionScores, rowIndices: keepIndices)
            try executor.copyRowsToPrefix(source: gatheredScores, destination: attentionScores, startRow: 0)
        }
        activeTokenCount = keepIndices.count
    }

    private func makeDSRKeepIndices(policy: QwenDSRKVCachePolicy) throws -> [Int] {
        let activeCount = activeTokenCount
        guard activeCount > 0 else {
            return []
        }

        let sinkCount = min(policy.sinkSize, activeCount)
        let recentStart = max(sinkCount, activeCount - policy.recentBudget)
        let sinkIndices = Array(0..<sinkCount)
        let recentIndices = recentStart < activeCount ? Array(recentStart..<activeCount) : []
        let middleStart = sinkCount
        let middleEnd = recentStart

        var heavyIndices: [Int] = []
        if middleEnd > middleStart, policy.heavyBudget > 0, hasAttentionScores, let attentionScores {
            let scoreValues = try attentionScores.readFloat32()
            let headCount = shape.keyValueHeadCount
            let scored = (middleStart..<middleEnd).map { tokenIndex -> (Int, Float) in
                let rowOffset = tokenIndex * headCount
                let score = scoreValues[rowOffset..<(rowOffset + headCount)].reduce(Float.zero, +)
                    / Float(headCount)
                return (tokenIndex, score)
            }
            heavyIndices = scored
                .sorted { lhs, rhs in
                    if lhs.1 == rhs.1 {
                        return lhs.0 < rhs.0
                    }
                    return lhs.1 > rhs.1
                }
                .prefix(min(policy.heavyBudget, scored.count))
                .map(\.0)
                .sorted()
        } else if middleEnd > middleStart {
            let fallbackBudget = max(0, policy.maxSize - sinkIndices.count - recentIndices.count)
            if fallbackBudget > 0 {
                heavyIndices = Array(Array(middleStart..<middleEnd).suffix(fallbackBudget))
            }
        }

        return Array(Set(sinkIndices + heavyIndices + recentIndices)).sorted()
    }

    public func reset() {
        keys.waitForPendingWork()
        values.waitForPendingWork()
        keys.buffer.contents().initializeMemory(
            as: UInt8.self,
            repeating: 0,
            count: keys.byteCount
        )
        values.buffer.contents().initializeMemory(
            as: UInt8.self,
            repeating: 0,
            count: values.byteCount
        )
        if let attentionScores {
            attentionScores.waitForPendingWork()
            attentionScores.buffer.contents().initializeMemory(
                as: UInt8.self,
                repeating: 0,
                count: attentionScores.byteCount
            )
        }
        tokenCount = 0
        activeTokenCount = 0
        hasAttentionScores = false
        tokensSinceEviction = 0
    }
}
