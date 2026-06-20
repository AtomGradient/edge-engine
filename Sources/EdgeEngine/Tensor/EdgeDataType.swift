// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import MetalPerformanceShadersGraph

public enum EdgeDataType: Equatable, Sendable {
    case float32

    public var byteSize: Int {
        switch self {
        case .float32:
            return 4
        }
    }

    var mpsDataType: MPSDataType {
        switch self {
        case .float32:
            return .float32
        }
    }
}
