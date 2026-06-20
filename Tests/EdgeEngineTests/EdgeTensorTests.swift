// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Testing
@testable import EdgeEngine

@Test func tensorStoresFloat32Values() throws {
    let runtime = try EdgeMetalRuntime()
    let tensor = try EdgeTensor(
        float32: [1, 2, 3, 4],
        shape: EdgeTensorShape([2, 2]),
        runtime: runtime
    )

    #expect(tensor.shape == EdgeTensorShape([2, 2]))
    #expect(try tensor.readFloat32() == [1, 2, 3, 4])
}

@Test func mpsGraphAdditionSmoke() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = MPSGraphExecutor(runtime: runtime)
    let lhs = try EdgeTensor(float32: [1, 2, 3, 4], shape: EdgeTensorShape([2, 2]), runtime: runtime)
    let rhs = try EdgeTensor(float32: [10, 20, 30, 40], shape: EdgeTensorShape([2, 2]), runtime: runtime)

    let result = try executor.add(lhs, rhs)

    #expect(try result.readFloat32() == [11, 22, 33, 44])
}

@Test func mpsGraphMatmulSmoke() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = MPSGraphExecutor(runtime: runtime)
    let lhs = try EdgeTensor(
        float32: [
            1, 2, 3,
            4, 5, 6,
        ],
        shape: EdgeTensorShape([2, 3]),
        runtime: runtime
    )
    let rhs = try EdgeTensor(
        float32: [
            7, 8,
            9, 10,
            11, 12,
        ],
        shape: EdgeTensorShape([3, 2]),
        runtime: runtime
    )

    let result = try executor.matmul(lhs, rhs)

    #expect(result.shape == EdgeTensorShape([2, 2]))
    #expect(try result.readFloat32() == [58, 64, 139, 154])
}
