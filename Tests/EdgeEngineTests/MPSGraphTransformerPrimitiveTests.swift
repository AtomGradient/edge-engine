// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Testing
@testable import EdgeEngine

@Test func mpsGraphSiLUMatchesCPUReference() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = MPSGraphExecutor(runtime: runtime)
    let inputValues: [Float] = [-1, 0, 1, 2]
    let input = try EdgeTensor(
        float32: inputValues,
        shape: EdgeTensorShape([4]),
        runtime: runtime
    )

    let result = try executor.silu(input)
    let expected = CPUReferenceOps.silu(inputValues)
    let error = try NumericComparison.maxAbsoluteError(try result.readFloat32(), expected)

    #expect(error < 1e-6)
}

@Test func mpsGraphRMSNormMatchesCPUReference() throws {
    let runtime = try EdgeMetalRuntime()
    let executor = MPSGraphExecutor(runtime: runtime)
    let inputValues: [Float] = [1, 2, 3, 4]
    let weightValues: [Float] = [0.5, 1, 1.5, 2]
    let input = try EdgeTensor(
        float32: inputValues,
        shape: EdgeTensorShape([4]),
        runtime: runtime
    )
    let weight = try EdgeTensor(
        float32: weightValues,
        shape: EdgeTensorShape([4]),
        runtime: runtime
    )

    let result = try executor.rmsNorm(input, weight: weight, epsilon: 1e-6)
    let expected = try CPUReferenceOps.rmsNorm(inputValues, weight: weightValues, epsilon: 1e-6)
    let error = try NumericComparison.maxAbsoluteError(try result.readFloat32(), expected)

    #expect(error < 1e-5)
}
