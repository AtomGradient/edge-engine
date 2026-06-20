// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import MetalPerformanceShadersGraph

public enum MPSGraphExecutorError: Error, Equatable {
    case dtypeMismatch
    case shapeMismatch
    case invalidMatrixRank
    case matrixDimensionMismatch
    case missingGraphResult
}

/// Small MPSGraph bridge used to prove system-framework execution for Phase 0.
public final class MPSGraphExecutor {
    private let runtime: EdgeMetalRuntime

    public init(runtime: EdgeMetalRuntime) {
        self.runtime = runtime
    }

    public func add(_ lhs: EdgeTensor, _ rhs: EdgeTensor) throws -> EdgeTensor {
        guard lhs.dataType == rhs.dataType else { throw MPSGraphExecutorError.dtypeMismatch }
        guard lhs.shape == rhs.shape else { throw MPSGraphExecutorError.shapeMismatch }

        let graph = MPSGraph()
        let lhsPlaceholder = graph.placeholder(
            shape: lhs.shape.mpsShape,
            dataType: lhs.dataType.mpsDataType,
            name: "lhs"
        )
        let rhsPlaceholder = graph.placeholder(
            shape: rhs.shape.mpsShape,
            dataType: rhs.dataType.mpsDataType,
            name: "rhs"
        )
        let result = graph.addition(lhsPlaceholder, rhsPlaceholder, name: "add")

        return try runBinaryGraph(
            graph: graph,
            lhsPlaceholder: lhsPlaceholder,
            rhsPlaceholder: rhsPlaceholder,
            result: result,
            lhs: lhs,
            rhs: rhs,
            outputShape: lhs.shape
        )
    }

    public func matmul(_ lhs: EdgeTensor, _ rhs: EdgeTensor) throws -> EdgeTensor {
        guard lhs.dataType == rhs.dataType else { throw MPSGraphExecutorError.dtypeMismatch }
        guard lhs.shape.rank == 2, rhs.shape.rank == 2 else {
            throw MPSGraphExecutorError.invalidMatrixRank
        }
        let m = lhs.shape.dimensions[0]
        let k = lhs.shape.dimensions[1]
        let rhsK = rhs.shape.dimensions[0]
        let n = rhs.shape.dimensions[1]
        guard k == rhsK else {
            throw MPSGraphExecutorError.matrixDimensionMismatch
        }

        let graph = MPSGraph()
        let lhsPlaceholder = graph.placeholder(
            shape: lhs.shape.mpsShape,
            dataType: lhs.dataType.mpsDataType,
            name: "lhs"
        )
        let rhsPlaceholder = graph.placeholder(
            shape: rhs.shape.mpsShape,
            dataType: rhs.dataType.mpsDataType,
            name: "rhs"
        )
        let result = graph.matrixMultiplication(
            primary: lhsPlaceholder,
            secondary: rhsPlaceholder,
            name: "matmul"
        )

        return try runBinaryGraph(
            graph: graph,
            lhsPlaceholder: lhsPlaceholder,
            rhsPlaceholder: rhsPlaceholder,
            result: result,
            lhs: lhs,
            rhs: rhs,
            outputShape: EdgeTensorShape([m, n])
        )
    }

    public func silu(_ input: EdgeTensor) throws -> EdgeTensor {
        let graph = MPSGraph()
        let placeholder = graph.placeholder(
            shape: input.shape.mpsShape,
            dataType: input.dataType.mpsDataType,
            name: "input"
        )
        let sigmoid = graph.sigmoid(with: placeholder, name: "sigmoid")
        let result = graph.multiplication(placeholder, sigmoid, name: "silu")

        return try runUnaryGraph(
            graph: graph,
            placeholder: placeholder,
            result: result,
            input: input,
            outputShape: input.shape
        )
    }

    public func rmsNorm(
        _ input: EdgeTensor,
        weight: EdgeTensor,
        epsilon: Float = 1e-6
    ) throws -> EdgeTensor {
        guard input.dataType == .float32, weight.dataType == .float32 else {
            throw MPSGraphExecutorError.dtypeMismatch
        }
        guard input.shape.rank == 1, weight.shape == input.shape else {
            throw MPSGraphExecutorError.shapeMismatch
        }

        let graph = MPSGraph()
        let inputPlaceholder = graph.placeholder(
            shape: input.shape.mpsShape,
            dataType: input.dataType.mpsDataType,
            name: "input"
        )
        let weightPlaceholder = graph.placeholder(
            shape: weight.shape.mpsShape,
            dataType: weight.dataType.mpsDataType,
            name: "weight"
        )
        let squared = graph.multiplication(inputPlaceholder, inputPlaceholder, name: "square")
        let summed = graph.reductionSum(with: squared, axis: 0, name: "sum")
        let count = graph.constant(
            Double(input.shape.elementCount),
            dataType: input.dataType.mpsDataType
        )
        let mean = graph.division(summed, count, name: "mean")
        let epsilonTensor = graph.constant(Double(epsilon), dataType: input.dataType.mpsDataType)
        let variance = graph.addition(mean, epsilonTensor, name: "variance")
        let exponent = graph.constant(-0.5, dataType: input.dataType.mpsDataType)
        let scale = graph.power(variance, exponent, name: "rsqrt")
        let normalized = graph.multiplication(inputPlaceholder, scale, name: "normalized")
        let result = graph.multiplication(normalized, weightPlaceholder, name: "weighted")

        return try runBinaryGraph(
            graph: graph,
            lhsPlaceholder: inputPlaceholder,
            rhsPlaceholder: weightPlaceholder,
            result: result,
            lhs: input,
            rhs: weight,
            outputShape: input.shape
        )
    }

    private func runUnaryGraph(
        graph: MPSGraph,
        placeholder: MPSGraphTensor,
        result: MPSGraphTensor,
        input: EdgeTensor,
        outputShape: EdgeTensorShape
    ) throws -> EdgeTensor {
        let feeds: [MPSGraphTensor: MPSGraphTensorData] = [
            placeholder: MPSGraphTensorData(
                input.buffer,
                shape: input.shape.mpsShape,
                dataType: input.dataType.mpsDataType
            )
        ]
        let results = graph.run(
            with: runtime.commandQueue,
            feeds: feeds,
            targetTensors: [result],
            targetOperations: nil
        )
        guard let resultData = results[result] else {
            throw MPSGraphExecutorError.missingGraphResult
        }

        let count = outputShape.elementCount
        var values = Array(repeating: Float.zero, count: count)
        resultData.mpsndarray().readBytes(&values, strideBytes: nil)
        return try EdgeTensor(float32: values, shape: outputShape, runtime: runtime)
    }

    private func runBinaryGraph(
        graph: MPSGraph,
        lhsPlaceholder: MPSGraphTensor,
        rhsPlaceholder: MPSGraphTensor,
        result: MPSGraphTensor,
        lhs: EdgeTensor,
        rhs: EdgeTensor,
        outputShape: EdgeTensorShape
    ) throws -> EdgeTensor {
        let feeds: [MPSGraphTensor: MPSGraphTensorData] = [
            lhsPlaceholder: MPSGraphTensorData(
                lhs.buffer,
                shape: lhs.shape.mpsShape,
                dataType: lhs.dataType.mpsDataType
            ),
            rhsPlaceholder: MPSGraphTensorData(
                rhs.buffer,
                shape: rhs.shape.mpsShape,
                dataType: rhs.dataType.mpsDataType
            ),
        ]
        let results = graph.run(
            with: runtime.commandQueue,
            feeds: feeds,
            targetTensors: [result],
            targetOperations: nil
        )
        guard let resultData = results[result] else {
            throw MPSGraphExecutorError.missingGraphResult
        }

        let count = outputShape.elementCount
        var values = Array(repeating: Float.zero, count: count)
        resultData.mpsndarray().readBytes(&values, strideBytes: nil)
        return try EdgeTensor(float32: values, shape: outputShape, runtime: runtime)
    }
}
