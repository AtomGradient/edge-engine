// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Metal
import QuartzCore

public enum DeviceBandwidthBenchmark {
    public static func run(
        bufferMB: Int? = nil,
        warmupIterations: Int = 10,
        measuredIterations: Int = 15
    ) async -> DeviceBandwidthBenchmarkResult? {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(
                    returning: runSync(
                        bufferMB: bufferMB,
                        warmupIterations: warmupIterations,
                        measuredIterations: measuredIterations
                    )
                )
            }
        }
    }

    private static func runSync(
        bufferMB: Int?,
        warmupIterations: Int,
        measuredIterations: Int
    ) -> DeviceBandwidthBenchmarkResult? {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            return nil
        }

        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void stream_triad(
            device const float4* a      [[buffer(0)]],
            device const float4* b      [[buffer(1)]],
            device       float4* dst    [[buffer(2)]],
            constant     float&  scalar [[buffer(3)]],
            uint gid [[thread_position_in_grid]]
        ) {
            dst[gid] = a[gid] + scalar * b[gid];
        }
        """

        guard let library = try? device.makeLibrary(source: shaderSource, options: nil),
              let function = library.makeFunction(name: "stream_triad"),
              let pipeline = try? device.makeComputePipelineState(function: function) else {
            return nil
        }

        #if os(iOS) || os(tvOS) || os(watchOS) || os(visionOS)
        let defaultBufferMB = 256
        #else
        let defaultBufferMB = 512
        #endif

        let bufferMB = max(16, bufferMB ?? defaultBufferMB)
        let elementSize = MemoryLayout<SIMD4<Float>>.stride
        let bufferBytes = bufferMB * 1_000_000
        let count = bufferBytes / elementSize

        guard let bufA = device.makeBuffer(length: bufferBytes, options: .storageModeShared),
              let bufB = device.makeBuffer(length: bufferBytes, options: .storageModeShared),
              let bufC = device.makeBuffer(length: bufferBytes, options: .storageModeShared) else {
            return nil
        }

        let pA = bufA.contents().bindMemory(to: SIMD4<Float>.self, capacity: count)
        let pB = bufB.contents().bindMemory(to: SIMD4<Float>.self, capacity: count)
        for i in 0..<count {
            pA[i] = SIMD4<Float>(1, 2, 3, 4)
            pB[i] = SIMD4<Float>(0.5, 0.5, 0.5, 0.5)
        }

        var scalar: Float = 2.0
        guard let scalarBuffer = device.makeBuffer(
            bytes: &scalar,
            length: MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            return nil
        }

        let threadgroupSize = MTLSize(width: 1024, height: 1, depth: 1)
        let gridSize = MTLSize(width: count, height: 1, depth: 1)

        func encode(_ commandBuffer: MTLCommandBuffer) -> Bool {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                return false
            }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(bufA, offset: 0, index: 0)
            encoder.setBuffer(bufB, offset: 0, index: 1)
            encoder.setBuffer(bufC, offset: 0, index: 2)
            encoder.setBuffer(scalarBuffer, offset: 0, index: 3)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
            return true
        }

        func dispatchAndWait() -> Bool {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  encode(commandBuffer) else {
                return false
            }
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            return commandBuffer.status == .completed && commandBuffer.error == nil
        }

        for _ in 0..<max(0, warmupIterations) {
            guard dispatchAndWait() else { return nil }
        }

        let iterations = max(1, measuredIterations)
        let bytesPerIteration = Double(bufferBytes) * 3.0
        var samples: [Double] = []
        samples.reserveCapacity(iterations)

        for _ in 0..<iterations {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  encode(commandBuffer) else {
                return nil
            }
            let t0 = CACurrentMediaTime()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            guard commandBuffer.status == .completed, commandBuffer.error == nil else {
                return nil
            }
            let elapsed = CACurrentMediaTime() - t0
            guard elapsed > 0 else { continue }
            samples.append(bytesPerIteration / elapsed / 1e9)
        }

        guard let peak = samples.max(), !samples.isEmpty else { return nil }
        let sorted = samples.sorted()
        let median = sorted[sorted.count / 2]
        let discounted = max(0, min(peak, median * 0.95))

        return DeviceBandwidthBenchmarkResult(
            peakGBs: peak,
            medianGBs: median,
            discountedGBs: discounted,
            deviceName: device.name
        )
    }
}
