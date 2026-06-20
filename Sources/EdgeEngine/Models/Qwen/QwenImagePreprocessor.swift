// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import CoreGraphics
import CoreImage
import Foundation
import ImageIO

public enum QwenImagePreprocessorError: Error, Equatable {
    case invalidDimension(height: Int, width: Int, factor: Int)
    case invalidAspectRatio(height: Int, width: Int)
    case invalidTargetSize(height: Int, width: Int)
    case unableToLoadImage(URL)
    case unableToRenderCIImage
    case unableToCreateImageContext
    case invalidPlanarRGBCount(expected: Int, actual: Int)
    case invalidPatchConfiguration(height: Int, width: Int, patchSize: Int, mergeSize: Int)
}

public struct QwenImageTargetSize: Equatable, Sendable {
    public var height: Int
    public var width: Int

    public init(height: Int, width: Int) {
        self.height = height
        self.width = width
    }
}

public struct QwenImageGridTHW: Codable, Equatable, Sendable {
    public var temporal: Int
    public var height: Int
    public var width: Int

    public init(temporal: Int, height: Int, width: Int) {
        self.temporal = temporal
        self.height = height
        self.width = width
    }

    public var product: Int {
        temporal * height * width
    }
}

public struct QwenImagePreprocessingResult: Equatable, Sendable {
    public var pixelValues: [Float]
    public var pixelValuesShape: [Int]
    public var imageGridTHW: QwenImageGridTHW

    public init(
        pixelValues: [Float],
        pixelValuesShape: [Int],
        imageGridTHW: QwenImageGridTHW
    ) {
        self.pixelValues = pixelValues
        self.pixelValuesShape = pixelValuesShape
        self.imageGridTHW = imageGridTHW
    }
}

public enum QwenImagePreprocessor {
    public static func targetSize(
        height: Int,
        width: Int,
        factor: Int,
        minPixels: Int,
        maxPixels: Int
    ) throws -> QwenImageTargetSize {
        if height < factor || width < factor {
            throw QwenImagePreprocessorError.invalidDimension(
                height: height,
                width: width,
                factor: factor
            )
        }
        if max(height, width) / min(height, width) > 200 {
            throw QwenImagePreprocessorError.invalidAspectRatio(height: height, width: width)
        }

        var targetHeight = max(factor, Int(round(Float(height) / Float(factor))) * factor)
        var targetWidth = max(factor, Int(round(Float(width) / Float(factor))) * factor)

        if targetHeight * targetWidth > maxPixels {
            let beta = sqrt(Float(height * width) / Float(maxPixels))
            targetHeight = Int(floor(Float(height) / beta / Float(factor))) * factor
            targetWidth = Int(floor(Float(width) / beta / Float(factor))) * factor
        } else if targetHeight * targetWidth < minPixels {
            let beta = sqrt(Float(minPixels) / Float(height * width))
            targetHeight = Int(ceil(Float(height) * beta / Float(factor))) * factor
            targetWidth = Int(ceil(Float(width) * beta / Float(factor))) * factor
        }

        targetHeight = (targetHeight / factor) * factor
        targetWidth = (targetWidth / factor) * factor
        guard targetHeight > 0, targetWidth > 0 else {
            throw QwenImagePreprocessorError.invalidTargetSize(
                height: targetHeight,
                width: targetWidth
            )
        }
        return QwenImageTargetSize(height: targetHeight, width: targetWidth)
    }

    public static func preprocessImage(
        at url: URL,
        configuration: QwenImageProcessorConfiguration
    ) throws -> QwenImagePreprocessingResult {
        let image = try loadCGImage(from: url)
        return try preprocessCGImage(image, configuration: configuration)
    }

    public static func preprocessCIImage(
        _ image: CIImage,
        configuration: QwenImageProcessorConfiguration
    ) throws -> QwenImagePreprocessingResult {
        let extent = image.extent.integral
        guard extent.width > 0, extent.height > 0 else {
            throw QwenImagePreprocessorError.unableToRenderCIImage
        }
        let context = CIContext(options: [.workingColorSpace: CGColorSpaceCreateDeviceRGB()])
        guard let cgImage = context.createCGImage(image, from: extent) else {
            throw QwenImagePreprocessorError.unableToRenderCIImage
        }
        return try preprocessCGImage(cgImage, configuration: configuration)
    }

    private static func preprocessCGImage(
        _ image: CGImage,
        configuration: QwenImageProcessorConfiguration
    ) throws -> QwenImagePreprocessingResult {
        let patchSize = configuration.patchSize ?? 16
        let mergeSize = configuration.mergeSize ?? 2
        let temporalPatchSize = configuration.temporalPatchSize ?? 2
        let factor = patchSize * mergeSize
        let target = try targetSize(
            height: image.height,
            width: image.width,
            factor: factor,
            minPixels: configuration.minPixels ?? 4 * 28 * 28,
            maxPixels: configuration.maxPixels ?? 16_384 * 28 * 28
        )
        let planarRGB = try loadNormalizedRGBPlanar(
            image: image,
            targetSize: target,
            mean: configuration.imageMean,
            std: configuration.imageStd
        )
        return try patchify(
            planarRGB: planarRGB,
            height: target.height,
            width: target.width,
            channelCount: 3,
            patchSize: patchSize,
            mergeSize: mergeSize,
            temporalPatchSize: temporalPatchSize
        )
    }

    public static func patchify(
        planarRGB: [Float],
        height: Int,
        width: Int,
        channelCount: Int = 3,
        patchSize: Int,
        mergeSize: Int,
        temporalPatchSize: Int
    ) throws -> QwenImagePreprocessingResult {
        let expectedCount = channelCount * height * width
        guard planarRGB.count == expectedCount else {
            throw QwenImagePreprocessorError.invalidPlanarRGBCount(
                expected: expectedCount,
                actual: planarRGB.count
            )
        }
        guard height % patchSize == 0,
              width % patchSize == 0,
              (height / patchSize) % mergeSize == 0,
              (width / patchSize) % mergeSize == 0
        else {
            throw QwenImagePreprocessorError.invalidPatchConfiguration(
                height: height,
                width: width,
                patchSize: patchSize,
                mergeSize: mergeSize
            )
        }

        let gridTemporal = 1
        let gridHeight = height / patchSize
        let gridWidth = width / patchSize
        let mergedHeight = gridHeight / mergeSize
        let mergedWidth = gridWidth / mergeSize
        let rowCount = gridTemporal * gridHeight * gridWidth
        let rowWidth = channelCount * temporalPatchSize * patchSize * patchSize
        var output: [Float] = []
        output.reserveCapacity(rowCount * rowWidth)

        for _ in 0..<gridTemporal {
            for blockH in 0..<mergedHeight {
                for blockW in 0..<mergedWidth {
                    for intraH in 0..<mergeSize {
                        for intraW in 0..<mergeSize {
                            let patchY = blockH * mergeSize + intraH
                            let patchX = blockW * mergeSize + intraW
                            appendPatch(
                                from: planarRGB,
                                to: &output,
                                height: height,
                                width: width,
                                channelCount: channelCount,
                                patchSize: patchSize,
                                temporalPatchSize: temporalPatchSize,
                                patchY: patchY,
                                patchX: patchX
                            )
                        }
                    }
                }
            }
        }

        return QwenImagePreprocessingResult(
            pixelValues: output,
            pixelValuesShape: [rowCount, rowWidth],
            imageGridTHW: QwenImageGridTHW(
                temporal: gridTemporal,
                height: gridHeight,
                width: gridWidth
            )
        )
    }

    private static func appendPatch(
        from planarRGB: [Float],
        to output: inout [Float],
        height: Int,
        width: Int,
        channelCount: Int,
        patchSize: Int,
        temporalPatchSize: Int,
        patchY: Int,
        patchX: Int
    ) {
        for channel in 0..<channelCount {
            for _ in 0..<temporalPatchSize {
                for y in 0..<patchSize {
                    let sourceY = patchY * patchSize + y
                    for x in 0..<patchSize {
                        let sourceX = patchX * patchSize + x
                        let index = (channel * height + sourceY) * width + sourceX
                        output.append(planarRGB[index])
                    }
                }
            }
        }
    }

    private static func loadNormalizedRGBPlanar(
        image: CGImage,
        targetSize: QwenImageTargetSize,
        mean: [Float],
        std: [Float]
    ) throws -> [Float] {
        let width = targetSize.width
        let height = targetSize.height
        let componentsPerPixel = 4
        let bytesPerRow = width * componentsPerPixel
        var rgba = [UInt8](repeating: 0, count: height * bytesPerRow)
        let colorSpace = image.colorSpace ?? CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &rgba,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
                | CGBitmapInfo.byteOrder32Big.rawValue
        ) else {
            throw QwenImagePreprocessorError.unableToCreateImageContext
        }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        let resolvedMean = mean.count == 3 ? mean : [0.5, 0.5, 0.5]
        let resolvedStd = std.count == 3 ? std : [0.5, 0.5, 0.5]
        var planar = Array(repeating: Float.zero, count: 3 * height * width)
        for y in 0..<height {
            for x in 0..<width {
                let pixelOffset = (y * width + x) * componentsPerPixel
                for channel in 0..<3 {
                    let raw = Float(rgba[pixelOffset + channel]) / 255.0
                    planar[(channel * height + y) * width + x] =
                        (raw - resolvedMean[channel]) / resolvedStd[channel]
                }
            }
        }
        return planar
    }

    private static func loadCGImage(from url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw QwenImagePreprocessorError.unableToLoadImage(url)
        }
        return image
    }
}
