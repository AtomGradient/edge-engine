// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Darwin
import Foundation
import Metal

public struct DeviceCapabilityProfile: Codable, Equatable, Sendable {
    public enum ThermalState: String, Codable, Sendable {
        case nominal
        case fair
        case serious
        case critical
        case unknown
    }

    public enum BandwidthSource: String, Codable, Sendable {
        case none
        case cached
        case measured
        case estimated
    }

    public enum Confidence: String, Codable, Sendable {
        case provisional
        case cached
        case measured
    }

    public let schemaVersion: Int
    public let machineIdentifier: String
    public let cpuBrandString: String?
    public let osVersion: String
    public let totalPhysicalMemoryMB: Int
    public let availableMemoryMB: Int
    public let footprintMB: Int
    public let jetsamLimitMB: Int
    public let thermalState: ThermalState
    public let metalDeviceName: String?
    public let metalFamilyTier: Int
    public let recommendedMaxWorkingSetMB: Int?
    public let measuredBandwidthGBs: Double?
    public let measuredBandwidthMedianGBs: Double?
    public let bandwidthSource: BandwidthSource
    public let confidence: Confidence
    public let measuredAt: Date?

    public init(
        schemaVersion: Int = DeviceCapabilityProbe.schemaVersion,
        machineIdentifier: String,
        cpuBrandString: String?,
        osVersion: String,
        totalPhysicalMemoryMB: Int,
        availableMemoryMB: Int,
        footprintMB: Int,
        jetsamLimitMB: Int,
        thermalState: ThermalState,
        metalDeviceName: String?,
        metalFamilyTier: Int,
        recommendedMaxWorkingSetMB: Int?,
        measuredBandwidthGBs: Double?,
        measuredBandwidthMedianGBs: Double?,
        bandwidthSource: BandwidthSource,
        confidence: Confidence,
        measuredAt: Date?
    ) {
        self.schemaVersion = schemaVersion
        self.machineIdentifier = machineIdentifier
        self.cpuBrandString = cpuBrandString
        self.osVersion = osVersion
        self.totalPhysicalMemoryMB = totalPhysicalMemoryMB
        self.availableMemoryMB = availableMemoryMB
        self.footprintMB = footprintMB
        self.jetsamLimitMB = jetsamLimitMB
        self.thermalState = thermalState
        self.metalDeviceName = metalDeviceName
        self.metalFamilyTier = metalFamilyTier
        self.recommendedMaxWorkingSetMB = recommendedMaxWorkingSetMB
        self.measuredBandwidthGBs = measuredBandwidthGBs
        self.measuredBandwidthMedianGBs = measuredBandwidthMedianGBs
        self.bandwidthSource = bandwidthSource
        self.confidence = confidence
        self.measuredAt = measuredAt
    }

    public var cacheIdentity: String {
        [
            "v\(schemaVersion)",
            machineIdentifier,
            osVersion,
            metalDeviceName ?? "no-metal",
            "family\(metalFamilyTier)",
        ].joined(separator: "|")
    }

    public var isPhone: Bool {
        machineIdentifier.lowercased().hasPrefix("iphone")
    }

    public var isMac: Bool {
        let lower = machineIdentifier.lowercased()
        return lower.hasPrefix("mac") || lower == "arm64"
    }

    public static func current(useCachedBenchmark: Bool = true) -> DeviceCapabilityProfile {
        DeviceCapabilityProbe.current(useCachedBenchmark: useCachedBenchmark)
    }
}

public struct DeviceBandwidthBenchmarkResult: Codable, Equatable, Sendable {
    public let peakGBs: Double
    public let medianGBs: Double
    public let discountedGBs: Double
    public let deviceName: String
    public let measuredAt: Date

    public init(
        peakGBs: Double,
        medianGBs: Double,
        discountedGBs: Double,
        deviceName: String,
        measuredAt: Date = Date()
    ) {
        self.peakGBs = peakGBs
        self.medianGBs = medianGBs
        self.discountedGBs = discountedGBs
        self.deviceName = deviceName
        self.measuredAt = measuredAt
    }
}

public enum DeviceCapabilityProbe {
    public static let schemaVersion = 1

    private static let cacheKey = "com.atomgradient.edgeengine.device-capability.bandwidth.v1"

    private struct CachedBenchmark: Codable {
        let cacheIdentity: String
        let result: DeviceBandwidthBenchmarkResult
    }

    public static func current(
        useCachedBenchmark: Bool = true,
        userDefaults: UserDefaults = .standard
    ) -> DeviceCapabilityProfile {
        let provisional = makeProfile(
            benchmark: nil,
            bandwidthSource: .none,
            confidence: .provisional
        )
        guard useCachedBenchmark,
              let cached = loadCachedBenchmark(
                  matching: provisional.cacheIdentity,
                  userDefaults: userDefaults
              ) else {
            return provisional
        }
        return makeProfile(
            benchmark: cached,
            bandwidthSource: .cached,
            confidence: .cached
        )
    }

    public static func refreshBenchmark(
        userDefaults: UserDefaults = .standard
    ) async -> DeviceCapabilityProfile {
        let provisional = makeProfile(
            benchmark: nil,
            bandwidthSource: .none,
            confidence: .provisional
        )
        guard let benchmark = await DeviceBandwidthBenchmark.run() else {
            return provisional
        }
        storeCachedBenchmark(
            benchmark,
            cacheIdentity: provisional.cacheIdentity,
            userDefaults: userDefaults
        )
        return makeProfile(
            benchmark: benchmark,
            bandwidthSource: .measured,
            confidence: .measured
        )
    }

    public static func totalPhysicalMemoryMB() -> Int {
        Int(ProcessInfo.processInfo.physicalMemory / 1_048_576)
    }

    public static func availableBeforeJetsamMB() -> Int {
        #if os(iOS) || os(tvOS) || os(watchOS) || os(visionOS)
        return Int(os_proc_available_memory()) / 1_048_576
        #else
        return max(0, totalPhysicalMemoryMB() - physFootprintMB())
        #endif
    }

    public static func physFootprintMB() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size
        )
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return Int(info.phys_footprint) / 1_048_576
    }

    public static func jetsamLimitMB() -> Int {
        physFootprintMB() + availableBeforeJetsamMB()
    }

    public static func sysctlString(_ key: String) -> String? {
        var size: Int = 0
        guard sysctlbyname(key, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [CChar](repeating: 0, count: size + 1)
        guard sysctlbyname(key, &buffer, &size, nil, 0) == 0 else { return nil }
        buffer[size] = 0
        let bytes = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self)
    }

    public static func metalFamilyTier(for device: MTLDevice? = MTLCreateSystemDefaultDevice()) -> Int {
        guard let device else { return 0 }
        let candidates: [(MTLGPUFamily, Int)] = [
            (.apple10, 10),
            (.apple9, 9),
            (.apple8, 8),
            (.apple7, 7),
            (.apple6, 6),
            (.apple5, 5),
            (.apple4, 4),
            (.apple3, 3),
            (.apple2, 2),
            (.apple1, 1),
        ]
        for (family, tier) in candidates where device.supportsFamily(family) {
            return tier
        }
        if device.supportsFamily(.mac2) { return 102 }
        return 0
    }

    public static func recommendedMaxWorkingSetMB(
        for device: MTLDevice? = MTLCreateSystemDefaultDevice()
    ) -> Int? {
        guard let bytes = device?.recommendedMaxWorkingSetSize, bytes > 0 else {
            return nil
        }
        return Int(bytes / 1_048_576)
    }

    private static func makeProfile(
        benchmark: DeviceBandwidthBenchmarkResult?,
        bandwidthSource: DeviceCapabilityProfile.BandwidthSource,
        confidence: DeviceCapabilityProfile.Confidence
    ) -> DeviceCapabilityProfile {
        let device = MTLCreateSystemDefaultDevice()
        return DeviceCapabilityProfile(
            machineIdentifier: sysctlString("hw.machine") ?? "unknown",
            cpuBrandString: sysctlString("machdep.cpu.brand_string"),
            osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
            totalPhysicalMemoryMB: totalPhysicalMemoryMB(),
            availableMemoryMB: availableBeforeJetsamMB(),
            footprintMB: physFootprintMB(),
            jetsamLimitMB: jetsamLimitMB(),
            thermalState: thermalState(),
            metalDeviceName: device?.name,
            metalFamilyTier: metalFamilyTier(for: device),
            recommendedMaxWorkingSetMB: recommendedMaxWorkingSetMB(for: device),
            measuredBandwidthGBs: benchmark?.discountedGBs,
            measuredBandwidthMedianGBs: benchmark?.medianGBs,
            bandwidthSource: bandwidthSource,
            confidence: confidence,
            measuredAt: benchmark?.measuredAt
        )
    }

    private static func loadCachedBenchmark(
        matching cacheIdentity: String,
        userDefaults: UserDefaults
    ) -> DeviceBandwidthBenchmarkResult? {
        guard let data = userDefaults.data(forKey: cacheKey),
              let cached = try? JSONDecoder().decode(CachedBenchmark.self, from: data),
              cached.cacheIdentity == cacheIdentity else {
            return nil
        }
        return cached.result
    }

    private static func storeCachedBenchmark(
        _ result: DeviceBandwidthBenchmarkResult,
        cacheIdentity: String,
        userDefaults: UserDefaults
    ) {
        let cached = CachedBenchmark(cacheIdentity: cacheIdentity, result: result)
        guard let data = try? JSONEncoder().encode(cached) else { return }
        userDefaults.set(data, forKey: cacheKey)
    }

    private static func thermalState() -> DeviceCapabilityProfile.ThermalState {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return .nominal
        case .fair: return .fair
        case .serious: return .serious
        case .critical: return .critical
        @unknown default: return .unknown
        }
    }
}
