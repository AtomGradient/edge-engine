// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeEngine
import XCTest

final class DeviceCapabilityProfileTests: XCTestCase {
    func testCurrentProfileHasStableCacheIdentity() {
        let profile = DeviceCapabilityProfile.current(useCachedBenchmark: false)

        XCTAssertGreaterThan(profile.totalPhysicalMemoryMB, 0)
        XCTAssertGreaterThanOrEqual(profile.availableMemoryMB, 0)
        XCTAssertGreaterThanOrEqual(profile.footprintMB, 0)
        XCTAssertFalse(profile.machineIdentifier.isEmpty)
        XCTAssertFalse(profile.cacheIdentity.isEmpty)
        XCTAssertEqual(profile.bandwidthSource, .none)
        XCTAssertEqual(profile.confidence, .provisional)
    }

    func testBandwidthBenchmarkResultCodableRoundTrip() throws {
        let result = DeviceBandwidthBenchmarkResult(
            peakGBs: 123.0,
            medianGBs: 100.0,
            discountedGBs: 95.0,
            deviceName: "unit-test",
            measuredAt: Date(timeIntervalSince1970: 1_234)
        )

        let data = try JSONEncoder().encode(result)
        let decoded = try JSONDecoder().decode(DeviceBandwidthBenchmarkResult.self, from: data)

        XCTAssertEqual(decoded, result)
    }

    func testDeviceClassConveniencePropertiesDeriveFromMachineIdentifier() {
        let phone = makeProfile(machineIdentifier: "iPhone18,5")
        XCTAssertTrue(phone.isPhone)
        XCTAssertFalse(phone.isMac)

        let mac = makeProfile(machineIdentifier: "Mac16,1")
        XCTAssertFalse(mac.isPhone)
        XCTAssertTrue(mac.isMac)

        let genericAppleSiliconMac = makeProfile(machineIdentifier: "arm64")
        XCTAssertFalse(genericAppleSiliconMac.isPhone)
        XCTAssertTrue(genericAppleSiliconMac.isMac)
    }

    private func makeProfile(machineIdentifier: String) -> DeviceCapabilityProfile {
        DeviceCapabilityProfile(
            machineIdentifier: machineIdentifier,
            cpuBrandString: nil,
            osVersion: "test-os",
            totalPhysicalMemoryMB: 8_192,
            availableMemoryMB: 4_096,
            footprintMB: 512,
            jetsamLimitMB: 6_144,
            thermalState: .nominal,
            metalDeviceName: "test-metal",
            metalFamilyTier: 10,
            recommendedMaxWorkingSetMB: nil,
            measuredBandwidthGBs: nil,
            measuredBandwidthMedianGBs: nil,
            bandwidthSource: .none,
            confidence: .provisional,
            measuredAt: nil
        )
    }
}
