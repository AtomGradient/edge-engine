// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Testing
@testable import EdgeEngine

@Test func qwenGDNReferenceStepUpdatesStateWithDecayAndOuterProduct() throws {
    let step = try QwenGDNReference.step(
        state: [0, 0, 0, 0],
        query: [1, 0],
        key: [2, 3],
        value: [4, 5],
        decay: [0.5],
        headCount: 1,
        headDimension: 2
    )

    #expect(step.updatedState == [
        8, 10,
        12, 15,
    ])
    #expect(step.output == [8, 10])
}

@Test func qwenGDNReferenceStepCarriesRecurrentStateAcrossTokens() throws {
    let first = try QwenGDNReference.step(
        state: [0, 0, 0, 0],
        query: [1, 0],
        key: [2, 3],
        value: [4, 5],
        decay: [0.5],
        headCount: 1,
        headDimension: 2
    )
    let second = try QwenGDNReference.step(
        state: first.updatedState,
        query: [0, 1],
        key: [2, 3],
        value: [4, 5],
        decay: [0.5],
        headCount: 1,
        headDimension: 2
    )

    #expect(second.updatedState == [
        12, 15,
        18, 22.5,
    ])
    #expect(second.output == [18, 22.5])
}

@Test func qwenGatedDeltaReferenceComputesDecay() throws {
    let decay = try QwenGDNReference.computeDecay(
        aLog: [0],
        a: [0],
        dtBias: [0],
        batchSize: 1,
        tokenCount: 1,
        valueHeadCount: 1
    )

    #expect(abs(decay[0] - 0.5) < 1e-6)
}

@Test func qwenGatedDeltaReferenceUpdatesSingleTokenState() throws {
    let update = try QwenGDNReference.gatedDeltaUpdate(
        query: [1, 0],
        key: [2, 3],
        value: [4, 5],
        a: [0],
        b: [0],
        aLog: [0],
        dtBias: [0],
        batchSize: 1,
        tokenCount: 1,
        keyHeadCount: 1,
        valueHeadCount: 1,
        keyHeadDimension: 2,
        valueHeadDimension: 2
    )

    #expect(try NumericComparison.maxAbsoluteError(update.output, [4, 5]) < 1e-6)
    #expect(try NumericComparison.maxAbsoluteError(update.updatedState, [4, 6, 5, 7.5]) < 1e-6)
}

@Test func qwenGatedDeltaReferenceCarriesStateAcrossTokens() throws {
    let update = try QwenGDNReference.gatedDeltaUpdate(
        query: [1, 0, 1, 0],
        key: [2, 3, 2, 3],
        value: [4, 5, 4, 5],
        a: [0, 0],
        b: [0, 0],
        aLog: [0],
        dtBias: [0],
        batchSize: 1,
        tokenCount: 2,
        keyHeadCount: 1,
        valueHeadCount: 1,
        keyHeadDimension: 2,
        valueHeadDimension: 2
    )

    #expect(try NumericComparison.maxAbsoluteError(update.output, [4, 5, -7, -8.75]) < 1e-6)
    #expect(try NumericComparison.maxAbsoluteError(update.updatedState, [-7, -10.5, -8.75, -13.125]) < 1e-6)
}

@Test func qwenGatedDeltaReferenceAcceptsInitialStateForIncrementalDecode() throws {
    let first = try QwenGDNReference.gatedDeltaUpdate(
        query: [1, 0],
        key: [2, 3],
        value: [4, 5],
        a: [0],
        b: [0],
        aLog: [0],
        dtBias: [0],
        batchSize: 1,
        tokenCount: 1,
        keyHeadCount: 1,
        valueHeadCount: 1,
        keyHeadDimension: 2,
        valueHeadDimension: 2
    )
    let second = try QwenGDNReference.gatedDeltaUpdate(
        query: [1, 0],
        key: [2, 3],
        value: [4, 5],
        a: [0],
        b: [0],
        aLog: [0],
        dtBias: [0],
        initialState: first.updatedState,
        batchSize: 1,
        tokenCount: 1,
        keyHeadCount: 1,
        valueHeadCount: 1,
        keyHeadDimension: 2,
        valueHeadDimension: 2
    )

    #expect(try NumericComparison.maxAbsoluteError(second.output, [-7, -8.75]) < 1e-6)
    #expect(try NumericComparison.maxAbsoluteError(second.updatedState, [-7, -10.5, -8.75, -13.125]) < 1e-6)
}

@Test func qwenGatedDeltaReferenceRepeatsKeyHeadsAcrossValueHeads() throws {
    let update = try QwenGDNReference.gatedDeltaUpdate(
        query: [1, 0],
        key: [2, 3],
        value: [4, 5],
        a: [0, 0],
        b: [0, 0],
        aLog: [0, 0],
        dtBias: [0, 0],
        batchSize: 1,
        tokenCount: 1,
        keyHeadCount: 1,
        valueHeadCount: 2,
        keyHeadDimension: 2,
        valueHeadDimension: 1
    )

    #expect(try NumericComparison.maxAbsoluteError(update.output, [4, 5]) < 1e-6)
    #expect(try NumericComparison.maxAbsoluteError(update.updatedState, [4, 6, 5, 7.5]) < 1e-6)
}
