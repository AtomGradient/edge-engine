// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Testing
@testable import EdgeEngine

@Test func qwenStopTokenMatcherMatchesSingleTokenStops() throws {
    var matcher = try QwenStopTokenMatcher(stopSequences: [[7]])

    #expect(matcher.append(tokenId: 1) == nil)
    #expect(matcher.bufferedTokenIds == [1])
    #expect(matcher.append(tokenId: 7) == QwenStopTokenMatch(tokenIds: [7]))
    #expect(matcher.currentMatch() == QwenStopTokenMatch(tokenIds: [7]))
    #expect(matcher.bufferedTokenIds == [7])
}

@Test func qwenStopTokenMatcherPrefersLongestSuffixMatch() throws {
    var matcher = try QwenStopTokenMatcher(
        stopSequences: [
            [3],
            [2, 3],
            [1, 2, 3],
        ]
    )

    #expect(matcher.append(tokenId: 1) == nil)
    #expect(matcher.append(tokenId: 2) == nil)
    #expect(matcher.append(tokenId: 3) == QwenStopTokenMatch(tokenIds: [1, 2, 3]))
    #expect(matcher.currentMatch() == QwenStopTokenMatch(tokenIds: [1, 2, 3]))
}

@Test func qwenStopTokenMatcherKeepsOnlyNeededSuffixWindow() throws {
    var matcher = try QwenStopTokenMatcher(stopSequences: [[4, 5, 6]])

    #expect(matcher.append(contentsOf: [1, 2, 4, 5]) == nil)
    #expect(matcher.bufferedTokenIds == [2, 4, 5])
    #expect(matcher.append(tokenId: 6) == QwenStopTokenMatch(tokenIds: [4, 5, 6]))
    #expect(matcher.bufferedTokenIds == [4, 5, 6])
}

@Test func qwenStopTokenMatcherStopsAppendContentsAtFirstMatch() throws {
    var matcher = try QwenStopTokenMatcher(stopSequences: [[2, 3]])

    let match = matcher.append(contentsOf: [1, 2, 3, 4])

    #expect(match == QwenStopTokenMatch(tokenIds: [2, 3]))
    #expect(matcher.bufferedTokenIds == [2, 3])
}

@Test func qwenStopTokenMatcherResetClearsBufferedTokens() throws {
    var matcher = try QwenStopTokenMatcher(stopSequences: [[1, 2]])

    #expect(matcher.append(tokenId: 1) == nil)
    matcher.reset()

    #expect(matcher.bufferedTokenIds.isEmpty)
    #expect(matcher.append(tokenId: 2) == nil)
}

@Test func qwenStopTokenMatcherRejectsEmptyStopSequences() throws {
    var rejectedEmpty = false

    do {
        _ = try QwenStopTokenMatcher(stopSequences: [[1], []])
    } catch QwenStopTokenMatcherError.emptyStopSequence(index: 1) {
        rejectedEmpty = true
    }

    #expect(rejectedEmpty)
}
