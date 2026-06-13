import XCTest
import ManifoldInference
import ManifoldLlamaKit
@_spi(Testing) import ManifoldLlamaKit

/// Unit tests for ``LlamaGenerationDriver/tailRepeats(_:phraseLen:minRepeats:)``.
///
/// The static function detects multi-token repetition loops in the generation
/// window and is used by the generation driver to break out early before
/// `maxTokens` is exhausted. These tests pin the edge-case contract that
/// the integration-level generation tests cannot exercise in isolation.
final class LlamaPhraseRepetitionTests: XCTestCase {

    // MARK: - Returns false when window is too short

    func test_emptyWindow_returnsFalse() {
        XCTAssertFalse(LlamaGenerationDriver.tailRepeats([], phraseLen: 2, minRepeats: 3))
    }

    func test_windowSmallerThanRequiredTokens_returnsFalse() {
        // need phraseLen(2) × minRepeats(3) = 6 tokens; only 5 present
        let window = ["a", "b", "a", "b", "a"]
        XCTAssertFalse(LlamaGenerationDriver.tailRepeats(window, phraseLen: 2, minRepeats: 3))
    }

    func test_exactMinimumWindow_noRepeat_returnsFalse() {
        // 6 tokens but the phrase doesn't repeat
        let window = ["a", "b", "c", "d", "e", "f"]
        XCTAssertFalse(LlamaGenerationDriver.tailRepeats(window, phraseLen: 2, minRepeats: 3))
    }

    // MARK: - Detects exact repetitions

    func test_threeConsecutiveIdenticalPhrases_returnsTrue() {
        // tail = [a, b, a, b, a, b] → phrase [a,b] repeated 3× at the tail
        let window = ["x", "a", "b", "a", "b", "a", "b"]
        XCTAssertTrue(LlamaGenerationDriver.tailRepeats(window, phraseLen: 2, minRepeats: 3))
    }

    func test_singleTokenPhrase_repeatedThreeTimes_returnsTrue() {
        let window = ["z", "q", "q", "q"]
        XCTAssertTrue(LlamaGenerationDriver.tailRepeats(window, phraseLen: 1, minRepeats: 3))
    }

    func test_longerPhrase_repeatedThreeTimes_returnsTrue() {
        // phrase = ["foo", "bar", "baz"], 3 reps = 9 tokens
        let phrase = ["foo", "bar", "baz"]
        let window = phrase + phrase + phrase
        XCTAssertTrue(LlamaGenerationDriver.tailRepeats(window, phraseLen: 3, minRepeats: 3))
    }

    // MARK: - Does not false-positive on partial repeats

    func test_twoRepetitions_belowMinRepeats_returnsFalse() {
        // Only 2 repeats of [a,b] when minRepeats=3
        let window = ["a", "b", "a", "b"]
        XCTAssertFalse(LlamaGenerationDriver.tailRepeats(window, phraseLen: 2, minRepeats: 3))
    }

    func test_phraseInterruptedAtEnd_returnsFalse() {
        // [a, b, a, b, a, X] — last position breaks the repeat
        let window = ["a", "b", "a", "b", "a", "X"]
        XCTAssertFalse(LlamaGenerationDriver.tailRepeats(window, phraseLen: 2, minRepeats: 3))
    }

    func test_repeatInMiddleNotAtTail_returnsFalse() {
        // Repeat is in the middle; tail does not end with a full repeat
        let window = ["a", "b", "a", "b", "a", "b", "c", "d"]
        XCTAssertFalse(LlamaGenerationDriver.tailRepeats(window, phraseLen: 2, minRepeats: 3))
    }

    // MARK: - Edge: phraseLen == 1

    func test_singleTokenPhrase_noRepeat_returnsFalse() {
        let window = ["a", "b", "c"]
        XCTAssertFalse(LlamaGenerationDriver.tailRepeats(window, phraseLen: 1, minRepeats: 3))
    }

    func test_singleTokenPhrase_twoRepeatsOnly_returnsFalse() {
        let window = ["a", "a"]
        XCTAssertFalse(LlamaGenerationDriver.tailRepeats(window, phraseLen: 1, minRepeats: 3))
    }

    // MARK: - Prefix tokens before the repeat are ignored

    func test_repeatAtTailWithArbitraryPrefix_returnsTrue() {
        let prefix = ["x", "y", "z", "p", "q"]
        let window = prefix + ["a", "b", "a", "b", "a", "b"]
        XCTAssertTrue(LlamaGenerationDriver.tailRepeats(window, phraseLen: 2, minRepeats: 3))
    }
}
