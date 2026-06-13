import XCTest
import ManifoldInference
import ManifoldLlamaKit
@_spi(Testing) import ManifoldLlamaKit

final class LlamaDRYSamplerPlumbingTests: XCTestCase {

    func test_descriptorIsNilWhenDRYUnset() {
        let descriptor = LlamaGenerationDriver.DRYSamplerDescriptor(
            config: GenerationConfig(),
            nCtxTrain: 4096
        )

        XCTAssertNil(descriptor)
    }

    func test_descriptorPreservesDRYOptionsAndTrainingContext() throws {
        let options = LlamaDRYSamplerOptions(
            multiplier: 0.8,
            base: 1.9,
            allowedLength: 4,
            penaltyLastN: 512,
            sequenceBreakers: ["\n", "</s>"]
        )
        let descriptor = try XCTUnwrap(LlamaGenerationDriver.DRYSamplerDescriptor(
            config: GenerationConfig(llamaDRY: options),
            nCtxTrain: 8192
        ))

        XCTAssertEqual(descriptor.nCtxTrain, 8192)
        XCTAssertEqual(descriptor.options, options)
    }

    func test_capabilitiesAdvertiseDRYSampler() {
        let backend = LlamaBackend()

        XCTAssertTrue(backend.capabilities.supportedParameters.contains(.llamaDRY))
    }

    // MARK: - Phrase-repetition detection (tailRepeats)
    //
    // ``LlamaGenerationDriver/tailRepeats(_:phraseLen:minRepeats:)`` is the
    // multi-token repetition guard that catches 2–20-token loops the single-token
    // window misses. Tests live here rather than in a standalone file so they
    // inherit this target's already-validated import environment on CI.

    func test_tailRepeats_emptyWindow_returnsFalse() {
        XCTAssertFalse(LlamaGenerationDriver.tailRepeats([], phraseLen: 2, minRepeats: 3))
    }

    func test_tailRepeats_windowShorterThanRequired_returnsFalse() {
        // need phraseLen(2) × minRepeats(3) = 6 tokens; only 5 present
        XCTAssertFalse(LlamaGenerationDriver.tailRepeats(["a","b","a","b","a"], phraseLen: 2, minRepeats: 3))
    }

    func test_tailRepeats_noRepeatInWindow_returnsFalse() {
        XCTAssertFalse(LlamaGenerationDriver.tailRepeats(["a","b","c","d","e","f"], phraseLen: 2, minRepeats: 3))
    }

    func test_tailRepeats_threeConsecutivePhrases_returnsTrue() {
        // tail = [a, b, a, b, a, b]
        XCTAssertTrue(LlamaGenerationDriver.tailRepeats(["x","a","b","a","b","a","b"], phraseLen: 2, minRepeats: 3))
    }

    func test_tailRepeats_singleTokenPhrase_repeatedThreeTimes_returnsTrue() {
        XCTAssertTrue(LlamaGenerationDriver.tailRepeats(["z","q","q","q"], phraseLen: 1, minRepeats: 3))
    }

    func test_tailRepeats_longerPhrase_repeatedThreeTimes_returnsTrue() {
        let phrase = ["foo","bar","baz"]
        XCTAssertTrue(LlamaGenerationDriver.tailRepeats(phrase + phrase + phrase, phraseLen: 3, minRepeats: 3))
    }

    func test_tailRepeats_twoRepetitionsOnly_returnsFalse() {
        XCTAssertFalse(LlamaGenerationDriver.tailRepeats(["a","b","a","b"], phraseLen: 2, minRepeats: 3))
    }

    func test_tailRepeats_phraseInterruptedAtEnd_returnsFalse() {
        // [a, b, a, b, a, X] — last position breaks the repeat
        XCTAssertFalse(LlamaGenerationDriver.tailRepeats(["a","b","a","b","a","X"], phraseLen: 2, minRepeats: 3))
    }

    func test_tailRepeats_repeatInMiddleNotAtTail_returnsFalse() {
        XCTAssertFalse(LlamaGenerationDriver.tailRepeats(["a","b","a","b","a","b","c","d"], phraseLen: 2, minRepeats: 3))
    }

    func test_tailRepeats_repeatAtTailWithArbitraryPrefix_returnsTrue() {
        let prefix = ["x","y","z","p","q"]
        XCTAssertTrue(LlamaGenerationDriver.tailRepeats(prefix + ["a","b","a","b","a","b"], phraseLen: 2, minRepeats: 3))
    }
}

