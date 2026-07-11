import XCTest
import ManifoldInference
import ManifoldLlama
@_spi(Testing) import ManifoldLlama

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

    // MARK: - Phrase guard grammar exemption (issue #141)
    //
    // ``LlamaGenerationDriver/phraseGuardShouldTrip(hasGrammar:window:maxPhraseLen:minRepeats:)``
    // is the extracted decision the generation loop calls before `break
    // generationLoop`-ing on a repeated phrase. These tests drive it directly
    // with synthetic decoded-token strings — no GGUF model required — proving
    // both halves of the issue's acceptance criteria:
    //   1. a grammar-active run whose tokens repeat a phrase >=3x must NOT trip
    //      the guard (repeated JSON blocks are legitimate under GBNF);
    //   2. a non-grammar run with the exact same repeated phrase still trips
    //      it (smollm2-class runaway protection is unchanged).
    //
    // Fully deterministic: no async, no model, no randomness.

    func test_phraseGuardShouldTrip_grammarActive_repeatedPhrase_doesNotTrip() {
        // Same repeated-phrase shape as test_tailRepeats_threeConsecutivePhrases_returnsTrue
        // (tail = [a, b, a, b, a, b]) — under grammar this must be exempted.
        let window = ["x", "a", "b", "a", "b", "a", "b"]

        XCTAssertFalse(
            LlamaGenerationDriver.phraseGuardShouldTrip(
                hasGrammar: true,
                window: window,
                maxPhraseLen: 20,
                minRepeats: 3
            ),
            "Grammar-constrained runs must be exempt from the phrase-repetition guard (#141): "
            + "a legitimately repeated structured-output phrase must not early-exit the loop."
        )
    }

    func test_phraseGuardShouldTrip_noGrammar_repeatedPhrase_stillTrips() {
        // Identical window to the grammar-active case above — only `hasGrammar`
        // differs. Non-grammar behavior must be unchanged from pre-#141.
        let window = ["x", "a", "b", "a", "b", "a", "b"]

        XCTAssertTrue(
            LlamaGenerationDriver.phraseGuardShouldTrip(
                hasGrammar: false,
                window: window,
                maxPhraseLen: 20,
                minRepeats: 3
            ),
            "Non-grammar runs must retain the original phrase-repetition early exit "
            + "(smollm2-class runaway protection) — the grammar exemption must not "
            + "weaken this path."
        )
    }

    func test_phraseGuardShouldTrip_grammarActive_noRepeat_doesNotTrip() {
        // Sabotage-adjacent: without a repeat at all, grammar-active must also
        // stay false — confirms the exemption isn't vacuously true because the
        // window itself never trips regardless of the flag.
        let window = ["a", "b", "c", "d", "e", "f"]

        XCTAssertFalse(
            LlamaGenerationDriver.phraseGuardShouldTrip(
                hasGrammar: true,
                window: window,
                maxPhraseLen: 20,
                minRepeats: 3
            )
        )
    }

    func test_phraseGuardShouldTrip_noGrammar_noRepeat_doesNotTrip() {
        let window = ["a", "b", "c", "d", "e", "f"]

        XCTAssertFalse(
            LlamaGenerationDriver.phraseGuardShouldTrip(
                hasGrammar: false,
                window: window,
                maxPhraseLen: 20,
                minRepeats: 3
            )
        )
    }
}

