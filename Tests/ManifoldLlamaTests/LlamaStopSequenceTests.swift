import XCTest
@_spi(Testing) import ManifoldLlama

/// Model-free unit tests for stop-sequence enforcement in the llama.cpp
/// generation driver. The driver previously dropped `GenerationConfig.stopSequences`
/// entirely — generation halted only on an EOG token ID, never on generated text —
/// so non-ChatML models that leak ChatML/other-family control strings as plain text
/// (no matching EOG token) ran on and fabricated fake multi-turn conversations.
///
/// These exercise the pure matcher (`StopSequenceMatcher`) and the resolution
/// helper (`resolveStopSequences`) directly, so they run in CI without a GGUF.
/// The matcher's hold-back / truncation logic is the streaming-correctness core:
/// a stop marker spanning multiple tokens must never be emitted, and output must
/// be truncated exactly at the stop boundary.
final class LlamaStopSequenceTests: XCTestCase {

    private typealias Matcher = LlamaGenerationDriver.StopSequenceMatcher

    /// Drives a matcher with a sequence of chunks and returns the concatenated
    /// emitted text plus whether a stop was hit, mirroring the driver's per-chunk
    /// loop + end-of-stream flush.
    private func run(stops: [String], chunks: [String]) -> (emitted: String, stopped: Bool) {
        var m = Matcher(stops: stops)
        var out = ""
        var stopped = false
        for chunk in chunks {
            let (emit, didStop) = m.push(chunk)
            out += emit
            if didStop { stopped = true; break }
        }
        if !stopped { out += m.flush() }
        return (out, stopped)
    }

    // MARK: - Empty stop set is a no-op

    func test_emptyStops_isPassthrough() {
        var m = Matcher(stops: [])
        XCTAssertTrue(m.isEmpty)
        let (emit, stopped) = m.push("hello <|im_end|> world")
        XCTAssertEqual(emit, "hello <|im_end|> world")
        XCTAssertFalse(stopped)
        XCTAssertEqual(m.flush(), "")
    }

    func test_emptyStrings_filteredOut_treatedAsEmpty() {
        // A stop set of only empty strings must behave like no stops at all.
        let m = Matcher(stops: ["", ""])
        XCTAssertTrue(m.isEmpty)
    }

    // MARK: - Single-chunk matches

    func test_stopInSingleChunk_truncatesAtBoundary_dropsMarkerAndAfter() {
        let (emit, stopped) = run(stops: ["<|im_end|>"], chunks: ["answer here<|im_end|>then garbage"])
        XCTAssertTrue(stopped)
        XCTAssertEqual(emit, "answer here", "Marker and everything after it must be dropped")
    }

    func test_stopAtExactBoundary_endOfChunk() {
        let (emit, stopped) = run(stops: ["</s>"], chunks: ["done</s>"])
        XCTAssertTrue(stopped)
        XCTAssertEqual(emit, "done")
    }

    func test_stopAtVeryStart_emitsNothing() {
        let (emit, stopped) = run(stops: ["[/INST]"], chunks: ["[/INST]rest"])
        XCTAssertTrue(stopped)
        XCTAssertEqual(emit, "")
    }

    func test_noStopPresent_emitsEverything() {
        let (emit, stopped) = run(stops: ["<|im_end|>"], chunks: ["a clean answer with no markers"])
        XCTAssertFalse(stopped)
        XCTAssertEqual(emit, "a clean answer with no markers")
    }

    // MARK: - Multi-token-spanning matches

    func test_stopSpansMultipleChunks() {
        // "<|im_end|>" arrives one character at a time, interleaved with real text.
        let chunks = ["Hi", "<|", "im", "_e", "nd", "|>", " trailing"]
        let (emit, stopped) = run(stops: ["<|im_end|>"], chunks: chunks)
        XCTAssertTrue(stopped)
        XCTAssertEqual(emit, "Hi")
    }

    func test_stopSplitAcrossTwoChunks() {
        let (emit, stopped) = run(stops: ["<end_of_turn>"], chunks: ["reply <end_of", "_turn> ignored"])
        XCTAssertTrue(stopped)
        XCTAssertEqual(emit, "reply ")
    }

    /// The hold-back must prevent a partially-emitted marker from streaming: after
    /// the prefix "<|im" no chunk should have leaked "<|im".
    func test_partialMarkerNeverLeaks_perChunk() {
        var m = Matcher(stops: ["<|im_end|>"])
        let r1 = m.push("text<|im")
        XCTAssertEqual(r1.emit, "text", "Partial marker prefix must be held back, not streamed")
        XCTAssertFalse(r1.stopped)
        let r2 = m.push("_end|>x")
        XCTAssertTrue(r2.stopped)
        XCTAssertEqual(r2.emit, "")
    }

    // MARK: - Partial-then-diverge (false match recovery)

    func test_partialThenDiverge_releasesHeldText() {
        // "<|im" looks like the start of "<|im_end|>" but then diverges to "<|imagine".
        // The held-back prefix must be released as genuine output, no stop.
        let (emit, stopped) = run(stops: ["<|im_end|>"], chunks: ["start <|im", "agine that"])
        XCTAssertFalse(stopped)
        XCTAssertEqual(emit, "start <|imagine that")
    }

    func test_partialPrefixAtStreamEnd_flushedAsOutput() {
        // Stream ends mid-prefix: "<|im_en" is a partial of "<|im_end|>" but no more
        // text arrives, so it is genuine trailing output and must be flushed.
        let (emit, stopped) = run(stops: ["<|im_end|>"], chunks: ["tail <|im_en"])
        XCTAssertFalse(stopped)
        XCTAssertEqual(emit, "tail <|im_en")
    }

    // MARK: - Longest-match-wins among overlapping stops

    func test_longestMatchWins_whenStopsOverlap() {
        // Both "</s>" and the longer "</s></s>" could match; either way text after
        // the earliest stop start is dropped, so output is the same. Assert the
        // boundary is at the first marker start.
        let (emit, stopped) = run(stops: ["</s>", "</s></s>"], chunks: ["bye</s></s>more"])
        XCTAssertTrue(stopped)
        XCTAssertEqual(emit, "bye")
    }

    func test_earliestStopWins_acrossDifferentMarkers() {
        // "<end_of_turn>" appears before "</s>"; the earliest boundary wins.
        let (emit, stopped) = run(
            stops: ["</s>", "<end_of_turn>"],
            chunks: ["first<end_of_turn>second</s>third"])
        XCTAssertTrue(stopped)
        XCTAssertEqual(emit, "first")
    }

    // MARK: - Custom (config) stop sequences

    func test_customStopSequence_isHonored() {
        let (emit, stopped) = run(stops: ["\nUser:"], chunks: ["The answer.\nUser: hi"])
        XCTAssertTrue(stopped)
        XCTAssertEqual(emit, "The answer.")
    }

    func test_latchedStop_emitsNothingAfter() {
        var m = Matcher(stops: ["STOP"])
        let r1 = m.push("aSTOPb")
        XCTAssertTrue(r1.stopped)
        XCTAssertEqual(r1.emit, "a")
        // Once latched, further pushes release nothing.
        let r2 = m.push("more text")
        XCTAssertTrue(r2.stopped)
        XCTAssertEqual(r2.emit, "")
        XCTAssertEqual(m.flush(), "")
    }

    // MARK: - resolveStopSequences

    func test_resolveStopSequences_unionsDefaultsWithConfig() {
        let resolved = LlamaGenerationDriver.resolveStopSequences(["\nUser:"])
        XCTAssertTrue(resolved.contains("\nUser:"))
        for marker in LlamaGenerationDriver.defaultControlMarkerStops {
            XCTAssertTrue(resolved.contains(marker), "Default marker \(marker) must be in the resolved set")
        }
    }

    func test_resolveStopSequences_dedupesAndDropsEmpty() {
        // A config entry duplicating a default, plus an empty string, must collapse.
        let resolved = LlamaGenerationDriver.resolveStopSequences(["</s>", ""])
        XCTAssertEqual(resolved.filter { $0 == "</s>" }.count, 1, "Duplicate must collapse")
        XCTAssertFalse(resolved.contains(""), "Empty strings must be dropped")
    }

    func test_resolveStopSequences_emptyConfig_isJustDefaults() {
        let resolved = LlamaGenerationDriver.resolveStopSequences([])
        XCTAssertEqual(Set(resolved), Set(LlamaGenerationDriver.defaultControlMarkerStops))
    }

    func test_defaultControlMarkers_coverCrossFamily() {
        let set = Set(LlamaGenerationDriver.defaultControlMarkerStops)
        // Spot-check the cross-family coverage the fix exists to provide.
        for required in ["<|im_end|>", "<|im_start|>", "<end_of_turn>",
                         "<|eot_id|>", "<|end_of_text|>", "</s>", "[/INST]", "[tool_call]"] {
            XCTAssertTrue(set.contains(required), "Missing required cross-family stop: \(required)")
        }
    }

    // MARK: - Default markers caught when leaked as plain text

    func test_leakedChatMLMarker_stopsNonChatMLModelOutput() {
        // The motivating bug: a Mistral/Gemma model leaks "<|im_end|>" as text.
        // With defaults unioned in, the matcher must catch it even with no custom stops.
        let resolved = LlamaGenerationDriver.resolveStopSequences([])
        let (emit, stopped) = run(stops: resolved, chunks: [
            "Here is my answer.<|im_end|>\n<|im_start|>user\nfake turn",
        ])
        XCTAssertTrue(stopped)
        XCTAssertEqual(emit, "Here is my answer.")
    }

    // MARK: - Unicode / grapheme safety

    func test_matchingIsGraphemeSafe_aroundMultibyteText() {
        let (emit, stopped) = run(stops: ["</s>"], chunks: ["café ☕ done</s>x"])
        XCTAssertTrue(stopped)
        XCTAssertEqual(emit, "café ☕ done")
    }
}
