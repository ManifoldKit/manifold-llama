import XCTest
@_spi(Testing) import ManifoldLlama

/// Model-free unit tests for the pure decision helpers extracted out of
/// `LlamaGenerationDriver.run()`. These paths were previously reachable only
/// through a live decode loop (requiring a loaded GGUF), so a regression in the
/// arithmetic or the C-string lifetime handling could not be caught in CI.
final class LlamaGenerationDriverHelpersTests: XCTestCase {

    // MARK: - withCStringArray (DRY sampler sequence-breaker bridge)

    /// The body must observe one valid, dereferenceable pointer per input string,
    /// in input order. Copying the C strings back inside the body proves the
    /// pointers are alive for the whole call (the property the nested
    /// `withCString` / `defer` unwind exists to guarantee).
    func test_withCStringArray_preservesValuesAndOrder() {
        let input = ["alpha", "b", "two words", "🌟unicode"]
        let copied = LlamaGenerationDriver.withCStringArray(input) { pointers -> [String] in
            pointers.map { $0.map { String(cString: $0) } ?? "<nil>" }
        }
        XCTAssertEqual(copied, input,
            "Each input string must round-trip through its C pointer, in order")
    }

    func test_withCStringArray_singleElement() {
        let copied = LlamaGenerationDriver.withCStringArray(["only"]) { ptrs in
            ptrs.map { String(cString: $0!) }
        }
        XCTAssertEqual(copied, ["only"])
    }

    /// Empty input must call `body` with an empty buffer (not skip the call) so
    /// the DRY sampler still receives a valid zero-length breaker array.
    func test_withCStringArray_emptyInput_invokesBodyWithEmptyBuffer() {
        var bodyCalled = false
        let count = LlamaGenerationDriver.withCStringArray([]) { ptrs -> Int in
            bodyCalled = true
            return ptrs.count
        }
        XCTAssertTrue(bodyCalled, "body must be invoked even for empty input")
        XCTAssertEqual(count, 0)
    }

    func test_withCStringArray_pointersAreDistinct() {
        LlamaGenerationDriver.withCStringArray(["x", "y", "z"]) { ptrs in
            let raw = ptrs.compactMap { $0 }
            XCTAssertEqual(raw.count, 3)
            XCTAssertEqual(Set(raw.map { UInt(bitPattern: $0) }).count, 3,
                "Each string must get its own distinct buffer (no pointer reuse)")
        }
    }

    // MARK: - alignedKVReuseLength

    /// The reuse floor must be a multiple of batchSize and never reach the final
    /// chunk's start — otherwise the decode loop would never run and there would
    /// be no logits to sample.
    func test_alignedKVReuse_capsBelowFinalChunk_whenFullPromptMatches() {
        // 64 tokens, batch 32: finalChunkStart = ((64-1)/32)*32 = 32.
        // reuseLen 64 would floor to 64, but must be capped at 32.
        let aligned = LlamaGenerationDriver.alignedKVReuseLength(
            tokenCount: 64, reuseLen: 64, batchSize: 32)
        XCTAssertEqual(aligned, 32)
    }

    func test_alignedKVReuse_floorsToBatchMultiple() {
        // reuseLen 50 floors to 32 (50/32*32); finalChunkStart for 100 tokens
        // = ((99)/32)*32 = 96, so the floor wins.
        let aligned = LlamaGenerationDriver.alignedKVReuseLength(
            tokenCount: 100, reuseLen: 50, batchSize: 32)
        XCTAssertEqual(aligned, 32)
    }

    func test_alignedKVReuse_zeroReuse_isZero() {
        XCTAssertEqual(
            LlamaGenerationDriver.alignedKVReuseLength(tokenCount: 100, reuseLen: 0, batchSize: 32),
            0)
    }

    func test_alignedKVReuse_emptyPrompt_isZero() {
        XCTAssertEqual(
            LlamaGenerationDriver.alignedKVReuseLength(tokenCount: 0, reuseLen: 0, batchSize: 32),
            0)
    }

    func test_alignedKVReuse_batchSizeOne_isClampedNotDivideByZeroOrPastFinal() {
        // batchSize 1: finalChunkStart = tokenCount-1, so reuse caps one below the end.
        XCTAssertEqual(
            LlamaGenerationDriver.alignedKVReuseLength(tokenCount: 10, reuseLen: 10, batchSize: 1),
            9)
        // Defensive: a zero batchSize must not trap (treated as 1).
        XCTAssertEqual(
            LlamaGenerationDriver.alignedKVReuseLength(tokenCount: 10, reuseLen: 10, batchSize: 0),
            9)
    }

    // MARK: - thinkingLoopBudget

    func test_thinkingBudget_parserOff_isZeroThinking() {
        let b = LlamaGenerationDriver.thinkingLoopBudget(
            contextCapacity: 4096, usedSlots: 10, useParser: false,
            maxThinkingTokens: 999, maxTokens: 256)
        XCTAssertEqual(b.effectiveThinkingBudget, 0)
        XCTAssertEqual(b.totalLoopBudget, 256)
    }

    func test_thinkingBudget_nilThinking_defaultsToMaxTokens() {
        // remaining = 4096-10 = 4086; cap = max(0, 4086-256)=3830; raw = maxTokens=256.
        let b = LlamaGenerationDriver.thinkingLoopBudget(
            contextCapacity: 4096, usedSlots: 10, useParser: true,
            maxThinkingTokens: nil, maxTokens: 256)
        XCTAssertEqual(b.effectiveThinkingBudget, 256)
        XCTAssertEqual(b.totalLoopBudget, 512)
    }

    /// The thinking budget must shrink to fit the remaining context window so the
    /// loop never decodes past it (the whole reason for the clamp).
    func test_thinkingBudget_clampedToRemainingContext() {
        // context 512, used 400 -> remaining 112; cap = max(0, 112-64)=48,
        // raw thinking = 1000 -> effective = 48; total = 64+48 = 112.
        let b = LlamaGenerationDriver.thinkingLoopBudget(
            contextCapacity: 512, usedSlots: 400, useParser: true,
            maxThinkingTokens: 1000, maxTokens: 64)
        XCTAssertEqual(b.effectiveThinkingBudget, 48)
        XCTAssertEqual(b.totalLoopBudget, 112)
    }

    func test_thinkingBudget_promptLargerThanContext_clampsToZero() {
        // remaining 0 -> cap 0 -> effective 0; total = maxTokens only.
        let b = LlamaGenerationDriver.thinkingLoopBudget(
            contextCapacity: 256, usedSlots: 300, useParser: true,
            maxThinkingTokens: 500, maxTokens: 64)
        XCTAssertEqual(b.effectiveThinkingBudget, 0)
        XCTAssertEqual(b.totalLoopBudget, 64)
    }

    // MARK: - RepeatWindow

    func test_repeatWindow_breaksExactlyAtLimit() {
        var w = LlamaGenerationDriver.RepeatWindow(limit: 3)
        XCTAssertFalse(w.observe("a"))  // count 1
        XCTAssertFalse(w.observe("a"))  // count 2
        XCTAssertTrue(w.observe("a"))   // count 3 -> break
    }

    func test_repeatWindow_resetsOnDifferentToken() {
        var w = LlamaGenerationDriver.RepeatWindow(limit: 3)
        XCTAssertFalse(w.observe("a"))
        XCTAssertFalse(w.observe("a"))
        XCTAssertFalse(w.observe("b"))  // reset
        XCTAssertFalse(w.observe("b"))
        XCTAssertFalse(w.observe("a"))  // reset again
        XCTAssertFalse(w.observe("a"))
        XCTAssertTrue(w.observe("a"))   // now hits limit 3 for "a"
    }

    func test_repeatWindow_neverBreaksBelowLimit() {
        var w = LlamaGenerationDriver.RepeatWindow(limit: 20)
        for _ in 0..<19 {
            XCTAssertFalse(w.observe("loop"))
        }
        XCTAssertTrue(w.observe("loop"))
    }
}
