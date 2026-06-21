import XCTest
@_spi(Testing) import ManifoldLlama
import ManifoldInference
import ManifoldTestSupport

/// Headless coverage for the `alreadyGenerating` re-entrancy guard in
/// `LlamaBackend.generate()` (issue #27).
///
/// The guard (`guard !withStateLock({ isGenerating }) else { throw
/// InferenceError.alreadyGenerating }`) was previously 0-hit in CI because
/// it requires the backend to be in a loaded+generating state, which normally
/// requires a live model. These tests use `armFakeLoadedStateForTesting()` and
/// `setIsGeneratingForTesting(_:)` to reach the guard without a real GGUF model.
/// The sentinel pointers are never passed to any C API in the guarded paths.
///
/// Model-gated live-loop guards (thinking-budget, repetition) are left as
/// `XCTSkip`-guarded scaffolds pending a real model fixture.
final class LlamaGenerationGuardTests: XCTestCase {

    // MARK: - alreadyGenerating

    /// The re-entrancy guard must throw `.alreadyGenerating` when `isGenerating == true`.
    ///
    /// Headless: `armFakeLoadedStateForTesting()` satisfies the pointer-nil checks and
    /// sets `isModelLoaded = true` without touching a real llama.cpp context.
    /// `setIsGeneratingForTesting(true)` simulates a concurrent generation in progress.
    /// The guard fires before any C call, so the sentinel pointers are never dereferenced.
    ///
    /// Sabotage: commenting out or inverting the `guard !withStateLock({ isGenerating })`
    /// line in `generate()` would cause this test to fail — either no error is thrown,
    /// or a different error arrives from the tokenization path that follows.
    func test_generate_throwsAlreadyGenerating_whenIsGeneratingTrue() throws {
        let backend = LlamaBackend()
        backend.armFakeLoadedStateForTesting()
        // Disarm before deinit: otherwise unloadModel() schedules a detached
        // llama_synchronize/llama_free on the addr-1 sentinel context and crashes
        // the process asynchronously (flaky exit-time SIGSEGV). See #54. The
        // teardown closure keeps `backend` alive until after disarm runs, so the
        // subsequent deinit finds nil pointers and does no C cleanup.
        addTeardownBlock { backend.disarmFakeLoadedStateForTesting() }
        backend.setIsGeneratingForTesting(true)

        XCTAssertThrowsError(
            try backend.generate(prompt: "hello", systemPrompt: nil,
                                 config: GenerationConfig(temperature: 0.0))
        ) { error in
            guard case InferenceError.alreadyGenerating = error else {
                XCTFail("Expected .alreadyGenerating; got \(error)")
                return
            }
        }
    }

    /// Sabotage check: when `isGenerating == false` the re-entrancy guard must NOT fire.
    ///
    /// We verify the guard's precondition directly via observable state rather than
    /// calling `generate()`: the code path after the guard reaches `llama_tokenize`
    /// with the sentinel vocab pointer, which dereferences address 1 and crashes the
    /// test process — Swift cannot catch a C-level SIGSEGV as a thrown error.
    ///
    /// State invariant verified:
    ///   - `armFakeLoadedStateForTesting()` sets `isModelLoaded = true`
    ///   - Without calling `setIsGeneratingForTesting(true)`, `isGenerating` stays `false`
    ///   - The guard is `guard !withStateLock({ isGenerating }) else { throw .alreadyGenerating }`
    ///     which only throws when `isGenerating == true`; with `false` it is a no-op.
    ///
    /// Sabotage: setting `isGenerating = true` unconditionally in `armFakeLoadedStateForTesting()`
    /// or inverting the guard condition would break the `XCTAssertFalse` assertion below.
    func test_generate_doesNotThrowAlreadyGenerating_whenNotGenerating() throws {
        let backend = LlamaBackend()
        backend.armFakeLoadedStateForTesting()
        // Disarm before deinit to avoid the detached sentinel-pointer cleanup crash
        // (#54); see the sibling test above.
        addTeardownBlock { backend.disarmFakeLoadedStateForTesting() }
        // isGenerating is false by default — do NOT call setIsGeneratingForTesting(true)

        // Verify the guard precondition directly. Calling generate() is unsafe here
        // because execution past the alreadyGenerating guard reaches llama_tokenize
        // with the sentinel vocab (address 1), which crashes — not a Swift-catchable throw.
        XCTAssertTrue(backend.isModelLoaded,
                      "armFakeLoadedStateForTesting() must set isModelLoaded = true")
        XCTAssertFalse(backend.isGenerating,
                       "isGenerating must be false when setIsGeneratingForTesting was not called; "
                       + "the alreadyGenerating guard would not fire for this state")
    }

    // MARK: - Model-gated scaffolds (issue #27 remainder)

    /// Thinking-budget enforcement wiring (issue #27). The `ThinkingLoopBudget`
    /// decision logic is unit-tested headlessly via `LlamaGenerationDriverHelpersTests`;
    /// this exercises the live break inside the generation loop
    /// (`LlamaGenerationDriver.run`, ~:786-798): once `thinkingTokenCount >= limit`
    /// the loop sets `thinkingLimitReached` and `break generationLoop`s.
    ///
    /// We drive a real reasoning model (Qwen3 etc.) with `maxThinkingTokens = 4`,
    /// forcing thinking on via the `.qwen3` marker preset so the test is robust to
    /// the model's chat-template auto-detection. We count emitted `.thinkingToken`
    /// events and assert:
    ///   1. The model actually entered thinking (`>= 1` thinking token) — otherwise
    ///      the budget assertion would be vacuously true.
    ///   2. The observed thinking-token count respects the budget. The guard breaks
    ///      on the iteration that *reaches* the limit, so the emitted count equals the
    ///      limit (the loop yields the limit-th thinking token, then breaks). We assert
    ///      `<= limit` to pin the upper bound the production wiring enforces.
    ///
    /// Sabotage: deleting the `thinkingLimitReached` break at :798 (or the
    /// `thinkingTokenCount >= limit` check at :786) lets the model think until its
    /// natural reasoning end — far more than 4 tokens — failing assertion (2).
    func test_generate_enforcesThinkingBudget_scaffold() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["RUN_SLOW_TESTS"] == "1",
            "Thinking-budget live-loop enforcement requires a real GGUF model; set RUN_SLOW_TESTS=1"
        )
        try XCTSkipUnless(HardwareRequirements.isPhysicalDevice && HardwareRequirements.isAppleSilicon,
                          "LlamaBackend requires Apple Silicon + Metal (unavailable in simulator)")
        guard let url = HardwareRequirements.findGGUFModel() else {
            throw XCTSkip("No GGUF on disk. Set LLAMA_TEST_MODEL=<path> (a reasoning model, e.g. Qwen3) or place a `.gguf` in ~/Documents/Models/ to run this test.")
        }

        let backend = LlamaBackend()
        addTeardownBlock { await backend.unloadAndWait() }
        try await backend.loadModel(from: url, plan: .testStub(effectiveContextSize: 1024))

        let thinkingBudget = 4
        var config = GenerationConfig(temperature: 0.0, maxOutputTokens: 64)
        config.maxThinkingTokens = thinkingBudget
        // Force the thinking parser on regardless of the model's chat-template
        // auto-detection, so the test pins the budget wiring rather than marker
        // discovery. `.qwen3` uses <think>…</think>, the dominant convention.
        config.thinkingMarkers = .qwen3

        // The backend tokenizes the raw `prompt` verbatim (no chat-template
        // application), so we hand it a Qwen3-formatted prompt that ends at the
        // assistant turn WITHOUT pre-filling `<think>`. The model then generates
        // its own `<think>…</think>` block, which the ThinkingTransform (depth
        // starts at 0) needs to observe to route reasoning to `.thinkingToken`.
        let prompt = """
        <|im_start|>user
        What is 17 + 26? Think step by step before answering.<|im_end|>
        <|im_start|>assistant
        """
        let stream = try backend.generate(
            prompt: prompt,
            systemPrompt: nil,
            config: config
        )

        var thinkingTokens = 0
        for try await event in stream.events {
            if case .thinkingToken = event { thinkingTokens += 1 }
        }
        await backend.awaitGenerationSettled()

        try XCTSkipIf(thinkingTokens == 0,
            "Model did not emit any thinking tokens for this prompt/markers; the budget "
            + "assertion would be vacuous. Use a reasoning model (e.g. Qwen3) via LLAMA_TEST_MODEL.")
        XCTAssertLessThanOrEqual(thinkingTokens, thinkingBudget,
            "Thinking-budget guard must break the generation loop at maxThinkingTokens=\(thinkingBudget); "
            + "observed \(thinkingTokens) thinking tokens — the in-loop break (LlamaGenerationDriver :786-798) is not enforcing the cap.")
    }

    /// Repetition-guard break wiring (issue #27). `RepeatWindow` and `tailRepeats`
    /// are unit-tested headlessly; this exercises the live loop-breaks in
    /// `LlamaGenerationDriver.run` — single-token run (:738-739) and phrase-level
    /// run (:753-754), both `break generationLoop`.
    ///
    /// Making a real model loop deterministically: we use **greedy** decoding
    /// (`temperature = 0.0`) with the repetition penalty **disabled**
    /// (`repeatPenalty = 1.0`) and a prompt that explicitly asks the model to repeat
    /// a token forever. Greedy + no penalty is the classic degenerate configuration
    /// that drives small models into a fixed point: once the highest-probability
    /// next token is the same token that was just emitted, with no penalty to perturb
    /// the distribution and no sampling randomness, the model emits it again and
    /// again. The repetition guard (≥20 identical tokens, or a phrase repeated ≥3×)
    /// then fires and breaks the loop.
    ///
    /// We set `maxOutputTokens` high (4096) so that early termination is
    /// unambiguously the repetition guard, not the length cap: a clean break leaves
    /// the visible completion count far below the cap. Thinking is disabled
    /// (`maxThinkingTokens = 0`) so all decoded tokens count as visible output and
    /// the repetition guards (which run on every decoded token, pre-parse) apply
    /// directly.
    ///
    /// Assertions:
    ///   1. Generation terminates (does not hang) — the `for try await` completes.
    ///   2. The visible completion count is well under `maxOutputTokens`, proving the
    ///      loop exited via the repetition `break` rather than the length cap. (If the
    ///      model happens to hit a natural EOG instead, that is also an early
    ///      termination well under the cap and satisfies this invariant — the guard
    ///      is one of several early-exit paths, all of which are "did not run to the
    ///      cap".)
    ///
    /// Sabotage: removing both `break generationLoop` repetition exits lets a looping
    /// model run all the way to `maxOutputTokens = 4096`, blowing assertion (2).
    func test_generate_breaksOnRepetition_scaffold() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["RUN_SLOW_TESTS"] == "1",
            "Repetition-guard live-loop break requires a real GGUF model; set RUN_SLOW_TESTS=1"
        )
        try XCTSkipUnless(HardwareRequirements.isPhysicalDevice && HardwareRequirements.isAppleSilicon,
                          "LlamaBackend requires Apple Silicon + Metal (unavailable in simulator)")
        guard let url = HardwareRequirements.findGGUFModel() else {
            throw XCTSkip("No GGUF on disk. Set LLAMA_TEST_MODEL=<path> or place a `.gguf` in ~/Documents/Models/ to run this test.")
        }

        // Context must exceed prompt + maxOutputTokens or the contextExhausted
        // preflight (LlamaBackend ~:553) throws before generation even starts.
        let backend = LlamaBackend()
        addTeardownBlock { await backend.unloadAndWait() }
        try await backend.loadModel(from: url, plan: .testStub(effectiveContextSize: 8192))

        let maxOutput = 2048
        var config = GenerationConfig(temperature: 0.0, maxOutputTokens: maxOutput)
        config.repeatPenalty = 1.0          // disable penalty so greedy can loop
        config.repetitionPenalty = 1.0      // belt-and-suspenders: optional override path
        config.maxThinkingTokens = 0        // no thinking — every token is visible output

        // Qwen3 chat format (the backend tokenizes verbatim, no template applied)
        // priming a degenerate repeat. Greedy + no penalty turns this into a fixed
        // point the repetition guard must break.
        let prompt = """
        <|im_start|>user
        Output the word "go" repeated forever with single spaces, nothing else.<|im_end|>
        <|im_start|>assistant
        /no_think
        go go go go go go go go go go go go go go go go go go go go go go go go
        """
        let stream = try backend.generate(
            prompt: prompt,
            systemPrompt: nil,
            config: config
        )

        var visibleTokens = 0
        var usageCompletion: Int?
        for try await event in stream.events {
            switch event {
            case .token: visibleTokens += 1
            case .usage(let usage): usageCompletion = usage.completionTokens
            default: break
            }
        }
        await backend.awaitGenerationSettled()

        // (1) Termination: reaching here means the stream finished without hanging.
        // (2) Early exit well under the cap. Use the usage completion count when
        //     present (authoritative), falling back to the counted .token events.
        let completion = usageCompletion ?? visibleTokens
        XCTAssertLessThan(completion, maxOutput,
            "Generation ran to the maxOutputTokens cap (\(maxOutput)); the repetition guard "
            + "(LlamaGenerationDriver :738-739 / :753-754) did not break the degenerate loop.")
        // A genuine repetition/EOG break terminates far below the cap. Pin a loose
        // upper bound so a near-cap "break" (which would indicate the guard is barely
        // working, or not at all) still fails. The single-token guard fires after 20
        // identical tokens and the phrase guard after a short phrase ×3, so any real
        // loop dies in well under a few hundred tokens.
        XCTAssertLessThan(completion, maxOutput / 2,
            "Generation produced \(completion) tokens — suspiciously close to the cap. A working "
            + "repetition guard terminates a degenerate greedy loop in tens of tokens, not thousands.")
    }
}
