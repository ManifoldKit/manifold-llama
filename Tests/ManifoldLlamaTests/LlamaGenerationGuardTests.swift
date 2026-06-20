import XCTest
@_spi(Testing) import ManifoldLlama
import ManifoldInference

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
    func test_generate_throwsAlreadyGenerating_whenIsGeneratingTrue() throws {
        let backend = LlamaBackend()
        backend.armFakeLoadedStateForTesting()
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
    /// `generate()` will proceed past the guard and fail later (at tokenization against a
    /// sentinel vocab), but the error must not be `.alreadyGenerating`. A regression that
    /// inverts or removes the guard would cause this test to fail by either:
    ///   - throwing `.alreadyGenerating` when it should not, or
    ///   - not throwing at all (if the guard is removed and tokenization somehow succeeds).
    func test_generate_doesNotThrowAlreadyGenerating_whenNotGenerating() throws {
        let backend = LlamaBackend()
        backend.armFakeLoadedStateForTesting()
        // isGenerating is false by default — do NOT call setIsGeneratingForTesting(true)

        let result = Result { try backend.generate(prompt: "hello", systemPrompt: nil,
                                                    config: GenerationConfig(temperature: 0.0)) }
        if case .failure(let e) = result,
           case InferenceError.alreadyGenerating = e {
            XCTFail("Must not throw .alreadyGenerating when isGenerating == false")
        }
        // Any other outcome (a different error from tokenization, or a stream) is expected.
    }

    // MARK: - Model-gated scaffolds (issue #27 remainder)

    /// Scaffold: thinking-budget enforcement wiring. The `ThinkingLoopBudget` decision
    /// logic is unit-tested headlessly via `LlamaGenerationDriverHelpersTests`; only the
    /// live break inside the generation loop needs a real model.
    func test_generate_enforcesThinkingBudget_scaffold() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["RUN_SLOW_TESTS"] == "1",
            "Thinking-budget live-loop enforcement requires a real GGUF model; set RUN_SLOW_TESTS=1"
        )
        // TODO: load model, generate with maxThinkingTokens=4, assert thinkingTokenCount <= 4
    }

    /// Scaffold: repetition-guard break wiring. `RepeatWindow` and `tailRepeats` logic is
    /// unit-tested headlessly; only the live loop-break needs a real model.
    func test_generate_breaksOnRepetition_scaffold() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["RUN_SLOW_TESTS"] == "1",
            "Repetition-guard live-loop break requires a real GGUF model; set RUN_SLOW_TESTS=1"
        )
        // TODO: load a small model known to loop, assert generation terminates
    }
}
