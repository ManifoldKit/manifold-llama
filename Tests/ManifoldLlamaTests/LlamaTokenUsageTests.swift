import XCTest
@_spi(Testing) import ManifoldLlama
@_spi(BackendInternals) import ManifoldContract
import ManifoldInference
import ManifoldTestSupport

/// Headless coverage for the `.usage(TokenUsage)` emission contract (issue #44).
///
/// Full end-to-end verification (driving `LlamaGenerationDriver.run()` through to
/// the usage event) requires a live `llama_context` and is therefore gated behind
/// `RUN_SLOW_TESTS=1`. The headless tests here cover:
///
///   1. `LlamaBackend` conforms to `TokenUsageProvider` — the type check is the
///      conformance guard; if the protocol shape changes these fail to compile.
///   2. `lastUsage` returns `nil` before any generation (zero-state invariant).
///
/// The `onUsage` wiring from `LlamaGenerationDriver.run()` through to
/// `LlamaBackend._lastUsage` is exercised by the slow integration path guarded
/// below; the headless tests verify the public surface without a model.
final class LlamaTokenUsageTests: XCTestCase {

    // MARK: - Protocol conformance (headless)

    /// `LlamaBackend` must conform to `TokenUsageProvider` so `InferenceService`
    /// can read `lastUsage` for local turns the same way it does for cloud turns.
    func test_conformsToTokenUsageProvider() {
        let backend = LlamaBackend()
        // A runtime cast is sufficient — if the extension is missing, this line fails.
        XCTAssertNotNil(backend as? any TokenUsageProvider,
            "LlamaBackend must conform to TokenUsageProvider (issue #44)")
    }

    /// Before any generation the property must be `nil` — callers that read
    /// `lastUsage` immediately after init must not see stale data.
    func test_lastUsage_nilByDefault() {
        let backend = LlamaBackend()
        XCTAssertNil(backend.lastUsage,
            "lastUsage must be nil before any generation turn completes")
    }

    // MARK: - End-to-end usage event (model-gated)

    /// Drives a full local generation turn and verifies that:
    ///   1. A `.usage(TokenUsage)` event is emitted at end-of-stream.
    ///   2. `backend.lastUsage` reflects the same prompt + completion counts.
    ///
    /// Skipped unless a `.gguf` is discoverable via `findGGUFModel()`
    /// (`LLAMA_TEST_MODEL` / `~/Documents/Models`) on Apple Silicon — the same
    /// gate every model-bound suite uses, so this runs in the model-bearing lane.
    /// This keeps CI green without a model while allowing local verification.
    func test_generate_emitsUsageEvent_andPopulatesLastUsage() async throws {
        guard let url = HardwareRequirements.findGGUFModel() else {
            throw XCTSkip("No GGUF on disk. Set LLAMA_TEST_MODEL=<path> or place a `.gguf` in ~/Documents/Models/ to run this test.")
        }
        try XCTSkipUnless(HardwareRequirements.isPhysicalDevice && HardwareRequirements.isAppleSilicon,
                          "LlamaBackend requires Apple Silicon + Metal (unavailable in simulator)")

        let backend = LlamaBackend()
        try await backend.loadModel(from: url, plan: .testStub(effectiveContextSize: 512))
        defer { backend.unloadModel() }

        let stream = try backend.generate(
            prompt: "Hello",
            systemPrompt: nil,
            config: GenerationConfig(maxOutputTokens: 5)
        )

        var usageEvent: TokenUsage?
        for try await event in stream.events {
            if case .usage(let usage) = event {
                usageEvent = usage
            }
        }
        await backend.awaitGenerationSettled()

        XCTAssertNotNil(usageEvent,
            "A .usage(TokenUsage) event must be emitted at end of local generation turn")
        XCTAssertGreaterThan(usageEvent?.promptTokens ?? 0, 0,
            "Prompt token count must be positive")
        XCTAssertGreaterThanOrEqual(usageEvent?.completionTokens ?? -1, 0,
            "Completion token count must be non-negative")

        let lastUsage = backend.lastUsage
        XCTAssertEqual(lastUsage?.promptTokens, usageEvent?.promptTokens,
            "backend.lastUsage.promptTokens must match the emitted event")
        XCTAssertEqual(lastUsage?.completionTokens, usageEvent?.completionTokens,
            "backend.lastUsage.completionTokens must match the emitted event")
    }
}
