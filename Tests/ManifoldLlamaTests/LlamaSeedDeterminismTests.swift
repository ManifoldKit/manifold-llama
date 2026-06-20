import XCTest
import ManifoldInference
import ManifoldTestSupport
import ManifoldLlama
@_spi(Testing) import ManifoldLlama

/// Verifies that ``GenerationConfig/seed`` makes ``LlamaBackend`` token streams
/// reproducible across runs. The driver feeds the seed into
/// `llama_sampler_init_dist`, so two `generate()` calls with identical prompt /
/// config / model state must produce the same token sequence.
///
/// Requires Apple Silicon and a real GGUF on disk — gated by
/// ``HardwareRequirements`` and skipped otherwise.
final class LlamaSeedDeterminismTests: XCTestCase {

    override func setUp() async throws {
        try await super.setUp()
        try XCTSkipUnless(HardwareRequirements.isPhysicalDevice,
                          "LlamaBackend requires Metal (unavailable in simulator)")
        try XCTSkipUnless(HardwareRequirements.isAppleSilicon,
                          "LlamaBackend requires Apple Silicon")
    }

    /// Two generations with the same `seed` produce identical token streams.
    ///
    /// We collect the full visible token sequence for two consecutive runs against
    /// the same loaded model. Under a correct seed implementation, `outputA == outputB`.
    ///
    /// Sabotage check: replace the seed plumbing in `LlamaGenerationDriver.run` with
    /// `UInt32.random(in: 0...UInt32.max)` (the prior behaviour). The two runs will
    /// diverge after the first sampled token and this assertion will fail.
    func test_sameSeed_producesIdenticalOutput() async throws {
        guard let modelURL = HardwareRequirements.findGGUFModel() else {
            throw XCTSkip("No GGUF on disk. Set LLAMA_TEST_MODEL=<path> or place a `.gguf` in ~/Documents/Models/ to run this test.")
        }

        let backend = LlamaBackend()
        addTeardownBlock { await backend.unloadAndWait() }
        try await backend.loadModel(from: modelURL, plan: .testStub(effectiveContextSize: 512))

        var config = GenerationConfig(temperature: 0.8, maxOutputTokens: 16)
        config.seed = 42

        let outputA = try await collectTokens(backend: backend, prompt: "List three colours:", config: config)
        backend.resetConversation()
        let outputB = try await collectTokens(backend: backend, prompt: "List three colours:", config: config)

        XCTAssertFalse(outputA.isEmpty, "Seeded generation must produce at least one token")
        XCTAssertEqual(outputA, outputB,
                       "Identical seeds must produce identical token streams; "
                     + "got A=\(outputA.debugDescription) B=\(outputB.debugDescription)")
    }

    /// Different-seed divergence — the seed-pinned-to-constant guard — lives in
    /// `LlamaModernSamplerIntegrationTests.test_differentSeeds_divergeOverLongStream`,
    /// which uses a tolerant oracle (long stream + retry across distinct seed
    /// pairs). It is deliberately NOT duplicated here as a single-pair
    /// `XCTAssertNotEqual` over a short stream: distinct seeds at temperature 1.0
    /// can legitimately coincide on a short prefix, so that form is flaky. See
    /// that test for the divergence assertion and its tolerance rationale.

    // MARK: - Helpers

    private func collectTokens(
        backend: LlamaBackend,
        prompt: String,
        config: GenerationConfig
    ) async throws -> String {
        let stream = try backend.generate(prompt: prompt, systemPrompt: nil, config: config)
        var text = ""
        for try await event in stream.events {
            if case .token(let chunk) = event {
                text += chunk
            }
        }
        // The stream's last element fires from `continuation.finish()`, but
        // `LlamaBackend` clears `isGenerating` in the generation task's `defer`,
        // which has no happens-before relationship with the consumer loop exiting
        // above. Without awaiting the task to settle, the *next* generate() can
        // race the defer and throw `.alreadyGenerating` (observed under
        // full-suite load). Awaiting the in-flight task guarantees the defer has
        // run and `isGenerating == false` before the caller starts gen #2.
        await backend.awaitGenerationSettled()
        return text
    }
}
