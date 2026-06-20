import XCTest
import ManifoldInference
import ManifoldTestSupport
import ManifoldLlama
@_spi(Testing) import ManifoldLlama

/// Integration determinism coverage for the XTC and Mirostat v2 sampler additions.
///
/// Both samplers consume the GenerationConfig seed; two runs at the **same** seed
/// must produce identical token streams. That is the only invariant asserted here
/// — these are same-seed determinism checks.
///
/// The same-seed checks below do NOT prove the seed actually *influences* output.
/// A regression that silently drops the seed and always uses 0 would still pass
/// them, because hardcoding the seed keeps same-seed runs identical. That gap is
/// closed by `test_differentSeeds_divergeOverLongStream` (#29): a tolerant
/// different-seed divergence guard that a seed-pinned-to-0 regression must fail.
///
/// Requires Apple Silicon and a real GGUF — gated and skipped otherwise.
final class LlamaModernSamplerIntegrationTests: XCTestCase {

    override func setUp() async throws {
        try await super.setUp()
        try XCTSkipUnless(HardwareRequirements.isPhysicalDevice,
                          "LlamaBackend requires Metal (unavailable in simulator)")
        try XCTSkipUnless(HardwareRequirements.isAppleSilicon,
                          "LlamaBackend requires Apple Silicon")
    }

    func test_xtc_sameSeed_isDeterministic() async throws {
        guard let modelURL = HardwareRequirements.findGGUFModel() else {
            throw XCTSkip("No GGUF on disk. Set LLAMA_TEST_MODEL=<path> or place a `.gguf` in ~/Documents/Models/ to run this test.")
        }

        let backend = LlamaBackend()
        addTeardownBlock { await backend.unloadAndWait() }
        try await backend.loadModel(from: modelURL, plan: .testStub(effectiveContextSize: 512))

        var config = GenerationConfig(
            temperature: 0.8,
            llamaXTC: LlamaXTCSamplerOptions(probability: 0.5, threshold: 0.10, minKeep: 1),
            maxOutputTokens: 24
        )
        config.seed = 42
        // Disable thinking: `findGGUFModel()` may pick a reasoning model (Qwen3-0.6B)
        // whose longer thinking phase both widens the back-to-back generation window
        // and adds a reasoning stream. The determinism property under test is about
        // the seeded sampler, not the thinking transform — keep the compared output
        // to the visible stream so the assertion is model-shape independent.
        config.maxThinkingTokens = 0

        let outputA = try await collectTokens(backend: backend, prompt: "Tell me a fact:", config: config)
        backend.resetConversation()
        let outputB = try await collectTokens(backend: backend, prompt: "Tell me a fact:", config: config)

        XCTAssertFalse(outputA.isEmpty, "Sampling must still produce tokens when XTC is active")
        XCTAssertEqual(outputA, outputB, "Same seed + same XTC config must produce identical streams")
    }

    func test_mirostatV2_sameSeed_isDeterministic() async throws {
        guard let modelURL = HardwareRequirements.findGGUFModel() else {
            throw XCTSkip("No GGUF on disk. Set LLAMA_TEST_MODEL=<path> or place a `.gguf` in ~/Documents/Models/ to run this test.")
        }

        let backend = LlamaBackend()
        addTeardownBlock { await backend.unloadAndWait() }
        try await backend.loadModel(from: modelURL, plan: .testStub(effectiveContextSize: 512))

        var config = GenerationConfig(
            llamaMirostatV2: LlamaMirostatV2SamplerOptions(tau: 5.0, eta: 0.1),
            maxOutputTokens: 24
        )
        config.seed = 1_337
        // See test_xtc_sameSeed_isDeterministic: disable thinking so the compared
        // streams are the seeded visible output, independent of which model is smallest.
        config.maxThinkingTokens = 0

        let outputA = try await collectTokens(backend: backend, prompt: "Tell me a fact:", config: config)
        backend.resetConversation()
        let outputB = try await collectTokens(backend: backend, prompt: "Tell me a fact:", config: config)

        XCTAssertFalse(outputA.isEmpty, "Mirostat v2 must still produce tokens")
        XCTAssertEqual(outputA, outputB, "Same seed + same Mirostat config must produce identical streams")
    }

    // MARK: - Different-seed divergence (the seed-pinned-to-0 guard)

    /// Two **distinct** seeds at `temperature = 1.0` must produce divergent token
    /// streams. This is the guard the same-seed determinism tests cannot provide:
    /// a regression that silently pins the internal seed to 0 (or any constant)
    /// keeps same-seed runs identical AND makes every different-seed run identical
    /// too — so only an *inequality* across seeds catches it. With a working seed,
    /// two different seeds drive `llama_sampler_init_dist` from different RNG state
    /// and the sampled streams diverge.
    ///
    /// ## Tolerant oracle (why this is not a single `XCTAssertNotEqual`)
    ///
    /// Distinct seeds do NOT *guarantee* divergence on every prefix: at temp 1.0 a
    /// peaked distribution can sample the same argmax token for several positions,
    /// and on a *short* stream two seeds can legitimately coincide entirely (a real
    /// collision, not a bug). A naive `assertNotEqual` over a few tokens would be
    /// flaky. Two tolerances make this robust while still failing a pinned seed:
    ///
    ///   1. **Long stream.** We request many tokens (`maxOutputTokens` below). The
    ///      probability that two independent RNG streams agree on *every* sampled
    ///      position falls off geometrically with length, so a long stream makes a
    ///      legitimate full-collision vanishingly rare while a pinned seed collides
    ///      with probability 1.
    ///   2. **Retry across seed pairs.** Even a long stream can coincide by chance
    ///      (greedy-ish prompt, early EOG). We try several *distinct* seed pairs and
    ///      pass as soon as ANY pair diverges. A correctly-seeded backend only needs
    ///      one diverging pair; a seed-pinned-to-0 backend diverges on NONE, so it
    ///      exhausts every pair and fails. False-pass requires ALL pairs to collide,
    ///      whose probability is the per-pair collision rate raised to the pair count
    ///      — negligible. False-fail requires a working seed to collide on every pair
    ///      over a long stream — also negligible.
    func test_differentSeeds_divergeOverLongStream() async throws {
        guard let modelURL = HardwareRequirements.findGGUFModel() else {
            throw XCTSkip("No GGUF on disk. Set LLAMA_TEST_MODEL=<path> or place a `.gguf` in ~/Documents/Models/ to run this test.")
        }

        let backend = LlamaBackend()
        addTeardownBlock { await backend.unloadAndWait() }
        try await backend.loadModel(from: modelURL, plan: .testStub(effectiveContextSize: 512))

        // Distinct seed pairs tried in order; pass on the first pair that diverges.
        // Each pair uses two clearly different seeds. Three pairs is enough: for any
        // plausible per-pair collision rate p over a 64-token stream, p^3 is far
        // below test-flake thresholds, while a pinned seed yields p = 1 on all three.
        let seedPairs: [(UInt64, UInt64)] = [
            (42, 1337),
            (7, 99_991),
            (2_024, 8_675_309),
        ]

        // A long stream: divergence probability per pair rises with length, so we
        // give the RNG room to separate rather than betting on a short prefix.
        let longStreamTokens = 64

        var observedNonEmpty = false
        var lastA = ""
        var lastB = ""

        for (seedA, seedB) in seedPairs {
            var configA = GenerationConfig(temperature: 1.0, maxOutputTokens: longStreamTokens)
            configA.seed = seedA
            // Disable thinking: `findGGUFModel()` may pick a reasoning model whose
            // thinking phase both widens the back-to-back window and changes which
            // stream we capture. The divergence property is about the seeded dist
            // sampler over the visible stream, independent of model shape — see the
            // same-seed tests above for the identical rationale.
            configA.maxThinkingTokens = 0
            var configB = configA
            configB.seed = seedB

            let outputA = try await collectTokens(backend: backend, prompt: "Tell me a story:", config: configA)
            backend.resetConversation()
            let outputB = try await collectTokens(backend: backend, prompt: "Tell me a story:", config: configB)
            backend.resetConversation()

            if !outputA.isEmpty { observedNonEmpty = true }
            lastA = outputA
            lastB = outputB

            if outputA != outputB {
                // Divergence observed — seed influences output. Done.
                return
            }
        }

        // If we never produced any tokens at all, the model/prompt is the problem,
        // not the seed — surface that distinctly rather than as a divergence failure.
        XCTAssertTrue(observedNonEmpty,
                      "No seed pair produced any tokens; cannot evaluate divergence")

        // Every distinct seed pair collided over a long stream. With a working seed
        // this is astronomically unlikely; the overwhelmingly likely cause is that
        // the seed is being ignored (pinned to a constant such as 0).
        XCTFail("All \(seedPairs.count) distinct seed pairs produced identical \(longStreamTokens)-token "
              + "streams at temperature=1.0. This is the signature of a seed-pinned-to-constant "
              + "regression — the seed is not reaching llama_sampler_init_dist. "
              + "Last pair: A=\(lastA.debugDescription) B=\(lastB.debugDescription)")
    }

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
        // The stream's terminal element fires from `continuation.finish()`, but
        // `LlamaBackend` clears `isGenerating` in the generation task's `defer`,
        // which has no happens-before relationship with this consumer loop exiting.
        // Draining the stream is NOT sufficient to guarantee the next generate()
        // won't race the defer and throw `.alreadyGenerating` — observed when a
        // longer (thinking-capable) model widens the window. Await the in-flight
        // task to settle so the defer has run and `isGenerating == false` before
        // the caller starts the second generation. This mirrors the documented
        // contract on `awaitGenerationSettled()` and LlamaSeedDeterminismTests.
        await backend.awaitGenerationSettled()
        return text
    }
}
