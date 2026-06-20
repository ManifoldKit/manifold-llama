import XCTest
import ManifoldInference
import ManifoldTestSupport
import ManifoldLlama

/// Re-homed from core's cross-family `KVCacheReuseRaceRegressionTests`
/// (Tests/ManifoldBackendsTests, retired in core PR C2 when the families
/// split out — ManifoldKit#1749). This is the Llama half: the real-GGUF
/// byte-exact determinism check across the KV-reuse boundary. The MLX half
/// (mock-driven #1382 stale-snapshot race guards) lives in manifold-mlx.
final class LlamaKVCacheReuseDeterminismTests: XCTestCase {

    private func drainEvents(_ stream: GenerationStream) async throws -> [GenerationEvent] {
        var events: [GenerationEvent] = []
        for try await event in stream.events {
            events.append(event)
        }
        return events
    }

    private func reuseCounts(in events: [GenerationEvent]) -> [Int] {
        events.compactMap { event in
            if case .kvCacheReuse(let count) = event { return count }
            return nil
        }
    }

    private func visibleText(in events: [GenerationEvent]) -> String {
        events.reduce(into: "") { acc, event in
            if case .token(let text) = event { acc += text }
        }
    }

    /// The true byte-exact correctness property #1382 threatened: a warm second
    /// turn (reuse ON, started from a cached prefix) must produce a token stream
    /// byte-identical to a cold second turn (a fresh backend that decodes the
    /// same prompt from scratch). Greedy decoding (`temperature: 0`) makes the
    /// argmax path deterministic, so any divergence means the reuse path's KV
    /// state was not equivalent to a full prefill — exactly the hazard class.
    ///
    /// The `LlamaBackend` reuses the matching KV prefix but only up to a `n_batch`
    /// boundary, resuming the re-decode at that aligned position (ManifoldKit#1677,
    /// superseding the #966 last-two-token cap). Because the resume point is a
    /// multiple of `n_batch`, the chunk that produces the sampling-position logits
    /// has the same Metal parallel-reduction path as the cold turn's, so the warm
    /// output is bit-identical; this test is the regression guard for that
    /// determinism. Genuine prefix reuse is preserved (O(new-tokens) prefill).
    ///
    /// Skips cleanly off-device or with no model — never faked.
    func test_llama_warmReuseTurnMatchesColdTurnByteForByte() async throws {
        try XCTSkipUnless(
            HardwareRequirements.isPhysicalDevice,
            "LlamaBackend requires Metal (unavailable in simulator)"
        )
        try XCTSkipUnless(
            HardwareRequirements.isAppleSilicon,
            "LlamaBackend requires Apple Silicon"
        )
        try XCTSkipUnless(
            HardwareRequirements.hasMetalDevice,
            "Requires a Metal device"
        )
        guard let modelURL = HardwareRequirements.findGGUFModel() else {
            throw XCTSkip("No GGUF model on disk. Set LLAMA_TEST_MODEL or place a .gguf in ~/Documents/Models/.")
        }

        let turn1Prompt = "The Swift programming language was created by"
        let turn2Prompt = turn1Prompt + " Apple, and it is used to build apps."
        // temperature 0 → greedy argmax → deterministic output, the precondition
        // for a meaningful byte-for-byte comparison.
        let config = GenerationConfig(temperature: 0.0, maxOutputTokens: 24)

        // Warm path: backend does turn 1, then turn 2 reusing the shared prefix.
        let warmBackend = LlamaBackend()
        addTeardownBlock { await warmBackend.unloadAndWait() }
        try await warmBackend.loadModel(from: modelURL, plan: .testStub(effectiveContextSize: 512))

        _ = try await drainEvents(try warmBackend.generate(prompt: turn1Prompt, systemPrompt: nil, config: config))
        // `isGenerating` is cleared in the generation task's `defer`, which runs
        // *after* the stream's `continuation.finish()` — draining turn 1's stream
        // does not guarantee the flag has flipped, so issuing turn 2 immediately
        // can race the defer and throw `.alreadyGenerating` (flaky under CI
        // timing). Await settlement between turns, the documented way to observe
        // post-generation readiness without unloading the model.
        await warmBackend.awaitGenerationSettled()
        let warmEvents = try await drainEvents(try warmBackend.generate(prompt: turn2Prompt, systemPrompt: nil, config: config))
        let warmReuse = reuseCounts(in: warmEvents)
        let warmText = visibleText(in: warmEvents)

        // Cold path: a fresh backend that has no prior KV state, decoding the
        // same turn-2 prompt from scratch.
        let coldBackend = LlamaBackend()
        addTeardownBlock { await coldBackend.unloadAndWait() }
        try await coldBackend.loadModel(from: modelURL, plan: .testStub(effectiveContextSize: 512))

        let coldEvents = try await drainEvents(try coldBackend.generate(prompt: turn2Prompt, systemPrompt: nil, config: config))
        let coldText = visibleText(in: coldEvents)

        XCTAssertGreaterThan(
            warmReuse.first ?? 0, 0,
            "Warm second turn must actually reuse a prefix (>0) or the determinism comparison is vacuous"
        )
        XCTAssertEqual(
            warmText, coldText,
            "Warm (KV-reuse) and cold (full-prefill) second turns must produce byte-identical greedy output — a mismatch is the #1382 non-exact-reuse hazard"
        )

        // Sabotage: in LlamaGenerationDriver, set the decode start to the raw
        // `reuseLen` instead of the batch-aligned `alignedReuseLen` (i.e. resume
        // mid-chunk). That changes the sampling-position chunk's batch shape vs the
        // cold path's, flips the argmax on near-tied logits for non-Qwen
        // architectures, and diverges warmText (ManifoldKit#1677).
    }
}
