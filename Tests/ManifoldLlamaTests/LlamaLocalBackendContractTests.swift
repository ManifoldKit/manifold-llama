import XCTest
import ManifoldInference
import ManifoldTestSupport
import ManifoldBackendTestKit
@_spi(Testing) import ManifoldLlama

/// Llama participant for the local-backend contract suite.
///
/// Moves to manifold-llama with the backend (#1749). Scenario implementations
/// live in ``ManifoldBackendTestKit/LocalBackendContractRunner``.
///
/// The `makeBackend` factory returns an unconfigured `LlamaBackend` (no model) —
/// its only consumer is the capability-gate scenario, which throws before
/// `generate()` and needs no model. The two `generate()` scenarios are
/// overridden with self-contained bodies that load their own GGUF and gate on
/// `findGGUFModel()` + Apple Silicon, because the shared runner's assertions are
/// structurally incompatible with a live sampler: it expects an exact golden
/// token stream (we sample at temperature 0.7 with a random seed) and reads
/// `isGenerating` immediately after draining the stream (the backend clears it
/// in a `defer` that runs after `continuation.finish()`). See each method.
///
/// The `maxContextTokens: 4096` value mirrors the default `_effectiveContextSize`
/// set at `LlamaBackend.init()` before any model is loaded.
final class LlamaLocalBackendContractTests: XCTestCase {

    private static let participant = LocalBackendContractParticipant(
        label: "llama.backend",
        fixtureDirectory: "llama",
        capabilities: BackendCapabilities(
            supportedParameters: [
                .temperature, .topP, .topK, .repeatPenalty,
                .minP, .repetitionPenalty, .presencePenalty, .frequencyPenalty,
                .llamaDRY, .llamaXTC, .llamaMirostatV2,
            ],
            maxContextTokens: 4096,
            requiresPromptTemplate: true,
            supportsSystemPrompt: true,
            supportsToolCalling: true,
            supportsStructuredOutput: false,
            supportsNativeJSONMode: false,
            cancellationStyle: .explicit,
            supportsTokenCounting: true,
            memoryStrategy: .mappable,
            maxOutputTokens: 4096,
            supportsStreaming: true,
            isRemote: false,
            supportsKVCachePersistence: true,
            supportsGrammarConstrainedSampling: true,
            supportsThinking: true,
            supportsVision: false
        ),
        requiresSlowTests: true,
        makeBackend: {
            // Only consumer is the capability-gate scenario, which throws before
            // `generate()` and needs no model. The two generation scenarios are
            // overridden below with self-contained bodies that load their own GGUF,
            // because the shared runner's exact-golden / immediate-flag assertions
            // are incompatible with a live sampler (see those methods).
            LlamaBackend()
        }
    )

    /// The shared runner's `assertSimplePromptEmitsTokensInOrder` compares the
    /// captured stream against an EXACT, totally-ordered golden `expected.jsonl`.
    /// That fits replay backends (claude/openai/ollama drive recorded SSE) but is
    /// structurally incompatible with a live local sampler: the runner drives
    /// `generate()` with a bare `GenerationConfig()` (temperature 0.7, no seed),
    /// and `LlamaGenerationDriver` falls back to a fresh `UInt32.random` seed when
    /// the caller requests no determinism (~line 288), so the exact token sequence
    /// differs on every run — a captured golden fixture would pass once and flake
    /// forever (which is why none was ever committed).
    ///
    /// We assert the deterministic structural invariant the scenario name actually
    /// promises instead: a live generation emits a non-empty, in-order `.token`
    /// stream that terminates with `isGenerating == false`. Re-unify with the
    /// shared runner if it ever gains a way to pin a deterministic seed for this
    /// scenario.
    func test_generate_simplePrompt_emitsTokensInOrder() async throws {
        guard let modelURL = HardwareRequirements.findGGUFModel() else {
            throw XCTSkip("No GGUF on disk. Set LLAMA_TEST_MODEL=<path> or place a `.gguf` in ~/Documents/Models/ to run this test.")
        }
        try XCTSkipUnless(HardwareRequirements.isPhysicalDevice && HardwareRequirements.isAppleSilicon,
                          "LlamaBackend requires Apple Silicon + Metal (unavailable in simulator)")

        let backend = LlamaBackend()
        addTeardownBlock { await backend.unloadAndWait() }
        try await backend.loadModel(from: modelURL, plan: .testStub(effectiveContextSize: 512))

        var config = GenerationConfig()
        config.maxOutputTokens = 16
        let stream = try backend.generate(prompt: "Hello", systemPrompt: nil, config: config)

        var tokens: [String] = []
        for try await event in stream.events {
            if case .token(let text) = event { tokens.append(text) }
        }

        XCTAssertFalse(tokens.isEmpty, "expected at least one token event from a live generation")
        // `isGenerating` is cleared in the generation task's `defer`, which runs
        // *after* the stream's `continuation.finish()` — draining `stream.events`
        // does NOT guarantee the flag has flipped (see `awaitGenerationSettled`
        // doc in LlamaBackend). Await settlement before asserting to avoid racing
        // the defer.
        await backend.awaitGenerationSettled()
        XCTAssertFalse(backend.isGenerating, "isGenerating must be false after the stream drains")
    }

    /// The shared runner's `assertStopsGeneratingAfterStreamEnd` drains the
    /// stream and asserts `isGenerating == false` immediately — but for this
    /// backend the flag is cleared in the generation task's `defer`, which runs
    /// *after* `continuation.finish()` (see `awaitGenerationSettled` doc in
    /// LlamaBackend). The runner has no happens-before edge to that defer, so the
    /// assertion races it and flakes. We drive the same scenario but await
    /// settlement — the documented way to observe the post-generation state — via
    /// the llama-specific SPI the shared runner can't reach.
    func test_generate_stopsGenerating_afterStreamEnd() async throws {
        guard let modelURL = HardwareRequirements.findGGUFModel() else {
            throw XCTSkip("No GGUF on disk. Set LLAMA_TEST_MODEL=<path> or place a `.gguf` in ~/Documents/Models/ to run this test.")
        }
        try XCTSkipUnless(HardwareRequirements.isPhysicalDevice && HardwareRequirements.isAppleSilicon,
                          "LlamaBackend requires Apple Silicon + Metal (unavailable in simulator)")

        let backend = LlamaBackend()
        addTeardownBlock { await backend.unloadAndWait() }
        try await backend.loadModel(from: modelURL, plan: .testStub(effectiveContextSize: 512))

        var config = GenerationConfig()
        config.maxOutputTokens = 16
        let stream = try backend.generate(prompt: "ping", systemPrompt: nil, config: config)
        for try await _ in stream.events {}

        await backend.awaitGenerationSettled()
        XCTAssertFalse(backend.isGenerating,
                       "isGenerating must be false after the stream ends and generation settles")
    }

    func test_capabilityGate_disclaimedRequirementThrows() async {
        await LocalBackendContractRunner.assertCapabilityGateDisclaimedRequirementThrows(
            participant: Self.participant
        )
    }
}

/// Real-driver coverage + adapter-shape checks for the Llama generation driver.
/// Split out of the shared `LocalBackendRealDriverCoverageTest` /
/// `LocalInferenceAdapterSmokeTests` so the file can move to manifold-llama.
final class LlamaLocalDriverCoverageTests: XCTestCase {

    func test_llamaDriverHasRealPathForEveryClaim() throws {
        let driver = LlamaGenerationDriver()
        try LocalDriverCoverageChecks.assertCoverage(
            adapter: driver,
            sourceFileSuffix: "Sources/ManifoldLlama/LlamaGenerationDriver.swift"
        )
    }

    func test_llamaGenerationDriverConformsToProtocol() {
        let driver = LlamaGenerationDriver()
        LocalDriverCoverageChecks.assertAdapterShape(driver, expectedName: "llama.generation")
    }
}
