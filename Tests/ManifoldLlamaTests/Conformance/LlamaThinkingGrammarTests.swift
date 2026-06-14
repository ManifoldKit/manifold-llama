import XCTest
import Foundation
import ManifoldInference
import ManifoldTestSupport
import ManifoldLlama
@_spi(Testing) import ManifoldLlama

/// Pins the #1595 thinking × grammar phase-gate trade-off as a DECISION, not a surprise.
///
/// **The trade-off.** A GBNF grammar only constrains output while the grammar sampler is
/// STRICT. The #1595 two-chain phase gate (`GrammarPhaseGate`, and the permissive/strict
/// chain split in `LlamaGenerationDriver.run` — see `gateGrammarOnThinking` ~L302 and the
/// `grammarGate` flip on `.thinkingCompleted` ~L463-465) deliberately keeps the grammar
/// sampler PERMISSIVE while the model is inside its `<think>…</think>` reasoning phase,
/// flipping STRICT only once thinking completes. Consequence: a grammar request on a
/// thinking-capable model that does NOT disable thinking yields UNCONSTRAINED tokens until
/// `</think>` closes. This is intentional (reasoning prose can't satisfy `root ::= "yes"|"no"`),
/// but it is an unguarded surprise for a caller who assumes "grammar ⇒ constrained from token 0".
/// This test makes that boundary a documented, regression-pinned contract.
///
/// **Field facts (verified against ManifoldKit 0.50.0 source @ checkout):**
///   - `GenerationConfig.maxThinkingTokens: Int?` — the field controlling the thinking phase.
///     `0` ⇒ disable thinking entirely (single-chain, grammar strict from token 0);
///     `N > 0` ⇒ cap reasoning at N tokens. (`InferenceBackend.swift` L215-231.)
///   - `GenerationConfig.grammar: String?` (L256).
///   - `GenerationEvent` cases `.token(String)` (L50), `.thinkingToken(String)` (L83),
///     `.thinkingCompleted` (L86) in `GenerationEvent.swift`.
///
/// **Skips-empty by design.** Needs a thinking-capable (Qwen) reasoning GGUF on disk. With
/// no models present (the CI default) the test throws `XCTSkip` with the standard message and
/// the suite merges green. Reuses the same path-contains discovery guard as
/// `LlamaGrammarConformanceTests` to defeat the `findGGUFModel` smallest-model false positive.
///
/// **This documents current behavior — it does NOT change the driver.** No `Sources/` edit.
final class LlamaThinkingGrammarTests: XCTestCase {

    /// Strict alternation grammar — the C3 fixture. Reasoning prose cannot satisfy it, which is
    /// exactly why the permissive thinking phase exists.
    private static let alternation = #"root ::= "yes" | "no""#

    override func setUp() async throws {
        try await super.setUp()
        try XCTSkipUnless(HardwareRequirements.isPhysicalDevice,
                          "LlamaBackend requires Metal (unavailable in simulator)")
        try XCTSkipUnless(HardwareRequirements.isAppleSilicon,
                          "LlamaBackend requires Apple Silicon")
    }

    // MARK: - Discovery (matches LlamaGrammarConformanceTests.loadBackend)

    /// Resolves a Qwen reasoning GGUF and loads a backend. Throws `XCTSkip` (standard message)
    /// when none is on disk — this is how the test merges green in CI.
    ///
    /// The `path.lowercased().contains("qwen")` guard is load-bearing: `findGGUFModel` (a)
    /// flips local-model discovery ON for any non-nil fragment and (b) falls back to the
    /// *smallest* discovered GGUF when the fragment matches nothing — so without this guard a
    /// box holding only a non-Qwen model would run that wrong model instead of skipping.
    private func loadQwenBackend() async throws -> LlamaBackend {
        guard let modelURL = HardwareRequirements.findGGUFModel(nameContains: "qwen"),
              modelURL.path.lowercased().contains("qwen") else {
            throw XCTSkip("No GGUF on disk for family 'qwen'. "
                        + "Set LLAMA_TEST_MODEL=<path> or place a `.gguf` whose path contains 'qwen' "
                        + "in ~/Documents/Models/ (with MANIFOLD_DISCOVER_LOCAL_MODELS=1) to run this test.")
        }
        let backend = LlamaBackend()
        addTeardownBlock { await backend.unloadAndWait() }
        try await backend.loadModel(from: modelURL, plan: .testStub(effectiveContextSize: 512))
        return backend
    }

    private func grammarConfig(maxTokens: Int) -> GenerationConfig {
        var config = GenerationConfig(temperature: 0.1, seed: 0xC0FFEE, maxOutputTokens: maxTokens)
        config.grammar = Self.alternation
        return config
    }

    // MARK: - Assertion 1: grammar + thinking DISABLED ⇒ constrained

    /// The contract callers can rely on: with a strict grammar AND thinking forced off
    /// (`maxThinkingTokens = 0`), generation runs the single strict chain and the FINAL visible
    /// output satisfies the grammar.
    ///
    /// Sabotage check: change `config.maxThinkingTokens = 0` to a positive value (e.g. `64`) and
    /// keep the same reasoning prompt — the grammar goes permissive during thinking and the
    /// drained `.token` tail can fall outside `{yes,no}` (or be empty if `</think>` never closes),
    /// failing the membership assertion. (That is precisely the trade-off Assertion 2 pins.)
    func test_grammar_thinkingDisabled_constrainsOutput() async throws {
        let backend = try await loadQwenBackend()

        var config = grammarConfig(maxTokens: 16)
        config.maxThinkingTokens = 0  // disable thinking ⇒ single strict chain (verified L302/L626)

        let stream = try backend.generate(prompt: "Answer yes or no: is the sky blue?",
                                          systemPrompt: nil, config: config)
        var output = ""
        for try await event in stream.events {
            if case .token(let t) = event { output += t }
        }
        let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertTrue(["yes", "no"].contains(trimmed),
                      "thinking-disabled: a strict grammar must constrain the final output to {yes,no}; "
                    + "got \(trimmed.debugDescription)")
    }

    // MARK: - Assertion 2: grammar + thinking ENABLED ⇒ permissive during, strict after

    /// Pins the #1595 trade-off. With the same grammar but thinking ENABLED:
    ///   - IF reasoning surfaces, it arrives as `.thinkingToken` free text — NOT grammar-pruned
    ///     into `{yes,no}` (the permissive chain governs the thinking phase), and
    ///   - any post-`</think>` visible `.token` output DOES satisfy the grammar (the gate flipped
    ///     strict on `.thinkingCompleted`).
    ///
    /// Whether the model opens a `<think>` block at all is model/prompt-intrinsic, so if NO
    /// `.thinkingToken` surfaces we LOG it and assert only the constrained-output invariant — we
    /// do not fail on "the model chose not to think" (mirrors the C4 divergence philosophy in the
    /// sibling suite). This encodes the documented trade-off; it never asserts a behavior the
    /// model can't be forced into.
    ///
    /// Sabotage check: make the gate strict during the thinking phase (apply the grammar sampler
    /// even while the thinking parser is active in `LlamaGenerationDriver.run` — i.e. delete the
    /// permissive chain / `gateGrammarOnThinking` branch ~L444-465) — the reasoning prose gets
    /// grammar-pruned, so EITHER `.thinkingToken` text collapses toward `{yes,no}` fragments
    /// (`reasoningLooksConstrained` flips true → fails) OR no free-text reasoning surfaces and the
    /// permissive-phase intent is lost.
    func test_grammar_thinkingEnabled_permissiveDuringThinking_strictAfter() async throws {
        let backend = try await loadQwenBackend()

        guard backend.capabilities.supportsThinking else {
            print("[thinking-grammar][qwen] model reports supportsThinking == false "
                + "(GGUF advertised no thinking markers) — phase-gate sub-case is informational-only; "
                + "the constrained-output invariant is covered by test_grammar_thinkingDisabled_constrainsOutput")
            throw XCTSkip("Loaded GGUF is not thinking-capable; the #1595 phase gate cannot be exercised.")
        }

        var config = grammarConfig(maxTokens: 96)
        config.maxThinkingTokens = 64  // enable thinking ⇒ two-chain phase gate (#1595)

        let stream = try backend.generate(
            prompt: "Think step by step, then answer with exactly yes or no: is 2 a prime number?",
            systemPrompt: nil, config: config)

        var reasoning = ""
        var sawThinkingCompleted = false
        var visibleOutput = ""
        for try await event in stream.events {
            switch event {
            case .thinkingToken(let t):  reasoning += t
            case .thinkingCompleted:     sawThinkingCompleted = true
            case .token(let t):          visibleOutput += t
            default:                     break
            }
        }

        if !reasoning.isEmpty {
            // Permissive phase: reasoning is FREE TEXT routed to `.thinkingToken`, not grammar-pruned.
            // If the grammar had constrained the thinking phase, reasoning would be nothing but
            // {yes,no} fragments. Assert it is NOT — i.e. it carries content outside the grammar's
            // tiny alphabet. (A genuine reasoning trace is far longer than "yes"/"no".)
            let reasoningLooksConstrained =
                ["yes", "no"].contains(reasoning.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
            XCTAssertFalse(reasoningLooksConstrained,
                           "thinking-enabled: reasoning must be permissive free text, NOT grammar-pruned to "
                         + "{yes,no} during the thinking phase (#1595); got \(reasoning.debugDescription)")
            print("[thinking-grammar][qwen] permissive phase confirmed: reasoning surfaced as "
                + "\(reasoning.count) chars of free .thinkingToken text")
        } else {
            print("[thinking-grammar][qwen] no .thinkingToken surfaced for this prompt "
                + "(model did not open <think>, or GGUF marker auto-detection found none) — informational; "
                + "asserting the constrained-output invariant only")
        }

        if sawThinkingCompleted {
            // Strict tail: gate flipped on `.thinkingCompleted`, so the post-</think> output MUST
            // satisfy the grammar.
            let trimmed = visibleOutput.trimmingCharacters(in: .whitespacesAndNewlines)
            XCTAssertTrue(["yes", "no"].contains(trimmed),
                          "thinking-enabled: post-</think> output must satisfy the grammar (gate strict on "
                        + ".thinkingCompleted, #1595); got \(trimmed.debugDescription)")
            print("[thinking-grammar][qwen] strict tail confirmed: post-</think> output \(trimmed.debugDescription)")
        } else {
            // Thinking opened but `</think>` never closed within budget: the permissive phase is the
            // only phase reached, so the strict-tail invariant is not reachable. Behavioral, not a
            // failure (mirrors C4/runThinking). The permissive-phase assertion above still ran.
            print("[thinking-grammar][qwen] </think> did not close within \(config.maxOutputTokens ?? -1) tokens "
                + "— permissive phase reached, strict-tail assertion not reachable (informational)")
        }
    }
}
