import XCTest
import Foundation
import ManifoldInference
import ManifoldTestSupport
import ManifoldLlama
@_spi(Testing) import ManifoldLlama

/// Model-family × grammar-shape conformance matrix for GBNF-constrained sampling.
///
/// Today grammar *behavior* across model families is verified by exactly one
/// happy-path Llama test (`LlamaGrammarSamplerTests.test_grammar_constrainsOutput`,
/// grammar `root ::= [0-9]+`). The trivial digit grammar is precisely the case the
/// Gemma carve-out says still works; the real failure modes (JSON-object stall,
/// alternation, tokenizer/leading-space accuracy, tool-call envelopes) are unguarded
/// on every family. This suite makes grammar conformance a per-family, per-shape matrix.
///
/// **Skips-empty by design.** Each family resolves a GGUF on disk by name fragment via
/// `HardwareRequirements.findGGUFModel(nameContains:)`, which honors `LLAMA_TEST_MODEL`
/// and `MANIFOLD_DISCOVER_LOCAL_MODELS=1`. With no models on disk (the CI default) every
/// family throws `XCTSkip` — the suite reports all-skipped, never fails. Drop a `.gguf`
/// whose path contains a family fragment into `~/Documents/Models/` (and set
/// `MANIFOLD_DISCOVER_LOCAL_MODELS=1`) to light a family up.
///
/// Spec: ManifoldKit/docs/plans/model-family-grammar-conformance-suite.md
///
/// Adaptations vs. the spec (verified against live source @ ManifoldKit 0.50.0):
///   - `GrammarFamily.toolCallDialect` uses a local `ToolDialect` string enum: the spec's
///     `ToolCallMarker.Dialect` type does not exist — Llama exposes two `ToolCallMarker`
///     delimiter pairs via `LlamaToolMarkers.markers()`, not a dialect enum.
///   - C5 is scoped to assert that grammar-constrained `<tool_call>{json}</tool_call>`
///     output parses as a JSON object carrying a `name` key (via `JSONSerialization`),
///     rather than driving it through the full `ToolCallTransform` — the transform's body
///     parser (`LlamaToolMarkers.parseCallBuffer`) is private to the module. See `runC5`.
///   - Gemma carve-out (1.3): the live `LlamaBackend` does NOT throw
///     `InferenceError.unsupportedGrammar` when a grammar reaches a Gemma model — the
///     driver applies the grammar sampler whenever `config.grammar != nil`, and the
///     `supportsGrammarConstrainedSampling` flag is purely advisory for callers. So the
///     conformance assertion is the capability flag being `false` after a *real* load; we
///     do not fabricate a throw expectation the backend doesn't make. See `runGemmaCarveOut`.
///   - C2 key-presence uses `JSONSerialization` (simpler, reachable) rather than core
///     `JSONSchemaValidator`.
final class LlamaGrammarConformanceTests: XCTestCase {

    override func setUp() async throws {
        try await super.setUp()
        try XCTSkipUnless(HardwareRequirements.isPhysicalDevice,
                          "LlamaBackend requires Metal (unavailable in simulator)")
        try XCTSkipUnless(HardwareRequirements.isAppleSilicon,
                          "LlamaBackend requires Apple Silicon")
    }

    // MARK: - Family table

    /// Tool-call envelope dialect for a family. Local to the test target — the live
    /// codebase has no `ToolCallMarker.Dialect`; `LlamaToolMarkers` ships two marker
    /// delimiter pairs (Gemma-4 native `<|tool_call>`…`<|end_of_turn>` and the JSON
    /// fallback `<tool_call>`…`</tool_call>`). This enum only records intent for the
    /// C5 envelope fixture; all non-Gemma families exercise the JSON fallback envelope.
    enum ToolDialect: String {
        case hermes      // <tool_call>{json}</tool_call>
        case qwenJSON    // same JSON envelope, Qwen-style fine-tunes
        case generic     // JSON envelope
        case gemma4      // <|tool_call> native — not exercised (Gemma skips grammar)
    }

    struct GrammarFamily {
        let id: String
        let nameFragment: String
        let expectsGrammarSupport: Bool
        let isThinkingCapable: Bool
        let toolCallDialect: ToolDialect
    }

    /// Default roster. Extend as models land in `~/Documents/Models/`.
    static let families: [GrammarFamily] = [
        GrammarFamily(id: "llama",   nameFragment: "llama",   expectsGrammarSupport: true,  isThinkingCapable: false, toolCallDialect: .hermes),
        GrammarFamily(id: "qwen",    nameFragment: "qwen",    expectsGrammarSupport: true,  isThinkingCapable: true,  toolCallDialect: .qwenJSON),
        GrammarFamily(id: "mistral", nameFragment: "mistral", expectsGrammarSupport: true,  isThinkingCapable: false, toolCallDialect: .generic),
        GrammarFamily(id: "gemma",   nameFragment: "gemma",   expectsGrammarSupport: false, isThinkingCapable: false, toolCallDialect: .gemma4),
        GrammarFamily(id: "phi",     nameFragment: "phi",     expectsGrammarSupport: true,  isThinkingCapable: false, toolCallDialect: .generic),
    ]

    // MARK: - Grammar fixtures (test-target only — hand-authored GBNF literals)

    /// Five literal GBNF grammars exercised by C1–C5. Kept as literals next to the cases
    /// (no schema→GBNF emitter exists; callers hand-author GBNF, which is what we test).
    enum GrammarFixtures {
        /// C1: digit-only smoke. Sampler-wiring proof for every family.
        static let digits = "root ::= [0-9]+"

        /// C2: JSON object with required keys `city` and `temp`. The Gemma stall class —
        /// open `{`, whitespace-loop to EOG. Required keys for the key-presence assertion.
        ///
        /// **Bounded fields (issue #20).** The earlier rules had unbounded repetition
        /// (`string ::= "\"" ([a-zA-Z ]*) "\""`, `number ::= "-"? [0-9]+ ("." [0-9]+)?`):
        /// the grammar IS followed, but a small/degenerate model (mistral 7B observed) keeps
        /// emitting fractional digits and never reaches the closing `}` within `maxOutputTokens`,
        /// so C2 fails by *truncation* rather than by grammar violation. We bound every
        /// repeating run to a small upper count so a well-behaved model can always close the
        /// object inside the token budget while still proving the grammar constrains structure.
        ///   - city string: 0–24 letters/spaces
        ///   - integer part: 1–3 digits; fractional part: optional `.` + 1–2 digits
        static let jsonObjectRequiredKeys: Set<String> = ["city", "temp"]
        static let jsonObject = #"""
        root   ::= "{" ws "\"city\"" ws ":" ws string ws "," ws "\"temp\"" ws ":" ws number ws "}"
        string ::= "\"" [a-zA-Z ]{0,24} "\""
        number ::= "-"? [0-9]{1,3} ("." [0-9]{1,2})?
        ws     ::= [ \t\n]*
        """#

        /// C3: alternation. The construct the dead pre-validator wrongly called inexpressible.
        static let alternation = #"root ::= "yes" | "no""#

        /// C4: C3 built two ways — without and with a leading optional-space prefix. Both
        /// must stay valid; we LOG branch agreement/divergence rather than fail on divergence
        /// (it's model-intrinsic; tokenizer leading-space sensitivity, Lost-in-Space).
        static let alternationNoLeadingSpace = #"root ::= "yes" | "no""#
        static let alternationLeadingSpace   = #"root ::= " "? ("yes" | "no")"#

        /// C5: tool-call envelope. Literal `<tool_call>{json}</tool_call>` constraining to a
        /// `{"name": ...}` body. Scoped (see `runC5` / class doc): we assert the inner JSON
        /// parses with a `name` key rather than driving the full transform.
        ///
        /// **Bounded name (issue #20).** The earlier `string ::= "\"" ([a-zA-Z_]+) "\""` rule
        /// let the tool name grow without limit. The grammar IS followed (a valid `<tool_call>`
        /// opens), but small models degenerate — mistral/qwen 7B were observed emitting names
        /// like `get_weather_tool_call_example_response_json_v_one_zero_zero…` that never close
        /// the quote, so the closing `</tool_call>` is never reached inside `maxOutputTokens`
        /// and C5 fails by *truncation*. Capping the name at 1–32 chars lets a well-behaved
        /// model always close the envelope while still proving the grammar constrains structure.
        ///
        /// NOTE: this envelope is a **hand-authored test fixture**, not output of production
        /// `ToolGrammarBuilder` (which lives in ManifoldKit's `ManifoldInference` and is
        /// compile-validated separately by `LlamaToolGrammarCompileTests`). So bounding it here
        /// is the right layer; there is no production grammar-generation gap behind this failure.
        ///
        /// **Bounded whitespace (issue #20, local-sweep follow-up).** The local real-model
        /// sweep found mistral C5 still truncated even with the 32-char name cap: the
        /// `ws ::= [ \t\n]*` rule was *unbounded*, so mistral emitted verbose newline
        /// indentation (`<tool_call>\n\n    {\n        "name": …`) that, combined with a
        /// budget-filling 32-char name, exhausted `maxOutputTokens` before `</tool_call>`.
        /// Unbounded `ws` is the same truncation class as the original unbounded name/number
        /// rules, so it is bounded to `{0,4}` — enough whitespace for any well-behaved model,
        /// not enough for a degenerate one to blow the budget. qwen/llama/phi (already passing)
        /// emit minimal whitespace and are unaffected.
        static let toolCallEnvelope = #"""
        root   ::= "<tool_call>" ws obj ws "</tool_call>"
        obj    ::= "{" ws "\"name\"" ws ":" ws string ws "}"
        string ::= "\"" [a-zA-Z_]{1,32} "\""
        ws     ::= [ \t\n]{0,4}
        """#

        /// Drained-output predicate helpers.
        static func isJSONObject(_ s: String, requiredKeys: Set<String>) -> Bool {
            guard let data = s.data(using: .utf8),
                  let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            else { return false }
            return requiredKeys.isSubset(of: Set(obj.keys))
        }
    }

    // MARK: - Deterministic config

    /// Shared determinism knobs (spec §2): low temperature, fixed seed, small token cap.
    ///
    /// `maxThinkingTokens = 0` is deliberate for the C1–C5 constraint cases: on a thinking
    /// model (Qwen3) the #1595 two-chain gate keeps the grammar PERMISSIVE until `</think>`
    /// closes, so with thinking enabled and a small token budget the grammar can legitimately
    /// never engage and output stays free-form. Disabling thinking forces the single-chain
    /// path where the grammar sampler constrains every token — the behavior C1–C5 assert. The
    /// dedicated `runThinking` case re-enables thinking to test the gate itself.
    private func grammarConfig(_ grammar: String, maxTokens: Int = 48) -> GenerationConfig {
        var config = GenerationConfig(temperature: 0.1, seed: 0xC0FFEE, maxOutputTokens: maxTokens)
        config.grammar = grammar
        config.maxThinkingTokens = 0
        return config
    }

    /// Drains a stream into plain `.token` text. Thinking tokens are excluded so callers
    /// asserting on grammar-constrained *output* see only the post-`</think>` payload.
    ///
    /// Settles `isGenerating` before returning. Draining the AsyncStream to its end does NOT
    /// guarantee the backend's generation Task has run its `defer { isGenerating = false }`
    /// block — `continuation.finish()` (which ends this loop) fires from inside
    /// `LlamaGenerationDriver.run`, but the flag clears only after `run` returns. Without the
    /// settle, the suite's back-to-back `generate` calls (C1→C2→…→C5→thinking on one backend)
    /// race the flag and the next `generate` throws `.alreadyGenerating`. Mirrors the
    /// drain-then-poll pattern in `LlamaKVReuseTests`/`LlamaGrammarSamplerTests`.
    private func drainTokens(_ stream: GenerationStream, _ backend: LlamaBackend) async throws -> String {
        var text = ""
        for try await event in stream.events {
            if case .token(let chunk) = event { text += chunk }
        }
        try await waitForGeneratingFalse(backend)
        return text
    }

    /// Polls `isGenerating` until false or a 3-second deadline. See `drainTokens` for why this
    /// is required between sequential `generate` calls on a shared backend.
    private func waitForGeneratingFalse(_ backend: LlamaBackend) async throws {
        let deadline = ContinuousClock.now + .seconds(3)
        while backend.isGenerating && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertFalse(backend.isGenerating, "isGenerating must settle false before the next generate call")
    }

    // MARK: - Per-family entry points

    /// Resolves a family's GGUF, loads a backend, and returns it (with teardown registered).
    /// Throws `XCTSkip` with the standard message when no model is on disk — this is how the
    /// suite merges green in CI.
    private func loadBackend(for family: GrammarFamily) async throws -> LlamaBackend {
        // ADAPTATION vs. spec: `findGGUFModel(nameContains:)` does NOT skip when the fragment
        // is absent. Two live behaviors (verified in ManifoldTestSupport/HardwareRequirements
        // @ 0.50.0): (a) passing any non-nil `nameContains` flips local-model discovery ON even
        // without MANIFOLD_DISCOVER_LOCAL_MODELS=1 (`shouldDiscoverLocalModels` returns true for
        // a non-nil substring); (b) `selectGGUFModel` falls back to the *smallest* discovered
        // model when the fragment matches nothing. So a fragment like "mistral" with no mistral
        // on disk silently returns whatever other GGUF is smallest — a false positive that would
        // run the wrong model. We defeat both by requiring the resolved path to actually contain
        // the fragment; otherwise this family skips. This restores the spec's "matching-only,
        // skips-empty" intent and keeps the suite green in CI (no GGUFs → every family skips).
        guard let modelURL = HardwareRequirements.findGGUFModel(nameContains: family.nameFragment),
              modelURL.path.lowercased().contains(family.nameFragment.lowercased()) else {
            throw XCTSkip("No GGUF on disk for family '\(family.id)' (fragment: '\(family.nameFragment)'). "
                        + "Set LLAMA_TEST_MODEL=<path> or place a `.gguf` whose path contains '\(family.nameFragment)' "
                        + "in ~/Documents/Models/ (with MANIFOLD_DISCOVER_LOCAL_MODELS=1) to run this family.")
        }
        let backend = LlamaBackend()
        addTeardownBlock { await backend.unloadAndWait() }
        try await backend.loadModel(from: modelURL, plan: .testStub(effectiveContextSize: 512))
        return backend
    }

    // MARK: - Family test methods (one per family; each is independent + skips-empty)

    func test_conformance_llama()   async throws { try await runFamily(family(for: "llama")) }
    func test_conformance_qwen()    async throws { try await runFamily(family(for: "qwen")) }
    func test_conformance_mistral() async throws { try await runFamily(family(for: "mistral")) }
    func test_conformance_phi()     async throws { try await runFamily(family(for: "phi")) }

    /// Gemma carve-out: assert grammar is gated OFF after a real load. Distinct entry
    /// point because the assertion shape differs from the grammar-supporting families.
    ///
    /// Sabotage check: remove `.lowercased().hasPrefix("gemma")` from the detection logic in
    /// `LlamaBackend.capabilities` — a loaded Gemma model would advertise grammar support and
    /// the `supportsGrammarConstrainedSampling == false` assertion below would fail.
    func test_conformance_gemma_carveOut() async throws {
        try await runGemmaCarveOut(family(for: "gemma"))
    }

    private func family(for id: String) -> GrammarFamily {
        // Force-unwrap: ids are compile-time constants matching the static table.
        Self.families.first { $0.id == id }!
    }

    // MARK: - Grammar-supporting family runner (C1–C5 + thinking)

    /// Runs the full grammar battery for a grammar-supporting family. Skips-empty when no
    /// model is on disk. C1–C5 each carry their own sabotage rationale below.
    private func runFamily(_ family: GrammarFamily) async throws {
        precondition(family.expectsGrammarSupport, "runFamily is only for grammar-supporting families")
        let backend = try await loadBackend(for: family)

        // A grammar-supporting family must advertise the capability after a real load.
        // Sabotage check: hardcode `supportsGrammar = false` in `LlamaBackend.capabilities` — fails here.
        XCTAssertTrue(backend.capabilities.supportsGrammarConstrainedSampling,
                      "[\(family.id)] non-Gemma family must report supportsGrammarConstrainedSampling == true after load")

        try await runC1(family, backend)
        try await runC2(family, backend)
        try await runC3(family, backend)
        try await runC4(family, backend)
        try await runC5(family, backend)

        if family.isThinkingCapable {
            try await runThinking(family, backend)
        }
    }

    /// C1 smoke: `root ::= [0-9]+` constrains output to digits only.
    ///
    /// Sabotage check: remove the grammar-sampler insertion block from
    /// `LlamaGenerationDriver.run` — the model emits non-digit tokens and `allSatisfy(isNumber)` fails.
    private func runC1(_ family: GrammarFamily, _ backend: LlamaBackend) async throws {
        let stream = try backend.generate(prompt: "Give me a random number.", systemPrompt: nil,
                                          config: grammarConfig(GrammarFixtures.digits, maxTokens: 16))
        let text = try await drainTokens(stream, backend)
        XCTAssertFalse(text.isEmpty, "[\(family.id)] C1: grammar-constrained generation must emit at least one token")
        XCTAssertTrue(text.allSatisfy { $0.isNumber },
                      "[\(family.id)] C1: 'root ::= [0-9]+' must constrain output to digits; got \(text.debugDescription)")
    }

    /// C2 json-object: output parses as a JSON object carrying the required keys. This is the
    /// Gemma stall class — for supporting families it must NOT stall.
    ///
    /// Sabotage check: corrupt the GBNF so the object never closes (drop the trailing `"}"`),
    /// or disable the grammar sampler — output stops being a valid keyed JSON object and the
    /// `isJSONObject` assertion fails.
    private func runC2(_ family: GrammarFamily, _ backend: LlamaBackend) async throws {
        let stream = try backend.generate(prompt: "Report the weather as JSON.", systemPrompt: nil,
                                          config: grammarConfig(GrammarFixtures.jsonObject, maxTokens: 64))
        let text = try await drainTokens(stream, backend)
        XCTAssertTrue(GrammarFixtures.isJSONObject(text, requiredKeys: GrammarFixtures.jsonObjectRequiredKeys),
                      "[\(family.id)] C2: grammar must yield a JSON object with keys "
                    + "\(GrammarFixtures.jsonObjectRequiredKeys.sorted()); got \(text.debugDescription)")
    }

    /// C3 alternation: `root ::= "yes" | "no"` constrains output to one of the two literals.
    ///
    /// Sabotage check: replace the grammar with `root ::= [a-z]+` (or disable the sampler) —
    /// the model emits free text and the membership assertion fails.
    private func runC3(_ family: GrammarFamily, _ backend: LlamaBackend) async throws {
        let stream = try backend.generate(prompt: "Answer yes or no: is the sky blue?", systemPrompt: nil,
                                          config: grammarConfig(GrammarFixtures.alternation, maxTokens: 8))
        let text = try await drainTokens(stream, backend).trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertTrue(["yes", "no"].contains(text),
                      "[\(family.id)] C3: alternation must constrain output to {yes,no}; got \(text.debugDescription)")
    }

    /// C4 leading-space: the same alternation grammar built with and without a leading
    /// optional-space prefix. BOTH variants must stay schema-valid (output ∈ {yes,no} after
    /// trimming). Branch agreement vs. divergence is LOGGED, never failed — leading-space
    /// sensitivity is model-intrinsic (tokenizer boundary, "Lost in Space"). A crash or an
    /// out-of-grammar result under either variant DOES fail.
    ///
    /// Sabotage check: break either variant's GBNF (e.g. make `alternationLeadingSpace` accept
    /// `[a-z]+`) — that variant's `{yes,no}` membership assertion fails.
    private func runC4(_ family: GrammarFamily, _ backend: LlamaBackend) async throws {
        let prompt = "Answer yes or no: is fire cold?"

        let plain = try await drainTokens(
            try backend.generate(prompt: prompt, systemPrompt: nil,
                                 config: grammarConfig(GrammarFixtures.alternationNoLeadingSpace, maxTokens: 8)),
            backend
        ).trimmingCharacters(in: .whitespacesAndNewlines)

        let spaced = try await drainTokens(
            try backend.generate(prompt: prompt, systemPrompt: nil,
                                 config: grammarConfig(GrammarFixtures.alternationLeadingSpace, maxTokens: 8)),
            backend
        ).trimmingCharacters(in: .whitespacesAndNewlines)

        XCTAssertTrue(["yes", "no"].contains(plain),
                      "[\(family.id)] C4: no-leading-space variant must stay valid; got \(plain.debugDescription)")
        XCTAssertTrue(["yes", "no"].contains(spaced),
                      "[\(family.id)] C4: leading-space variant must stay valid; got \(spaced.debugDescription)")

        // Behavioral characteristic — recorded, not asserted.
        if plain == spaced {
            print("[grammar-conformance][\(family.id)] C4 branch AGREEMENT: both variants → \(plain.debugDescription)")
        } else {
            print("[grammar-conformance][\(family.id)] C4 branch DIVERGENCE (model-intrinsic): "
                + "no-space → \(plain.debugDescription), with-space → \(spaced.debugDescription)")
        }
    }

    /// C5 tool-envelope (SCOPED). Constrains output to a literal
    /// `<tool_call>{"name": ...}</tool_call>` envelope and asserts the inner object parses as
    /// JSON with a `name` key. We do NOT drive the full `ToolCallTransform` here: its body
    /// parser (`LlamaToolMarkers.parseCallBuffer`) is `private` to ManifoldLlama, and forcing
    /// the streaming transform from a grammar test adds coupling without strengthening the
    /// grammar guarantee. The end-to-end transform path is covered by `LlamaToolCallParserTests`.
    ///
    /// Sabotage check: drop the `<tool_call>`…`</tool_call>` literals from the grammar (or
    /// disable the sampler) — the output stops matching the envelope and JSON-with-`name`
    /// extraction fails.
    private func runC5(_ family: GrammarFamily, _ backend: LlamaBackend) async throws {
        // 64-token cap (was 48): even when a model fills the now-bounded 32-char name to its
        // limit, the literal `<tool_call>…</tool_call>` wrapper plus quoting needs headroom to
        // emit the closing marker within budget. See `toolCallEnvelope` re issue #20.
        let stream = try backend.generate(prompt: "Call the get_weather tool.", systemPrompt: nil,
                                          config: grammarConfig(GrammarFixtures.toolCallEnvelope, maxTokens: 64))
        let text = try await drainTokens(stream, backend).trimmingCharacters(in: .whitespacesAndNewlines)

        guard let open = text.range(of: "<tool_call>"),
              let close = text.range(of: "</tool_call>") else {
            XCTFail("[\(family.id)] C5: grammar-constrained output must contain a <tool_call>…</tool_call> envelope; "
                  + "got \(text.debugDescription)")
            return
        }
        let body = String(text[open.upperBound..<close.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = body.data(using: .utf8),
              let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            XCTFail("[\(family.id)] C5: tool-call envelope body must be a JSON object; got \(body.debugDescription)")
            return
        }
        XCTAssertNotNil(obj["name"],
                        "[\(family.id)] C5: tool-call JSON must carry a 'name' key; got \(body.debugDescription)")
    }

    /// Thinking-capable runner (Qwen3): with thinking enabled, the #1595 two-chain gate keeps
    /// the grammar PERMISSIVE during the reasoning phase and flips it STRICT on
    /// `.thinkingCompleted`. The behavioral guarantees this asserts:
    ///   - when reasoning surfaces, it arrives as `.thinkingToken` (never as grammar-pruned
    ///     plain `.token`), and
    ///   - the post-`</think>` plain output satisfies the grammar (gate went strict).
    ///
    /// Whether the model actually OPENS a `<think>` block for a given prompt within the token
    /// budget — and whether marker auto-detection from the GGUF surfaced any markers at all —
    /// is model/prompt-intrinsic, NOT a grammar-conformance guarantee. So "no reasoning surfaced"
    /// is LOGGED as a characteristic, not failed (mirrors the C4 divergence philosophy). The
    /// hard assertion is the conditional one: IF thinking completed, the constrained tail must
    /// be grammar-valid.
    ///
    /// Sabotage check: force the grammar gate strict during the thinking phase (apply the
    /// grammar sampler even when a thinking parser is active in `LlamaGenerationDriver`) — the
    /// reasoning prose gets grammar-pruned and either surfaces as grammar-mangled `.token`
    /// text or the post-think tail stops satisfying the grammar.
    private func runThinking(_ family: GrammarFamily, _ backend: LlamaBackend) async throws {
        // Read the per-model flag, NOT `capabilities.supportsThinking`: the latter is hardcoded
        // `true` on `LlamaBackend.capabilities` (it advertises the family's *potential*), whereas
        // `manifest.supportsThinking` reflects whether THIS loaded GGUF auto-detected thinking
        // markers from its chat template (`LlamaBackend.loadModel` sets it to
        // `autoDetectedThinkingMarkers != nil`). A "qwen"-matched GGUF with no thinking markers
        // must degrade to informational, not run the phase-gate sub-case against a model that
        // can never open `<think>`.
        guard backend.manifest?.supportsThinking == true else {
            print("[grammar-conformance][\(family.id)] thinking: loaded GGUF advertised no thinking "
                + "markers (manifest.supportsThinking == false) — thinking sub-case is informational-only")
            return
        }
        var config = grammarConfig(GrammarFixtures.alternation, maxTokens: 96)
        // Re-enable thinking (grammarConfig disables it for the C1–C5 single-chain cases).
        config.maxThinkingTokens = 64

        let stream = try backend.generate(prompt: "Think step by step, then answer with exactly yes or no: is 2 a prime number?",
                                          systemPrompt: nil, config: config)

        var thinkingText = ""
        var sawThinkingCompleted = false
        var postThinkOutput = ""
        for try await event in stream.events {
            switch event {
            case .thinkingToken(let t):     thinkingText += t
            case .thinkingCompleted:        sawThinkingCompleted = true
            case .token(let t):             postThinkOutput += t
            default:                        break
            }
        }
        try await waitForGeneratingFalse(backend)

        if sawThinkingCompleted {
            // Gate flipped strict: the constrained tail MUST satisfy the grammar.
            let trimmed = postThinkOutput.trimmingCharacters(in: .whitespacesAndNewlines)
            XCTAssertTrue(["yes", "no"].contains(trimmed),
                          "[\(family.id)] thinking: post-</think> output must satisfy the grammar (gate strict); got \(trimmed.debugDescription)")
            print("[grammar-conformance][\(family.id)] thinking: completed — reasoning \(thinkingText.count) chars, "
                + "constrained tail \(trimmed.debugDescription)")
        } else if !thinkingText.isEmpty {
            // Reasoning surfaced as .thinkingToken (correct routing) but never closed within
            // budget — permissive phase confirmed, strict tail not reachable. Behavioral, not a failure.
            print("[grammar-conformance][\(family.id)] thinking: reasoning surfaced as .thinkingToken "
                + "(\(thinkingText.count) chars) but </think> did not close within \(config.maxOutputTokens ?? -1) tokens "
                + "— permissive phase confirmed, strict-tail assertion not reachable")
        } else {
            // No reasoning at all: model didn't open <think> for this prompt, or no markers were
            // auto-detected from the GGUF. Model/prompt-intrinsic — logged, not failed.
            print("[grammar-conformance][\(family.id)] thinking: no .thinkingToken surfaced for this prompt "
                + "(model did not open <think>, or GGUF marker auto-detection found none) — informational")
        }
    }

    // MARK: - Gemma carve-out runner

    /// Asserts the grammar carve-out holds end-to-end for a non-supporting family: after a
    /// real load, `supportsGrammarConstrainedSampling == false`.
    ///
    /// NOTE (adaptation vs. spec §1.3): the spec suggested asserting a thrown
    /// `InferenceError.unsupportedGrammar` when a grammar reaches a Gemma backend. The live
    /// `LlamaBackend` does NOT throw — `LlamaGenerationDriver.run` applies the grammar sampler
    /// whenever `config.grammar != nil`, and the capability flag is advisory for *callers*.
    /// Fabricating a throw expectation would make the test wrong, so the conformance assertion
    /// is the capability flag alone. If a future change wires a contract throw into the
    /// backend, tighten this to assert the throw.
    ///
    /// Sabotage check: remove `.lowercased().hasPrefix("gemma")` from `LlamaBackend.capabilities`
    /// — a loaded Gemma model advertises grammar support and the `== false` assertion fails.
    private func runGemmaCarveOut(_ family: GrammarFamily) async throws {
        precondition(!family.expectsGrammarSupport, "runGemmaCarveOut is only for grammar-gated families")
        let backend = try await loadBackend(for: family)
        XCTAssertFalse(backend.capabilities.supportsGrammarConstrainedSampling,
                       "[\(family.id)] grammar-gated family must report supportsGrammarConstrainedSampling == false "
                     + "after a real load — Gemma truncates under structured GBNF")
    }
}
