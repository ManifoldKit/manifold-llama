import XCTest
import Foundation
import ManifoldInference
import ManifoldTestSupport
import ManifoldLlama
@_spi(Testing) import ManifoldLlama

/// COMPILE-VALIDATION for `ToolGrammarBuilder`'s emitted GBNF against the *real*
/// vendored llama.cpp parser.
///
/// ## Why this exists (the structural gap)
///
/// `ToolGrammarBuilder` (ManifoldKit `ManifoldInference`, #1859) lowers a tool
/// list into a GBNF grammar. Its own unit suite (`ToolGrammarBuilderTests`) pins
/// the emitted grammar **byte-for-byte** — and says so explicitly: *"No live GBNF
/// compiler in CI, so pin the exact bytes."* That is the gap: nothing actually
/// feeds the emitted grammar to `llama_sampler_init_grammar`. A 100%-broken
/// grammar (the underscore-rule-name class — `args_0` misparsed as `args`
/// followed by a syntax error, because llama.cpp's `is_word_char` accepts only
/// `[a-zA-Z0-9-]` in rule names) passes every byte-golden while being rejected by
/// the real parser at runtime. That near-miss is exactly why this test exists:
/// it compiles the builder's output against the vendored llama.cpp this package
/// links, closing the "no CI job compiles GBNF" hole.
///
/// ## How the compile signal works
///
/// An invalid grammar makes `llama_sampler_init_grammar` return `nil`, which
/// `LlamaGenerationDriver` turns into `setPhase(.failed("Failed to parse GBNF
/// grammar"))` → the `generate` stream THROWS
/// `InferenceError.inferenceFailure("Failed to parse GBNF grammar string")`. So:
///   - a **valid** grammar generates (≥1 token, output matches the envelope);
///   - an **invalid** grammar THROWS the grammar-parse failure through the stream.
/// This test distinguishes on that signal — it is the live proof the byte-goldens
/// cannot give.
///
/// ## Skips-empty by design
///
/// Resolves a GGUF on disk by name fragment via `HardwareRequirements.findGGUFModel`,
/// with the same path-contains guard the sibling conformance suites use (defeats
/// `findGGUFModel`'s "flip discovery on + fall back to smallest model" false
/// positive). With no model on disk (the CI default) every test throws `XCTSkip`
/// and the suite merges green. On a box WITH a model it RUNS and PASSES — the real
/// proof the shipped grammar compiles against vendored llama.cpp.
///
/// Hardware-gated (Apple Silicon + physical device) — `LlamaBackend` requires
/// Metal, unavailable in the simulator.
final class LlamaToolGrammarCompileTests: XCTestCase {

    private let builder = ToolGrammarBuilder()

    override func setUp() async throws {
        try await super.setUp()
        try XCTSkipUnless(HardwareRequirements.isPhysicalDevice,
                          "LlamaBackend requires Metal (unavailable in simulator)")
        try XCTSkipUnless(HardwareRequirements.isAppleSilicon,
                          "LlamaBackend requires Apple Silicon")
    }

    // MARK: - Schema helpers (mirror ManifoldKit ToolGrammarBuilderTests idiom)

    /// Builds a JSON-Schema `object` document expressed as `JSONSchemaValue`
    /// (the shape `ToolDefinition.parameters` carries). Required keys in declared
    /// order; `properties` keyed by name.
    private func objectSchema(_ props: [(String, JSONSchemaValue)], required: [String]) -> JSONSchemaValue {
        var properties: [String: JSONSchemaValue] = [:]
        for (k, v) in props { properties[k] = v }
        return .object([
            "type": .string("object"),
            "properties": .object(properties),
            "required": .array(required.map { .string($0) })
        ])
    }

    // MARK: - Discovery (matches LlamaGrammarConformanceTests.loadBackend)

    /// Resolves any grammar-capable GGUF on disk and loads a backend (teardown
    /// registered). Throws `XCTSkip` (standard message) when none is present — how
    /// the suite merges green in CI.
    ///
    /// The `path.lowercased().contains(fragment)` guard is load-bearing:
    /// `findGGUFModel` (a) flips local-model discovery ON for any non-nil fragment
    /// and (b) falls back to the *smallest* discovered GGUF when the fragment
    /// matches nothing — so without the guard a box holding only a non-matching
    /// model would run that wrong model instead of skipping. We accept any of a
    /// few common grammar-capable families (this suite is grammar-shape-agnostic;
    /// it only needs a model that applies a grammar sampler).
    private func loadGrammarCapableBackend() async throws -> LlamaBackend {
        let fragments = ["qwen", "llama", "mistral", "phi"]
        var resolved: URL?
        for fragment in fragments {
            if let url = HardwareRequirements.findGGUFModel(nameContains: fragment),
               url.path.lowercased().contains(fragment) {
                resolved = url
                break
            }
        }
        guard let modelURL = resolved else {
            throw XCTSkip("No grammar-capable GGUF on disk (looked for fragments \(fragments)). "
                        + "Set LLAMA_TEST_MODEL=<path> or place a `.gguf` whose path contains one of "
                        + "\(fragments) in ~/Documents/Models/ (with MANIFOLD_DISCOVER_LOCAL_MODELS=1) "
                        + "to run this suite.")
        }
        let backend = LlamaBackend()
        addTeardownBlock { await backend.unloadAndWait() }
        try await backend.loadModel(from: modelURL, plan: .testStub(effectiveContextSize: 512))

        // A grammar-gated family (Gemma) would never constrain — skip rather than
        // exercise the compile path against a backend that ignores the grammar.
        try XCTSkipUnless(backend.capabilities.supportsGrammarConstrainedSampling,
                          "Resolved GGUF does not advertise grammar-constrained sampling "
                        + "(grammar-gated family); cannot compile-validate the tool grammar against it.")
        return backend
    }

    /// Deterministic config carrying a tool grammar. `maxThinkingTokens = 0` is
    /// load-bearing (mirrors the conformance suite): on a thinking-capable model the
    /// #1595 two-chain gate keeps the grammar PERMISSIVE until `</think>` closes, so
    /// the grammar would never constrain within a small budget. Disabling thinking
    /// forces the single strict chain where the grammar sampler is applied from
    /// token 0 — and, crucially, where an UN-COMPILABLE grammar throws the parse
    /// failure (the signal this suite reads).
    private func grammarConfig(_ grammar: String, tools: [ToolDefinition], maxTokens: Int = 96) -> GenerationConfig {
        var config = GenerationConfig(temperature: 0.1, seed: 0xC0FFEE, maxOutputTokens: maxTokens)
        config.grammar = grammar
        config.tools = tools
        config.maxThinkingTokens = 0
        return config
    }

    // MARK: - Stream draining + settle (mirrors sibling suites)

    /// Drains a stream into plain `.token` text. Rethrows any error — for this suite
    /// a thrown `InferenceError.inferenceFailure("Failed to parse GBNF grammar string")`
    /// is the *failure* signal we explicitly do NOT want, so callers let it propagate
    /// and the test fails with the grammar-parse message.
    ///
    /// Settles `isGenerating` before returning: draining the AsyncStream to its end
    /// does not guarantee the backend's `defer { isGenerating = false }` has run
    /// (`continuation.finish()` ends the loop from inside `LlamaGenerationDriver.run`,
    /// while the flag clears only after `run` returns). Mirrors `LlamaKVReuseTests`.
    private func drainTokens(_ stream: GenerationStream, _ backend: LlamaBackend) async throws -> String {
        var text = ""
        for try await event in stream.events {
            if case .token(let chunk) = event { text += chunk }
        }
        try await waitForGeneratingFalse(backend)
        return text
    }

    private func waitForGeneratingFalse(_ backend: LlamaBackend) async throws {
        let deadline = ContinuousClock.now + .seconds(3)
        while backend.isGenerating && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertFalse(backend.isGenerating, "isGenerating must settle false before the next generate call")
    }

    /// Extracts the JSON object from grammar-constrained tool-call output. The
    /// builder's envelope is the bare object `{"name": ..., "arguments": {...}}`
    /// (no `<tool_call>` markers — those belong to the streaming transform, not the
    /// grammar). Returns the parsed dictionary or fails with the raw text.
    private func parseEnvelope(_ text: String, _ label: String) -> [String: Any]? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = trimmed.data(using: .utf8),
              let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            XCTFail("\(label): grammar-constrained output must parse as a JSON object; got \(trimmed.debugDescription)")
            return nil
        }
        return obj
    }

    // MARK: - 1. Compile + constrain (core)

    /// Builds a non-trivial tool (object with a required typed field, a
    /// string-enum field, and an array field — exercising the per-param lowerer),
    /// compiles its grammar via `ToolGrammarBuilder`, applies it to a real model,
    /// and asserts (a) generation does NOT throw the grammar-parse failure — this
    /// catches the underscore class of bug — (b) ≥1 token is produced, and (c) the
    /// output parses as the `{"name", "arguments"}` envelope with the tool's name
    /// and an object `arguments`.
    ///
    /// Sabotage check: revert `ToolGrammarBuilder`'s hyphenated rule names to
    /// underscores (`args-0` → `args_0`) → `llama_sampler_init_grammar` returns nil
    /// → `LlamaGenerationDriver` sets `.failed("Failed to parse GBNF grammar")` →
    /// `generate` throws `InferenceError.inferenceFailure("Failed to parse GBNF
    /// grammar string")` → `drainTokens` rethrows and this test fails with the
    /// grammar-parse message.
    func test_compile_singleNonTrivialTool_constrainsEnvelope() async throws {
        let backend = try await loadGrammarCapableBackend()

        let tool = ToolDefinition(
            name: "get_forecast",
            description: "Get the weather forecast for a city.",
            parameters: objectSchema([
                ("city", .object(["type": .string("string")])),
                ("unit", .object([
                    "type": .string("string"),
                    "enum": .array([.string("celsius"), .string("fahrenheit")])
                ])),
                ("days", .object([
                    "type": .string("array"),
                    "items": .object(["type": .string("integer")])
                ]))
            ], required: ["city", "unit"])
        )

        let grammar = try XCTUnwrap(builder.buildGrammar(for: [tool]),
                                    "ToolGrammarBuilder must emit a grammar for a non-empty tool list")

        let stream = try backend.generate(
            prompt: "Call the get_forecast tool for Paris in celsius.",
            systemPrompt: nil,
            config: grammarConfig(grammar, tools: [tool]))

        // (a) MUST NOT throw the grammar-parse failure — drainTokens rethrows it.
        let text = try await drainTokens(stream, backend)

        // (b) at least one token.
        XCTAssertFalse(text.isEmpty,
                       "compiled tool grammar must emit at least one constrained token; got empty output")

        // (c) parses as the discriminated-union envelope.
        guard let obj = parseEnvelope(text, "single-tool") else { return }
        XCTAssertEqual(obj["name"] as? String, "get_forecast",
                       "envelope 'name' must pin to the tool name; got \(String(describing: obj["name"]))")
        XCTAssertTrue(obj["arguments"] is [String: Any],
                      "envelope 'arguments' must be a JSON object; got \(String(describing: obj["arguments"]))")
    }

    // MARK: - 2. Multi-tool union compiles

    /// Two tools with different schemas → `root ::= toolcall-0 | toolcall-1`.
    /// Asserts the union grammar compiles (no grammar-parse throw) and the output
    /// names ONE of the two tools.
    ///
    /// Sabotage check: same underscore-rule-name revert as test 1 — the
    /// multi-branch union fails to parse and `generate` throws the grammar-parse
    /// failure, failing here.
    func test_compile_multiToolUnion_constrainsToOneTool() async throws {
        let backend = try await loadGrammarCapableBackend()

        let weather = ToolDefinition(
            name: "get_weather",
            description: "Get current weather.",
            parameters: objectSchema([
                ("city", .object(["type": .string("string")]))
            ], required: ["city"]))

        let setTimer = ToolDefinition(
            name: "set_timer",
            description: "Set a countdown timer.",
            parameters: objectSchema([
                ("seconds", .object(["type": .string("integer")])),
                ("label", .object(["type": .string("string")]))
            ], required: ["seconds"]))

        let tools = [weather, setTimer]
        let grammar = try XCTUnwrap(builder.buildGrammar(for: tools),
                                    "ToolGrammarBuilder must emit a union grammar for two tools")

        let stream = try backend.generate(
            prompt: "Set a 60 second timer.",
            systemPrompt: nil,
            config: grammarConfig(grammar, tools: tools))

        let text = try await drainTokens(stream, backend)
        XCTAssertFalse(text.isEmpty, "compiled union grammar must emit at least one token")

        guard let obj = parseEnvelope(text, "multi-tool") else { return }
        let name = obj["name"] as? String
        XCTAssertTrue(["get_weather", "set_timer"].contains(name ?? ""),
                      "union envelope 'name' must be one of the two tools; got \(String(describing: name))")
    }

    // MARK: - 3. Fallback (generic `value`) tool compiles

    /// A tool whose `arguments` schema uses an unmodeled combiner (`anyOf`) so the
    /// lowerer degrades that branch's `args-N` to the shared generic `value` rule.
    /// Asserts the grammar (envelope + generic tail) STILL compiles against real
    /// llama.cpp and constrains the envelope — graceful degradation must not emit a
    /// grammar the parser rejects.
    ///
    /// Sabotage check: same underscore-rule-name revert — the generic-rule block
    /// (`value`/`object`/`array`/…) and the `args-0` reference fail to parse and
    /// `generate` throws the grammar-parse failure, failing here.
    func test_compile_fallbackGenericValueTool_constrainsEnvelope() async throws {
        let backend = try await loadGrammarCapableBackend()

        // `anyOf` at the schema root is unmodeled → whole-tool args-0 degrades to
        // the shared generic `value` rule (still emitted, still referenced).
        let tool = ToolDefinition(
            name: "freeform",
            description: "A tool with free-form arguments.",
            parameters: .object([
                "anyOf": .array([
                    .object(["type": .string("string")]),
                    .object(["type": .string("object")])
                ])
            ]))

        let grammar = try XCTUnwrap(builder.buildGrammar(for: [tool]),
                                    "ToolGrammarBuilder must emit a grammar even for a fallback (generic value) tool")

        let stream = try backend.generate(
            prompt: "Call the freeform tool with any arguments.",
            systemPrompt: nil,
            config: grammarConfig(grammar, tools: [tool]))

        let text = try await drainTokens(stream, backend)
        XCTAssertFalse(text.isEmpty, "compiled fallback grammar must emit at least one token")

        guard let obj = parseEnvelope(text, "fallback-tool") else { return }
        XCTAssertEqual(obj["name"] as? String, "freeform",
                       "fallback envelope 'name' must still pin to the tool name; got \(String(describing: obj["name"]))")
        // `arguments` for a generic-value fallback can be any JSON value — only the
        // envelope + name are guaranteed constrained. Presence is the assertion.
        XCTAssertNotNil(obj["arguments"],
                        "fallback envelope must still carry an 'arguments' key; got \(obj.keys.sorted())")
    }
}
