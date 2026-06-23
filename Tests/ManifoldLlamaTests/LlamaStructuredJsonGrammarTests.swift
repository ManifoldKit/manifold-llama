import XCTest
import Foundation
import ManifoldInference
import ManifoldTools
import ManifoldLlama
@_spi(Testing) import ManifoldLlama

/// Lever 2 of #100 — grammar-constrained decoding for structured-json
/// extraction scenarios.
///
/// The soak found that models producing correct key/value pairs still fail the
/// `containsAll` assertion because they wrap the JSON in markdown fences or
/// prose. A GBNF grammar on the final-answer turn eliminates that: the sampler
/// can only emit tokens that extend a valid JSON object matching the extraction
/// schema, so fences and prose are structurally impossible.
///
/// The activation logic lives in the executable target (`main.swift`) and
/// cannot be imported, so this suite mirrors the same rules locally — matching
/// the approach in `LlamaResultGroundingPromptTests`. A drift in the id prefix
/// guard or the Gemma carve-out will break these tests, which is the guard we
/// want around those coupling points.
///
/// No model is loaded — pure data/string assertions that run in CI.
final class LlamaStructuredJsonGrammarTests: XCTestCase {

    // MARK: - Mirror of harness logic

    /// Mirror of `structuredJsonExtractionSchema` in `main.swift`.
    private static let extractionSchema: JSONSchemaValue = .object([
        "type": .string("object"),
        "properties": .object([
            "invoice_id": .object(["type": .string("string")]),
            "total": .object(["type": .string("string")]),
            "currency": .object(["type": .string("string")])
        ]),
        "required": .array([.string("invoice_id"), .string("total"), .string("currency")])
    ])

    /// Mirror of `grammarForScenario(_:backend:)` in `main.swift`.
    ///
    /// Kept in lockstep: if the id-prefix guard or the capability check changes
    /// in the harness, update this too — the duplication is deliberate and makes
    /// the coupling visible.
    private static func grammarForScenario(_ scenario: Scenario, backend: LlamaBackend) -> String? {
        guard scenario.requiredTools.isEmpty,
              scenario.id.hasPrefix("structured-json")
        else { return nil }
        guard backend.capabilities.supportsGrammarConstrainedSampling else { return nil }
        return ToolGrammarBuilder().buildObjectGrammar(for: extractionSchema)
    }

    // MARK: - Helpers

    /// Resolves the vendored scenario JSON directory from this file's source
    /// location, matching `LlamaResultGroundingPromptTests`.
    private func scenarioDirectory() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // ManifoldLlamaTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // package root
            .appendingPathComponent("Sources/manifold-tools-llama/Scenarios", isDirectory: true)
    }

    /// Builds a `LlamaBackend` with an injected architecture so capability
    /// flags reflect a specific model family without loading a GGUF.
    private func backend(architecture: String) -> LlamaBackend {
        let b = LlamaBackend()
        b.injectArchitectureForTesting(architecture)
        return b
    }

    // MARK: - 1. Grammar is emitted for the structured-json extraction scenario

    /// `grammarForScenario` must return a non-nil grammar string for the
    /// `structured-json-extraction` scenario on a grammar-capable (non-Gemma)
    /// backend, and the grammar must begin with the GBNF root rule.
    func test_grammarForScenario_structuredJsonExtraction_returnsGrammar() throws {
        let scenarios = try ScenarioLoader.load(from: scenarioDirectory())
        let target = try XCTUnwrap(
            scenarios.first(where: { $0.id == "structured-json-extraction" }),
            "structured-json-extraction scenario not found in Scenarios/")

        let g = try XCTUnwrap(
            Self.grammarForScenario(target, backend: backend(architecture: "llama")),
            "grammarForScenario must return a grammar for structured-json-extraction on a llama backend")

        XCTAssertTrue(g.hasPrefix("root ::="),
            "GBNF grammar must open with the root rule; got:\n\(g.prefix(120))")
    }

    // MARK: - 2. Grammar is withheld for Gemma (grammar-gated family)

    /// Gemma truncates under structured JSON grammars.
    /// `LlamaBackend.capabilities` reports `supportsGrammarConstrainedSampling
    /// == false` for the family. `grammarForScenario` must return `nil` rather
    /// than shipping a grammar that would silence the model.
    func test_grammarForScenario_gemmaBackend_returnsNil() throws {
        let scenarios = try ScenarioLoader.load(from: scenarioDirectory())
        let target = try XCTUnwrap(
            scenarios.first(where: { $0.id == "structured-json-extraction" }),
            "structured-json-extraction scenario not found")

        let gemmaBackend = backend(architecture: "gemma3")
        XCTAssertFalse(gemmaBackend.capabilities.supportsGrammarConstrainedSampling,
            "test precondition: Gemma backend must report grammar unsupported")

        XCTAssertNil(Self.grammarForScenario(target, backend: gemmaBackend),
            "grammarForScenario must return nil for a Gemma backend — the family truncates under structured GBNF grammars")
    }

    // MARK: - 3. Grammar is withheld for tool-using scenarios

    /// Tool-call turn grammar must NOT be constrained by the extraction grammar —
    /// that would mask the tool-dispatch envelope. `grammarForScenario` must
    /// return `nil` for any scenario with `requiredTools` populated.
    func test_grammarForScenario_toolUsingScenarios_returnNil() throws {
        let scenarios = try ScenarioLoader.load(from: scenarioDirectory())
        let toolUsing = scenarios.filter { !$0.requiredTools.isEmpty }
        XCTAssertFalse(toolUsing.isEmpty,
            "test precondition: expected tool-using scenarios to be present")

        let llamaBackend = backend(architecture: "llama")
        for scenario in toolUsing {
            XCTAssertNil(Self.grammarForScenario(scenario, backend: llamaBackend),
                "grammarForScenario must return nil for tool-using scenario '\(scenario.id)' — tool dispatch must remain unconstrained")
        }
    }

    // MARK: - 4. Grammar is withheld for non-structured-json toolless scenarios

    /// A future toolless scenario that is NOT a structured-json extraction must
    /// not receive the extraction grammar.
    func test_grammarForScenario_nonStructuredJsonId_returnsNil() throws {
        let data = """
        {
          "id": "plain-text-generation",
          "description": "A toolless scenario that should not receive a grammar",
          "systemPrompt": "Be helpful.",
          "userPrompt": "Say hello.",
          "requiredTools": [],
          "assertions": [],
          "backend": { "kind": "mock", "model": "mock", "temperature": 0.0 }
        }
        """.data(using: .utf8)!
        let scenario = try JSONDecoder().decode(Scenario.self, from: data)

        let llamaBackend = backend(architecture: "llama")
        XCTAssertNil(Self.grammarForScenario(scenario, backend: llamaBackend),
            "grammarForScenario must return nil for a toolless scenario whose id does not start with 'structured-json'")
    }

    // MARK: - 5. Extraction schema grammar constrains to the right three keys

    /// Sanity-check that `buildObjectGrammar` on the extraction schema emits a
    /// grammar that references the three required keys. Not a byte-golden (those
    /// live in `ToolGrammarBuilderTests` in ManifoldKit) — structural only.
    func test_extractionSchema_grammarContainsRequiredKeys() throws {
        let grammar = try XCTUnwrap(
            ToolGrammarBuilder().buildObjectGrammar(for: Self.extractionSchema),
            "buildObjectGrammar must return a grammar for the extraction schema")

        for key in ["invoice_id", "total", "currency"] {
            XCTAssertTrue(grammar.contains(key),
                "extraction schema grammar must reference key '\(key)'; got:\n\(grammar.prefix(300))")
        }
    }
}
