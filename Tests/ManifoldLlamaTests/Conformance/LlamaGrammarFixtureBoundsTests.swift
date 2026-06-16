import XCTest
import Foundation

/// HEADLESS structural guard for the C2/C5 grammar fixtures bounded under issue #20.
///
/// ## Why this exists
///
/// Issue #20: `LlamaGrammarConformanceTests` C2 (JSON object) and C5 (tool-call
/// envelope) failed by *truncation* on small models (mistral/qwen 7B) — the grammar
/// WAS followed, but unbounded repetition rules (`number ::= "-"? [0-9]+ ("." [0-9]+)?`,
/// `string ::= "\"" ([a-zA-Z_]+) "\""`) let a degenerate model keep emitting digits /
/// name characters and never reach the closing delimiter within `maxOutputTokens`.
/// PR #23 bounded those fixture rules so a well-behaved model can always close.
///
/// The conformance suite that owns those fixtures is **model-gated** (Apple Silicon +
/// a GGUF on disk) — in CI it XCTSkips, so a regression that *removes* the bounds would
/// merge green there and only resurface on the next model-bearing sweep. This suite is
/// the missing **headless tripwire**: it asserts the bounds (and the parser-safety
/// invariants llama.cpp's GBNF parser requires) directly on the fixture strings, with
/// no model and no Metal, so it runs on every CI lane.
///
/// ## Scope
///
/// This is a *structural* guard on the fixture text — NOT a compile against vendored
/// llama.cpp. The live compile-against-the-real-parser proof lives in
/// `LlamaToolGrammarCompileTests` (model-gated) and the behavioral close-within-budget
/// proof in `LlamaGrammarConformanceTests` C2/C5 (model-gated). Production grammar
/// generation is `ToolGrammarBuilder` in ManifoldKit core (`ManifoldInference`), not in
/// this package — its envelope pins `name` to each tool's *literal* name per branch, so
/// the C5 unbounded-name degeneration cannot occur in production output. The C2/C5
/// grammars here are hand-authored test fixtures, which is why bounding (and guarding)
/// them lives at the test layer.
final class LlamaGrammarFixtureBoundsTests: XCTestCase {

    private typealias Fixtures = LlamaGrammarConformanceTests.GrammarFixtures

    /// Every rule name in a fixture grammar must be hyphen/alnum only — never an
    /// underscore. llama.cpp's GBNF parser uses `is_word_char = [a-zA-Z0-9-]`, so an
    /// underscore terminates the rule name early (`args_0` parses as `args` then a
    /// syntax error). The C2/C5 fixtures use single-word names (`root`, `string`,
    /// `number`, `obj`, `ws`), but this guards against a future multi-word rename
    /// reaching for `_`.
    ///
    /// Sabotage check: rename `obj` to `tool_obj` in `toolCallEnvelope` — this fails.
    func test_fixtureRuleNames_areParserSafe_noUnderscores() {
        for (label, grammar) in [("C2/jsonObject", Fixtures.jsonObject),
                                 ("C5/toolCallEnvelope", Fixtures.toolCallEnvelope)] {
            for line in grammar.split(separator: "\n") {
                guard let arrow = line.range(of: "::=") else { continue }
                let name = line[..<arrow.lowerBound].trimmingCharacters(in: .whitespaces)
                XCTAssertFalse(name.isEmpty, "\(label): a rule line must declare a name before '::='")
                XCTAssertFalse(name.contains("_"),
                               "\(label): GBNF rule name '\(name)' contains '_'; llama.cpp parser "
                             + "is_word_char=[a-zA-Z0-9-] stops at '_' and mis-parses the name. Use hyphens.")
                XCTAssertTrue(name.allSatisfy { $0.isLetter || $0.isNumber || $0 == "-" },
                              "\(label): rule name '\(name)' has a non-[a-zA-Z0-9-] character")
            }
        }
    }

    /// C2: the `number` rule must cap fractional digits (the truncation cause). An
    /// unbounded `[0-9]+` fractional run is exactly what made mistral 7B emit
    /// `70.5644444…` past the token cap and never close the object.
    ///
    /// Sabotage check: revert `number` to `"-"? [0-9]+ ("." [0-9]+)?` — the bounded
    /// `{1,3}` / `{1,2}` quantifiers vanish and this fails.
    func test_C2_numberRule_boundsFractionalDigits() {
        let g = Fixtures.jsonObject
        XCTAssertTrue(g.contains("[0-9]{1,3}"),
                      "C2 number integer part must be bounded ({1,3}); got grammar:\n\(g)")
        XCTAssertTrue(g.contains("[0-9]{1,2}"),
                      "C2 number fractional part must be bounded ({1,2}); got grammar:\n\(g)")
        XCTAssertFalse(g.contains("[0-9]+"),
                       "C2 must not carry an unbounded [0-9]+ run (the truncation cause); got grammar:\n\(g)")
    }

    /// C2: the `string` (city) rule must cap its character run so the value cannot
    /// grow unbounded before the object can close.
    ///
    /// Sabotage check: revert to `string ::= "\"" [a-zA-Z ]* "\""` — the `{0,24}`
    /// bound vanishes and this fails.
    func test_C2_stringRule_boundsCharacterRun() {
        let g = Fixtures.jsonObject
        XCTAssertTrue(g.contains("[a-zA-Z ]{0,24}"),
                      "C2 city string must be length-bounded ({0,24}); got grammar:\n\(g)")
        XCTAssertFalse(g.contains("[a-zA-Z ]*"),
                       "C2 city string must not use an unbounded '*' run; got grammar:\n\(g)")
    }

    /// C5: the tool-call `name` rule must be length-bounded. Unbounded
    /// `[a-zA-Z_]+` is what let qwen/mistral 7B degenerate into giant names that
    /// never close the quote, so `</tool_call>` was never reached in budget.
    ///
    /// Sabotage check: revert to `string ::= "\"" [a-zA-Z_]+ "\""` — the `{1,32}`
    /// bound vanishes and this fails.
    func test_C5_nameRule_boundsLength() {
        let g = Fixtures.toolCallEnvelope
        XCTAssertTrue(g.contains("{1,32}"),
                      "C5 name string must be length-bounded ({1,32}); got grammar:\n\(g)")
        XCTAssertFalse(g.contains("[a-zA-Z_]+"),
                       "C5 name string must not use an unbounded '+' run (the truncation cause); got grammar:\n\(g)")
    }

    /// C5: the whitespace rule must be length-bounded too. The local real-model sweep
    /// found mistral still truncated with a bounded name because `ws ::= [ \t\n]*` was
    /// unbounded — mistral spent the token budget on newline indentation before reaching
    /// `</tool_call>`. Unbounded `ws` is the same truncation class as an unbounded name.
    ///
    /// Sabotage check: revert to `ws ::= [ \t\n]*` — the `{0,4}` bound vanishes and this fails.
    func test_C5_whitespaceRule_boundsLength() {
        let g = Fixtures.toolCallEnvelope
        XCTAssertTrue(g.contains(#"ws     ::= [ \t\n]{0,4}"#),
                      "C5 ws rule must be length-bounded ({0,4}); got grammar:\n\(g)")
        XCTAssertFalse(g.contains(#"[ \t\n]*"#),
                       "C5 ws must not use an unbounded '*' run (a truncation cause); got grammar:\n\(g)")
    }

    /// Both fixtures must still END with their closing delimiter in the root rule —
    /// bounding the inner runs must not have removed the structure the cases assert.
    func test_fixtures_retainClosingDelimiters() {
        XCTAssertTrue(Fixtures.jsonObject.contains(#""}""#),
                      "C2 root must still close the JSON object with '}'")
        XCTAssertTrue(Fixtures.toolCallEnvelope.contains("</tool_call>"),
                      "C5 root must still close the tool-call envelope with </tool_call>")
        XCTAssertTrue(Fixtures.toolCallEnvelope.contains("<tool_call>"),
                      "C5 root must still open the tool-call envelope with <tool_call>")
    }
}
