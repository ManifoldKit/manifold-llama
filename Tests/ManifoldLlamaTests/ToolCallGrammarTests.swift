import XCTest
import ManifoldModelCatalog
import ManifoldHardware
@testable import ManifoldLlama

/// Model-free unit tests for the GBNF tool-call grammar generator
/// (`ToolCallGrammar`). No GGUF, no Metal, no generation — these run anywhere,
/// including CI. They assert the *shape* of the emitted grammar: the name enum is
/// present and keyed to the advertised tools, the JSON envelope is correct, the
/// dialect's delimiters are honoured, and the empty / unsupported paths throw.
final class ToolCallGrammarTests: XCTestCase {

    /// The Llama-3.1 dialect (`.buried`, bare JSON): no delimiters, `.json` args.
    private let llamaDialect = ChatTemplateToolDescriptor.ToolCallDialect(
        openDelimiter: nil, closeDelimiter: nil, argEncoding: .json)

    // MARK: - name enum

    func test_json_emitsNameEnumForEachAdvertisedTool() throws {
        let g = try ToolCallGrammar.grammar(
            for: llamaDialect, toolNames: ["now", "calc", "read_file"])

        // The `name` production must list each advertised tool as a quoted literal.
        XCTAssertTrue(g.contains(#"\"now\""#), "name enum missing `now`")
        XCTAssertTrue(g.contains(#"\"calc\""#), "name enum missing `calc`")
        XCTAssertTrue(g.contains(#"\"read_file\""#), "name enum missing `read_file`")
        // Alternation separates the names.
        let nameLine = g.split(separator: "\n").first { $0.hasPrefix("name") }
        XCTAssertNotNil(nameLine, "no `name` production found")
        XCTAssertTrue(nameLine!.contains("|"), "name enum is not an alternation")

        // Sabotage: a tool NOT advertised must not appear in the name enum.
        XCTAssertFalse(g.contains(#"\"send_email\""#),
                       "name enum leaked an un-advertised tool")
    }

    func test_json_singleTool_hasNoAlternationButNamesIt() throws {
        let g = try ToolCallGrammar.grammar(for: llamaDialect, toolNames: ["now"])
        XCTAssertTrue(g.contains(#"\"now\""#))
        let nameLine = g.split(separator: "\n").first { $0.hasPrefix("name") }!
        // One tool ⇒ no `|` on the name production.
        XCTAssertFalse(nameLine.contains("|"), "single-tool name enum should not alternate")
    }

    // MARK: - JSON envelope

    func test_json_envelopeIsCompactAndWellFormed() throws {
        let g = try ToolCallGrammar.grammar(for: llamaDialect, toolNames: ["calc"])

        // Compact JSON envelope — keyed on `name` then `arguments`, no whitespace
        // between `{` and `"name"` (so it satisfies LlamaToolMarkers' `{"name"`
        // anchor). The `call` production literal is the load-bearing shape.
        XCTAssertTrue(g.contains(#"call    ::= "{\"name\":" name ",\"arguments\":" object "}""#),
                      "call production is not the expected compact {\"name\":…,\"arguments\":…} shape")

        // Standard JSON value productions are present so `arguments` is well-formed.
        for production in ["root", "call", "name", "object", "member", "array",
                           "value", "string", "number"] {
            XCTAssertTrue(g.contains("\(production)"),
                          "missing JSON production `\(production)`")
        }
    }

    func test_json_bareDialect_rootIsTheCallNoDelimiters() throws {
        let g = try ToolCallGrammar.grammar(for: llamaDialect, toolNames: ["now"])
        // Bare-JSON dialect: root is just `call`, no wrapping delimiter literals.
        let rootLine = g.split(separator: "\n").first { $0.hasPrefix("root") }!
        XCTAssertTrue(rootLine.contains("call"))
        XCTAssertFalse(rootLine.contains(#"""#) && rootLine.contains("tool_call"),
                       "bare dialect root must not wrap the call in delimiters")
    }

    // MARK: - delimiter-bearing dialect (forward-compat scaffolding)

    func test_json_delimiterDialect_wrapsCallInDelimiters() throws {
        // A Qwen/Hermes-style delimited JSON dialect — exercises the parameterised
        // envelope path even though the spike fully targets Llama's bare form.
        let qwen = ChatTemplateToolDescriptor.ToolCallDialect(
            openDelimiter: "<tool_call>", closeDelimiter: "</tool_call>", argEncoding: .json)
        let g = try ToolCallGrammar.grammar(for: qwen, toolNames: ["now"])
        let rootLine = g.split(separator: "\n").first { $0.hasPrefix("root") }!
        XCTAssertTrue(rootLine.contains(#""<tool_call>""#), "open delimiter not in root")
        XCTAssertTrue(rootLine.contains(#""</tool_call>""#), "close delimiter not in root")
    }

    // MARK: - error paths

    func test_emptyToolList_throws() {
        XCTAssertThrowsError(try ToolCallGrammar.grammar(for: llamaDialect, toolNames: [])) { error in
            XCTAssertEqual(error as? ToolCallGrammar.GenerationError, .emptyToolList)
        }
    }

    func test_unsupportedEncoding_throwsWithEncoding() {
        for enc: ChatTemplateToolDescriptor.ArgEncoding in [.keyValue, .keyEqualsValue] {
            let dialect = ChatTemplateToolDescriptor.ToolCallDialect(
                openDelimiter: "<|tool_call>", closeDelimiter: "<|/tool_call>", argEncoding: enc)
            XCTAssertThrowsError(try ToolCallGrammar.grammar(for: dialect, toolNames: ["now"])) { error in
                XCTAssertEqual(error as? ToolCallGrammar.GenerationError, .unsupportedEncoding(enc))
            }
        }
    }

    // MARK: - schema-bearing overload

    func test_toolWithSchema_stillProducesValidEnvelope() throws {
        let schema = JSONSchemaValue.object([
            "type": .string("object"),
            "properties": .object(["a": .object(["type": .string("number")])]),
            "required": .array([.string("a")])
        ])
        let g = try ToolCallGrammar.grammar(
            for: llamaDialect,
            tools: [ToolCallGrammar.Tool(name: "calc", parametersSchema: schema)])
        XCTAssertTrue(g.contains(#"\"calc\""#))
        XCTAssertTrue(g.contains("object"))
    }
}
