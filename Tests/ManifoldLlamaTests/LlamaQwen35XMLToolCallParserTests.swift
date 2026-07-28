import XCTest
import ManifoldInference
import ManifoldLlama
@_spi(Testing) import ManifoldLlama

/// Unit tests for the Qwen3.5 **XML** tool-call dialect (#158).
///
/// Qwen3.5's embedded chat template teaches a nested-XML call shape, NOT the
/// Qwen2.5 Hermes-JSON shape the backend historically implemented:
///
/// ```
/// <tool_call>
/// <function=get_weather>
/// <parameter=city>
/// London
/// </parameter>
/// </function>
/// </tool_call>
/// ```
///
/// The bodies below are transcribed from the template's own few-shot
/// instruction block (`tokenizer.chat_template` in
/// `Qwen_Qwen3.5-9B-Q4_K_M.gguf`), so they are the literal shape the model is
/// told to emit rather than an invented approximation.
///
/// Before the fix the `<tool_call>` … `</tool_call>` marker matched, the body
/// failed `JSONSerialization`, and `parseBody` returned `nil` — the call was
/// silently dropped and the model scored 0 dispatches on every scenario.
///
/// These are pure parser tests: no GGUF is loaded, so they run everywhere
/// (unlike the model-gated soak suites) — the model-agnostic coverage #158 asks
/// for.
final class LlamaQwen35XMLToolCallParserTests: XCTestCase {

    /// Drives the unified transform under Llama markers, matching the shim in
    /// `LlamaToolCallParserTests`.
    private struct Parser {
        private var transform = ToolCallTransform(markers: LlamaToolMarkers.markers())

        mutating func process(_ chunk: String) -> [GenerationEvent] {
            transform.process([.token(chunk)])
        }

        mutating func finalize() -> [GenerationEvent] {
            transform.finalize()
        }
    }

    private func toolCalls(_ events: [GenerationEvent]) -> [ToolCall] {
        events.compactMap { event in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
    }

    private func decodeArgs(_ call: ToolCall) throws -> [String: Any] {
        let data = try XCTUnwrap(call.arguments.data(using: .utf8))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    // MARK: - Core dialect

    func test_qwen35XMLCall_multipleParameters_emitsToolCallEvent() throws {
        var parser = Parser()
        let input = """
        <tool_call>
        <function=get_weather>
        <parameter=city>
        London
        </parameter>
        <parameter=units>
        celsius
        </parameter>
        </function>
        </tool_call>
        """
        let calls = toolCalls(parser.process(input))

        XCTAssertEqual(calls.count, 1)
        let call = try XCTUnwrap(calls.first)
        XCTAssertEqual(call.toolName, "get_weather")

        let args = try decodeArgs(call)
        XCTAssertEqual(args["city"] as? String, "London")
        XCTAssertEqual(args["units"] as? String, "celsius")
    }

    func test_qwen35XMLCall_noParameters_emitsCallWithEmptyArguments() throws {
        // The `01-now` scenario shape: a zero-argument tool. This is the
        // scenario that scored 0/1 on Qwen3.5-9B.
        var parser = Parser()
        let input = "<tool_call>\n<function=now>\n</function>\n</tool_call>"
        let calls = toolCalls(parser.process(input))

        XCTAssertEqual(calls.count, 1)
        let call = try XCTUnwrap(calls.first)
        XCTAssertEqual(call.toolName, "now")
        XCTAssertEqual(try decodeArgs(call).count, 0)
    }

    func test_qwen35XMLCall_multilineParameterValue_preservesInteriorNewlines() throws {
        // The template explicitly documents a value that "can span multiple
        // lines". Only the single delimiting newline either side of the value
        // is structural; interior newlines are content.
        var parser = Parser()
        let input = """
        <tool_call>
        <function=write_note>
        <parameter=body>
        line one
        line two
        </parameter>
        </function>
        </tool_call>
        """
        let calls = toolCalls(parser.process(input))

        let call = try XCTUnwrap(calls.first)
        XCTAssertEqual(call.toolName, "write_note")
        XCTAssertEqual(try decodeArgs(call)["body"] as? String, "line one\nline two")
    }

    func test_qwen35XMLCall_scalarValues_areTypedNotStrings() throws {
        // Tool schemas are typed; a bare numeric / boolean / null value must
        // round-trip as its JSON type so argument validation downstream does
        // not reject an integer arriving as "41".
        var parser = Parser()
        let input = """
        <tool_call>
        <function=calc>
        <parameter=a>
        7823
        </parameter>
        <parameter=b>
        41.5
        </parameter>
        <parameter=exact>
        true
        </parameter>
        </function>
        </tool_call>
        """
        let calls = toolCalls(parser.process(input))

        let args = try decodeArgs(try XCTUnwrap(calls.first))
        XCTAssertEqual(args["a"] as? Int, 7823)
        XCTAssertEqual(args["b"] as? Double, 41.5)
        XCTAssertEqual(args["exact"] as? Bool, true)
    }

    func test_qwen35XMLCall_valueContainingAngleBrackets_isNotTruncated() throws {
        // A parameter value may legitimately contain `<` — the close is the
        // literal `</parameter>` tag, not the next angle bracket.
        var parser = Parser()
        let input = """
        <tool_call>
        <function=echo>
        <parameter=text>
        a < b and c <notatag> d
        </parameter>
        </function>
        </tool_call>
        """
        let calls = toolCalls(parser.process(input))

        let args = try decodeArgs(try XCTUnwrap(calls.first))
        XCTAssertEqual(args["text"] as? String, "a < b and c <notatag> d")
    }

    // MARK: - Streaming

    func test_qwen35XMLCall_splitAcrossChunks_stillParses() throws {
        // Real generation arrives token by token; the dialect must survive a
        // split mid-tag, which is what the transform's holdback protects.
        var parser = Parser()
        let input = "<tool_call>\n<function=now>\n</function>\n</tool_call>"
        var events: [GenerationEvent] = []
        for character in input {
            events += parser.process(String(character))
        }
        events += parser.finalize()

        let calls = toolCalls(events)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.toolName, "now")
    }

    func test_qwen35XMLCall_afterThinkBlock_isStillParsed() throws {
        // Qwen3.5 opens `<think>` every turn (the template hardcodes it unless
        // `enable_thinking` is false). Prose before the call — which the
        // template explicitly permits — must not suppress the dispatch.
        var parser = Parser()
        let input = """
        <think>
        The user wants the time. I should call the now function.
        </think>

        Let me check the current time.
        <tool_call>
        <function=now>
        </function>
        </tool_call>
        """
        let calls = toolCalls(parser.process(input))

        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.toolName, "now")
    }

    // MARK: - Coexistence with the Qwen2.5 JSON dialect

    func test_qwen25JSONCall_stillParses_afterXMLSupportAdded() throws {
        // The XML dialect shares its `<tool_call>` … `</tool_call>` delimiters
        // with the Qwen2.5 Hermes-JSON dialect, so body dispatch must keep
        // routing a JSON body to the JSON parser. Guards against the XML
        // addition regressing every existing Qwen2.5 fine-tune.
        var parser = Parser()
        let input = "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>"
        let calls = toolCalls(parser.process(input))

        XCTAssertEqual(calls.count, 1)
        let call = try XCTUnwrap(calls.first)
        XCTAssertEqual(call.toolName, "get_weather")
        XCTAssertEqual(try decodeArgs(call)["city"] as? String, "Paris")
    }

    // MARK: - Malformed bodies

    func test_qwen35XMLCall_missingFunctionName_isDropped() throws {
        // An empty name must not dispatch a nameless tool.
        var parser = Parser()
        let input = "<tool_call>\n<function=>\n</function>\n</tool_call>"
        XCTAssertTrue(toolCalls(parser.process(input)).isEmpty)
    }

    func test_qwen35XMLCall_unterminatedFunctionTag_isDropped() throws {
        var parser = Parser()
        let input = "<tool_call>\n<function=now\n</function>\n</tool_call>"
        XCTAssertTrue(toolCalls(parser.process(input)).isEmpty)
    }
}
