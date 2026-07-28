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

    // MARK: - Non-finite numbers (crash regression)

    /// `Double(String)` accepts `nan` / `inf` / `1e400`, and
    /// `JSONSerialization.data(withJSONObject:)` raises an ObjC
    /// `NSInvalidArgumentException` — NOT a Swift error — on a non-finite
    /// number. `try?` cannot catch an ObjC exception, so coercing one of these
    /// to a `Double` aborted the whole process (exit 134, uncaught NSException):
    /// a model emitting `nan` from a divide-by-zero, or extraction over a CSV
    /// containing `inf`, would take the host down with no error and no degraded
    /// path.
    ///
    /// These MUST stay strings. If this regresses the test does not fail — it
    /// crashes the test runner, which is the loudest possible signal.
    func test_qwen35XMLCall_nonFiniteValues_doNotCrashAndStayStrings() throws {
        for spelling in ["nan", "NaN", "inf", "Inf", "infinity", "-inf", "1e400", "-1e400"] {
            var parser = Parser()
            let input = """
            <tool_call>
            <function=calc>
            <parameter=a>
            \(spelling)
            </parameter>
            </function>
            </tool_call>
            """
            let calls = toolCalls(parser.process(input))
            XCTAssertEqual(calls.count, 1, "\(spelling) should still dispatch")
            let value = try decodeArgs(try XCTUnwrap(calls.first))["a"]
            XCTAssertEqual(value as? String, spelling,
                           "\(spelling) must be preserved as text, never coerced to a non-finite Double")
        }
    }

    func test_qwen35XMLCall_overflowingNumberInsideJSONArray_doesNotCrash() throws {
        // Checks the other candidate route to a non-finite: an overflowing
        // literal inside a JSON-shaped value. `JSONSerialization.jsonObject`
        // REJECTS `1e400` outright (it does not decode to `inf`), so the array
        // decode fails and the value falls through to text. Asserted here so
        // the behaviour is pinned rather than assumed — the guarantee that
        // matters is that the process survives and nothing non-finite is
        // constructed.
        var parser = Parser()
        let input = """
        <tool_call>
        <function=calc>
        <parameter=xs>
        [1e400]
        </parameter>
        </function>
        </tool_call>
        """
        let args = try decodeArgs(try XCTUnwrap(toolCalls(parser.process(input)).first))
        XCTAssertEqual(args["xs"] as? String, "[1e400]")
    }

    func test_qwen35XMLCall_finiteScientificNotation_stillTypesAsDouble() throws {
        // The non-finite guard must not reject legitimate exponent notation.
        var parser = Parser()
        let input = "<tool_call>\n<function=calc>\n<parameter=a>\n1e3\n</parameter>\n</function>\n</tool_call>"
        let args = try decodeArgs(try XCTUnwrap(toolCalls(parser.process(input)).first))
        XCTAssertEqual(args["a"] as? Double, 1000.0)
    }

    // MARK: - Structured (array / object) parameter values

    func test_qwen35XMLCall_arrayValue_decodesAsArrayNotString() throws {
        // The template renders structured args as `args_value | tojson`, so the
        // model is TAUGHT to emit JSON text here. Handing the tool the string
        // `"[\"a\",\"b\"]"` instead of a list fails schema validation downstream.
        var parser = Parser()
        let input = """
        <tool_call>
        <function=list_dir>
        <parameter=names>
        ["a","b"]
        </parameter>
        </function>
        </tool_call>
        """
        let args = try decodeArgs(try XCTUnwrap(toolCalls(parser.process(input)).first))
        XCTAssertEqual(args["names"] as? [String], ["a", "b"])
    }

    func test_qwen35XMLCall_objectValue_decodesAsObjectNotString() throws {
        var parser = Parser()
        let input = """
        <tool_call>
        <function=configure>
        <parameter=opts>
        {"depth":2,"recursive":true}
        </parameter>
        </function>
        </tool_call>
        """
        let args = try decodeArgs(try XCTUnwrap(toolCalls(parser.process(input)).first))
        let opts = try XCTUnwrap(args["opts"] as? [String: Any])
        XCTAssertEqual(opts["depth"] as? Int, 2)
        XCTAssertEqual(opts["recursive"] as? Bool, true)
    }

    func test_qwen35XMLCall_bracketLeadingProse_isNotMangled() throws {
        // A value that merely STARTS with `[` but is not JSON must survive as
        // text rather than being dropped or half-parsed.
        var parser = Parser()
        let input = """
        <tool_call>
        <function=echo>
        <parameter=text>
        [draft] not json at all
        </parameter>
        </function>
        </tool_call>
        """
        let args = try decodeArgs(try XCTUnwrap(toolCalls(parser.process(input)).first))
        XCTAssertEqual(args["text"] as? String, "[draft] not json at all")
    }

    // MARK: - Whitespace-padded scalars

    func test_qwen35XMLCall_whitespacePaddedScalar_stillTypesAsNumber() throws {
        var parser = Parser()
        let input = "<tool_call>\n<function=calc>\n<parameter=a>\n 41 \n</parameter>\n</function>\n</tool_call>"
        let args = try decodeArgs(try XCTUnwrap(toolCalls(parser.process(input)).first))
        XCTAssertEqual(args["a"] as? Int, 41)
    }

    func test_qwen35XMLCall_freeTextValue_keepsItsInternalSpacing() throws {
        // Trimming is a probe for scalar detection only — it must not silently
        // rewrite the content of a genuine free-text parameter.
        var parser = Parser()
        let input = """
        <tool_call>
        <function=write_note>
        <parameter=body>
        line one
          indented two
        </parameter>
        </function>
        </tool_call>
        """
        let args = try decodeArgs(try XCTUnwrap(toolCalls(parser.process(input)).first))
        XCTAssertEqual(args["body"] as? String, "line one\n  indented two")
    }

    // MARK: - Multiple function blocks in one delimiter pair

    func test_qwen35XMLCall_twoFunctionBlocks_doNotMergeParameters() throws {
        // Off-spec emission, but the failure mode is a silent WRONG dispatch:
        // without bounding the parameter scan at `</function>`, both parameter
        // sets merge into one call under the FIRST function's name.
        var parser = Parser()
        let input = """
        <tool_call>
        <function=first>
        <parameter=a>
        1
        </parameter>
        </function>
        <function=second>
        <parameter=b>
        2
        </parameter>
        </function>
        </tool_call>
        """
        let call = try XCTUnwrap(toolCalls(parser.process(input)).first)
        XCTAssertEqual(call.toolName, "first")
        let args = try decodeArgs(call)
        XCTAssertEqual(args["a"] as? Int, 1)
        XCTAssertNil(args["b"], "second function's parameters must not leak into the first call")
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
