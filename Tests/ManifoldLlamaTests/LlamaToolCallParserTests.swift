import XCTest
import ManifoldInference
import ManifoldLlama
@_spi(Testing) import ManifoldLlama

/// Test shim driving the unified `ToolCallTransform` (configured with
/// `LlamaToolMarkers.markers()`) through the old `LlamaToolCallParser`-shaped
/// API, so this suite keeps regression-testing the unified transform after the
/// parser unification (#1593).
private struct LlamaToolCallParser {
    private var transform = ToolCallTransform(markers: LlamaToolMarkers.markers())

    mutating func process(_ chunk: String) -> [GenerationEvent] {
        transform.process([.token(chunk)])
    }

    mutating func finalize() -> [GenerationEvent] {
        transform.finalize()
    }
}

/// Unit tests for the unified tool-call transform under Llama markers
/// (formerly `LlamaToolCallParser`).
///
/// These tests exercise the parser logic only — no GGUF model is loaded and
/// no hardware-specific symbols are invoked. They run under
/// `swift test --filter ManifoldBackendsTests --disable-default-traits`.
final class LlamaToolCallParserTests: XCTestCase {

    // MARK: - Gemma 4 native format

    func test_gemma4NativeCall_singleCall_emitsToolCallEvent() throws {
        var parser = LlamaToolCallParser()
        let input = "<|tool_call>\ncall:get_weather{city:<|\"|>London<|\"|>,units:<|\"|>celsius<|\"|>}\n<tool_call|>"
        let events = parser.process(input)

        let toolCalls = events.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertEqual(toolCalls.count, 1)
        let tc = try XCTUnwrap(toolCalls.first)
        XCTAssertEqual(tc.toolName, "get_weather")

        let args = try XCTUnwrap(tc.arguments.data(using: .utf8))
        let decoded = try XCTUnwrap(try? JSONSerialization.jsonObject(with: args) as? [String: Any])
        XCTAssertEqual(decoded["city"] as? String, "London")
        XCTAssertEqual(decoded["units"] as? String, "celsius")
    }

    func test_gemma4NativeCall_mixedScalarArgs_areTypedNotStrings() throws {
        // Gemma 4's brace body has unquoted JSON-like keys, with values that
        // are either `<|"|>...<|"|>`-quoted strings or bare numeric / boolean
        // literals. Earlier versions handed the body to JSONSerialization
        // directly, which always failed (unquoted keys aren't JSON), so every
        // native call fell back to `arguments == "{}"`. This test exists so
        // that regression cannot reappear silently.
        var parser = LlamaToolCallParser()
        let input = "<|tool_call>\ncall:rate{score:5,active:true,note:<|\"|>ok<|\"|>}\n<tool_call|>"
        let events = parser.process(input)
        let toolCalls = events.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertEqual(toolCalls.count, 1)
        let tc = try XCTUnwrap(toolCalls.first)
        XCTAssertEqual(tc.toolName, "rate")
        XCTAssertNotEqual(tc.arguments, "{}", "Native call body must not silently fall back to empty args")

        let args = try XCTUnwrap(tc.arguments.data(using: .utf8))
        let decoded = try XCTUnwrap(try? JSONSerialization.jsonObject(with: args) as? [String: Any])
        XCTAssertEqual(decoded["score"] as? Int, 5)
        XCTAssertEqual(decoded["active"] as? Bool, true)
        XCTAssertEqual(decoded["note"] as? String, "ok")
    }

    func test_gemma4NativeCall_noArgs_emitsToolCallWithEmptyArgs() {
        var parser = LlamaToolCallParser()
        let input = "<|tool_call>\ncall:list_files{}\n<tool_call|>"
        let events = parser.process(input)

        let toolCalls = events.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertEqual(toolCalls.count, 1)
        XCTAssertEqual(toolCalls.first?.toolName, "list_files")
    }

    func test_gemma4NativeCall_prefixText_emitsTokenBeforeToolCall() {
        var parser = LlamaToolCallParser()
        let input = "Sure, let me check that.<|tool_call>\ncall:get_time{}\n<tool_call|>"
        let events = parser.process(input)

        let tokens = events.compactMap { event -> String? in
            if case .token(let t) = event { return t }
            return nil
        }
        let toolCalls = events.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertFalse(tokens.isEmpty, "Expected token events before tool call")
        XCTAssertEqual(toolCalls.count, 1)
        XCTAssertEqual(toolCalls.first?.toolName, "get_time")
    }

    // MARK: - JSON fallback format

    func test_jsonFallback_singleCall_emitsToolCallEvent() {
        var parser = LlamaToolCallParser()
        let input = """
        <tool_call>
        {"name":"search","arguments":{"query":"swift concurrency"}}
        </tool_call>
        """
        let events = parser.process(input)
        let toolCalls = events.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertEqual(toolCalls.count, 1)
        XCTAssertEqual(toolCalls.first?.toolName, "search")
    }

    func test_jsonFallback_multipleCalls_emitsMultipleToolCallEvents() {
        var parser = LlamaToolCallParser()
        let input = """
        <tool_call>{"name":"tool_a","arguments":{}}</tool_call>\
        <tool_call>{"name":"tool_b","arguments":{}}</tool_call>
        """
        let events = parser.process(input)
        let toolCalls = events.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertEqual(toolCalls.count, 2)
        XCTAssertEqual(toolCalls[0].toolName, "tool_a")
        XCTAssertEqual(toolCalls[1].toolName, "tool_b")
    }

    // MARK: - Chunk safety

    func test_tagSplitAcrossChunks_parsesCorrectly() {
        var parser = LlamaToolCallParser()

        // Split "<|tool_call>" across two chunks.
        let events1 = parser.process("<|tool_")
        let events2 = parser.process("call>\ncall:ping{}\n<tool_call|>")
        let all = events1 + events2

        let toolCalls = all.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertEqual(toolCalls.count, 1)
        XCTAssertEqual(toolCalls.first?.toolName, "ping")
    }

    func test_closeTagSplitAcrossChunks_parsesCorrectly() {
        var parser = LlamaToolCallParser()
        var all: [GenerationEvent] = []
        all += parser.process("<tool_call>{\"name\":\"foo\",\"arguments\":{}}</")
        all += parser.process("tool_call>")

        let toolCalls = all.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertEqual(toolCalls.count, 1)
        XCTAssertEqual(toolCalls.first?.toolName, "foo")
    }

    func test_singleByteChunks_parsesCorrectly() {
        var parser = LlamaToolCallParser()
        let full = "<tool_call>{\"name\":\"byte_test\",\"arguments\":{}}</tool_call>"
        var all: [GenerationEvent] = []
        for char in full.unicodeScalars {
            all += parser.process(String(char))
        }
        all += parser.finalize()

        let toolCalls = all.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertEqual(toolCalls.count, 1)
        XCTAssertEqual(toolCalls.first?.toolName, "byte_test")
    }

    // MARK: - Invalid / malformed input

    func test_invalidJSON_inJSONFallback_isDiscarded() {
        var parser = LlamaToolCallParser()
        let events = parser.process("<tool_call>this is not json</tool_call>")
        let toolCalls = events.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertTrue(toolCalls.isEmpty, "Malformed JSON should be discarded silently")
    }

    func test_missingNameField_inJSONFallback_isDiscarded() {
        var parser = LlamaToolCallParser()
        let events = parser.process("<tool_call>{\"arguments\":{}}</tool_call>")
        let toolCalls = events.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertTrue(toolCalls.isEmpty)
    }

    // MARK: - finalize

    func test_finalize_emitsBufferedPartialTagPrefix() {
        // When the stream ends with bytes that *could* be the start of an
        // open tag (here: "<|tool_") the parser holds them back during
        // `process()` so a later chunk can complete or cancel the tag.
        // `finalize()` must flush that held-back text once we know no more
        // chunks are coming.
        var parser = LlamaToolCallParser()
        let processEvents = parser.process("Hello <|tool_")
        let processTokens = processEvents.compactMap { event -> String? in
            if case .token(let t) = event { return t }
            return nil
        }
        // The visible "Hello " portion may be emitted during process;
        // the partial tag suffix "<|tool_" must NOT be — verify by reuniting
        // the joined token stream and checking it never leaks the partial tag.
        XCTAssertFalse(processTokens.joined().contains("<|tool_"),
                       "Partial open-tag bytes must not leak as visible tokens")

        let finalEvents = parser.finalize()
        let finalTokens = finalEvents.compactMap { event -> String? in
            if case .token(let t) = event { return t }
            return nil
        }
        XCTAssertFalse(finalTokens.joined().isEmpty,
                       "finalize must flush bytes held back as a candidate open-tag prefix")
    }

    func test_finalize_discardsPartialToolCallBlock() {
        var parser = LlamaToolCallParser()
        _ = parser.process("<tool_call>{\"name\":\"partial\"")
        // No close tag — incomplete block.
        let events = parser.finalize()
        let toolCalls = events.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertTrue(toolCalls.isEmpty, "Incomplete tool-call block must be discarded on finalize")
    }

    func test_finalize_calledOnFreshParser_returnsEmpty() {
        var parser = LlamaToolCallParser()
        XCTAssertTrue(parser.finalize().isEmpty)
    }

    // MARK: - Mistral [TOOL_CALLS] format (#70)
    //
    // Mistral v0.3 emits a bare JSON ARRAY of calls right after a literal
    // `[TOOL_CALLS]` token with NO closing tag — the block ends at EOS. These
    // tests are synthetic/unit-level: end-to-end verification against a real
    // Mistral model is deferred to #69 / ManifoldKit#1983 (the scenario harness
    // does not render tools yet).

    func test_mistral_singleCall_extractsNameAndArguments() throws {
        var parser = LlamaToolCallParser()
        var all = parser.process(#"[TOOL_CALLS][{"name": "calc", "arguments": {"a": 1}}]"#)
        all += parser.finalize()

        let toolCalls = all.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertEqual(toolCalls.count, 1)
        let tc = try XCTUnwrap(toolCalls.first)
        XCTAssertEqual(tc.toolName, "calc")

        let args = try XCTUnwrap(tc.arguments.data(using: .utf8))
        let decoded = try XCTUnwrap(try? JSONSerialization.jsonObject(with: args) as? [String: Any])
        XCTAssertEqual(decoded["a"] as? Int, 1)
    }

    func test_mistral_multipleCalls_fromOneArray() {
        var parser = LlamaToolCallParser()
        var all = parser.process(#"[TOOL_CALLS][{"name": "calc", "arguments": {"a": 1}}, {"name": "now", "arguments": {}}]"#)
        all += parser.finalize()

        let toolCalls = all.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertEqual(toolCalls.count, 2)
        XCTAssertEqual(toolCalls[0].toolName, "calc")
        XCTAssertEqual(toolCalls[1].toolName, "now")
        // count == 2 (not 1) proves the array multi-call path (`parseBodyMulti`)
        // is exercised — a single-call fallback would collapse this to one call.
    }

    func test_mistral_plainText_withoutToolCallsToken_isUntouched() {
        var parser = LlamaToolCallParser()
        var all = parser.process("Here is a plain answer with no tool calls.")
        all += parser.finalize()

        let tokens = all.compactMap { event -> String? in
            if case .token(let t) = event { return t }
            return nil
        }
        let toolCalls = all.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertTrue(toolCalls.isEmpty, "Text without [TOOL_CALLS] must not yield tool calls")
        XCTAssertEqual(tokens.joined(), "Here is a plain answer with no tool calls.")
    }

    func test_mistral_eosClose_unterminatedBlockAtStreamEnd_yieldsCall() throws {
        // Mistral has NO close tag: the block is only terminated by EOS. The
        // call must therefore not appear until finalize() drains the buffered
        // body (the #1982 `closesAtEnd` path).
        var parser = LlamaToolCallParser()
        let processEvents = parser.process(#"[TOOL_CALLS][{"name": "lookup", "arguments": {"q": "swift"}}]"#)

        let midToolCalls = processEvents.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertTrue(midToolCalls.isEmpty,
                      "No close tag means the call is buffered until EOS — process() must not emit it yet")

        let finalToolCalls = parser.finalize().compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertEqual(finalToolCalls.count, 1, "finalize() (EOS) must flush the buffered Mistral call")
        let tc = try XCTUnwrap(finalToolCalls.first)
        XCTAssertEqual(tc.toolName, "lookup")
        let args = try XCTUnwrap(tc.arguments.data(using: .utf8))
        let decoded = try XCTUnwrap(try? JSONSerialization.jsonObject(with: args) as? [String: Any])
        XCTAssertEqual(decoded["q"] as? String, "swift")
    }

    func test_mistral_prefixText_emittedBeforeToolCalls() {
        var parser = LlamaToolCallParser()
        var all = parser.process(#"Sure.[TOOL_CALLS][{"name": "go", "arguments": {}}]"#)
        all += parser.finalize()

        let tokens = all.compactMap { event -> String? in
            if case .token(let t) = event { return t }
            return nil
        }
        let toolCalls = all.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertEqual(tokens.joined(), "Sure.")
        XCTAssertEqual(toolCalls.count, 1)
        XCTAssertEqual(toolCalls.first?.toolName, "go")
    }

    // MARK: - Llama 3.1 native bare-JSON format (#76)
    //
    // llama3.1's native `llama3` tool template emits a bare top-level JSON
    // object with NO open/close marker, keyed on `parameters`:
    //   {"name": "calc", "parameters": {"a": 7823, "b": 41, "op": "*"}}
    // The block ends at EOS (the #1982 `closesAtEnd` path). These tests are
    // model-free; real-model dispatch is verified via manifold-tools-llama.

    func test_llama3BareJSON_singleCall_extractsNameAndParameters() throws {
        var parser = LlamaToolCallParser()
        var all = parser.process(#"{"name": "calc", "parameters": {"a": 7823, "b": 41, "op": "*"}}"#)
        all += parser.finalize()

        let toolCalls = all.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertEqual(toolCalls.count, 1)
        let tc = try XCTUnwrap(toolCalls.first)
        XCTAssertEqual(tc.toolName, "calc")

        let args = try XCTUnwrap(tc.arguments.data(using: .utf8))
        let decoded = try XCTUnwrap(try? JSONSerialization.jsonObject(with: args) as? [String: Any])
        // `parameters` must be mapped onto `arguments` — not dropped.
        XCTAssertEqual(decoded["a"] as? Int, 7823)
        XCTAssertEqual(decoded["b"] as? Int, 41)
        XCTAssertEqual(decoded["op"] as? String, "*")
        XCTAssertNotEqual(tc.arguments, "{}",
                          "parameters payload must populate arguments, not fall back to empty")
    }

    func test_llama3BareJSON_noCallUntilFinalize() {
        // No marker and no close tag: the object is buffered until EOS, so
        // process() alone must not emit it (the `closesAtEnd` contract).
        var parser = LlamaToolCallParser()
        let mid = parser.process(#"{"name": "now", "parameters": {}}"#)
        let midCalls = mid.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertTrue(midCalls.isEmpty,
                      "Bare-JSON call has no close tag — it must not surface before finalize()")

        let finalCalls = parser.finalize().compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertEqual(finalCalls.count, 1)
        XCTAssertEqual(finalCalls.first?.toolName, "now")
    }

    func test_llama3BareJSON_argumentsKeyAlsoAccepted() throws {
        // The bare-JSON dialect must also accept the historical `arguments` key,
        // and when BOTH are present `arguments` wins (no silent shape change).
        var parser = LlamaToolCallParser()
        var all = parser.process(#"{"name": "calc", "arguments": {"a": 1}, "parameters": {"a": 999}}"#)
        all += parser.finalize()
        let toolCalls = all.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertEqual(toolCalls.count, 1)
        let tc = try XCTUnwrap(toolCalls.first)
        let args = try XCTUnwrap(tc.arguments.data(using: .utf8))
        let decoded = try XCTUnwrap(try? JSONSerialization.jsonObject(with: args) as? [String: Any])
        XCTAssertEqual(decoded["a"] as? Int, 1,
                       "arguments must take priority over parameters when both are present")
    }

    func test_parametersAlias_worksForMarkerWrappedJSON() throws {
        // The `parameters` alias is not bare-JSON-only — a marker-wrapped JSON
        // fallback call keyed on `parameters` must populate arguments too.
        var parser = LlamaToolCallParser()
        var all = parser.process(#"<tool_call>{"name":"search","parameters":{"q":"swift"}}</tool_call>"#)
        all += parser.finalize()
        let toolCalls = all.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertEqual(toolCalls.count, 1)
        let tc = try XCTUnwrap(toolCalls.first)
        XCTAssertEqual(tc.toolName, "search")
        let args = try XCTUnwrap(tc.arguments.data(using: .utf8))
        let decoded = try XCTUnwrap(try? JSONSerialization.jsonObject(with: args) as? [String: Any])
        XCTAssertEqual(decoded["q"] as? String, "swift")
    }

    func test_llama3BareJSON_plainProse_withoutNameKey_isUntouched() {
        // Critical false-positive guard: ordinary prose with no `{"name"` anchor
        // must pass straight through as visible text — never suppressed as a body.
        var parser = LlamaToolCallParser()
        var all = parser.process("The result is 320743. Let me know if you need anything else.")
        all += parser.finalize()

        let tokens = all.compactMap { event -> String? in
            if case .token(let t) = event { return t }
            return nil
        }
        let toolCalls = all.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertTrue(toolCalls.isEmpty)
        XCTAssertEqual(tokens.joined(), "The result is 320743. Let me know if you need anything else.")
    }

    func test_llama3BareJSON_nameKeyButNotAToolCall_yieldsNoCall() {
        // A body that begins with the `{"name"` anchor but is NOT a valid call
        // (e.g. a missing-name shape after decode, or non-object JSON) must be
        // rejected by the decoder — no spurious dispatch.
        var parser = LlamaToolCallParser()
        // `{"name": 42, ...}` — name is not a string, so parseJSONCall rejects it.
        var all = parser.process(#"{"name": 42, "parameters": {}}"#)
        all += parser.finalize()
        let toolCalls = all.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertTrue(toolCalls.isEmpty,
                      "A non-string name must not produce a tool call even with the {\"name\" anchor")
    }

    // MARK: - Tool call ID uniqueness

    func test_multipleToolCalls_haveDistinctIDs() {
        var parser = LlamaToolCallParser()
        let events = parser.process("""
        <tool_call>{"name":"a","arguments":{}}</tool_call>\
        <tool_call>{"name":"b","arguments":{}}</tool_call>
        """)
        let ids = events.compactMap { event -> String? in
            if case .toolCall(let tc) = event { return tc.id }
            return nil
        }
        XCTAssertEqual(ids.count, 2)
        XCTAssertNotEqual(ids[0], ids[1], "Each tool call must receive a distinct ID")
    }
}

// MARK: - Edge-case and malformed-input tests
//
// Covers boundary and failure contracts: malformed brace bodies, bare-literal
// argument types (double, null), missing braces, empty names, unterminated
// strings, and JSON-fallback boundary conditions. Appended here (rather than
// a standalone file) to avoid the Xcode 26 / Swift 6.3 parallel-compilation
// race that drops new standalone files with "no such module 'ManifoldLlamaKit'".

final class LlamaToolMarkerEdgeCaseTests: XCTestCase {

    private struct Parser {
        private var transform = ToolCallTransform(markers: LlamaToolMarkers.markers())

        mutating func process(_ chunk: String) -> [GenerationEvent] {
            transform.process([.token(chunk)])
        }

        mutating func finalize() -> [GenerationEvent] {
            transform.finalize()
        }
    }

    private func toolCalls(from events: [GenerationEvent]) -> [ToolCall] {
        events.compactMap { if case .toolCall(let tc) = $0 { return tc } else { return nil } }
    }

    // MARK: - Gemma 4: missing brace body → no tool call

    func test_gemma4_noBraces_producesNoToolCall() {
        var p = Parser()
        let events = p.process("<|tool_call>\ncall:tool_no_braces\n<tool_call|>")
        XCTAssertTrue(toolCalls(from: events).isEmpty,
                      "A call: body without braces must not produce a tool call")
    }

    // MARK: - Gemma 4: empty name → no tool call

    func test_gemma4_emptyName_producesNoToolCall() {
        var p = Parser()
        let events = p.process("<|tool_call>\ncall:{}\n<tool_call|>")
        XCTAssertTrue(toolCalls(from: events).isEmpty,
                      "A call: body with an empty tool name must not produce a tool call")
    }

    // MARK: - Gemma 4: unterminated quoted string → arguments fall back to {}

    func test_gemma4_unterminatedString_producesToolCallWithEmptyArgs() {
        var p = Parser()
        let events = p.process("<|tool_call>\ncall:my_tool{key:<|\"|>no closing quote\n<tool_call|>")
        let calls = toolCalls(from: events)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.toolName, "my_tool")
        XCTAssertEqual(calls.first?.arguments, "{}",
                       "Unterminated string must fall back to empty arguments rather than crashing")
    }

    // MARK: - Gemma 4: double-precision bare value

    func test_gemma4_doubleValue_isPreservedInArguments() throws {
        var p = Parser()
        let events = p.process("<|tool_call>\ncall:score{threshold:0.75}\n<tool_call|>")
        let calls = toolCalls(from: events)
        XCTAssertEqual(calls.count, 1)
        let tc = try XCTUnwrap(calls.first)
        let data = try XCTUnwrap(tc.arguments.data(using: .utf8))
        let decoded = try XCTUnwrap(try? JSONSerialization.jsonObject(with: data) as? [String: Any])
        let threshold = try XCTUnwrap(decoded["threshold"] as? Double)
        XCTAssertEqual(threshold, 0.75, accuracy: 1e-9)
    }

    // MARK: - Gemma 4: null bare value

    func test_gemma4_nullValue_isPreservedInArguments() throws {
        var p = Parser()
        let events = p.process("<|tool_call>\ncall:op{cursor:null}\n<tool_call|>")
        let calls = toolCalls(from: events)
        XCTAssertEqual(calls.count, 1)
        let tc = try XCTUnwrap(calls.first)
        let data = try XCTUnwrap(tc.arguments.data(using: .utf8))
        let decoded = try XCTUnwrap(try? JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertTrue(decoded["cursor"] is NSNull,
                      "bare 'null' value must be preserved as JSON null")
    }

    // MARK: - Gemma 4: multiple heterogeneous bare values

    func test_gemma4_mixedBareValues_allTypesPresent() throws {
        var p = Parser()
        let events = p.process("<|tool_call>\ncall:multi{n:7,ratio:1.5,ok:false,ptr:null}\n<tool_call|>")
        let calls = toolCalls(from: events)
        XCTAssertEqual(calls.count, 1)
        let tc = try XCTUnwrap(calls.first)
        let data = try XCTUnwrap(tc.arguments.data(using: .utf8))
        let decoded = try XCTUnwrap(try? JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(decoded["n"] as? Int, 7)
        XCTAssertEqual(decoded["ratio"] as? Double ?? 0, 1.5, accuracy: 1e-9)
        XCTAssertEqual(decoded["ok"] as? Bool, false)
        XCTAssertTrue(decoded["ptr"] is NSNull)
    }

    // MARK: - JSON fallback: no arguments key → {}

    func test_jsonFallback_noArgumentsKey_producesEmptyArgs() {
        var p = Parser()
        let events = p.process("<tool_call>{\"name\":\"ping\"}</tool_call>")
        let calls = toolCalls(from: events)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.toolName, "ping")
        XCTAssertEqual(calls.first?.arguments, "{}",
                       "Missing arguments key must fall back to empty object")
    }

    // MARK: - JSON fallback: string arguments field is preserved

    func test_jsonFallback_stringArgumentsField_isPreservedVerbatim() {
        var p = Parser()
        let events = p.process(#"<tool_call>{"name":"raw","arguments":"{\"x\":1}"}</tool_call>"#)
        let calls = toolCalls(from: events)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.toolName, "raw")
        XCTAssertEqual(calls.first?.arguments, "{\"x\":1}",
                       "A pre-serialized string arguments field must be preserved verbatim")
    }

    // MARK: - JSON fallback: empty name → no tool call

    func test_jsonFallback_emptyName_producesNoToolCall() {
        var p = Parser()
        let events = p.process("<tool_call>{\"name\":\"\",\"arguments\":{}}</tool_call>")
        XCTAssertTrue(toolCalls(from: events).isEmpty,
                      "A JSON body with an empty name must not produce a tool call")
    }

    // MARK: - JSON fallback: whitespace-only body → no tool call

    func test_jsonFallback_whitespaceBody_producesNoToolCall() {
        var p = Parser()
        let events = p.process("<tool_call>   \n   </tool_call>")
        XCTAssertTrue(toolCalls(from: events).isEmpty,
                      "A whitespace-only body must not produce a tool call")
    }
}
