import XCTest
import ManifoldInference
import ManifoldLlamaKit
@_spi(Testing) import ManifoldLlamaKit

/// Edge-case and malformed-input tests for ``LlamaToolMarkers``.
///
/// The main ``LlamaToolCallParserTests`` suite covers the happy paths. This
/// suite pins the boundary and failure contracts: malformed brace bodies,
/// bare-literal argument types (double, null), missing braces, empty names,
/// unterminated strings, and JSON-fallback boundary conditions.
///
/// All tests drive the full ``ToolCallTransform`` pipeline (identical to the
/// production code path) rather than calling private parse functions directly.
final class LlamaToolMarkerEdgeCaseTests: XCTestCase {

    // MARK: - Test shim (mirrors LlamaToolCallParserTests)

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
        // "call:name" with no brace body — name extraction fails
        let events = p.process("<|tool_call>\ncall:tool_no_braces\n<|end_of_turn>")
        XCTAssertTrue(toolCalls(from: events).isEmpty,
                      "A call: body without braces must not produce a tool call")
    }

    // MARK: - Gemma 4: empty name → no tool call

    func test_gemma4_emptyName_producesNoToolCall() {
        var p = Parser()
        // "call:{}" — nothing before the opening brace → name is ""
        let events = p.process("<|tool_call>\ncall:{}\n<|end_of_turn>")
        XCTAssertTrue(toolCalls(from: events).isEmpty,
                      "A call: body with an empty tool name must not produce a tool call")
    }

    // MARK: - Gemma 4: unterminated quoted string → arguments fall back to {}

    func test_gemma4_unterminatedString_producesToolCallWithEmptyArgs() {
        var p = Parser()
        // The closing quote token is missing — parseGemma4Arguments returns nil → "{}"
        let events = p.process("<|tool_call>\ncall:my_tool{key:<|\"|>no closing quote\n<|end_of_turn>")
        let calls = toolCalls(from: events)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.toolName, "my_tool")
        XCTAssertEqual(calls.first?.arguments, "{}",
                       "Unterminated string must fall back to empty arguments rather than crashing")
    }

    // MARK: - Gemma 4: double-precision bare value

    func test_gemma4_doubleValue_isPreservedInArguments() throws {
        var p = Parser()
        let events = p.process("<|tool_call>\ncall:score{threshold:0.75}\n<|end_of_turn>")
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
        let events = p.process("<|tool_call>\ncall:op{cursor:null}\n<|end_of_turn>")
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
        let events = p.process("<|tool_call>\ncall:multi{n:7,ratio:1.5,ok:false,ptr:null}\n<|end_of_turn>")
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
