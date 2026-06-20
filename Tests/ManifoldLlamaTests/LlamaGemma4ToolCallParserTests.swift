import XCTest
import ManifoldInference
import ManifoldLlama
@_spi(Testing) import ManifoldLlama

/// Fixture tests for the gemma-4 native tool-call output parser path.
///
/// These tests exercise `LlamaToolMarkers.markers()`, `ToolCallTransform`, and
/// `LlamaBackend.setStructuredHistory()` with no GGUF model loaded — they run
/// entirely in-process and complete in milliseconds.
///
/// Appended to the `LlamaToolCallParserTests.swift` compilation unit pattern:
/// a companion file of the parser suite that focuses specifically on:
///   1. Marker inventory — confirms gemma-4 native delimiters are present
///   2. End-to-end parse — gemma-4-style text chunk → `.toolCall` event
///   3. Truncation surface — incomplete body with `surfaceTruncatedToolBody: true`
///      emits `.toolCallTruncated` (coverage added in PR #50)
///   4. `setStructuredHistory` storage — the backend stores the value under its
///      state lock and exposes it via `@_spi(Testing) structuredHistoryForTesting`

// MARK: - Helpers

private func toolCalls(from events: [GenerationEvent]) -> [ToolCall] {
    events.compactMap {
        if case .toolCall(let tc) = $0 { return tc } else { return nil }
    }
}

// MARK: - Marker inventory

final class LlamaGemma4MarkerInventoryTests: XCTestCase {

    func test_markers_containsGemma4NativeOpenTag() {
        // Sabotage: change `LlamaToolMarkers.gemma4OpenTag` in LlamaToolMarkers.swift
        // to "<|tool_call|>" (trailing pipe) and this test fails — binding the fixture
        // to the real constant so any drift is immediately surfaced.
        let markers = LlamaToolMarkers.markers()
        let openTags = markers.map(\.open)
        XCTAssertTrue(openTags.contains("<|tool_call>"),
                      "markers() must include the gemma-4 native open token <|tool_call>")
    }

    func test_markers_containsGemma4NativeCloseTag() {
        let markers = LlamaToolMarkers.markers()
        // Pair: open == "<|tool_call>" should have close == "<|end_of_turn>"
        let gemma4Pair = markers.first { $0.open == "<|tool_call>" }
        XCTAssertNotNil(gemma4Pair,
                        "markers() must contain a pair whose open is <|tool_call>")
        XCTAssertEqual(gemma4Pair?.close, "<|end_of_turn>",
                       "The gemma-4 native pair must use <|end_of_turn> as its close token")
    }

    func test_markers_containsJSONFallbackPair() {
        let markers = LlamaToolMarkers.markers()
        let jsonPair = markers.first { $0.open == "<tool_call>" }
        XCTAssertNotNil(jsonPair,
                        "markers() must contain the JSON-fallback <tool_call> pair")
        XCTAssertEqual(jsonPair?.close, "</tool_call>")
    }

    func test_markers_gemma4IsFirst() {
        // Gemma-4 must come before JSON fallback so ToolCallTransform's
        // earliest-open-wins tie-break favours the native format.
        let markers = LlamaToolMarkers.markers()
        let firstOpen = markers.first?.open
        XCTAssertEqual(firstOpen, "<|tool_call>",
                       "Gemma-4 native marker must be the first entry so it wins ties")
    }
}

// MARK: - End-to-end parse (no model required)

/// Drives `ToolCallTransform` with gemma-4-style synthetic output and asserts
/// the expected `GenerationEvent` shape is emitted.
final class LlamaGemma4ToolCallParserE2ETests: XCTestCase {

    private struct Parser {
        private var transform = ToolCallTransform(markers: LlamaToolMarkers.markers())

        mutating func process(_ chunk: String) -> [GenerationEvent] {
            transform.process([.token(chunk)])
        }

        mutating func finalize() -> [GenerationEvent] {
            transform.finalize()
        }
    }

    // MARK: - Single tool call — minimal

    func test_gemma4JSON_singleTurn_emitsToolCallEvent() throws {
        // The issue background notes that `LlamaBackend` is NOT tool-aware —
        // tool-call *parsing* is done by `ToolCallTransform` in the driver.
        // This test simulates the exact token stream the driver sees when
        // gemma-4's chat template emits a native tool call.
        //
        // Sabotage: remove the gemma-4 marker pair from `LlamaToolMarkers.markers()`
        // and `calls.count` drops to 0, failing the XCTAssertEqual below.
        var parser = Parser()
        let chunk = "<|tool_call>\ncall:get_weather{city:<|\"|>Paris<|\"|>}\n<|end_of_turn>"
        let events = parser.process(chunk)

        let calls = toolCalls(from: events)
        XCTAssertEqual(calls.count, 1, "A complete gemma-4 tool-call block must emit exactly one .toolCall event")

        let tc = try XCTUnwrap(calls.first)
        XCTAssertEqual(tc.toolName, "get_weather")

        let data = try XCTUnwrap(tc.arguments.data(using: .utf8))
        let decoded = try XCTUnwrap(
            try? JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(decoded["city"] as? String, "Paris")
    }

    // MARK: - JSON fallback form (for completeness in this file)

    func test_jsonFallback_singleTurn_emitsToolCallEvent() {
        var parser = Parser()
        let chunk = #"<tool_call>{"name":"search","arguments":{"query":"swift tools"}}</tool_call>"#
        let events = parser.process(chunk)
        let calls = toolCalls(from: events)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.toolName, "search")
    }

    // MARK: - Multi-token streaming

    func test_gemma4_chunkedAcrossTagBoundary_emitsToolCallEvent() {
        var parser = Parser()
        // Split the open tag across two chunks to verify holdback / reassembly.
        var all: [GenerationEvent] = []
        all += parser.process("<|tool_")
        all += parser.process("call>\ncall:ping{}\n<|end_of_turn>")

        let calls = toolCalls(from: all)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.toolName, "ping")
    }
}

// MARK: - Truncation surfacing (PR #50)

/// Verifies that `ToolCallTransform(surfaceTruncatedToolBody: true)` emits
/// `.toolCallTruncated` when a tool-call block is never closed.
///
/// This exercises the `surfaceTruncatedToolBody` path that PR #50 wired up;
/// the default (`false`) silently discards the body — not tested here because
/// `LlamaToolCallParserTests.test_finalize_discardsPartialToolCallBlock` already
/// covers that contract.
final class LlamaGemma4TruncationTests: XCTestCase {

    func test_gemma4_incompleteBlock_withSurfaceTruncated_emitsToolCallTruncated() {
        // Open tag present, body starts, but the model was cut off before the
        // close tag — simulates token-limit truncation mid-tool-call.
        //
        // Sabotage: remove `surfaceTruncatedToolBody: true` (or change it to `false`)
        // and `truncated` will be empty, failing the XCTAssertFalse below. This guards
        // the PR #50 wiring — a ToolCallTransform constructed without the flag silently
        // discards the partial body and the test must catch that regression.
        var transform = ToolCallTransform(
            markers: LlamaToolMarkers.markers(),
            surfaceTruncatedToolBody: true)

        _ = transform.process([.token("<|tool_call>\ncall:get_weather{city:<|\"|>Lon")])
        // No close tag — finalize() must surface the partial body.
        let events = transform.finalize()

        let truncated = events.compactMap { event -> String? in
            if case .toolCallTruncated(let body) = event { return body }
            return nil
        }
        XCTAssertFalse(truncated.isEmpty,
                       "An incomplete gemma-4 tool-call block must emit .toolCallTruncated when surfaceTruncatedToolBody is true")
    }

    func test_jsonFallback_incompleteBlock_withSurfaceTruncated_emitsToolCallTruncated() {
        var transform = ToolCallTransform(
            markers: LlamaToolMarkers.markers(),
            surfaceTruncatedToolBody: true)

        _ = transform.process([.token(#"<tool_call>{"name":"partial","argu"#)])
        let events = transform.finalize()

        let truncated = events.compactMap { event -> String? in
            if case .toolCallTruncated(let body) = event { return body }
            return nil
        }
        XCTAssertFalse(truncated.isEmpty,
                       "An incomplete JSON-fallback tool-call block must emit .toolCallTruncated when surfaceTruncatedToolBody is true")
    }

    func test_surfaceTruncatedFalse_incompleteBlock_emitsNothing() {
        // Baseline: default behaviour discards without surfacing.
        var transform = ToolCallTransform(
            markers: LlamaToolMarkers.markers(),
            surfaceTruncatedToolBody: false)
        _ = transform.process([.token("<|tool_call>\ncall:lost{")])
        let events = transform.finalize()

        let truncated = events.compactMap { event -> String? in
            if case .toolCallTruncated(let body) = event { return body }
            return nil
        }
        XCTAssertTrue(truncated.isEmpty,
                      "Default surfaceTruncatedToolBody:false must silently drop incomplete blocks")
    }
}

// MARK: - setStructuredHistory storage

/// Verifies that `LlamaBackend.setStructuredHistory(_:)` stores the supplied
/// history under its state lock and that the stored value is retrievable via
/// the `@_spi(Testing)` accessor added in issue #45.
///
/// No model load is performed — the backend is instantiated and the setter is
/// called directly, exercising only the lock + assignment path.
final class LlamaBackendStructuredHistoryStorageTests: XCTestCase {

    func test_setStructuredHistory_storesMessages() {
        // Sabotage: remove the `withStateLock` guard from `structuredHistoryForTesting`
        // (or from `setStructuredHistory`) and this test still passes functionally —
        // but the guard is exercised by the concurrent-access sanitiser (TSan). To catch
        // the lock removal structurally, change `_structuredHistory = []` to not store
        // the value and `stored.count` drops to 0, failing the XCTAssertEqual below.
        let backend = LlamaBackend()
        let messages: [StructuredMessage] = [
            StructuredMessage(role: "user", content: "What is the weather in Tokyo?"),
            StructuredMessage(role: "assistant", content: "<|tool_call>\ncall:get_weather{city:<|\"|>Tokyo<|\"|>}\n<|end_of_turn>"),
        ]

        backend.setStructuredHistory(messages)

        let stored = backend.structuredHistoryForTesting
        XCTAssertEqual(stored.count, 2)
        XCTAssertEqual(stored[0].role, "user")
        XCTAssertEqual(stored[1].role, "assistant")
    }

    func test_setStructuredHistory_replacesExistingHistory() {
        let backend = LlamaBackend()

        backend.setStructuredHistory([
            StructuredMessage(role: "user", content: "First turn"),
        ])
        backend.setStructuredHistory([
            StructuredMessage(role: "user", content: "Second turn"),
            StructuredMessage(role: "assistant", content: "Response"),
        ])

        let stored = backend.structuredHistoryForTesting
        XCTAssertEqual(stored.count, 2,
                       "setStructuredHistory must replace, not append to, the prior history")
        XCTAssertEqual(stored[0].textContent, "Second turn")
    }

    func test_setStructuredHistory_withToolResultPart_isAccepted() {
        // Structural check — verifies the backend accepts a history entry
        // whose parts include a ToolResult without crashing or discarding.
        let backend = LlamaBackend()
        let toolResultMessage = StructuredMessage(
            role: "tool",
            parts: [.toolResult(ToolResult(callId: "llama-get_weather-abc123",
                                           content: "Sunny, 22 °C"))])

        backend.setStructuredHistory([toolResultMessage])

        let stored = backend.structuredHistoryForTesting
        XCTAssertEqual(stored.count, 1)
        XCTAssertEqual(stored[0].role, "tool")
    }

    func test_setStructuredHistory_withEmptyArray_clearsHistory() {
        let backend = LlamaBackend()
        backend.setStructuredHistory([
            StructuredMessage(role: "user", content: "Hello"),
        ])
        backend.setStructuredHistory([])

        XCTAssertTrue(backend.structuredHistoryForTesting.isEmpty,
                      "setStructuredHistory([]) must clear any previously stored history")
    }
}
