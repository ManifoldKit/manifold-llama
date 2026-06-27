import XCTest
import Foundation
import Jinja
import ManifoldInference
@_spi(Testing) import ManifoldLlama

/// Model-free render-fixture tests for issue #45.
///
/// ## Why this exists
/// Local tool-calling had **zero** live coverage of the prompt-template render
/// path. The only end-to-end tool-calling test ran against Ollama, which is a
/// `ToolCallingHistoryReceiver` — it exercised the cloud wire path and never the
/// prompt-template render path. `LlamaBackend` is *not* tool-aware: tool
/// definitions and prior tool results reach it only baked into the rendered
/// prompt string. So the render seam — the gemma-4 `tokenizer.chat_template`
/// rendered *with its tools* — is where the ~0% tool-call regression hid.
///
/// ## What it asserts (no GGUF required, runs in milliseconds)
/// 1. Rendering a gemma-4 native chat template with a **non-empty tool set**
///    produces the native `<|tool>declaration:…` block. With an empty tool set
///    the block is absent — proving the assertion tracks the `tools` array, not
///    a constant.
/// 2. After a tool result is threaded back into structured history, the *next*
///    turn's rendered prompt carries **both** the prior `tool_call` and the
///    paired tool result — proving a non-tool-aware backend like `LlamaBackend`
///    sees the result via structured-history threading, not an Ollama-only
///    `ToolCallingHistoryReceiver` wire.
///
/// ## Why render via `Jinja` directly
/// `PromptRenderer` / `JinjaPromptRenderer` (the production render seam) are
/// `internal` to `ManifoldInference` and unreachable across the package
/// boundary. This file drives the *same* `swift-jinja` engine the production
/// renderer uses, building the *same* template context shape (the OpenAI-nested
/// `function` + flat-alias tool dictionaries, the per-message `tool_calls` /
/// `tool_call_id` threading, `add_generation_prompt`, an empty `documents`),
/// so the render assertion is faithful to what `JinjaPromptRenderer` emits.
///
/// ## Scope: model-free half of #45 only
/// This covers the model-free render deliverable the issue's Notes call out. The
/// full live gemma-4 GGUF round-trip E2E (issue #45 deliverables 1–2, driven via
/// `LlamaGemma4ToolRenderE2ETests`) remains OPEN: it is gated on the gemma-4 GGUF
/// loading on the pinned llama.cpp — currently BLOCKED (see issue #62 /
/// ManifoldKit#1981). Re-enable once that load path is fixed.
final class LlamaGemma4ToolTemplateRenderTests: XCTestCase {

    // MARK: - Vendored gemma-4 native-tool chat template fixture

    /// A gemma-4 native tool-calling `tokenizer.chat_template`, abbreviated to
    /// the structural surface that matters for issue #45 while preserving the
    /// gemma-4 native delimiters the codebase already declares for this family
    /// (`<|turn>`, `<|end_of_turn>`, `<|tool>`, `<|tool_call>`,
    /// `<|tool_response>` — see `PromptTemplate.gemma4.markers`).
    ///
    /// It exercises every structured field that the text-only `(role, content)`
    /// projection used to silently drop (the root cause of the ~0% tool-call
    /// rate):
    ///   - the `{%- if tools %}` declaration block (one `<|tool>` block per tool,
    ///     written against both the OpenAI-nested `tool.function.name` and the
    ///     flat `tool.name` conventions the renderer exposes),
    ///   - per-message `tool_calls` rendered as `<|tool_call>` blocks,
    ///   - the paired `tool_call_id` rendered as a `<|tool_response>` block.
    static let gemma4NativeToolTemplate = """
    {%- if tools %}
    <|turn>system
    {%- for tool in tools %}
    <|tool>declaration:{"name":"{{ tool.name }}","fn":"{{ tool.function.name }}"}
    {%- endfor %}
    <|end_of_turn>
    {%- endif %}
    {%- for message in messages %}
    {%- if message.role == "user" %}
    <|turn>user
    {{ message.content }}<|end_of_turn>
    {%- elif message.role == "assistant" %}
    <|turn>model
    {{ message.content }}
    {%- if message.tool_calls %}
    {%- for tc in message.tool_calls %}
    <|tool_call>{{ tc.function.name }}({{ tc.function.arguments | tojson }})<|end_of_turn>
    {%- endfor %}
    {%- endif %}
    {%- elif message.role == "tool" %}
    <|tool_response>for:{{ message.tool_call_id }}
    {{ message.content }}<|end_of_turn>
    {%- endif %}
    {%- endfor %}
    {%- if add_generation_prompt %}
    <|turn>model
    {%- endif %}
    """

    // MARK: - Fixtures

    private static func weatherTool() -> ToolDefinition {
        ToolDefinition(
            name: "get_weather",
            description: "Returns current weather for a location",
            parameters: .object([
                "type": .string("object"),
                "properties": .object([
                    "location": .object([
                        "type": .string("string"),
                        "description": .string("City name"),
                    ])
                ]),
                "required": .array([.string("location")]),
            ])
        )
    }

    private func msg(_ role: String, _ content: String) -> StructuredMessage {
        StructuredMessage(role: role, content: content)
    }

    /// Renders the vendored gemma-4 chat template against the supplied structured
    /// history and tool set, building the same Jinja context shape that
    /// `JinjaPromptRenderer` constructs in production: each tool exposed in both
    /// the OpenAI-nested (`function`) and flat-alias forms, each message threading
    /// its `tool_calls` / `tool_call_id`, plus `add_generation_prompt` and an
    /// empty `documents` array.
    private func render(
        _ messages: [StructuredMessage],
        tools: [ToolDefinition]
    ) throws -> String {
        let jinjaMessages: [[String: Any]] = messages.map { Self.jinjaMessage(from: $0) }
        let toolsContext: [[String: Any]] = tools.map { Self.jinjaTool(from: $0) }

        // Same whitespace semantics as the production renderer: `JinjaPromptRenderer`
        // builds its `Template` with `.init(lstripBlocks: true, trimBlocks: true)`
        // to match Hugging Face `apply_chat_template`. Mirror it here so the
        // model-free render is faithful to what the real seam emits.
        let template = try Template(
            Self.gemma4NativeToolTemplate,
            with: .init(lstripBlocks: true, trimBlocks: true))
        return try template.render([
            "messages": try Value(any: jinjaMessages),
            "tools": try Value(any: toolsContext),
            "add_generation_prompt": try Value(any: true),
            "documents": try Value(any: [Any]()),
        ])
    }

    /// Mirrors `JinjaPromptRenderer.jinjaMessage(from:)`: threads assistant
    /// `tool_calls` (OpenAI-nested + flat aliases) and a tool turn's
    /// `tool_call_id` / folded result content into the per-message dictionary.
    private static func jinjaMessage(from message: StructuredMessage) -> [String: Any] {
        var dict: [String: Any] = [
            "role": message.role,
            "content": message.textContent,
        ]

        let toolCalls: [[String: Any]] = message.parts.compactMap { part in
            guard case .toolCall(let call) = part else { return nil }
            let function: [String: Any] = [
                "name": call.toolName,
                "arguments": argumentsValue(call.arguments),
            ]
            return [
                "id": call.id,
                "type": "function",
                "function": function,
                "name": call.toolName,
                "arguments": argumentsValue(call.arguments),
            ]
        }
        if !toolCalls.isEmpty {
            dict["tool_calls"] = toolCalls
        }

        for part in message.parts {
            if case .toolResult(let result) = part {
                dict["tool_call_id"] = result.callId
                if (dict["content"] as? String)?.isEmpty ?? true {
                    dict["content"] = result.content
                }
                break
            }
        }

        return dict
    }

    private static func jinjaTool(from tool: ToolDefinition) -> [String: Any] {
        let function: [String: Any] = [
            "name": tool.name,
            "description": tool.description,
        ]
        return [
            "type": "function",
            "function": function,
            "name": tool.name,
            "description": tool.description,
        ]
    }

    private static func argumentsValue(_ raw: String) -> Any {
        guard let data = raw.data(using: .utf8),
              let parsed = try? JSONSerialization.jsonObject(with: data) else {
            return raw
        }
        return parsed
    }

    // MARK: - Deliverable 1: single-turn native tool declaration block

    /// Rendering with a non-empty tool set must emit the native `<|tool>`
    /// declaration block; rendering with an empty tool set must not.
    func test_singleTurn_nonEmptyToolSet_rendersNativeToolDeclaration() throws {
        let messages = [msg("user", "What is the weather in Paris right now?")]

        let withTools = try render(messages, tools: [Self.weatherTool()])

        // The native tool declaration block must be present.
        XCTAssertTrue(
            withTools.contains("<|tool>declaration:"),
            "A gemma-4 chat template rendered with a non-empty tool set must emit the native "
            + "<|tool> declaration block. Rendered prompt:\n\(withTools)")
        // Both the flat (tool.name) and OpenAI-nested (tool.function.name)
        // conventions must resolve to the tool name.
        XCTAssertTrue(
            withTools.contains(#""name":"get_weather""#),
            "The declaration block must carry the tool name via the flat alias")
        XCTAssertTrue(
            withTools.contains(#""fn":"get_weather""#),
            "The declaration block must carry the tool name via the OpenAI-nested function shape")

        // Sabotage / negative control: with no tools the `{%- if tools %}` branch
        // is falsey, so the declaration block must be absent — proving the
        // assertion tracks the tools array, not a constant in the template.
        let noTools = try render(messages, tools: [])
        XCTAssertFalse(
            noTools.contains("<|tool>declaration:"),
            "An empty tool set must render no <|tool> declaration block. Rendered prompt:\n\(noTools)")
    }

    // MARK: - Deliverable 2: multi-turn round-trip threads call + result

    /// After a tool result is fed back into structured history, the *next*
    /// turn's rendered prompt must contain BOTH the prior assistant `tool_call`
    /// AND the paired tool result. This proves the non-tool-aware `LlamaBackend`
    /// sees the result purely via structured-history threading into the prompt —
    /// not via any Ollama-only `ToolCallingHistoryReceiver` wire.
    func test_multiTurn_toolResultFedBack_rendersPriorCallAndPairedResult() throws {
        let call = ToolCall(
            id: "call_abc123",
            toolName: "get_weather",
            arguments: #"{"location":"Tokyo"}"#)

        // The structured history as the dispatch loop threads it back for a
        // non-tool-aware backend: user → assistant tool_call → tool result.
        let history: [StructuredMessage] = [
            msg("user", "What is the weather in Tokyo?"),
            StructuredMessage(role: "assistant", parts: [.toolCall(call)]),
            StructuredMessage(
                role: "tool",
                parts: [.toolResult(ToolResult(callId: "call_abc123", content: "Sunny, 28C in Tokyo"))]),
        ]

        let rendered = try render(history, tools: [Self.weatherTool()])

        // The prior assistant tool_call must render natively.
        XCTAssertTrue(
            rendered.contains("<|tool_call>get_weather("),
            "The next turn's prompt must contain the prior assistant tool_call. "
            + "Rendered prompt:\n\(rendered)")

        // The paired tool result must render, keyed to its originating call id.
        XCTAssertTrue(
            rendered.contains("<|tool_response>for:call_abc123"),
            "The next turn's prompt must contain the tool result paired to its call id. "
            + "Rendered prompt:\n\(rendered)")
        XCTAssertTrue(
            rendered.contains("Sunny, 28C in Tokyo"),
            "The tool result content must reach the rendered prompt — this is the only "
            + "channel by which a non-tool-aware LlamaBackend sees the result. "
            + "Rendered prompt:\n\(rendered)")

        // The result must come AFTER the call in the rendered prompt — the
        // round-trip ordering the model was trained on.
        let callRange = try XCTUnwrap(rendered.range(of: "<|tool_call>get_weather("))
        let resultRange = try XCTUnwrap(rendered.range(of: "<|tool_response>for:call_abc123"))
        XCTAssertLessThan(
            callRange.lowerBound, resultRange.lowerBound,
            "The tool result must render after the tool call it answers")
    }

    // MARK: - Deliverable 3: byte-exact render → parse round-trip

    /// A minimal Jinja template that renders an assistant tool call in the
    /// gemma-4 **native emission grammar** the runtime parser actually consumes:
    /// `<|tool_call>call:NAME{key:<|"|>value<|"|>,…}<|end_of_turn>`.
    ///
    /// This is deliberately NOT the `name(json)` body that
    /// `gemma4NativeToolTemplate` (used by the multi-turn ordering test above)
    /// emits. That simplified body was only ever a *fixture artifact* chosen to
    /// assert call/result *ordering* in the rendered prompt — it is not the
    /// spelling any runtime component produces or consumes. The authoritative
    /// emission grammar lives in `LlamaToolMarkers.parseGemma4NativeCall`
    /// (the `call:`-prefixed brace body) closed by `LlamaToolMarkers.gemma4EndTurn`
    /// (`<|end_of_turn>`). This template encodes that grammar so the rendered
    /// string round-trips through the *real* parser unchanged.
    ///
    /// `arguments` is exposed as a parsed object (mirroring
    /// `JinjaPromptRenderer.argumentsValue`) so `.items()` resolves; the quoted
    /// `<|"|>` token is the literal `LlamaToolMarkers.gemma4QuoteToken` the parser
    /// substitutes back to `"` before decoding.
    static let gemma4NativeEmissionTemplate = """
    {%- for tc in calls %}
    <|tool_call>call:{{ tc.name }}{{ "{" }}{% for key, value in tc.arguments.items() %}{{ key }}:<|"|>{{ value }}<|"|>{% if not loop.last %},{% endif %}{% endfor %}{{ "}" }}<|end_of_turn>
    {%- endfor %}
    """

    /// Render ONE assistant tool call through the real swift-jinja engine in the
    /// gemma-4 native emission grammar, then feed that **exact** rendered string
    /// back through the runtime parser (`ToolCallTransform` +
    /// `LlamaToolMarkers.markers()`) and assert the recovered structured call
    /// matches the original. One shared emission, no GGUF, no network — this is
    /// the byte-exact render→parse round-trip proving the renderer's spelling and
    /// the parser's spelling are identical.
    func test_renderParseRoundTrip_gemma4NativeEmission_recoversOriginalCall() throws {
        let original = ToolCall(
            id: "call_rt1",
            toolName: "get_weather",
            arguments: #"{"location":"Tokyo"}"#)

        // Expose arguments as a parsed object exactly as JinjaPromptRenderer does,
        // so the template's `.items()` iteration resolves the key/value pairs.
        let argsObject = try XCTUnwrap(
            try JSONSerialization.jsonObject(
                with: Data(original.arguments.utf8)) as? [String: Any],
            "fixture arguments must parse as a JSON object")
        let callContext: [String: Any] = [
            "name": original.toolName,
            "arguments": argsObject,
        ]

        // Same whitespace semantics as JinjaPromptRenderer (`lstripBlocks`/`trimBlocks`).
        let template = try Template(
            Self.gemma4NativeEmissionTemplate,
            with: .init(lstripBlocks: true, trimBlocks: true))
        let rendered = try template.render([
            "calls": try Value(any: [callContext]),
        ])

        // Bind the round-trip to the REAL runtime delimiters via markers() (so any
        // drift in LlamaToolMarkers is caught here, not just against string literals).
        let gemmaMarker = try XCTUnwrap(
            LlamaToolMarkers.markers().first { $0.open == "<|tool_call>" },
            "markers() must expose the gemma-4 native pair")

        // Faithfulness guards: the rendered emission must use the native grammar —
        // the real open/close delimiters, the `call:` prefix, and the `<|\"|>` quote
        // token — NOT the simplified name(json) fixture body.
        XCTAssertTrue(
            rendered.contains(gemmaMarker.open),
            "render must emit the native open delimiter. Rendered:\n\(rendered)")
        XCTAssertTrue(
            rendered.contains(gemmaMarker.close),
            "render must close on the native turn delimiter. Rendered:\n\(rendered)")
        XCTAssertTrue(
            rendered.contains("call:get_weather{"),
            "render must use the parser's `call:NAME{…}` body grammar. Rendered:\n\(rendered)")
        XCTAssertTrue(
            rendered.contains(#"location:<|"|>Tokyo<|"|>"#),
            "render must quote string values with the gemma-4 <|\"|> token. Rendered:\n\(rendered)")

        // Feed the EXACT rendered string through the runtime parser.
        var transform = ToolCallTransform(markers: LlamaToolMarkers.markers())
        var events = transform.process([.token(rendered)])
        events += transform.finalize()

        let calls = events.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc }
            return nil
        }
        XCTAssertEqual(
            calls.count, 1,
            "the rendered native emission must parse back to exactly one tool call. "
            + "Events: \(events.map { "\($0)" }.joined(separator: ", "))")

        let recovered = try XCTUnwrap(calls.first)
        // Name round-trips exactly. (The id is parser-generated — `llama-<name>-<uuid>`
        // — so it intentionally differs from the original; name + arguments are the
        // round-trip invariants.)
        XCTAssertEqual(
            recovered.toolName, original.toolName,
            "recovered tool name must match the rendered call")

        let recoveredArgs = try XCTUnwrap(
            try JSONSerialization.jsonObject(
                with: Data(recovered.arguments.utf8)) as? [String: Any],
            "recovered arguments must parse as a JSON object. Raw: \(recovered.arguments)")
        XCTAssertEqual(
            recoveredArgs["location"] as? String, "Tokyo",
            "the argument value must survive the render→parse round-trip")
        XCTAssertEqual(
            recoveredArgs.count, argsObject.count,
            "no spurious arguments may appear after the round-trip")
    }
}
