import XCTest
import ManifoldInference
import ManifoldTestSupport
import ManifoldLlama
@_spi(Testing) import ManifoldLlama

/// Model-gated E2E scaffolds for `LlamaBackend`'s local tool-call path.
///
/// ## Issue #45
/// `LlamaBackend`'s local tool-call path had zero E2E coverage prior to this
/// file. The only tool-calling E2E ran against Ollama (`ToolCallingHistoryReceiver`),
/// leaving the prompt-template render path exercised by `LlamaBackend` untested.
///
/// ## Architecture recap
/// `LlamaBackend` is NOT a `ToolCallingHistoryReceiver`. Tool definitions reach it
/// pre-baked into the rendered prompt string via `InferenceService`'s
/// `PromptAssembler`. The backend generates text; `LlamaGenerationDriver` parses
/// tool calls via `ToolCallTransform`. Tool results for multi-turn round-trips flow
/// via `setStructuredHistory()` / `StructuredHistoryReceiver`.
///
/// ## Gating
/// These tests require a locally available gemma-4 GGUF and are skipped unless:
///   - `RUN_SLOW_TESTS=1` is set in the environment, **and**
///   - `LLAMA_TEST_MODEL` is set to an absolute path to a gemma-4-compatible GGUF
///     (or a GGUF exists in `~/Documents/Models/`).
///
/// Example:
/// ```
/// RUN_SLOW_TESTS=1 LLAMA_TEST_MODEL=/path/to/gemma-4-it-Q4_K_M.gguf \
///   swift test --filter LlamaToolCallE2ETests
/// ```
final class LlamaToolCallE2ETests: XCTestCase {

    // MARK: - Gate / setup

    override func setUp() async throws {
        try await super.setUp()
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["RUN_SLOW_TESTS"] == "1",
            "Set RUN_SLOW_TESTS=1 to run model-gated E2E tests")
        try XCTSkipUnless(
            HardwareRequirements.isPhysicalDevice,
            "LlamaBackend requires Metal (unavailable in simulator)")
        try XCTSkipUnless(
            HardwareRequirements.isAppleSilicon,
            "LlamaBackend requires Apple Silicon")
    }

    private func requireModelURL() throws -> URL {
        guard let url = HardwareRequirements.findGGUFModel() else {
            throw XCTSkip(
                "No GGUF model found on disk. Set LLAMA_TEST_MODEL=<path> or place a .gguf file "
                + "in ~/Documents/Models/ to run this test.")
        }
        return url
    }

    // MARK: - Single-turn: tool-call event is emitted

    /// Single-turn E2E: load a gemma-4 model, generate with a non-empty tool
    /// set, drain the stream, and assert at least one `.toolCall` event fires.
    ///
    /// This exercises the full prompt-template render → generation → `ToolCallTransform`
    /// parsing pipeline that issue #45 identified as having zero coverage.
    func test_singleTurn_localToolCall_emitsToolCallEvent() async throws {
        let modelURL = try requireModelURL()

        let backend = LlamaBackend()
        addTeardownBlock { await backend.unloadAndWait() }
        try await backend.loadModel(from: modelURL, plan: .testStub(effectiveContextSize: 4096))

        // Build a minimal tool definition for the weather query scenario.
        let weatherTool = ToolDefinition(
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

        // Pre-render the prompt the way InferenceService's PromptAssembler would:
        // tool definitions are baked into the prompt string before the backend
        // receives it (LlamaBackend is NOT tool-aware at the wire level).
        let prompt = PromptTemplate.gemma4.format(
            messages: [(role: "user", content: "What is the weather in Paris right now?")],
            systemPrompt: nil,
            tools: [weatherTool]
        )

        var config = GenerationConfig()
        config.maxOutputTokens = 200
        config.temperature = 0.0 // greedy — maximises determinism

        var collectedEvents: [GenerationEvent] = []
        let stream = try backend.generate(prompt: prompt, systemPrompt: nil, config: config)
        for try await event in stream {
            collectedEvents.append(event)
        }

        let toolCallEvents = collectedEvents.compactMap { event -> ToolCall? in
            if case .toolCall(let tc) = event { return tc } else { return nil }
        }

        XCTAssertFalse(
            toolCallEvents.isEmpty,
            "A gemma-4 model asked about weather with a registered tool must emit at least one "
            + ".toolCall event. Collected events: "
            + collectedEvents.map { "\($0)" }.joined(separator: ", "))

        if let firstCall = toolCallEvents.first {
            XCTAssertFalse(firstCall.toolName.isEmpty, "Tool call must carry a non-empty name")
            XCTAssertFalse(firstCall.id.isEmpty, "Tool call must carry a non-empty ID")
        }
    }

    // MARK: - Multi-turn: tool result accepted without error

    /// Multi-turn structural check: after a tool result is fed back via
    /// `setStructuredHistory`, assert the backend accepts it without error.
    ///
    /// Content verification is intentionally omitted — the model's follow-up
    /// response to a tool result is model-weight-dependent and not stable enough
    /// for a deterministic assertion. This test validates the structural wiring
    /// only: the backend must accept the history, store it, and start a second
    /// generation without throwing.
    func test_multiTurn_toolResult_acceptedByBackend() async throws {
        let modelURL = try requireModelURL()

        let backend = LlamaBackend()
        addTeardownBlock { await backend.unloadAndWait() }
        try await backend.loadModel(from: modelURL, plan: .testStub(effectiveContextSize: 4096))

        let weatherTool = ToolDefinition(
            name: "get_weather",
            description: "Returns current weather for a location",
            parameters: .object([
                "type": .string("object"),
                "properties": .object([
                    "location": .object(["type": .string("string")])
                ]),
                "required": .array([.string("location")]),
            ])
        )

        var config = GenerationConfig()
        config.maxOutputTokens = 200
        config.temperature = 0.0

        // Turn 1: generate a tool call.
        let turn1Prompt = PromptTemplate.gemma4.format(
            messages: [(role: "user", content: "What is the weather in Tokyo?")],
            systemPrompt: nil,
            tools: [weatherTool]
        )

        var turn1Events: [GenerationEvent] = []
        let stream1 = try backend.generate(prompt: turn1Prompt, systemPrompt: nil, config: config)
        for try await event in stream1 {
            turn1Events.append(event)
        }

        // Extract the tool call emitted in turn 1 (if any); fall back to a
        // synthetic ID so the structural assertion below still runs even when
        // the model chose not to call the tool.
        let callId = turn1Events.compactMap { event -> String? in
            if case .toolCall(let tc) = event { return tc.id } else { return nil }
        }.first ?? "llama-get_weather-synthetic"

        // Turn 2: feed back a synthetic tool result via setStructuredHistory,
        // then issue a follow-up generation. The backend must not throw.
        let history: [StructuredMessage] = [
            StructuredMessage(role: "user", content: "What is the weather in Tokyo?"),
            StructuredMessage(
                role: "assistant",
                parts: [.text("<|tool_call>\ncall:get_weather{city:<|\"|>Tokyo<|\"|>}\n<|end_of_turn>")]),
            StructuredMessage(
                role: "tool",
                parts: [.toolResult(ToolResult(
                    callId: callId,
                    content: "Sunny, 28 °C in Tokyo"))]),
        ]

        backend.setStructuredHistory(history)

        // Structural assertion: the history was stored under the state lock.
        let stored = backend.structuredHistoryForTesting
        XCTAssertEqual(stored.count, 3,
                       "Backend must store all three structured history entries")

        // Turn 2 prompt: follow-up asking the model to use the tool result.
        let turn2Prompt = PromptTemplate.gemma4.format(
            messages: [
                (role: "user",      content: "What is the weather in Tokyo?"),
                (role: "assistant", content: "Let me check the weather for you."),
                (role: "tool",      content: "Sunny, 28 °C"),
                (role: "user",      content: "Thanks — should I bring an umbrella?"),
            ],
            systemPrompt: nil,
            tools: [weatherTool]
        )

        // Drain the second generation; the key assertion is that it does NOT throw.
        var turn2Events: [GenerationEvent] = []
        let stream2 = try backend.generate(prompt: turn2Prompt, systemPrompt: nil, config: config)
        for try await event in stream2 {
            turn2Events.append(event)
        }

        XCTAssertFalse(turn2Events.isEmpty,
                       "Second-turn generation must produce at least one event after a tool result is fed back")
    }
}
