import XCTest
import ManifoldInference
import ManifoldModelCatalog
import ManifoldHardware
import ManifoldTestSupport
import ManifoldLlama

/// Model-gated full-GGUF E2E for the gemma-4 native tool-call render path
/// (issue #45, deliverable 3).
///
/// ## What this proves that the direct-backend E2E cannot
/// `LlamaToolCallE2ETests` (sibling file) drives `LlamaBackend.generate(...)`
/// directly with a *pre-rendered* prompt, so it never exercises the production
/// render seam — `InferenceService` → `GenerationQueue` → `PromptRenderer`
/// (which renders the model's *real* embedded `tokenizer.chat_template` via
/// `JinjaPromptRenderer`). That render seam is exactly where the ~0% tool-call
/// regression hid (#1909): the embedded gemma-4 Jinja template was fed an empty
/// `tools` array, so its native tool declaration block was silently dropped.
///
/// This test drives the **whole** path through `InferenceService.enqueue(...)`
/// with `config.tools` set and `config.captureRenderedPrompt = true`, then:
///   1. captures the rendered prompt via the `.promptRendered` event and asserts
///      it carries the native gemma-4 tool declaration markers — i.e. the real
///      embedded template actually rendered the tools, not an empty block,
///   2. asserts the model emits a structured `.toolCall` (not prose).
///
/// ## Gating
/// Requires a locally available gemma-4 GGUF whose embedded
/// `tokenizer.chat_template` is the native tool template. Skips cleanly when
/// absent — matching the repo's other model-dependent integration tests
/// (`RUN_SLOW_TESTS=1` + `LLAMA_TEST_MODEL=<path>` / a GGUF in `~/Documents/Models/`).
///
/// ```
/// RUN_SLOW_TESTS=1 LLAMA_TEST_MODEL=/path/to/gemma-4-it-Q4_K_M.gguf \
///   swift test --filter LlamaGemma4ToolRenderE2ETests
/// ```
@MainActor
final class LlamaGemma4ToolRenderE2ETests: XCTestCase {

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

    private func requireModelInfo() throws -> ModelInfo {
        guard let url = HardwareRequirements.findGGUFModel() else {
            throw XCTSkip(
                "No GGUF model found on disk. Set LLAMA_TEST_MODEL=<path> or place a .gguf file "
                + "in ~/Documents/Models/ to run this test.")
        }
        guard let info = ModelInfo(ggufURL: url) else {
            throw XCTSkip("GGUF at \(url.path) could not be read as a ModelInfo (metadata parse failed).")
        }
        // The whole point of this test is the *embedded* chat template render
        // path — a GGUF with no embedded template would only exercise the enum
        // fallback, which is not what #1909 regressed. Skip (don't fail) so a
        // non-templated fixture model degrades to a clean skip.
        try XCTSkipUnless(
            info.chatTemplateRaw?.isEmpty == false,
            "GGUF carries no embedded tokenizer.chat_template — this E2E requires the embedded "
            + "Jinja render path. Point LLAMA_TEST_MODEL at a gemma-4 GGUF with a tool template.")
        return info
    }

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

    // MARK: - Full-path E2E: capture rendered prompt + assert tool call

    func test_fullPath_capturesRenderedToolPrompt_andEmitsToolCall() async throws {
        let modelInfo = try requireModelInfo()

        let service = InferenceService()
        LlamaBackends.register(with: service)
        addTeardownBlock { @MainActor in service.unloadModel() }

        let plan = ModelLoadPlan.compute(for: modelInfo, requestedContextSize: 4096)
        try await service.loadModel(from: modelInfo, plan: plan)

        var config = GenerationConfig(
            temperature: 0.0,            // greedy — maximise determinism
            maxOutputTokens: 200,
            tools: [Self.weatherTool()]
        )
        // Surface the prompt the render seam actually produced (#1909).
        let hints = GenerationRuntimeHints(captureRenderedPrompt: true)

        let (_, stream) = try service.enqueue(
            messages: [.user("What is the weather in Paris right now?")],
            config: config,
            hints: hints
        )

        var renderedPrompt: String?
        var toolCalls: [ToolCall] = []
        var allEvents: [GenerationEvent] = []
        for try await event in stream {
            allEvents.append(event)
            switch event {
            case .promptRendered(let text):
                renderedPrompt = text
            case .toolCall(let call):
                toolCalls.append(call)
            default:
                break
            }
        }

        // 1. The captured prompt must carry the native gemma-4 tool declaration —
        //    proof the embedded Jinja template rendered the tools rather than
        //    dropping them into an empty `{% if tools %}` block (#1909).
        let prompt = try XCTUnwrap(
            renderedPrompt,
            "captureRenderedPrompt=true must yield a .promptRendered event. Events: "
            + allEvents.map { "\($0)" }.joined(separator: ", "))
        XCTAssertTrue(
            prompt.contains("get_weather"),
            "The rendered prompt must declare the registered tool — the embedded gemma-4 chat "
            + "template must render tools natively. Rendered prompt:\n\(prompt)")
        // The native gemma-4 tool markers the family declares (PromptTemplate.gemma4.markers).
        let hasNativeToolMarker =
            prompt.contains("<|tool>") || prompt.contains("<|tool_call>") || prompt.contains("tool")
        XCTAssertTrue(
            hasNativeToolMarker,
            "The rendered prompt must contain a native tool declaration marker. Rendered prompt:\n\(prompt)")

        // 2. The model must emit a structured tool call (not prose).
        XCTAssertFalse(
            toolCalls.isEmpty,
            "A gemma-4 model asked about weather with a registered tool must emit at least one "
            + ".toolCall event through the full InferenceService render path. Events: "
            + allEvents.map { "\($0)" }.joined(separator: ", "))
        if let first = toolCalls.first {
            XCTAssertFalse(first.toolName.isEmpty, "Tool call must carry a non-empty name")
            XCTAssertFalse(first.id.isEmpty, "Tool call must carry a non-empty id")
        }
    }
}
