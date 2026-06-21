import XCTest
import ManifoldInference
@testable import ManifoldLlama

/// Model-free unit tests for the harness's templateless-model mitigation
/// (`LlamaToolSystemPromptFallback`): the "does the template render tools?"
/// detector and the JSON tool-call instruction builder. No GGUF model is loaded.
final class LlamaToolSystemPromptFallbackTests: XCTestCase {

    // MARK: - Detector: embedded template present

    /// A Phi-3.5-style template: only system/user/assistant turns, never iterates
    /// `tools`. Tools will NOT render → detector must return false (inject).
    func test_detector_templatelessTemplate_doesNotRenderTools() {
        let phiLike = """
        {{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{'<|system|>\\n' + message['content'] + '<|end|>\\n'}}{% elif message['role'] == 'user' %}{{'<|user|>\\n' + message['content'] + '<|end|>\\n'}}{% else %}{{'<|assistant|>\\n' + message['content'] + '<|end|>\\n'}}{% endif %}{% endfor %}
        """
        XCTAssertFalse(
            LlamaToolSystemPromptFallback.templateRendersTools(
                chatTemplateRaw: phiLike,
                templateRendersToolsNatively: false),
            "A template that never references the tools variable must be detected as NOT rendering tools.")
    }

    /// A tool-aware ChatML/Qwen3-style template that iterates `tools` inside a
    /// Jinja block. Tools WILL render → detector returns true (do NOT inject).
    func test_detector_toolAwareTemplate_rendersTools() {
        let qwenLike = """
        {% if tools %}{% for tool in tools %}{{ tool.function.name }}{% endfor %}{% endif %}{% for message in messages %}{{ message.content }}{% endfor %}
        """
        XCTAssertTrue(
            LlamaToolSystemPromptFallback.templateRendersTools(
                chatTemplateRaw: qwenLike,
                templateRendersToolsNatively: false),
            "A template that iterates the tools variable in a Jinja block must be detected as rendering tools.")
    }

    /// `tool_calls` / `<|tools|>` outside a Jinja delimiter must NOT count — a
    /// bare textual occurrence does not render the tool grammar.
    func test_detector_toolsWordOutsideDelimiter_doesNotCount() {
        // `tool_calls` appears, but only as an attribute access — the bare
        // identifier `tools` never appears word-bounded in a delimiter.
        let template = """
        {% for message in messages %}{% if message.tool_calls %}{{ message.content }}{% endif %}{% endfor %}<|tools|>
        """
        XCTAssertFalse(
            LlamaToolSystemPromptFallback.templateRendersTools(
                chatTemplateRaw: template,
                templateRendersToolsNatively: false),
            "`tool_calls` and a literal `<|tools|>` token must not be mistaken for a tools-iterating template.")
    }

    // MARK: - Detector: no embedded template (enum fallback)

    func test_detector_noTemplate_gemma4_rendersTools() {
        XCTAssertTrue(
            LlamaToolSystemPromptFallback.templateRendersTools(
                chatTemplateRaw: nil,
                templateRendersToolsNatively: true),
            "With no embedded template, a native-tool enum (gemma4) renders tools.")
    }

    func test_detector_noTemplate_nonNativeEnum_doesNotRenderTools() {
        XCTAssertFalse(
            LlamaToolSystemPromptFallback.templateRendersTools(
                chatTemplateRaw: nil,
                templateRendersToolsNatively: false),
            "With no embedded template and a non-native enum, tools do not render.")
    }

    func test_detector_emptyTemplate_fallsBackToEnum() {
        // An empty/whitespace template is unusable → fall back to the enum flag.
        XCTAssertTrue(
            LlamaToolSystemPromptFallback.templateRendersTools(
                chatTemplateRaw: "   \n  ",
                templateRendersToolsNatively: true))
        XCTAssertFalse(
            LlamaToolSystemPromptFallback.templateRendersTools(
                chatTemplateRaw: "",
                templateRendersToolsNatively: false))
    }

    // MARK: - Instruction builder

    private func calcDefinition() -> ToolDefinition {
        ToolDefinition(
            name: "calc",
            description: "Evaluates a simple arithmetic expression.",
            parameters: .object([
                "type": .string("object"),
                "properties": .object([
                    "a": .object(["type": .string("number")]),
                    "op": .object(["type": .string("string")]),
                    "b": .object(["type": .string("number")]),
                ]),
                "required": .array([.string("a"), .string("op"), .string("b")]),
            ]))
    }

    private func readFileDefinition() -> ToolDefinition {
        ToolDefinition(
            name: "read_file",
            description: "Reads a file from the fixture root.",
            parameters: .object([
                "type": .string("object"),
                "properties": .object([
                    "path": .object(["type": .string("string")]),
                ]),
                "required": .array([.string("path")]),
            ]))
    }

    func test_instruction_emptyForNoTools() {
        XCTAssertEqual(LlamaToolSystemPromptFallback.toolInstruction(for: []), "")
    }

    func test_instruction_includesCanonicalJSONShape() {
        let text = LlamaToolSystemPromptFallback.toolInstruction(for: [calcDefinition()])
        XCTAssertTrue(text.contains(#"{"name": "<tool_name>", "arguments": { <parameters> }}"#),
                      "Instruction must spell out the exact JSON dialect LlamaToolMarkers parses.")
    }

    func test_instruction_listsRealParameterNamesAndTypes() {
        let text = LlamaToolSystemPromptFallback.toolInstruction(for: [calcDefinition()])
        XCTAssertTrue(text.contains("- calc:"), "Tool name must be listed.")
        // Real param names + their types, deterministically sorted (a, b, op).
        XCTAssertTrue(text.contains(#""a": <number>"#), "calc's `a` param (number) must be instructed.")
        XCTAssertTrue(text.contains(#""b": <number>"#), "calc's `b` param (number) must be instructed.")
        XCTAssertTrue(text.contains(#""op": <string>"#), "calc's `op` param (string) must be instructed.")
        // It must NOT invent param names that aren't in the schema. (Check the
        // `arguments:` line specifically — "expression" legitimately appears in
        // the tool's prose description.)
        let argsLine = text.split(separator: "\n").first { $0.contains("arguments:") }
        XCTAssertNotNil(argsLine)
        XCTAssertFalse(argsLine?.contains("expression") ?? true,
                       "Must not instruct a param the schema does not declare.")
    }

    func test_instruction_singleParamTool() {
        let text = LlamaToolSystemPromptFallback.toolInstruction(for: [readFileDefinition()])
        XCTAssertTrue(text.contains("- read_file:"))
        XCTAssertTrue(text.contains(#""path": <string>"#), "read_file's `path` param must be instructed.")
    }

    func test_instruction_marksOptionalParameters() {
        let def = ToolDefinition(
            name: "create_note",
            description: "Saves a note.",
            parameters: .object([
                "type": .string("object"),
                "properties": .object([
                    "title": .object(["type": .string("string")]),
                    "content": .object(["type": .string("string")]),
                ]),
                "required": .array([.string("content")]),
            ]))
        let text = LlamaToolSystemPromptFallback.toolInstruction(for: [def])
        XCTAssertTrue(text.contains(#""title": <string> (optional)"#),
                      "A non-required param must be marked optional.")
        XCTAssertTrue(text.contains(#""content": <string>"#))
        XCTAssertFalse(text.contains(#""content": <string> (optional)"#),
                       "A required param must not be marked optional.")
    }

    func test_instruction_isDeterministicAcrossToolOrder() {
        let a = LlamaToolSystemPromptFallback.toolInstruction(for: [calcDefinition(), readFileDefinition()])
        let b = LlamaToolSystemPromptFallback.toolInstruction(for: [readFileDefinition(), calcDefinition()])
        XCTAssertEqual(a, b, "Instruction must be stable regardless of input tool ordering.")
    }
}
