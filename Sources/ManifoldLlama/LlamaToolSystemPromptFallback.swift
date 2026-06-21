import Foundation
import ManifoldInference

/// Harness-level mitigation for GGUF models whose chat template gives the model
/// NO way to emit a tool call.
///
/// **The problem.** Some local GGUFs (Phi-3.5-mini, the Mistral-7B-Instruct
/// GGUF) ship an embedded `tokenizer.chat_template` that only knows
/// system/user/assistant turns — it never iterates `tools` and never defines a
/// tool-call output shape. When such a model is *told* tools exist (via
/// `GenerationConfig.tools`) but its template can't render them, the model
/// improvises an unparseable format (e.g. a Pythonic
/// `tool_call(calc, "7823 * 41")`) that `ToolCallTransform` never recognises, so
/// nothing dispatches. The detection here mirrors the production
/// `PromptRenderer.rendersToolsNatively` decision exactly.
///
/// **The proper fix** belongs in ManifoldKit's `ToolSystemPromptBuilder` /
/// `GenerationQueue` fold (tracked as MK#1856): when the renderer reports the
/// template can't render tools natively, the queue should fold a canonical
/// tool-call instruction into the system prompt itself. Until that lands and
/// reaches this companion package's pin, this is the harness-level mitigation so
/// the tool-calling test campaign can exercise these models. The instruction it
/// injects targets the exact JSON dialect `LlamaToolMarkers` already parses
/// (`{"name": "<tool>", "arguments": { … }}`).
public enum LlamaToolSystemPromptFallback {

    // MARK: - Detection

    /// Whether a model with the given embedded chat template and detected
    /// template enum will render tool definitions into its prompt.
    ///
    /// This is a faithful mirror of the production
    /// `PromptRenderer.rendersToolsNatively` logic (which is `internal` to
    /// ManifoldInference and so not reachable from this companion package):
    ///
    /// - When an embedded Jinja `chat_template` is present and non-empty, the
    ///   only reliable signal is whether the template *references the `tools`
    ///   variable inside a Jinja delimiter* — a template can only render the tool
    ///   grammar by branching on / iterating `tools`, which is always inside a
    ///   `{% … %}` or `{{ … }}` block. (A bare textual `tools` in static prose or
    ///   a literal token like `<|tools|>` does NOT count.)
    /// - When there is no usable embedded template, only the `.gemma4` enum
    ///   renders tools natively; every other enum silently discards `tools`.
    ///
    /// - Parameters:
    ///   - chatTemplateRaw: the model's embedded `tokenizer.chat_template`, or
    ///     `nil`/empty when absent.
    ///   - templateRendersToolsNatively: the detected `PromptTemplate`'s
    ///     `rendersToolsNatively` value (pass `false` when no template was
    ///     detected — the conservative default, matching the renderer's ChatML
    ///     fallback which does not render tools).
    /// - Returns: `true` when tools WILL be rendered into the prompt (no
    ///   injection needed); `false` when the model is templateless w.r.t. tools
    ///   and needs the system-prompt addendum.
    public static func templateRendersTools(
        chatTemplateRaw: String?,
        templateRendersToolsNatively: Bool
    ) -> Bool {
        if let chatTemplateRaw,
           !chatTemplateRaw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return templateReferencesToolsVariable(chatTemplateRaw)
        }
        return templateRendersToolsNatively
    }

    /// Whether the template references the bare `tools` identifier inside a Jinja
    /// statement (`{% … %}`) or expression (`{{ … }}`) block.
    ///
    /// Mirrors `PromptRenderer.templateReferencesToolsVariable(_:)`: scan only
    /// Jinja-delimited regions for the word-bounded identifier `tools` so that
    /// `tool_calls` / `get_tools` / `<|tools|>` in prose do not false-positive.
    /// A heuristic, not a full parse, but it cannot false-negative a template
    /// that genuinely renders tools.
    static func templateReferencesToolsVariable(_ template: String) -> Bool {
        let scalars = Array(template.unicodeScalars)
        guard scalars.count > 1 else { return false }
        var i = 0
        while i < scalars.count - 1 {
            guard scalars[i] == "{", scalars[i + 1] == "%" || scalars[i + 1] == "{" else {
                i += 1
                continue
            }
            let closer: Unicode.Scalar = scalars[i + 1] == "%" ? "%" : "}"
            let blockStart = i + 2
            var j = blockStart
            while j < scalars.count - 1, !(scalars[j] == closer && scalars[j + 1] == "}") {
                j += 1
            }
            let blockEnd = min(j, scalars.count)
            if scalarsContainToolsWord(scalars, from: blockStart, to: blockEnd) {
                return true
            }
            i = j + 2
        }
        return false
    }

    private static func scalarsContainToolsWord(
        _ scalars: [Unicode.Scalar],
        from: Int,
        to: Int
    ) -> Bool {
        let word: [Unicode.Scalar] = ["t", "o", "o", "l", "s"]
        guard to - from >= word.count else { return false }
        var k = from
        while k <= to - word.count {
            if Array(scalars[k..<k + word.count]) == word {
                let prevIsIdent = k > from && isIdentifierScalar(scalars[k - 1])
                let nextIndex = k + word.count
                let nextIsIdent = nextIndex < to && isIdentifierScalar(scalars[nextIndex])
                if !prevIsIdent && !nextIsIdent { return true }
            }
            k += 1
        }
        return false
    }

    private static func isIdentifierScalar(_ s: Unicode.Scalar) -> Bool {
        s == "_" || s.properties.isAlphabetic || ("0"..."9").contains(s)
    }

    // MARK: - Injection

    /// Builds a concise system-prompt addendum instructing a templateless model
    /// to emit tool calls in the canonical JSON dialect `LlamaToolMarkers`
    /// already parses: `{"name": "<tool>", "arguments": { … }}`.
    ///
    /// The instruction lists each available tool with its **actual** parameter
    /// names (read from the `ToolDefinition` parameter schema) so the emitted
    /// arguments match each tool's schema (e.g. `calc` →
    /// `{"a": <number>, "op": "<string>", "b": <number>}`). The block is clearly
    /// demarcated so it reads as an addendum, not part of the scenario's own
    /// system prompt.
    ///
    /// Returns an empty string when `definitions` is empty (nothing to instruct),
    /// so callers can unconditionally append the result.
    public static func toolInstruction(for definitions: [ToolDefinition]) -> String {
        guard !definitions.isEmpty else { return "" }

        var lines: [String] = []
        lines.append("# Tool calling")
        lines.append("")
        lines.append("You have access to the following tools. When a tool is needed to answer the user, respond with ONLY a single JSON object on its own line and nothing else, in exactly this shape:")
        lines.append("")
        lines.append(#"{"name": "<tool_name>", "arguments": { <parameters> }}"#)
        lines.append("")
        lines.append("Do not wrap the JSON in code fences or prose. Use the exact parameter names listed below. Available tools:")
        lines.append("")
        for def in definitions.sorted(by: { $0.name < $1.name }) {
            lines.append("- \(def.name): \(def.description)")
            let schema = parameterSummary(for: def)
            lines.append("  arguments: \(schema)")
        }
        return lines.joined(separator: "\n")
    }

    /// Renders a tool's parameter object schema as a compact
    /// `{"name": <type>, …}` example so the model sees the exact keys and value
    /// types to emit. Falls back to `{}` for a tool with no object parameters.
    private static func parameterSummary(for definition: ToolDefinition) -> String {
        guard case .object(let root) = definition.parameters,
              case .object(let properties)? = root["properties"],
              !properties.isEmpty
        else {
            return "{}"
        }
        let required: Set<String> = {
            if case .array(let items)? = root["required"] {
                return Set(items.compactMap { if case .string(let s) = $0 { return s } else { return nil } })
            }
            return []
        }()

        // Stable, deterministic ordering so the instruction is reproducible.
        let pairs = properties.keys.sorted().map { key -> String in
            let placeholder = typePlaceholder(for: properties[key])
            let suffix = required.contains(key) ? "" : " (optional)"
            return "\"\(key)\": \(placeholder)\(suffix)"
        }
        return "{ \(pairs.joined(separator: ", ")) }"
    }

    /// Maps a parameter's JSON-Schema `type` to a human-readable placeholder the
    /// model can pattern-match (e.g. `<number>`, `<string>`).
    private static func typePlaceholder(for schema: JSONSchemaValue?) -> String {
        guard case .object(let fields)? = schema,
              case .string(let type)? = fields["type"]
        else {
            return "<value>"
        }
        switch type {
        case "integer": return "<integer>"
        case "number":  return "<number>"
        case "boolean": return "<true|false>"
        case "array":   return "<array>"
        case "object":  return "<object>"
        default:        return "<string>"
        }
    }
}
