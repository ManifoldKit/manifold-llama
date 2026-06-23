import ManifoldInference
@_spi(BackendInternals) import ManifoldHardware

/// Infers the tool-call dialect a loaded GGUF model will emit from its
/// `general.architecture` string.
///
/// The mapping is coarse but correct for the families llama.cpp ships today:
/// it relies only on the architecture prefix, which is stable across model
/// generations within a family. A `nil` architecture (pre-load state) yields
/// `nil` so consumers can distinguish "not yet known" from "unknown family".
///
/// This is a conservative start — no chat-template parsing. A future pass can
/// narrow the dialect within a family (e.g. Hermes vs plain JSON fine-tune)
/// once the Jinja renderer exposes the selected tool block at load time.
enum LlamaToolCallDialect {

    /// Returns the dialect inferred from `architecture`, or `nil` when
    /// `architecture` is `nil` (model not yet loaded).
    static func infer(from architecture: String?) -> ToolCallDialect? {
        guard let arch = architecture?.lowercased() else { return nil }

        if arch.hasPrefix("gemma") {
            // Gemma 3 / gemma2 / gemma emit a native `<|tool_call>` block with a
            // custom `call:name{key:value}` body; closed by `<|end_of_turn>`.
            // The JSON body alternative (`<tool_call>…</tool_call>`) is also
            // supported via LlamaToolMarkers but the native path is primary.
            return ToolCallDialect(
                family: .gemma,
                openDelimiter: "<|tool_call|>",
                closeDelimiter: "<|end_of_turn>",
                argEncoding: .json,
                extractability: .buried
            )
        } else if arch.hasPrefix("qwen") {
            // Qwen2.5-Instruct emits `<tool_call>\n{json}\n</tool_call>`.
            return ToolCallDialect(
                family: .qwen,
                openDelimiter: "<tool_call>",
                closeDelimiter: "</tool_call>",
                argEncoding: .json,
                extractability: .clean
            )
        } else if arch.hasPrefix("llama") {
            // Llama 3.x bare-JSON custom tool: no open/close delimiter, the
            // call is a top-level JSON object (`{"name":…,"parameters":{…}}`).
            // Buried because there is no opening delimiter to anchor extraction.
            return ToolCallDialect(
                family: .llamaPythonTag,
                openDelimiter: nil,
                closeDelimiter: nil,
                argEncoding: .json,
                extractability: .buried
            )
        } else if arch.hasPrefix("mistral") || arch.hasPrefix("mixtral") {
            // Mistral v0.3 / Mixtral: `[TOOL_CALLS] [{…}]` with no close tag.
            return ToolCallDialect(
                family: .mistral,
                openDelimiter: "[TOOL_CALLS]",
                closeDelimiter: nil,
                argEncoding: .json,
                extractability: .buried
            )
        }

        // Unknown architecture — report the family as unknown rather than nil
        // so consumers know the backend loaded a model but can't classify it.
        return ToolCallDialect(family: .unknown, extractability: .toolLess)
    }
}
