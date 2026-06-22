import Foundation
import ManifoldModelCatalog
// JSONSchemaValue lives in ManifoldHardware (ToolTypes.swift).
import ManifoldHardware

/// Generates a GBNF grammar that constrains a model's output to a single,
/// well-formed tool call in a given dialect (#2005 follow-on spike).
///
/// **Why this exists.** `ChatTemplateToolDescriptor` (MK 0.59, Layer 1) classifies
/// a model's tool-call dialect. The most parse-prone class is `.buried` with
/// `argEncoding: .json` (Llama-3.1): the call is bare top-level JSON with no
/// delimiter, so the model is free to *narrate* a tool call ("This response is a
/// JSON object with the name of the function 'now'…") instead of emitting one a
/// host can parse. A GBNF grammar removes that freedom: it constrains the FIRST
/// real output token onward to a valid call object, so the model cannot prose its
/// way out of a parseable call.
///
/// **Honest scope.** A grammar constrains tool-call *syntax / dispatch* — whether
/// the emitted bytes parse into a `ToolCall`. It does NOT constrain second-turn
/// *grounding* (whether the model then uses the tool's result in its final
/// answer); grounding is a separate, downstream property. See the spike report.
///
/// **Wire-format coupling (load-bearing).** Llama-3.1's bare-JSON call is anchored
/// by `LlamaToolMarkers.llama3OpenTag` == `{"name"` — i.e. the parser keys on
/// *compact* JSON with NO space after the opening brace and NO space after the
/// `"name"` key's colon-less prefix. The emitted grammar therefore produces
/// **compact JSON** (`{"name":...}`, no leading whitespace) so a grammar-valid
/// call also satisfies the scanner's anchor. A pretty-printing grammar would emit
/// valid JSON that the Llama scanner would nonetheless miss.
public enum ToolCallGrammar {

    /// Errors a caller can distinguish from "grammar produced".
    public enum GenerationError: Error, Equatable {
        /// No tools were advertised — there is no call to constrain to.
        case emptyToolList
        /// The dialect's `argEncoding` is recognised but not yet implemented by
        /// this generator. Carries the encoding so a caller can log/branch.
        case unsupportedEncoding(ChatTemplateToolDescriptor.ArgEncoding)
    }

    /// A minimal description of one advertised tool: its name and, optionally, its
    /// JSON-Schema parameter object. When `parametersSchema` is `nil` the grammar
    /// falls back to a permissive (any-shape) JSON object for that tool's args —
    /// still a valid call envelope, just not key-constrained.
    public struct Tool: Sendable, Equatable {
        public let name: String
        /// The tool's JSON-Schema parameters object (the `parameters` field of a
        /// `ToolDefinition`). Currently used only to detect the "no properties"
        /// case; per-key argument constraining is a documented TODO below.
        public let parametersSchema: JSONSchemaValue?

        public init(name: String, parametersSchema: JSONSchemaValue? = nil) {
            self.name = name
            self.parametersSchema = parametersSchema
        }
    }

    /// Builds a GBNF grammar constraining output to a valid tool call in `dialect`
    /// for one of `tools`.
    ///
    /// - The `argEncoding` selects the body shape (only `.json` is implemented this
    ///   spike — `.keyValue` / `.keyEqualsValue` throw `unsupportedEncoding`).
    /// - `openDelimiter` / `closeDelimiter`, when present, wrap the call (so the
    ///   generator is dialect-parameterised and will extend to Qwen/Mistral/Gemma
    ///   once their body encodings are filled in). For Llama-3.1 both are `nil`,
    ///   yielding a bare top-level object.
    ///
    /// - Throws: ``GenerationError/emptyToolList`` when `tools` is empty;
    ///   ``GenerationError/unsupportedEncoding`` for `.keyValue` / `.keyEqualsValue`.
    public static func grammar(
        for dialect: ChatTemplateToolDescriptor.ToolCallDialect,
        tools: [Tool]
    ) throws -> String {
        guard !tools.isEmpty else { throw GenerationError.emptyToolList }

        switch dialect.argEncoding {
        case .json:
            return jsonGrammar(
                openDelimiter: dialect.openDelimiter,
                closeDelimiter: dialect.closeDelimiter,
                tools: tools
            )
        case .keyValue, .keyEqualsValue:
            // TODO: implement the Gemma `<|tool_call>` newline `key: value` body
            // (.keyValue) and the Llama python-tag `key=value` body
            // (.keyEqualsValue). The envelope/delimiter scaffolding above is
            // already dialect-parameterised; only the body production differs.
            throw GenerationError.unsupportedEncoding(dialect.argEncoding)
        }
    }

    /// Convenience overload taking advertised tool *names* only (no schemas).
    public static func grammar(
        for dialect: ChatTemplateToolDescriptor.ToolCallDialect,
        toolNames: [String]
    ) throws -> String {
        try grammar(for: dialect, tools: toolNames.map { Tool(name: $0) })
    }

    // MARK: - JSON body

    /// Emits the `.json` grammar:
    ///
    /// ```gbnf
    /// root      ::= "<open>" call "<close>"          # delimiters omitted when nil
    /// call      ::= "{\"name\":" name ",\"arguments\":" args "}"
    /// name      ::= "\"now\"" | "\"calc\"" | …       # enum of advertised names
    /// args      ::= object                           # any JSON object
    /// ```
    ///
    /// The `name` production is a literal enum of the advertised tool names, so the
    /// model can only ever name a real tool. The args object is an unconstrained
    /// (but well-formed) JSON object — enough to guarantee a parseable call; per-key
    /// constraining from `parametersSchema` is a documented follow-up.
    ///
    /// `arguments` is the emitted key (Llama-3.1's native template uses
    /// `parameters`, but `LlamaToolMarkers` accepts both via the #76 alias, and
    /// `arguments` is the cross-dialect canonical key — so a single grammar works
    /// for Llama's parser and any Qwen/Hermes/Mistral extension).
    private static func jsonGrammar(
        openDelimiter: String?,
        closeDelimiter: String?,
        tools: [Tool]
    ) -> String {
        let nameAlternatives = tools
            .map { "\"\\\"\($0.name)\\\"\"" }   // GBNF string literal of `"name"`
            .joined(separator: " | ")

        // Root wraps the call in the dialect's delimiters when it has them. The
        // bare-JSON (Llama-3.1) case has neither, so root IS the call object.
        let rootRHS: String
        if let open = openDelimiter, let close = closeDelimiter {
            rootRHS = "\(gbnfLiteral(open)) call \(gbnfLiteral(close))"
        } else if let open = openDelimiter {
            rootRHS = "\(gbnfLiteral(open)) call"
        } else {
            rootRHS = "call"
        }

        // Compact JSON (no whitespace after `{` or after `:`) so the produced
        // bytes satisfy LlamaToolMarkers.llama3OpenTag (`{"name"`). ws is permitted
        // *inside* the args object's standard JSON productions only.
        return """
        # GBNF tool-call grammar (dialect: argEncoding=.json) — spike #2005 follow-on.
        # Constrains output to a single valid tool call naming one advertised tool.
        root    ::= \(rootRHS)
        call    ::= "{\\"name\\":" name ",\\"arguments\\":" object "}"
        name    ::= \(nameAlternatives)

        # --- Generic well-formed JSON value productions (args object) ---
        object  ::= "{" ws ( member ( ws "," ws member )* )? ws "}"
        member  ::= string ws ":" ws value
        array   ::= "[" ws ( value ( ws "," ws value )* )? ws "]"
        value   ::= object | array | string | number | "true" | "false" | "null"
        string  ::= "\\"" char* "\\""
        char    ::= [^"\\\\] | "\\\\" escape
        escape  ::= ["\\\\/bfnrt] | "u" hex hex hex hex
        hex     ::= [0-9a-fA-F]
        number  ::= "-"? int frac? exp?
        int     ::= "0" | [1-9] [0-9]*
        frac    ::= "." [0-9]+
        exp     ::= [eE] [-+]? [0-9]+
        ws      ::= [ \\t\\n]*
        """
    }

    /// Escapes an arbitrary delimiter string as a GBNF double-quoted literal.
    /// Backslash and double-quote are the only chars that need escaping inside a
    /// GBNF string literal.
    private static func gbnfLiteral(_ s: String) -> String {
        let escaped = s
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        return "\"\(escaped)\""
    }
}
