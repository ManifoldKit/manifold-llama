import Foundation
import ManifoldInference

/// Llama tool-call dialects for ``ToolCallTransform``.
///
/// Replaces the former `LlamaToolCallParser`: the scanning, N-candidate
/// earliest-open-wins selection, holdback, and chunk safety now live once in
/// `ToolCallTransform`. Only the two Llama-specific delimiter pairs and their
/// body parsers stay here, injected as `@Sendable` closures.
///
/// Two dialects, in preference order (Gemma-4 native before JSON fallback —
/// `ToolCallTransform` resolves ties by array order, matching the original
/// parser's earliest-of-two behavior):
///
/// 1. **Gemma 4 native** — `<|tool_call>` … `<tool_call|>` with a
///    `call:name{key:<|"|>value<|"|>}` brace body. The close delimiter is the
///    `<tool_call|>` token (pipe BEFORE the angle bracket), NOT `<|end_of_turn>`:
///    a real gemma-4 GGUF emits `<|tool_call>call:NAME{…}<tool_call|>` and then
///    an end-of-turn EOG *special token* (which the driver consumes as EOG, never
///    as literal text). Pairing the open with `<|end_of_turn>` — a string the
///    model never emits as text — left the whole call body held back and dropped
///    at finalize (0 dispatches). Verified byte-for-byte against the text-only
///    gemma-4-E4B GGUF.
/// 2. **JSON fallback** — `<tool_call>` … `</tool_call>` with a
///    `{"name":…,"arguments":…}` body (Qwen-style fine-tunes). The SAME
///    delimiter pair also carries Qwen3.5's nested-**XML** body
///    (`<function=name><parameter=key>value</parameter></function>`, #158);
///    `parseCallBuffer` dispatches on the body shape, so both Qwen generations
///    share one marker.
/// 3. **Mistral `[TOOL_CALLS]`** — a literal `[TOOL_CALLS]` open token followed
///    by a bare JSON *array* of `{"name":…,"arguments":…}` objects with NO
///    closing tag; the block ends at EOS/end-of-generation. Uses the #1982
///    EOS-keyed close (`closesAtEnd: true`) + multi-call body parser
///    (`parseBodyMulti`) so every element of the array becomes a `ToolCall`.
/// 4. **Llama 3.1 native bare-JSON** — a top-level JSON object the model emits
///    DIRECTLY, with NO open/close marker:
///    `{"name": "calc", "parameters": {"a": 7823, "b": 41, "op": "*"}}`. There
///    is no delimiter at all, so the only reliable anchor is the literal
///    `{"name"` key prefix the native `llama3` template emits (compact JSON, no
///    space after `{`). Like Mistral it has no closing tag — the object ends at
///    EOS — so it also uses the #1982 `closesAtEnd` path; the body parser
///    re-prepends the stripped open token and JSON-decodes the object, keyed on
///    `parameters` (with `arguments` honored as the alias) (#76).
///
/// NOTE: end-to-end verification against a real Mistral model is deferred to
/// #69 / ManifoldKit#1983 — the scenario harness does not render tools yet, so
/// the parser never sees real Mistral output today. The Mistral coverage in
/// `LlamaToolCallParserTests` is therefore synthetic/unit-level (#70).
// @_spi(Testing): published only for backend test targets (companion-package split, #1749).
@_spi(Testing) public enum LlamaToolMarkers {

    /// Gemma 4 native open token.
    static let gemma4OpenTag = "<|tool_call>"
    /// Gemma 4 native close token. A gemma-4 GGUF closes a tool call with the
    /// `<tool_call|>` token (pipe before the bracket) — NOT `<|end_of_turn>`.
    /// The turn-end that follows is an EOG special token the driver consumes
    /// directly, so it never reaches the parser as literal text.
    static let gemma4CloseTag = "<tool_call|>"
    /// Gemma 4 string-quoting token substituted with `"` before parsing.
    static let gemma4QuoteToken = "<|\"|>"

    /// Standard JSON open tag (fallback for non-native tool-call fine-tunes).
    static let jsonOpenTag  = "<tool_call>"
    /// Standard JSON close tag.
    static let jsonCloseTag = "</tool_call>"

    /// Mistral v0.3 open token. The body is a bare JSON array emitted right after
    /// this token with NO closing delimiter — the block ends at EOS.
    static let mistralOpenTag = "[TOOL_CALLS]"

    /// Llama 3.1 native bare-JSON "open token". The `llama3` tool template emits
    /// a top-level JSON object with NO marker at all — the call IS the object:
    /// `{"name": "calc", "parameters": {...}}`. The most specific reliable anchor
    /// is the literal `{"name"` key prefix (compact `tojson` output — no space
    /// after the brace). Anything looser (a bare `{`) would mis-fire on ordinary
    /// JSON the model might emit in prose; `{"name"` is rare outside an actual
    /// tool call, and the body parser still JSON-validates and rejects non-calls.
    /// The token is stripped from the buffered body by the scanner, so the body
    /// parser re-prepends it before decoding (see `parseLlama3BareJSON`).
    static let llama3OpenTag = #"{"name""#

    /// The ordered marker set Llama hands to a `ToolCallTransform`.
    /// Gemma-4 first so it wins ties against the JSON fallback. Mistral last —
    /// its `[TOOL_CALLS]` open token cannot collide with the angle-bracket
    /// dialects, so order is immaterial for correctness; it sits last for
    /// readability.
    public static func markers() -> [ToolCallMarker] {
        [
            ToolCallMarker(open: gemma4OpenTag, close: gemma4CloseTag) { body in
                parseCallBuffer(body)
            },
            ToolCallMarker(open: jsonOpenTag, close: jsonCloseTag) { body in
                parseCallBuffer(body)
            },
            // Mistral `[TOOL_CALLS]` — EOS-keyed close (#1982 `closesAtEnd`) and
            // a multi-call array body (#1982 `parseBodyMulti`). The `close`
            // string is irrelevant when `closesAtEnd` is true; the scanner
            // drains the body to the stream end and `finalize()` parses it.
            ToolCallMarker(open: mistralOpenTag, closesAtEnd: true) { body in
                parseMistralArray(body)
            },
            // Llama 3.1 native bare-JSON — no marker; the `{"name"` key prefix is
            // the open anchor and the object ends at EOS (`closesAtEnd: true`,
            // #1982). Placed LAST: its open token starts with `{`, so it cannot
            // collide with the angle-bracket dialects, and in a Mistral
            // `[TOOL_CALLS][{"name"…}]` stream the `[TOOL_CALLS]` open is earlier
            // in the buffer, so earliest-open-wins selects Mistral first (#76).
            ToolCallMarker(open: llama3OpenTag, closesAtEnd: true) { body in
                parseLlama3BareJSON(body)
            }
        ]
    }

    // MARK: - Body dispatch

    /// Dispatches a buffered call body to the Gemma-4 native, Qwen3.5 XML, or
    /// JSON parser by inspecting its prefix — preserving the original
    /// `LlamaToolCallParser.parseCallBuffer` behaviour, which dispatched on the
    /// body shape rather than on which open tag matched. A `call:`-prefixed body
    /// is Gemma-4 native; a `<function=`-prefixed body is Qwen3.5 XML (#158);
    /// anything else is treated as JSON.
    ///
    /// Dispatching on shape rather than on the matched open tag is what lets
    /// Qwen2.5 (JSON) and Qwen3.5 (XML) share the one `<tool_call>` marker —
    /// their delimiters are byte-identical and only the body differs.
    private static func parseCallBuffer(_ raw: String) -> ToolCall? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if trimmed.hasPrefix("call:") {
            return parseGemma4NativeCall(trimmed)
        }
        if trimmed.hasPrefix(qwen35FunctionOpenPrefix) {
            return parseQwen35XMLCall(trimmed)
        }
        return parseJSONCall(trimmed)
    }

    // MARK: - Qwen3.5 XML body parsing

    /// Opening prefix of the Qwen3.5 inner function tag: `<function=`.
    private static let qwen35FunctionOpenPrefix = "<function="
    /// Closing tag of a Qwen3.5 function block.
    private static let qwen35FunctionCloseTag = "</function>"
    /// Opening prefix of a Qwen3.5 parameter tag: `<parameter=`.
    private static let qwen35ParameterOpenPrefix = "<parameter="
    /// Closing tag of a Qwen3.5 parameter block.
    private static let qwen35ParameterCloseTag = "</parameter>"

    /// Parses the Qwen3.5 native tool-call body: a nested-XML block inside the
    /// shared `<tool_call>` … `</tool_call>` delimiters (#158).
    ///
    /// ```
    /// <function=get_weather>
    /// <parameter=city>
    /// London
    /// </parameter>
    /// </function>
    /// ```
    ///
    /// Transcribed from `tokenizer.chat_template` in the Qwen3.5 GGUFs, whose
    /// instruction block spells this shape out verbatim and requires the
    /// `<function=…>` block to be nested inside the `<tool_call>` tags. The
    /// `qwen35` architecture covers the Qwen3.5 *and* Qwen3.6 releases — both
    /// report `general.architecture == "qwen35"` and ship the same XML block.
    ///
    /// Returns `nil` on a missing/malformed function name so the transform
    /// surfaces `.toolCallParseFailed` rather than dispatching a nameless tool.
    private static func parseQwen35XMLCall(_ raw: String) -> ToolCall? {
        guard let nameRange = raw.range(of: qwen35FunctionOpenPrefix),
              let nameEnd = raw[nameRange.upperBound...].firstIndex(of: ">")
        else { return nil }

        let name = String(raw[nameRange.upperBound..<nameEnd])
        guard isValidQwen35FunctionName(name) else { return nil }

        // Bound the parameter scan at this function's OWN close tag. Without it
        // a stream carrying two `<function=…>` blocks inside one `<tool_call>`
        // pair would merge both parameter sets into a single call under the
        // first function's name (same-named keys last-wins) — a silent WRONG
        // dispatch rather than a drop. Off-spec emission, but cheap to exclude.
        var body = String(raw[raw.index(after: nameEnd)...])
        if let functionClose = body.range(of: qwen35FunctionCloseTag) {
            body = String(body[..<functionClose.lowerBound])
        }
        let arguments = parseQwen35Parameters(body)

        // `JSONSerialization.data` raises an ObjC `NSInvalidArgumentException`
        // (NOT a Swift error) on a non-finite Double, and `try?` cannot catch
        // an ObjC exception — the process would abort (exit 134). The live
        // defence is `coerceQwen35Value`, which refuses to construct a
        // non-finite scalar; the nested-JSON route is already closed upstream
        // because `JSONSerialization.jsonObject` rejects `1e400` rather than
        // decoding it to `inf`. This check is therefore defence-in-depth with
        // no currently-reachable trigger — kept because the failure it guards
        // is an uncatchable process abort, and one cheap validity call is a
        // trivial price for making that unreachable by construction.
        guard JSONSerialization.isValidJSONObject(arguments),
              let data = try? JSONSerialization.data(withJSONObject: arguments, options: [.sortedKeys]),
              let argsString = String(data: data, encoding: .utf8)
        else { return nil }

        let id = "llama-\(name)-\(UUID().uuidString.prefix(8))"
        return ToolCall(id: id, toolName: name, arguments: argsString)
    }

    /// A function name is a bare identifier. Rejecting whitespace and angle
    /// brackets is what stops an UNTERMINATED `<function=now` tag from
    /// swallowing the following `</function>` — the naive "scan to the next
    /// `>`" would otherwise yield the name `"now\n</function"` and dispatch a
    /// bogus tool.
    private static func isValidQwen35FunctionName(_ name: String) -> Bool {
        guard !name.isEmpty else { return false }
        return !name.contains(where: { $0.isWhitespace || $0 == "<" || $0 == ">" || $0 == "/" })
    }

    /// Extracts every `<parameter=key>value</parameter>` pair from a Qwen3.5
    /// function body, in emission order.
    ///
    /// Values are delimited by the literal `</parameter>` tag, NOT by the next
    /// angle bracket — the template documents values that "span multiple lines",
    /// and a value may legitimately contain a `<`. The single newline the
    /// template emits either side of the value is structural and stripped;
    /// interior newlines are content and preserved.
    private static func parseQwen35Parameters(_ body: String) -> [String: Any] {
        var parameters: [String: Any] = [:]
        var cursor = body.startIndex

        while let open = body.range(of: qwen35ParameterOpenPrefix, range: cursor..<body.endIndex) {
            guard let keyEnd = body[open.upperBound...].firstIndex(of: ">") else { break }
            let key = String(body[open.upperBound..<keyEnd]).trimmingCharacters(in: .whitespaces)
            let valueStart = body.index(after: keyEnd)

            // An unterminated final parameter runs to the end of the body; take
            // what is there rather than dropping the whole call.
            let close = body.range(of: qwen35ParameterCloseTag, range: valueStart..<body.endIndex)
            let valueEnd = close?.lowerBound ?? body.endIndex
            cursor = close?.upperBound ?? body.endIndex

            guard !key.isEmpty else { continue }
            parameters[key] = coerceQwen35Value(String(body[valueStart..<valueEnd]))
        }

        return parameters
    }

    /// Strips the structural newlines around a parameter value and coerces bare
    /// scalars, arrays, and objects to their JSON types.
    ///
    /// The XML wire format has no quoting, so the value arrives as raw text and
    /// the type must be recovered from its shape — the same trade-off (and the
    /// same rules) the Gemma-4 bare-literal path already makes. Tool schemas are
    /// typed, so leaving `41` as the string `"41"` fails argument validation
    /// downstream; the cost is that a genuinely string-typed value that looks
    /// numeric is typed as a number.
    ///
    /// Scalar detection runs against a whitespace-trimmed probe so a padded
    /// `\n 41 \n` still types as `41`, but a value that is NOT a scalar is
    /// returned with only its structural newlines removed — interior and
    /// surrounding spaces are content in a free-text parameter and must survive.
    private static func coerceQwen35Value(_ raw: String) -> Any {
        var value = raw
        if value.hasPrefix("\n") { value.removeFirst() }
        if value.hasSuffix("\n") { value.removeLast() }

        let probe = value.trimmingCharacters(in: .whitespacesAndNewlines)

        switch probe {
        case "true":  return true
        case "false": return false
        case "null":  return NSNull()
        default: break
        }
        if let intValue = Int(probe) { return intValue }
        if let doubleValue = Double(probe) {
            // `Double("nan"/"inf"/"1e400")` succeeds and yields a non-finite
            // value. Serialising one raises an UNCATCHABLE ObjC exception (see
            // `parseQwen35XMLCall`), so refuse to produce it — fall through and
            // keep the literal text, which is lossless and safe.
            if doubleValue.isFinite { return doubleValue }
        }
        // The template renders structured arguments as `args_value | tojson`
        // inside `<parameter=…>`, so an array/object parameter arrives as JSON
        // TEXT. Decode it rather than handing the tool a string that merely
        // looks like a list — matching the Gemma-4 path, which re-parses `[`/`{`
        // literals for the same reason.
        if probe.hasPrefix("[") || probe.hasPrefix("{"),
           let data = probe.data(using: .utf8),
           let parsed = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) {
            return parsed
        }
        return value
    }

    // MARK: - Gemma 4 native body parsing

    /// Parses Gemma 4 native format: `call:name{param1:<|"|>val<|"|>,param2:42}`.
    ///
    /// The brace body uses **unquoted JSON-like keys** with values that are
    /// either Gemma 4's `<|"|>...<|"|>` quoted-string token, or a bare
    /// numeric / boolean / null literal. JSON itself requires quoted keys, so
    /// the body cannot be handed to `JSONSerialization` directly — the
    /// dedicated tokenizer below quotes keys and (when needed) values, then
    /// round-trips through `JSONSerialization` for canonicalisation.
    private static func parseGemma4NativeCall(_ raw: String) -> ToolCall? {
        // Caller (`parseCallBuffer`) has already trimmed and verified the
        // `call:` prefix.
        let body = String(raw.dropFirst("call:".count))
        let substituted = body.replacingOccurrences(of: gemma4QuoteToken, with: "\"")

        guard let braceIndex = substituted.firstIndex(of: "{") else { return nil }
        let name = String(substituted[..<braceIndex]).trimmingCharacters(in: .whitespaces)
        guard !name.isEmpty else { return nil }

        let braceBody = String(substituted[braceIndex...])
        let argsString = parseGemma4Arguments(braceBody) ?? "{}"
        let id = "llama-\(name)-\(UUID().uuidString.prefix(8))"
        return ToolCall(id: id, toolName: name, arguments: argsString)
    }

    /// Tokenises Gemma 4's `{key:value,key:value}` brace body into a JSON object.
    ///
    /// Returns the canonical JSON string on success, or `nil` on parse failure
    /// so the caller can fall back to `"{}"` (the same behaviour the JSON
    /// fallback path uses for malformed call bodies).
    private static func parseGemma4Arguments(_ braceBody: String) -> String? {
        var trimmed = braceBody.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("{"), trimmed.hasSuffix("}") else { return nil }
        trimmed.removeFirst()
        trimmed.removeLast()
        let inner = trimmed.trimmingCharacters(in: .whitespacesAndNewlines)
        if inner.isEmpty {
            return "{}"
        }

        var dict: [String: Any] = [:]
        var idx = inner.startIndex
        let end = inner.endIndex

        while idx < end {
            // Skip whitespace and stray commas between pairs.
            while idx < end, inner[idx].isWhitespace || inner[idx] == "," {
                idx = inner.index(after: idx)
            }
            if idx >= end { break }

            // Read unquoted key up to the next `:`.
            guard let colon = inner[idx...].firstIndex(of: ":") else { return nil }
            let key = inner[idx..<colon].trimmingCharacters(in: .whitespaces)
            guard !key.isEmpty else { return nil }
            idx = inner.index(after: colon)

            // Skip whitespace before the value.
            while idx < end, inner[idx].isWhitespace {
                idx = inner.index(after: idx)
            }
            if idx >= end { return nil }

            // Read value: either a quoted string, or a bare literal up to the next comma.
            let value: Any
            if inner[idx] == "\"" {
                // Quoted string: scan for the closing quote, honouring `\"` escapes.
                var cursor = inner.index(after: idx)
                var raw = ""
                var escaped = false
                while cursor < end {
                    let ch = inner[cursor]
                    if escaped {
                        raw.append(ch)
                        escaped = false
                    } else if ch == "\\" {
                        raw.append(ch)
                        escaped = true
                    } else if ch == "\"" {
                        break
                    } else {
                        raw.append(ch)
                    }
                    cursor = inner.index(after: cursor)
                }
                guard cursor < end else { return nil } // unterminated string
                idx = inner.index(after: cursor)
                // Decode JSON escapes via JSONSerialization on a wrapped string.
                if let data = "\"\(raw)\"".data(using: .utf8),
                   let decoded = try? JSONSerialization.jsonObject(
                       with: data, options: [.fragmentsAllowed]) as? String {
                    value = decoded
                } else {
                    value = raw
                }
            } else if inner[idx] == "[" || inner[idx] == "{" {
                // Array or nested object — scan to the matching close bracket,
                // then parse the captured literal as JSON.
                var cursor = idx
                var depth = 0
                while cursor < end {
                    let ch = inner[cursor]
                    if ch == "[" || ch == "{" { depth += 1 }
                    else if ch == "]" || ch == "}" { depth -= 1; if depth == 0 { cursor = inner.index(after: cursor); break } }
                    cursor = inner.index(after: cursor)
                }
                let literal = inner[idx..<cursor].trimmingCharacters(in: .whitespaces)
                idx = cursor
                if let data = literal.data(using: .utf8),
                   let parsed = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) {
                    value = parsed
                } else {
                    value = literal
                }
            } else {
                // Bare literal — number, true, false, or null. Read up to next comma.
                var cursor = idx
                while cursor < end, inner[cursor] != "," {
                    cursor = inner.index(after: cursor)
                }
                let literal = inner[idx..<cursor].trimmingCharacters(in: .whitespaces)
                idx = cursor
                guard !literal.isEmpty else { return nil }
                if literal == "true" {
                    value = true
                } else if literal == "false" {
                    value = false
                } else if literal == "null" {
                    value = NSNull()
                } else if let intVal = Int(literal) {
                    value = intVal
                } else if let dblVal = Double(literal) {
                    value = dblVal
                } else {
                    // Treat as a bare string when the model omits Gemma's quote token.
                    value = literal
                }
            }

            dict[key] = value
        }

        guard let data = try? JSONSerialization.data(withJSONObject: dict, options: [.sortedKeys]),
              let str  = String(data: data, encoding: .utf8)
        else { return nil }
        return str
    }

    // MARK: - JSON fallback body parsing

    /// Parses JSON fallback format: `{"name":"...","arguments":{...}}`.
    /// `arguments` and the llama3.1 `parameters` alias are both honored via
    /// `coerceArguments` (#76); the llama3.1 bare-JSON dialect reuses this decoder.
    private static func parseJSONCall(_ raw: String) -> ToolCall? {
        let json = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !json.isEmpty,
              let data = json.data(using: .utf8),
              let obj  = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let name = obj["name"] as? String, !name.isEmpty
        else { return nil }

        let id = "llama-\(name)-\(UUID().uuidString.prefix(8))"
        return ToolCall(id: id, toolName: name, arguments: coerceArguments(from: obj))
    }

    /// Coerces a call object's arguments field into a serialized JSON string.
    ///
    /// Accepts both `arguments` (the JSON / Mistral / Qwen key) and `parameters`
    /// (the llama3.1 native key) — `arguments` wins when BOTH are present so the
    /// historical key keeps priority and no existing call changes shape (#76).
    /// The value may be a nested object (serialized canonically) or an already
    /// serialized string (preserved verbatim); anything else falls back to `{}`.
    private static func coerceArguments(from obj: [String: Any]) -> String {
        let value = obj["arguments"] ?? obj["parameters"]
        if let argsDict = value as? [String: Any],
           let serialized = try? JSONSerialization.data(withJSONObject: argsDict),
           let str = String(data: serialized, encoding: .utf8) {
            return str
        } else if let rawStr = value as? String {
            return rawStr
        }
        return "{}"
    }

    // MARK: - Llama 3.1 native bare-JSON body parsing

    /// Parses the llama3.1 native dialect: a bare top-level JSON object,
    /// `{"name": "calc", "parameters": {…}}`, with NO open/close marker.
    ///
    /// The scanner matches `{"name"` as the open token and STRIPS it from the
    /// buffered body, so this re-prepends it before decoding. Returns a one-element
    /// array on success (the marker uses `parseBodyMulti` for the `closesAtEnd`
    /// init); a non-object, name-less, or otherwise malformed body yields `[]`,
    /// which the transform surfaces as `.toolCallParseFailed` — crucially, plain
    /// prose that merely contained the literal `{"name"` decodes to nothing and is
    /// dropped rather than mis-dispatched.
    private static func parseLlama3BareJSON(_ body: String) -> [ToolCall] {
        // The open token `{"name"` was consumed by the scanner; rebuild the object.
        let reconstructed = llama3OpenTag + body
        guard let call = parseJSONCall(reconstructed) else { return [] }
        return [call]
    }

    // MARK: - Mistral `[TOOL_CALLS]` body parsing

    /// Parses Mistral's `[TOOL_CALLS]` body: a bare JSON **array** of call
    /// objects, e.g. `[{"name":"calc","arguments":{"a":1}}, {"name":"now"}]`.
    ///
    /// Returns ALL calls in the array (the #1982 multi-call contract). Each
    /// element is the same `{name|fn, arguments}` shape the JSON-fallback dialect
    /// uses; `parseJSONElement` is tolerant of the `fn` alias and of `arguments`
    /// being either a nested object or a pre-serialized string. Malformed
    /// elements are skipped rather than aborting the whole array — a single bad
    /// entry should not drop its well-formed siblings. An entirely unparseable
    /// body yields `[]`, which the transform surfaces as `.toolCallParseFailed`.
    private static func parseMistralArray(_ raw: String) -> [ToolCall] {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              let data = trimmed.data(using: .utf8),
              let parsed = try? JSONSerialization.jsonObject(with: data)
        else { return [] }

        // Tolerant of both the array form (the spec'd Mistral shape) and a lone
        // object (some fine-tunes emit a single call without the array wrapper).
        let elements: [[String: Any]]
        if let array = parsed as? [[String: Any]] {
            elements = array
        } else if let object = parsed as? [String: Any] {
            elements = [object]
        } else {
            return []
        }

        return elements.compactMap(parseJSONElement)
    }

    /// Maps one Mistral array element (`{name|fn, arguments}`) to a `ToolCall`.
    /// Shares the JSON-fallback argument-coercion rules; accepts `fn` as an alias
    /// for `name`.
    private static func parseJSONElement(_ obj: [String: Any]) -> ToolCall? {
        let name = (obj["name"] as? String) ?? (obj["fn"] as? String)
        guard let name, !name.isEmpty else { return nil }

        let id = "llama-\(name)-\(UUID().uuidString.prefix(8))"
        return ToolCall(id: id, toolName: name, arguments: coerceArguments(from: obj))
    }
}
