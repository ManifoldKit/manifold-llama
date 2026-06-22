// GrammarToolProbe — before/after measurement of grammar-constrained tool-call
// dispatch for Llama-3.1 (#2005 follow-on spike).
//
// Why this path (not ScenarioRunner): `ScenarioRunner` exposes no grammar hook —
// it builds its own `GenerationConfig` internally with no way to inject a GBNF
// string. The spike brief explicitly sanctions driving the model directly via
// `InferenceService.enqueue(..., config:)` (which carries `GenerationConfig.grammar`)
// when the runner has no hook. This probe replicates a SINGLE tool turn for each
// of the 01-now / 02-calc / 03-read prompts and measures the first-turn outcome:
//
//   parseable  — the stream yielded a `.toolCall(ToolCall)` (a host can dispatch it)
//   parseFail  — the stream yielded `.toolCallParseFailed` (call-shaped but unparseable)
//   narrated   — neither: the model emitted prose (the Llama-3.1 failure mode)
//
// It runs each prompt twice: grammar OFF (baseline) then grammar ON (the GBNF
// emitted by `ToolCallGrammar` keyed to the model's `ChatTemplateToolDescriptor`
// dialect). Tool dispatch itself is capped at one iteration — we measure the
// FIRST assistant turn's call emission (dispatch/parse), NOT second-turn grounding.
import Foundation
import ManifoldInference
import ManifoldModelCatalog
import ManifoldContract
import ManifoldTools
import ManifoldLlama

/// One probe prompt mirroring a scenario's system+user turn.
private struct ProbePrompt {
    let id: String
    let tool: String
    let system: String
    let user: String
}

enum GrammarToolProbe {

    /// The three single-tool prompts (verbatim from 01-now / 02-calc / 03-read).
    private static let prompts: [ProbePrompt] = [
        ProbePrompt(
            id: "01-now", tool: "now",
            system: "You have tools. When a tool can answer, call it. Never guess time, dates, or numeric calculations — always call the relevant tool and quote its result verbatim.",
            user: "What time is it? Reply with exactly the timestamp the tool returns."),
        ProbePrompt(
            id: "02-calc", tool: "calc",
            system: "You have tools. When a tool can answer, call it. Never perform arithmetic in your head — always call `calc` and quote its answer.",
            user: "What is 7823 multiplied by 41? Reply with only the number the tool returns."),
        ProbePrompt(
            id: "03-read", tool: "read_file",
            system: "You have tools. When the user asks about a file, always call `read_file` and quote the file's contents verbatim in your reply. Never guess file contents.",
            user: "Read the file example.txt in the fixtures sandbox and reply with exactly its contents."),
    ]

    private enum Outcome: String {
        case parseable
        case parseFail
        case narrated
    }

    /// Runs the before/after probe. The model is already loaded into `service`.
    @MainActor
    static func run(service: InferenceService, registry: ToolRegistry, modelInfo: ModelInfo) async -> Int32 {
        // Derive the dialect from the model's embedded chat template (Layer 1).
        let descriptor = ChatTemplateToolDescriptor(parsingChatTemplate: modelInfo.chatTemplateRaw)
        guard let dialect = descriptor.declaredDialect else {
            print("grammar-tools: model declares no recognised tool dialect — nothing to constrain.")
            return 1
        }
        print("=== grammar-tools probe ===")
        print("Dialect: open=\(dialect.openDelimiter ?? "(none)") close=\(dialect.closeDelimiter ?? "(none)") args=\(dialect.argEncoding.rawValue) extractability=\(descriptor.extractability.rawValue)")

        let allDefs = registry.definitions
        func defs(for tool: String) -> [ToolDefinition] { allDefs.filter { $0.name == tool } }

        var offParseable = 0, onParseable = 0
        let total = prompts.count

        for grammarOn in [false, true] {
            print("\n--- grammar \(grammarOn ? "ON" : "OFF") ---")
            for p in prompts {
                let toolDefs = defs(for: p.tool)
                guard let def = toolDefs.first else {
                    print("  \(p.id): SKIP — tool '\(p.tool)' not registered")
                    continue
                }

                var grammar: String? = nil
                if grammarOn {
                    do {
                        // Schema-bearing generator call so the emitted GBNF is keyed
                        // to the advertised tool's real name (and, when we extend it,
                        // its params). For the single-tool prompts the name enum is
                        // exactly that one tool.
                        grammar = try ToolCallGrammar.grammar(
                            for: dialect,
                            tools: [ToolCallGrammar.Tool(name: def.name, parametersSchema: def.parameters)])
                    } catch {
                        print("  \(p.id): grammar generation failed: \(error)")
                        continue
                    }
                }

                let outcome = await probeOne(
                    service: service, prompt: p, toolDefs: toolDefs, grammar: grammar)
                print("  \(p.id) [\(p.tool)] → \(outcome.rawValue)")
                if outcome == .parseable {
                    if grammarOn { onParseable += 1 } else { offParseable += 1 }
                }
            }
        }

        func pct(_ n: Int) -> String { String(format: "%.0f%%", Double(n) / Double(total) * 100) }
        print("\n=== RESULT (parseable tool-call rate, first turn) ===")
        print("  grammar OFF: \(offParseable)/\(total) (\(pct(offParseable)))")
        print("  grammar ON : \(onParseable)/\(total) (\(pct(onParseable)))")
        print("  NOTE: this measures dispatch/parse only — NOT second-turn grounding.")
        return 0
    }

    /// Drives one prompt for one (grammar) condition and classifies the first-turn
    /// tool-call outcome.
    @MainActor
    private static func probeOne(
        service: InferenceService,
        prompt: ProbePrompt,
        toolDefs: [ToolDefinition],
        grammar: String?
    ) async -> Outcome {
        // Deterministic single-turn config: greedy (temp 0, topK 1), fixed seed,
        // one tool iteration so we observe the FIRST assistant turn's emission.
        let config = GenerationConfig(
            temperature: 0.0,
            topK: 1,
            seed: 42,
            maxOutputTokens: 256,
            tools: toolDefs,
            toolChoice: .auto,
            maxToolIterations: 1,
            grammar: grammar)

        do {
            let (_, stream) = try service.enqueue(
                messages: [.user(prompt.user)],
                systemPrompt: prompt.system,
                config: config)

            var sawParseable = false
            var sawParseFail = false
            for try await event in stream.events {
                switch event {
                case .toolCall:
                    sawParseable = true
                case .toolCallParseFailed, .toolCallTruncated:
                    sawParseFail = true
                default:
                    break
                }
                // First-turn decision: stop once a tool call resolves either way.
                if sawParseable { break }
            }
            if sawParseable { return .parseable }
            if sawParseFail { return .parseFail }
            return .narrated
        } catch {
            print("    (enqueue/stream error for \(prompt.id): \(error))")
            return .narrated
        }
    }
}
