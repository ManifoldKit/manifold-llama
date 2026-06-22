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

    /// Probe difficulty. `.easy` advertises only the required tool and decodes
    /// greedily (temp 0, topK 1) — the saturated, deterministic baseline. `.hard`
    /// advertises ALL registered tools and decodes at temp 0.7 (the default,
    /// non-greedy regime where Llama-3.1 narrates instead of emitting a call) —
    /// this is the regime that reproduces the soak's dispatch failure and where a
    /// grammar can bite.
    enum Difficulty { case easy, hard }

    /// Runs the before/after probe at both difficulties. Model already loaded.
    @MainActor
    static func run(service: InferenceService, registry: ToolRegistry, modelInfo: ModelInfo) async -> Int32 {
        let descriptor = ChatTemplateToolDescriptor(parsingChatTemplate: modelInfo.chatTemplateRaw)
        guard let dialect = descriptor.declaredDialect else {
            print("grammar-tools: model declares no recognised tool dialect — nothing to constrain.")
            return 1
        }
        print("=== grammar-tools probe ===")
        print("Dialect: open=\(dialect.openDelimiter ?? "(none)") close=\(dialect.closeDelimiter ?? "(none)") args=\(dialect.argEncoding.rawValue) extractability=\(descriptor.extractability.rawValue)")
        _ = await runOne(service: service, registry: registry, dialect: dialect, difficulty: .easy)
        _ = await runOne(service: service, registry: registry, dialect: dialect, difficulty: .hard)
        return 0
    }

    @MainActor
    private static func runOne(service: InferenceService, registry: ToolRegistry, dialect: ChatTemplateToolDescriptor.ToolCallDialect, difficulty: Difficulty) async -> Int32 {
        let allDefs = registry.definitions
        func defs(for tool: String) -> [ToolDefinition] { allDefs.filter { $0.name == tool } }

        // `.hard` advertises every registered tool to each prompt (selection
        // pressure + a multi-tool name-enum in the grammar) and decodes at the
        // default temperature; `.easy` advertises only the required tool greedily.
        let label = difficulty == .easy
            ? "EASY (only required tool advertised, greedy temp=0)"
            : "HARD (all \(allDefs.count) tools advertised, temp=0.7)"
        print("\n############ \(label) ############")

        var offParseable = 0, onParseable = 0
        let total = prompts.count

        for grammarOn in [false, true] {
            print("\n--- grammar \(grammarOn ? "ON" : "OFF") ---")
            for p in prompts {
                // Tools advertised to the model this prompt.
                let advertised = difficulty == .easy ? defs(for: p.tool) : allDefs
                guard advertised.contains(where: { $0.name == p.tool }) else {
                    print("  \(p.id): SKIP — tool '\(p.tool)' not registered")
                    continue
                }

                var grammar: String? = nil
                if grammarOn {
                    do {
                        // The grammar's name-enum lists exactly the advertised tools
                        // (one for .easy, all for .hard), keyed to their real names
                        // and schemas — so a grammar-valid call must name a real tool.
                        grammar = try ToolCallGrammar.grammar(
                            for: dialect,
                            tools: advertised.map {
                                ToolCallGrammar.Tool(name: $0.name, parametersSchema: $0.parameters)
                            })
                    } catch {
                        print("  \(p.id): grammar generation failed: \(error)")
                        continue
                    }
                }

                let outcome = await probeOne(
                    service: service, prompt: p, toolDefs: advertised,
                    grammar: grammar, difficulty: difficulty)
                print("  \(p.id) [want \(p.tool)] → \(outcome.rawValue)")
                if outcome == .parseable {
                    if grammarOn { onParseable += 1 } else { offParseable += 1 }
                }
            }
        }

        func pct(_ n: Int) -> String { String(format: "%.0f%%", Double(n) / Double(total) * 100) }
        print("\n=== RESULT [\(difficulty == .easy ? "easy" : "hard")] parseable-tool-call rate (first turn) ===")
        print("  grammar OFF: \(offParseable)/\(total) (\(pct(offParseable)))")
        print("  grammar ON : \(onParseable)/\(total) (\(pct(onParseable)))")
        print("  NOTE: dispatch/parse only — NOT second-turn grounding.")
        return 0
    }

    /// Drives one prompt for one (grammar) condition and classifies the first-turn
    /// tool-call outcome.
    @MainActor
    private static func probeOne(
        service: InferenceService,
        prompt: ProbePrompt,
        toolDefs: [ToolDefinition],
        grammar: String?,
        difficulty: Difficulty
    ) async -> Outcome {
        // .easy: greedy (temp 0, topK 1) — deterministic saturated baseline.
        // .hard: default temp 0.7 (the regime Llama-3.1 narrates in), fixed seed
        // for reproducibility. One tool iteration: we observe the FIRST turn only.
        let config = GenerationConfig(
            temperature: difficulty == .easy ? 0.0 : 0.7,
            topK: difficulty == .easy ? 1 : nil,
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
