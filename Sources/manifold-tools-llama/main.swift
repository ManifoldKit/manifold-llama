// manifold-tools-llama — run ManifoldKit's tool-calling validation scenarios
// against a real llama.cpp / GGUF model.
//
// This reuses the published `ManifoldTools` library product from ManifoldKit
// (bundled scenarios + fixture tree + reference tools + scenario runner +
// `ScenarioCLIHarness`, MK 0.64+) and drives them through this package's
// `LlamaBackend`. There are NO changes to ManifoldKit core — the only
// Llama-specific wiring is the backend construction, model load, decoy-tool
// padding, result-grounding system-prompt injection, and grammar-constrained
// final-answer decoding (none of which `ScenarioCLIHarness` owns — it
// deliberately leaves model-loading policy and registry scoping to each
// consumer; see its doc comment). Four scenarios still carry a vendored
// override — see `Sources/manifold-tools-llama/ScenarioOverrides/` and
// `loadScenarios()` below.
//
// Real-hardware tool: requires Apple Silicon + Metal (llama.cpp uses a
// process-global backend init and has no Metal support in the simulator) and
// a local `.gguf` model. Compilation does not need a model; running does.
import Foundation
import ManifoldInference
import ManifoldModelCatalog
import ManifoldTools
import ManifoldLlama

/// Hand-rolled argument parser for the flags `ScenarioCLIHarness.parseCommonFlags`
/// doesn't own (`--model`, `--describe`, `--bench` and its sub-flags). Common
/// flags (`--scenario`, `--output`, `--fixtures-root`, `--extra-tools`,
/// `--list`, `--help`/`-h`) are parsed by the shared harness in `CLI.parse`
/// below; this struct only carries this CLI's own remainder.
struct CLI {

    /// Flags shared with the companion CLIs (`manifold-tools`,
    /// `manifold-tools-mlx`) — parsed by `ScenarioCLIHarness`.
    var common: ScenarioCLIHarness.Options
    var modelPath: String? = nil
    /// Static tool-call capability report (issue #2005 layers 1+2): parses the
    /// model's embedded chat template into a `ChatTemplateToolDescriptor` and
    /// runs `RenderConsistencyChecker`. Reads GGUF metadata only — NO weights,
    /// no Metal, no generation. Distinct from the scenario soak (the measured
    /// positive verdict) and from `--bench` (timing).
    var describe: Bool = false

    /// Cold-vs-warm generation benchmark mode (see Benchmark.swift). Bypasses the
    /// scenario runner entirely — drives `LlamaBackend` directly to time the
    /// one-time per-process Metal pipeline warm-up the first generation pays.
    var bench: Bool = false
    /// Flash-attention setting(s) for `--bench`: on | off | both.
    var flash: String = "on"
    /// Prompt the benchmark generates from. Fixed so every run does identical work.
    var benchPrompt: String = "Write a short paragraph about the ocean."
    /// Tokens to generate per benchmark run.
    var maxTokens: Int = 64
    /// Number of warm (post-cold) generations to average for `--bench`.
    var warmRuns: Int = 3
    /// Requested context size (tokens) for `--bench`. The load planner may clamp
    /// it; the benchmark prints the plan-effective value.
    var context: Int = 4096

    /// Argument errors exit with status 2 via `exit(2)` + stderr rather than
    /// `precondition` / `fatalError` (those trap with SIGABRT in debug builds,
    /// producing a confusing stack trace instead of the clean "bad arguments"
    /// exit code the usage text documents).
    private static func fail(_ message: String) -> Never {
        FileHandle.standardError.write(Data("manifold-tools-llama: \(message)\n".utf8))
        exit(2)
    }

    /// Parses the flags common to every scenario-CLI harness consumer via
    /// `ScenarioCLIHarness`, then walks the remainder for this CLI's own
    /// flags (`--model`, `--describe`, `--bench`, `--flash`, `--bench-prompt`,
    /// `--max-tokens`, `--warm-runs`, `--context`).
    static func parse(_ argv: [String]) -> CLI {
        let commonOptions: ScenarioCLIHarness.Options
        let remainder: [String]
        switch ScenarioCLIHarness.parseCommonFlags(argv, defaultOutput: defaultOutputURL()) {
        case .options(let options, let rest):
            commonOptions = options
            remainder = rest
        case .helpRequested:
            printUsage()
            exit(0)
        case .failure(let message):
            fail(message)
        }

        var cli = CLI(common: commonOptions)
        var i = 0
        while i < remainder.count {
            let arg = remainder[i]
            switch arg {
            case "--model":
                i += 1
                guard i < remainder.count else { fail("--model requires a value") }
                cli.modelPath = remainder[i]
            case "--bench":
                cli.bench = true
            case "--flash":
                i += 1
                guard i < remainder.count else { fail("--flash requires a value (on|off|both)") }
                cli.flash = remainder[i]
            case "--bench-prompt":
                i += 1
                guard i < remainder.count else { fail("--bench-prompt requires a value") }
                cli.benchPrompt = remainder[i]
            case "--max-tokens":
                i += 1
                guard i < remainder.count else { fail("--max-tokens requires a value") }
                guard let n = Int(remainder[i]), n > 0 else { fail("--max-tokens requires a positive integer") }
                cli.maxTokens = n
            case "--warm-runs":
                i += 1
                guard i < remainder.count else { fail("--warm-runs requires a value") }
                guard let n = Int(remainder[i]), n >= 0 else { fail("--warm-runs requires a non-negative integer") }
                cli.warmRuns = n
            case "--context":
                i += 1
                guard i < remainder.count else { fail("--context requires a value") }
                guard let n = Int(remainder[i]), n > 0 else { fail("--context requires a positive integer") }
                cli.context = n
            case "--describe":
                cli.describe = true
            default:
                fail("unknown argument: \(arg)")
            }
            i += 1
        }
        return cli
    }

    static func defaultOutputURL() -> URL {
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        return cwd.appendingPathComponent("tmp/manifold-tools-llama/\(TranscriptLogger.defaultFilename())")
    }

    static func printUsage() {
        let text = """
        manifold-tools-llama — tool-calling validation against a real GGUF model

        USAGE
          manifold-tools-llama --model <path.gguf> [--scenario <id|all>]
                    [--output <path.jsonl>] [--fixtures-root <dir>] [--list]
          manifold-tools-llama --bench --model <path.gguf> [--flash on|off|both]
                    [--bench-prompt <text>] [--max-tokens <n>] [--warm-runs <n>]

        FLAGS
          --model <path>        Path to the .gguf model file. REQUIRED (except for --list / --help).
          --scenario <id>       Scenario id (matches JSON 'id') or 'all'. Default: all.
          --output <path>       Transcript JSONL destination. Default: tmp/manifold-tools-llama/<iso>.jsonl.
          --fixtures-root <dir> Override the file/dir tool fixture root. Default: bundled fixtures.

        BENCHMARK (--bench)
          --bench               Run the cold-vs-warm generation benchmark instead of
                                the scenario harness. Times model load, the first
                                (cold) generation, and N warm generations; the
                                cold−warm delta ≈ the one-time per-process Metal
                                pipeline warm-up (incl. flash-attention kernels).
          --flash on|off|both   FA setting. Default: on. For an honest cold
                                comparison run on/off in SEPARATE processes — Metal
                                caches pipelines per-process, so 'both' under-reports
                                the second config's cold time (a warning is printed).
          --bench-prompt <text> Prompt to generate from. Default: a fixed sentence.
          --max-tokens <n>      Tokens generated per run. Default: 64.
          --warm-runs <n>       Warm generations to average. Default: 3.
          --context <n>         Requested context size in tokens. Default: 4096.
                                The load planner may clamp it; the run prints the
                                plan-effective value.
          --extra-tools <N>     Advertise N decoy (distractor) tools alongside each
                                scenario's required tool(s). Decoys are plausible but
                                never the correct answer; success still requires the
                                REAL tool to be dispatched. Default: 0. Max useful: 24.
          --describe            Print the STATIC tool-call capability report for
                                --model (issue #2005 layers 1+2): the
                                ChatTemplateToolDescriptor (toolsExpressible,
                                declared dialect, extractability) and the
                                RenderConsistencyChecker verdict. Reads GGUF
                                metadata only — no weights, no Metal, no
                                generation. Runs anywhere, including CI.
          --list                Print available scenarios and exit (no model needed).
          --help                Show this text.

        EXIT
          0 — all scenarios passed.
          1 — at least one scenario or assertion failed (or a non-load setup error).
          2 — bad arguments.
          3 — the model FAILED TO LOAD (arch / llama.cpp version skew). Distinct
              from 1 so a sweep can tell "never loaded" from "loaded, no dispatch".

        REQUIREMENTS
          Apple Silicon + Metal (llama.cpp has no simulator Metal support) and a
          local .gguf model (e.g. a gemma GGUF). The transcript is one JSONL line
          per event so downstream tooling can diff runs without parsing stdout.
        """
        print(text)
    }
}

/// Holds the `LlamaBackend` the registered factory constructs so the harness can
/// `unloadAndWait()` it deterministically before `exit()` — the coordinator's own
/// `unloadModel()` is fire-and-forget and would race the process exit.
@MainActor
final class BackendBox {
    var backend: LlamaBackend?
}

/// Builds a `ToolRegistry` containing every reference tool.
///
/// `ScenarioRunner` filters the registry to each scenario's `requiredTools`
/// before advertising them to the model (so a scenario is still only ever shown
/// the tools it needs — the per-scenario scoping of #66 is preserved by the
/// runner, not by rebuilding a registry per scenario). A single all-tools
/// registry therefore lets the harness load the model once and reuse one
/// service across every scenario.
///
/// The file/dir tools read against the resolved (bundled or overridden) fixture
/// root rather than ManifoldTools' default, which points at a non-existent
/// ManifoldKit test path here.
@MainActor
func makeFullRegistry(fixturesRoot: URL) -> ToolRegistry {
    let registry = ToolRegistry()
    registry.register(NowTool.makeExecutor())
    registry.register(CalcTool.makeExecutor())
    registry.register(ReadFileTool.makeExecutor(root: fixturesRoot))
    registry.register(ListDirTool.makeExecutor(root: fixturesRoot))
    registry.register(SampleRepoSearchTool.makeExecutor(root: fixturesRoot))
    registry.register(HttpGetFixtureTool.makeExecutor())
    return registry
}

// MARK: - Decoy tools (--extra-tools)

/// A fixed pool of plausible, distinct decoy `ToolDefinition`s used to pad the
/// advertised tool set when `--extra-tools N` is passed. None of these is ever
/// the correct answer for the single-tool scenarios (`now` / `calc` / file
/// reads), so a model that dispatches one has been distracted — the scenario's
/// own assertions (which require the REAL tool) still gate pass/fail.
///
/// Names and parameter schemas are deliberately realistic and varied so the
/// model faces genuine selection pressure rather than obvious throwaways. There
/// are 24 entries so `--extra-tools` can pad well past 20.
enum DecoyTools {

    /// Helper to build a one-string-parameter object schema.
    private static func obj(_ props: [(String, String)], required: [String]) -> JSONSchemaValue {
        var properties: [String: JSONSchemaValue] = [:]
        for (name, desc) in props {
            properties[name] = .object([
                "type": .string("string"),
                "description": .string(desc)
            ])
        }
        return .object([
            "type": .string("object"),
            "properties": .object(properties),
            "required": .array(required.map(JSONSchemaValue.string))
        ])
    }

    /// The ordered decoy pool. `--extra-tools N` advertises the first N entries.
    static let pool: [ToolDefinition] = [
        ToolDefinition(name: "get_weather", description: "Returns the current weather for a city.",
                       parameters: obj([("city", "City name")], required: ["city"])),
        ToolDefinition(name: "send_email", description: "Sends an email to a recipient.",
                       parameters: obj([("to", "Recipient address"), ("subject", "Subject line"), ("body", "Email body")], required: ["to", "body"])),
        ToolDefinition(name: "search_web", description: "Searches the web and returns result snippets.",
                       parameters: obj([("query", "Search query")], required: ["query"])),
        ToolDefinition(name: "translate_text", description: "Translates text into a target language.",
                       parameters: obj([("text", "Text to translate"), ("target_language", "Target language code")], required: ["text", "target_language"])),
        ToolDefinition(name: "set_timer", description: "Starts a countdown timer for the given duration.",
                       parameters: obj([("duration", "Duration, e.g. '10 minutes'")], required: ["duration"])),
        ToolDefinition(name: "currency_convert", description: "Converts an amount between two currencies.",
                       parameters: obj([("amount", "Amount to convert"), ("from", "Source currency code"), ("to", "Target currency code")], required: ["amount", "from", "to"])),
        ToolDefinition(name: "create_event", description: "Creates a calendar event.",
                       parameters: obj([("title", "Event title"), ("start", "Start time"), ("end", "End time")], required: ["title", "start"])),
        ToolDefinition(name: "get_stock_price", description: "Returns the latest price for a stock ticker.",
                       parameters: obj([("ticker", "Stock ticker symbol")], required: ["ticker"])),
        ToolDefinition(name: "roll_dice", description: "Rolls dice and returns the total.",
                       parameters: obj([("notation", "Dice notation, e.g. '2d6'")], required: ["notation"])),
        ToolDefinition(name: "unit_convert", description: "Converts a value between measurement units.",
                       parameters: obj([("value", "Numeric value"), ("from_unit", "Source unit"), ("to_unit", "Target unit")], required: ["value", "from_unit", "to_unit"])),
        ToolDefinition(name: "send_sms", description: "Sends a text message to a phone number.",
                       parameters: obj([("phone", "Destination phone number"), ("message", "Message text")], required: ["phone", "message"])),
        ToolDefinition(name: "get_directions", description: "Returns driving directions between two places.",
                       parameters: obj([("origin", "Starting location"), ("destination", "Ending location")], required: ["origin", "destination"])),
        ToolDefinition(name: "play_music", description: "Plays a song or playlist.",
                       parameters: obj([("query", "Song, artist, or playlist name")], required: ["query"])),
        ToolDefinition(name: "set_reminder", description: "Creates a reminder at a given time.",
                       parameters: obj([("text", "Reminder text"), ("time", "When to remind")], required: ["text", "time"])),
        ToolDefinition(name: "get_news", description: "Returns recent news headlines for a topic.",
                       parameters: obj([("topic", "News topic or category")], required: ["topic"])),
        ToolDefinition(name: "book_flight", description: "Searches and books a flight.",
                       parameters: obj([("origin", "Departure airport"), ("destination", "Arrival airport"), ("date", "Travel date")], required: ["origin", "destination", "date"])),
        ToolDefinition(name: "get_definition", description: "Returns the dictionary definition of a word.",
                       parameters: obj([("word", "Word to define")], required: ["word"])),
        ToolDefinition(name: "create_note", description: "Saves a note to the user's notebook.",
                       parameters: obj([("title", "Note title"), ("content", "Note body")], required: ["content"])),
        ToolDefinition(name: "get_traffic", description: "Returns current traffic conditions for a route.",
                       parameters: obj([("route", "Route or area name")], required: ["route"])),
        ToolDefinition(name: "shorten_url", description: "Creates a shortened URL.",
                       parameters: obj([("url", "URL to shorten")], required: ["url"])),
        ToolDefinition(name: "get_recipe", description: "Returns a recipe for a dish.",
                       parameters: obj([("dish", "Dish name")], required: ["dish"])),
        ToolDefinition(name: "track_package", description: "Returns the delivery status of a package.",
                       parameters: obj([("tracking_number", "Carrier tracking number")], required: ["tracking_number"])),
        ToolDefinition(name: "get_horoscope", description: "Returns the daily horoscope for a star sign.",
                       parameters: obj([("sign", "Zodiac sign")], required: ["sign"])),
        ToolDefinition(name: "convert_timezone", description: "Converts a time between two time zones.",
                       parameters: obj([("time", "Time to convert"), ("from_zone", "Source time zone"), ("to_zone", "Target time zone")], required: ["time", "from_zone", "to_zone"])),
    ]

    /// Result shape every decoy executor returns. Decoys should never actually be
    /// dispatched on a passing run; if a model does call one, this benign payload
    /// keeps the runner loop alive so the transcript records the wrong-tool call.
    struct DecoyResult: Encodable, Sendable {
        let note: String
    }

    /// Builds a no-op executor for a decoy definition. Accepts any arguments
    /// (`EmptyArgs` is permissive) and returns a fixed marker so a wrong-tool
    /// dispatch is visible in the transcript without crashing the run.
    static func makeExecutor(for definition: ToolDefinition) -> TypedToolExecutor<EmptyArgs, DecoyResult> {
        TypedToolExecutor(definition: definition) { _ in
            DecoyResult(note: "decoy tool '\(definition.name)' is not the right tool for this task")
        }
    }
}

/// Registers the first `count` decoy tools (clamped to the pool size) into the
/// registry and returns their names so the caller can add them to the scenario's
/// advertised set. Returns an empty array when `count <= 0`.
@MainActor
func registerDecoys(count: Int, into registry: ToolRegistry) -> [String] {
    guard count > 0 else { return [] }
    let n = min(count, DecoyTools.pool.count)
    let chosen = Array(DecoyTools.pool.prefix(n))
    for definition in chosen {
        registry.register(DecoyTools.makeExecutor(for: definition))
    }
    return chosen.map(\.name)
}

/// Returns a copy of `scenario` whose `requiredTools` also lists `decoyNames`.
///
/// `ScenarioRunner` advertises exactly the tools named in `requiredTools` (it
/// filters the registry by that set), so padding the list is how decoys reach
/// the model. The scenario's assertions are unchanged — success still requires
/// the original required tool to be dispatched with correct args. `Scenario` has
/// no public memberwise initialiser, so we round-trip through its `Codable`
/// conformance and splice the names in via JSON.
func padScenario(_ scenario: Scenario, advertisingAlso decoyNames: [String]) throws -> Scenario {
    guard !decoyNames.isEmpty else { return scenario }
    let data = try JSONEncoder().encode(scenario)
    guard var dict = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        return scenario
    }
    let existing = (dict["requiredTools"] as? [String]) ?? []
    // Required tool(s) first so the real tool keeps a stable, leading position.
    dict["requiredTools"] = existing + decoyNames.filter { !existing.contains($0) }
    let patched = try JSONSerialization.data(withJSONObject: dict)
    return try JSONDecoder().decode(Scenario.self, from: patched)
}

/// Result-grounding directive appended to every scenario's system prompt
/// (lever 1 of #100).
///
/// The four-model soak found tool-call *dispatch* is solid but the second turn
/// — the one AFTER a tool returns a result — is where llama/gemma fail: they
/// narrate ("I called the tool and it said…", or a paraphrase) instead of
/// answering FROM the tool result. The harness can't edit the second-turn
/// instruction the orchestrator emits (that lives in MK's
/// `GenerationToolDispatchLoop`), but the system prompt is in force on every
/// turn including that one, so strengthening it here is the cheapest, highest-
/// headroom grounding lift. Kept generic so it stacks on top of each scenario's
/// own (already grounding-flavoured) instruction without contradicting it; the
/// `structured-json` extraction scenario advertises no tools, so it is left
/// untouched (see `groundScenario`).
let resultGroundingDirective =
    "When a tool returns a result, answer USING that result directly — quote its "
    + "values verbatim where the user asks for them. Do NOT narrate that you called "
    + "a tool, do NOT paraphrase or recompute the result, and do NOT add facts the "
    + "tool did not return. The tool result is the ground truth for your answer."

/// Composes a scenario's grounded system prompt: its own instruction followed by
/// the shared ``resultGroundingDirective``. Scenarios that advertise no tools
/// (e.g. structured-JSON extraction) are returned unchanged — there is no tool
/// result for them to ground in, and the directive would only add noise.
func groundedSystemPrompt(base: String, requiredTools: [String]) -> String {
    guard !requiredTools.isEmpty else { return base }
    let trimmed = base.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return resultGroundingDirective }
    return trimmed + " " + resultGroundingDirective
}

/// Returns a copy of `scenario` whose `systemPrompt` carries the
/// ``resultGroundingDirective`` (lever 1 of #100). Mirrors ``padScenario``'s
/// Codable round-trip because `Scenario` has no public memberwise initialiser.
func groundScenario(_ scenario: Scenario) throws -> Scenario {
    let grounded = groundedSystemPrompt(
        base: scenario.systemPrompt,
        requiredTools: scenario.requiredTools)
    guard grounded != scenario.systemPrompt else { return scenario }
    let data = try JSONEncoder().encode(scenario)
    guard var dict = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        return scenario
    }
    dict["systemPrompt"] = grounded
    let patched = try JSONSerialization.data(withJSONObject: dict)
    return try JSONDecoder().decode(Scenario.self, from: patched)
}

// MARK: - Lever 2: grammar-constrained final-answer decoding (#100)

/// JSON-Schema for the `structured-json-extraction` scenario's expected output:
/// `{ invoice_id: string, total: string, currency: string }`.
///
/// Using string for `total` (not number) because models frequently emit the
/// value as a quoted string (e.g. `"123.45"`), matching the `containsAll`
/// assertion which accepts either form — the grammar must not be stricter than
/// the assertion gate.
let structuredJsonExtractionSchema: JSONSchemaValue = .object([
    "type": .string("object"),
    "properties": .object([
        "invoice_id": .object(["type": .string("string")]),
        "total": .object(["type": .string("string")]),
        "currency": .object(["type": .string("string")])
    ]),
    "required": .array([.string("invoice_id"), .string("total"), .string("currency")])
])

/// Returns a GBNF grammar constraining output to the expected JSON object for
/// `structured-json` extraction scenarios, or `nil` when the scenario is not a
/// no-tool extraction scenario or the backend does not support grammar-
/// constrained sampling (e.g. Gemma family).
///
/// The grammar activates ONLY on the final-answer (synthesis) turn. Extraction
/// scenarios have no tool calls so this is always the only turn; the constraint
/// is never applied on a tool-call turn.
func grammarForScenario(_ scenario: Scenario, backend: LlamaBackend) -> String? {
    guard scenario.requiredTools.isEmpty,
          scenario.id.hasPrefix("structured-json")
    else { return nil }
    guard backend.capabilities.supportsGrammarConstrainedSampling else { return nil }
    return ToolGrammarBuilder().buildObjectGrammar(for: structuredJsonExtractionSchema)
}

/// Minimal outcome from ``runScenarioWithGrammar``.
///
/// `ScenarioRunner.Outcome`'s memberwise initialiser is `internal`, so the
/// grammar-constrained runner returns its own parallel type. The call site in
/// ``runCLI`` only reads `finalAnswer`, `assertions`, and `passed`, which are
/// available on both types.
struct GrammarRunOutcome: Sendable {
    let finalAnswer: String
    let assertions: [AssertionOutcome]
    var passed: Bool { assertions.allSatisfy(\.passed) }
}

/// Runs `scenario` through `service` with `grammar` injected into the
/// generation config, collecting events and evaluating assertions.
///
/// Mirrors `ScenarioRunner.run` but adds `grammar` to the `GenerationConfig`.
/// `ScenarioRunner` is `final` with no grammar hook, so grammar injection
/// requires driving `service.enqueue` directly for the constrained case.
@MainActor
func runScenarioWithGrammar(
    _ scenario: Scenario,
    grammar: String,
    service: InferenceService,
    logger: TranscriptLogger?
) async throws -> GrammarRunOutcome {
    logger?.append(.prompt(
        scenarioId: scenario.id,
        system: scenario.systemPrompt,
        user: scenario.userPrompt,
        requiredTools: scenario.requiredTools))

    let messages: [StructuredMessage] = [
        StructuredMessage(role: "user", content: scenario.userPrompt)
    ]

    var config = GenerationConfig(
        temperature: Float(scenario.backend.temperature ?? 0.0),
        topP: 0.9,
        repeatPenalty: 1.1,
        topK: scenario.backend.topK.map(Int32.init),
        maxOutputTokens: 1024,
        maxToolIterations: 6
    )
    // Disable thinking explicitly: the #1595 grammar-phase gate holds the grammar
    // permissive until </think> closes. Extraction scenarios have no thinking
    // block, but a thinking-capable model might emit one; disabling it forces the
    // single strict sampler chain where the grammar applies from token 0.
    config.maxThinkingTokens = 0
    config.grammar = grammar

    var accumulatedText = ""

    let (_, stream) = try service.enqueue(
        structuredMessages: messages,
        systemPrompt: scenario.systemPrompt,
        config: config)

    for try await event in stream.events {
        switch event {
        case .token(let text):
            accumulatedText += text
            logger?.append(.tokenDelta(scenarioId: scenario.id, text: text))
        case .generationCompleted:
            continue
        default:
            continue
        }
    }

    logger?.append(.final(scenarioId: scenario.id, text: accumulatedText))

    var assertionOutcomes: [AssertionOutcome] = []
    for assertion in scenario.assertions {
        let outcome = AssertionEvaluator.evaluate(
            assertion,
            finalAnswer: accumulatedText)
        assertionOutcomes.append(outcome)
        logger?.append(.assertion(
            scenarioId: scenario.id,
            passed: outcome.passed,
            message: outcome.message))
    }

    return GrammarRunOutcome(finalAnswer: accumulatedText, assertions: assertionOutcomes)
}

/// Loads the tool-calling scenario corpus: ManifoldKit core's bundled
/// `built-in` scenarios (MK 0.64+ `ScenarioLoader.loadBuiltIn()`) with four
/// llama/gemma-tolerant overrides spliced in by id.
///
/// Nine of the ten scenario ids core ships are used verbatim. Four —
/// `shopping-list-budget`, `parallel-readme-comparison`,
/// `oversize-tool-output`, `structured-json-extraction` — carry intentional
/// wording differences in this package (looser `containsAny`/`containsAll`
/// assertion sets tuned from real llama/gemma soak runs; core's copies use
/// stricter literal-match wording). Those four stay vendored under
/// `ScenarioOverrides/` (a bundled `.copy` resource) and replace the
/// core-sourced scenario of the same id here.
func loadScenarios() throws -> [Scenario] {
    let base = try ScenarioLoader.loadBuiltIn()

    guard let overridesDir = Bundle.module.url(forResource: "ScenarioOverrides", withExtension: nil) else {
        throw NSError(
            domain: "manifold-tools-llama", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "bundled ScenarioOverrides directory not found in resource bundle"])
    }
    let overrides = try ScenarioLoader.load(from: overridesDir)
    let overridesByID = Dictionary(uniqueKeysWithValues: overrides.map { ($0.id, $0) })

    return base.map { overridesByID[$0.id] ?? $0 }
}

/// Resolves the fixture root the file/dir tools read against. Prefers an
/// explicit `--fixtures-root`, otherwise `ManifoldTools`'s own bundled fixture
/// tree (`ToolFixtures.bundledRoot()`, via `ScenarioCLIHarness`) — no longer
/// vendored in this package.
func resolveFixturesRoot(_ override: URL?) -> URL {
    ScenarioCLIHarness.resolveFixturesRoot(override)
}

/// `--describe`: static tool-call capability report (issue #2005 layers 1+2).
///
/// Reads GGUF metadata only (`ModelInfo.load`), builds the layer-1
/// `ChatTemplateToolDescriptor` from the embedded `tokenizer.chat_template`, and
/// runs the layer-2 `RenderConsistencyChecker`. No weights are mapped, no Metal
/// context is created, and nothing is generated — so this runs anywhere, in the
/// simulator, and in CI.
///
/// Exit codes mirror the rest of the tool: `0` success, `1` metadata read
/// failure. The capability verdict itself is informational (printed), not an
/// exit code — a `toolless` model is a legitimate, successful describe.
func describeModel(_ modelURL: URL) -> Int32 {
    let modelInfo: ModelInfo
    do {
        modelInfo = try ModelInfo.load(ggufURL: modelURL)
    } catch {
        FileHandle.standardError.write(Data("failed to read GGUF metadata: \(error)\n".utf8))
        return 1
    }

    let raw = modelInfo.chatTemplateRaw
    let descriptor = ChatTemplateToolDescriptor(parsingChatTemplate: raw)
    let consistency = RenderConsistencyChecker.check(chatTemplateRaw: raw)

    func dialectString(_ d: ChatTemplateToolDescriptor.ToolCallDialect?) -> String {
        guard let d else { return "—" }
        let open = d.openDelimiter ?? "(none)"
        let close = d.closeDelimiter ?? "(none)"
        return "open=\(open) close=\(close) args=\(d.argEncoding.rawValue)"
    }

    print("Model: \(modelURL.lastPathComponent)")
    print("  embedded chat_template: \(raw != nil ? "present" : "ABSENT")")
    print("  — Layer 1 (ChatTemplateToolDescriptor, static) —")
    print("    toolsExpressible : \(descriptor.toolsExpressible)")
    print("    declaredDialect  : \(dialectString(descriptor.declaredDialect))")
    print("    extractability   : \(descriptor.extractability.rawValue)")
    print("  — Layer 2 (RenderConsistencyChecker, static) —")
    print("    status                  : \(consistency.status)")
    print("    toolDefinitionRendered  : \(consistency.toolDefinitionRendered)")
    print("    declaredDelimiterRendered: \(consistency.declaredDelimiterRendered.map(String.init(describing:)) ?? "n/a")")
    if !consistency.detail.isEmpty {
        print("    detail                  : \(consistency.detail)")
    }
    return 0
}

@MainActor
func runCLI() async -> Int32 {
    let argv = Array(CommandLine.arguments.dropFirst())
    let cli = CLI.parse(argv)

    let scenarios: [Scenario]
    do {
        scenarios = try loadScenarios()
    } catch {
        FileHandle.standardError.write(Data("failed to load scenarios: \(error)\n".utf8))
        return 1
    }

    if cli.common.list {
        print("Available scenarios:")
        for s in scenarios {
            print("  \(s.id) — \(s.description)")
        }
        return 0
    }

    guard let modelPath = cli.modelPath else {
        FileHandle.standardError.write(Data(
            "manifold-tools-llama: --model <path.gguf> is required (use --list to inspect scenarios)\n".utf8))
        return 2
    }
    let modelURL = URL(fileURLWithPath: modelPath)
    guard FileManager.default.fileExists(atPath: modelURL.path) else {
        FileHandle.standardError.write(Data("model file not found: \(modelURL.path)\n".utf8))
        return 1
    }

    // Describe mode short-circuits everything: it reads GGUF metadata only and
    // reports the STATIC tool-call capability (issue #2005 layers 1+2) — no
    // weights, no Metal, no generation. This is the free/cheap signal that
    // precedes the scenario soak: a model whose template cannot express tools
    // (no tools guard) is honestly `unsupported` without ever running it, and a
    // template that declares a dialect MK's renderer does not emit is flagged by
    // the render-consistency check (the #1909 class) with no model run.
    if cli.describe {
        return describeModel(modelURL)
    }

    // Benchmark mode short-circuits the scenario harness: it drives LlamaBackend
    // directly to time the cold (first) vs warm (subsequent) generations and the
    // one-time per-process Metal pipeline warm-up between them.
    if cli.bench {
        return await Benchmark.run(
            modelURL: modelURL,
            flash: cli.flash,
            prompt: cli.benchPrompt,
            maxTokens: cli.maxTokens,
            warmRuns: cli.warmRuns,
            contextSize: cli.context)
    }

    let filtered: [Scenario]
    do {
        filtered = try ScenarioCLIHarness.filterScenarios(scenarios, matching: cli.common.scenarioFilter)
    } catch {
        FileHandle.standardError.write(Data("\(error)\n".utf8))
        return 1
    }

    let fixturesRoot = resolveFixturesRoot(cli.common.fixturesRoot)
    print("Fixtures root: \(fixturesRoot.path)")

    let logger: TranscriptLogger
    do {
        logger = try TranscriptLogger(url: cli.common.output)
    } catch {
        FileHandle.standardError.write(Data("failed to open log: \(error)\n".utf8))
        return 1
    }
    print("Logging to \(logger.destination.path)")

    // Build one InferenceService, register ALL reference tools once, and load
    // the GGUF through the *production* load path so the model's native chat
    // template is rendered (#69).
    //
    // Why the production load path (not the `init(backend:)` seam): the renderer
    // (`PromptRenderer`/`JinjaPromptRenderer`) only injects the native tool block
    // when `GenerationQueue` is given the model's embedded `tokenizer.chat_template`
    // (its `selectedChatTemplateRaw`). That raw template is set on
    // `ModelLifecycleCoordinator` *only* from `ModelInfo.chatTemplateRaw` inside
    // `InferenceService.loadModel(from:plan:)`. The `init(backend:)` seam the
    // harness previously used never sets it, so the queue fell back to the
    // ChatML enum — which renders tools only for `.gemma4`. The result was 0 tool
    // dispatches and `<|im_start|>/<|im_end|>` markers on non-ChatML models
    // (llama3.1 / Mistral). Loading via `loadModel(from: ModelInfo, plan:)`
    // reads the GGUF metadata (`ModelInfo.load(ggufURL:)`) and threads the
    // embedded template + detected enum into the renderer; native stop tokens
    // already come from the loaded model via llama.cpp, not the template enum.
    //
    // Per-scenario tool scoping is handled by `ScenarioRunner` itself (it filters
    // the service's registry by `scenario.requiredTools`), so a single
    // all-tools registry + a single service is sufficient — no per-scenario
    // service churn (#66 scoping is preserved by the runner's own filter).
    let registry = makeFullRegistry(fixturesRoot: fixturesRoot)
    // Decoy padding (--extra-tools): register N distractor tools into the shared
    // registry so the runner can advertise them alongside each scenario's real
    // tool. Names are spliced into each scenario's `requiredTools` below.
    let decoyNames = registerDecoys(count: cli.common.extraTools, into: registry)
    if !decoyNames.isEmpty {
        print("Decoy tools (--extra-tools \(cli.common.extraTools)): \(decoyNames.joined(separator: ", "))")
    }
    let service = InferenceService(toolRegistry: registry)
    // Register the GGUF backend factory so `loadModel(from:plan:)` constructs and
    // installs the backend through the coordinator (the path that captures the
    // embedded chat template). We capture the constructed instance so we can
    // `unloadAndWait()` it deterministically before `exit()` — the coordinator's
    // own `unloadModel()` is fire-and-forget and would race the process exit
    // (the Metal residency-set SIGABRT the harness teardown guards against).
    let backendBox = BackendBox()
    service.registerBackendFactory { modelType in
        guard modelType == .gguf else { return nil }
        let backend = LlamaBackend()
        backendBox.backend = backend
        return backend
    }
    service.declareSupport(for: .gguf)

    let modelInfo: ModelInfo
    do {
        modelInfo = try ModelInfo.load(ggufURL: modelURL)
    } catch {
        FileHandle.standardError.write(Data("failed to read GGUF metadata: \(error)\n".utf8))
        return 1
    }
    if modelInfo.chatTemplateRaw == nil {
        // Not fatal — the renderer falls back to the detected enum — but flag it,
        // since a templateless GGUF cannot render the native tool block.
        FileHandle.standardError.write(Data(
            "manifold-tools-llama: WARNING — \(modelURL.lastPathComponent) has no embedded tokenizer.chat_template; tool rendering falls back to the \(modelInfo.detectedPromptTemplate.map(String.init(describing:)) ?? "ChatML") enum\n".utf8))
    }

    do {
        print("Loading model: \(modelURL.path)")
        print("  embedded chat_template: \(modelInfo.chatTemplateRaw != nil ? "present" : "ABSENT")"
            + ", detected template: \(modelInfo.detectedPromptTemplate.map(String.init(describing:)) ?? "nil")")
        try await service.loadModel(from: modelInfo, plan: .systemManaged(requestedContextSize: 4096))
    } catch {
        // Fix 2 — load failure is a DISTINCT, loud outcome (separate exit code 3),
        // so a sweep can tell "model never loaded" (arch/llama.cpp version skew —
        // e.g. qwen3.5-4b's `rope.dimension_sections` mismatch, gemma4-e4b's
        // `unsupportedModelArchitecture`) apart from "loaded but did not dispatch"
        // (empty/garbage transcript, exit 1). Previously both collapsed to a
        // generic "failed to load model" + exit 1 that read as "no dispatch" in
        // the campaign and caused a misdiagnosis. Surface the underlying error
        // verbatim (`errorDescription` when available, then the raw value) so the
        // root cause is visible without re-running under a debugger.
        let detail = (error as? LocalizedError)?.errorDescription ?? "\(error)"
        let banner = "LOAD FAILED: \(modelURL.lastPathComponent): \(detail)"
        FileHandle.standardError.write(Data((banner + "\n").utf8))
        print(banner)
        print("  (model never loaded — this is a load failure, NOT a tool-dispatch failure)")
        // Best-effort teardown of any half-constructed backend before exit.
        if let backend = backendBox.backend {
            await backend.unloadAndWait()
        }
        return 3
    }
    // Teardown must be awaited before `exit()` reclaims the process, so a fire-
    // and-forget `Task` in `defer` would race the exit and routinely never run.
    // Run the scenarios, then await `unloadAndWait()` on every exit path below.

    // Templateless-model tool-format instruction is now handled UPSTREAM: as of
    // ManifoldKit 0.58 (MK#2002), `GenerationQueue.toolAugmentedSystemPrompt`
    // folds `ToolSystemPromptBuilder.preferTools(for:)` — which spells out the
    // exact `{"name": …, "arguments": {…}}` envelope, named-argument enumeration,
    // and the "no Python-style call" prohibition — into the system prompt for any
    // model whose renderer does NOT emit tools natively (Phi-3.5, Mistral-7B, and
    // every non-`gemma4` enum template). Since the harness routes through the
    // production `InferenceService` → `GenerationQueue` path (#69), that preamble
    // already reaches templateless models. The harness no longer injects its own
    // instruction — doing so would double-instruct. (Verified: Phi-3.5 dispatches
    // `calc` with correct args on 0.58 with no harness injection.)

    var allPassed = true
    for baseScenario in filtered {
        // #99 — `--scenario all` reuses a single `service` (one backend, one KV
        // cache, one conversation) across every scenario. The orchestrator
        // appends each `enqueue` turn to that shared conversation and reuses the
        // resident KV cache (it emits `.kvCacheReuse`), so a later scenario
        // prefills on top of an earlier scenario's tokens and can run out of
        // context budget mid-answer — observed as intermittent truncation (e.g.
        // qwen3-0.6B's `structured-json-extraction` clipped to ` ```json\n `).
        // Reset the conversation and zero the KV cache BEFORE each scenario so
        // every scenario starts from a clean context, matching the deterministic
        // behaviour of an isolated `--scenario <id>` run. Done at the top of the
        // loop (not the bottom) so it still runs after a `continue`, and is a
        // harmless no-op on the very first iteration.
        service.resetConversation()
        service.secureWipe()

        let scenario: Scenario
        do {
            // Decoy padding extends `requiredTools` so `--extra-tools` decoys are
            // advertised alongside the scenario's real tool. The templateless
            // tool-format instruction is supplied upstream by 0.58's
            // `ToolSystemPromptBuilder` fold (MK#2002), so no per-scenario system
            // prompt injection happens here for tool *format*.
            //
            // Tool-*result grounding* (lever 1 of #100) IS injected here: the
            // system prompt is in force on the second turn (after a tool returns)
            // where the soak found llama/gemma narrate instead of grounding, and
            // the harness cannot reach the orchestrator's second-turn instruction.
            let padded = try padScenario(baseScenario, advertisingAlso: decoyNames)
            scenario = try groundScenario(padded)
        } catch {
            allPassed = false
            print("\n── \(baseScenario.id) — ERROR preparing scenario: \(error)")
            continue
        }
        print("\n── \(scenario.id) (\(scenario.description)) ──")
        print("  advertising tool(s): \(scenario.requiredTools.joined(separator: ", "))")
        do {
            // Lever 2 of #100: for structured-json extraction scenarios, apply a
            // GBNF grammar on the final-answer turn. The grammar constrains output
            // to the expected JSON object shape, eliminating markdown fences and
            // prose wrapping that cause `containsAll` assertions to fail even when
            // the model has the right values. Falls through to the standard runner
            // when the backend does not support grammar sampling (Gemma family).
            let assertions: [AssertionOutcome]
            let finalAnswer: String
            let passed: Bool
            if let grammar = backendBox.backend.flatMap({ grammarForScenario(scenario, backend: $0) }) {
                print("  grammar: structured-JSON extraction constraint active")
                let outcome = try await runScenarioWithGrammar(
                    scenario,
                    grammar: grammar,
                    service: service,
                    logger: logger)
                assertions = outcome.assertions
                finalAnswer = outcome.finalAnswer
                passed = outcome.passed
            } else {
                // Standard path: drive scenarios through the production
                // InferenceService → GenerationQueue → dispatch-loop. That path
                // renders the chat template and injects each scenario's tool
                // definitions (#1983/#1985); the runner filters the registry by
                // `scenario.requiredTools` so each run advertises only its own tools.
                let runner = ScenarioRunner(service: service, logger: logger)
                let outcome = try await runner.run(scenario)
                assertions = outcome.assertions
                finalAnswer = outcome.finalAnswer
                passed = outcome.passed
            }
            for assertion in assertions {
                let marker = assertion.passed ? "  PASS" : "  FAIL"
                print("\(marker) \(assertion.message)")
            }
            if !passed {
                allPassed = false
                print("  final answer: \(finalAnswer.prefix(200))")
            }
        } catch {
            allPassed = false
            print("  ERROR \(error)")
        }
    }

    if let backend = backendBox.backend {
        await backend.unloadAndWait()
    } else {
        service.unloadModel()
    }

    return ScenarioCLIHarness.finish(allPassed: allPassed, transcriptPath: logger.destination)
}

let exitCode = await runCLI()
exit(exitCode)
