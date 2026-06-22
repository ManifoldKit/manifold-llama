// manifold-tools-llama — run ManifoldKit's tool-calling validation scenarios
// against a real llama.cpp / GGUF model.
//
// This reuses the published `ManifoldTools` library product from ManifoldKit
// (bundled scenarios + reference tools + scenario runner) and drives them
// through this package's `LlamaBackend`. There are NO changes to ManifoldKit
// core — the only Llama-specific wiring is the backend construction and model
// load, plus a vendored copy of the fixture tree the file/dir tools read
// (ManifoldTools' `ReadFileTool.defaultRoot()` resolves to a ManifoldKit test
// path that does not travel with the library product).
//
// Real-hardware tool: requires Apple Silicon + Metal (llama.cpp uses a
// process-global backend init and has no Metal support in the simulator) and
// a local `.gguf` model. Compilation does not need a model; running does.
import Foundation
import ManifoldInference
import ManifoldModelCatalog
import ManifoldTools
import ManifoldLlama

/// Hand-rolled argument parser — mirrors `manifold-tools` in ManifoldKit core.
/// Pulling in swift-argument-parser for a ~150-line harness is not worth the
/// Package.swift churn; the syntax is small enough to parse in place.
struct CLI {

    var scenarioFilter: String = "all"
    var modelPath: String? = nil
    var output: URL? = nil
    var fixturesRoot: URL? = nil
    var list: Bool = false
    /// Static tool-call capability report (issue #2005 layers 1+2): parses the
    /// model's embedded chat template into a `ChatTemplateToolDescriptor` and
    /// runs `RenderConsistencyChecker`. Reads GGUF metadata only — NO weights,
    /// no Metal, no generation. Distinct from the scenario soak (the measured
    /// positive verdict) and from `--bench` (timing).
    var describe: Bool = false
    /// Number of decoy (distractor) tools to advertise alongside each scenario's
    /// required tool(s). Used to measure how a model's tool selection degrades as
    /// the advertised tool set grows. Default 0 preserves the original behaviour
    /// (advertise only the scenario's `requiredTools`).
    var extraTools: Int = 0

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

    static func parse(_ argv: [String]) -> CLI {
        var cli = CLI()
        var i = 0
        while i < argv.count {
            let arg = argv[i]
            switch arg {
            case "--scenario":
                i += 1
                guard i < argv.count else { fail("--scenario requires a value") }
                cli.scenarioFilter = argv[i]
            case "--model":
                i += 1
                guard i < argv.count else { fail("--model requires a value") }
                cli.modelPath = argv[i]
            case "--output":
                i += 1
                guard i < argv.count else { fail("--output requires a value") }
                cli.output = URL(fileURLWithPath: argv[i])
            case "--fixtures-root":
                i += 1
                guard i < argv.count else { fail("--fixtures-root requires a value") }
                cli.fixturesRoot = URL(fileURLWithPath: argv[i], isDirectory: true)
            case "--extra-tools":
                i += 1
                guard i < argv.count else { fail("--extra-tools requires a value") }
                guard let n = Int(argv[i]), n >= 0 else { fail("--extra-tools requires a non-negative integer") }
                cli.extraTools = n
            case "--bench":
                cli.bench = true
            case "--flash":
                i += 1
                guard i < argv.count else { fail("--flash requires a value (on|off|both)") }
                cli.flash = argv[i]
            case "--bench-prompt":
                i += 1
                guard i < argv.count else { fail("--bench-prompt requires a value") }
                cli.benchPrompt = argv[i]
            case "--max-tokens":
                i += 1
                guard i < argv.count else { fail("--max-tokens requires a value") }
                guard let n = Int(argv[i]), n > 0 else { fail("--max-tokens requires a positive integer") }
                cli.maxTokens = n
            case "--warm-runs":
                i += 1
                guard i < argv.count else { fail("--warm-runs requires a value") }
                guard let n = Int(argv[i]), n >= 0 else { fail("--warm-runs requires a non-negative integer") }
                cli.warmRuns = n
            case "--context":
                i += 1
                guard i < argv.count else { fail("--context requires a value") }
                guard let n = Int(argv[i]), n > 0 else { fail("--context requires a positive integer") }
                cli.context = n
            case "--describe":
                cli.describe = true
            case "--list":
                cli.list = true
            case "--help", "-h":
                printUsage()
                exit(0)
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

/// Loads the bundled tool-calling scenarios.
///
/// `ScenarioLoader.loadBuiltIn()` is unusable here: it resolves a ManifoldKit
/// *source*-relative path (`<cwd>/Sources/ManifoldTools/Scenarios/built-in`)
/// that exists only when run from the ManifoldKit package root. We vendor the
/// scenario JSONs as a bundled `.copy` resource and drive the public
/// `ScenarioLoader.load(from:)` against that directory instead.
func loadScenarios() throws -> [Scenario] {
    guard let dir = Bundle.module.url(forResource: "Scenarios", withExtension: nil) else {
        throw NSError(
            domain: "manifold-tools-llama", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "bundled Scenarios directory not found in resource bundle"])
    }
    return try ScenarioLoader.load(from: dir)
}

/// Resolves the fixture root the file/dir tools read against. Prefers an
/// explicit `--fixtures-root`, otherwise the bundled copy resource. Exits 2 if
/// neither resolves — the file/dir scenarios would otherwise fail confusingly.
func resolveFixturesRoot(_ override: URL?) -> URL {
    if let override {
        return override
    }
    // SwiftPM's `.copy("Fixtures/manifold-tools")` flattens to the trailing
    // path component, so the tree lands at the bundle root as `manifold-tools/`
    // — there is no `Fixtures/` subdirectory in the built bundle. Look it up at
    // the root (no `subdirectory:`); passing `subdirectory: "Fixtures"` returns
    // nil and would wrongly exit(2) on every real run.
    guard let bundled = Bundle.module.url(
        forResource: "manifold-tools",
        withExtension: nil)
    else {
        FileHandle.standardError.write(Data(
            "manifold-tools-llama: bundled fixtures not found — pass --fixtures-root <dir>\n".utf8))
        exit(2)
    }
    return bundled
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

    if cli.list {
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
    if cli.scenarioFilter == "all" {
        filtered = scenarios
    } else {
        filtered = scenarios.filter { $0.id == cli.scenarioFilter }
        if filtered.isEmpty {
            FileHandle.standardError.write(Data("no scenario matches id '\(cli.scenarioFilter)'\n".utf8))
            return 1
        }
    }

    let fixturesRoot = resolveFixturesRoot(cli.fixturesRoot)
    print("Fixtures root: \(fixturesRoot.path)")

    let logger: TranscriptLogger
    do {
        logger = try TranscriptLogger(url: cli.output ?? CLI.defaultOutputURL())
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
    let decoyNames = registerDecoys(count: cli.extraTools, into: registry)
    if !decoyNames.isEmpty {
        print("Decoy tools (--extra-tools \(cli.extraTools)): \(decoyNames.joined(separator: ", "))")
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
            // Drive scenarios through the production InferenceService →
            // GenerationQueue → dispatch-loop path. That path renders the chat
            // template and injects each scenario's tool definitions the model
            // needs to see (#1983/#1985); the runner filters the registry by
            // `scenario.requiredTools` so each run advertises only its own tools.
            let runner = ScenarioRunner(service: service, logger: logger)
            let outcome = try await runner.run(scenario)
            for assertion in outcome.assertions {
                let marker = assertion.passed ? "  PASS" : "  FAIL"
                print("\(marker) \(assertion.message)")
            }
            if !outcome.passed {
                allPassed = false
                print("  final answer: \(outcome.finalAnswer.prefix(200))")
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

    if allPassed {
        print("\nAll scenarios passed.")
        return 0
    } else {
        print("\nOne or more scenarios failed — see \(logger.destination.path)")
        return 1
    }
}

let exitCode = await runCLI()
exit(exitCode)
