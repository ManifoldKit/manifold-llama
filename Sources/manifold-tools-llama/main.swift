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

        FLAGS
          --model <path>        Path to the .gguf model file. REQUIRED (except for --list / --help).
          --scenario <id>       Scenario id (matches JSON 'id') or 'all'. Default: all.
          --output <path>       Transcript JSONL destination. Default: tmp/manifold-tools-llama/<iso>.jsonl.
          --fixtures-root <dir> Override the file/dir tool fixture root. Default: bundled fixtures.
          --list                Print available scenarios and exit (no model needed).
          --help                Show this text.

        EXIT
          0 — all scenarios passed.
          1 — at least one scenario or assertion failed (or a load/setup error).
          2 — bad arguments.

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
        FileHandle.standardError.write(Data("failed to load model: \(error)\n".utf8))
        return 1
    }
    // Teardown must be awaited before `exit()` reclaims the process, so a fire-
    // and-forget `Task` in `defer` would race the exit and routinely never run.
    // Run the scenarios, then await `unloadAndWait()` on every exit path below.

    var allPassed = true
    for scenario in filtered {
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
