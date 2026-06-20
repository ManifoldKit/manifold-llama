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

    // Register all six reference tools. The file/dir tools read against the
    // resolved (bundled or overridden) fixture root rather than ManifoldTools'
    // default, which points at a non-existent ManifoldKit test path here.
    let registry = ToolRegistry()
    registry.register(NowTool.makeExecutor())
    registry.register(CalcTool.makeExecutor())
    registry.register(ReadFileTool.makeExecutor(root: fixturesRoot))
    registry.register(ListDirTool.makeExecutor(root: fixturesRoot))
    registry.register(SampleRepoSearchTool.makeExecutor(root: fixturesRoot))
    registry.register(HttpGetFixtureTool.makeExecutor())

    // Load the GGUF once and reuse the backend across every scenario — model
    // load dominates wall-clock and llama.cpp uses a process-global backend.
    let backend = LlamaBackend()
    do {
        print("Loading model: \(modelURL.path)")
        try await backend.loadModel(from: modelURL, plan: .systemManaged(requestedContextSize: 4096))
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
        do {
            let runner = ScenarioRunner(backend: backend, registry: registry, logger: logger)
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

    await backend.unloadAndWait()

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
