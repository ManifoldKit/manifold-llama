// manifold-llama-eval — run ONE raw-prompt generation through LlamaBackend and
// emit exactly one `RawRun` JSON object on stdout.
//
// This is the llama.cpp leg of the manifold-eval same-GGUF cross-backend
// differential (vs Ollama). It feeds the `--prompt-file` bytes verbatim to
// `LlamaBackend.generate(prompt:systemPrompt:config:)` — no chat template — and
// records the prompt hash, input token ids, output, and sampler/tooling
// metadata in the fixed `RawRun` contract. See ManifoldKit
// docs/plans/manifold-eval-repo-v2-override.md §13b.
//
// Real-hardware tool: requires Apple Silicon + Metal (llama.cpp has no simulator
// Metal support) and a local `.gguf`. Only the single RawRun JSON object is
// written to stdout; all diagnostics go to stderr.
import Foundation
import ManifoldLlamaEvalKit

/// Hand-rolled argument parser — mirrors `manifold-tools-llama`; pulling in
/// swift-argument-parser for a handful of flags is not worth the churn.
struct CLI {
    var modelPath: String?
    var promptFile: String?
    var temperature: Double = 0.0
    var seed: Int = 0
    var maxTokens: Int = 256
    var repeatIndex: Int = 0
    var topK: Int = 0
    var repeatPenalty: Double = 1.0

    /// Argument errors exit 2 via stderr + `exit(2)` rather than trapping.
    private static func fail(_ message: String) -> Never {
        FileHandle.standardError.write(Data("manifold-llama-eval: \(message)\n".utf8))
        exit(2)
    }

    static func parse(_ argv: [String]) -> CLI {
        var cli = CLI()
        var i = 0
        while i < argv.count {
            let arg = argv[i]
            func nextValue(_ label: String) -> String {
                i += 1
                guard i < argv.count else { fail("\(label) requires a value") }
                return argv[i]
            }
            switch arg {
            case "--model":
                cli.modelPath = nextValue("--model")
            case "--prompt-file":
                cli.promptFile = nextValue("--prompt-file")
            case "--temperature":
                let raw = nextValue("--temperature")
                guard let value = Double(raw), value >= 0 else {
                    fail("--temperature requires a non-negative number")
                }
                cli.temperature = value
            case "--seed":
                let raw = nextValue("--seed")
                guard let value = Int(raw) else { fail("--seed requires an integer") }
                cli.seed = value
            case "--max-tokens":
                let raw = nextValue("--max-tokens")
                guard let value = Int(raw), value > 0 else {
                    fail("--max-tokens requires a positive integer")
                }
                cli.maxTokens = value
            case "--repeat-index":
                let raw = nextValue("--repeat-index")
                guard let value = Int(raw), value >= 0 else {
                    fail("--repeat-index requires a non-negative integer")
                }
                cli.repeatIndex = value
            case "--top-k":
                let raw = nextValue("--top-k")
                // Upper-bounded by Int32.max: `GenerationConfig.topK` is `Int32?`
                // (see ManifoldKit's ManifoldContract), and `EvalRunner` converts
                // via the non-failable `Int32(_:)` initializer, which TRAPS on
                // overflow rather than failing gracefully. Reject out-of-range
                // values here so a bad value is a clean exit(2), not a crash.
                guard let value = Int(raw), value >= 0, value <= Int(Int32.max) else {
                    fail("--top-k requires a non-negative integer no greater than \(Int32.max)")
                }
                cli.topK = value
            case "--repeat-penalty":
                let raw = nextValue("--repeat-penalty")
                // Only rejects <= 0 (not <= 1.0): llama.cpp's own CLI doesn't
                // reject sub-1.0 values either, and the differential's more
                // permissive default mode may want to explore below the no-op
                // threshold. Note this means values in (0, 1.0] are accepted but
                // inert — see the driver's `effectiveRepetitionPenalty > 1.0`
                // gate in LlamaGenerationDriver.swift, and the FLAGS note below.
                guard let value = Double(raw), value > 0 else {
                    fail("--repeat-penalty requires a positive number")
                }
                cli.repeatPenalty = value
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

    static func printUsage() {
        let text = """
        manifold-llama-eval — one raw-prompt generation → one RawRun JSON object

        USAGE
          manifold-llama-eval --model <path.gguf> --prompt-file <path> \\
              [--temperature <d>] [--seed <n>] [--max-tokens <n>] [--repeat-index <n>] \\
              [--top-k <n>] [--repeat-penalty <d>]

        FLAGS
          --model <path>          Path to the .gguf model file. REQUIRED.
          --prompt-file <path>    File whose bytes are the raw prompt (fed verbatim,
                                  NO chat template). REQUIRED.
          --temperature <d>       Sampling temperature. 0 → greedy. Default: 0.
          --seed <n>              Sampling seed. Default: 0.
          --max-tokens <n>        Max tokens to generate. Default: 256.
          --repeat-index <n>      0-based index within a repeat sweep. Default: 0.
          --top-k <n>             Top-k sampling cutoff. 0 → not applied (the
                                  neutral default; also has no effect at
                                  temperature 0, which always decodes greedily
                                  regardless of this value). Default: 0.
          --repeat-penalty <d>    Repetition penalty (1.0 = no-op, llama.cpp's
                                  neutral default). Must be > 0; values <= 1.0
                                  are accepted but have no effect (the driver
                                  only applies the penalty above 1.0).
                                  Default: 1.0.

        OUTPUT
          Exactly one RawRun JSON object on stdout. Diagnostics go to stderr.

        EXIT
          0 — success (one RawRun written).
          1 — runtime failure (load/generation/IO error; message on stderr).
          2 — bad arguments.

        REQUIREMENTS
          Apple Silicon + Metal and a local .gguf model.
        """
        print(text)
    }
}

func runMain() async -> Int32 {
    let argv = Array(CommandLine.arguments.dropFirst())
    let cli = CLI.parse(argv)

    guard let modelPath = cli.modelPath else {
        FileHandle.standardError.write(Data("manifold-llama-eval: --model <path.gguf> is required\n".utf8))
        return 2
    }
    guard let promptFile = cli.promptFile else {
        FileHandle.standardError.write(Data("manifold-llama-eval: --prompt-file <path> is required\n".utf8))
        return 2
    }

    let options = EvalOptions(
        modelPath: modelPath,
        promptFile: promptFile,
        temperature: cli.temperature,
        seed: cli.seed,
        maxTokens: cli.maxTokens,
        repeatIndex: cli.repeatIndex,
        topK: cli.topK,
        repeatPenalty: cli.repeatPenalty)

    do {
        let run = try await EvalRunner.run(options)
        let line = try run.encodedJSONLine()
        // The single RawRun object — the only thing on stdout.
        print(line)
        return 0
    } catch {
        let detail = (error as? LocalizedError)?.errorDescription ?? "\(error)"
        FileHandle.standardError.write(Data("manifold-llama-eval: \(detail)\n".utf8))
        return 1
    }
}

let exitCode = await runMain()
exit(exitCode)
