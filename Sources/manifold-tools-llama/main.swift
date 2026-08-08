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
import ManifoldLlama
import ManifoldModelCatalog
import ManifoldTools

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

  /// Write the normalized `[ConformanceRecord]` schema (ManifoldKit #2041) to
  /// this path after the run, matching the shape and semantics
  /// `manifold-tools score --emit-records` and `manifold-tools-mlx
  /// --emit-records` already produce — so a downstream collator can fold all
  /// three legs without re-deriving per-backend vocabulary (manifold-llama#178).
  var emitRecords: URL? = nil

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
        guard let n = Int(remainder[i]), n > 0 else {
          fail("--max-tokens requires a positive integer")
        }
        cli.maxTokens = n
      case "--warm-runs":
        i += 1
        guard i < remainder.count else { fail("--warm-runs requires a value") }
        guard let n = Int(remainder[i]), n >= 0 else {
          fail("--warm-runs requires a non-negative integer")
        }
        cli.warmRuns = n
      case "--context":
        i += 1
        guard i < remainder.count else { fail("--context requires a value") }
        guard let n = Int(remainder[i]), n > 0 else {
          fail("--context requires a positive integer")
        }
        cli.context = n
      case "--describe":
        cli.describe = true
      case "--emit-records":
        i += 1
        guard i < remainder.count else { fail("--emit-records requires a value") }
        cli.emitRecords = URL(fileURLWithPath: remainder[i])
      default:
        fail("unknown argument: \(arg)")
      }
      i += 1
    }
    return cli
  }

  static func defaultOutputURL() -> URL {
    let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    return cwd.appendingPathComponent(
      "tmp/manifold-tools-llama/\(TranscriptLogger.defaultFilename())")
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
        --emit-records <path>  Write the normalized [ConformanceRecord] JSON
                              (the cross-leg eval schema, ManifoldKit #2041)
                              to <path> after the run. Additive — the
                              transcript at --output is still written.
                              Absence (a missing GGUF, a failed load) is
                              recorded as a notMeasured/loadFail hole for
                              every requested scenario, never as a measured
                              failure — see ConformanceRecord's CellStatus.
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

/// Registers the first `count` decoys from `ManifoldTools.DecoyTools` — the
/// shared, deterministic pool published from ManifoldKit core specifically so
/// this repo, `manifold-tools-mlx`, and core's own `manifold-tools` all
/// advertise the SAME distractor set at the same `--extra-tools N` level
/// (its own doc comment: "the companions, from their next adoption pass,
/// [should] delete their local copy and depend on one deterministic pool").
///
/// This CLI previously hand-rolled its own 24-entry local pool, which had
/// already drifted from core's: 4 of the 24 names weren't in core's 46-entry
/// pool at all, so `--extra-tools 10` on this leg advertised `decoyLevel: 7`
/// once scored against core's pool — and a sweep folding this leg with
/// Ollama/MLX at the "same" decoy level was actually comparing two DIFFERENT
/// advertised sets, confounding decoy-identity divergence with genuine
/// runtime divergence (found live during the collate probe for #178).
///
/// Returns the registered names, in pool order, so the caller can add them to
/// a scenario's advertised set. Returns `[]` for `count <= 0`; `DecoyTools`
/// itself clamps `count` to its pool size (`maxCount`), so no local clamping
/// is needed here.
@MainActor
func registerDecoys(count: Int, into registry: ToolRegistry) -> [String] {
  guard count > 0 else { return [] }
  for executor in DecoyTools.executors(count) {
    registry.register(executor)
  }
  return DecoyTools.names(count)
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
/// ``resultGroundingDirective`` (lever 1 of #100). Round-trips through
/// `Scenario`'s `Codable` conformance because it has no public memberwise
/// initialiser.
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
    "currency": .object(["type": .string("string")]),
  ]),
  "required": .array([.string("invoice_id"), .string("total"), .string("currency")]),
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
  logger?.append(
    .prompt(
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
    logger?.append(
      .assertion(
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

  guard let overridesDir = Bundle.module.url(forResource: "ScenarioOverrides", withExtension: nil)
  else {
    throw NSError(
      domain: "manifold-tools-llama", code: 1,
      userInfo: [
        NSLocalizedDescriptionKey:
          "bundled ScenarioOverrides directory not found in resource bundle"
      ])
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
  print(
    "    declaredDelimiterRendered: \(consistency.declaredDelimiterRendered.map(String.init(describing:)) ?? "n/a")"
  )
  if !consistency.detail.isEmpty {
    print("    detail                  : \(consistency.detail)")
  }
  return 0
}

// MARK: - ConformanceRecord emission (--emit-records)

/// Best-effort quantization label parsed from a GGUF file name — the llama.cpp
/// analogue of `manifold-tools`' `quantLabel(from:)` (Sources/manifold-tools/main.swift
/// in ManifoldKit core), which parses an Ollama tag instead. llama.cpp model
/// identity is a file path, not a tag, so this keys off the last path component.
/// Returns `"unknown"` when nothing matches (`quant` still gets a value — every
/// leg stamps a non-optional `quant` on `ConformanceRecord`, unlike core's
/// scorer row where it's optional).
func quantLabel(fromFileName name: String) -> String {
  // BLOCKER 2 (rev-183 on #183): the previous separator-split approach
  // fragmented the label BEFORE the detector ever saw it — splitting on "_"
  // broke "Q4_K_M" into "Q4"/"K"/"M" tokens, so the detector matched only
  // the leading "Q4" and silently dropped the "_K_M" sub-label (13 of 14
  // real GGUFs on the reviewer's machine truncated this way). It also never
  // matched I-quants at all (`hasPrefix("q")` fails on "iq4_xs" — the "I"
  // prefix), and its reversed-token scan had no ordering guard between the
  // quant and precision branches, so a name like "m-Q4_K_M-f16.gguf" could
  // return the wrong, later-scanned "f16" instead of the real "Q4_K_M".
  //
  // Fix: match the FULL quant marker as one boundary-delimited run against
  // the ORIGINAL (unsplit) string, and always prefer a quant match over a
  // precision match regardless of where either sits in the name.
  if let quant = firstMatch(in: name, pattern: #"\bI?Q\d+(?:_[A-Za-z0-9]+)*\b"#) {
    return quant  // Q4_K_M, Q8_0, Q5_K_S, IQ4_XS, IQ2_XXS, ...
  }
  if let precision = firstMatch(in: name, pattern: #"\b(?:fp16|fp32|bf16|f16|f32)\b"#) {
    return precision
  }
  if let intN = firstMatch(in: name, pattern: #"\bint\d+\b"#) {
    return intN  // int4, int8
  }
  return "unknown"
}

/// Returns the first case-insensitive regex match of `pattern` in `string`,
/// verbatim (original casing, e.g. `"Q4_K_M"` not `"q4_k_m"`).
///
/// `pattern` is always a fixed string literal at every call site above, so a
/// compile failure here would be a build-time-catchable typo, not a runtime
/// condition — logging and treating it as "not found" (rather than trapping
/// the whole CLI) keeps a cosmetic regex mistake from taking down a run over
/// nothing worse than a missing Quant-column label.
func firstMatch(in string: String, pattern: String) -> String? {
  let regex: NSRegularExpression
  do {
    regex = try NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
  } catch {
    FileHandle.standardError.write(
      Data(
        "manifold-tools-llama: WARNING — invalid quant-label regex '\(pattern)': \(error)\n".utf8))
    return nil
  }
  let range = NSRange(string.startIndex..., in: string)
  guard let match = regex.firstMatch(in: string, options: [], range: range),
    let matchRange = Range(match.range, in: string)
  else {
    return nil
  }
  return String(string[matchRange])
}

/// The caller-declared provenance stamped on every emitted record.
///
/// `renderer` is `"jinja-prompt"` — llama.cpp renders the model's own embedded
/// chat template through `JinjaPromptRenderer`/`PromptRenderer` (see the
/// production-load-path comment in `runCLI` below), the same label
/// `ConformanceRecord.renderer`'s doc comment names as an example. `coreCommit`
/// prefers `$MANIFOLD_CORE_COMMIT` — the same env var `manifold-tools score
/// --core-commit` defaults from in ManifoldKit core — so a sweep that exports it
/// gets a real, comparable value instead of the hardcoded `"unknown"`
/// `manifold-tools-mlx` currently ships (a known limitation flagged in
/// ManifoldKit's `scripts/local-integration-sweep.sh`).
func recordContext(transcriptRef: URL) -> ConformanceScorer.RecordContext {
  ConformanceScorer.RecordContext(
    renderer: "jinja-prompt",
    coreCommit: ProcessInfo.processInfo.environment["MANIFOLD_CORE_COMMIT"] ?? "unknown",
    transcriptRef: transcriptRef.path
  )
}

/// One `ExpectedCell` per scenario, all naming the same (backend, model, quant)
/// cell — the coordinates an absence hole must carry so a downstream matrix can
/// attribute it to the right row instead of silently having no row at all.
func expectedCells(for scenarios: [Scenario], model: String, quant: String) -> [ConformanceScorer
  .ExpectedCell]
{
  scenarios.map {
    ConformanceScorer.ExpectedCell(
      backend: "llama.cpp", model: model, quant: quant, scenario: $0.id)
  }
}

/// Resolves the scenario list to attribute absence records to, best-effort: an
/// invalid `--scenario` filter here just yields no absence records rather than
/// aborting — the real scenario-filter failure path (`ScenarioCLIHarness.filterScenarios`
/// in the main run) still surfaces the actual error to the operator whenever the
/// run gets far enough to reach it. Used only by the two early-return absence
/// branches below, which run before the main flow's own `filtered` is computed.
func resolveScenariosForAbsence(_ scenarios: [Scenario], filter: String) -> [Scenario] {
  do {
    return try ScenarioCLIHarness.filterScenarios(scenarios, matching: filter)
  } catch {
    return []
  }
}

/// Warns on stderr when `--emit-records` was requested but the CLI is about
/// to take an early-return path (`--list`, `--describe`, `--bench`, a missing
/// `--model`) that never runs the scenario harness — so `--emit-records`
/// would otherwise silently no-op with no file and no explanation (rev-183
/// finding on #183). A no-op here is correct behavior (none of these modes
/// have a cell to report on), but it must not be a SILENT no-op.
func warnEmitRecordsIgnored(_ emitRecords: URL?, becauseOf reason: String) {
  guard let emitRecords else { return }
  FileHandle.standardError.write(
    Data(
      "manifold-tools-llama: WARNING — --emit-records \(emitRecords.path) ignored: \(reason) never runs the scenario harness, so there is no cell to report on\n"
        .utf8))
}

/// Writes `records` as the `[ConformanceRecord]` JSON payload to `url`.
///
/// Record emission is additive instrumentation layered on top of a run whose
/// exit code is already decided by the scenario results (or by the load/absence
/// path that called this) — a write failure here must not change that exit
/// code, so the error is logged (do/catch, not `try?`) rather than propagated.
func writeRecords(_ records: [ConformanceRecord], to url: URL) {
  do {
    let data = try ConformanceScorer.encodeJSON(records)
    try data.write(to: url)
    print("Records written to \(url.path) (\(records.count) record(s))")
  } catch {
    FileHandle.standardError.write(
      Data(
        "manifold-tools-llama: WARNING — failed to write --emit-records payload to \(url.path): \(error)\n"
          .utf8))
  }
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
    warnEmitRecordsIgnored(cli.emitRecords, becauseOf: "--list")
    print("Available scenarios:")
    for s in scenarios {
      print("  \(s.id) — \(s.description)")
    }
    return 0
  }

  guard let modelPath = cli.modelPath else {
    warnEmitRecordsIgnored(cli.emitRecords, becauseOf: "a missing --model")
    FileHandle.standardError.write(
      Data(
        "manifold-tools-llama: --model <path.gguf> is required (use --list to inspect scenarios)\n"
          .utf8))
    return 2
  }
  let modelURL = URL(fileURLWithPath: modelPath)
  guard FileManager.default.fileExists(atPath: modelURL.path) else {
    FileHandle.standardError.write(Data("model file not found: \(modelURL.path)\n".utf8))
    // Absence is not failure (ConformanceRecord.CellStatus doc): a missing
    // GGUF is a `notMeasured` hole for every scenario this invocation would
    // have run, never a measured `fail` verdict.
    if let emitURL = cli.emitRecords {
      let scenariosForHole = resolveScenariosForAbsence(
        scenarios, filter: cli.common.scenarioFilter)
      let cells = expectedCells(
        for: scenariosForHole,
        model: modelURL.lastPathComponent,
        quant: quantLabel(fromFileName: modelURL.lastPathComponent)
      )
      let context = recordContext(transcriptRef: cli.common.output)
      let records = cells.map {
        ConformanceScorer.notMeasuredRecord(
          $0, context: context,
          status: .notMeasured("gguf model file not found at \(modelURL.path)"))
      }
      writeRecords(records, to: emitURL)
    }
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
    warnEmitRecordsIgnored(cli.emitRecords, becauseOf: "--describe")
    return describeModel(modelURL)
  }

  // Benchmark mode short-circuits the scenario harness: it drives LlamaBackend
  // directly to time the cold (first) vs warm (subsequent) generations and the
  // one-time per-process Metal pipeline warm-up between them.
  if cli.bench {
    warnEmitRecordsIgnored(cli.emitRecords, becauseOf: "--bench")
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
    filtered = try ScenarioCLIHarness.filterScenarios(
      scenarios, matching: cli.common.scenarioFilter)
  } catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    return 1
  }

  let fixturesRoot = resolveFixturesRoot(cli.common.fixturesRoot)
  print("Fixtures root: \(fixturesRoot.path)")

  // Stamp backend/model/quant onto every transcript record so
  // `ConformanceScorer.resolve(jsonl:)` — and this run's own --emit-records
  // re-scoring below — group rows by the real cell instead of falling back to
  // "unknown"/"unknown"/"unknown" (the default when a record carries none).
  let modelLabel = modelURL.lastPathComponent
  let quantLabelForRun = quantLabel(fromFileName: modelLabel)
  let logger: TranscriptLogger
  do {
    logger = try TranscriptLogger(
      url: cli.common.output,
      backend: "llama.cpp",
      model: modelLabel,
      quant: quantLabelForRun
    )
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
    print(
      "Decoy tools (--extra-tools \(cli.common.extraTools)): \(decoyNames.joined(separator: ", "))")
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
    FileHandle.standardError.write(
      Data(
        "manifold-tools-llama: WARNING — \(modelURL.lastPathComponent) has no embedded tokenizer.chat_template; tool rendering falls back to the \(modelInfo.detectedPromptTemplate.map(String.init(describing:)) ?? "ChatML") enum\n"
          .utf8))
  }

  do {
    print("Loading model: \(modelURL.path)")
    print(
      "  embedded chat_template: \(modelInfo.chatTemplateRaw != nil ? "present" : "ABSENT")"
        + ", detected template: \(modelInfo.detectedPromptTemplate.map(String.init(describing:)) ?? "nil")"
    )
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
    // Absence is not failure: the model existed but its weights/arch could
    // not be loaded, so every requested scenario is a `loadFail` hole, not a
    // measured `fail` verdict.
    if let emitURL = cli.emitRecords {
      let cells = expectedCells(for: filtered, model: modelLabel, quant: quantLabelForRun)
      let context = recordContext(transcriptRef: logger.destination)
      let records = cells.map {
        ConformanceScorer.notMeasuredRecord($0, context: context, status: .loadFail(detail))
      }
      writeRecords(records, to: emitURL)
    }
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
      // `--extra-tools` decoys are NOT spliced into `requiredTools` here —
      // that was the bug this fix removes (see `ScenarioRunner`'s own
      // `passAllRegisteredTools` mode below, which advertises decoys
      // without inflating the scored expected-tool set; the sibling
      // `manifold-tools-mlx` harness's doc comment calls the old
      // requiredTools-splicing approach this replaced "the old
      // requiredTools-patching hack"). The templateless tool-format
      // instruction is supplied upstream by 0.58's `ToolSystemPromptBuilder`
      // fold (MK#2002), so no per-scenario system prompt injection happens
      // here for tool *format*.
      //
      // Tool-*result grounding* (lever 1 of #100) IS injected here: the
      // system prompt is in force on the second turn (after a tool returns)
      // where the soak found llama/gemma narrate instead of grounding, and
      // the harness cannot reach the orchestrator's second-turn instruction.
      scenario = try groundScenario(baseScenario)
    } catch {
      allPassed = false
      print("\n── \(baseScenario.id) — ERROR preparing scenario: \(error)")
      // A preparation failure (groundScenario) means this scenario never
      // even reaches the point of logging a `.prompt` event — without an
      // explicit `.error` here it vanishes from the transcript entirely,
      // so `ConformanceScorer.resolve` never creates a group for it at
      // all: a hole that isn't even a hole (rev-183 finding on #183).
      // `TranscriptLogger.append` stamps this CLI's backend/model/quant
      // onto the event, so a group forms from this one line, `errored`
      // maps it to a `loadFail` record.
      logger.append(
        .error(scenarioId: baseScenario.id, message: "scenario preparation failed: \(error)"))
      continue
    }
    print("\n── \(scenario.id) (\(scenario.description)) ──")
    print("  required tool(s): \(scenario.requiredTools.joined(separator: ", "))")
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
        // definitions (#1983/#1985). `passAllRegisteredTools` advertises
        // every registered tool (required + decoys) when decoys are
        // present — the harness-native way to expose distractors without
        // touching `scenario.requiredTools` (the scored expected-tool
        // set), mirroring `manifold-tools-mlx`'s identical flag. Without
        // it, the runner filters the registry to `scenario.requiredTools`
        // and no decoy is ever advertised.
        let runner = ScenarioRunner(
          service: service,
          logger: logger,
          passAllRegisteredTools: cli.common.extraTools > 0
        )
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
      // BLOCKER 1 (rev-183 on #183): a scenario that streams at least one
      // token/tool_call and THEN throws (mid-generation decode error,
      // context overflow, ...) leaves `producedModelTurn == true` and
      // `errored == false` with zero assertions in the transcript —
      // ConformanceScorer.resolve's verdict(passed:0, failed:0,
      // errored:false) resolves to `.fail`, so the record emitter (which
      // only routes to a notMeasured/loadFail hole when `row.errored ||
      // !row.producedModelTurn`) would emit this as a FABRICATED
      // `.measured` record with `verdict: .fail` — a cell that was never
      // actually measured. Logging the `.error` event here — the same
      // event `TranscriptLogger`/`ConformanceScorer` already define for
      // exactly this ("so the scorer can positively distinguish an infra
      // failure from a model that ran and declined to call a tool") —
      // marks the group `errored`, which routes it to `loadFail` instead.
      logger.append(.error(scenarioId: scenario.id, message: "scenario run failed: \(error)"))
    }
  }

  // Re-score the just-written transcript into ConformanceRecord[] for
  // --emit-records. Reuses ConformanceScorer.records(fileAt:context:) — the
  // same reduction `manifold-tools score --emit-records` and
  // `manifold-tools-mlx --emit-records` apply — so per-scenario absence
  // (a scenario whose run errored before producing a transcript group) also
  // comes back as a notMeasured/loadFail hole automatically, not just the
  // whole-invocation absence cases handled above.
  if let emitURL = cli.emitRecords {
    let context = recordContext(transcriptRef: logger.destination)
    let records = ConformanceScorer.records(fileAt: logger.destination, context: context)
    writeRecords(records, to: emitURL)
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
