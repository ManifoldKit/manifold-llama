# manifold-llama — Claude Code Instructions

llama.cpp (GGUF) inference backend family for [ManifoldKit](https://github.com/ManifoldKit/ManifoldKit), split out of core in v0.48 (ManifoldKit#1749) so core's `swift build` never drags in the xcframework. Module: `ManifoldLlama`. Core conventions (concurrency, testing philosophy, commit style, PR workflow) live in ManifoldKit's own `CLAUDE.md` and `Tests/README.md` — this file only covers what's specific to this companion.

## Targets

| Target | Role |
|--------|------|
| `ManifoldLlama` | GGUF loading, streaming generation, KV-cache persistence, embeddings, reranking, sampling (grammar/DRY/XTC/Mirostat), GGUF tool-call parsing. Depends on ManifoldKit's `ManifoldInference` + `ManifoldHardware` + `ManifoldContract`, plus `LlamaSwift`. |
| `LlamaSwift` | Thin `@_exported @preconcurrency import llama` shim over the vendored `llama-cpp` binary target. |
| `ManifoldLlamaEvalKit` | `RawRun` JSON record + metadata (prompt SHA-256, quant parse, core-commit resolution) + `EvalRunner` — the raw-prompt eval core, unit-testable without a model. |
| `manifold-llama-eval` | Thin CLI over `ManifoldLlamaEvalKit`: loads a GGUF, runs one raw-prompt generation, emits one `RawRun` JSON object to stdout. The llama.cpp leg of manifold-eval's cross-backend differential. |
| `manifold-tools-llama` | Tool-calling scenario CLI. Links ManifoldKit's published `ManifoldTools` product + this repo's `ManifoldLlama`. Ships its own vendored copies of the scenario JSONs and fixture tree (see Vendored data below). The scenario copies predate core 0.62, where `ScenarioLoader.loadBuiltIn()` became `Bundle.module`-based (#2042) — they can now be replaced by `loadBuiltIn()` (planned migration); the fixture copies are still required because `ReadFileTool.defaultRoot()` resolves a source-relative path until core ships a bundled accessor. |

## Testing

```bash
swift build
swift test   # NEVER --parallel
```

`ManifoldBackendTestKit`'s `BackendContractChecks` claims registry is process-global — explicit parallelism interleaves backend test classes and races it (same constraint as core).

Most suites in `Tests/ManifoldLlamaTests` are model-gated: they `XCTSkip` unless a `.gguf` is on disk. Set `LLAMA_TEST_MODEL=<path>` or drop a model under `~/Documents/Models/` to run them for real; otherwise they skip cleanly. `.github/workflows/model-tests.yml` is the nightly lane that provisions a pinned model (Qwen3-0.6B + a MiniLM embedding GGUF) so these guarantees can't silently cover zero behavior for long — it is not a PR gate (too slow/flaky to block merges).

## Pin / release model

This repo pins ManifoldKit with `.upToNextMinor(from: "…")` in `Package.swift`. `.github/workflows/core-bump.yml` listens for ManifoldKit's `core-release` repository_dispatch, rewrites the pin, builds/tests, and admin-merges a `fix:` PR — which trips this repo's own release-please into cutting a patch release. Never hand-edit the pin or hand-tag a release. Conventional Commits are required for release-please to version correctly — unlike core, this repo has no CI job that lints PR titles (the only required check is `test`); self-police the format.

## Vendored data — orphan risk

As of the D1 refactor (MK 0.64+), this repo no longer hand-copies core's full scenario corpus or fixture tree — both are consumed live from the published `ManifoldTools` product (`ScenarioLoader.loadBuiltIn()` / `ToolFixtures.bundledRoot()`, reached through `ScenarioCLIHarness`). The only vendored content left is `Sources/manifold-tools-llama/ScenarioOverrides/*.json` — four scenario ids (`shopping-list-budget`, `parallel-readme-comparison`, `oversize-tool-output`, `structured-json-extraction`) whose assertion wording is deliberately tuned for llama/gemma soak behavior and is *meant* to diverge from core's copy at the same id (spliced in by id in `loadScenarios()`, `Sources/manifold-tools-llama/main.swift`).

`scripts/check-vendored-sync.sh` no longer does content-drift comparison (that would false-positive on every override, since divergence is the point). It instead checks that each override still has a same-named counterpart in core's bundled corpus at the resolved pin (`raw.githubusercontent.com`) — an ORPHAN means core renamed or retired that scenario id and the override silently stopped being spliced in. It hard-fails in CI by default (`--strict`); network failures still exit 0:

```bash
scripts/check-vendored-sync.sh          # strict mode (default), exit 1 on ORPHAN
scripts/check-vendored-sync.sh --warn   # advisory only, always exits 0
```

## Other references

- `docs/LLAMA_CONTRACT.md` — the llama.cpp C-API contract (every `llama_*` symbol called, threading/ordering/ownership rules) and the xcframework upgrade procedure.
- `scripts/repackage-xcframework.sh` — rebuilds the slim (macOS + iOS-only, dSYM-stripped) xcframework from an upstream `ggml-org/llama.cpp` release asset and prints the `url`/`checksum` pair for `Package.swift`.
