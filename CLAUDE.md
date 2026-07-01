# manifold-llama — Claude Code Instructions

llama.cpp (GGUF) inference backend family for [ManifoldKit](https://github.com/ManifoldKit/ManifoldKit), split out of core in v0.48 (ManifoldKit#1749) so core's `swift build` never drags in the xcframework. Module: `ManifoldLlama`. Core conventions (concurrency, testing philosophy, commit style, PR workflow) live in ManifoldKit's own `CLAUDE.md` and `Tests/README.md` — this file only covers what's specific to this companion.

## Targets

| Target | Role |
|--------|------|
| `ManifoldLlama` | GGUF loading, streaming generation, KV-cache persistence, embeddings, reranking, sampling (grammar/DRY/XTC/Mirostat), GGUF tool-call parsing. Depends on ManifoldKit's `ManifoldInference` + `ManifoldHardware` + `ManifoldContract`, plus `LlamaSwift`. |
| `LlamaSwift` | Thin `@_exported @preconcurrency import llama` shim over the vendored `llama-cpp` binary target. |
| `ManifoldLlamaEvalKit` | `RawRun` JSON record + metadata (prompt SHA-256, quant parse, core-commit resolution) + `EvalRunner` — the raw-prompt eval core, unit-testable without a model. |
| `manifold-llama-eval` | Thin CLI over `ManifoldLlamaEvalKit`: loads a GGUF, runs one raw-prompt generation, emits one `RawRun` JSON object to stdout. The llama.cpp leg of manifold-eval's cross-backend differential. |
| `manifold-tools-llama` | Tool-calling scenario CLI. Links ManifoldKit's published `ManifoldTools` product + this repo's `ManifoldLlama`. Ships its own vendored copies of the scenario JSONs and fixture tree (see Vendored data below) because `ScenarioLoader.loadBuiltIn()` / `ReadFileTool.defaultRoot()` resolve source-relative paths that don't exist in this package. |

## Testing

```bash
swift build
swift test   # NEVER --parallel
```

`ManifoldBackendTestKit`'s `BackendContractChecks` claims registry is process-global — explicit parallelism interleaves backend test classes and races it (same constraint as core).

Most suites in `Tests/ManifoldLlamaTests` are model-gated: they `XCTSkip` unless a `.gguf` is on disk. Set `LLAMA_TEST_MODEL=<path>` or drop a model under `~/Documents/Models/` to run them for real; otherwise they skip cleanly. `.github/workflows/model-tests.yml` is the nightly lane that provisions a pinned model (Qwen3-0.6B + a MiniLM embedding GGUF) so these guarantees can't silently cover zero behavior for long — it is not a PR gate (too slow/flaky to block merges).

## Pin / release model

This repo pins ManifoldKit with `.upToNextMinor(from: "…")` in `Package.swift`. `.github/workflows/core-bump.yml` listens for ManifoldKit's `core-release` repository_dispatch, rewrites the pin, builds/tests, and admin-merges a `fix:` PR — which trips this repo's own release-please into cutting a patch release. Never hand-edit the pin or hand-tag a release. Conventional Commits are required for release-please to version correctly — unlike core, this repo has no CI job that lints PR titles (the only required check is `test`); self-police the format.

## Vendored data — drift risk

`Sources/manifold-tools-llama/Scenarios/*.json` and `Sources/manifold-tools-llama/Fixtures/manifold-tools/**` are hand-copied from ManifoldKit core (`Sources/ManifoldTools/Scenarios/built-in/` and `Tests/Fixtures/manifold-tools/` respectively) and nothing keeps them in sync automatically — core can change a scenario's assertions and this repo's copy silently goes stale.

`scripts/check-vendored-sync.sh` compares this repo's vendored files against core at the tag matching the resolved ManifoldKit pin (via `raw.githubusercontent.com`). It runs warn-only in CI (`continue-on-error: true`); run it locally with `--strict` to fail on real drift:

```bash
scripts/check-vendored-sync.sh          # warn mode, always exits 0
scripts/check-vendored-sync.sh --strict # exit 1 on DRIFT/MISSING-UPSTREAM
```

As of this writing it reports real drift on 4 of 9 scenario files (`06`, `07`, `08`, `09`) — investigate and reconcile by hand; the script only detects, it does not resync.

## Other references

- `docs/LLAMA_CONTRACT.md` — the llama.cpp C-API contract (every `llama_*` symbol called, threading/ordering/ownership rules) and the xcframework upgrade procedure.
- `scripts/repackage-xcframework.sh` — rebuilds the slim (macOS + iOS-only, dSYM-stripped) xcframework from an upstream `ggml-org/llama.cpp` release asset and prints the `url`/`checksum` pair for `Package.swift`.
