# manifold-llama — llama.cpp (GGUF) inference backend for ManifoldKit

llama.cpp (GGUF) inference backend family for [ManifoldKit](https://github.com/ManifoldKit/ManifoldKit), split out of core in v0.48 (ManifoldKit#1749) so core's `swift build` never drags in the xcframework. Module: `ManifoldLlama`. Core conventions (concurrency, testing philosophy, commit style, PR workflow) live in ManifoldKit's own `AGENTS.md`/`CLAUDE.md` and `Tests/README.md` — this file only covers what's specific to this companion.

## Session bootstrap (any harness)

1. Read `~/Repos/roryford/estate/policies/DIGEST.md` — the standing policies,
   compiled to one page.
2. Read this repo's known-issues buffer — solved non-obvious failures.
   **Check it before diagnosing**, and append to it when you solve one.
   Which file:
   - `.agents/known-issues.md` if present; otherwise `.claude/known-issues.md`
     (the older path, still in use in most repos). "Present" means it has
     content — a file that is empty, or only newlines and spaces, does not
     count. (Claude's hook prefers the committed copy on the default branch
     when *choosing* which file to read, then injects the **union** of that
     copy and yours — so an entry you wrote locally and have not pushed is
     normally still shown. Not always: if the union cannot be computed — a
     buffer with no entry markers, an unbalanced code fence, awk missing — the
     hook says so and withholds your local-only entries. A `NOTE` mentioning
     either is the hook refusing to guess, not a bug. Read the file yourself
     and you are never subject to any of this.)
   - **Append to the one that exists; never create the other.** Two files means
     half the entries stop being injected.
   - If **both** exist the buffer is already forked — **report it, don't repair
     it here.** Which half is dark depends on the machine's hook, so the repair
     has a per-machine precondition: see
     `~/Repos/roryford/estate/policies/knowledge-capture.md`, "Both files
     present".
   - If **neither** exists, create `.agents/known-issues.md` — the neutral
     path. (You are reading this section, so you will find it next time
     regardless of what any harness hook does.)
3. Then follow this file's Targets / Build & test / Constraints & gotchas /
   Conventions.

**If you are not Claude Code, none of the above happens for you automatically —
just do steps 1-3.** Claude Code *may* have injected 1 and 2 already via its
own SessionStart hook, but that hook is installed per machine and does not run
for Grok, Codex, Cursor or OpenCode whatever is on the box. Reading the files
yourself is always correct and never wrong, which is why the steps say "read".

Pointers only — this section never restates DIGEST content.

## Targets

| Target | Role |
|--------|------|
| `ManifoldLlama` | GGUF loading, streaming generation, KV-cache persistence, embeddings, reranking, sampling (grammar/DRY/XTC/Mirostat), GGUF tool-call parsing. Depends on ManifoldKit's `ManifoldInference` + `ManifoldHardware` + `ManifoldContract`, plus `LlamaSwift`. |
| `LlamaSwift` | Thin `@_exported @preconcurrency import llama` shim over the vendored `llama-cpp` binary target. |
| `ManifoldLlamaEvalKit` | `RawRun` JSON record + metadata (prompt SHA-256, quant parse, core-commit resolution) + `EvalRunner` — the raw-prompt eval core, unit-testable without a model. |
| `manifold-llama-eval` | Thin CLI over `ManifoldLlamaEvalKit`: loads a GGUF, runs one raw-prompt generation, emits one `RawRun` JSON object to stdout. The llama.cpp leg of manifold-eval's cross-backend differential. |
| `manifold-tools-llama` | Tool-calling scenario CLI. Links ManifoldKit's published `ManifoldTools` product + this repo's `ManifoldLlama`. As of the D1 refactor (MK 0.64+, #130), it consumes core's bundled scenario corpus and fixture tree *live* (`ScenarioLoader.loadBuiltIn()` / `ToolFixtures.bundledRoot()`, via the shared `ScenarioCLIHarness`) — nothing is copied into this repo for those. The only vendored content is `Sources/manifold-tools-llama/ScenarioOverrides/*.json` (see Vendored data below). |

## Build & test

```sh
swift build
swift test   # NEVER --parallel
```

`ManifoldBackendTestKit`'s `BackendContractChecks` claims registry is process-global — explicit parallelism interleaves backend test classes and races it (same constraint as core). CI's `test` job (`.github/workflows/ci.yml`) runs both commands verbatim (plus a toolchain-select step, see below) and is the repo's only required status check.

Most suites in `Tests/ManifoldLlamaTests` are model-gated: they `XCTSkip` unless a `.gguf` is on disk. Set `LLAMA_TEST_MODEL=<path>` or drop a model under `~/Documents/Models/` to run them for real; otherwise they skip cleanly. `.github/workflows/model-tests.yml` is the nightly lane that provisions a pinned model (Qwen3-0.6B + a MiniLM embedding GGUF) so these guarantees can't silently cover zero behavior for long — it is not a PR gate (too slow/flaky to block merges).

## Constraints & gotchas

- **Model-gated suites run for real on any machine with local GGUFs on disk, and some fail there** — e.g. `LlamaGrammarConformanceTests.test_conformance_gemma_carveOut` against a real gemma4 GGUF. CI's default `ci.yml` lane provisions no model, so every model-gated suite `XCTSkip`s there and stays green. A local `swift test` failure in a model-gated suite is almost always a local-baseline issue, not a regression — gate on CI, not local `swift test`.
- The llama.cpp pin is a `binaryTarget` named `llama-cpp` in **this repo's** `Package.swift` (~lines 60-63), pulling a prebuilt xcframework straight from a `ggml-org/llama.cpp` release tag. **ManifoldKit no longer depends on llama.cpp at all** — file llama.cpp version/bump/arch-support issues and GGUF backend docs here, not in ManifoldKit. ManifoldKit issues are only for the `ManifoldContract`/render/parser API surface.
- This repo pins ManifoldKit with `.upToNextMinor(from: "…")` in `Package.swift`. `.github/workflows/core-bump.yml` listens for ManifoldKit's `core-release` repository_dispatch, rewrites the pin, builds/tests, and admin-merges the PR — which trips this repo's own release-please into cutting a patch release. Never hand-edit the pin or hand-tag a release. A pure pin republish is committed as `deps:` (+ a `Release-As:` trailer forcing the patch), so it lands under the CHANGELOG's **Dependencies** heading rather than **Bug Fixes** — it isn't a bug fix and shouldn't be counted as one. The shared workflow falls back to `fix(deps):` when any `feat`/`fix`/breaking change is already queued since the last tag (letting release-please compute the version rather than forcing a patch that could under-version it). The `deps` section is made visible by the explicit `changelog-sections` in `release-please-config.json`; release-please's empty-config default would discard it. Convention owned centrally — see ManifoldKit `AGENTS.md` → "Companion pin-bump releases".
- Conventional Commits are required for release-please to version correctly — unlike core, this repo has no CI job that lints PR titles (the only required check is `test`); self-police the format.
- CI (`ci.yml`) explicitly selects the newest installed Xcode 26 toolchain before building: the runner image default mis-resolves the dependency graph (fails to prune an `AnyLanguageModel`/`mlx-swift-lm` trait-disabled edge), producing a bogus "could not be resolved" conflict. Match that toolchain locally if `swift build`/`swift test` hits a resolution error CI doesn't.
- Publishing (or re-cutting) the vendored xcframework is a `.github/workflows/vendor-release.yml` dispatch (or a `vendor-llama-*` tag push), **not** a manual `gh release create`/`upload`. It's macOS-only and deliberately never runs on PR/push-to-main (10x-billed runner). It refuses to overwrite an already-published `vendor-llama-<BUILD>` asset unless the dispatch's `force` input is set — the repackage is not byte-reproducible, so a silent re-publish would change the bytes behind `Package.swift`'s existing pin. `.github/workflows/vendor-release-rehearsal.yml` (workflow_dispatch only, read-only permissions) rehearses the repackage step for real without publishing or attesting anything. See `docs/LLAMA_CONTRACT.md` → "Signed build-provenance attestation".

## Vendored data — orphan risk

As of the D1 refactor (MK 0.64+, #130), this repo no longer hand-copies core's full scenario corpus or fixture tree — both are consumed live from the published `ManifoldTools` product (`ScenarioLoader.loadBuiltIn()` / `ToolFixtures.bundledRoot()`, reached through `ScenarioCLIHarness`). The only vendored content left is `Sources/manifold-tools-llama/ScenarioOverrides/*.json` — four scenario ids (`shopping-list-budget`, `parallel-readme-comparison`, `oversize-tool-output`, `structured-json-extraction`) whose assertion wording is deliberately tuned for llama/gemma soak behavior and is *meant* to diverge from core's copy at the same id (spliced in by id in `loadScenarios()`, `Sources/manifold-tools-llama/main.swift`).

`scripts/check-vendored-sync.sh` no longer does content-drift comparison (that would false-positive on every override, since divergence is the point). It instead checks that each override's scenario id still has a same-named counterpart in core's bundled corpus at the resolved pin (`raw.githubusercontent.com`) — an ORPHAN means core renamed or retired that scenario id and the override silently stopped being spliced in. It hard-fails in CI by default (`--strict`); network failures still exit 0:

```sh
scripts/check-vendored-sync.sh          # strict mode (default), exit 1 on ORPHAN
scripts/check-vendored-sync.sh --warn   # advisory only, always exits 0
```

## Other references

- `docs/LLAMA_CONTRACT.md` — the llama.cpp C-API contract (every `llama_*` symbol called, threading/ordering/ownership rules) and the xcframework upgrade procedure.
- `scripts/repackage-xcframework.sh` — rebuilds the slim (macOS + iOS-only, dSYM-stripped) xcframework from an upstream `ggml-org/llama.cpp` release asset. A local run is diagnostic-only (see the script's header); the artifact + checksum that actually get pinned come only from a `.github/workflows/vendor-release.yml` run.
- `.github/workflows/vendor-release.yml` / `vendor-release-rehearsal.yml` — publish (with a signed build-provenance attestation) and rehearse, respectively, the vendored xcframework release. See `docs/LLAMA_CONTRACT.md`.

## Conventions

- Estate-wide rules apply (worktrees, secrets via `op run --env-file .env.tpl`,
  conventional commits) — see `~/Repos/estate/estate.yaml` `conventions:`.
