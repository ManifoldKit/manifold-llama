# manifold-tools-llama

A real-hardware tool-calling validation CLI. It runs ManifoldKit's bundled
tool-calling **scenarios** against a real llama.cpp / GGUF model and prints a
PASS/FAIL verdict per assertion.

It reuses the published `ManifoldTools` library product from ManifoldKit — the
bundled scenario corpus, the bundled fixture tree, the six reference tools,
the scenario runner, the JSONL transcript logger, and the shared
`ScenarioCLIHarness` (common flag parsing, bundled-scenario loading,
transcript-summary printing) all come from there (MK 0.64+). Llama-specific
wiring — backend construction and model load, decoy-tool padding, tool-result
grounding, and grammar-constrained final-answer decoding — is layered on top;
**no changes to ManifoldKit core are needed**.

The bundled scenario corpus and fixture tree are consumed *live* from
`ManifoldTools`'s own resource bundle (`ScenarioLoader.loadBuiltIn()` /
`ToolFixtures.bundledRoot()`, reached through `ScenarioCLIHarness`) — nothing
is copied into this repo for those. The one exception is
`Sources/manifold-tools-llama/ScenarioOverrides/` — four scenario ids
(`shopping-list-budget`, `parallel-readme-comparison`, `oversize-tool-output`,
`structured-json-extraction`) whose assertion wording is deliberately tuned
for llama/gemma soak behavior (looser `containsAny`/`containsAll` sets than
core's stricter literal-match wording). `loadScenarios()` in `main.swift`
loads core's full corpus, then splices these four in by id.

> **The four overrides are intentional divergences, not drift to reconcile.**
> `scripts/check-vendored-sync.sh` checks only that each override still
> targets a scenario id core still ships — not content equality (byte
> comparison would always "fail" on these four by design).

## Usage

```sh
swift run manifold-tools-llama --model /path/to/gemma-2-2b-it-Q4_K_M.gguf --scenario all
```

Inspect the available scenarios without a model:

```sh
swift run manifold-tools-llama --list
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--model <path>` | — (required) | Path to the `.gguf` model file. Required except for `--list` / `--help`. |
| `--scenario <id\|all>` | `all` | Scenario id (matches the JSON `id`) or `all`. |
| `--output <path.jsonl>` | `tmp/manifold-tools-llama/<iso>.jsonl` | Transcript JSONL destination. |
| `--fixtures-root <dir>` | bundled fixtures | Override the file/dir tool fixture root. |
| `--list` | — | Print available scenarios and exit (no model needed). |
| `--help` | — | Show usage. |

### Exit codes

- `0` — all scenarios passed.
- `1` — at least one scenario or assertion failed, or a load/setup error.
- `2` — bad arguments.

## Requirements

Running scenarios needs a **real model and real hardware**:

- **Apple Silicon + Metal.** llama.cpp has no Metal support in the iOS
  Simulator and uses a process-global backend init, so this is a
  device/desktop tool, not a CI smoke test.
- A local `.gguf` model (e.g. a gemma GGUF). The model is loaded once and
  reused across every scenario.

Compilation does not need a model — `swift build` and
`swift run manifold-tools-llama --list` work without one.
