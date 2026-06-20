# manifold-tools-llama

A real-hardware tool-calling validation CLI. It runs ManifoldKit's bundled
tool-calling **scenarios** against a real llama.cpp / GGUF model and prints a
PASS/FAIL verdict per assertion.

It reuses the published `ManifoldTools` library product from ManifoldKit — the
scenarios, the six reference tools, the scenario runner, and the JSONL
transcript logger all come from there. The only Llama-specific wiring is
constructing `LlamaBackend` and loading the GGUF; **no changes to ManifoldKit
core are needed**.

Two resources are vendored here as bundled `.copy` resources because the
corresponding ManifoldTools defaults resolve to ManifoldKit source/test paths
that do not travel with the library product:

- **The scenario JSONs** (`Scenarios/`). `ScenarioLoader.loadBuiltIn()`
  resolves `<cwd>/Sources/ManifoldTools/Scenarios/built-in`, which only exists
  when run from the ManifoldKit package root. We ship our own copy and drive
  the public `ScenarioLoader.load(from:)` against the bundled directory.
- **The fixture tree** (`Fixtures/manifold-tools/`) the file/dir tools read.
  `ReadFileTool.defaultRoot()` resolves to a ManifoldKit test path; we pass the
  bundled root (or a `--fixtures-root` override) explicitly.

> **These are vendored copies and can drift from ManifoldKit.** The scenario
> JSONs and fixtures are hand-copied from ManifoldKit's `Sources/ManifoldTools/`
> (no remote fetch — vendoring is deliberate). If ManifoldKit changes the
> bundled scenarios or fixtures, re-copy them here. Do not de-duplicate via a
> network fetch.

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
