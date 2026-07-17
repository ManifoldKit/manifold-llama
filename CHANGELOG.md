# Changelog

## [0.3.4](https://github.com/ManifoldKit/manifold-llama/compare/v0.3.3...v0.3.4) (2026-07-17)


### Bug Fixes

* bump ManifoldKit pin to v0.72.0 ([#150](https://github.com/ManifoldKit/manifold-llama/issues/150)) ([10b6386](https://github.com/ManifoldKit/manifold-llama/commit/10b638641ffd30b97e28cf8602d5246985d30f91))

## [0.3.3](https://github.com/ManifoldKit/manifold-llama/compare/v0.3.2...v0.3.3) (2026-07-13)


### Highlights

#### ManifoldKit 0.71.0

Re-pins the core dependency to [ManifoldKit 0.71.0](https://github.com/ManifoldKit/ManifoldKit/releases/tag/v0.71.0) — the Phase A API-surface tightening (29 core internals demoted to `package`) and the experimental-tier declaration. No llama.cpp-side source changes were needed; the full build+test gate passed against the new core unchanged. See [#148](https://github.com/ManifoldKit/manifold-llama/issues/148).

## [0.3.2](https://github.com/ManifoldKit/manifold-llama/compare/v0.3.1...v0.3.2) (2026-07-11)


### Bug Fixes

* bump ManifoldKit pin to v0.70.0 ([#145](https://github.com/ManifoldKit/manifold-llama/issues/145)) ([e94f71f](https://github.com/ManifoldKit/manifold-llama/commit/e94f71f9b37d284ba91b4804104a260246cf3df1))

## [0.3.1](https://github.com/ManifoldKit/manifold-llama/compare/v0.3.0...v0.3.1) (2026-07-11)


### Bug Fixes

* bump ManifoldKit pin to v0.68.0 ([#138](https://github.com/ManifoldKit/manifold-llama/issues/138)) ([e3e6b08](https://github.com/ManifoldKit/manifold-llama/commit/e3e6b08684e2ef1330d437cd0f0b1fc664794802))
* bump ManifoldKit pin to v0.69.0 (wave-2 ClaimRegistry adapt) ([#140](https://github.com/ManifoldKit/manifold-llama/issues/140)) ([e1744c7](https://github.com/ManifoldKit/manifold-llama/commit/e1744c71bb07ad55e0b0f95b81daf82c9ca3fd9d))
* **driver:** exempt grammar-constrained runs from the phrase-repetition early exit ([#143](https://github.com/ManifoldKit/manifold-llama/issues/143)) ([2d8e385](https://github.com/ManifoldKit/manifold-llama/commit/2d8e3856032b955bfef4a1acdf16683dc7fbc9c8))

## [0.3.0](https://github.com/ManifoldKit/manifold-llama/compare/v0.2.19...v0.3.0) (2026-07-09)


### ⚠ BREAKING CHANGES

* adopt GenerationRuntimeHints (config→hints split, core #2152) ([#136](https://github.com/ManifoldKit/manifold-llama/issues/136))

### Features

* adopt GenerationRuntimeHints (config→hints split, core [#2152](https://github.com/ManifoldKit/manifold-llama/issues/2152)) ([#136](https://github.com/ManifoldKit/manifold-llama/issues/136)) ([2446fda](https://github.com/ManifoldKit/manifold-llama/commit/2446fda4660047630b544ba2e9066a0c17acb2e3))

## [0.2.19](https://github.com/ManifoldKit/manifold-llama/compare/v0.2.18...v0.2.19) (2026-07-07)


### Bug Fixes

* bump ManifoldKit pin to v0.66.0 ([#134](https://github.com/ManifoldKit/manifold-llama/issues/134)) ([7ce3fec](https://github.com/ManifoldKit/manifold-llama/commit/7ce3fec62d91d8cb7ae631a8bd49e4fe9e80ec7a))

## [0.2.18](https://github.com/ManifoldKit/manifold-llama/compare/v0.2.17...v0.2.18) (2026-07-03)

### Highlights

**Tracks ManifoldKit 0.65** ([#131](https://github.com/ManifoldKit/manifold-llama/issues/131)) — the core pin moves to `.upToNextMinor(from: "0.65.0")`, the release that honors advertised structured-output and cache-usage capabilities on cloud backends and removes dead public surface flagged by the inert-code audit. Re-resolved, built, and tested green against the new core.

**Refreshed vendored llama.cpp** ([#127](https://github.com/ManifoldKit/manifold-llama/issues/127)) — the prebuilt llama.cpp xcframework pin advances to build b9859.

### Bug Fixes

* Bump ManifoldKit pin to v0.65.0 ([#131](https://github.com/ManifoldKit/manifold-llama/issues/131))
* **deps:** bump vendored llama.cpp xcframework pin to b9859 ([#127](https://github.com/ManifoldKit/manifold-llama/issues/127))

## [0.2.17](https://github.com/ManifoldKit/manifold-llama/compare/v0.2.16...v0.2.17) (2026-07-02)

### Highlights

**manifold-eval differential leg: raw-prompt eval runner** ([#121](https://github.com/ManifoldKit/manifold-llama/issues/121)) — new `manifold-llama-eval` CLI loads a GGUF, runs one raw-prompt generation through `LlamaBackend` (no chat template), and emits a single `RawRun` JSON object — the llama.cpp leg of the manifold-eval same-GGUF cross-backend differential against Ollama. `--top-k`/`--repeat-penalty` are now overridable on the CLI ([#122](https://github.com/ManifoldKit/manifold-llama/issues/122)) so the runner can force-match samplers across legs.

**Tracks ManifoldKit 0.64** ([#125](https://github.com/ManifoldKit/manifold-llama/issues/125)) — the core pin moves to `.upToNextMinor(from: "0.64.0")`, the release that ships the sticky "approve for the run" tool-approval policy on `UIToolApprovalGate` and the public `ToolFixtures`/`VLModelDetector`/`ScenarioCLIHarness` conformance-harness surface. Re-resolved, built, and tested green against the new core.

### Features

* **eval:** add `manifold-llama-eval`, a raw-prompt eval runner emitting `RawRun` for the manifold-eval differential leg ([#121](https://github.com/ManifoldKit/manifold-llama/issues/121))
* **eval:** make `--top-k`/`--repeat-penalty` overridable in the eval runner ([#122](https://github.com/ManifoldKit/manifold-llama/issues/122))

### Bug Fixes

* **deps:** bump ManifoldKit pin to v0.64.0 ([#125](https://github.com/ManifoldKit/manifold-llama/issues/125))

## [0.2.16](https://github.com/ManifoldKit/manifold-llama/compare/v0.2.15...v0.2.16) (2026-06-28)

### Highlights

**Gemma-4 tool calls now terminate at the right delimiter (0/25 → 16/25 BFCL AST).** The Gemma-4 family's tool-call close delimiter was wrong, so generated calls ran past their boundary and failed to parse — scoring 0/25 on the BFCL AST track. Correcting the delimiter lifts Gemma-4 to 16/25, restoring native tool calling for the family ([#116](https://github.com/ManifoldKit/manifold-llama/issues/116)). Text-only Gemma-4 GGUFs also load now, after dropping a stale architecture-denylist entry that had been rejecting them ([#115](https://github.com/ManifoldKit/manifold-llama/issues/115)).

**Tracks ManifoldKit 0.63** ([#120](https://github.com/ManifoldKit/manifold-llama/issues/120)) — the core pin moves to `.upToNextMinor(from: "0.63.0")`, the release that ships the on-device `Score`/`EvalScorer` eval surface, the `ManifoldTelemetryOTLP` OTLP/HTTP span exporter, and AGENTS.md ambient-instruction skills support. Re-resolved, built, and tested green against the new core.

### Features

* **reranker:** adopt `CancellableModelLoading` on `LlamaReranker`, so a host can observe, cooperatively cancel, and await the true completion of an in-flight reranker model load ([#113](https://github.com/ManifoldKit/manifold-llama/issues/113), [#118](https://github.com/ManifoldKit/manifold-llama/issues/118))

### Bug Fixes

* **tools:** correct the Gemma-4 tool-call close delimiter — 0/25 → 16/25 BFCL AST ([#116](https://github.com/ManifoldKit/manifold-llama/issues/116))
* **llama:** load text-only Gemma-4 GGUFs by dropping a stale arch-denylist entry ([#115](https://github.com/ManifoldKit/manifold-llama/issues/115))
* **deps:** bump ManifoldKit pin to v0.63.0 ([#120](https://github.com/ManifoldKit/manifold-llama/issues/120))

## [0.2.15](https://github.com/roryford/manifold-llama/compare/v0.2.14...v0.2.15) (2026-06-27)

### Highlights

**`LlamaBackend` now conforms to `CancellableModelLoading`** ([#110](https://github.com/roryford/manifold-llama/pull/110), [#2037](https://github.com/roryford/manifold-llama/issues/2037)) — a host can now observe, cooperatively cancel, and await the true completion of an in-flight native model load. `llama_model_load_from_file` ignores Swift `Task` cancellation; when a host's load deadline fires the native call keeps mutating the backend on a background thread, and touching it then SIGSEGVs in `ggml_backend_graph_compute_async`. The new conformance wires a `progress_callback` abort path (`cancelModelLoad()`), a true-completion signal (`awaitModelLoadSettled()` returns only after the C call unwinds), and an `isModelLoadInFlight` flag so hosts can latch precisely instead of guessing. Companion to ManifoldKit [#2054](https://github.com/roryford/ManifoldKit/issues/2054).

**Tracks ManifoldKit 0.62** ([#114](https://github.com/roryford/manifold-llama/pull/114)) — the core pin moves to `.upToNextMinor(from: "0.62.0")`, which is the release that ships `CancellableModelLoading`, the `ConformanceRecord`/`MatrixRenderer` cross-backend scoring APIs, and the `RenderConsistencyChecker` load-time gate.

### Features

* **backend:** adopt CancellableModelLoading on LlamaBackend ([#110](https://github.com/roryford/manifold-llama/pull/110), [#2037](https://github.com/roryford/manifold-llama/issues/2037)) ([8955b73](https://github.com/roryford/manifold-llama/commit/8955b73be567eec8db97e2662455b2dbeb9db9ac))

### Bug Fixes

* **backend:** address code-review findings on CancellableModelLoading ([#110](https://github.com/roryford/manifold-llama/pull/110)) ([f73abfe](https://github.com/roryford/manifold-llama/commit/f73abfe144eb421fb78927517773cc37e99a62e4))
* **deps:** bump ManifoldKit to 0.62.0 ([#114](https://github.com/roryford/manifold-llama/pull/114)) ([22116012](https://github.com/roryford/manifold-llama/commit/22116012))

## [0.2.14](https://github.com/roryford/manifold-llama/compare/v0.2.13...v0.2.14) (2026-06-25)

### Highlights

**Tracks ManifoldKit 0.61** ([#107](https://github.com/roryford/manifold-llama/issues/107)) — the core pin moves to `.upToNextMinor(from: "0.61.0")` to build against the 0.61 release. 0.61 lands the SwiftData-backed `ToolCallConformanceCache` adapter — measured `(model × quant × backend)` tool-call verdicts now persist across launches, wired automatically through `ManifoldBootstrap` — plus tool-calling fixes that fold tool results into the user turn for alternation-strict (Mistral-family) chat templates and adjudicate the Gemma close delimiter to `<|end_of_turn|>`. No source changes required — bump and rebuild.

## [0.2.13](https://github.com/roryford/manifold-llama/compare/v0.2.12...v0.2.13) (2026-06-23)


### Features

* **llama:** surface tool-call dialect on BackendCapabilities ([#104](https://github.com/roryford/manifold-llama/issues/104)) ([be4a72c](https://github.com/roryford/manifold-llama/commit/be4a72c23d9e21f78f8af40d3dd4084c856504b4))
* **tools-cli:** grammar-constrained final-answer decoding for structured-json scenarios ([#100](https://github.com/roryford/manifold-llama/issues/100)) ([#105](https://github.com/roryford/manifold-llama/issues/105)) ([2de6b2d](https://github.com/roryford/manifold-llama/commit/2de6b2d359826f682488a6f9cdc408b6baa153c0))

## [0.2.12](https://github.com/roryford/manifold-llama/compare/v0.2.11...v0.2.12) (2026-06-22)

### Highlights

**Tracks ManifoldKit 0.60** ([#103](https://github.com/roryford/manifold-llama/issues/103), [#96](https://github.com/roryford/manifold-llama/issues/96)) — the core pin moves to `.upToNextMinor(from: "0.60.0")`, jumping past 0.59 to build against the 0.60 release. 0.60 lands the measured tool-call conformance spine — a `ToolCallConformance` cache port, tool-call *dialect* surfaced on `BackendCapabilities`, transcript attribution + a conformance scorer, and a public JSON-Schema → GBNF surface — plus the Mistral renderer fix that folds the system prompt into the first user turn for alternation-strict chat templates. No source changes required — bump and rebuild.

**Tool-call conformance CLI** ([#97](https://github.com/roryford/manifold-llama/issues/97), [#102](https://github.com/roryford/manifold-llama/issues/102), [#99](https://github.com/roryford/manifold-llama/issues/99)) — `manifold-tools-llama` gains `--describe`, which surfaces ManifoldKit 0.59's static tool-call capability claim (`toolsExpressible` + declared dialect) for a GGUF without running inference; a result-grounding prompt (lever 1 of the conformance build-out) that steers weak models to consume a tool result instead of re-calling the tool; and a fix to reset context between scenarios under `--scenario all` so one scenario's history no longer leaks into the next.

**Stability fixes** — honor `flashAttention=false` by mapping to `DISABLED` rather than `AUTO` ([#86](https://github.com/roryford/manifold-llama/issues/86), [#92](https://github.com/roryford/manifold-llama/issues/92)); drain GPU work and clear the KV cache before `llama_free` to avoid a #1394-class SIGABRT in the reranker ([#93](https://github.com/roryford/manifold-llama/issues/93)); make the structured-json extraction assertion format-tolerant ([#95](https://github.com/roryford/manifold-llama/issues/95)); and harden `repackage-xcframework` against silent partial output ([#90](https://github.com/roryford/manifold-llama/issues/90)).

## [0.2.11](https://github.com/roryford/manifold-llama/compare/v0.2.10...v0.2.11) (2026-06-21)


### Features

* **tools:** slim the llama.cpp xcframework artifact (627 MB → 24 MB) ([#87](https://github.com/roryford/manifold-llama/issues/87)) ([bd26b6b](https://github.com/roryford/manifold-llama/commit/bd26b6b54ca8e3957f22fb4cb7f6fbf17141b15b))

## [0.2.10](https://github.com/roryford/manifold-llama/compare/v0.2.9...v0.2.10) (2026-06-21)


### Features

* **tools:** add cold-vs-warm generation benchmark to manifold-tools-llama ([#83](https://github.com/roryford/manifold-llama/issues/83)) ([7cc99d4](https://github.com/roryford/manifold-llama/commit/7cc99d4d1ede50a1164a55f858e83090d5a81b37))

## [0.2.9](https://github.com/roryford/manifold-llama/compare/v0.2.8...v0.2.9) (2026-06-21)


### Bug Fixes

* add llama3.1 bare-JSON tool-call dialect + parameters key alias ([#76](https://github.com/roryford/manifold-llama/issues/76)) ([#77](https://github.com/roryford/manifold-llama/issues/77)) ([bc8874e](https://github.com/roryford/manifold-llama/commit/bc8874e54bd961257881335f54d9956bf52c9bc7))
* bump ManifoldKit pin to v0.58.0 ([#81](https://github.com/roryford/manifold-llama/issues/81)) ([62515cd](https://github.com/roryford/manifold-llama/commit/62515cde3deaf13c2c548f997fb81c5812a938b8))

## [0.2.8](https://github.com/roryford/manifold-llama/compare/v0.2.7...v0.2.8) (2026-06-21)


### Features

* add manifold-tools-llama CLI for running tool-calling scenarios against real GGUF models ([#60](https://github.com/roryford/manifold-llama/issues/60)) ([71089ee](https://github.com/roryford/manifold-llama/commit/71089ee86f7a27e01549d8bb58932bf3e23f04ed))
* add Mistral [TOOL_CALLS] tool-call dialect to GGUF parser ([#70](https://github.com/roryford/manifold-llama/issues/70)) ([#74](https://github.com/roryford/manifold-llama/issues/74)) ([51d1c29](https://github.com/roryford/manifold-llama/commit/51d1c290cdb2777cd15f51d3d882b611f1e5a2a8))


### Bug Fixes

* register only each scenario's requiredTools in manifold-tools-llama (was advertising all 6, overloading small models) ([#66](https://github.com/roryford/manifold-llama/issues/66)) ([baf7d3f](https://github.com/roryford/manifold-llama/commit/baf7d3fbe5b62d4c54518d264b440b127257fbf5))
* render harness prompts with the model's embedded GGUF chat_template ([#69](https://github.com/roryford/manifold-llama/issues/69)) ([#75](https://github.com/roryford/manifold-llama/issues/75)) ([6cf066c](https://github.com/roryford/manifold-llama/commit/6cf066c66e35796c685169019028f8e8a28aba02))
* surface typed error for fused-multimodal gemma4/gemma3n GGUFs ([#62](https://github.com/roryford/manifold-llama/issues/62)) ([#68](https://github.com/roryford/manifold-llama/issues/68)) ([c3629c2](https://github.com/roryford/manifold-llama/commit/c3629c2e01b626b015c0348ce8c9f58ae3d7577f))

## [0.2.7](https://github.com/roryford/manifold-llama/compare/v0.2.6...v0.2.7) (2026-06-20)


### Bug Fixes

* bump ManifoldKit pin to 0.56.0 (lands toolChoice-aware tool-grammar fix, [#55](https://github.com/roryford/manifold-llama/issues/55)) ([#59](https://github.com/roryford/manifold-llama/issues/59)) ([bad6d71](https://github.com/roryford/manifold-llama/commit/bad6d71a013808e58f4139c7c9f600fd88f77a59))
* bump ManifoldKit pin to v0.56.0 ([#57](https://github.com/roryford/manifold-llama/issues/57)) ([e66f2a1](https://github.com/roryford/manifold-llama/commit/e66f2a127f732aa0c085f832d39b9aa7b6680d68))

## [0.2.6](https://github.com/roryford/manifold-llama/compare/v0.2.5...v0.2.6) (2026-06-20)


### Features

* emit .usage(TokenUsage) at end-of-turn for local generation ([#44](https://github.com/roryford/manifold-llama/issues/44)) ([#49](https://github.com/roryford/manifold-llama/issues/49)) ([b928d99](https://github.com/roryford/manifold-llama/commit/b928d9945449e7ab011a8138ea7081372303674d))
* emit prefillProgress events, surface truncated tool calls, claim supportsParallelToolCalls ([#45](https://github.com/roryford/manifold-llama/issues/45) prep) ([#50](https://github.com/roryford/manifold-llama/issues/50)) ([3e98afb](https://github.com/roryford/manifold-llama/commit/3e98afb71c87d48d2ba65e924782ca2eeb9de893))


### Bug Fixes

* bump ManifoldKit pin to v0.55.0 ([#46](https://github.com/roryford/manifold-llama/issues/46)) ([8db0627](https://github.com/roryford/manifold-llama/commit/8db0627c5afcd086e9c281ec5462a33dfbc63543))
* **llama:** drain GPU before freeing embedding context; repair model lane and CI teardown crashes ([#53](https://github.com/roryford/manifold-llama/issues/53)) ([7bcce66](https://github.com/roryford/manifold-llama/commit/7bcce66b8604b2ad21ae80d8c99a21e2162b5c82))

## [0.2.5](https://github.com/roryford/manifold-llama/compare/v0.2.4...v0.2.5) (2026-06-18)

### Highlights

**Tracks ManifoldKit 0.54** ([#41](https://github.com/roryford/manifold-llama/issues/41)) — the core pin moves to `.upToNextMinor(from: "0.54.0")`, building against the 0.54 release. Most relevant here: GGUF models now render their embedded **Jinja chat templates** via swift-jinja instead of a hand-rolled approximation, so prompts match each model family's exact turn formatting. Also in 0.54: a server-side HTTP/SSE transport for the MCP host, and continued pre-1.0 Contract API hardening (backend-neutral `InferenceError.idleTimeout`, the `streamsToolCallArgumentDeltas` capability-alias deprecation, and documented `EmbeddingBackend` guarantees). No source changes required — bump and rebuild.

## [0.2.4](https://github.com/roryford/manifold-llama/compare/v0.2.3...v0.2.4) (2026-06-17)

### Highlights

**Tracks ManifoldKit 0.53** ([#39](https://github.com/roryford/manifold-llama/issues/39)) — the core pin moves to `.upToNextMinor(from: "0.53.0")`, building against the 0.53 release. No source changes required — bump and rebuild.

**llama.cpp now comes straight from upstream** ([#40](https://github.com/roryford/manifold-llama/issues/40)) — the `mattt/llama.swift` wrapper is dropped; the xcframework is pinned directly from the `ggml-org/llama.cpp` releases via a local `.binaryTarget(url:checksum:)` (build b9553) plus a one-line `LlamaSwift` re-export shim. The binary is bit-identical, so `ManifoldLlama` sources are unchanged — but resolution is now deterministic (`url` + `checksum`, no git-tag resolution), removing the wrapper's auto-tag CI-drift hazard. Ships a vendored `docs/vendor/llama.h` and an updated upgrade procedure in the C-API contract doc.

**Weak models now close the C5 grammar envelope** ([#38](https://github.com/roryford/manifold-llama/issues/38), [#20](https://github.com/roryford/manifold-llama/issues/20)) — the C5 tool-call fixture's whitespace rule was unbounded, so small models (mistral 7B) spent the token budget on newline indentation before reaching `</tool_call>`. Bounding `ws` to `{0,4}` keeps output compact enough to close, with a headless tripwire guarding the bound.

**Model-bearing nightly test lane** ([#31](https://github.com/roryford/manifold-llama/issues/31), [#25](https://github.com/roryford/manifold-llama/issues/25)) — a scheduled CI lane runs the hardware-gated real-model suites against on-disk GGUFs nightly, so the skips-empty model tests actually execute on a cadence instead of only locally.

## [0.2.3](https://github.com/roryford/manifold-llama/compare/v0.2.2...v0.2.3) (2026-06-15)

### Highlights

**Tracks ManifoldKit 0.52** ([#17](https://github.com/roryford/manifold-llama/issues/17)) — the core pin moves to `.upToNextMinor(from: "0.52.0")`, building against the 0.52 release (opt-in rendered-prompt observability via `GenerationConfig.captureRenderedPrompt`, batteries-included context-compression policies, idle model auto-unload, and headless model selection). No source changes required — bump and rebuild.

## [0.2.2](https://github.com/roryford/manifold-llama/compare/v0.2.1...v0.2.2) (2026-06-14)

### Highlights

**Tracks ManifoldKit 0.51** ([#12](https://github.com/roryford/manifold-llama/issues/12)) — the core pin moves to `.upToNextMinor(from: "0.51.0")`, building against the 0.51 release (grammar-constrained tool calling derived from `config.tools`, per-tool parameter-schema GBNF lowering, model-capability flags, and the pre-1.0 Contract wire-type freeze). `v0.2.1` still pinned `0.50.0`, which **excludes** 0.51 — so this is the release to take if you're on ManifoldKit 0.51.0. No source changes are required — bump and rebuild.

**Model-family grammar conformance suite** ([#11](https://github.com/roryford/manifold-llama/issues/11)) — a hardware-gated, skips-empty test suite that exercises GBNF grammars across model families (Llama / Qwen / Mistral / Gemma / Phi): digit, JSON-object, alternation, leading-space, and tool-call envelope cases, plus the Gemma grammar carve-out and the thinking-phase grammar gate (ManifoldKit #1595). Test-only — no runtime change.

## [0.2.1](https://github.com/roryford/manifold-llama/compare/v0.2.0...v0.2.1) (2026-06-13)

### Highlights

**Upgrade from 0.2.0 to pick up the ManifoldKit 0.50 core.** `v0.2.0` was pinned to ManifoldKit 0.49.0; this is the recommended successor for anyone on `from: "0.2.0"`. The core pin moves to `.upToNextMinor(from: "0.50.0")` ([#8](https://github.com/roryford/manifold-llama/issues/8)), building against ManifoldKit 0.50 (device-aware model recommender, zero-config NLEmbedding RAG, image-gen preview contract, and more). No source changes are required — bump and rebuild.

**Greedy KV-reuse is now deterministic on every architecture** ([#5](https://github.com/roryford/manifold-llama/issues/5)) — multi-turn greedy decoding with KV-prefix reuse could flip the sampled token on non-Qwen models, because the re-decode used a different batch shape than the first turn and Metal's attention kernels take a different parallel-reduction path per batch size. The re-decode now aligns to an `n_batch` boundary so the sampling chunk is re-run with a byte-identical batch shape — the Metal reduction path matches every turn, restoring determinism while keeping the O(new-tokens) prefix-reuse speedup. Verified on real Apple-Silicon Metal against the non-Qwen `llama3.1-8b` model. Resolves [ManifoldKit#1677](https://github.com/roryford/ManifoldKit/issues/1677).

**Versioning reconciled** ([#9](https://github.com/roryford/manifold-llama/issues/9)) — a manual `v0.2.0` tag had been pushed out-of-band, so release-please (which versions from `.release-please-manifest.json`, not git tags) cut a `v0.1.1` *below* it, leaving the published high tag pointing at older code. The manifest is reset to `0.2.0` so the version line resumes correctly at `0.2.1` and can never regress below `v0.2.0` again.

## [0.1.1](https://github.com/roryford/manifold-llama/compare/v0.1.0...v0.1.1) (2026-06-13)

### Highlights

**Greedy KV-reuse is now deterministic on every architecture** ([#5](https://github.com/roryford/manifold-llama/issues/5)) — multi-turn greedy decoding with KV-prefix reuse could flip the sampled token on non-Qwen models, because the re-decode used a different batch shape than the first turn and Metal's attention kernels take a different parallel-reduction path per batch size. The re-decode now aligns to an `n_batch` boundary so the sampling chunk is re-run with a byte-identical batch shape — the Metal reduction path matches every turn, restoring determinism while keeping the O(new-tokens) prefix-reuse speedup. Verified on real Apple-Silicon Metal against the non-Qwen `llama3.1-8b` model. Resolves [ManifoldKit#1677](https://github.com/roryford/ManifoldKit/issues/1677).

**Tracks ManifoldKit 0.50** ([#8](https://github.com/roryford/manifold-llama/issues/8)) — the core pin moves to `.upToNextMinor(from: "0.50.0")`, building against the 0.50 release (device-aware model recommender, zero-config NLEmbedding RAG, image-gen preview contract, and more).

## 0.1.0 (2026-06-12)

The llama.cpp (GGUF) inference backend now ships as its own package. It was split out of ManifoldKit core in the v0.48 packaging release ([ManifoldKit#1749](https://github.com/roryford/ManifoldKit/issues/1749)) so that `swift build` of core never drags the llama.cpp xcframework — the heavy backend is one `.package` line and one registrar call away.

### Highlights

**`ManifoldLlama` is now a companion package** ([#2](https://github.com/roryford/manifold-llama/issues/2)) — GGUF model loading, streaming generation, KV-cache persistence/reuse, embeddings, reranking, grammar/DRY/XTC/Mirostat sampling, and GGUF tool-call parsing move out of core and plug back in through a single `LlamaBackends` registrar. It wraps llama.cpp via the exact-pinned [`mattt/llama.swift`](https://github.com/mattt/llama.swift) xcframework, behind ManifoldKit's `InferenceBackend` contract. Module names are restored to their canonical form ahead of this first tag — no temporary `Kit`-suffixed targets.

```swift
// Package.swift
.package(url: "https://github.com/roryford/ManifoldKit", from: "0.48.0"),
.package(url: "https://github.com/roryford/manifold-llama", from: "0.1.0"),

// App entry point
import ManifoldKit
import ManifoldLlama

let kit = try await ManifoldKit.quickStart(backends: [LlamaBackends.self])
```

**Pinned to ManifoldKit 0.48.x** — this release tracks core via `.upToNextMinor(from: "0.48.0")` and builds against the post-split core, where the backend seam and registrar surface are frozen and verified by the out-of-package split proof.
