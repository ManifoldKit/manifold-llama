# Changelog

## [0.2.2](https://github.com/ManifoldKit/manifold-llama/compare/v0.2.17...v0.2.2) (2026-07-02)


### ⚠ BREAKING CHANGES

* restore canonical module names and adopt the family registrar ([#2](https://github.com/ManifoldKit/manifold-llama/issues/2))

### Features

* add manifold-tools-llama CLI for running tool-calling scenarios against real GGUF models ([#60](https://github.com/ManifoldKit/manifold-llama/issues/60)) ([71089ee](https://github.com/ManifoldKit/manifold-llama/commit/71089ee86f7a27e01549d8bb58932bf3e23f04ed))
* add Mistral [TOOL_CALLS] tool-call dialect to GGUF parser ([#70](https://github.com/ManifoldKit/manifold-llama/issues/70)) ([#74](https://github.com/ManifoldKit/manifold-llama/issues/74)) ([51d1c29](https://github.com/ManifoldKit/manifold-llama/commit/51d1c290cdb2777cd15f51d3d882b611f1e5a2a8))
* **backend:** adopt CancellableModelLoading on LlamaBackend ([6840f2e](https://github.com/ManifoldKit/manifold-llama/commit/6840f2ed2fed4828df908f7186e7be4efa6d569e))
* **backend:** adopt CancellableModelLoading on LlamaBackend ([#2037](https://github.com/ManifoldKit/manifold-llama/issues/2037)) ([8955b73](https://github.com/ManifoldKit/manifold-llama/commit/8955b73be567eec8db97e2662455b2dbeb9db9ac))
* **ci:** add model-bearing nightly test lane ([#25](https://github.com/ManifoldKit/manifold-llama/issues/25)) ([#31](https://github.com/ManifoldKit/manifold-llama/issues/31)) ([8420592](https://github.com/ManifoldKit/manifold-llama/commit/8420592c87ac5335a197f1eeaa98eda498158e91))
* emit .usage(TokenUsage) at end-of-turn for local generation ([#44](https://github.com/ManifoldKit/manifold-llama/issues/44)) ([#49](https://github.com/ManifoldKit/manifold-llama/issues/49)) ([b928d99](https://github.com/ManifoldKit/manifold-llama/commit/b928d9945449e7ab011a8138ea7081372303674d))
* emit prefillProgress events, surface truncated tool calls, claim supportsParallelToolCalls ([#45](https://github.com/ManifoldKit/manifold-llama/issues/45) prep) ([#50](https://github.com/ManifoldKit/manifold-llama/issues/50)) ([3e98afb](https://github.com/ManifoldKit/manifold-llama/commit/3e98afb71c87d48d2ba65e924782ca2eeb9de893))
* import ManifoldLlama from ManifoldKit ([f1e30bc](https://github.com/ManifoldKit/manifold-llama/commit/f1e30bc589ff78ca770d67b5e925ede0863811ce))
* **llama:** surface tool-call dialect on BackendCapabilities ([#104](https://github.com/ManifoldKit/manifold-llama/issues/104)) ([be4a72c](https://github.com/ManifoldKit/manifold-llama/commit/be4a72c23d9e21f78f8af40d3dd4084c856504b4))
* make top-k and repeat-penalty overridable in the eval runner ([#122](https://github.com/ManifoldKit/manifold-llama/issues/122)) ([21cd93b](https://github.com/ManifoldKit/manifold-llama/commit/21cd93b90222f4ac0b13dabe53afa6c86b5c27bf))
* P2.2 raw-prompt eval runner emitting RawRun (manifold-eval differential leg) ([#121](https://github.com/ManifoldKit/manifold-llama/issues/121)) ([bc7ec0b](https://github.com/ManifoldKit/manifold-llama/commit/bc7ec0b6b8069314216b8b3325d48ff98fca8819))
* pin ManifoldKit core to .upToNextMinor(from: 0.48.0) ([#3](https://github.com/ManifoldKit/manifold-llama/issues/3)) ([626bf93](https://github.com/ManifoldKit/manifold-llama/commit/626bf93e74ad27aa0846630f4eee2de5ab20c4db))
* **reranker:** adopt CancellableModelLoading on LlamaReranker ([#113](https://github.com/ManifoldKit/manifold-llama/issues/113)) ([#118](https://github.com/ManifoldKit/manifold-llama/issues/118)) ([a4ee11c](https://github.com/ManifoldKit/manifold-llama/commit/a4ee11cd89c026f06e95ac75e38ff76522386c61))
* restore canonical module names and adopt the family registrar ([#2](https://github.com/ManifoldKit/manifold-llama/issues/2)) ([8592eb8](https://github.com/ManifoldKit/manifold-llama/commit/8592eb8c2844f32223e22e17e620cf25d67b3d91))
* **tools-cli:** add --describe (MK 0.59 static tool-call capability) ([#97](https://github.com/ManifoldKit/manifold-llama/issues/97)) ([ac1bcc8](https://github.com/ManifoldKit/manifold-llama/commit/ac1bcc87eeb4d9440644c3c53ee5d71df6b2c5d8))
* **tools-cli:** grammar-constrained final-answer decoding for structured-json scenarios ([#100](https://github.com/ManifoldKit/manifold-llama/issues/100)) ([#105](https://github.com/ManifoldKit/manifold-llama/issues/105)) ([2de6b2d](https://github.com/ManifoldKit/manifold-llama/commit/2de6b2d359826f682488a6f9cdc408b6baa153c0))
* **tools-cli:** result-grounding prompt (lever 1 of [#100](https://github.com/ManifoldKit/manifold-llama/issues/100)) ([#102](https://github.com/ManifoldKit/manifold-llama/issues/102)) ([1606554](https://github.com/ManifoldKit/manifold-llama/commit/1606554d9d2c1b1effb19be21ca43fdd40dd381e))
* **tools:** add cold-vs-warm generation benchmark to manifold-tools-llama ([#83](https://github.com/ManifoldKit/manifold-llama/issues/83)) ([7cc99d4](https://github.com/ManifoldKit/manifold-llama/commit/7cc99d4d1ede50a1164a55f858e83090d5a81b37))
* **tools:** slim the llama.cpp xcframework artifact (627 MB → 24 MB) ([#87](https://github.com/ManifoldKit/manifold-llama/issues/87)) ([bd26b6b](https://github.com/ManifoldKit/manifold-llama/commit/bd26b6b54ca8e3957f22fb4cb7f6fbf17141b15b))


### Bug Fixes

* add llama3.1 bare-JSON tool-call dialect + parameters key alias ([#76](https://github.com/ManifoldKit/manifold-llama/issues/76)) ([#77](https://github.com/ManifoldKit/manifold-llama/issues/77)) ([bc8874e](https://github.com/ManifoldKit/manifold-llama/commit/bc8874e54bd961257881335f54d9956bf52c9bc7))
* **backend:** address code-review findings on CancellableModelLoading ([#2037](https://github.com/ManifoldKit/manifold-llama/issues/2037)) ([f73abfe](https://github.com/ManifoldKit/manifold-llama/commit/f73abfe144eb421fb78927517773cc37e99a62e4))
* batch-aligned KV-reuse re-decode for greedy determinism (preserves prefix reuse) ([#5](https://github.com/ManifoldKit/manifold-llama/issues/5)) ([99d61e0](https://github.com/ManifoldKit/manifold-llama/commit/99d61e09325af500729bfd0725b6ba2c9f096267))
* bump ManifoldKit pin to 0.50.0 ([#8](https://github.com/ManifoldKit/manifold-llama/issues/8)) ([2365eaa](https://github.com/ManifoldKit/manifold-llama/commit/2365eaa9c80410774c224ff1a383f53b86f65b18))
* bump ManifoldKit pin to 0.56.0 (lands toolChoice-aware tool-grammar fix, [#55](https://github.com/ManifoldKit/manifold-llama/issues/55)) ([#59](https://github.com/ManifoldKit/manifold-llama/issues/59)) ([bad6d71](https://github.com/ManifoldKit/manifold-llama/commit/bad6d71a013808e58f4139c7c9f600fd88f77a59))
* bump ManifoldKit pin to v0.52.0 ([#17](https://github.com/ManifoldKit/manifold-llama/issues/17)) ([2d84ca9](https://github.com/ManifoldKit/manifold-llama/commit/2d84ca92517ac5c1f99c612884c7e369b352da9c))
* bump ManifoldKit pin to v0.53.0 ([#39](https://github.com/ManifoldKit/manifold-llama/issues/39)) ([e7295f9](https://github.com/ManifoldKit/manifold-llama/commit/e7295f9230f4efefb0145620ff2b3d51062634b8))
* bump ManifoldKit pin to v0.54.0 ([#41](https://github.com/ManifoldKit/manifold-llama/issues/41)) ([d5b3bbf](https://github.com/ManifoldKit/manifold-llama/commit/d5b3bbff6d46dada0acb83bc80c48c1068b78250))
* bump ManifoldKit pin to v0.55.0 ([#46](https://github.com/ManifoldKit/manifold-llama/issues/46)) ([8db0627](https://github.com/ManifoldKit/manifold-llama/commit/8db0627c5afcd086e9c281ec5462a33dfbc63543))
* bump ManifoldKit pin to v0.56.0 ([#57](https://github.com/ManifoldKit/manifold-llama/issues/57)) ([e66f2a1](https://github.com/ManifoldKit/manifold-llama/commit/e66f2a127f732aa0c085f832d39b9aa7b6680d68))
* bump ManifoldKit pin to v0.58.0 ([#81](https://github.com/ManifoldKit/manifold-llama/issues/81)) ([62515cd](https://github.com/ManifoldKit/manifold-llama/commit/62515cde3deaf13c2c548f997fb81c5812a938b8))
* bump ManifoldKit pin to v0.59.0 ([#96](https://github.com/ManifoldKit/manifold-llama/issues/96)) ([a455702](https://github.com/ManifoldKit/manifold-llama/commit/a45570260379071a89b8acb92bbf83cae1970e1e))
* bump ManifoldKit pin to v0.60.0 ([#103](https://github.com/ManifoldKit/manifold-llama/issues/103)) ([6e6c8aa](https://github.com/ManifoldKit/manifold-llama/commit/6e6c8aa446d8cf6cd1f7774b5d22f05074adca9c))
* bump ManifoldKit pin to v0.61.0 ([#107](https://github.com/ManifoldKit/manifold-llama/issues/107)) ([863fce6](https://github.com/ManifoldKit/manifold-llama/commit/863fce64ee7bf48f34acf92e89e50c9cc5c82196))
* bump ManifoldKit pin to v0.63.0 ([#120](https://github.com/ManifoldKit/manifold-llama/issues/120)) ([b97f2ff](https://github.com/ManifoldKit/manifold-llama/commit/b97f2ff94fa621f8f2c6ea55f1bf157610284aff))
* consume llama.cpp xcframework directly from upstream ggml-org ([#40](https://github.com/ManifoldKit/manifold-llama/issues/40)) ([fe48c9b](https://github.com/ManifoldKit/manifold-llama/commit/fe48c9b80754d9ecfa36d5ae02b6aa2c73bd9b07))
* **deps:** bump ManifoldKit pin to v0.64.0 ([#125](https://github.com/ManifoldKit/manifold-llama/issues/125)) ([3c21499](https://github.com/ManifoldKit/manifold-llama/commit/3c21499db44802524799eb36e946918f350ba7c0))
* **deps:** bump vendored llama.cpp xcframework pin to b9859 ([#127](https://github.com/ManifoldKit/manifold-llama/issues/127)) ([6ff3f3f](https://github.com/ManifoldKit/manifold-llama/commit/6ff3f3f137f6fea3801e10767daf9d1c80d6831c))
* **llama:** bound C5 conformance whitespace so weak models close the envelope ([#20](https://github.com/ManifoldKit/manifold-llama/issues/20)) ([#38](https://github.com/ManifoldKit/manifold-llama/issues/38)) ([55341d5](https://github.com/ManifoldKit/manifold-llama/commit/55341d505f9595bf0a245dd38cf9b673a4f0c32f))
* **llama:** drain GPU before freeing embedding context; repair model lane and CI teardown crashes ([#53](https://github.com/ManifoldKit/manifold-llama/issues/53)) ([7bcce66](https://github.com/ManifoldKit/manifold-llama/commit/7bcce66b8604b2ad21ae80d8c99a21e2162b5c82))
* **llama:** load text-only gemma4 GGUFs by dropping arch denylist entry ([#115](https://github.com/ManifoldKit/manifold-llama/issues/115)) ([659f20f](https://github.com/ManifoldKit/manifold-llama/commit/659f20f2c1d33f9e396c2112faac9942c9d15da0))
* **loader:** honor flashAttention=false (map to DISABLED, not AUTO) ([#86](https://github.com/ManifoldKit/manifold-llama/issues/86)) ([#92](https://github.com/ManifoldKit/manifold-llama/issues/92)) ([7345f7a](https://github.com/ManifoldKit/manifold-llama/commit/7345f7a44d40ae6c0f12256346cc53ed6f034861))
* reconcile release-please manifest to 0.2.0 (was regressed to 0.1.1) ([#9](https://github.com/ManifoldKit/manifold-llama/issues/9)) ([af9a110](https://github.com/ManifoldKit/manifold-llama/commit/af9a11081f31c998ef00c152a9b25822423cb200))
* register only each scenario's requiredTools in manifold-tools-llama (was advertising all 6, overloading small models) ([#66](https://github.com/ManifoldKit/manifold-llama/issues/66)) ([baf7d3f](https://github.com/ManifoldKit/manifold-llama/commit/baf7d3fbe5b62d4c54518d264b440b127257fbf5))
* render harness prompts with the model's embedded GGUF chat_template ([#69](https://github.com/ManifoldKit/manifold-llama/issues/69)) ([#75](https://github.com/ManifoldKit/manifold-llama/issues/75)) ([6cf066c](https://github.com/ManifoldKit/manifold-llama/commit/6cf066c66e35796c685169019028f8e8a28aba02))
* **reranker:** drain GPU work + clear KV before llama_free to avoid [#1394](https://github.com/ManifoldKit/manifold-llama/issues/1394) SIGABRT ([#93](https://github.com/ManifoldKit/manifold-llama/issues/93)) ([216b9a0](https://github.com/ManifoldKit/manifold-llama/commit/216b9a0b280cf66e30245ef4ab5418a9447e2702))
* **scenarios:** make structured-json extraction assertion format-tolerant ([#95](https://github.com/ManifoldKit/manifold-llama/issues/95)) ([8da5e35](https://github.com/ManifoldKit/manifold-llama/commit/8da5e35b8fa8b577869c08cbe3406195b7ebd4a8))
* **scripts:** harden repackage-xcframework against silent partial output ([#90](https://github.com/ManifoldKit/manifold-llama/issues/90)) ([d60a1f6](https://github.com/ManifoldKit/manifold-llama/commit/d60a1f67af7704ca40ebeeb9c0ca32dfc7eb7e06))
* surface typed error for fused-multimodal gemma4/gemma3n GGUFs ([#62](https://github.com/ManifoldKit/manifold-llama/issues/62)) ([#68](https://github.com/ManifoldKit/manifold-llama/issues/68)) ([c3629c2](https://github.com/ManifoldKit/manifold-llama/commit/c3629c2e01b626b015c0348ce8c9f58ae3d7577f))
* **tools-cli:** reset context between scenarios in --scenario all ([#99](https://github.com/ManifoldKit/manifold-llama/issues/99)) ([#101](https://github.com/ManifoldKit/manifold-llama/issues/101)) ([c2cfadb](https://github.com/ManifoldKit/manifold-llama/commit/c2cfadb2ec07aae2df708ac5d7aabbf561256789))
* **tools:** correct gemma-4 tool-call close delimiter (0/25 → 16/25 AST) ([#116](https://github.com/ManifoldKit/manifold-llama/issues/116)) ([ee383ea](https://github.com/ManifoldKit/manifold-llama/commit/ee383ea37e92ec76070cade8da5ee8266eafbe5a))


### Miscellaneous Chores

* release 0.2.2 ([#13](https://github.com/ManifoldKit/manifold-llama/issues/13)) ([ee70051](https://github.com/ManifoldKit/manifold-llama/commit/ee7005132a857c12c2cd14a2beeb3b1ad277fbcf))

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
