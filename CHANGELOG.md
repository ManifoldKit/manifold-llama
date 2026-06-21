# Changelog

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
