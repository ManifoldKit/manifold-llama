# Changelog

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
