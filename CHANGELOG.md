# Changelog

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
