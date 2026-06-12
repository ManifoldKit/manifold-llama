# Changelog

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
