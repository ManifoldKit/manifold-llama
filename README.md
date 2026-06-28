# manifold-llama

llama.cpp (GGUF) inference backend for [ManifoldKit](https://github.com/ManifoldKit/ManifoldKit) — the `ManifoldLlama` module, split out of the core package as part of the v0.48 packaging release (ManifoldKit#1749) so that `swift build` of core never drags the llama.cpp xcframework, and heavy backends are one `.package` line away.

It wraps llama.cpp (via the prebuilt xcframework from the upstream [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) releases, pinned by `url` + `checksum` in a local `.binaryTarget`) behind ManifoldKit's `InferenceBackend` contract: GGUF model loading, streaming generation, KV-cache persistence/reuse, embeddings, reranking, grammar/DRY/XTC/Mirostat sampling, and GGUF tool-call parsing.

> **Temporary module name — pre-0.48 only.** Until ManifoldKit's C2 removal PR deletes the in-core `ManifoldLlama` target, SwiftPM's graph-wide target-name uniqueness forces this package to ship the module as **`ManifoldLlama`**. It is renamed to `ManifoldLlama` in one commit before the first `0.1.0` tag (see the `NOTE(C2)` in `Package.swift`). If you are reading this after a 0.1.0 tag exists and still see `Kit` names, file an issue.

## Install

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/ManifoldKit/ManifoldKit", branch: "main"),
    .package(url: "https://github.com/ManifoldKit/manifold-llama", branch: "main"),
],
targets: [
    .target(name: "MyApp", dependencies: [
        .product(name: "ManifoldKit", package: "ManifoldKit"),
        .product(name: "ManifoldLlama", package: "manifold-llama"),
    ]),
]
```

Register the backend via the `LlamaBackends` registrar (the seam shipped in core's B2 work; the registrar moved here in core's C2 split):

```swift
import ManifoldKit
import ManifoldLlama

let kit = try await ManifoldKit.quickStart(backends: [LlamaBackends.self])
```

## Compatibility

| manifold-llama | ManifoldKit |
|---|---|
| `main` | `main` (pre-0.48) |
| `0.1.0` (not yet tagged) | `0.48.x` (`.upToNextMinor` pin) |

Pre-tag, this package tracks core `main` by branch; the pin flips to `.upToNextMinor(from: "0.48.0")` at the 0.48 release train.

## Tests

```bash
swift build
swift test   # no --parallel: the contract-suite claims registry is process-global
```

The suite is the Llama-family subset of core's `ManifoldBackendsTests` plus the shared `ManifoldBackendTestKit` contract/conformance checks. Tests that need a real GGUF model on disk skip themselves on machines without one.

## Provenance & history

Imported as a fresh copy from `ManifoldKit/ManifoldKit` (see the `Imported-From:` trailer on the import commit). **History before 2026-06 lives in [ManifoldKit](https://github.com/ManifoldKit/ManifoldKit)** — `git log` there for the archaeology. The llama.cpp upgrade procedure and C-API contract notes are in [`docs/LLAMA_CONTRACT.md`](docs/LLAMA_CONTRACT.md).

## License

MIT — see [LICENSE](LICENSE).
