# ``ManifoldLlama``

The llama.cpp / GGUF backend family for ManifoldKit — on-device text
inference, embeddings, and reranking from quantized `.gguf` models, powered
by [llama.cpp](https://github.com/ggml-org/llama.cpp) via the
`mattt/llama.swift` xcframework.

## Overview

`ManifoldLlama` is one of ManifoldKit's companion backend packages. It plugs
into the ``InferenceService`` request queue through the ``LlamaBackends``
registrar and adds three capabilities backed by the same loaded GGUF weights:

- **Text inference** — ``LlamaBackend`` runs GGUF-format language models on
  CPU/GPU through the llama.cpp C API, streaming ``GenerationEvent`` values
  through the standard `InferenceBackend` protocol. Prompt formatting is
  handled by ``InferenceService`` from the detected prompt template.
- **Embeddings** — ``LlamaEmbeddingBackend`` produces pooled embeddings from
  an embedding GGUF (e.g. `nomic-embed`, `all-MiniLM`) for RAG indexing and
  semantic search, conforming to `EmbeddingBackend`.
- **Reranking** — ``LlamaReranker`` scores query/candidate pairs with a
  cross-encoder rank head, squashing the raw logit to a relevance probability
  in `(0, 1)` so scores are comparable across calls and safe to cite.

The package lives outside the main ManifoldKit repository because of its heavy
native dependency (the pre-built ~100 MB llama.cpp xcframework). It depends on
`ManifoldInference` from the ManifoldKit package and is opt-in — you only link
llama.cpp if you add this package.

## Getting started

Add the companion package alongside ManifoldKit, then pass the
``LlamaBackends`` registrar to `quickStart`. That registers the GGUF text
backend with the shared ``InferenceService`` so any `.gguf` model can load.

```swift,no-build
// Package.swift
.package(url: "https://github.com/roryford/manifold-llama", from: "0.1.0"),
```

```swift,no-build
import ManifoldKit
import ManifoldLlama

let kit = try await ManifoldKit.quickStart(backends: [LlamaBackends.self])
```

The embedding and reranking backends are constructed directly against a loaded
GGUF — see ``LlamaEmbeddingBackend`` and ``LlamaReranker`` for the RAG wiring.

## Topics

### Registering the family

- ``LlamaBackends``

### Text inference

- ``LlamaBackend``

### Embeddings & reranking

- ``LlamaEmbeddingBackend``
- ``LlamaReranker``
