// swift-tools-version: 6.1
import PackageDescription

// NOTE(C2, resolved): the target/product/module carried a temporary `Kit`
// suffix while core still declared a `ManifoldLlama` target (SwiftPM requires
// target names to be unique across the package graph). Core's C2 removal PR
// deletes that target; this branch restores the canonical `ManifoldLlama`
// name and merges immediately after core's C2.
let package = Package(
    name: "manifold-llama",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
    ],
    products: [
        .library(name: "ManifoldLlama", targets: ["ManifoldLlama"]),
        // Real-hardware tool-calling validation CLI: runs ManifoldKit's bundled
        // tool-calling scenarios against a real llama.cpp / GGUF model. Reuses
        // the published `ManifoldTools` library product from ManifoldKit — no
        // changes to core are needed.
        .executable(name: "manifold-tools-llama", targets: ["manifold-tools-llama"]),
        // Raw-prompt eval runner: loads a GGUF, runs ONE raw-prompt generation
        // through LlamaBackend (no chat template), and emits one `RawRun` JSON
        // object on stdout — the llama.cpp leg of the manifold-eval same-GGUF
        // cross-backend differential (vs Ollama). See
        // ManifoldKit docs/plans/manifold-eval-repo-v2-override.md §13b.
        .executable(name: "manifold-llama-eval", targets: ["manifold-llama-eval"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ManifoldKit/ManifoldKit", .upToNextMinor(from: "0.66.0")),
        // swift-jinja (test-only): lets the gemma-4 render-fixture tests render the
        // vendored `tokenizer.chat_template` string directly — `PromptRenderer` /
        // `JinjaPromptRenderer` are `internal` to ManifoldInference and unreachable
        // across the package boundary, so the model-free render assertions drive the
        // real Jinja engine the same way the production renderer does (issue #45).
        // Pinned to the same major swift-jinja ManifoldKit consumes.
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.0.0"),
    ],
    targets: [
        // llama.cpp (GGUF) inference, generation driver, process-lifecycle
        // refcount, embedding backend, GGUF-specific tool call parser,
        // tokenizer adapters. Imported from ManifoldKit/ManifoldKit (see the
        // Imported-From commit trailer); the `#if Llama` / `#if HuggingFace`
        // trait gates were stripped at import — both are always-on here.
        .target(
            name: "ManifoldLlama",
            dependencies: [
                .product(name: "ManifoldInference", package: "ManifoldKit"),
                // ManifoldHardware provides BackendCapabilities, GGUFParser, and
                // device-capability types consumed by LlamaGenerationDriver —
                // a direct import, not via core's @_exported chain (which P7
                // retires).
                .product(name: "ManifoldHardware", package: "ManifoldKit"),
                // @_spi(BackendInternals) import in LlamaBackend.swift requires
                // the direct product edge.
                .product(name: "ManifoldContract", package: "ManifoldKit"),
                "LlamaSwift",
            ],
            path: "Sources/ManifoldLlama"
        ),
        // llama.cpp consumed as a self-hosted *slim* repackage of the upstream
        // ggml-org b9859 release asset (dSYMs stripped, and the unused
        // tvOS/visionOS slices dropped — only the macOS + iOS-device +
        // iOS-simulator slices this package declares remain). Produced once by
        // scripts/repackage-xcframework.sh and hosted as the manifold-llama
        // `vendor-llama-b9859` release asset; this cut the pinned artifact from
        // ~257 MB to ~11 MB (769 MB → 30 MB extracted). No third-party wrapper,
        // no git-tag resolution: `url` + `checksum` still pin the exact
        // pre-built xcframework deterministically, so the "unable to read tree"
        // CI flake that floating wrapper tags caused cannot occur. Bump the
        // build (and re-verify the C API contract) per docs/LLAMA_CONTRACT.md's
        // upgrade procedure, which covers re-running the repackage script and
        // cutting a new vendor-llama-<build> release.
        .binaryTarget(
            name: "llama-cpp",
            url: "https://github.com/ManifoldKit/manifold-llama/releases/download/vendor-llama-b9859/llama-b9859-slim.xcframework.zip",
            checksum: "88e382d47d12e41c786fcf36d5c829c24004d0ee7e3a11483bf4fb63e1d7b190"
        ),
        // Thin re-export shim: `@_exported @preconcurrency import llama` so the
        // ManifoldLlama sources keep importing `LlamaSwift` unchanged, and the
        // `@preconcurrency` keeps the C symbols quiet under Swift 6 strict
        // concurrency.
        .target(
            name: "LlamaSwift",
            dependencies: ["llama-cpp"],
            path: "Sources/LlamaSwift"
        ),
        // Tool-calling scenario CLI. Links the published `ManifoldTools`
        // library product (bundled scenarios + fixture tree + reference tools
        // + `ScenarioCLIHarness` travel with it, MK 0.64+) plus this package's
        // `ManifoldLlama` for the real GGUF backend.
        //
        // The fixture tree the file/dir tools read is NO LONGER vendored here
        // — `ManifoldTools.ToolFixtures.bundledRoot()` (via
        // `ScenarioCLIHarness.resolveFixturesRoot`) resolves it from
        // `ManifoldTools`'s own resource bundle. Five of this package's nine
        // scenario JSONs are likewise no longer vendored — they're byte-
        // identical to core's bundled `built-in` corpus (verified by content
        // diff) and load via `ScenarioLoader.loadBuiltIn()`. Four scenarios
        // (`shopping-list-budget`, `parallel-readme-comparison`,
        // `oversize-tool-output`, `structured-json-extraction`) carry
        // intentional llama/gemma-tolerant assertion wording that DIFFERS from
        // core's copies — those stay vendored under `ScenarioOverrides/` and
        // are spliced in by id at load time (see `loadScenarios()` in
        // main.swift).
        .executableTarget(
            name: "manifold-tools-llama",
            dependencies: [
                .product(name: "ManifoldTools", package: "ManifoldKit"),
                .product(name: "ManifoldInference", package: "ManifoldKit"),
                // ManifoldModelCatalog (MK 0.59+) provides
                // `ChatTemplateToolDescriptor` — the static, render-honest
                // tool-call dialect/negative-gate descriptor (issue #2005 layer
                // 1). Used by `--describe` to report capability without loading
                // weights. `RenderConsistencyChecker` (layer 2) lives in
                // ManifoldInference (already linked above).
                .product(name: "ManifoldModelCatalog", package: "ManifoldKit"),
                "ManifoldLlama",
            ],
            path: "Sources/manifold-tools-llama",
            // README lives inside the target dir; exclude it so SwiftPM doesn't
            // emit an "unhandled resource" warning for it.
            exclude: ["README.md"],
            resources: [
                // Only the four scenarios whose assertions genuinely diverge
                // from core's bundled corpus are vendored — see the target
                // comment above.
                .copy("ScenarioOverrides"),
            ]
        ),
        // Raw-prompt eval runner core: the `RawRun` wire record + metadata
        // helpers (prompt SHA-256, quant parse, core-commit resolution) + the
        // `EvalRunner` that drives one raw-prompt generation through
        // `LlamaBackend`. Split out of the executable so the pure JSON-encoding
        // and metadata logic is unit-testable in CI without a model.
        .target(
            name: "ManifoldLlamaEvalKit",
            dependencies: [
                "ManifoldLlama",
                .product(name: "ManifoldInference", package: "ManifoldKit"),
                .product(name: "ManifoldContract", package: "ManifoldKit"),
            ],
            path: "Sources/ManifoldLlamaEvalKit"
        ),
        // Thin CLI over `ManifoldLlamaEvalKit`: arg parsing + one RawRun to stdout.
        .executableTarget(
            name: "manifold-llama-eval",
            dependencies: ["ManifoldLlamaEvalKit"],
            path: "Sources/manifold-llama-eval"
        ),
        .testTarget(
            name: "ManifoldLlamaTests",
            dependencies: [
                "ManifoldLlama",
                // Raw-prompt eval runner core, exercised by the RawRun JSON
                // unit test (CI) and the model-gated EvalRunner integration test.
                "ManifoldLlamaEvalKit",
                .product(name: "ManifoldInference", package: "ManifoldKit"),
                .product(name: "ManifoldHardware", package: "ManifoldKit"),
                // Test-only: load the vendored tool-calling scenario JSONs via
                // `ScenarioLoader` to pin lever 1's result-grounding prompt (#100).
                .product(name: "ManifoldTools", package: "ManifoldKit"),
                .product(name: "ManifoldRuntime", package: "ManifoldKit"),
                .product(name: "ManifoldPersistenceSwiftData", package: "ManifoldKit"),
                .product(name: "ManifoldTestSupport", package: "ManifoldKit"),
                .product(name: "ManifoldBackendTestKit", package: "ManifoldKit"),
                // ManifoldModelCatalog (MK 0.59+): `ChatTemplateToolDescriptor`
                // exercised by the `--describe` integration test (layers 1+2).
                .product(name: "ManifoldModelCatalog", package: "ManifoldKit"),
                // Test-only: render the vendored gemma-4 chat_template fixture
                // through the real Jinja engine (issue #45 render assertions).
                .product(name: "Jinja", package: "swift-jinja"),
            ]
        ),
    ]
)
