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
    ],
    dependencies: [
        // The ManifoldBackendTestKit / ManifoldTestSupport products this package
        // needs exist only on main until the 0.48 tags ship.
        // traits: [] builds core's products trait-less (the post-C2 world).
        .package(url: "https://github.com/roryford/ManifoldKit", .upToNextMinor(from: "0.61.0")),
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
        // tokenizer adapters. Imported from roryford/ManifoldKit (see the
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
        // ggml-org b9744 release asset (dSYMs stripped, and the unused
        // tvOS/visionOS slices dropped — only the macOS + iOS-device +
        // iOS-simulator slices this package declares remain). Produced once by
        // scripts/repackage-xcframework.sh and hosted as the manifold-llama
        // `vendor-llama-b9744` release asset; this cut the pinned artifact from
        // ~208 MB to ~8 MB (627 MB → 24 MB extracted). No third-party wrapper,
        // no git-tag resolution: `url` + `checksum` still pin the exact
        // pre-built xcframework deterministically, so the "unable to read tree"
        // CI flake that floating wrapper tags caused cannot occur. Bump the
        // build (and re-verify the C API contract) per docs/LLAMA_CONTRACT.md's
        // upgrade procedure, which covers re-running the repackage script and
        // cutting a new vendor-llama-<build> release.
        .binaryTarget(
            name: "llama-cpp",
            url: "https://github.com/roryford/manifold-llama/releases/download/vendor-llama-b9744/llama-b9744-slim.xcframework.zip",
            checksum: "ec27ea729522906d06f2618ea30c9cf9427dd45d2552dc4377224486cfd6339c"
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
        // library product (the bundled scenarios + reference tools travel with
        // it) plus this package's `ManifoldLlama` for the real GGUF backend.
        // The fixture tree the file/dir tools read is vendored as a bundled
        // `.copy` resource — `ReadFileTool.defaultRoot()` resolves to a
        // ManifoldKit test path that does not exist here.
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
                .copy("Fixtures/manifold-tools"),
                // The bundled scenario JSONs are vendored here too:
                // `ScenarioLoader.loadBuiltIn()` resolves a ManifoldKit
                // source-relative path (`<cwd>/Sources/ManifoldTools/...`) that
                // does not exist in this package, so we ship our own copy and
                // drive the public `ScenarioLoader.load(from:)` against the
                // bundled directory instead.
                .copy("Scenarios"),
            ]
        ),
        .testTarget(
            name: "ManifoldLlamaTests",
            dependencies: [
                "ManifoldLlama",
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
