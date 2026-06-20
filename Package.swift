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
        .package(url: "https://github.com/roryford/ManifoldKit", .upToNextMinor(from: "0.56.0")),
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
        // llama.cpp consumed straight from the upstream ggml-org release asset
        // (no third-party wrapper, no git-tag resolution). `url` + `checksum`
        // pin the exact pre-built xcframework deterministically — there is no
        // clone phase to drift, so the "unable to read tree" CI flake that
        // floating wrapper tags caused cannot occur. Bump the build (and
        // re-verify the C API contract) per docs/LLAMA_CONTRACT.md's upgrade
        // procedure, which also points at the upstream release URL + checksum.
        .binaryTarget(
            name: "llama-cpp",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b9553/llama-b9553-xcframework.zip",
            checksum: "8d7d15297300c2724d4630c855d5eb7d92a4eca6c3fd037cdb28b55854e49a67"
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
                "ManifoldLlama",
            ],
            path: "Sources/manifold-tools-llama",
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
                .product(name: "ManifoldRuntime", package: "ManifoldKit"),
                .product(name: "ManifoldPersistenceSwiftData", package: "ManifoldKit"),
                .product(name: "ManifoldTestSupport", package: "ManifoldKit"),
                .product(name: "ManifoldBackendTestKit", package: "ManifoldKit"),
                // Test-only: render the vendored gemma-4 chat_template fixture
                // through the real Jinja engine (issue #45 render assertions).
                .product(name: "Jinja", package: "swift-jinja"),
            ]
        ),
    ]
)
