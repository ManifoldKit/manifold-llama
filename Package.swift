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
    ],
    dependencies: [
        // The ManifoldBackendTestKit / ManifoldTestSupport products this package
        // needs exist only on main until the 0.48 tags ship.
        // traits: [] builds core's products trait-less (the post-C2 world).
        .package(url: "https://github.com/roryford/ManifoldKit", .upToNextMinor(from: "0.53.0")),
        // Pinned EXACT to 2.9505.0 (Package.resolved rev 11efdff6cfadc8ed2f998dc6f50d68d3e35237f9).
        // Wraps llama.cpp as a pre-built xcframework binary. mattt/llama.swift auto-tags a new
        // version per upstream commit; a floating `from:` lets CI resolution drift to the newest
        // tag, and the cached SwiftPM clone can land in an `unable to read tree` state for a
        // just-pushed revision — breaking every CI run repo-wide regardless of the lockfile. Exact
        // pinning keeps resolution deterministic. Bump intentionally (and re-verify the C API
        // contract) per docs/LLAMA_CONTRACT.md's upgrade procedure.
        .package(url: "https://github.com/mattt/llama.swift", exact: "2.9553.0"),
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
                .product(name: "LlamaSwift", package: "llama.swift"),
            ],
            path: "Sources/ManifoldLlama"
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
            ]
        ),
    ]
)
