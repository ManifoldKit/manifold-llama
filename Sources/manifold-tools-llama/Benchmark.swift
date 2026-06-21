// Cold-vs-warm generation benchmark (--bench).
//
// Measures the one-time, per-process cost that the first generation pays over
// every subsequent ("warm") generation. On Metal that cost is dominated by
// ggml-metal building `MTLComputePipelineState` objects from the *pre-compiled*
// llama.cpp metallib the first time each kernel is touched — including the
// flash-attention kernels when FA is enabled (LlamaModelLoader.swift:258). The
// metallib itself is shipped pre-compiled in the pinned xcframework, so there is
// no `.metal` shader-source compilation here; this benchmark isolates only the
// runtime pipeline warm-up.
//
// IMPORTANT — cold is a *per-process* phenomenon: ggml-metal caches pipeline
// states on the Metal device, which outlives context creation. So once FA
// kernels are built in this process, a *second* load's "cold" run no longer
// re-pays for them. For an honest FA-on vs FA-off comparison, run the benchmark
// twice in separate processes (`--flash on`, then `--flash off`). `--flash both`
// is offered for convenience but the second config's cold number is contaminated
// by the shared cache — the harness prints a warning when used.
import Foundation
import ManifoldInference
import ManifoldLlama

/// One configuration's measurements. Times are wall-clock milliseconds.
struct BenchResult {
    let flashAttention: Bool
    /// Context size requested via `.systemManaged(requestedContextSize:)`.
    let requestedContext: Int
    /// Plan-effective context after the load planner's clamp, read back from
    /// `backend.manifest?.contextWindow`. May be < requested if the planner
    /// trimmed it to fit memory / the model's trained context. This is what was
    /// wired into `ctxParams.n_ctx`; llama.cpp may still adjust internally
    /// (`docs/LLAMA_CONTRACT.md:137`), but no public API re-queries `llama_n_ctx`.
    let effectiveContext: Int?
    let loadMs: Double
    let coldMs: Double
    let warmMs: [Double]

    var warmAvgMs: Double { warmMs.isEmpty ? .nan : warmMs.reduce(0, +) / Double(warmMs.count) }
    /// The cold-over-warm delta — the per-process one-time cost the cold run
    /// pays that the warm runs do not (pipeline-state creation + first-touch).
    var deltaMs: Double { coldMs - warmAvgMs }
}

enum Benchmark {

    /// Wall-clock milliseconds elapsed while running `body`.
    private static func timed(_ body: () async throws -> Void) async rethrows -> Double {
        let start = ContinuousClock.now
        try await body()
        let elapsed = ContinuousClock.now - start
        let c = elapsed.components
        return Double(c.seconds) * 1_000 + Double(c.attoseconds) / 1_000_000_000_000_000
    }

    /// Drives one full generation to completion, draining the token stream and
    /// awaiting the in-flight task so the next `generate()` cannot race the
    /// `isGenerating` defer (see `LlamaSeedDeterminismTests.collectTokens`).
    private static func runOnce(
        backend: LlamaBackend,
        prompt: String,
        config: GenerationConfig
    ) async throws {
        let stream = try backend.generate(prompt: prompt, systemPrompt: nil, config: config)
        for try await event in stream.events {
            if case .token = event { continue }
        }
        await backend.awaitGenerationSettled()
    }

    /// Loads `modelURL` with the given FA setting and measures load, a single
    /// cold generation, and `warmRuns` warm generations. Each generation uses an
    /// identical prompt + fixed seed and is preceded (warm runs) by
    /// `resetConversation()` so the KV cache is cleared and every run re-prefills
    /// exactly the same work — the only systematic difference between cold and
    /// warm is the one-time per-process pipeline warm-up.
    static func measure(
        modelURL: URL,
        flashAttention: Bool,
        prompt: String,
        maxTokens: Int,
        warmRuns: Int,
        contextSize: Int
    ) async throws -> BenchResult {
        let backend = LlamaBackend()
        // Teardown is awaited explicitly at the end (defer cannot await).
        backend.setLoadOptions(BackendLoadOptions(flashAttention: flashAttention))

        let loadMs = try await timed {
            try await backend.loadModel(
                from: modelURL,
                plan: .systemManaged(requestedContextSize: contextSize))
        }
        let effectiveContext = backend.manifest?.contextWindow

        var config = GenerationConfig(temperature: 0.8, maxOutputTokens: maxTokens)
        config.seed = 42

        let coldMs = try await timed {
            try await runOnce(backend: backend, prompt: prompt, config: config)
        }

        var warmMs: [Double] = []
        for _ in 0..<max(0, warmRuns) {
            backend.resetConversation()
            let ms = try await timed {
                try await runOnce(backend: backend, prompt: prompt, config: config)
            }
            warmMs.append(ms)
        }

        await backend.unloadAndWait()
        return BenchResult(
            flashAttention: flashAttention, requestedContext: contextSize,
            effectiveContext: effectiveContext, loadMs: loadMs, coldMs: coldMs, warmMs: warmMs)
    }

    private static func fmt(_ ms: Double) -> String {
        ms.isNaN ? "n/a" : String(format: "%.1f ms", ms)
    }

    /// Entry point for `--bench`. Returns a process exit code.
    static func run(
        modelURL: URL,
        flash: String,
        prompt: String,
        maxTokens: Int,
        warmRuns: Int,
        contextSize: Int
    ) async -> Int32 {
        // No explicit hardware gate: this is a macOS CLI that always runs on the
        // host (never the iOS simulator), and a non-Metal host surfaces a loud
        // load failure below — same as the scenario harness in main.swift.
        let settings: [Bool]
        switch flash.lowercased() {
        case "on":  settings = [true]
        case "off": settings = [false]
        case "both":
            settings = [true, false]
            FileHandle.standardError.write(Data(
                ("manifold-tools-llama: WARNING — --flash both runs both configs in ONE process; "
               + "the second config's COLD time is contaminated by the shared Metal pipeline cache "
               + "and will under-report the warm-up. For an honest cold comparison run --flash on and "
               + "--flash off in separate processes.\n").utf8))
        default:
            FileHandle.standardError.write(Data(
                "manifold-tools-llama: --flash must be on|off|both (got '\(flash)')\n".utf8))
            return 2
        }

        print("Cold-vs-warm benchmark")
        print("  model:        \(modelURL.lastPathComponent)")
        print("  prompt:       \"\(prompt)\"")
        print("  max tokens:   \(maxTokens)")
        print("  warm runs:    \(warmRuns)")
        print("  context req:  \(contextSize)")
        print("")

        var results: [BenchResult] = []
        for fa in settings {
            print("── flash attention: \(fa ? "ON" : "OFF") ──")
            do {
                let r = try await Benchmark.measure(
                    modelURL: modelURL, flashAttention: fa, prompt: prompt,
                    maxTokens: maxTokens, warmRuns: warmRuns, contextSize: contextSize)
                results.append(r)
                let ctxNote = r.effectiveContext.map { eff in
                    eff == r.requestedContext ? "\(eff)" : "\(eff)  (clamped from \(r.requestedContext))"
                } ?? "unknown (no manifest)"
                print("  eff context: \(ctxNote)")
                print("  load:        \(fmt(r.loadMs))")
                print("  cold gen:    \(fmt(r.coldMs))")
                let warmList = r.warmMs.map { fmt($0) }.joined(separator: ", ")
                print("  warm gen:    avg \(fmt(r.warmAvgMs))  [\(warmList)]")
                print("  cold−warm:   \(fmt(r.deltaMs))   (≈ one-time per-process warm-up)")
                print("")
            } catch {
                let detail = (error as? LocalizedError)?.errorDescription ?? "\(error)"
                FileHandle.standardError.write(Data("  BENCH FAILED (fa=\(fa)): \(detail)\n".utf8))
                return 1
            }
        }

        // When both settings ran, surface the FA-attributable slice of the cold
        // delta — but only the FA-ON cold figure is trustworthy (see warning).
        if results.count == 2,
           let on = results.first(where: { $0.flashAttention }),
           let off = results.first(where: { !$0.flashAttention }) {
            print("── summary ──")
            print("  FA-on  cold−warm: \(fmt(on.deltaMs))  (trustworthy: FA kernels built fresh this process)")
            print("  FA-off cold−warm: \(fmt(off.deltaMs))  (contaminated: pipelines already cached)")
        }

        return 0
    }
}
