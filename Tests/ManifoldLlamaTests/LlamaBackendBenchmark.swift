import XCTest
import ManifoldInference
import ManifoldTestSupport
import ManifoldLlama

/// LOCAL, UNCOMMITTED throughput benchmark — mirror of core's
/// `OllamaBackendBenchmark` and manifold-mlx's `MLXBackendBenchmark` so the
/// MK→llama.cpp in-process lane can be measured with the *identical* prompt,
/// config, warmup, and run count. Prints a `BENCH_RESULT` sentinel line.
///
/// Run:
///   LLAMA_TEST_MODEL=/path/to/Model-Q4_K_M.gguf \
///     swift test -c release --filter LlamaBackendBenchmark
private let benchPrompt = "Write a short story about a robot learning to paint. Be concise."
private let benchRuns   = 4
private let benchTokens = 300

@MainActor
final class LlamaBackendBenchmark: XCTestCase {

    private var backend: LlamaBackend!
    private var modelName = "unknown"

    override func setUp() async throws {
        try await super.setUp()
        try XCTSkipUnless(HardwareRequirements.isPhysicalDevice,
                          "LlamaBackend requires Metal (unavailable in simulator)")
        try XCTSkipUnless(HardwareRequirements.isAppleSilicon,
                          "LlamaBackend requires Apple Silicon")
        guard let modelURL = HardwareRequirements.findGGUFModel() else {
            throw XCTSkip("No GGUF — set LLAMA_TEST_MODEL=<path> or place a .gguf in ~/Documents/Models/")
        }
        modelName = modelURL.deletingPathExtension().lastPathComponent
        backend = LlamaBackend()
        // Match Ollama's runtime context budget (-c 8192).
        try await backend.loadModel(from: modelURL, plan: .testStub(effectiveContextSize: 8192))
    }

    override func tearDown() async throws {
        await backend?.unloadAndWait()
        backend = nil
        try await super.tearDown()
    }

    private func timedGenerate() async throws -> (ttftMs: Double, totalMs: Double, tokens: Int) {
        let config = GenerationConfig(temperature: 0.3, maxOutputTokens: benchTokens)
        let t0 = ContinuousClock.now
        var t1: ContinuousClock.Instant?
        var count = 0
        let stream = try backend.generate(prompt: benchPrompt, systemPrompt: nil, config: config)
        for try await event in stream.events {
            if case .token = event {
                if t1 == nil { t1 = ContinuousClock.now }
                count += 1
            }
        }
        let t2 = ContinuousClock.now
        func ms(_ d: Duration) -> Double {
            Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15
        }
        guard let first = t1 else { return (0, ms(t2 - t0), 0) }
        return (ms(first - t0), ms(t2 - t0), count)
    }

    func test_throughput() async throws {
        let config = GenerationConfig(temperature: 0.3, maxOutputTokens: benchTokens)
        // Warmup
        let warmup = try backend.generate(prompt: benchPrompt, systemPrompt: nil, config: config)
        for try await _ in warmup.events {}
        backend.resetConversation()

        var results: [(ttftMs: Double, totalMs: Double, tokens: Int)] = []
        for _ in 1...benchRuns {
            results.append(try await timedGenerate())
            backend.resetConversation()
        }
        for (i, r) in results.enumerated() {
            let tps = Double(r.tokens) / (r.totalMs / 1000)
            print(String(format: "  [ManifoldKit→llama.cpp run %d] TTFT=%.0fms  total=%.0fms  tokens=%d  TPS=%.1f",
                         i + 1, r.ttftMs, r.totalMs, r.tokens, tps))
        }
        let sortedTTFT = results.map(\.ttftMs).sorted()
        let sortedTPS  = results.map { Double($0.tokens) / ($0.totalMs / 1000) }.sorted()
        func median(_ xs: [Double]) -> Double {
            let n = xs.count
            return n.isMultiple(of: 2) ? (xs[n / 2 - 1] + xs[n / 2]) / 2 : xs[n / 2]
        }
        print(String(format: "BENCH_RESULT label=ManifoldKit→llama.cpp model=%@ median_ttft_ms=%.0f median_tps=%.1f",
                     modelName, median(sortedTTFT), median(sortedTPS)))
        XCTAssertGreaterThan(sortedTPS.max() ?? 0, 1)
    }
}
