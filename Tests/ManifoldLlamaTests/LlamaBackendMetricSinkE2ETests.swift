import XCTest
import ManifoldInference
import ManifoldTestSupport
@_spi(Testing) import ManifoldLlama

/// Model-gated end-to-end coverage for the `LlamaBackend` → `metricSink` wiring
/// itself (#142) — the class of sabotage `LlamaBackendMetricSinkTests`
/// (headless, tests `LlamaMetricTracker.emitMetric` directly) cannot reach:
/// deleting the `emitMetric(...)` call inside `generate()`'s `defer`,
/// hardcoding `metricsEnabled = false`, deleting the `onToken:` argument at
/// the `driver.run(...)` call site, or deleting `onToken?()` inside the
/// driver's `.token` case. All of those require a real generation to
/// actually run for their absence to be observable.
///
/// This suite is a HAPPY-PATH turn only, so it does NOT reach the backend's
/// `onError:` wiring at the `driver.run(...)` call site or the
/// `recordError("backendUnloaded")` branch in `generate()`'s early bail —
/// both would need a forced decode failure or a lifecycle race against a
/// real `llama_context`, which this test does not attempt. See Tier 3 in
/// the PR body for that gap; nothing in this repo's test tree currently
/// covers it.
///
/// Gated on Apple Silicon + Metal + a discoverable GGUF, mirroring
/// `RawPromptEvalRunnerTests`/`LlamaLocalBackendContractTests` — `XCTSkip`s
/// cleanly on CI (no model provisioned) and is picked up by the nightly
/// `model-tests.yml` lane via its `SUITES` allowlist.
final class LlamaBackendMetricSinkE2ETests: XCTestCase {

    /// A real `generate()` call, with a stub sink installed, must deliver
    /// EXACTLY ONE plausible `InferenceMetric`: positive TTFT (a real decode
    /// takes measurable wall-clock time), `completionTokens` matching the
    /// number of `.token` events actually observed on the stream,
    /// `promptTokens` > 0, `errorClass` nil (a clean successful turn), and
    /// `provider == "llama"`.
    ///
    /// A sabotage that deleted the `emitMetric(...)` call in `generate()`'s
    /// `defer`, hardcoded `metricsEnabled = false`, or dropped `onToken:` at
    /// the `driver.run(...)` call site would each be caught here by an
    /// empty/zero-shaped metric or no metric at all. Dropping `onToken?()`
    /// inside the driver's `.token` case is caught by `completionTokens ==
    /// 0` while `tokenEventCount > 0`. Dropping `onError:` at the
    /// `driver.run(...)` call site is NOT caught here — this is a clean
    /// successful turn, so `errorClass == nil` regardless of whether
    /// `onError:` is wired at all (see the type-level doc comment above).
    func test_generate_realTurn_emitsExactlyOnePlausibleMetric() async throws {
        try XCTSkipUnless(HardwareRequirements.isPhysicalDevice,
                          "LlamaBackend requires Metal (unavailable in simulator)")
        try XCTSkipUnless(HardwareRequirements.isAppleSilicon,
                          "LlamaBackend requires Apple Silicon")
        guard let modelURL = HardwareRequirements.findGGUFModel() else {
            throw XCTSkip("No GGUF on disk. Set LLAMA_TEST_MODEL=<path> or "
                        + "MANIFOLD_DISCOVER_LOCAL_MODELS=1 with a `.gguf` in ~/Documents/Models/.")
        }

        let backend = LlamaBackend()
        addTeardownBlock { await backend.unloadAndWait() }

        let sink = StubMetricSink()
        backend.metricSink = sink

        try await backend.loadModel(from: modelURL, plan: .testStub(effectiveContextSize: 512))

        var config = GenerationConfig()
        config.temperature = 0.0
        config.maxOutputTokens = 16
        let stream = try backend.generate(prompt: "The capital of France is", systemPrompt: nil, config: config)

        var tokenEventCount = 0
        for try await event in stream.events {
            if case .token = event { tokenEventCount += 1 }
        }
        await backend.awaitGenerationSettled()

        XCTAssertGreaterThan(tokenEventCount, 0, "a real greedy generation must produce at least one visible token")

        let received = await sink.waitForRecords(count: 1)
        XCTAssertEqual(received.count, 1,
            "a single generate() call must dispatch exactly one metric — this is the exact bug class #142 exists to catch")
        let metric = try XCTUnwrap(received.first)

        XCTAssertEqual(metric.provider, "llama")
        XCTAssertGreaterThan(metric.promptTokens, 0, "the tokenized prompt must be non-empty")
        XCTAssertEqual(metric.completionTokens, tokenEventCount,
            "completionTokens must match the number of .token events actually observed on the stream")
        XCTAssertGreaterThan(metric.timeToFirstToken, .zero,
            "a real decode takes measurable wall-clock time before the first token")
        XCTAssertGreaterThan(metric.wallClockDuration, .zero)
        XCTAssertEqual(metric.cachedPromptTokens, 0)
        XCTAssertNil(metric.errorClass, "a clean, successful turn must not carry an errorClass")
    }
}

// MARK: - Test double

/// Records every metric handed to `record(_:)`, in order. Mirrors
/// `LlamaBackendMetricSinkTests`'s `StubMetricSink` — duplicated rather than
/// shared because that one is `private` to its file and this suite is gated
/// independently (model-gated vs. headless); keeping them separate avoids
/// coupling the CI-always suite's compile to this nightly-only one.
private actor StubMetricSink: InferenceMetricSink {
    private(set) var received: [InferenceMetric] = []

    func record(_ metric: InferenceMetric) async {
        received.append(metric)
    }

    /// Polls until at least `count` metrics have been recorded or `timeout`
    /// elapses. `emitMetric(...)` dispatches via a fire-and-forget `Task`, so
    /// there is no synchronous happens-before edge back to the caller.
    func waitForRecords(count: Int, timeout: Duration = .seconds(5)) async -> [InferenceMetric] {
        let deadline = ContinuousClock.now + timeout
        while received.count < count, ContinuousClock.now < deadline {
            try? await Task.sleep(for: .milliseconds(10))
        }
        return received
    }
}
