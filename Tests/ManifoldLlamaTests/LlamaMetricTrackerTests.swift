import XCTest
import ManifoldInference
@testable import ManifoldLlama

/// Headless coverage for `LlamaMetricTracker` — the local mirror of
/// `ManifoldInference`'s package-scoped `GenerationMetricTracker` that
/// `LlamaBackend.generate()` uses to build the `InferenceMetric` it hands to
/// `metricSink` (#142). None of these tests touch a real `llama_context`, so
/// they run unconditionally in CI (no `.gguf` required).
final class LlamaMetricTrackerTests: XCTestCase {

    // MARK: - Zero-token shape

    /// A generation that never emits a single token (e.g. the backend bails
    /// before the driver ever runs, or a decode fails on the very first
    /// chunk) must report a zero TTFT and zero mean inter-token latency, not
    /// crash or leave stale timing state from tracker construction.
    func test_buildMetric_zeroTokens_reportsZeroTTFTAndZeroITL() {
        let tracker = LlamaMetricTracker()
        tracker.start()
        let metric = tracker.buildMetric(provider: "llama", model: "test-model", promptTokens: 12)

        XCTAssertEqual(metric.timeToFirstToken, .zero)
        XCTAssertEqual(metric.meanInterTokenLatency, .zero)
        XCTAssertEqual(metric.completionTokens, 0)
        XCTAssertEqual(metric.promptTokens, 12)
        XCTAssertEqual(metric.provider, "llama")
        XCTAssertEqual(metric.model, "test-model")
        XCTAssertNil(metric.errorClass)
        XCTAssertEqual(metric.cachedPromptTokens, 0)
    }

    // MARK: - Token accounting

    /// `completionTokens` on the built metric must equal the number of
    /// `recordToken()` calls — this is the only source of truth for
    /// completion-token count (see `LlamaBackend.generate()`, which does not
    /// thread the driver's separate `onUsage` completion count into the
    /// metric). A sabotage that dropped every other `recordToken()` call
    /// would fail this assertion.
    func test_recordToken_countsExactlyEachCall() {
        let tracker = LlamaMetricTracker()
        tracker.start()
        for _ in 0..<5 {
            tracker.recordToken()
        }
        let metric = tracker.buildMetric(provider: "llama", model: "m", promptTokens: 0)
        XCTAssertEqual(metric.completionTokens, 5)
    }

    /// First token defines TTFT; the tracker must not report a TTFT of zero
    /// once at least one token has been recorded after a real elapsed gap.
    func test_recordToken_firstTokenProducesNonZeroTTFT() {
        let tracker = LlamaMetricTracker()
        tracker.start()
        Thread.sleep(forTimeInterval: 0.02)
        tracker.recordToken()
        let metric = tracker.buildMetric(provider: "llama", model: "m", promptTokens: 0)
        XCTAssertGreaterThan(metric.timeToFirstToken, .zero)
    }

    /// A single token produces no inter-token gap (need ≥2 to measure one) —
    /// `meanInterTokenLatency` must stay zero, not divide-by-zero or crash.
    func test_recordToken_singleToken_zeroMeanInterTokenLatency() {
        let tracker = LlamaMetricTracker()
        tracker.start()
        tracker.recordToken()
        let metric = tracker.buildMetric(provider: "llama", model: "m", promptTokens: 0)
        XCTAssertEqual(metric.meanInterTokenLatency, .zero)
    }

    /// Two or more tokens with a real gap between them must produce a
    /// positive mean inter-token latency.
    func test_recordToken_multipleTokens_positiveMeanInterTokenLatency() {
        let tracker = LlamaMetricTracker()
        tracker.start()
        tracker.recordToken()
        Thread.sleep(forTimeInterval: 0.02)
        tracker.recordToken()
        let metric = tracker.buildMetric(provider: "llama", model: "m", promptTokens: 0)
        XCTAssertGreaterThan(metric.meanInterTokenLatency, .zero)
    }

    // MARK: - Error recording

    /// `recordError` sets `errorClass` on the built metric — this is the
    /// failure-path contract `LlamaGenerationDriver`'s `onError` hook relies
    /// on.
    func test_recordError_setsErrorClassOnMetric() {
        let tracker = LlamaMetricTracker()
        tracker.start()
        tracker.recordError("decodeFailed")
        let metric = tracker.buildMetric(provider: "llama", model: "m", promptTokens: 0)
        XCTAssertEqual(metric.errorClass, "decodeFailed")
    }

    /// Only the FIRST recorded error label wins. `LlamaGenerationDriver`'s
    /// failure branches are mutually exclusive today, but the tracker must
    /// stay honest (report the first real failure, not the last) if that
    /// ever changes — e.g. a decode failure followed by teardown noise should
    /// not silently overwrite the original cause.
    func test_recordError_firstCallWins() {
        let tracker = LlamaMetricTracker()
        tracker.start()
        tracker.recordError("decodeFailed")
        tracker.recordError("memoryInsufficient")
        let metric = tracker.buildMetric(provider: "llama", model: "m", promptTokens: 0)
        XCTAssertEqual(metric.errorClass, "decodeFailed")
    }

    /// A failed generation can still have streamed tokens before the failure
    /// (e.g. a decode error mid-generation-loop, after several tokens were
    /// already yielded) — `errorClass` and `completionTokens` must both be
    /// present on the same metric, not one clobbering the other.
    func test_recordError_afterTokens_preservesBothTokenCountAndErrorClass() {
        let tracker = LlamaMetricTracker()
        tracker.start()
        tracker.recordToken()
        tracker.recordToken()
        tracker.recordError("decodeFailed")
        let metric = tracker.buildMetric(provider: "llama", model: "m", promptTokens: 3)
        XCTAssertEqual(metric.completionTokens, 2)
        XCTAssertEqual(metric.errorClass, "decodeFailed")
    }

    // MARK: - cachedPromptTokens

    /// Local GGUF inference has no provider-side prompt cache — this field
    /// must always be `0`, regardless of prompt/completion token counts.
    func test_buildMetric_cachedPromptTokensAlwaysZero() {
        let tracker = LlamaMetricTracker()
        tracker.start()
        tracker.recordToken()
        let metric = tracker.buildMetric(provider: "llama", model: "m", promptTokens: 999)
        XCTAssertEqual(metric.cachedPromptTokens, 0)
    }
}
