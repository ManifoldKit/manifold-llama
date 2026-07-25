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

    /// Three tokens with two DISTINCT real gaps (~100ms then ~200ms) must
    /// produce a mean inter-token latency close to their average (~150ms) —
    /// not just "greater than zero", and NOT the value a wrong-divisor
    /// sabotage would produce.
    ///
    /// The gap sizes and tolerance here are chosen deliberately, not
    /// arbitrarily — an earlier version of this test used ~20ms/~40ms gaps
    /// (true mean 30ms) with a ±40ms tolerance, giving bounds of −10ms to
    /// 70ms. That tolerance window is WIDER than the distance from the true
    /// value to either wrong value it was meant to exclude: the negative
    /// lower bound clamps to accepting anything non-negative (including
    /// `.zero`), and `wallClock / tokenCount` (≈60ms / 3 = 20ms) also falls
    /// inside 0–70ms. Both sabotages this test claims to catch would
    /// silently pass. The rule a tolerance must satisfy: it can never exceed
    /// the distance between the true value and the wrong value being
    /// excluded. With 100ms/200ms gaps, the true mean is 150ms while
    /// `wallClock / tokenCount` ≈ 300ms / 3 = 100ms — a 50ms discriminating
    /// distance — so a ±40ms tolerance (110ms–190ms) excludes both `.zero`
    /// and the wrong-divisor value while still absorbing real `Thread.sleep`
    /// / CI scheduling jitter. `XCTAssertGreaterThan(lowerBound, .zero)` below
    /// is a self-check: if a future tolerance change ever pushes the lower
    /// bound negative again, THIS test's own arithmetic has broken and it
    /// must fail loudly rather than silently stop discriminating.
    ///
    /// This has been verified against the actual sabotage it names: with
    /// `meanITL` production code temporarily replaced by `wallClock /
    /// Duration(tokenCount)`, this test failed (observed ~100ms, outside the
    /// 110–190ms window) before the fix was reverted.
    func test_recordToken_multipleTokens_meanInterTokenLatencyMatchesKnownGaps() {
        let tracker = LlamaMetricTracker()
        tracker.start()
        tracker.recordToken()
        Thread.sleep(forTimeInterval: 0.1)
        tracker.recordToken()
        Thread.sleep(forTimeInterval: 0.2)
        tracker.recordToken()
        let metric = tracker.buildMetric(provider: "llama", model: "m", promptTokens: 0)

        let expectedMean = Duration.milliseconds(150)
        let tolerance = Duration.milliseconds(40)
        let lowerBound = expectedMean - tolerance
        let upperBound = expectedMean + tolerance
        XCTAssertGreaterThan(lowerBound, .zero,
            "sanity check on this test's own arithmetic — a non-positive lower bound means the tolerance has stopped discriminating")
        XCTAssertGreaterThan(metric.meanInterTokenLatency, lowerBound,
            "mean ITL \(metric.meanInterTokenLatency) is implausibly low for ~100ms/~200ms gaps — a wrong-divisor sabotage (e.g. wallClock/tokenCount ≈ 100ms) would land here")
        XCTAssertLessThan(metric.meanInterTokenLatency, upperBound,
            "mean ITL \(metric.meanInterTokenLatency) is implausibly high for ~100ms/~200ms gaps")
    }

    // MARK: - nanoseconds(of:) — the ≥1s Duration-truncation regression

    /// Direct regression coverage for the bug this helper was extracted to
    /// fix: reading `Duration.components.attoseconds` alone silently drops
    /// the whole-seconds part. Pure function, no sleeping required — a
    /// synthetic `Duration` reproduces the ≥1s case deterministically.
    func test_nanosecondsOf_wholeSecondGap_isNotTruncatedToZero() {
        XCTAssertEqual(LlamaMetricTracker.nanoseconds(of: .seconds(2)), 2_000_000_000)
    }

    /// A gap with both a whole-second part AND a fractional part must fold
    /// both in — the historical bug reported ONLY the fractional 0.5s here
    /// (500_000_000 ns) instead of the true 1.5s (1_500_000_000 ns).
    func test_nanosecondsOf_wholeAndFractionalSecondGap_foldsBothParts() {
        XCTAssertEqual(LlamaMetricTracker.nanoseconds(of: .milliseconds(1_500)), 1_500_000_000)
    }

    /// Sub-second gaps (the common case for cloud SSE token streaming) must
    /// remain exact — this is the case the pre-fix code already got right,
    /// and the fix must not regress it.
    func test_nanosecondsOf_subSecondGap_isExact() {
        XCTAssertEqual(LlamaMetricTracker.nanoseconds(of: .milliseconds(20)), 20_000_000)
    }

    func test_nanosecondsOf_zeroGap_isZero() {
        XCTAssertEqual(LlamaMetricTracker.nanoseconds(of: .zero), 0)
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
