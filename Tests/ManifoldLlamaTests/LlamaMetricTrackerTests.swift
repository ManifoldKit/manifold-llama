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

    /// Three tokens with two DISTINCT real gaps (~100ms then ~200ms, true
    /// mean 150ms) must produce a mean inter-token latency ABOVE a floor that
    /// excludes both `.zero` and the named `wallClock / tokenCount` sabotage
    /// (≈100ms) — not just "greater than zero".
    ///
    /// **Bound history, in order — read this before touching the numbers.**
    /// v1 used ~20ms/~40ms gaps (true mean 30ms) with a symmetric ±40ms
    /// tolerance: bounds of −10ms to 70ms. The negative lower bound clamped
    /// to accepting anything non-negative including `.zero`, and the
    /// wrong-divisor result (≈20ms there) also fell inside the window —
    /// neither excluded sabotage was actually excluded. v2 widened the gaps
    /// to ~100ms/~200ms and kept a symmetric ±40ms tolerance: bounds of
    /// 110ms–190ms. The LOWER bound (110ms) genuinely discriminates
    /// (verified — see below). The UPPER bound (190ms) does no
    /// discriminating work at all — both excluded values sit BELOW the true
    /// mean, so nothing needs excluding above it.
    ///
    /// v3 replaced the symmetric tolerance with an explicit 120ms floor (the
    /// real, discriminating assertion) and a loose 1s ceiling, reasoning from
    /// a LOCAL flake estimate: ~20–50ms of `Thread.sleep` overshoot per call
    /// on a loaded machine. v3 was then reverted back to v2's symmetric
    /// ±40ms window on the belief that the estimate was too pessimistic —
    /// measured locally (8 idle runs + 3 more under 40 spinning `yes`
    /// processes) the total overshoot across both sleeps was only 6–11ms
    /// against an 80ms budget, comfortably inside 110–190ms.
    ///
    /// **v4 (current) — restores v3's floor/loose-ceiling shape, because the
    /// v2 symmetric window went red in CI, not just locally.** On a GitHub
    /// macOS runner, `swift test` observed a mean of **287ms** for this exact
    /// test (`XCTAssertLessThan` failed: 0.2875s is not less than 0.19s) —
    /// roughly **137ms of overshoot per `Thread.sleep` call**, more than
    /// 10x the 6–11ms *total* measured on a loaded local machine. The
    /// conclusion: a virtualized CI runner does not schedule like a loaded
    /// local machine, so local load-testing — however thorough — does not
    /// model CI scheduling and cannot justify a tight ceiling. The floor
    /// (120ms) is unaffected by overshoot (overshoot only pushes the
    /// observed value UP, away from the floor) and remains the real,
    /// load-bearing assertion; the ceiling is restored to a loose 1s pure
    /// sanity check, which the CI-observed 287ms clears with wide margin.
    /// This means the test does not catch a hypothetical `meanITL =
    /// wallClock` (no averaging at all, ≈300ms here) sabotage — a real but
    /// unnamed regression this test does not claim to guard against. Do NOT
    /// re-tighten this ceiling based on ANY local measurement, however
    /// extensive — only CI's own observed distribution is evidence for what
    /// bound is safe here.
    ///
    /// `XCTAssertGreaterThan(floor, .zero)` below is a self-check: if a
    /// future edit ever pushes the floor non-positive, THIS test's own
    /// arithmetic has broken and it must fail loudly rather than silently
    /// stop discriminating.
    ///
    /// Verified against the actual sabotage it names: with `meanITL`
    /// production code temporarily replaced by `wallClock /
    /// Duration(tokenCount)`, this test failed (observed ≈100ms, below the
    /// 120ms floor) before the fix was reverted.
    func test_recordToken_multipleTokens_meanInterTokenLatencyMatchesKnownGaps() {
        let tracker = LlamaMetricTracker()
        tracker.start()
        tracker.recordToken()
        Thread.sleep(forTimeInterval: 0.1)
        tracker.recordToken()
        Thread.sleep(forTimeInterval: 0.2)
        tracker.recordToken()
        let metric = tracker.buildMetric(provider: "llama", model: "m", promptTokens: 0)

        let floor = Duration.milliseconds(120)
        let looseCeiling = Duration.seconds(1)
        XCTAssertGreaterThan(floor, .zero,
            "sanity check on this test's own arithmetic — a non-positive floor means the assertion below has stopped discriminating")
        XCTAssertGreaterThan(metric.meanInterTokenLatency, floor,
            "mean ITL \(metric.meanInterTokenLatency) is implausibly low for ~100ms/~200ms gaps (true mean 150ms) — a wrong-divisor sabotage (e.g. wallClock/tokenCount ≈ 100ms) would land at or below this floor")
        XCTAssertLessThan(metric.meanInterTokenLatency, looseCeiling,
            "mean ITL \(metric.meanInterTokenLatency) is wildly implausible for ~100ms/~200ms gaps — this is a loose sanity ceiling, not a discriminator (see doc comment)")
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
