import ManifoldInference
import XCTest

@testable import ManifoldLlama

/// Headless coverage for the `LlamaBackend` → `metricSink` seam itself
/// (#142 AC 4) — as opposed to `LlamaMetricTrackerTests`, which covers the
/// tracker's internal timing/counting arithmetic in isolation. None of these
/// tests touch a real `llama_context`, so they run unconditionally in CI.
///
/// `LlamaBackend.generate()`'s own `defer` block (the actual emission site)
/// requires a live model to reach — its `guard let (context, vocab) = ...`
/// bails without one. `LlamaMetricTracker.emitMetric(from:to:provider:model:
/// promptTokens:)` is the assemble-and-dispatch logic extracted OUT of that
/// `defer` block so it can be exercised with a stub sink instead, mirroring
/// the same "inject the seam" move `LlamaGenerationDriver.finishDecodeFailure`
/// already uses. A sabotage that deleted the `defer` block's call to
/// `emitMetric(...)`, hardcoded `metricsEnabled = false`, emitted twice, or
/// passed the wrong `provider`/`promptTokens` would all be caught by testing
/// `emitMetric` directly with the exact values `LlamaBackend.generate()`
/// passes it.
final class LlamaBackendMetricSinkTests: XCTestCase {

  // MARK: - metricSink property

  /// Matches `SSECloudBackend`/`FoundationBackend`: a fresh backend
  /// defaults to a non-nil sink so callers can read recent metrics without
  /// any configuration.
  func test_metricSink_defaultsToNonNil() {
    let backend = LlamaBackend()
    XCTAssertNotNil(backend.metricSink)
  }

  /// Setting `metricSink = nil` must be honored — this is the documented
  /// opt-out mechanism.
  func test_metricSink_isSettableToNil() {
    let backend = LlamaBackend()
    backend.metricSink = nil
    XCTAssertNil(backend.metricSink)
  }

  /// Setting a custom sink must stick — the property is a plain stored var,
  /// but a sabotage that made the setter a no-op (or that re-read a stale
  /// captured value instead of the live property) would fail this.
  func test_metricSink_isSettableToCustomSink() {
    let backend = LlamaBackend()
    let stub = StubMetricSink()
    backend.metricSink = stub
    XCTAssertTrue(backend.metricSink === stub)
  }

  // MARK: - emitMetric seam: shape and single-dispatch

  /// The success-path shape: `provider`, `model`, `promptTokens`, and
  /// `completionTokens` (from the tracker's own token count) all land on
  /// the metric the sink receives, `errorClass` is nil, and — critically —
  /// the sink receives EXACTLY ONE metric, not zero and not two.
  func test_emitMetric_dispatchesExactlyOneMetricOfExpectedShape() async throws {
    let tracker = LlamaMetricTracker()
    tracker.start()
    tracker.recordToken()
    tracker.recordToken()
    let sink = StubMetricSink()

    LlamaMetricTracker.emitMetric(
      from: tracker,
      to: sink,
      provider: "llama",
      model: "qwen3-0.6b.gguf",
      promptTokens: 42
    )

    let received = await sink.waitForRecords(count: 1)
    XCTAssertEqual(
      received.count, 1,
      "must dispatch exactly one metric — no double-emit, no silent drop")
    let metric = try XCTUnwrap(received.first)
    XCTAssertEqual(metric.provider, "llama")
    XCTAssertEqual(metric.model, "qwen3-0.6b.gguf")
    XCTAssertEqual(metric.promptTokens, 42)
    XCTAssertEqual(metric.completionTokens, 2)
    XCTAssertNil(metric.errorClass)
  }

  /// A `nil` sink (the documented opt-out) must dispatch nothing and must
  /// not crash — mirrors `LlamaBackend.generate()`'s `metricsEnabled`
  /// early-out, which skips this call entirely when no sink is attached.
  func test_emitMetric_nilSink_dispatchesNothing() {
    let tracker = LlamaMetricTracker()
    tracker.start()
    tracker.recordToken()

    // No sink to observe — the contract under test is "does not crash
    // and performs no work", which a `nil`-safe no-op satisfies.
    LlamaMetricTracker.emitMetric(
      from: tracker, to: nil, provider: "llama", model: "m", promptTokens: 0)
  }

  /// The failure-path shape: `errorClass` set by `recordError(...)` must
  /// reach the emitted metric unchanged.
  func test_emitMetric_errorClassPropagates() async {
    let tracker = LlamaMetricTracker()
    tracker.start()
    tracker.recordError("decodeFailed")
    let sink = StubMetricSink()

    LlamaMetricTracker.emitMetric(
      from: tracker, to: sink, provider: "llama", model: "m", promptTokens: 5)

    let received = await sink.waitForRecords(count: 1)
    XCTAssertEqual(received.count, 1)
    XCTAssertEqual(received.first?.errorClass, "decodeFailed")
  }
}

// MARK: - Test double

/// Records every metric handed to `record(_:)`, in order. An `actor` so
/// concurrent access from the fire-and-forget dispatch `Task` inside
/// `emitMetric(...)` and the test's own polling are both safe without an
/// explicit lock.
private actor StubMetricSink: InferenceMetricSink {
  private(set) var received: [InferenceMetric] = []

  func record(_ metric: InferenceMetric) async {
    received.append(metric)
  }

  /// Polls until at least `count` metrics have been recorded or `timeout`
  /// elapses, then returns whatever was recorded. `emitMetric(...)`
  /// dispatches via a fire-and-forget `Task { await sink.record(metric) }`
  /// (mirrors `SSEGenerationTaskRunner`) — there is no synchronous
  /// happens-before edge back to the caller, so a bare synchronous read
  /// immediately after calling `emitMetric` would be racy. The bound is
  /// generous (2s) to stay robust under CI scheduling jitter while still
  /// failing a genuinely broken (zero-dispatch) test in finite time; the
  /// common case returns almost immediately once the child Task runs.
  func waitForRecords(count: Int, timeout: Duration = .seconds(2)) async -> [InferenceMetric] {
    let deadline = ContinuousClock.now + timeout
    while received.count < count, ContinuousClock.now < deadline {
      try? await Task.sleep(for: .milliseconds(5))
    }
    return received
  }
}
