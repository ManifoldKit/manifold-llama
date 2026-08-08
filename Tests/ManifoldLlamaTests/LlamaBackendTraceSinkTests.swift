import ManifoldInference
import XCTest

@testable import ManifoldLlama

/// Headless coverage for the assemble-and-dispatch half of the `traceSink`
/// seam (#164) — `LlamaMetricTracker.emitMetric(...)` with a stub
/// `RecordingTraceSink`. Mirrors `LlamaBackendMetricSinkTests`.
///
/// These tests run unconditionally in CI (no `llama_context`). They prove
/// span shape, dual-sink dispatch, and both-nil no-op. They do **not**
/// exercise `LlamaBackend.generate()`'s call site — deleting `traceSink:`
/// there, or gating `metricsEnabled` on the metric sink alone, is covered by
/// the model-gated `LlamaBackendTraceSinkE2ETests` suite instead.
final class LlamaBackendTraceSinkTests: XCTestCase {

  // MARK: - traceSink property

  /// Matches `SSECloudBackend`/`FoundationBackend`: a fresh backend leaves
  /// `traceSink` nil (opt-in).
  func test_traceSink_defaultsToNil() {
    let backend = LlamaBackend()
    XCTAssertNil(backend.traceSink)
  }

  /// Setting a custom sink must stick.
  func test_traceSink_isSettable() {
    let backend = LlamaBackend()
    let sink = RecordingTraceSink()
    backend.traceSink = sink
    XCTAssertTrue(backend.traceSink === sink)
    backend.traceSink = nil
    XCTAssertNil(backend.traceSink)
  }

  // MARK: - Dual-sink emit

  /// Success path: exactly one `.llm` span of the expected shape lands on
  /// the trace sink, derived from a tracker that actually recorded tokens
  /// (so a trace-sink-only configuration is not all-zeros).
  func test_emitMetric_dispatchesExactlyOneSpanOfExpectedShape() async throws {
    let tracker = LlamaMetricTracker()
    tracker.start()
    tracker.recordToken()
    tracker.recordToken()
    let sink = RecordingTraceSink()

    LlamaMetricTracker.emitMetric(
      from: tracker,
      to: nil,
      traceSink: sink,
      provider: "llama",
      model: "qwen3-0.6b.gguf",
      promptTokens: 42
    )

    let spans = await waitForSpans(on: sink, count: 1)
    XCTAssertEqual(
      spans.count, 1,
      "must dispatch exactly one span — no double-emit, no silent drop")
    let span = try XCTUnwrap(spans.first)
    XCTAssertEqual(span.kind, .llm)
    XCTAssertEqual(span.name, "qwen3-0.6b.gguf")
    XCTAssertEqual(span.status, .ok)
    XCTAssertEqual(span.attributes[GenAIAttributeKeys.system], .string("llama"))
    XCTAssertEqual(span.attributes[GenAIAttributeKeys.requestModel], .string("qwen3-0.6b.gguf"))
    XCTAssertEqual(span.attributes[GenAIAttributeKeys.usagePromptTokens], .int(42))
    XCTAssertEqual(span.attributes[GenAIAttributeKeys.usageCompletionTokens], .int(2))
    XCTAssertEqual(span.attributes[GenAIAttributeKeys.usageCachedPromptTokens], .int(0))
    // completionTokens == 2 proves the tracker was started and saw tokens —
    // the core trap of building a span from an unstarted tracker yields 0.
    XCTAssertNil(span.context.parentSpanID, "bare generation span is a trace root")
  }

  /// Failure path: `errorClass` from `recordError` becomes span status
  /// `.error(...)` and `error.type`, matching the metric path and core's
  /// adapter tests.
  func test_emitMetric_errorClassPropagatesToSpan() async throws {
    let tracker = LlamaMetricTracker()
    tracker.start()
    tracker.recordError("decodeFailed")
    let sink = RecordingTraceSink()

    LlamaMetricTracker.emitMetric(
      from: tracker,
      to: nil,
      traceSink: sink,
      provider: "llama",
      model: "m",
      promptTokens: 5
    )

    let spans = await waitForSpans(on: sink, count: 1)
    XCTAssertEqual(spans.count, 1)
    let span = try XCTUnwrap(spans.first)
    XCTAssertEqual(span.status, .error("decodeFailed"))
    XCTAssertEqual(span.attributes[GenAIAttributeKeys.errorType], .string("decodeFailed"))
  }

  /// Both sinks attached: each receives exactly one record of the same
  /// generation (metric + span).
  func test_emitMetric_bothSinksReceiveOneRecord() async throws {
    let tracker = LlamaMetricTracker()
    tracker.start()
    tracker.recordToken()
    let metricSink = StubMetricSink()
    let traceSink = RecordingTraceSink()

    LlamaMetricTracker.emitMetric(
      from: tracker,
      to: metricSink,
      traceSink: traceSink,
      provider: "llama",
      model: "m",
      promptTokens: 10
    )

    let metrics = await metricSink.waitForRecords(count: 1)
    let spans = await waitForSpans(on: traceSink, count: 1)
    XCTAssertEqual(metrics.count, 1)
    XCTAssertEqual(spans.count, 1)
    XCTAssertEqual(metrics.first?.completionTokens, 1)
    XCTAssertEqual(
      spans.first?.attributes[GenAIAttributeKeys.usageCompletionTokens],
      .int(1)
    )
  }

  /// Metric-only config must not invent a span (trace sink stays empty when
  /// not passed). Covered implicitly by existing metric tests; this asserts
  /// the dual-emit path still no-ops the absent trace sink.
  func test_emitMetric_metricOnly_doesNotRequireTraceSink() async throws {
    let tracker = LlamaMetricTracker()
    tracker.start()
    let metricSink = StubMetricSink()

    LlamaMetricTracker.emitMetric(
      from: tracker,
      to: metricSink,
      // traceSink defaults to nil
      provider: "llama",
      model: "m",
      promptTokens: 0
    )

    let metrics = await metricSink.waitForRecords(count: 1)
    XCTAssertEqual(metrics.count, 1)
  }

  /// Neither sink attached: pure no-op, no crash.
  func test_emitMetric_neitherSink_dispatchesNothing() {
    let tracker = LlamaMetricTracker()
    tracker.start()
    tracker.recordToken()
    LlamaMetricTracker.emitMetric(
      from: tracker,
      to: nil,
      traceSink: nil,
      provider: "llama",
      model: "m",
      promptTokens: 0
    )
  }

  // MARK: - Helpers

  /// Polls `RecordingTraceSink` until at least `count` spans land or
  /// `timeout` elapses. `emitMetric` dispatches via fire-and-forget
  /// `Task`, so a bare read immediately after the call is racy.
  private func waitForSpans(
    on sink: RecordingTraceSink,
    count: Int,
    timeout: Duration = .seconds(2)
  ) async -> [GenSpan] {
    let deadline = ContinuousClock.now + timeout
    var spans = await sink.recordedSpans()
    while spans.count < count, ContinuousClock.now < deadline {
      try? await Task.sleep(for: .milliseconds(5))
      spans = await sink.recordedSpans()
    }
    return spans
  }
}

// MARK: - Shared test double (metric path)

/// Same shape as the private double in `LlamaBackendMetricSinkTests` —
/// redeclared here because that one is file-private. Actor so concurrent
/// dispatch from `emitMetric`'s fire-and-forget `Task` is safe.
private actor StubMetricSink: InferenceMetricSink {
  private(set) var received: [InferenceMetric] = []

  func record(_ metric: InferenceMetric) async {
    received.append(metric)
  }

  func waitForRecords(count: Int, timeout: Duration = .seconds(2)) async -> [InferenceMetric] {
    let deadline = ContinuousClock.now + timeout
    while received.count < count, ContinuousClock.now < deadline {
      try? await Task.sleep(for: .milliseconds(5))
    }
    return received
  }
}
