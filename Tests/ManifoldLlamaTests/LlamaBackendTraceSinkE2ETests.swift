import ManifoldInference
@_spi(Testing) import ManifoldLlama
import ManifoldTestSupport
import XCTest

/// Model-gated end-to-end coverage for the `LlamaBackend` → `traceSink`
/// wiring itself (#164) — the class of sabotage `LlamaBackendTraceSinkTests`
/// (headless, tests `LlamaMetricTracker.emitMetric` directly) cannot reach:
/// deleting the `traceSink:` argument at the `generate()` emit call site,
/// gating `metricsEnabled` on the metric sink alone, or dropping `onToken:`
/// when only a trace sink is attached. All of those require a real generation
/// to actually run for their absence to be observable.
///
/// Gated on Apple Silicon + Metal + a discoverable GGUF, mirroring
/// `LlamaBackendMetricSinkE2ETests` — `XCTSkip`s cleanly on CI (no model
/// provisioned) and is picked up by the nightly `model-tests.yml` lane via
/// its `SUITES` allowlist.
final class LlamaBackendTraceSinkE2ETests: XCTestCase {

  /// Trace-sink-only configuration: `metricSink = nil`, `traceSink` set.
  /// A real `generate()` must deliver exactly one `.llm` span whose
  /// `usage.completion_tokens` matches the observed `.token` count.
  ///
  /// Sabotages caught here:
  /// - Dropping `traceSink:` at the emit call site → zero spans
  /// - Gating `metricsEnabled` on metric sink alone → span with
  ///   `completion_tokens == 0` while `tokenEventCount > 0` (the core trap)
  /// - Dropping `onToken:` / `onToken?()` → same zero-completion shape
  func test_generate_traceOnly_emitsExactlyOnePlausibleSpan() async throws {
    try XCTSkipUnless(
      HardwareRequirements.isPhysicalDevice,
      "LlamaBackend requires Metal (unavailable in simulator)")
    try XCTSkipUnless(
      HardwareRequirements.isAppleSilicon,
      "LlamaBackend requires Apple Silicon")
    guard let modelURL = HardwareRequirements.findGGUFModel() else {
      throw XCTSkip(
        "No GGUF on disk. Set LLAMA_TEST_MODEL=<path> or "
          + "MANIFOLD_DISCOVER_LOCAL_MODELS=1 with a `.gguf` in ~/Documents/Models/.")
    }

    let backend = LlamaBackend()
    addTeardownBlock { await backend.unloadAndWait() }

    // The exact host configuration that surfaces the core trap: no metric
    // sink, only a trace sink. Tracker bookkeeping must still run.
    backend.metricSink = nil
    let sink = RecordingTraceSink()
    backend.traceSink = sink

    try await backend.loadModel(from: modelURL, plan: .testStub(effectiveContextSize: 512))

    var config = GenerationConfig()
    config.temperature = 0.0
    config.maxOutputTokens = 16
    let stream = try backend.generate(
      prompt: "The capital of France is", systemPrompt: nil, config: config)

    var tokenEventCount = 0
    for try await event in stream.events {
      if case .token = event { tokenEventCount += 1 }
    }
    await backend.awaitGenerationSettled()

    XCTAssertGreaterThan(
      tokenEventCount, 0, "a real greedy generation must produce at least one visible token")

    let spans = await waitForSpans(on: sink, count: 1)
    XCTAssertEqual(
      spans.count, 1,
      "a single generate() call must dispatch exactly one GenSpan — deleting traceSink: at the emit site would yield zero"
    )
    let span = try XCTUnwrap(spans.first)

    XCTAssertEqual(span.kind, .llm)
    XCTAssertEqual(span.status, .ok)
    XCTAssertEqual(span.attributes[GenAIAttributeKeys.system], .string("llama"))
    XCTAssertEqual(
      span.attributes[GenAIAttributeKeys.usageCompletionTokens],
      .int(tokenEventCount),
      "completion_tokens must match observed .token events; zero here with tokenEventCount > 0 means metricsEnabled ignored the trace sink"
    )
    if case .int(let promptTokens)? = span.attributes[GenAIAttributeKeys.usagePromptTokens] {
      XCTAssertGreaterThan(promptTokens, 0, "tokenized prompt must be non-empty")
    } else {
      XCTFail("usage.prompt_tokens attribute missing or wrong type")
    }
  }
}

// MARK: - Helpers

/// Polls `RecordingTraceSink` until at least `count` spans land or timeout.
/// `emitMetric` dispatches via fire-and-forget `Task`.
private func waitForSpans(
  on sink: RecordingTraceSink,
  count: Int,
  timeout: Duration = .seconds(5)
) async -> [GenSpan] {
  let deadline = ContinuousClock.now + timeout
  var spans = await sink.recordedSpans()
  while spans.count < count, ContinuousClock.now < deadline {
    try? await Task.sleep(for: .milliseconds(10))
    spans = await sink.recordedSpans()
  }
  return spans
}
