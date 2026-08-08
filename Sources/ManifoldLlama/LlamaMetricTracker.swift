import Foundation
import ManifoldInference

/// Accumulates per-token timing data for a single `LlamaBackend.generate()` call
/// and assembles the resulting ``InferenceMetric``.
///
/// Mirrors the shape of `ManifoldInference`'s `GenerationMetricTracker` (see
/// `ManifoldCloudCore`/`ManifoldFoundation`), which this backend cannot reuse
/// directly — that type is `package`-scoped to `ManifoldInference` and
/// unreachable from a companion package (#142). Every mutable field is guarded
/// by `lock`, and the lock is never held across an `await` — callers hand this
/// tracker synchronous `onToken`/`onError` closures fired from inside the
/// generation task, mirroring `FoundationBackend`'s `metricTracker.recordToken()`
/// pattern rather than the SSE-relay wrapper cloud backends use.
final class LlamaMetricTracker: @unchecked Sendable {
  private let lock = NSLock()
  private var wallStart: ContinuousClock.Instant = ContinuousClock.now
  private var dispatchDate: Date = Date()
  private var firstTokenInstant: ContinuousClock.Instant?
  private var lastTokenInstant: ContinuousClock.Instant?
  private var interTokenGapsNs: [Int64] = []
  /// Count of visible `.token` events observed (excludes `.thinkingToken`,
  /// per `InferenceMetric.timeToFirstToken`'s documented exclusion). Used as
  /// the `completionTokens` value for every emitted metric — including
  /// failure paths, where it reports however many tokens streamed before
  /// the failure.
  private var tokenCount = 0
  /// First failure label recorded via ``recordError(_:)``, or `nil` on a
  /// clean run. Only the first call wins — driver failure sites are
  /// mutually exclusive today, but this keeps the tracker honest if that
  /// ever changes.
  private var errorClass: String?

  /// Resets the tracker's clock. Call once, synchronously, at generation
  /// dispatch — before any `.token` event can arrive.
  func start() {
    lock.lock()
    defer { lock.unlock() }
    wallStart = ContinuousClock.now
    dispatchDate = Date()
  }

  /// Records one visible output token. Call synchronously from the same
  /// task that will eventually call ``buildMetric(provider:model:promptTokens:)``
  /// — never from a detached task or another thread — so no `await` ever
  /// separates a read from a write here.
  func recordToken() {
    lock.lock()
    defer { lock.unlock() }
    let now = ContinuousClock.now
    if firstTokenInstant == nil {
      firstTokenInstant = now
    } else if let last = lastTokenInstant {
      // Monotonic clamp, matching `buildMetric`'s `wallStart <= wallEnd`
      // / `wallStart <= first` defensive clamps below. Unreachable in
      // practice with `ContinuousClock` (never goes backwards), but
      // this file already pays for that defense twice — pay it here
      // too rather than being the one path that would silently feed a
      // negative nanosecond count into the running sum (and hence the
      // mean) if it ever were reached.
      interTokenGapsNs.append(now >= last ? Self.nanoseconds(of: now - last) : 0)
    }
    lastTokenInstant = now
    tokenCount += 1
  }

  /// Converts a `Duration` to whole nanoseconds. Pure and testable without a
  /// real clock — see the regression coverage for the ≥1s truncation bug
  /// this exists to prevent.
  ///
  /// `ManifoldKit#2382` tracks the IDENTICAL bug in core's own
  /// `GenerationMetricTracker.swift:35` (confirmed live at both v0.73.0 and
  /// v0.74.0, the pin this backend has been built against). This helper is
  /// a DELIBERATE divergence from that upstream code, not drift — do not
  /// resync this file against core's tracker until #2382 lands there.
  ///
  /// `Duration.components.attoseconds` is only the SUB-SECOND remainder —
  /// it does NOT include whole seconds. Reading it alone silently truncates
  /// any duration ≥ 1s to just its fractional part (e.g. a real 2.0s gap
  /// reports as 0ns; a 1.5s gap reports as 0.5s). Sub-second gaps are the
  /// common case for cloud SSE streaming, which is presumably why this
  /// exact mistake exists upstream in `ManifoldInference`'s package-scoped
  /// `GenerationMetricTracker` too — but local GGUF decode on a pressured
  /// device routinely exceeds 1s/token, where this bug would silently
  /// report FASTER inter-token latency the SLOWER generation actually is.
  /// Must fold in the whole-seconds component explicitly.
  static func nanoseconds(of duration: Duration) -> Int64 {
    let components = duration.components
    return components.seconds * 1_000_000_000 + components.attoseconds / 1_000_000_000
  }

  /// Records a short, stable failure-class label. Safe to call even when no
  /// generation ever started (e.g. a lifecycle race bails before the driver
  /// runs) — the resulting metric still reports the failure.
  func recordError(_ label: String) {
    lock.lock()
    defer { lock.unlock() }
    if errorClass == nil { errorClass = label }
  }

  /// Builds the terminal ``InferenceMetric`` for this generation. Call
  /// exactly once, after the generation task has fully settled (success,
  /// failure, or cancellation) — typically from a `defer` block so it fires
  /// on every exit path exactly once.
  ///
  /// `cachedPromptTokens` is always `0`: local GGUF inference has no
  /// provider-side prompt cache to report against (unlike cloud backends'
  /// server-side prompt caching). KV-cache prefix reuse
  /// (`.kvCacheReuse(promptTokensReused:)`) is a distinct, backend-local
  /// concept — deliberately not mapped onto this field.
  func buildMetric(provider: String, model: String, promptTokens: Int) -> InferenceMetric {
    lock.lock()
    defer { lock.unlock() }
    let wallEnd = ContinuousClock.now
    let wallClock: Duration = wallStart <= wallEnd ? wallEnd - wallStart : .zero

    let ttft: Duration
    if let first = firstTokenInstant {
      ttft = wallStart <= first ? first - wallStart : .zero
    } else {
      ttft = .zero
    }

    let meanITL: Duration
    if interTokenGapsNs.isEmpty {
      meanITL = .zero
    } else {
      let sumNs = interTokenGapsNs.reduce(Int64(0), +)
      let avgNs = sumNs / Int64(interTokenGapsNs.count)
      meanITL = .nanoseconds(avgNs)
    }

    return InferenceMetric(
      provider: provider,
      model: model,
      promptTokens: promptTokens,
      cachedPromptTokens: 0,
      completionTokens: tokenCount,
      timeToFirstToken: ttft,
      meanInterTokenLatency: meanITL,
      wallClockDuration: wallClock,
      errorClass: errorClass,
      timestamp: dispatchDate
    )
  }

  /// Assembles the terminal metric from `tracker` and dispatches it to the
  /// metric and/or trace sinks via fire-and-forget `Task`s, mirroring
  /// `SSEGenerationTaskRunner`'s dual-sink emit:
  /// ```
  /// if let sink = context.metricSink { Task { await sink.record(metric) } }
  /// if let sink = context.traceSink  { Task { await sink.record(metric.asGenSpan()) } }
  /// ```
  /// When both sinks are `nil`, this is a no-op — no metric is built, matching
  /// `LlamaBackend.generate()`'s `metricsEnabled` early-out. When only the
  /// trace sink is attached, the metric is still built (from a tracker that
  /// must have been started — see `metricsEnabled` gating on either sink)
  /// and adapted via ``InferenceMetric/asGenSpan(context:name:)``.
  ///
  /// Extracted out of `LlamaBackend.generate()`'s emit-on-every-exit-path
  /// `defer` block (#142 / #164) so the seam — assemble exactly one metric
  /// of the right shape, dispatch metric + span at most once each — is
  /// exercisable headlessly via `@testable import ManifoldLlama`, without a
  /// live `llama_context`. Mirrors the same "inject the seam" move
  /// `LlamaGenerationDriver.finishDecodeFailure(...)` already uses for its
  /// own headless coverage.
  static func emitMetric(
    from tracker: LlamaMetricTracker,
    to metricSink: (any InferenceMetricSink)?,
    traceSink: (any TraceSink)? = nil,
    provider: String,
    model: String,
    promptTokens: Int
  ) {
    guard metricSink != nil || traceSink != nil else { return }
    let metric = tracker.buildMetric(provider: provider, model: model, promptTokens: promptTokens)
    if let metricSink {
      Task { await metricSink.record(metric) }
    }
    if let traceSink {
      Task { await traceSink.record(metric.asGenSpan()) }
    }
  }
}
