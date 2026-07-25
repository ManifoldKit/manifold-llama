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
            let gapNs = Int64((now - last).components.attoseconds / 1_000_000_000)
            interTokenGapsNs.append(gapNs)
        }
        lastTokenInstant = now
        tokenCount += 1
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
}
