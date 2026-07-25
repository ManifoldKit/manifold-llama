import XCTest
@_spi(Testing) import ManifoldLlama
@_spi(Testing) import ManifoldInference
import ManifoldContract

/// Headless coverage for the `onError` metrics hook `LlamaGenerationDriver`
/// fires on its decode-failure path (#142). Reuses the same
/// `finishDecodeFailure(...)` seam as `LlamaGenerationDriverDecodeFailureTests`
/// — a live `llama_context` is not required. Runs unconditionally in CI.
final class LlamaGenerationDriverMetricHookTests: XCTestCase {

    // MARK: - Helpers

    /// Builds a `GenerationStream` plus the raw continuation that feeds it,
    /// mirroring `LlamaGenerationDriverDecodeFailureTests.makeStream()`.
    private func makeStream() -> (GenerationStream, AsyncThrowingStream<GenerationEvent, Error>.Continuation) {
        var capturedContinuation: AsyncThrowingStream<GenerationEvent, Error>.Continuation!
        let raw = AsyncThrowingStream<GenerationEvent, Error> { continuation in
            capturedContinuation = continuation
        }
        let stream = GenerationStream(raw)
        return (stream, capturedContinuation)
    }

    private final class ErrorRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private var _labels: [String] = []
        func record(_ label: String) {
            lock.lock(); defer { lock.unlock() }
            _labels.append(label)
        }
        var labels: [String] {
            lock.lock(); defer { lock.unlock() }
            return _labels
        }
    }

    // MARK: - onError wiring

    /// A decode failure must fire `onError("decodeFailed")` exactly once.
    /// The label is asserted via `recorder`.
    func test_finishDecodeFailure_firesOnErrorWithDecodeFailedLabel() async throws {
        let (stream, continuation) = makeStream()
        let recorder = ErrorRecorder()

        let drainTask = Task<Error?, Never> {
            do {
                for try await _ in stream.events { }
                return nil
            } catch {
                return error
            }
        }

        let coherent = await LlamaGenerationDriver.finishDecodeFailure(
            message: "synthetic decode failure",
            synchronize: {},
            generationStream: stream,
            continuation: continuation,
            onError: { label in recorder.record(label) }
        )

        let thrown = await drainTask.value
        XCTAssertFalse(coherent, "a decode failure must report KV state as incoherent")
        XCTAssertNotNil(thrown, "the continuation must finish with a thrown error")
        XCTAssertEqual(recorder.labels, ["decodeFailed"])
    }

    /// `onError` must fire BEFORE `finishDecodeFailure` finishes the
    /// continuation with the thrown error — asserted DETERMINISTICALLY, not
    /// via a concurrent drain task racing to observe an interleaving.
    ///
    /// An earlier version of this test used a second `Task` draining
    /// `stream.events` and had it append `"stream-finished"` to the SAME
    /// recorder `onError` writes to, asserting the two labels landed in
    /// `["decodeFailed", "stream-finished"]` order. That version passed
    /// under the correct code — but it could not have caught the sabotage it
    /// claimed to: `finishDecodeFailure`'s body from `onError?(...)` through
    /// `continuation.finish(throwing:)` has no suspension point (`onError` →
    /// `synchronize()` → `await MainActor.run` → `finish` — the `await` runs
    /// AFTER both calls, not between them), so the calling task always
    /// records BOTH labels to completion, back-to-back, before yielding to
    /// the scheduler at all. Moving `onError?(...)` to after
    /// `continuation.finish(throwing:)` inside `finishDecodeFailure`
    /// therefore ALSO produces `["decodeFailed", "stream-finished"]` — a
    /// standalone repro of that exact call pattern reproduced the "expected"
    /// order in 500/500 trials. The old test could not distinguish the
    /// correct code from the sabotage.
    ///
    /// This version probes `continuation`'s own live/terminated state
    /// directly, from inside the `onError` closure itself, with no second
    /// Task involved: `AsyncThrowingStream.Continuation.yield(_:)` returns
    /// `.enqueued(remaining:)` while the stream is still open, and
    /// `.terminated` once `finish`/`finish(throwing:)` has already run. If
    /// `onError` genuinely fires before `finish`, the probe observes
    /// `.enqueued`; if a sabotage moves `onError` after `finish`, the SAME
    /// call observes `.terminated` instead — a real, load-bearing signal
    /// with no timing dependency.
    func test_finishDecodeFailure_firesOnErrorBeforeContinuationFinishes() async throws {
        let (stream, continuation) = makeStream()

        // Drain concurrently so the stream doesn't back up — not used for
        // ordering evidence (see doc comment above for why that approach
        // doesn't work).
        let drainTask = Task<Error?, Never> {
            do {
                for try await _ in stream.events { }
                return nil
            } catch {
                return error
            }
        }

        nonisolated(unsafe) var yieldResultAtOnError: AsyncThrowingStream<GenerationEvent, Error>.Continuation.YieldResult?

        let coherent = await LlamaGenerationDriver.finishDecodeFailure(
            message: "synthetic decode failure",
            synchronize: {},
            generationStream: stream,
            continuation: continuation,
            onError: { _ in
                // Injects one harmless extra `.token` event ahead of the
                // eventual thrown error; the drain loop above only cares
                // about the terminal error, so this is a no-op for it.
                yieldResultAtOnError = continuation.yield(.token(""))
            }
        )

        let thrown = await drainTask.value
        XCTAssertFalse(coherent)
        XCTAssertNotNil(thrown)

        guard case .enqueued = yieldResultAtOnError else {
            XCTFail("onError must fire BEFORE continuation.finish(throwing:) — "
                + "expected .enqueued (stream still live), observed "
                + "\(String(describing: yieldResultAtOnError)) (stream already finished)")
            return
        }
    }

    /// `onError` defaults to `nil` — existing callers (including this
    /// backend's own two `finishDecodeFailure(...)` call sites before #142,
    /// and every pre-existing direct test caller) must keep compiling and
    /// behaving identically without passing it.
    func test_finishDecodeFailure_withoutOnError_stillCompletesTheContract() async throws {
        let (stream, continuation) = makeStream()

        let drainTask = Task<Error?, Never> {
            do {
                for try await _ in stream.events { }
                return nil
            } catch {
                return error
            }
        }

        let coherent = await LlamaGenerationDriver.finishDecodeFailure(
            message: "synthetic decode failure",
            synchronize: {},
            generationStream: stream,
            continuation: continuation
        )

        let thrown = await drainTask.value
        XCTAssertFalse(coherent)
        XCTAssertNotNil(thrown)
    }
}
