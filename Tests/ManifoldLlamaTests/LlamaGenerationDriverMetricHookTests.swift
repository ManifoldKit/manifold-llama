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

    /// A decode failure must fire `onError("decodeFailed")` exactly once,
    /// BEFORE the continuation finishes with the thrown error — a sabotage
    /// that dropped the `onError?(...)` call (or moved it after
    /// `continuation.finish`) would fail this either on the missing call or
    /// on ordering relative to the drain below.
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
