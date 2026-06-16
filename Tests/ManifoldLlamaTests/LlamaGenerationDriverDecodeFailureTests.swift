import XCTest
@_spi(Testing) import ManifoldLlama
@_spi(Testing) import ManifoldInference
import ManifoldContract

/// Headless coverage for the decode-failure and KV-coherence contracts that were
/// previously 0% covered because they require a live `llama_context` returning a
/// nonzero decode result (issue #26).
///
/// `LlamaGenerationDriver.finishDecodeFailure(...)` is the seam extracted from the
/// two `llama_decode != 0` sites (prompt-chunk decode ~542/574, generation-loop
/// decode ~794). It captures the load-bearing ordering — synchronize FIRST, then
/// `.failed`, then finish with `.inferenceFailure`, then return `false` — without
/// a model, by injecting the `synchronize` call as a closure.
///
/// `LlamaBackend.applyKVCoherenceForTesting(_:)` is the seam over the post-decode
/// `if !kvCoherent { sessionKVState = nil }` guard (~LlamaBackend.swift), which
/// drops the cached prefix after a failed decode so the next turn cannot reuse
/// stale positions.
final class LlamaGenerationDriverDecodeFailureTests: XCTestCase {

    // MARK: - Helpers

    /// Records the order in which the decode-failure teardown steps run, so a test
    /// can prove `synchronize` happens BEFORE the stream is finished (the ordering
    /// that prevents the documented Metal command-buffer race).
    private final class OrderRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private var _steps: [String] = []
        func record(_ step: String) {
            lock.lock(); defer { lock.unlock() }
            _steps.append(step)
        }
        var steps: [String] {
            lock.lock(); defer { lock.unlock() }
            return _steps
        }
    }

    /// Builds a `GenerationStream` plus the raw continuation that feeds it, mirroring
    /// how `LlamaBackend.generate()` wires the driver. Returns both so the test can
    /// observe the terminal phase and the thrown error.
    private func makeStream() -> (GenerationStream, AsyncThrowingStream<GenerationEvent, Error>.Continuation) {
        var capturedContinuation: AsyncThrowingStream<GenerationEvent, Error>.Continuation!
        let raw = AsyncThrowingStream<GenerationEvent, Error> { continuation in
            capturedContinuation = continuation
        }
        let stream = GenerationStream(raw)
        return (stream, capturedContinuation)
    }

    // MARK: - finishDecodeFailure ordering contract

    /// The canonical failure contract: synchronize FIRST, then `.failed`, then the
    /// continuation finishes with `.inferenceFailure`, and the call returns `false`.
    func test_finishDecodeFailure_synchronizesBeforeFinishingAndReturnsFalse() async throws {
        let (stream, continuation) = makeStream()
        let recorder = OrderRecorder()

        // Drain the stream concurrently so we can observe the terminal error the
        // continuation is finished with.
        let drain = Task { () -> Error? in
            do {
                for try await _ in stream.events {}
                return nil
            } catch {
                recorder.record("stream-finished")
                return error
            }
        }

        let result = await LlamaGenerationDriver.finishDecodeFailure(
            message: "Failed to decode prompt",
            synchronize: { recorder.record("synchronize") },
            generationStream: stream,
            continuation: continuation
        )

        XCTAssertFalse(result,
            "A decode failure must return false so the caller clears sessionKVState")

        let thrown = await drain.value

        // Ordering: synchronize must precede the stream finishing. Skipping the
        // synchronize (or finishing first) is exactly the Metal command-buffer race
        // this contract guards against — this assertion catches both.
        XCTAssertEqual(recorder.steps, ["synchronize", "stream-finished"],
            "synchronize must run BEFORE the continuation finishes (Metal race guard)")

        let phase = await MainActor.run { stream.phase }
        guard case .failed(let msg) = phase else {
            return XCTFail("Expected .failed phase, got \(phase)")
        }
        XCTAssertEqual(msg, "Failed to decode prompt")

        guard let inferenceError = thrown as? InferenceError,
              case .inferenceFailure(let errMsg) = inferenceError else {
            return XCTFail("Expected InferenceError.inferenceFailure, got \(String(describing: thrown))")
        }
        XCTAssertEqual(errMsg, "Failed to decode prompt")
    }

    /// The generation-loop decode-error site uses a distinct message; verify it
    /// flows through the same contract verbatim.
    func test_finishDecodeFailure_generationLoopMessage_propagates() async throws {
        let (stream, continuation) = makeStream()

        let drain = Task { () -> Error? in
            do { for try await _ in stream.events {}; return nil }
            catch { return error }
        }

        let result = await LlamaGenerationDriver.finishDecodeFailure(
            message: "Decode failed during generation",
            synchronize: {},
            generationStream: stream,
            continuation: continuation
        )

        XCTAssertFalse(result)
        let thrown = await drain.value
        guard let err = thrown as? InferenceError,
              case .inferenceFailure(let msg) = err else {
            return XCTFail("Expected InferenceError.inferenceFailure, got \(String(describing: thrown))")
        }
        XCTAssertEqual(msg, "Decode failed during generation")

        let phase = await MainActor.run { stream.phase }
        XCTAssertEqual(phase, .failed("Decode failed during generation"))
    }

    // MARK: - KV-coherence guard

    /// After an incoherent (failed) decode the cached prefix must be discarded so
    /// the next turn cannot reuse never-coherently-decoded positions.
    func test_applyKVCoherence_false_clearsSessionKVState() {
        let backend = LlamaBackend()
        backend.seedSessionKVStateForTesting(tokenCount: 12)
        XCTAssertEqual(backend.sessionKVTokenCountForTesting, 12,
            "precondition: a synthetic prefix is cached")

        backend.applyKVCoherenceForTesting(false)

        XCTAssertNil(backend.sessionKVTokenCountForTesting,
            "Incoherent decode must clear sessionKVState (stale-prefix-reuse guard)")
    }

    /// A coherent decode must leave the cached prefix intact so legitimate prefix
    /// reuse still happens on the next turn.
    func test_applyKVCoherence_true_preservesSessionKVState() {
        let backend = LlamaBackend()
        backend.seedSessionKVStateForTesting(tokenCount: 7)

        backend.applyKVCoherenceForTesting(true)

        XCTAssertEqual(backend.sessionKVTokenCountForTesting, 7,
            "Coherent decode must not discard the reusable prefix")
    }
}
