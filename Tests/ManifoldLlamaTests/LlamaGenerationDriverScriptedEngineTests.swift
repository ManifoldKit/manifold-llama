import XCTest
import LlamaSwift
import ManifoldContract
import ManifoldInference
@_spi(Testing) @testable import ManifoldLlama

/// Drives `LlamaGenerationDriver.run(...)` directly against a
/// ``ScriptedLlamaEngine`` (#165). Complements
/// `LlamaBackendGenerateOrchestrationTests`, which drives the same loop from
/// the top through `LlamaBackend.generate()`: this suite reaches the driver
/// parameters the backend does not currently vary — notably the prefill
/// headroom sampler, which is what makes the third of the three
/// previously-unreachable failure branches (`memoryInsufficient`) testable
/// without a model.
final class LlamaGenerationDriverScriptedEngineTests: XCTestCase {

    // MARK: - Harness

    private func makeStream() -> (GenerationStream, AsyncThrowingStream<GenerationEvent, Error>.Continuation) {
        var capturedContinuation: AsyncThrowingStream<GenerationEvent, Error>.Continuation!
        let raw = AsyncThrowingStream<GenerationEvent, Error> { continuation in
            capturedContinuation = continuation
        }
        return (GenerationStream(raw), capturedContinuation)
    }

    private final class LabelRecorder: @unchecked Sendable {
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

    /// Lock-guarded counter: the driver fires its hooks synchronously from the
    /// generation loop, but the closures are `@Sendable` so a plain `var`
    /// capture will not compile under strict concurrency.
    private final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var value = 0
        /// Returns the pre-increment value.
        @discardableResult func next() -> Int {
            lock.lock(); defer { lock.unlock() }
            let current = value
            value += 1
            return current
        }
        var count: Int {
            lock.lock(); defer { lock.unlock() }
            return value
        }
    }

    private func config(maxOutputTokens: Int = 32) -> GenerationConfig {
        var config = GenerationConfig(temperature: 0.0)
        config.maxOutputTokens = maxOutputTokens
        return config
    }

    // MARK: - memoryInsufficient (#165 AC 3)

    /// The pre-chunk prefill abort guard: when the learned per-token footprint
    /// predicts the next chunk would overrun free memory, the driver must
    /// decline the chunk, fire `onError("memoryInsufficient")`, synchronize,
    /// throw `InferenceError.memoryInsufficient`, and report the KV cache as
    /// incoherent.
    ///
    /// Reaching this without a model needs two things at once: an engine that
    /// answers `decodePromptChunk` without a `llama_context` (so the estimator
    /// can accumulate a sample from chunk 1), and a scripted footprint /
    /// headroom pair. The estimator stays dormant until the first accepted
    /// sample, hence the two-chunk prompt: chunk 1 is decoded and teaches it a
    /// per-token cost, chunk 2 is refused.
    func test_run_prefillHeadroomExhausted_reportsMemoryInsufficient() async throws {
        let engine = ScriptedLlamaEngine(script: ["never"], batchSize: 4)
        let (stream, continuation) = makeStream()
        let recorder = LabelRecorder()

        let drainTask = Task<Error?, Never> {
            do {
                for try await _ in stream.events { }
                return nil
            } catch { return error }
        }

        // Chunk 1 grows resident memory by 400 MB over 4 tokens (100 MB/token),
        // so the guard predicts 4 × 100 MB × 1.5 = 600 MB for chunk 2 against
        // only 1 MB free.
        let footprints: [UInt64] = [0, 400_000_000, 400_000_000, 400_000_000]
        let footprintIndex = Counter()

        let kvCoherent = await LlamaGenerationDriver().run(
            engine: engine,
            tokens: [1, 2, 3, 4, 5, 6],
            reuseLen: 0,
            maxTokens: 8,
            config: config(),
            markers: nil,
            isCancelled: { false },
            generationStream: stream,
            continuation: continuation,
            prefillFootprintSampler: {
                let index = footprintIndex.next()
                return footprints[min(index, footprints.count - 1)]
            },
            prefillHeadroomSampler: { 1_000_000 },
            prefillSafetyFactor: 1.5,
            onError: { recorder.record($0) }
        )

        let thrown = await drainTask.value
        XCTAssertFalse(kvCoherent,
            "a partially-decoded prefix must be reported incoherent so the next turn clears it")
        XCTAssertEqual(recorder.labels, ["memoryInsufficient"])
        XCTAssertEqual(engine.promptChunks.count, 1, "the second chunk must be declined, not decoded")
        XCTAssertGreaterThanOrEqual(engine.synchronizeCount, 1,
            "the abort must drain the GPU before surfacing the error")
        guard case InferenceError.memoryInsufficient = try XCTUnwrap(thrown) else {
            return XCTFail("expected .memoryInsufficient; got \(String(describing: thrown))")
        }
    }

    // MARK: - Token hook fidelity

    /// `onToken` fires once per VISIBLE `.token` event and never for a
    /// `.thinkingToken` — the exclusion `InferenceMetric.timeToFirstToken`
    /// documents. Deleting `onToken?()` from the driver's `.token` case, or
    /// moving it into the `.thinkingToken` case, both land here.
    func test_run_onTokenFiresPerVisibleTokenOnly_notForThinkingTokens() async throws {
        let engine = ScriptedLlamaEngine(script: ["<think>", "reasoning", "</think>", "answer"])
        let (stream, continuation) = makeStream()
        let tokenHits = Counter()

        let collector = Task<(visible: Int, thinking: Int), Never> {
            var visible = 0
            var thinking = 0
            do {
                for try await event in stream.events {
                    switch event {
                    case .token: visible += 1
                    case .thinkingToken: thinking += 1
                    default: break
                    }
                }
            } catch { }
            return (visible, thinking)
        }

        _ = await LlamaGenerationDriver().run(
            engine: engine,
            tokens: [1, 2, 3],
            reuseLen: 0,
            maxTokens: 8,
            config: config(),
            markers: .qwen3,
            isCancelled: { false },
            generationStream: stream,
            continuation: continuation,
            onToken: { _ = tokenHits.next() }
        )

        let counts = await collector.value
        XCTAssertGreaterThan(counts.thinking, 0, "the scripted script must produce reasoning tokens")
        // Without this bound the equality below is satisfiable by 0 == 0 — a
        // future change that made the thinking parser swallow all visible
        // output would leave `onToken` provably broken and this test green.
        XCTAssertGreaterThan(counts.visible, 0,
            "the scripted script must produce visible output after the reasoning block")
        XCTAssertEqual(tokenHits.count, counts.visible,
            "onToken must fire exactly once per visible .token event and never for .thinkingToken")
    }

    // MARK: - Resource contract

    /// Every sampler chain the engine hands out is freed, and the generation
    /// batch is released exactly once — including on the failure paths that
    /// return before the loop.
    func test_run_freesEverySamplerAndReleasesDecodeResources() async throws {
        let engine = ScriptedLlamaEngine(script: ["a", "b"])
        let (stream, continuation) = makeStream()
        let drainTask = Task<Void, Never> {
            do { for try await _ in stream.events { } } catch { }
        }

        _ = await LlamaGenerationDriver().run(
            engine: engine,
            tokens: [1, 2],
            reuseLen: 0,
            maxTokens: 8,
            config: config(),
            markers: nil,
            isCancelled: { false },
            generationStream: stream,
            continuation: continuation
        )
        await drainTask.value

        XCTAssertEqual(engine.samplerRequests, [false],
            "no grammar and no thinking gate means exactly one chain")
        XCTAssertEqual(engine.freedSamplerIDs, [1], "the one chain handed out must be freed")
        XCTAssertEqual(engine.releaseDecodeResourcesCount, 1)
    }

    /// Cancellation is a clean completion, not a failure: no `onError`, no
    /// `.usage` event, and the KV cache stays coherent.
    func test_run_cancellationIsCleanAndEmitsNoError() async throws {
        let engine = ScriptedLlamaEngine(script: ["a", "b", "c"])
        let (stream, continuation) = makeStream()
        let recorder = LabelRecorder()
        let collector = Task<[GenerationEvent], Never> {
            var events: [GenerationEvent] = []
            do { for try await event in stream.events { events.append(event) } } catch { }
            return events
        }

        let kvCoherent = await LlamaGenerationDriver().run(
            engine: engine,
            tokens: [1, 2],
            reuseLen: 0,
            maxTokens: 8,
            config: config(),
            markers: nil,
            isCancelled: { true },
            generationStream: stream,
            continuation: continuation,
            onError: { recorder.record($0) }
        )

        let events = await collector.value
        XCTAssertTrue(kvCoherent, "a cancelled turn leaves the KV cache coherent")
        XCTAssertTrue(recorder.labels.isEmpty, "cancellation must not be recorded as an error")
        XCTAssertFalse(events.contains { if case .usage = $0 { return true }; return false },
            "no usage may be emitted for a turn that never completed")
    }
}
