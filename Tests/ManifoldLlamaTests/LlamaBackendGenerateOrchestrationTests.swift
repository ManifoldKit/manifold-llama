import XCTest
import LlamaSwift
import ManifoldContract
import ManifoldInference
@_spi(Testing) @testable import ManifoldLlama

/// Headless coverage for `LlamaBackend.generate()`'s **Task body** (#165) — the
/// region that previously required a live `llama_context` and so was reachable
/// only from the nightly model-gated lane.
///
/// What is under test here is the orchestration, not a reimplementation of it.
/// `installGenerationSeamForTesting(_:)` swaps exactly two things — prompt
/// tokenization and construction of the ``LlamaEngine`` — and everything
/// between them runs as production code: the context-window preflight, the KV
/// shared-prefix scan, the `isGenerating` flip, the `metricsEnabled` gate, the
/// tracker `start()`, the `onToken`/`onError` hook construction, the
/// `driver.run(...)` call with all of its arguments, the `defer`-based
/// `emitMetric(...)`, and the post-decode `applyKVCoherence(...)` guard.
///
/// Sabotage resistance is the point (see #165's acceptance criteria). Each of
/// these fails after the corresponding one-line sabotage:
///
/// | Sabotage | Caught by |
/// |---|---|
/// | delete the `emitMetric(...)` call | `test_generate_emitsExactlyOneMetricOfExpectedShape` (no metric arrives) |
/// | hardcode `metricsEnabled = false` | same test (`completionTokens == 0`, tracker never started) |
/// | delete `onToken:` / `onError:` at the `driver.run(...)` call site | same test / `test_generate_promptDecodeFailure_...` |
/// | delete `onToken?()` in the driver's `.token` case | same test (`completionTokens == 0`) |
/// | pass a wrong `provider` / `model` / `promptTokens` | same test (each is asserted against an independently known value) |
final class LlamaBackendGenerateOrchestrationTests: XCTestCase {

    // MARK: - Harness

    /// Builds a backend armed with sentinel pointers, a scripted engine, and a
    /// fixed tokenization, and registers the disarm teardown that #54 requires.
    ///
    /// `promptTokens` is deliberately a value the test knows independently of
    /// the prompt string, so the emitted metric's `promptTokens` field can only
    /// be right if `generate()` really threaded `tokens.count` through.
    private func makeBackend(
        engine: ScriptedLlamaEngine,
        promptTokens: [llama_token],
        modelIdentifier: String = "scripted-model.gguf"
    ) -> LlamaBackend {
        let backend = LlamaBackend()
        backend.armFakeLoadedStateForTesting()
        addTeardownBlock { backend.disarmFakeLoadedStateForTesting() }
        backend.injectModelIdentifierForTesting(modelIdentifier)
        backend.installGenerationSeamForTesting(LlamaGenerationSeam(
            tokenize: { _, _ in promptTokens },
            makeEngine: { _, _ in engine }
        ))
        return backend
    }

    private func config(maxOutputTokens: Int = 32, grammar: String? = nil) -> GenerationConfig {
        var config = GenerationConfig(temperature: 0.0)
        config.maxOutputTokens = maxOutputTokens
        config.grammar = grammar
        return config
    }

    /// Drains `stream`, returning the visible token texts and the terminal
    /// error (if the stream finished by throwing).
    private func drain(_ stream: GenerationStream) async -> (tokens: [String], events: [GenerationEvent], error: Error?) {
        var tokens: [String] = []
        var events: [GenerationEvent] = []
        do {
            for try await event in stream.events {
                events.append(event)
                if case .token(let text) = event { tokens.append(text) }
            }
            return (tokens, events, nil)
        } catch {
            return (tokens, events, error)
        }
    }

    // MARK: - The emit site

    /// The success path, end to end through the Task body: exactly one metric,
    /// carrying the provider/model/promptTokens `generate()` is supposed to
    /// pass and the completion count the driver's `onToken` hook is supposed
    /// to have accumulated.
    ///
    /// This is the test #165 exists for: before the seam, deleting the
    /// `emitMetric(...)` call from `generate()`'s `defer` passed the entire
    /// headless suite.
    func test_generate_emitsExactlyOneMetricOfExpectedShape() async throws {
        let engine = ScriptedLlamaEngine(script: ["Hello", ",", " world"])
        let backend = makeBackend(engine: engine, promptTokens: [1, 2, 3, 4, 5])
        let sink = StubMetricSink()
        backend.metricSink = sink

        let stream = try backend.generate(prompt: "hi", systemPrompt: nil, config: config())
        let (tokens, _, error) = await drain(stream)
        await backend.awaitGenerationSettled()

        XCTAssertNil(error, "a clean scripted turn must not throw")
        XCTAssertEqual(tokens, ["Hello", ",", " world"])

        let received = await sink.waitForRecords(count: 1)
        XCTAssertEqual(received.count, 1,
            "generate() must emit exactly one metric per call — deleting the emitMetric(...) "
            + "call in its defer, or gating it off, lands here")
        let metric = try XCTUnwrap(received.first)
        XCTAssertEqual(metric.provider, "llama",
            "provider must be BackendName.llama.rawValue")
        XCTAssertEqual(metric.model, "scripted-model.gguf",
            "model must come from the loaded manifest, not a literal")
        XCTAssertEqual(metric.promptTokens, 5,
            "promptTokens must be the tokenized prompt's count, not maxTokens or the string length")
        XCTAssertEqual(metric.completionTokens, 3,
            "completionTokens comes from the tracker's onToken hook — a nil onToken: at the "
            + "driver.run(...) call site, a deleted onToken?() in the driver's .token case, or "
            + "a hardcoded metricsEnabled = false all report 0 here")
        XCTAssertNil(metric.errorClass)
        XCTAssertEqual(metric.cachedPromptTokens, 0)
    }

    /// The `metricsEnabled` gate in the other direction: with both sinks
    /// detached, nothing is emitted and the turn still completes normally.
    func test_generate_withNoSinksAttached_emitsNothingAndStillGenerates() async throws {
        let engine = ScriptedLlamaEngine(script: ["a", "b"])
        let backend = makeBackend(engine: engine, promptTokens: [1, 2, 3])
        backend.metricSink = nil
        backend.traceSink = nil

        let stream = try backend.generate(prompt: "hi", systemPrompt: nil, config: config())
        let (tokens, _, error) = await drain(stream)
        await backend.awaitGenerationSettled()

        XCTAssertNil(error)
        XCTAssertEqual(tokens, ["a", "b"], "generation must be unaffected by having no sinks")
        XCTAssertFalse(backend.isGenerating, "the defer must clear isGenerating on the success path")
    }

    /// A trace-sink-only configuration must still start the tracker and fire
    /// the per-token hooks — the trap core's own `SSEGenerationTaskRunner` falls
    /// into (gates emission on either sink but only enables token recording for
    /// the metric sink) and that this backend deliberately does not reproduce.
    func test_generate_withTraceSinkOnly_stillCountsTokens() async throws {
        let engine = ScriptedLlamaEngine(script: ["x", "y", "z", "!"])
        let backend = makeBackend(engine: engine, promptTokens: [7, 8])
        backend.metricSink = nil
        let traceSink = RecordingTraceSink()
        backend.traceSink = traceSink

        let stream = try backend.generate(prompt: "hi", systemPrompt: nil, config: config())
        _ = await drain(stream)
        await backend.awaitGenerationSettled()

        let spans = await waitForSpans(on: traceSink, count: 1)
        XCTAssertEqual(spans.count, 1)
        let span = try XCTUnwrap(spans.first)
        XCTAssertEqual(span.attributes[GenAIAttributeKeys.usageCompletionTokens], .int(4),
            "a trace-only configuration must still record tokens — an all-zeros span means the "
            + "tracker was never started or onToken was never wired")
        XCTAssertEqual(span.attributes[GenAIAttributeKeys.usagePromptTokens], .int(2))
    }

    /// Polls `RecordingTraceSink` until at least `count` spans land or timeout.
    /// `emitMetric(...)` dispatches spans from a fire-and-forget `Task`.
    private func waitForSpans(
        on sink: RecordingTraceSink, count: Int, timeout: Duration = .seconds(5)
    ) async -> [GenSpan] {
        let deadline = ContinuousClock.now + timeout
        var spans = await sink.recordedSpans()
        while spans.count < count, ContinuousClock.now < deadline {
            try? await Task.sleep(for: .milliseconds(10))
            spans = await sink.recordedSpans()
        }
        return spans
    }

    // MARK: - Batch shapes and KV reuse (ManifoldKit#1677)

    /// The prompt is decoded in `batchSize` chunks, only the final token of the
    /// final chunk carries logits, and the generated tokens are decoded one at
    /// a time at consecutive positions starting at `tokens.count`.
    func test_generate_decodesPromptInBatchSizedChunks_withLogitsOnlyOnTheLastToken() async throws {
        let engine = ScriptedLlamaEngine(script: ["ok"], batchSize: 4)
        let backend = makeBackend(engine: engine, promptTokens: [10, 11, 12, 13, 14, 15])

        let stream = try backend.generate(prompt: "hi", systemPrompt: nil, config: config())
        _ = await drain(stream)
        await backend.awaitGenerationSettled()

        XCTAssertEqual(engine.promptChunks, [
            .init(tokens: [10, 11, 12, 13], startPosition: 0, logitsOnLastToken: false),
            .init(tokens: [14, 15], startPosition: 4, logitsOnLastToken: true),
        ], "batch shapes are load-bearing for greedy determinism (ManifoldKit#1677)")
        XCTAssertEqual(engine.kvReusePrefixes, [0], "a cold turn reuses nothing")
        XCTAssertEqual(engine.generatedDecodes, [.init(token: 0, position: 6)])
        XCTAssertEqual(engine.releaseDecodeResourcesCount, 1,
            "the generation batch must be released exactly once per run")
    }

    /// A second turn with the same prompt reuses the shared prefix floored to a
    /// `batchSize` boundary, re-decodes only the final chunk, and surfaces the
    /// full (unaligned) shared-prefix length as the `.kvCacheReuse` signal.
    func test_generate_secondTurn_reusesBatchAlignedPrefixAndReDecodesFinalChunk() async throws {
        let promptTokens: [llama_token] = [10, 11, 12, 13, 14, 15]

        let firstEngine = ScriptedLlamaEngine(script: ["a"], batchSize: 4)
        let backend = makeBackend(engine: firstEngine, promptTokens: promptTokens)
        _ = await drain(try backend.generate(prompt: "hi", systemPrompt: nil, config: config()))
        await backend.awaitGenerationSettled()

        let secondEngine = ScriptedLlamaEngine(script: ["b"], batchSize: 4)
        backend.installGenerationSeamForTesting(LlamaGenerationSeam(
            tokenize: { _, _ in promptTokens },
            makeEngine: { _, _ in secondEngine }
        ))
        let (_, events, _) = await drain(
            try backend.generate(prompt: "hi", systemPrompt: nil, config: config()))
        await backend.awaitGenerationSettled()

        XCTAssertEqual(secondEngine.kvReusePrefixes, [4],
            "reuse must be floored to a batchSize boundary and capped below the final chunk")
        XCTAssertEqual(secondEngine.promptChunks, [
            .init(tokens: [14, 15], startPosition: 4, logitsOnLastToken: true),
        ], "the chunk producing the N-1 logits must be re-decoded with the cold turn's shape")

        let reuseSignals: [Int] = events.compactMap {
            if case .kvCacheReuse(let reused) = $0 { return reused }
            return nil
        }
        XCTAssertEqual(reuseSignals, [6],
            ".kvCacheReuse reports the full detected shared prefix, not the aligned one")
    }

    // MARK: - Failure wiring

    /// A prompt-decode failure must reach `finishDecodeFailure`, fire
    /// `onError("decodeFailed")` through the hook `generate()` installs, throw
    /// on the stream, and — via the `applyKVCoherence(false)` call at the end
    /// of the Task body — discard the cached KV prefix.
    func test_generate_promptDecodeFailure_recordsErrorClassAndClearsSessionKV() async throws {
        let engine = ScriptedLlamaEngine(script: ["never"], promptDecodeStatuses: [1])
        let backend = makeBackend(engine: engine, promptTokens: [1, 2, 3])
        let sink = StubMetricSink()
        backend.metricSink = sink

        let stream = try backend.generate(prompt: "hi", systemPrompt: nil, config: config())
        let (tokens, _, error) = await drain(stream)
        await backend.awaitGenerationSettled()

        XCTAssertTrue(tokens.isEmpty)
        XCTAssertNotNil(error, "a failed decode must surface as a thrown error, not a silent end")

        let records = await sink.waitForRecords(count: 1)
        let metric = try XCTUnwrap(records.first)
        XCTAssertEqual(metric.errorClass, "decodeFailed",
            "onError: must be wired at the driver.run(...) call site")
        XCTAssertEqual(metric.completionTokens, 0)
        XCTAssertNil(backend.sessionKVTokenCountForTesting,
            "an incoherent KV cache must clear sessionKVState — this is applyKVCoherence(kvCoherent) "
            + "at the end of generate()'s Task body")
    }

    /// Sampler-chain allocation failure — `LlamaGenerationDriver.swift`'s
    /// `.chainInitFailed` branch, unreachable with a real model short of an
    /// allocation failure (#165 AC 3).
    func test_generate_samplerInitFailure_reportsSamplerInitFailed() async throws {
        let engine = ScriptedLlamaEngine(script: ["never"], samplerBehavior: .chainInitFailed)
        let backend = makeBackend(engine: engine, promptTokens: [1, 2, 3])
        let sink = StubMetricSink()
        backend.metricSink = sink

        let stream = try backend.generate(prompt: "hi", systemPrompt: nil, config: config())
        let (_, _, error) = await drain(stream)
        await backend.awaitGenerationSettled()

        XCTAssertNotNil(error)
        XCTAssertTrue(engine.promptChunks.isEmpty, "no decode may run once the sampler failed")
        let records = await sink.waitForRecords(count: 1)
        let metric = try XCTUnwrap(records.first)
        XCTAssertEqual(metric.errorClass, "samplerInitFailed")
        XCTAssertNil(backend.sessionKVTokenCountForTesting,
            "the driver returns kvCoherent == false for a sampler-init failure")
    }

    /// GBNF parse failure — the `.grammarParseFailed` branch. Distinct from
    /// `.chainInitFailed` in both label and KV-coherence verdict: no decode has
    /// run, so the cached prefix stays valid.
    func test_generate_grammarParseFailure_reportsGrammarParseFailedAndKeepsKVPrefix() async throws {
        let engine = ScriptedLlamaEngine(script: ["never"], samplerBehavior: .grammarParseFailed)
        let backend = makeBackend(engine: engine, promptTokens: [1, 2, 3])
        let sink = StubMetricSink()
        backend.metricSink = sink

        let stream = try backend.generate(
            prompt: "hi", systemPrompt: nil, config: config(grammar: "root ::= \"x\""))
        let (_, _, error) = await drain(stream)
        await backend.awaitGenerationSettled()

        XCTAssertNotNil(error)
        XCTAssertEqual(engine.samplerRequests, [true], "the strict chain carries the grammar")
        let records = await sink.waitForRecords(count: 1)
        let metric = try XCTUnwrap(records.first)
        XCTAssertEqual(metric.errorClass, "grammarParseFailed")
        XCTAssertEqual(backend.sessionKVTokenCountForTesting, 3,
            "no decode ran, so the KV cache is still coherent and the prefix must survive")
    }

    /// The lifecycle race in `generate()`'s Task body: the backend is unloaded
    /// before the Task re-reads the pointers, so the driver never runs. The
    /// metric must still be emitted, tagged `backendUnloaded`.
    func test_generate_unloadedBeforeTaskRuns_recordsBackendUnloaded() async throws {
        let engine = ScriptedLlamaEngine(script: ["never"])
        let backend = makeBackend(engine: engine, promptTokens: [1, 2, 3])
        let sink = StubMetricSink()
        backend.metricSink = sink

        let stream = try backend.generate(prompt: "hi", systemPrompt: nil, config: config())
        // Clears context/vocab without any C call, exactly as unloadModel()
        // would have from the caller's point of view.
        backend.disarmFakeLoadedStateForTesting()
        _ = await drain(stream)
        await backend.awaitGenerationSettled()

        let records = await sink.waitForRecords(count: 1)
        let metric = try XCTUnwrap(records.first)
        XCTAssertEqual(metric.errorClass, "backendUnloaded")
        XCTAssertTrue(engine.promptChunks.isEmpty)
    }
}

// MARK: - Test doubles

/// Records every metric handed to `record(_:)`, in order. Mirrors the
/// same-named actor in `LlamaBackendMetricSinkTests` /
/// `LlamaBackendMetricSinkE2ETests` — duplicated rather than shared for the
/// same reason they duplicate each other: each is `private` to its own file so
/// suites gated differently never couple at compile time.
private actor StubMetricSink: InferenceMetricSink {
    private(set) var received: [InferenceMetric] = []

    func record(_ metric: InferenceMetric) async {
        received.append(metric)
    }

    /// Polls until at least `count` metrics have been recorded or `timeout`
    /// elapses. `emitMetric(...)` dispatches via a fire-and-forget `Task`, so
    /// there is no synchronous happens-before edge back to the caller.
    func waitForRecords(count: Int, timeout: Duration = .seconds(5)) async -> [InferenceMetric] {
        let deadline = ContinuousClock.now + timeout
        while received.count < count, ContinuousClock.now < deadline {
            try? await Task.sleep(for: .milliseconds(10))
        }
        return received
    }
}
