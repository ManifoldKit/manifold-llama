import XCTest
import ManifoldInference
import ManifoldTestSupport
import ManifoldLlama
@_spi(Testing) import ManifoldLlama
@_spi(BackendInternals) import ManifoldContract

/// Tests for ``LlamaBackend``'s ``CancellableModelLoading`` conformance.
///
/// Split into two lanes:
///
/// **Headless** — run on any Apple Silicon physical device (no GGUF required).
/// Cover the protocol's invariants at the flag-plumbing level: initial state,
/// fast-path settle, and no-op cancel.
///
/// **Model-gated** — additionally require a GGUF on disk. Cover the live
/// cooperative-cancel path: request cancel while the C load is in flight,
/// verify `CancellationError` is thrown, `awaitModelLoadSettled()` returns
/// only after the native work unwinds, `isModelLoadInFlight` is false at
/// return, and a subsequent load + generate succeeds with no corrupted context.
///
/// All tests require Apple Silicon (LlamaBackend uses Metal). Load tests are
/// NOT run concurrently — each creates its own `LlamaBackend` and awaits
/// `unloadAndWait()` in teardown to drain detached cleanup tasks.
final class LlamaLoadCancelTests: XCTestCase {

    override func setUp() async throws {
        try await super.setUp()
        try XCTSkipUnless(
            HardwareRequirements.isPhysicalDevice,
            "LlamaBackend requires Metal (unavailable in simulator)"
        )
        try XCTSkipUnless(
            HardwareRequirements.isAppleSilicon,
            "LlamaBackend requires Apple Silicon"
        )
    }

    // MARK: - Headless

    func test_isModelLoadInFlight_falseByDefault() {
        let backend = LlamaBackend()
        XCTAssertFalse(backend.isModelLoadInFlight, "No load started — flag must be false")
    }

    func test_cancelModelLoad_isNoOp_whenNotLoading() {
        let backend = LlamaBackend()
        // Must not crash; flag must remain false.
        backend.cancelModelLoad()
        XCTAssertFalse(backend.isModelLoadInFlight)
    }

    func test_awaitModelLoadSettled_returnsImmediately_whenIdle() async {
        let backend = LlamaBackend()
        // Fast path: nothing in flight → returns without suspending.
        await backend.awaitModelLoadSettled()
        XCTAssertFalse(backend.isModelLoadInFlight)
    }

    func test_conformsToCancellableModelLoading() {
        let backend = LlamaBackend()
        // The cast is the assertion: LlamaBackend must conform at runtime.
        XCTAssertNotNil(backend as? any CancellableModelLoading,
                        "LlamaBackend must conform to CancellableModelLoading")
    }

    // MARK: - Model-gated: cooperative cancel

    /// Starts a real GGUF load, requests cooperative cancel while it is in
    /// flight, and asserts that:
    ///   1. `cancelModelLoad()` causes the in-flight load to throw `CancellationError`.
    ///   2. `awaitModelLoadSettled()` returns only after the native work unwinds.
    ///   3. `isModelLoadInFlight` is `false` the instant `awaitModelLoadSettled()` returns.
    func test_cancelModelLoad_abortsInFlightLoad_andSettles() async throws {
        guard let modelURL = HardwareRequirements.findGGUFModel() else {
            throw XCTSkip("No GGUF model on disk — set LLAMA_TEST_MODEL or place a .gguf in ~/Documents/Models/.")
        }

        let backend = LlamaBackend()
        addTeardownBlock { await backend.unloadAndWait() }

        // Launch the load as a Task<Result<Void, any Error>, Never> so we can
        // collect the outcome without a mutable capture (Swift 6 strict concurrency).
        let loadTask = Task<Result<Void, any Error>, Never> {
            do {
                try await backend.loadModel(
                    from: modelURL,
                    plan: .testStub(effectiveContextSize: 512)
                )
                return .success(())
            } catch {
                return .failure(error)
            }
        }

        // Wait until the native load is truly in flight (progress callback has fired
        // at least once, setting isModelLoadInFlight = true), then request cancel.
        // Tight poll with a generous outer deadline so we don't spin forever on fast SSD.
        let deadline = Date().addingTimeInterval(30)
        while !backend.isModelLoadInFlight && Date() < deadline {
            try await Task.sleep(nanoseconds: 2_000_000)  // 2 ms
        }

        guard backend.isModelLoadInFlight else {
            // Model loaded before we could observe the in-flight window —
            // this is valid (fast SSD, tiny model). Skip rather than fail.
            _ = await loadTask.value
            throw XCTSkip("Model loaded before in-flight window was observable — fast-load path, not a failure.")
        }

        // Request cooperative cancel. The progress callback will return false on
        // its next invocation, instructing llama_model_load_from_file to abort.
        backend.cancelModelLoad()

        // Collect the load outcome after cancel.
        let loadResult = await loadTask.value

        // Await the native work truly settling. This is the key contract:
        // awaitModelLoadSettled() must not return while the native load is
        // still mutating state on the background thread.
        await backend.awaitModelLoadSettled()

        // isModelLoadInFlight must be false the instant settle returns.
        XCTAssertFalse(
            backend.isModelLoadInFlight,
            "isModelLoadInFlight must be false the moment awaitModelLoadSettled() returns"
        )

        if case .failure(let error) = loadResult {
            // The cancel took effect during the load — verify it threw CancellationError,
            // not some unrelated load failure.
            XCTAssertTrue(
                error is CancellationError,
                "cancelled load must throw CancellationError; got: \(error)"
            )
            XCTAssertFalse(backend.isModelLoaded, "a cancelled load must not mark the model loaded")
        }
        // .success means the model loaded before cancel arrived — that's fine;
        // awaitModelLoadSettled() still verified the in-flight contract above.
    }

    /// After a cooperative cancel, a subsequent `loadModel` must succeed and
    /// `generate` must produce output — proving no corrupted half-built context
    /// from the prior aborted load poisons the next load cycle.
    func test_cancelledLoad_followedBySuccessfulLoadAndGenerate() async throws {
        guard let modelURL = HardwareRequirements.findGGUFModel() else {
            throw XCTSkip("No GGUF model on disk — set LLAMA_TEST_MODEL or place a .gguf in ~/Documents/Models/.")
        }

        let backend = LlamaBackend()
        addTeardownBlock { await backend.unloadAndWait() }

        // ── Phase 1: attempt a cancel ────────────────────────────────────────
        let firstLoadTask = Task<Void, any Error> {
            try await backend.loadModel(
                from: modelURL,
                plan: .testStub(effectiveContextSize: 512)
            )
        }

        let deadline = Date().addingTimeInterval(30)
        while !backend.isModelLoadInFlight && Date() < deadline {
            try await Task.sleep(nanoseconds: 2_000_000)
        }

        if backend.isModelLoadInFlight {
            backend.cancelModelLoad()
        }

        _ = try? await firstLoadTask.value
        await backend.awaitModelLoadSettled()
        XCTAssertFalse(backend.isModelLoadInFlight, "phase-1 must have settled")

        // ── Phase 2: reload and generate ────────────────────────────────────
        // Load a fresh model into the same backend. The RAII handles on the
        // aborted phase-1 load are already freed (LlamaModelHandle/ContextHandle
        // deinits ran when initializeModel threw). A subsequent load must work
        // from a clean slate.
        try await backend.loadModel(
            from: modelURL,
            plan: .testStub(effectiveContextSize: 512)
        )
        XCTAssertTrue(backend.isModelLoaded, "phase-2 load must succeed")
        XCTAssertFalse(backend.isModelLoadInFlight, "no load in flight after phase-2 completes")

        // Generate a token to confirm the context is sound (not half-built from phase 1).
        let stream = try backend.generate(
            prompt: "<bos>Hello",
            systemPrompt: nil,
            config: GenerationConfig(temperature: 0, maxOutputTokens: 1)
        )
        var receivedToken = false
        for try await event in stream.events {
            if case .token = event { receivedToken = true }
        }
        XCTAssertTrue(receivedToken, "generate after cancelled-then-reloaded backend must produce a token")
    }
}
