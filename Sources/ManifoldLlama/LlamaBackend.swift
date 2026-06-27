import Foundation
import LlamaSwift
import os
import Synchronization
import ManifoldInference
// BackendInternals SPI: MemoryPressureHandler (ManifoldHardware) and
// HeuristicTokenizer (ManifoldContract) are part of the frozen backend seam
// published for the companion family packages (#1749).
@_spi(BackendInternals) import ManifoldHardware
@_spi(BackendInternals) import ManifoldContract

/// llama.cpp inference backend for GGUF-format models.
///
/// Uses the llama.cpp C API via `mattt/llama.swift` (pre-built xcframework).
/// Models are loaded from local `.gguf` files. Prompt formatting is handled
/// externally by `InferenceService` using the detected `PromptTemplate`.
public final class LlamaBackend: InferenceBackend, @unchecked Sendable {

    // MARK: - Logging

    private static let logger = Logger(
        subsystem: ManifoldConfiguration.shared.logSubsystem,
        category: "inference"
    )

    // MARK: - State

    public private(set) var isModelLoaded = false
    public private(set) var isGenerating = false

    // MARK: - CancellableModelLoading state (guarded by stateLock)

    /// True from the moment the detached native load task starts mutating
    /// backend state until it truly finishes (success, failure, or cooperative
    /// cancel). Distinct from the `async loadModel` await: a host's deadline
    /// can fire while the native load is still running, so `isModelLoadInFlight`
    /// stays `true` even after `loadModel` throws `CancellationError`.
    private var _isModelLoadInFlight = false

    /// The detached Task that drives the in-flight native load. Stored so
    /// `awaitModelLoadSettled()` can suspend until the task truly finishes
    /// regardless of whether `loadModel`'s own await has already resumed.
    private var _activeLoadTask: Task<LlamaModelLoader.LoadedResources, Error>?

    // MARK: - Locking

    /// Guards mutable runtime state and pending async lifecycle work.
    /// These values are read from @MainActor callers and written from detached
    /// tasks that may outlive the initiating method call.
    private let stateLock = NSLock()

    private func withStateLock<T>(_ body: () -> T) -> T {
        stateLock.lock()
        defer { stateLock.unlock() }
        return body()
    }

    // MARK: - Capabilities

    private var _effectiveContextSize: Int32 = 4096

    /// Manifest captured at the most recent successful load. Mirrors
    /// ``_effectiveContextSize`` and ``_autoDetectedThinkingMarkers`` in a
    /// single structured value so consumers (``ContextWindowManager``, the
    /// conformance harness) can introspect the loaded model uniformly.
    /// Guarded by `stateLock`.
    private var _manifest: ModelManifest?

    public var manifest: ModelManifest? { withStateLock { _manifest } }

    /// `general.architecture` declared by the most recently loaded GGUF (e.g.
    /// `llama`, `qwen2`, `gemma3`), or `nil` before any load. Drives
    /// architecture-family capability gating (see ``capabilities``).
    /// Guarded by `stateLock`.
    private var _architecture: String?

    /// Injects a GGUF `general.architecture` string for unit tests that need to
    /// assert on capability flags gated by model architecture (e.g. Gemma
    /// grammar detection) without performing a real GGUF load.
    ///
    /// Accessible via `@testable import ManifoldLlama`. Never call this in
    /// production code — it bypasses the normal `loadModel` lifecycle.
    @_spi(Testing) public func injectArchitectureForTesting(_ architecture: String) {
        withStateLock { _architecture = architecture }
    }

    // MARK: - Context-window preflight

    /// Pure predicate behind `generate()`'s `.contextExhausted` preflight: does a
    /// prompt of `promptTokens` plus `maxOutputTokens` of generation headroom fit
    /// inside `contextSize`?
    ///
    /// Extracted as a model-free static so the `< vs <=` boundary can be unit-tested
    /// without a loaded vocab/context (issue #27). The relation must be `<=`: a
    /// prompt that exactly fills the window (`promptTokens + maxOutputTokens ==
    /// contextSize`) is still serviceable — every prompt token and every requested
    /// output token has a KV slot. A `<` slip would spuriously reject that exact-fit
    /// case, and callers that retry on `.contextExhausted` would loop or truncate
    /// needlessly.
    @_spi(Testing) public static func contextWindowFits(
        promptTokens: Int, maxOutputTokens: Int, contextSize: Int
    ) -> Bool {
        promptTokens + maxOutputTokens <= contextSize
    }

    /// Injects a llama.cpp `vocab` pointer for unit tests that exercise the
    /// production tokenization callsites (``tokenCount(_:)`` /
    /// ``countTokens(_:)``) against a fixture vocabulary loaded with
    /// `vocab_only`, without a full Metal-backed ``loadModel(from:plan:)``.
    ///
    /// The caller owns the underlying `llama_model` and must keep it alive for
    /// the duration of the test (the vocab pointer is borrowed, not retained).
    /// Accessible via `@_spi(Testing) import ManifoldLlama`. Never call this in
    /// production code — it bypasses the normal `loadModel` lifecycle and the
    /// vocab-lifetime ownership the loader otherwise guarantees.
    @_spi(Testing) public func injectVocabForTesting(_ vocab: OpaquePointer?) {
        withStateLock { self.vocab = vocab }
    }

    /// Per-token resident cost (bytes) learned from the most recent prefill via
    /// ``PrefillFootprintEstimator`` (issue #1592), or `nil` if no prefill has
    /// produced a stable sample yet. Callers that rebuild a ``ModelLoadPlan`` for
    /// the loaded model can pass this as `measuredBytesPerToken` so the plan
    /// recomputes `effectiveContextSize` against a measured budget instead of the
    /// static heuristic. Guarded by `stateLock`.
    private var _lastMeasuredBytesPerToken: UInt64?

    public var lastMeasuredBytesPerToken: UInt64? { withStateLock { _lastMeasuredBytesPerToken } }

    /// Token usage reported by the most recently completed local generation turn,
    /// or `nil` before the first successful turn. Populated by the `onUsage`
    /// callback fired by `LlamaGenerationDriver.run()` just before it finishes the
    /// stream. Guarded by `stateLock`.
    private var _lastUsage: (promptTokens: Int, completionTokens: Int)?

    public var capabilities: BackendCapabilities {
        let ctxSize = withStateLock { _effectiveContextSize }
        let architecture = withStateLock { _architecture }
        // Gemma emits malformed/truncated output under structured (JSON-object)
        // GBNF grammars: it opens the object then gets trapped emitting
        // whitespace until EOG, never completing the structure — so extraction
        // yields nothing and the FiresideMemory pipeline heuristic-fallbacks with
        // 0 entities. Trivial grammars (e.g. `[0-9]+`) are unaffected, and the
        // identical grammar produces valid JSON on Llama; the failure is
        // Gemma-specific. Targeting only complex grammars isn't worth the
        // complexity, so disable grammar wholesale for the Gemma family and fall
        // through to JSON-mode-only parsing, which works correctly. Detect by
        // declared GGUF architecture (gemma / gemma2 / gemma3); an unloaded
        // backend (architecture == nil) keeps grammar enabled — the pre-load
        // default for non-Gemma callers.
        let supportsGrammar = !(architecture?.lowercased().hasPrefix("gemma") ?? false)
        // MLX: KV cache reuse deferred — MLX manages its own context lifecycle via
        // MLXModelContainer and does not expose a KV-trim API.
        return BackendCapabilities(
            supportedParameters: [
                .temperature, .topP, .topK, .repeatPenalty,
                .minP, .repetitionPenalty, .presencePenalty, .frequencyPenalty,
                .llamaDRY, .llamaXTC, .llamaMirostatV2,
            ],
            maxContextTokens: ctxSize,
            requiresPromptTemplate: true,
            supportsSystemPrompt: true,
            supportsToolCalling: true,
            supportsStructuredOutput: false,
            supportsNativeJSONMode: false,
            cancellationStyle: .explicit,
            supportsTokenCounting: true,
            memoryStrategy: .mappable,
            maxOutputTokens: 4096,
            supportsStreaming: true,
            isRemote: false,
            supportsKVCachePersistence: true,
            supportsGrammarConstrainedSampling: supportsGrammar,
            supportsThinking: true,
            supportsVision: BackendVisionCapability.llamaSupportsImageInput,
            supportsParallelToolCalls: false,
            toolDialect: LlamaToolCallDialect.infer(from: architecture)
        )
    }

    // MARK: - Private

    private var model: OpaquePointer?
    private var context: OpaquePointer?
    /// Accessible to tests via `@_spi(Testing)` for vocabulary-level assertions.
    @_spi(Testing) public internal(set) var vocab: OpaquePointer?
    private var generationTask: Task<Void, Never>?
    /// Cancellation flag shared between the decode loop (background task) and
    /// `stopGeneration()` / `unloadModel()` (any thread/actor).
    ///
    /// `Atomic<Bool>` (Swift 6 `Synchronization` stdlib) makes every read and
    /// write sequentially consistent without requiring a lock, eliminating the
    /// data race that existed when a plain `Bool` was written from the main actor
    /// and read on a detached background task. This is also safe to write from
    /// a memory-pressure handler callback (#415) running on an arbitrary thread.
    private let cancelled = Atomic<Bool>(false)
    private var cleanupTask: Task<Void, Never>?
    private var nextLoadToken: UInt64 = 0
    private var activeLoadToken: UInt64 = 0

    /// Guarded by `stateLock`. Set by `setLoadProgressHandler(_:)` before each load.
    private var _loadProgressHandler: (@Sendable (Double) async -> Void)?

    /// Backend tuning knobs applied at the next ``loadModel(from:plan:)`` call.
    /// Guarded by `stateLock`. Set via ``setLoadOptions(_:)``; defaults preserve
    /// historical behaviour bit-for-bit.
    private var _loadOptions: BackendLoadOptions = .default

    /// Test-only read-side accessor that snapshots `_loadOptions` under the
    /// state lock. Lets plumbing tests assert the setter persisted the value
    /// without needing a real model load.
    @_spi(Testing) public var loadOptionsForTesting: BackendLoadOptions { withStateLock { _loadOptions } }

    // MARK: - Test seams (no live model required)

    /// Snapshot of the `cancelled` atomic. Lets a test assert that the
    /// memory-pressure callback (or `stopGeneration()`) actually flipped the flag
    /// without driving a real decode loop.
    @_spi(Testing) public var isCancelledForTesting: Bool {
        cancelled.load(ordering: .sequentiallyConsistent)
    }

    /// Fires the registered memory-pressure callback synchronously so tests can
    /// exercise the `.warning` / `.critical` / `.nominal` dispatch body
    /// (`registerMemoryPressureCallback`) on an unloaded backend.
    @_spi(Testing) public func simulateMemoryPressure(_ level: MemoryPressureLevel) {
        memoryPressure.fireCallbacks(level: level)
    }

    /// Seeds a synthetic `sessionKVState` so tests can observe that
    /// `secureWipe()` / `resetConversation()` actually clear the cached prefix —
    /// the field is otherwise only populated after a real decode.
    @_spi(Testing) public func seedSessionKVStateForTesting(tokenCount: Int) {
        withStateLock {
            sessionKVState = SessionKVState(
                tokens: Array(repeating: llama_token(0), count: max(0, tokenCount)))
        }
    }

    /// Token count of the cached `sessionKVState`, or `nil` when it has been
    /// cleared. Read side for the seed seam above.
    @_spi(Testing) public var sessionKVTokenCountForTesting: Int? {
        withStateLock { sessionKVState?.tokens.count }
    }

    /// Applies the post-decode KV-coherence guard: when the driver reports the KV
    /// cache is incoherent (a decode failed), the cached prefix must be discarded so
    /// the next turn does not reuse positions that were never coherently decoded.
    /// A coherent decode leaves `sessionKVState` untouched.
    ///
    /// Factored out of the `generate()` task so the guard can be exercised headlessly
    /// — `kvCoherent` is otherwise always `true` in tests because no fake context can
    /// force a real `llama_decode` failure.
    private func applyKVCoherence(_ kvCoherent: Bool) {
        if !kvCoherent {
            withStateLock { sessionKVState = nil }
        }
    }

    /// Headless seam over ``applyKVCoherence(_:)`` so the post-decode KV-coherence
    /// guard can be tested without a live decode loop. Seed a synthetic prefix with
    /// ``seedSessionKVStateForTesting(tokenCount:)``, drive this with `false`, and
    /// assert ``sessionKVTokenCountForTesting`` cleared to `nil`; driving it with
    /// `true` must leave the prefix intact.
    @_spi(Testing) public func applyKVCoherenceForTesting(_ kvCoherent: Bool) {
        applyKVCoherence(kvCoherent)
    }

    /// Installs sentinel (non-nil) model/context/vocab pointers and sets
    /// `isModelLoaded = true` so headless tests can reach guards that fire
    /// BEFORE any C API call is made on those pointers.
    ///
    /// The sentinel value (`OpaquePointer(bitPattern: 1)!`) is never passed to
    /// any llama.cpp C function in the guarded paths — the `alreadyGenerating`
    /// guard fires before tokenization, which is the first site that
    /// dereferences a pointer. Calling any C API on this backend after arming
    /// this seam will crash.
    ///
    /// The sentinel is a bit-pattern integer cast, not a heap allocation, so
    /// there is nothing to free. Not calling `llama_free` / `llama_free_model`
    /// on it is intentional — `unloadModel()` would crash if it tried to free
    /// address 1.
    ///
    /// ONLY call this from test targets. Never call in production code.
    @_spi(Testing) public func armFakeLoadedStateForTesting() {
        withStateLock {
            let sentinel = OpaquePointer(bitPattern: 1)!
            model        = sentinel
            context      = sentinel
            vocab        = sentinel
            isModelLoaded = true
        }
    }

    /// Clears the sentinel state armed by ``armFakeLoadedStateForTesting()`` WITHOUT
    /// invoking any llama.cpp C API. A test that arms fake state MUST disarm before
    /// the backend deinits: otherwise `deinit` → ``unloadModel()`` captures the addr-1
    /// sentinel `context` and schedules a `Task.detached` that calls
    /// `llama_synchronize(ctx)` / `llama_free(ctx)` on it — dereferencing address 1
    /// and crashing the process with SIGSEGV. Because that cleanup is detached, the
    /// crash surfaces asynchronously (often after the test bundle reports all tests
    /// passed), making it a flaky exit-time segfault. Niling the pointers here makes
    /// `unloadModel()`'s `capturedContext != nil` guard early-return with no C call.
    /// See #54.
    ///
    /// ONLY call this from test targets. Never call in production code.
    @_spi(Testing) public func disarmFakeLoadedStateForTesting() {
        withStateLock {
            model        = nil
            context      = nil
            vocab        = nil
            isModelLoaded = false
        }
    }

    /// Directly sets `isGenerating` under `stateLock`. Lets headless tests put the
    /// backend into a simulated mid-generation state so the `alreadyGenerating` guard
    /// can be exercised without a real decode loop.
    ///
    /// ONLY call this from test targets. Never call in production code.
    @_spi(Testing) public func setIsGeneratingForTesting(_ value: Bool) {
        withStateLock { isGenerating = value }
    }

    /// Snapshot of the structured history most recently supplied via
    /// ``StructuredHistoryReceiver/setStructuredHistory(_:)``. Lets headless tests
    /// assert that the value was stored without running a real decode.
    @_spi(Testing) public var structuredHistoryForTesting: [StructuredMessage] {
        withStateLock { _structuredHistory }
    }

    // MARK: - Multimodal Projector

    /// URL of the mmproj companion file, set by ``MultimodalProjectorConfigurable`` before each load.
    ///
    /// Guarded by `stateLock`. Non-nil when a vision-capable model's projector is staged for load.
    /// Cleared by ``unloadModel()``. The current vendored LlamaSwift xcframework does not expose
    /// the CLIP / mtmd C APIs needed to turn images into embeddings, so ``capabilities`` continues
    /// to advertise `supportsVision = false` until that binding exists.
    private var _mmprojURL: URL?

    /// Structured history set by ``StructuredHistoryReceiver``. Guarded by `stateLock`.
    /// Used in ``generate(prompt:systemPrompt:config:)`` to detect image parts in the current turn.
    private var _structuredHistory: [StructuredMessage] = []

    /// Owns the serialized model-load path and the C-level parameter/progress-callback bridging.
    private let modelLoader = LlamaModelLoader()

    // MARK: - KV Cache State

    /// Captures the full prompt token sequence of the most recently completed
    /// decode so the next turn can skip re-decoding a shared prefix.
    private struct SessionKVState {
        /// Full token array of the last successfully completed prompt decode.
        var tokens: [llama_token]
    }

    /// Guarded by `stateLock`. Non-nil after a successful prompt decode; nil after reset.
    private var sessionKVState: SessionKVState?

    /// Thinking-marker pair auto-detected from the GGUF's `tokenizer.chat_template`.
    /// `nil` when the model is not loaded, the chat template is missing, or no
    /// known marker pair was found in the template. `GenerationConfig.thinkingMarkers`
    /// always overrides this — see the generate path below.
    /// Guarded by `stateLock`.
    private var _autoDetectedThinkingMarkers: ThinkingMarkers?

    // MARK: - Memory Pressure

    /// Monitors OS-level memory pressure so the decode loop can be aborted before
    /// the OS revokes Metal buffers, which would cause llama_decode to dereference
    /// a freed pointer and crash with SIGSEGV / EXC_BAD_ACCESS. See issue #415.
    ///
    /// `LlamaBackend` owns this handler and registers its callback in `init`, so
    /// pressure events are handled here regardless of whether a `ChatViewModel` or
    /// any other higher-level observer is also listening.
    private let memoryPressure = MemoryPressureHandler()

    // MARK: - Init / Deinit

    public init() {
        LlamaBackendProcessLifecycle.retain()
        registerMemoryPressureCallback()
        memoryPressure.startMonitoring()
    }

    deinit {
        memoryPressure.removeCallback(for: self)
        memoryPressure.stopMonitoring()
        unloadModel()
        LlamaBackendProcessLifecycle.release()
    }

    // MARK: - Memory Pressure Wiring

    /// Registers the backend-level memory pressure callback.
    ///
    /// On `.warning`: calls `stopGeneration()` immediately so the decode loop exits
    /// cleanly before the OS escalates. `stopGeneration()` uses `Atomic<Bool>` and is
    /// safe to call from any thread (PR #456).
    ///
    /// On `.critical`: calls `stopGeneration()` AND schedules a `Task.detached` to call
    /// `unloadAndWait()`, releasing Metal buffers before the OS reclaims them forcibly.
    /// `Task.detached` is used explicitly so the task does not inherit any actor isolation
    /// from the GCD callback's execution context, and the GCD callback returns immediately.
    /// A weak capture prevents a retain cycle with the handler's closure storage.
    private func registerMemoryPressureCallback() {
        memoryPressure.addPressureCallback(for: self) { [weak self] level in
            guard let self else { return }
            switch level {
            case .warning:
                Self.logger.warning("Memory pressure: warning — stopping generation to prevent Metal buffer revocation (#415)")
                self.stopGeneration()
            case .critical:
                Self.logger.warning("Memory pressure: critical — stopping generation and scheduling model unload (#415)")
                self.stopGeneration()
                Task.detached { [weak self] in
                    await self?.unloadAndWait()
                }
            case .nominal:
                break
            }
        }
    }

    // MARK: - Model Lifecycle

    /// Plan-aware model load. The plan's ``ModelLoadPlan/effectiveContextSize``
    /// is authoritative — no clamping happens inside llama.cpp's initializer.
    ///
    /// - Precondition: `plan.verdict != .deny`. Callers must check the verdict
    ///   before invoking; the backend assumes the plan is allow/warn.
    public func loadModel(from url: URL, plan: ModelLoadPlan) async throws {
        assert(plan.verdict != .deny,
               "ModelLoadPlan was denied; callers must check verdict before invoking backend")

        unloadModel()
        await waitForPendingCleanup()

        let loadToken = withStateLock {
            nextLoadToken &+= 1
            activeLoadToken = nextLoadToken
            return activeLoadToken
        }

        // Snapshotting the handler here means calling setLoadProgressHandler(nil)
        // mid-load will not cancel in-flight Task callbacks already dispatched by
        // the C progress hook. Stale callbacks become no-ops at the consumer:
        // InferenceService.applyLoadProgress(_:for:) drops values whose request
        // token no longer matches the active loading phase.
        let capturedHandler = withStateLock { _loadProgressHandler }
        let capturedLoadOptions = withStateLock { _loadOptions }
        let effectiveContextSize = Int32(plan.effectiveContextSize)
        let capturedLoadToken = loadToken

        // Store the detached task handle so CancellableModelLoading can observe
        // and await the native load even after this async function has thrown
        // (e.g. the host's deadline fired while the C load was still in flight).
        // The task body sets _isModelLoadInFlight = true at its start and clears
        // it in a defer at its end — keeping both writes on the same thread to
        // eliminate the TOCTOU where a fast-failing task's defer fired before the
        // spawning thread could reach its own withStateLock{_isModelLoadInFlight = true}.
        let loadTask = Task.detached(priority: .userInitiated) { [weak self, modelLoader] in
            // Set in-flight before any work so the flag tracks native load lifetime.
            self?.withStateLock {
                if self?.activeLoadToken == capturedLoadToken {
                    self?._isModelLoadInFlight = true
                }
            }
            defer {
                self?.withStateLock {
                    // Guard: unloadModel() bumps activeLoadToken so a superseded
                    // detached task cannot mistakenly flip the flag after unload.
                    if self?.activeLoadToken == capturedLoadToken {
                        self?._isModelLoadInFlight = false
                        self?._activeLoadTask = nil
                    }
                }
            }
            return try modelLoader.serializedModelLoad(
                at: url,
                effectiveContextSize: effectiveContextSize,
                loadOptions: capturedLoadOptions,
                progressHandler: capturedHandler
            )
        }

        withStateLock {
            _activeLoadTask = loadTask
        }

        let loadedResources = try await loadTask.value

        let didCommit = withStateLock {
            guard activeLoadToken == loadToken else {
                return false
            }
            // steal() transfers ownership; unloadModel's explicit ordered cleanup takes over.
            self.model = loadedResources.model.steal()
            self.context = loadedResources.context.steal()
            self.vocab = loadedResources.vocab
            self.isModelLoaded = true
            self._effectiveContextSize = loadedResources.effectiveContextSize
            self._autoDetectedThinkingMarkers = loadedResources.autoDetectedThinkingMarkers
            self._architecture = loadedResources.architecture
            self._manifest = ModelManifest(
                contextWindow: Int(loadedResources.effectiveContextSize),
                supportsTools: true,
                supportsThinking: loadedResources.autoDetectedThinkingMarkers != nil,
                thinkingMarkers: loadedResources.autoDetectedThinkingMarkers,
                supportsSeed: true,
                supportedSamplingParameters: [
                    .temperature, .topP, .topK, .repeatPenalty,
                    .presencePenalty, .frequencyPenalty,
                ],
                modelIdentifier: url.lastPathComponent,
                producerKind: .local
            )
            return true
        }

        guard didCommit else {
            // loadedResources goes out of scope here. steal() was never called, so
            // LlamaContextHandle/LlamaModelHandle deinits free the C memory automatically.
            throw CancellationError()
        }

        Self.logger.info("Llama backend loaded \(url.lastPathComponent) with context \(loadedResources.effectiveContextSize)")
    }

    // MARK: - Generation

    public func generate(
        prompt: String,
        systemPrompt: String?,
        config: GenerationConfig
    ) throws -> GenerationStream {
        // The Task body re-reads context under stateLock below to avoid the
        // use-after-free window between here and `self.generationTask = task`
        // install. We only need to verify model is loaded up front — the
        // captured pointers are accessed through the re-read, not these
        // outer locals.
        // Reject image-bearing turns before any other state check so the caller
        // gets the actionable xcframework-limitation message regardless of
        // whether a model happens to be loaded. The check is independent of
        // model state — it only inspects the structured history we cached
        // from the coordinator. CLIP / mtmd C APIs are not present in the
        // bundled xcframework (mattt/llama.swift 2.8772.0, llama.cpp build
        // b8772); upgrading to a build that exposes clip.h / mtmd.h will
        // unblock real image embedding.
        let history = withStateLock { _structuredHistory }
        let hasImageParts = history.contains { msg in
            msg.parts.contains {
                if case .image = $0 { return true }
                return false
            }
        }
        if hasImageParts {
            throw InferenceError.inferenceFailure(
                "LlamaBackend: multimodal image inference is not yet available. "
                + "The vendored llama.cpp xcframework (build b8772) does not include "
                + "clip.h / mtmd.h. Upgrade the LlamaSwift dependency to a build that "
                + "exposes those headers to enable image token embedding."
            )
        }

        guard isModelLoaded, context != nil, vocab != nil, model != nil else {
            throw InferenceError.inferenceFailure("No model loaded")
        }
        guard !withStateLock({ isGenerating }) else {
            throw InferenceError.alreadyGenerating
        }

        // Tokenize up front (pure vocab lookup — doesn't touch context KV
        // state) so we can preflight prompt + output against the context
        // window before flipping `isGenerating`. If we failed this check
        // after the flip, callers who retry on `.contextExhausted` would see
        // an unnecessary `.alreadyGenerating` on the next call.
        //
        // Snapshot vocab under stateLock before handing it to LlamaTokenization:
        // without this, the read races with unloadModel() niling `self.vocab`
        // and freeing the backing model — a use-after-free. The outer
        // `vocab != nil` check is advisory (Swift pointer reads are atomic at
        // machine level) but does not survive across the unprotected tokenize().
        guard let preflightVocab = withStateLock({ vocab }) else {
            throw InferenceError.inferenceFailure("No model loaded")
        }
        let tokens = LlamaTokenization.tokenize(prompt, vocab: preflightVocab, addBos: true)
        guard !tokens.isEmpty else {
            throw InferenceError.inferenceFailure("Failed to tokenize prompt")
        }

        let maxTokens = config.maxOutputTokens ?? 2048
        let contextSize = Int(withStateLock { _effectiveContextSize })
        guard Self.contextWindowFits(
            promptTokens: tokens.count, maxOutputTokens: maxTokens, contextSize: contextSize
        ) else {
            throw InferenceError.contextExhausted(
                promptTokens: tokens.count,
                maxOutputTokens: maxTokens,
                contextSize: contextSize
            )
        }

        // Compute how many leading tokens match the previous turn's KV state.
        // We read sessionKVState here (before the isGenerating flip) because
        // stopGeneration() never clears it — only unloadModel() and
        // resetConversation() do — so there is no race between this read and
        // those two paths under stateLock.
        //
        // `reuseLen` is the full count of matching leading tokens. It is surfaced
        // as `.kvCacheReuse(promptTokensReused:)` AND drives genuine prefix reuse —
        // but the driver only reuses the prefix up to a `batchSize` boundary so the
        // chunk producing the sampling-position (N-1) logits keeps a batch shape
        // bit-identical to the first turn's. That is what guarantees greedy
        // determinism across non-Qwen architectures (ManifoldKit#1677): the old
        // -2-cap path (PR #966) could resume mid-chunk, a different batch shape that
        // flips the argmax on near-tied logits for kernels that reduce differently
        // across batch sizes. Batch-aligned reuse keeps O(new-tokens) prefill while
        // remaining deterministic.
        //
        // The KV tail trim (`llama_memory_seq_rm`) and the full-clear-vs-keep
        // decision now both live in the driver, which owns `batchSize`
        // (`llama_n_batch`) and runs inside the generation Task already serialized
        // with unloadModel(). See LlamaGenerationDriver.
        let previousTokens = withStateLock { sessionKVState?.tokens ?? [] }
        let reuseLen = zip(tokens, previousTokens).prefix(while: { $0.0 == $0.1 }).count

        // Reset the cancellation flag and flip isGenerating atomically under the
        // same lock that stopGeneration() holds when it touches generationTask.
        // Keeping both writes inside a single critical section means a concurrent
        // stopGeneration() call that races this startup cannot observe a window
        // where cancelled == false but isGenerating == false (not yet set), which
        // would let it skip the task cancel and leave the loop running uncancelled.
        withStateLock {
            cancelled.store(false, ordering: .sequentiallyConsistent)
            isGenerating = true
            // Optimistically record the current prompt tokens as the new KV state.
            // If generation is cancelled mid-stream, this is still safe: the prefix
            // up to `reuseLen` is intact in the C KV cache, and any tokens decoded
            // beyond that are the start of the new turn's output — not prompt tokens.
            // resetConversation() / unloadModel() will wipe this if needed.
            sessionKVState = SessionKVState(tokens: tokens)
        }
        Self.logger.debug("Llama generate started")

        let (stream, continuation) = AsyncThrowingStream.makeStream(of: GenerationEvent.self)
        let generationStream = GenerationStream(stream)

        // Hold stateLock across Task creation AND generationTask assignment
        // (see install block below). The Task body's first action is a
        // stateLock re-read, which blocks until we release. That guarantees
        // the Task body cannot observe `self.generationTask == nil` when its
        // re-read runs — so unloadModel() always either sees the installed
        // task (and awaits it) or runs entirely before the task's re-read
        // (and nils `self.context`, causing the task to bail).
        stateLock.lock()
        let task = Task { [weak self, generationStream] in
            guard let self else {
                continuation.finish()
                return
            }

            defer {
                self.withStateLock { self.isGenerating = false }
                Self.logger.debug("Llama generate finished")
            }

            // Re-acquire context and vocab under stateLock so we serialize
            // with unloadModel(). The parent installs `generationTask = task`
            // under stateLock below before releasing; that guarantees either:
            //   (a) unloadModel() ran first → `self.context` is nil → we bail
            //       cleanly without touching any freed pointer, or
            //   (b) the parent installed generationTask first → unloadModel()
            //       now observes the task and awaits it before calling
            //       llama_free / llama_model_free on the captured pointers.
            // Performing all context-touching work (KV clear, decode, sample)
            // inside this task keeps it under the lifecycle that
            // unloadModel() already knows how to wait for.
            let pointers = self.withStateLock { () -> (OpaquePointer, OpaquePointer)? in
                guard let ctx = self.context, let voc = self.vocab else { return nil }
                return (ctx, voc)
            }
            guard let (context, vocab) = pointers else {
                continuation.finish()
                return
            }

            let driver = LlamaGenerationDriver()
            // Manual override beats auto-detection. When neither is present,
            // markers stays nil and the driver skips ThinkingTransform entirely.
            let autoDetected = self.withStateLock { self._autoDetectedThinkingMarkers }
            let resolvedMarkers = config.thinkingMarkers ?? autoDetected
            let kvCoherent = await driver.run(
                context: context,
                vocab: vocab,
                tokens: tokens,
                reuseLen: reuseLen,
                maxTokens: maxTokens,
                config: config,
                markers: resolvedMarkers,
                isCancelled: {
                    Task.isCancelled || self.cancelled.load(ordering: .sequentiallyConsistent)
                },
                generationStream: generationStream,
                continuation: continuation,
                onPrefillEstimate: { [self] measured in
                    self.withStateLock { self._lastMeasuredBytesPerToken = measured }
                },
                onUsage: { [self] promptTokens, completionTokens in
                    self.withStateLock {
                        self._lastUsage = (promptTokens: promptTokens, completionTokens: completionTokens)
                    }
                }
            )
            // A decode failure leaves the C KV cache in an undefined state.
            // Clear sessionKVState so the next turn does not attempt prefix reuse
            // against positions that were never coherently decoded.
            self.applyKVCoherence(kvCoherent)
        }

        // Assignment and unlock complete the critical section opened above.
        // unloadModel() will now observe `generationTask` whenever it beats
        // the task body to the lock — or, if unloadModel() ran fully before
        // we acquired the lock, the task body's re-read will see nil context
        // and bail out without touching freed pointers.
        self.generationTask = task
        stateLock.unlock()

        continuation.onTermination = { @Sendable _ in
            task.cancel()
        }

        return generationStream
    }

    // MARK: - Control

    /// Awaits completion of the in-flight generation task, if any.
    ///
    /// `isGenerating` is cleared in the generation task's `defer` block, which
    /// runs *after* the stream's `continuation.finish()` — so a caller that has
    /// just drained `generate(...)`'s stream has no happens-before guarantee that
    /// the flag has flipped back to `false`. Issuing the next `generate(...)`
    /// immediately can therefore race the defer and trip `.alreadyGenerating`.
    ///
    /// Await this between back-to-back generations on the same loaded model when
    /// deterministic readiness matters (the determinism tests, programmatic
    /// regenerate loops). It awaits the same `Task` whose `defer` releases the
    /// guard, so on return `isGenerating == false` is guaranteed without
    /// unloading the model — unlike ``unloadAndWait()``, the loaded context and
    /// KV state are preserved for the next turn.
    public func awaitGenerationSettled() async {
        let task = withStateLock { generationTask }
        await task?.value
    }

    public func stopGeneration() {
        // Set the atomic flag first so the decode loop can break on its very next
        // iteration check — even before the lock is acquired below.
        cancelled.store(true, ordering: .sequentiallyConsistent)
        // Capture and nil-out generationTask under stateLock. generationTask is
        // a mutable var guarded by stateLock everywhere else (generate() assigns
        // it under the lock, unloadModel() captures it under the lock). Accessing
        // it here without the lock would be a data race under TSan.
        let taskToCancel = withStateLock {
            let t = generationTask
            generationTask = nil
            return t
        }
        taskToCancel?.cancel()
    }

    /// Invalidates the KV cache prefix so the next turn starts with a clean
    /// context rather than attempting to reuse state from a prior conversation.
    ///
    /// Call this when the conversation history is cleared or a new session
    /// begins. `stopGeneration()` intentionally does NOT call this — a
    /// cancelled turn preserves the prefix so the model can continue from
    /// where it left off on the next `generate()`.
    public func resetConversation() {
        // Run the C KV flush inside stateLock alongside the sessionKVState reset.
        // Snapshotting `context` and releasing the lock before calling
        // llama_memory_clear() would let a concurrent unloadModel()/secureWipe()
        // free the context between the read and the C call (use-after-free).
        withStateLock {
            sessionKVState = nil
            // Also flush the actual KV cache in the C context so any leftover
            // positional state is gone before the next turn decodes from position 0.
            if let ctx = context, let mem = llama_get_memory(ctx) {
                llama_memory_clear(mem, false)
            }
        }
    }

    /// Zeros the KV tensor data in the active llama.cpp context.
    ///
    /// Unlike ``resetConversation()``, which passes `false` to
    /// `llama_memory_clear` (metadata-only clear), this passes `true` to
    /// write zeros into the underlying key and value matrices. This closes the
    /// window during which prior-session KV tensors remain in process memory.
    ///
    /// The KV state cache pointer is also nil-ed so the next
    /// ``generate(_:config:)`` call starts with a fresh context.
    public func secureWipe() {
        // C KV zeroing runs inside stateLock with the sessionKVState reset for
        // the same use-after-free reason as resetConversation(): teardown frees
        // `context` under this lock.
        withStateLock {
            sessionKVState = nil
            if let ctx = context, let mem = llama_get_memory(ctx) {
                // true = zero the actual KV tensor data (key + value matrices),
                // not just the positional/sequence metadata.
                llama_memory_clear(mem, true)
            }
        }
    }

    public func unloadModel() {
        // Signal the decode loop to stop before acquiring stateLock. The atomic
        // write is visible to the background task immediately, so the loop can
        // break on its next iteration check without waiting for the lock.
        cancelled.store(true, ordering: .sequentiallyConsistent)
        stateLock.lock()
        nextLoadToken &+= 1
        activeLoadToken = nextLoadToken

        let previousCleanup = cleanupTask
        cleanupTask = nil
        let capturedTask = generationTask
        let capturedContext = context
        let capturedModel = model

        // Clear state immediately so callers see the backend as unloaded
        // without waiting for C memory deallocation.
        generationTask = nil
        context = nil
        model = nil
        vocab = nil
        isModelLoaded = false
        isGenerating = false
        sessionKVState = nil
        _autoDetectedThinkingMarkers = nil
        _manifest = nil
        _mmprojURL = nil
        _structuredHistory = []
        _lastUsage = nil
        // An in-flight load is superseded: clear the in-flight flag immediately
        // so callers see the unloaded state. _activeLoadTask is intentionally
        // NOT cleared here — awaitModelLoadSettled() reads it to suspend until
        // the native C work truly finishes (the detached task may still be
        // running). The task's own defer skips clearing (bumped activeLoadToken)
        // so the reference stays valid until the next loadModel overwrites it.
        _isModelLoadInFlight = false
        stateLock.unlock()

        capturedTask?.cancel()

        Self.logger.info("Llama backend unloaded")

        guard capturedTask != nil || capturedContext != nil || capturedModel != nil else {
            withStateLock {
                cleanupTask = previousCleanup
            }
            return
        }

        LlamaBackendProcessLifecycle.retain()

        // Defer llama_free off the calling thread — InferenceService is @MainActor,
        // so blocking here would freeze the UI for the duration of the spin-wait.
        // We await the generation task to ensure the C loop has stopped before
        // touching the pointers, preventing a use-after-free crash.
        //
        // Before calling llama_free we must:
        //   1. Drain GPU work via llama_synchronize — the generation loop may have
        //      been cancelled mid-stride, leaving Metal command buffers enqueued
        //      but not yet committed. llama_free releases the Metal residency set
        //      while those buffers still reference it, tripping:
        //        GGML_ASSERT([rsets->data count] == 0)   (ggml-metal-device.m:618)
        //      which aborts the process with SIGABRT (issue #1394).
        //   2. Clear the KV cache — ensures ggml_metal_device_free finds an empty
        //      residency set and does not assert on leftover context allocations.
        let newCleanupTask = Task.detached(priority: .utility) {
            await previousCleanup?.value
            await capturedTask?.value
            if let ctx = capturedContext {
                // Synchronize before clearing: llama_memory_clear enqueues Metal
                // ops internally; calling it before the GPU drains would race.
                llama_synchronize(ctx)
                if let mem = llama_get_memory(ctx) {
                    llama_memory_clear(mem, false)
                }
                // Second synchronize to drain the KV-clear Metal pass before free.
                llama_synchronize(ctx)
                llama_free(ctx)
            }
            if let mdl = capturedModel { llama_model_free(mdl) }
            LlamaBackendProcessLifecycle.release()
        }
        withStateLock {
            self.cleanupTask = newCleanupTask
        }
    }

    /// Schedules the same tear-down as `unloadModel()` and awaits completion of
    /// the detached cleanup task that frees the llama.cpp context and model.
    ///
    /// Use this before process exit or between back-to-back load cycles when
    /// deterministic teardown matters. Production code that drops the backend
    /// and immediately exits can keep calling fire-and-forget `unloadModel()` —
    /// but tests, programmatic reload loops, and anywhere Metal's `MTLDevice`
    /// deinit might race with `llama_free` should await this method instead.
    ///
    /// Without this, Metal's device tear-down can trip
    /// `ggml-metal-device.m:612: GGML_ASSERT([rsets->data count] == 0) failed`
    /// when the context still holds command-buffer resource sets at exit, which
    /// aborts the process with SIGABRT (swift-test exit code 1 even on a green
    /// suite). See issue #391.
    public func unloadAndWait() async {
        unloadModel()
        await waitForPendingCleanup()
    }

    // MARK: - Cleanup

    private func waitForPendingCleanup() async {
        let task = withStateLock {
            let task = cleanupTask
            cleanupTask = nil
            return task
        }
        await task?.value
    }
}

// MARK: - LoadProgressReporting

extension LlamaBackend: LoadProgressReporting {
    /// Installs a progress handler that receives fractional progress values in `[0.0, 1.0]`
    /// delivered by the llama.cpp `progress_callback` during `llama_model_load_from_file`.
    /// The handler fires from the loader thread via an unstructured Task.
    public func setLoadProgressHandler(_ handler: (@Sendable (Double) async -> Void)?) {
        withStateLock { _loadProgressHandler = handler }
    }

    /// Installs backend tuning knobs (KV cache quantization, Flash Attention,
    /// prefill batch size) that take effect on the **next** ``loadModel(from:plan:)``
    /// call. Defaults use Q8 KV cache and platform-gated Flash Attention; pass
    /// explicit ``BackendLoadOptions`` to choose different memory/perf tradeoffs.
    ///
    /// Calling this after a model is already loaded does not retune the live
    /// context — applying KV-quantization changes requires rebuilding the
    /// llama context, which only happens at load time.
    public func setLoadOptions(_ options: BackendLoadOptions) {
        withStateLock { _loadOptions = options }
    }
}

// MARK: - TokenizerVendor

extension LlamaBackend: TokenizerVendor, TokenizerProvider {
    /// Vends `self` as the synchronous tokenizer.
    ///
    /// `llama_tokenize` is a pure vocabulary lookup — safe to call from any thread
    /// while the model is loaded. `LlamaBackend` is already `@unchecked Sendable`.
    public var tokenizer: any TokenizerProvider { self }

    /// Returns the number of tokens in `text` using the loaded llama.cpp vocabulary.
    ///
    /// Falls back to the 4-chars-per-token heuristic if no vocabulary is loaded.
    /// Callers should prefer accessing this through `InferenceService.tokenizer`.
    public func tokenCount(_ text: String) -> Int {
        // Snapshot vocab under stateLock to avoid a use-after-free race with
        // unloadModel() — mirrors the `countTokens(_:)` pattern below.
        guard let currentVocab = withStateLock({ vocab }) else {
            return HeuristicTokenizer.tokenCount(text)
        }
        let tokens = LlamaTokenization.tokenize(text, vocab: currentVocab, addBos: false, parseSpecial: false)
        return tokens.isEmpty ? HeuristicTokenizer.tokenCount(text) : tokens.count
    }
}

// MARK: - TokenCountingBackend

extension LlamaBackend: TokenCountingBackend {
    /// Returns the exact token count for `text` using the loaded model's vocabulary.
    ///
    /// This calls `llama_tokenize` directly — a pure vocabulary lookup with no
    /// context state involved. Safe to call from any thread while the model is loaded.
    ///
    /// - Throws: ``InferenceError/inferenceFailure(_:)`` when the model is not loaded
    ///   or when `llama_tokenize` returns a negative value (buffer sizing failure).
    /// - Note: Call only after a successful `loadModel`. The model pointer is guarded
    ///   under `stateLock` to prevent a use-after-free race with `unloadModel()`.
    public func countTokens(_ text: String) throws -> Int {
        // Snapshot the vocab pointer under stateLock and use it directly for
        // llama_tokenize. Without this snapshot, calling tokenize() outside the
        // lock would re-read `self.vocab` and race with unloadModel() setting it
        // to nil and freeing the backing model — a use-after-free crash.
        // Holding the lock only for the snapshot (not the whole C call) keeps
        // `unloadModel()` responsive while still preventing the race.
        guard let currentVocab = withStateLock({ vocab }) else {
            throw InferenceError.inferenceFailure("countTokens called before model was loaded")
        }
        let utf8 = text.utf8CString
        let maxTokens = Int32(utf8.count) + 1  // +1 for BOS
        var tokens = [llama_token](repeating: 0, count: Int(maxTokens))
        let count = llama_tokenize(currentVocab, text, Int32(text.utf8.count), &tokens, maxTokens, true, true)
        guard count >= 0 || text.isEmpty else {
            throw InferenceError.inferenceFailure("countTokens: llama_tokenize failed for text of length \(text.utf8.count)")
        }
        return text.isEmpty ? 0 : Int(count)
    }
}

// MARK: - MultimodalProjectorConfigurable

extension LlamaBackend: MultimodalProjectorConfigurable {
    /// Stages the mmproj companion URL before the next ``loadModel(from:plan:)`` call.
    ///
    /// Called by ``ModelLifecycleCoordinator`` with ``ModelInfo/mmprojURL`` before loading.
    /// The URL is retained for future multimodal loading, but it does not flip
    /// ``BackendCapabilities/supportsVision`` until this backend can embed images.
    public func setMmprojURL(_ url: URL?) {
        withStateLock { _mmprojURL = url }
    }
}

// MARK: - StructuredHistoryReceiver

extension LlamaBackend: StructuredHistoryReceiver {
    /// Caches the structured conversation history so ``generate(prompt:systemPrompt:config:)``
    /// can detect image parts and surface a clear error when they are present.
    public func setStructuredHistory(_ messages: [StructuredMessage]) {
        withStateLock { _structuredHistory = messages }
    }
}

// MARK: - CancellableModelLoading

extension LlamaBackend: CancellableModelLoading {

    /// Whether a native model load is currently mutating state on a background
    /// thread. Stays `true` from the moment the detached load task starts until
    /// it truly finishes — which may be *after* the `async loadModel` await has
    /// already thrown (e.g. a host deadline fired mid-load). Guarded by
    /// `stateLock`.
    public var isModelLoadInFlight: Bool {
        withStateLock { _isModelLoadInFlight }
    }

    /// Requests that the in-flight native load unwind at its next progress
    /// callback. Best-effort and cooperative — if no progress callback fires
    /// before the load completes, this is a no-op. Always follow with
    /// `awaitModelLoadSettled()` before reusing the backend.
    public func cancelModelLoad() {
        modelLoader.requestCancelLoad()
    }

    /// Suspends until any in-flight native load has truly finished — whether it
    /// completed normally, failed, or unwound cooperatively via
    /// `cancelModelLoad()`. Returns immediately when no load is in flight.
    /// `isModelLoadInFlight` is `false` the instant this returns.
    public func awaitModelLoadSettled() async {
        let task = withStateLock { _activeLoadTask }
        do {
            _ = try await task?.value
        } catch {
            // The load error is surfaced to the original loadModel caller.
            // awaitModelLoadSettled() provides a timing guarantee only
            // (isModelLoadInFlight false on return), not an error contract.
            Self.logger.debug("LlamaBackend.awaitModelLoadSettled: settled with error (already surfaced to caller): \(error)")
        }
    }
}

// MARK: - TokenUsageProvider

extension LlamaBackend: TokenUsageProvider {
    /// Token usage from the most recently completed local-generation turn.
    ///
    /// Populated by `LlamaGenerationDriver.run()` firing the `onUsage` callback just
    /// before finishing the stream. Mirrors the contract that cloud backends satisfy so
    /// `InferenceService` can surface `promptTokens`/`completionTokens` for local turns.
    /// Returns `nil` before the first successful generation on this instance.
    public var lastUsage: (promptTokens: Int, completionTokens: Int)? {
        withStateLock { _lastUsage }
    }
}
