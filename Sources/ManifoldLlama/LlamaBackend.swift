import Foundation
import LlamaSwift
@_spi(BackendInternals) import ManifoldContract
// BackendInternals SPI: MemoryPressureHandler (ManifoldHardware) and
// HeuristicTokenizer (ManifoldContract) are part of the frozen backend seam
// published for the companion family packages (#1749).
@_spi(BackendInternals) import ManifoldHardware
import ManifoldInference
import Synchronization
import os

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
  public var isGenerating: Bool { withStateLock { _isGenerating } }
  private var _isGenerating = false

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

  /// Tokenizes `prompt` exactly as ``generate(prompt:systemPrompt:config:)``
  /// does — against the loaded vocab, with `addBos: true` and special-token
  /// parsing on — and returns the resulting token ids as `[Int]`.
  ///
  /// This exposes the *exact* input token sequence the next raw-prompt
  /// generation will decode from. The `manifold-llama-eval` runner records it
  /// as `inputTokenIds` in the `RawRun` record so the manifold-eval
  /// differential harness can compare the tokenization of the same GGUF string
  /// across backends (llama.cpp vs Ollama). Mirrors `generate()`'s
  /// lock-through-lookup discipline so unload cannot free the owning model
  /// while tokenization is reading its vocabulary. Returns `[]`
  /// when no model (and hence no vocab) is loaded.
  ///
  /// `@_spi(Testing)` rather than fully public: it mirrors the existing
  /// `vocab` / ``LlamaTokenization`` test surface and is consumed by the
  /// eval-runner kit and its tests via `@_spi(Testing) import ManifoldLlama`,
  /// not by the library's stable public API.
  @_spi(Testing) public func inputTokenIds(forPrompt prompt: String) -> [Int] {
    withStateLock {
      guard let vocab else { return [] }
      return tokenizePromptLocked(prompt, vocab: vocab).map(Int.init)
    }
  }

  /// Caller holds stateLock for the entire lookup. Both generation and eval
  /// use the same tokenizer (including its test seam) and model lifetime.
  private func tokenizePromptLocked(_ prompt: String, vocab: OpaquePointer) -> [llama_token] {
    if let seam = _generationSeam { return seam.tokenize(prompt, vocab) }
    return LlamaTokenization.tokenize(prompt, vocab: vocab, addBos: true)
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

  /// The sink that receives an ``InferenceMetric`` after every generation call
  /// (success, failure, or cancellation) — see ``generate(prompt:systemPrompt:config:hints:)``.
  ///
  /// Defaults to ``InMemoryMetricSink/shared`` so callers can read recent
  /// metrics without any configuration, matching ``SSECloudBackend`` and
  /// ``FoundationBackend``. Set to `nil` to disable metric emission.
  public var metricSink: (any InferenceMetricSink)? = InMemoryMetricSink.shared

  /// The sink that receives an OTel-shaped ``GenSpan`` after every generation
  /// call (success, failure, or cancellation), derived from the same
  /// ``InferenceMetric`` via ``InferenceMetric/asGenSpan(context:name:)``.
  ///
  /// Defaults to `nil` (opt-in), matching ``SSECloudBackend`` and
  /// ``FoundationBackend``. When attached, spans are emitted alongside the
  /// metric sink so an OTLP exporter wired only to cloud/Foundation lanes
  /// does not leave the GGUF path invisible. Tracker bookkeeping is enabled
  /// when **either** sink is attached (#164).
  public var traceSink: (any TraceSink)?

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
      // Probed form (ManifoldKit #2381 / MIGRATION-llama-vision-probe.md).
      // Keep both halves false until mtmd/clip image embedding is wired
      // (manifold-llama#152) — a staged mmproj URL alone must not advertise
      // vision. See MultimodalProjectorConfigurable's contract.
      supportsVision: BackendVisionCapability.llamaSupportsImageInput(
        projectorStaged: false,
        engineSupportsImageEmbedding: false
      ),
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
  /// Tail of all C-context operations. A stopped generation remains in this
  /// chain until it actually releases native ownership.
  private var nativeWorkTask: Task<Void, Never>?
  private var nativeWorkRevision: UInt64 = 0
  private var cacheRevision: UInt64 = 0

  private final class GenerationCancellation: Sendable {
    let requested = Atomic<Bool>(false)
  }

  /// Replaced on admission; a newer generation must not reset an older
  /// generation's flag while that generation is still unwinding.
  private var generationCancellation = GenerationCancellation()
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
  @_spi(Testing) public var loadOptionsForTesting: BackendLoadOptions {
    withStateLock { _loadOptions }
  }

  // MARK: - Test seams (no live model required)

  /// Snapshot of the `cancelled` atomic. Lets a test assert that the
  /// memory-pressure callback (or `stopGeneration()`) actually flipped the flag
  /// without driving a real decode loop.
  @_spi(Testing) public var isCancelledForTesting: Bool {
    withStateLock { generationCancellation.requested.load(ordering: .sequentiallyConsistent) }
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
      model = sentinel
      context = sentinel
      vocab = sentinel
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
      model = nil
      context = nil
      vocab = nil
      isModelLoaded = false
    }
  }

  /// Installs a fabricated ``ModelManifest`` carrying `identifier` so headless
  /// tests can assert the `model` field of the emitted ``InferenceMetric`` is
  /// read from the manifest rather than hardcoded. Without this the metric
  /// always reports `"unknown"` on an unloaded backend, which cannot
  /// distinguish "read the manifest" from "wrote a literal".
  ///
  /// ONLY call this from test targets. Never call in production code — it
  /// bypasses the normal `loadModel` lifecycle.
  @_spi(Testing) public func injectModelIdentifierForTesting(_ identifier: String) {
    withStateLock {
      _manifest = ModelManifest(
        contextWindow: Int(_effectiveContextSize),
        supportsTools: true,
        supportsThinking: false,
        thinkingMarkers: nil,
        supportsSeed: true,
        supportedSamplingParameters: [.temperature],
        modelIdentifier: identifier,
        producerKind: .local
      )
    }
  }

  /// Replaces the two llama.cpp-touching steps of
  /// ``generate(prompt:systemPrompt:config:hints:)`` — prompt tokenization and
  /// the construction of the ``LlamaEngine`` handed to
  /// ``LlamaGenerationDriver`` — so the whole generation Task body runs with
  /// no `.gguf` and no Metal (#165).
  ///
  /// Everything BETWEEN those two steps is production code and stays on the
  /// real path: the context-window preflight, the KV shared-prefix scan, the
  /// `isGenerating` flip, the `metricsEnabled` gate, the tracker start, the
  /// `onToken`/`onError` hook construction, the `driver.run(...)` call, the
  /// `defer`-based metric emit, and the post-decode KV-coherence guard. That
  /// is the point: a test drives the orchestration, not a reimplementation of
  /// it.
  ///
  /// Pass `nil` to restore the production path.
  ///
  /// ONLY call this from test targets. Never call in production code.
  @_spi(Testing) public func installGenerationSeamForTesting(_ seam: LlamaGenerationSeam?) {
    withStateLock { _generationSeam = seam }
  }

  /// Guarded by `stateLock`. Nil in production — see
  /// ``installGenerationSeamForTesting(_:)``.
  private var _generationSeam: LlamaGenerationSeam?

  /// Directly sets `isGenerating` under `stateLock`. Lets headless tests put the
  /// backend into a simulated mid-generation state so the `alreadyGenerating` guard
  /// can be exercised without a real decode loop.
  ///
  /// ONLY call this from test targets. Never call in production code.
  @_spi(Testing) public func setIsGeneratingForTesting(_ value: Bool) {
    withStateLock { _isGenerating = value }
  }

  // MARK: - Multimodal Projector

  /// URL of the mmproj companion file, set by ``MultimodalProjectorConfigurable`` before each load.
  ///
  /// Guarded by `stateLock`. Non-nil when a vision-capable model's projector is staged for load.
  /// Cleared by ``unloadModel()``. The current vendored LlamaSwift xcframework does not expose
  /// the CLIP / mtmd C APIs needed to turn images into embeddings, so ``capabilities`` continues
  /// to advertise `supportsVision = false` until that binding exists.
  private var _mmprojURL: URL?

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
        Self.logger.warning(
          "Memory pressure: warning — stopping generation to prevent Metal buffer revocation (#415)"
        )
        self.stopGeneration()
      case .critical:
        Self.logger.warning(
          "Memory pressure: critical — stopping generation and scheduling model unload (#415)")
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
    assert(
      plan.verdict != .deny,
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

    Self.logger.info(
      "Llama backend loaded \(url.lastPathComponent) with context \(loadedResources.effectiveContextSize)"
    )
  }

  // MARK: - Generation

  public func generate(
    prompt: String,
    systemPrompt: String?,
    config: GenerationConfig,
    hints: GenerationRuntimeHints
  ) throws -> GenerationStream {
    // Reject image-bearing turns before any other state check so the caller
    // gets the actionable xcframework-limitation message regardless of
    // whether a model happens to be loaded. The check is independent of
    // model state — it only inspects the structured history we cached
    // from the coordinator. CLIP / mtmd C APIs are not present in the
    // bundled xcframework (mattt/llama.swift 2.8772.0, llama.cpp build
    // b8772); upgrading to a build that exposes clip.h / mtmd.h will
    // unblock real image embedding.
    // Conversation history arrives per-call on `hints.history` (#2312) — not
    // from shared instance state. LlamaBackend consumes the pre-rendered
    // `prompt`; it only inspects the structured history here to reject
    // image-bearing turns the vendored xcframework can't embed.
    let history = hints.history
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

    // Tokenization is a vocabulary-only lookup. Hold model ownership through
    // lookup, preflight, and task publication so unload cannot swap the model
    // between these steps, and concurrent callers cannot both reserve a turn.
    stateLock.lock()
    defer { stateLock.unlock() }
    guard isModelLoaded, context != nil, let vocab, model != nil else {
      throw InferenceError.inferenceFailure("No model loaded")
    }
    guard !_isGenerating else {
      throw InferenceError.alreadyGenerating
    }
    let generationSeam = _generationSeam
    let tokens = tokenizePromptLocked(prompt, vocab: vocab)
    guard !tokens.isEmpty else {
      throw InferenceError.inferenceFailure("Failed to tokenize prompt")
    }

    let maxTokens = config.maxOutputTokens ?? 2048
    let contextSize = Int(_effectiveContextSize)
    guard
      Self.contextWindowFits(
        promptTokens: tokens.count, maxOutputTokens: maxTokens, contextSize: contextSize
      )
    else {
      throw InferenceError.contextExhausted(
        promptTokens: tokens.count,
        maxOutputTokens: maxTokens,
        contextSize: contextSize
      )
    }

    let cancellation = GenerationCancellation()
    generationCancellation = cancellation
    _isGenerating = true
    let predecessor = nativeWorkTask
    nativeWorkRevision &+= 1
    let workRevision = nativeWorkRevision
    let loadToken = activeLoadToken
    let initialCacheRevision = cacheRevision
    Self.logger.debug("Llama generate started")

    let (stream, continuation) = AsyncThrowingStream.makeStream(of: GenerationEvent.self)
    let generationStream = GenerationStream(stream)

    // Metrics + trace seam (#142 / #164). `metricTracker` is a lock-guarded
    // accumulator (see LlamaMetricTracker) captured by reference into the
    // Task below; sinks / `promptTokenCount` / `modelIdentifier` are
    // snapshotted here, outside the Task, so the eventual `sink.record(...)`
    // call doesn't need to re-read backend state after the model may have
    // been unloaded. Snapshotting via `withStateLock { … }` mirrors
    // `FoundationBackend.generate()`'s capture idiom verbatim, even though
    // the sink properties themselves are plain (unlocked) vars on both
    // backends.
    //
    // `metricsEnabled` mirrors `SSEGenerationTaskRunner`'s
    // `guard context.metricSink != nil || context.traceSink != nil else {
    // return }` — gated on **either** sink so a trace-sink-only config
    // still starts the tracker and fires per-token hooks. Core's own
    // runner has a live trap: it gates emission on either sink but only
    // passes the tracker (enabling `.recordToken()`) when `metricSink !=
    // nil`, so a trace-only setup silently builds an all-zeros span from
    // an unstarted tracker. We deliberately do **not** reproduce that
    // trap: when either sink is attached, `start()` and `onToken`/`onError`
    // run. When neither is listening we skip them entirely so a generation
    // with no sink attached pays no per-token timing/lock overhead.
    let metricTracker = LlamaMetricTracker()
    let capturedMetricSink = metricSink
    let capturedTraceSink = traceSink
    let metricsEnabled = capturedMetricSink != nil || capturedTraceSink != nil
    let promptTokenCount = tokens.count
    let modelIdentifier = _manifest?.modelIdentifier ?? "unknown"

    let task = Task.detached(priority: .userInitiated) { [weak self, generationStream] in
      guard let self else {
        continuation.finish()
        return
      }

      // Include time queued behind a stopped turn in perceived latency.
      if metricsEnabled { metricTracker.start() }

      defer {
        self.withStateLock {
          if self.generationCancellation === cancellation {
            self._isGenerating = false
            self.generationTask = nil
          }
          if self.nativeWorkRevision == workRevision {
            self.nativeWorkTask = nil
          }
        }
        Self.logger.debug("Llama generate finished")
        // Emit exactly once per generation, success or failure — this
        // `defer` runs on every exit path of THIS TASK BODY, once the
        // body has begun (Swift's `defer` semantics), including the
        // early "pointers == nil" bail below, which is why that
        // branch calls `metricTracker.recordError(...)` before
        // returning. The pre-Task guards above (no model loaded,
        // already generating, tokenize failure, `.contextExhausted`)
        // throw before this Task is ever created and so emit nothing
        // — consistent with core's cloud/Foundation backends, which
        // likewise only emit once request dispatch has begun.
        LlamaMetricTracker.emitMetric(
          from: metricTracker,
          to: capturedMetricSink,
          traceSink: capturedTraceSink,
          provider: BackendName.llama.rawValue,
          model: modelIdentifier,
          promptTokens: promptTokenCount
        )
      }

      // Admission and native ownership are distinct: a replacement can be
      // returned to its caller now, but must wait for all prior native work.
      await predecessor?.value
      if cancellation.requested.load(ordering: .sequentiallyConsistent) || Task.isCancelled {
        await MainActor.run { generationStream.setPhase(.done) }
        continuation.finish()
        return
      }

      let state = self.withStateLock {
        () -> (OpaquePointer, OpaquePointer, [llama_token], ThinkingMarkers?)? in
        guard self.activeLoadToken == loadToken,
          let ctx = self.context, let voc = self.vocab
        else { return nil }
        return (ctx, voc, self.sessionKVState?.tokens ?? [], self._autoDetectedThinkingMarkers)
      }
      guard let (context, vocab, previousTokens, autoDetected) = state else {
        if metricsEnabled { metricTracker.recordError("backendUnloaded") }
        continuation.finish()
        return
      }
      let reuseLen = zip(tokens, previousTokens).prefix(while: { $0.0 == $0.1 }).count

      // The second (and last) llama.cpp touch point in this method —
      // everything the driver does to the C API goes through this
      // witness. Production builds `LlamaCAPIEngine`, a verbatim
      // transcription of the calls that used to be inline in the driver.
      let engine: any LlamaEngine
      if let generationSeam {
        engine = generationSeam.makeEngine(context, vocab)
      } else {
        engine = LlamaCAPIEngine(context: context, vocab: vocab)
      }

      let driver = LlamaGenerationDriver()
      // Manual override beats auto-detection. When neither is present,
      // markers stays nil and the driver skips ThinkingTransform entirely.
      let resolvedMarkers = hints.thinkingMarkers ?? autoDetected
      // Typed out with explicit if/else (rather than a ternary or an
      // inline expression in the call below) to sidestep a compiler
      // diagnostic-generation crash observed with both of those forms.
      var onTokenHook: (@Sendable () -> Void)?
      var onGeneratedTokenHook: (@Sendable () -> Void)?
      var onErrorHook: (@Sendable (String) -> Void)?
      if metricsEnabled {
        onTokenHook = { metricTracker.recordToken() }
        onGeneratedTokenHook = { metricTracker.recordGeneratedToken() }
        onErrorHook = { label in metricTracker.recordError(label) }
      }
      var prefillStarted = false
      var promptDecoded = false
      let kvCoherent = await driver.run(
        engine: engine,
        tokens: tokens,
        reuseLen: reuseLen,
        maxTokens: maxTokens,
        config: config,
        markers: resolvedMarkers,
        isCancelled: {
          Task.isCancelled || cancellation.requested.load(ordering: .sequentiallyConsistent)
        },
        generationStream: generationStream,
        continuation: continuation,
        onPrefillEstimate: { [self] measured in
          self.withStateLock {
            if self.activeLoadToken == loadToken { self._lastMeasuredBytesPerToken = measured }
          }
        },
        onUsage: { [self] promptTokens, completionTokens in
          self.withStateLock {
            guard self.activeLoadToken == loadToken,
              self.generationCancellation === cancellation
            else { return }
            self._lastUsage = (promptTokens: promptTokens, completionTokens: completionTokens)
          }
        },
        onPrefillStarted: { prefillStarted = true },
        onPromptDecoded: { promptDecoded = true },
        onToken: onTokenHook,
        onGeneratedToken: onGeneratedTokenHook,
        onError: onErrorHook
      )
      // A cancelled partial prefill must not advertise tokens the C cache
      // never consumed. A completely decoded prompt may still be reusable
      // after stopping in the sampling loop.
      if !kvCoherent || prefillStarted {
        self.withStateLock {
          guard self.activeLoadToken == loadToken,
            self.cacheRevision == initialCacheRevision
          else { return }
          self.sessionKVState = kvCoherent && promptDecoded ? SessionKVState(tokens: tokens) : nil
        }
      }
    }

    // Publish both handles before unlocking, so Stop reaches this particular
    // stream and unload joins the whole chain. Stream disposal only cancels
    // its own task; finishing an older stream cannot cancel a replacement.
    generationTask = task
    nativeWorkTask = task
    continuation.onTermination = { @Sendable reason in
      if case .cancelled = reason { task.cancel() }
    }

    return generationStream
  }

  // MARK: - Control

  /// Joins the currently scheduled native decode, cache clearing, and unload
  /// work. Stop makes the backend available for a replacement immediately,
  /// but does not interrupt an in-flight C call or relinquish native ownership.
  public func awaitGenerationSettled() async {
    let tasks = withStateLock { (nativeWorkTask, cleanupTask) }
    await tasks.0?.value
    await tasks.1?.value
  }

  public func stopGeneration() {
    let taskToCancel = withStateLock {
      generationCancellation.requested.store(true, ordering: .sequentiallyConsistent)
      _isGenerating = false
      return generationTask
    }
    taskToCancel?.cancel()
  }

  /// Invalidates in-memory KV reuse when clearing or switching conversations.
  public func resetConversation() {
    clearConversationCache(zeroData: false)
  }

  /// Zeros the KV tensors, after any native decode already using the context.
  /// A later generation cannot touch the context until the wipe completes.
  public func secureWipe() {
    clearConversationCache(zeroData: true)
  }

  private func clearConversationCache(zeroData: Bool) {
    withStateLock {
      sessionKVState = nil
      cacheRevision &+= 1
      guard let context else { return }
      let seam = _generationSeam
      guard let predecessor = nativeWorkTask else {
        Self.clearCache(context: context, zeroData: zeroData, seam: seam)
        return
      }

      // The captured context stays alive: unload takes ownership of the tail
      // and waits for it before freeing the context. Keep the clear itself
      // out of the caller's thread when native work is already in flight.
      nativeWorkRevision &+= 1
      let revision = nativeWorkRevision
      nativeWorkTask = Task.detached { [weak self] in
        await predecessor.value
        Self.clearCache(context: context, zeroData: zeroData, seam: seam)
        self?.withStateLock {
          if self?.nativeWorkRevision == revision { self?.nativeWorkTask = nil }
        }
      }
    }
  }

  private static func clearCache(
    context: OpaquePointer, zeroData: Bool, seam: LlamaGenerationSeam?
  ) {
    if let seam {
      seam.clearCache(zeroData)
    } else if let memory = llama_get_memory(context) {
      llama_memory_clear(memory, zeroData)
      llama_synchronize(context)
    }
  }

  public func unloadModel() {
    stateLock.lock()
    generationCancellation.requested.store(true, ordering: .sequentiallyConsistent)
    nextLoadToken &+= 1
    activeLoadToken = nextLoadToken
    nativeWorkRevision &+= 1
    cacheRevision &+= 1

    let previousCleanup = cleanupTask
    let taskToCancel = generationTask
    let capturedTask = nativeWorkTask
    let capturedContext = context
    let capturedModel = model

    // Clear state immediately so callers see the backend as unloaded
    // without waiting for C memory deallocation.
    generationTask = nil
    nativeWorkTask = nil
    context = nil
    model = nil
    vocab = nil
    isModelLoaded = false
    _isGenerating = false
    sessionKVState = nil
    _autoDetectedThinkingMarkers = nil
    _manifest = nil
    _mmprojURL = nil
    _lastUsage = nil
    // An in-flight load is superseded: clear the in-flight flag immediately
    // so callers see the unloaded state. _activeLoadTask is intentionally
    // NOT cleared here — awaitModelLoadSettled() reads it to suspend until
    // the native C work truly finishes (the detached task may still be
    // running). The task's own defer skips clearing (bumped activeLoadToken)
    // so the reference stays valid until the next loadModel overwrites it.
    _isModelLoadInFlight = false
    guard capturedTask != nil || capturedContext != nil || capturedModel != nil else {
      stateLock.unlock()
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
    // Publish ownership before releasing the lock. Concurrent unloads and
    // waiters must never see a gap between detaching pointers and registering
    // the task that still owns them.
    self.cleanupTask = newCleanupTask
    stateLock.unlock()
    taskToCancel?.cancel()
    Self.logger.info("Llama backend unloaded")
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
    // Keep the handle for every waiter, including reload and repeated unload.
    // A waiter observing the task must not consume another caller's join.
    let task = withStateLock { cleanupTask }
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
    withStateLock {
      guard let vocab else { return HeuristicTokenizer.tokenCount(text) }
      let tokens = LlamaTokenization.tokenize(
        text, vocab: vocab, addBos: false, parseSpecial: false)
      return tokens.isEmpty ? HeuristicTokenizer.tokenCount(text) : tokens.count
    }
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
    // Retain model ownership until the synchronous C lookup returns.
    stateLock.lock()
    defer { stateLock.unlock() }
    guard let currentVocab = vocab else {
      throw InferenceError.inferenceFailure("countTokens called before model was loaded")
    }
    let utf8 = text.utf8CString
    let maxTokens = Int32(utf8.count) + 1  // +1 for BOS
    var tokens = [llama_token](repeating: 0, count: Int(maxTokens))
    let count = llama_tokenize(
      currentVocab, text, Int32(text.utf8.count), &tokens, maxTokens, true, true)
    guard count >= 0 || text.isEmpty else {
      throw InferenceError.inferenceFailure(
        "countTokens: llama_tokenize failed for text of length \(text.utf8.count)")
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
      Self.logger.debug(
        "LlamaBackend.awaitModelLoadSettled: settled with error (already surfaced to caller): \(error)"
      )
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
