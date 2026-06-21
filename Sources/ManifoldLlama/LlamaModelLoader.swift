import Foundation
import LlamaSwift
import os
import ManifoldInference

/// Owns the C-level model load path: parameter setup, progress-callback ABI
/// bridging, and serialization of concurrent loads.
///
/// `llama_model_load_from_file` and `llama_free` are not safe to call
/// concurrently. A `LlamaModelLoader` instance owns `loadSerializationLock`
/// so every load through this loader is serialized against every other load
/// through the same loader. `LlamaBackend` keeps a single instance for its
/// lifetime, so all loads on one backend share this lock.
// @_spi(Testing): published only for backend test targets (companion-package split, #1749).
@_spi(Testing) public final class LlamaModelLoader: @unchecked Sendable {

    private static let logger = Logger(
        subsystem: ManifoldConfiguration.shared.logSubsystem,
        category: "inference"
    )

    /// Serializes concurrent `initializeModel` C-level calls.
    ///
    /// Blocking is acceptable here because the lock is only held inside a
    /// detached task.
    private let loadSerializationLock = NSLock()

    struct LoadedResources: @unchecked Sendable {
        let model: LlamaModelHandle
        let context: LlamaContextHandle
        let effectiveContextSize: Int32
        /// Auto-detected thinking-marker pair sniffed from
        /// `tokenizer.chat_template` GGUF metadata. `nil` when the metadata
        /// is absent or no known marker pair appears in the template.
        let autoDetectedThinkingMarkers: ThinkingMarkers?
        /// `general.architecture` declared by the GGUF (e.g. `llama`, `qwen2`,
        /// `gemma3`). `nil` when the key is absent. Consumers use this for
        /// architecture-family capability gating without re-reading metadata.
        let architecture: String?
        var vocab: OpaquePointer? { context.vocabPtr }
    }

    // MARK: - RAII Pointer Wrappers
    //
    // These types own C pointers and free them on deinit, making error-path
    // cleanup in initializeModel automatic. On the successful load path,
    // steal() transfers ownership to the instance vars so that unloadModel's
    // explicit ordered cleanup (context before model, both before
    // llama_backend_free) is unaffected.

    /// Owns a `llama_model *`. Calls `llama_model_free` on deinit unless
    /// ownership was transferred via `steal()`.
    final class LlamaModelHandle: @unchecked Sendable {
        private(set) var pointer: OpaquePointer?
        init(_ pointer: OpaquePointer) { self.pointer = pointer }
        /// Transfers ownership to the caller. Subsequent deinit is a no-op.
        func steal() -> OpaquePointer? { defer { pointer = nil }; return pointer }
        deinit { if let p = pointer { llama_model_free(p) } }
    }

    /// Owns a `llama_context *`. Calls `llama_free` on deinit unless
    /// ownership was transferred via `steal()`.
    final class LlamaContextHandle: @unchecked Sendable {
        private(set) var pointer: OpaquePointer?
        let vocabPtr: OpaquePointer?
        init(context: OpaquePointer, vocab: OpaquePointer?) {
            self.pointer = context
            self.vocabPtr = vocab
        }
        /// Transfers ownership to the caller. Subsequent deinit is a no-op.
        func steal() -> OpaquePointer? { defer { pointer = nil }; return pointer }
        deinit { if let p = pointer { llama_free(p) } }
    }

    /// Heap-allocated box used to bridge a Swift async progress handler through the C callback ABI.
    ///
    /// `llama_model_params.progress_callback` is a C function pointer — it cannot capture Swift
    /// context directly. We store the handler in this class, pass an `Unmanaged` retain into
    /// `progress_callback_user_data`, then release it after `llama_model_load_from_file` returns.
    // @_spi(Testing): the box type itself must be reachable so the ABI
    // round-trip helper below can be unit-tested without loading a model.
    @_spi(Testing) public final class ProgressCallbackContext: @unchecked Sendable {
        @_spi(Testing) public let handler: @Sendable (Double) async -> Void
        @_spi(Testing) public init(_ handler: @escaping @Sendable (Double) async -> Void) {
            self.handler = handler
        }
    }

    /// The retain/release-balanced `Unmanaged` round-trip used to smuggle a
    /// Swift `ProgressCallbackContext` through llama.cpp's `@convention(c)`
    /// `progress_callback_user_data` pointer.
    ///
    /// Extracted from `initializeModel` so the ARC contract — exactly one
    /// `passRetained` balanced by exactly one `release`, with the boxed value
    /// surviving the opaque-pointer round-trip in between — can be exercised
    /// headlessly (no GGUF, no `llama_*` call). `initializeModel` inlines the
    /// same three steps because it must interleave them with `modelParams`
    /// mutation and the C load call; this helper is the canonical, tested
    /// statement of the contract.
    ///
    /// `body` receives the opaque pointer that would be stored in
    /// `progress_callback_user_data`; whatever it returns is forwarded. The
    /// retained box is released exactly once when `body` returns (or throws).
    @_spi(Testing) public static func withProgressCallbackBox<R>(
        _ context: ProgressCallbackContext,
        _ body: (UnsafeMutableRawPointer) throws -> R
    ) rethrows -> R {
        let ref = Unmanaged.passRetained(context)
        defer { ref.release() }
        return try body(ref.toOpaque())
    }

    /// Recovers the boxed context from an opaque pointer the way the
    /// `@convention(c)` progress callback does — `fromOpaque` +
    /// `takeUnretainedValue`, which does NOT touch the retain count. Exposed so
    /// the round-trip test can assert the recovered instance is the same object
    /// that was boxed.
    @_spi(Testing) public static func progressContext(
        fromOpaque ptr: UnsafeMutableRawPointer
    ) -> ProgressCallbackContext {
        Unmanaged<ProgressCallbackContext>.fromOpaque(ptr).takeUnretainedValue()
    }

    /// The typed error thrown when `llama_init_from_model` returns nil while the
    /// model itself loaded — an allocator failure at the requested context size,
    /// distinct (code -2) from the model-load failure (code -1).
    ///
    /// Extracted from `initializeModel`'s nil-context guard so the contract — the
    /// `LlamaBackend` domain, the -2 code, and the size-bearing message — can be
    /// asserted headlessly. Forcing `llama_init_from_model` to return nil while
    /// `llama_model_load_from_file` succeeded cannot be triggered reliably from a
    /// config, so this seam stands in for that branch the way `finishDecodeFailure`
    /// stands in for the decode-error teardown. The real guard throws exactly this.
    @_spi(Testing) public static func contextCreationFailure(
        effectiveContextSize: Int32
    ) -> InferenceError {
        InferenceError.modelLoadFailed(underlying: NSError(
            domain: "LlamaBackend",
            code: -2,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "Failed to create llama context at \(effectiveContextSize) tokens. "
                    + "The memory estimate did not account for an allocator failure at this size. "
                    + "Retry with a smaller requested context size.",
            ]
        ))
    }

    /// Synchronous wrapper that holds `loadSerializationLock` while calling the
    /// C-level model init. Called from a detached task so the lock/unlock stays
    /// in a synchronous context (required by Swift 6.3 strict concurrency).
    func serializedModelLoad(
        at url: URL,
        effectiveContextSize: Int32,
        loadOptions: BackendLoadOptions,
        progressHandler: (@Sendable (Double) async -> Void)?
    ) throws -> LoadedResources {
        loadSerializationLock.lock()
        defer { loadSerializationLock.unlock() }
        return try Self.initializeModel(
            at: url,
            effectiveContextSize: effectiveContextSize,
            loadOptions: loadOptions,
            progressHandler: progressHandler
        )
    }

    static func initializeModel(
        at url: URL,
        effectiveContextSize: Int32,
        loadOptions: BackendLoadOptions = .default,
        progressHandler: (@Sendable (Double) async -> Void)? = nil
    ) throws -> LoadedResources {
        var modelParams = llama_model_default_params()
        // Hardware-selection policy lives in `LlamaSamplingPolicy` so the
        // generation-load and embedding-load paths cannot drift again.
        modelParams.n_gpu_layers = LlamaSamplingPolicy.gpuLayerCount()

        // Wire up the progress callback when a handler is installed.
        // The C callback fires on the loader thread; we bridge to async by
        // creating an unstructured Task so the synchronous C callback returns
        // quickly. The Unmanaged retain is released once the load call returns.
        var callbackContextRef: Unmanaged<ProgressCallbackContext>?
        if let handler = progressHandler {
            let ctx = ProgressCallbackContext(handler)
            callbackContextRef = Unmanaged.passRetained(ctx)
            modelParams.progress_callback_user_data = callbackContextRef!.toOpaque()
            modelParams.progress_callback = { progress, userData -> Bool in
                guard let ptr = userData else { return true }
                // `takeUnretainedValue()` does not bump ARC here — the Task closure below
                // captures `ctx` as a Swift reference, which provides its own ARC retain
                // for the Task's lifetime. The Unmanaged retain managed by the outer defer
                // in `loadModel` is separate and only responsible for keeping the context
                // alive during the synchronous C load call.
                let ctx = LlamaModelLoader.progressContext(fromOpaque: ptr)
                let value = Double(progress)
                Task { await ctx.handler(value) }
                return true
            }
        }
        defer { callbackContextRef?.release() }

        // Pre-load architecture preflight. The post-load denylist check below
        // (kept as defense-in-depth) is unreachable for architectures the
        // pinned llama.cpp build cannot construct: those abort inside
        // `llama_model_load_from_file` and return nil before we hold a model
        // pointer, collapsing to a cryptic "Failed to load GGUF model" error.
        // Reading `general.architecture` straight from the GGUF header (no
        // tensor data touched) lets us throw the typed
        // `unsupportedModelArchitecture` for those cases too. A nil result
        // means "couldn't read the header arch" — assume supported and let the
        // real load decide, preserving prior behavior. See issue #62.
        if let headerArch = Self.readArchitectureFromHeader(at: url) {
            try Self.preflightArchitecture(headerArch)
        }

        guard let rawModel = llama_model_load_from_file(url.path, modelParams) else {
            throw InferenceError.modelLoadFailed(underlying: NSError(
                domain: "LlamaBackend",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to load GGUF model from \(url.lastPathComponent)"]
            ))
        }
        let modelHandle = LlamaModelHandle(rawModel)

        // Preflight architecture check: GGUF files declare their model role via
        // `general.architecture`. Vision encoders, embedding-only models, and
        // speech/diffusion checkpoints crash inside `llama_decode` (or silently
        // produce garbage) because they do not expose a causal-LM decode path.
        // Throwing here gives callers a typed error instead of a mid-stream crash.
        // modelHandle owns `rawModel`; throwing lets its deinit call `llama_model_free`.
        let architecture = Self.readArchitectureMetadata(model: rawModel)
        try Self.preflightArchitecture(architecture)

        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx = UInt32(effectiveContextSize)
        ctxParams.n_threads = LlamaSamplingPolicy.threadCount()
        ctxParams.n_threads_batch = ctxParams.n_threads

        // Apply BackendLoadOptions. Defaults prefer Q8 KV cache and Flash
        // Attention on physical devices; callers can still choose llama.cpp's
        // F16/no-FA library defaults explicitly.
        switch loadOptions.kvCacheQuantization {
        case .f16:
            // Library default; leave ctxParams.type_k / type_v as-is (GGML_TYPE_F16).
            break
        case .q8:
            ctxParams.type_k = GGML_TYPE_Q8_0
            ctxParams.type_v = GGML_TYPE_Q8_0
        case .q4:
            ctxParams.type_k = GGML_TYPE_Q4_0
            ctxParams.type_v = GGML_TYPE_Q4_0
        }

        // Flash attention. Simulator path stays disabled regardless of the
        // requested value: simulator Metal does not reliably support FA kernels.
        #if !targetEnvironment(simulator)
        if loadOptions.flashAttention {
            ctxParams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED
        }
        #endif

        if let prefillBatch = loadOptions.prefillBatchSize {
            ctxParams.n_batch = UInt32(max(1, prefillBatch))
        }

        let prefillBatchDescription = loadOptions.prefillBatchSize.map(String.init) ?? "default"
        let kvDescription = loadOptions.kvCacheQuantization.rawValue
        let faDescription = loadOptions.flashAttention ? "on" : "off"
        Self.logger.info("""
            LlamaBackend: initializing context at \(effectiveContextSize, privacy: .public) tokens (plan-authoritative); \
            kv=\(kvDescription, privacy: .public), fa=\(faDescription, privacy: .public), \
            prefillBatch=\(prefillBatchDescription, privacy: .public)
            """)

        // Single attempt. The plan is authoritative — it has already clamped the
        // context to a memory-safe value. If llama_init_from_model still returns
        // nil at this size, we surface a typed error so the caller can request a
        // smaller plan rather than silently allocating half of what was asked for.
        guard let ctx = llama_init_from_model(rawModel, ctxParams) else {
            // modelHandle goes out of scope here → llama_model_free called automatically
            throw Self.contextCreationFailure(effectiveContextSize: effectiveContextSize)
        }

        let contextHandle = LlamaContextHandle(
            context: ctx,
            vocab: llama_model_get_vocab(rawModel)
        )

        // Sniff the GGUF's chat template for known thinking-marker pairs. A
        // missing or empty template is fine — callers treat nil as "no
        // auto-detected markers" and fall back to whatever the caller passes
        // explicitly via `GenerationConfig.thinkingMarkers`.
        let autoMarkers = Self.readChatTemplateMetadata(model: rawModel)
            .flatMap { ThinkingMarkers.fromChatTemplate($0) }

        return LoadedResources(
            model: modelHandle,
            context: contextHandle,
            effectiveContextSize: effectiveContextSize,
            autoDetectedThinkingMarkers: autoMarkers,
            architecture: architecture
        )
    }

    // MARK: - Architecture Preflight

    /// GGUF architecture strings that are NOT causal chat/instruct LMs.
    ///
    /// Denylist (vs. allowlist) because the set of legitimate causal-LM
    /// architectures grows every month (`llama`, `qwen`, `qwen2`, `qwen3`,
    /// `mistral`, `gemma`, `gemma2`, `gemma3`, `phi`, `phi3`, `falcon`,
    /// `mamba`, `gptneox`, …) and rejecting by omission would break new
    /// models the day they land. The known-bad set — vision encoders,
    /// embedding-only models, speech/diffusion — is small and stable.
    ///
    /// Values are lowercased before comparison so `CLIP` / `clip` both match.
    /// Internal for testability — `LlamaBackendTests.test_unsupportedArchitecture_denylistMatches`
    /// validates this set without needing a real GGUF.
    static let unsupportedArchitectures: Set<String> = [
        "clip",         // vision encoders (CLIP-L/B)
        "llava",        // multimodal LLaVA fused weights that need the MM projector
        "mllama",       // Meta multimodal llama variants loaded through llama.cpp's MM path
        "whisper",      // speech-to-text
        "bert",         // embedding-only (no decode path)
        "nomic-bert",   // nomic embedder
        "jina-bert-v2", // jina embedder variant
        "t5encoder",    // T5 encoder-only checkpoints
        "stablediffusion", // diffusion UNet weights
        "sd3",          // stable-diffusion-3
        // Fused-multimodal Gemma checkpoints (Gemma 3n / "gemma4"): a single
        // GGUF carries the text tower plus separate audio (`a.*`) and vision
        // (`v.*`) towers and an mm projector (`mm.*`). The pinned llama.cpp
        // build (b9744) recognizes the `gemma4` arch but its text-model loader
        // only claims the text-tower tensors, then aborts in
        // `done_getting_tensors` ("wrong number of tensors; expected N, got M")
        // because the audio/vision/mmproj tensors are unclaimed. Until the
        // pinned build can load these (or the model is re-packaged as a
        // text-only GGUF + standalone mmproj), surface a typed
        // `unsupportedModelArchitecture` instead of a cryptic nil-load failure.
        // See issue #62.
        "gemma4",       // Gemma 4 fused-multimodal (audio + vision + text)
        "gemma3n",      // Gemma 3n fused-multimodal (the HF/upstream arch name)
    ]

    /// Returns true when `architecture` is on the non-LM denylist.
    @_spi(Testing) public static func isUnsupportedArchitecture(_ architecture: String) -> Bool {
        unsupportedArchitectures.contains(architecture.lowercased())
    }

    /// The wired preflight throw extracted from `initializeModel`: given the
    /// `general.architecture` string read from a loaded GGUF (or `nil` when the
    /// key is absent), throws `InferenceError.unsupportedModelArchitecture`
    /// for denylisted non-LM architectures and returns normally otherwise.
    ///
    /// Extracted so the metadata→`isUnsupportedArchitecture`→throw *wiring*
    /// (not just the predicate, which `LlamaArchitecturePreflightTests` already
    /// covers) can be driven headlessly — no GGUF, no `llama_*` call. A `nil`
    /// architecture means "unknown, assume supported" to avoid false positives
    /// on exotic-but-legitimate LM GGUFs.
    @_spi(Testing) public static func preflightArchitecture(_ architecture: String?) throws {
        if let architecture, Self.isUnsupportedArchitecture(architecture) {
            throw InferenceError.unsupportedModelArchitecture(architecture)
        }
    }

    /// Reads `tokenizer.chat_template` from the loaded GGUF model's metadata.
    ///
    /// Returns `nil` when the key is absent or the metadata read fails.
    /// Chat templates can be several KB of Jinja so we probe for the size
    /// first, then allocate a correctly-sized buffer for the second call.
    /// `llama_model_meta_val_str` returns the byte length the value would
    /// require (excluding the null terminator) when the supplied buffer is
    /// too small, or a negative value when the key is not present.
    @_spi(Testing) public static func readChatTemplateMetadata(model: OpaquePointer) -> String? {
        let key = "tokenizer.chat_template"
        // Probe with a single-byte buffer to learn the required size. The
        // function still returns the value's length even when the buffer
        // can't fit it; a negative return means the key wasn't found.
        var probe: [CChar] = [0]
        let needed = probe.withUnsafeMutableBufferPointer { ptr in
            llama_model_meta_val_str(model, key, ptr.baseAddress, ptr.count)
        }
        guard needed > 0 else { return nil }
        // Allocate one extra byte for the null terminator.
        var buffer = [CChar](repeating: 0, count: Int(needed) + 1)
        let written = buffer.withUnsafeMutableBufferPointer { ptr in
            llama_model_meta_val_str(model, key, ptr.baseAddress, ptr.count)
        }
        guard written > 0 else { return nil }
        return buffer.withUnsafeBytes { ptr in String(decoding: ptr.prefix(Int(written)), as: UTF8.self) }
    }

    /// Reads `general.architecture` directly from a GGUF file header, without
    /// loading the model or touching tensor data.
    ///
    /// Used by the pre-load preflight so denylisted architectures the pinned
    /// llama.cpp build cannot even construct (which abort inside
    /// `llama_model_load_from_file` and return nil) still surface a typed
    /// ``InferenceError/unsupportedModelArchitecture`` rather than a generic
    /// load failure. Returns `nil` on any parse problem — callers treat that as
    /// "unknown, assume supported" and fall through to the real load. See #62.
    ///
    /// Parses only the GGUF header + metadata key-value section (the file's
    /// prefix); the multi-GB tensor blob is never read. Mirrors the layout of
    /// `ManifoldHardware`'s `GGUFMetadataReader`, reimplemented here because that
    /// type is `package`-scoped and unreachable across the package boundary.
    @_spi(Testing) public static func readArchitectureFromHeader(at url: URL) -> String? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }

        // GGUF type-code → fixed scalar byte width. `nil` for string/array,
        // which are length-prefixed and handled specially.
        func scalarSize(_ type: UInt32) -> Int? {
            switch type {
            case 0, 1, 7: return 1          // uint8/int8/bool
            case 2, 3: return 2             // uint16/int16
            case 4, 5, 6: return 4          // uint32/int32/float32
            case 10, 11, 12: return 8       // uint64/int64/float64
            default: return nil             // 8 = string, 9 = array
            }
        }

        // Sequential little-endian readers over the file handle. Any short read
        // or malformed length aborts the whole parse (returns nil upstream).
        func read(_ count: Int) -> Data? {
            guard count >= 0, let d = try? handle.read(upToCount: count), d.count == count else { return nil }
            return d
        }
        func readU32() -> UInt32? { read(4).map { $0.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) } } }
        func readU64() -> UInt64? { read(8).map { $0.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) } } }
        func readString() -> String? {
            guard let len = readU64(), len <= 64 * 1024, let bytes = read(Int(len)) else { return nil }
            return String(decoding: bytes, as: UTF8.self)
        }
        /// Advances past a metadata value of the given type without retaining it.
        func skipValue(type: UInt32) -> Bool {
            if type == 8 { return readString() != nil }      // string
            if type == 9 {                                    // array
                guard let elemType = readU32(), let count = readU64() else { return false }
                if elemType == 8 {
                    for _ in 0..<count where readString() == nil { return false }
                    return true
                }
                guard let elemSize = scalarSize(elemType) else { return false }
                return read(elemSize * Int(count)) != nil
            }
            guard let size = scalarSize(type) else { return false }
            return read(size) != nil
        }

        guard let magic = read(4), magic == Data([0x47, 0x47, 0x55, 0x46]) else { return nil } // "GGUF"
        guard let version = readU32(), version == 2 || version == 3 else { return nil }
        guard readU64() != nil else { return nil }            // tensor count (unused)
        guard let kvCount = readU64() else { return nil }

        for _ in 0..<kvCount {
            guard let key = readString(), let valueType = readU32() else { return nil }
            if key == "general.architecture" {
                guard valueType == 8 else { return nil }       // must be a string
                return readString()
            }
            guard skipValue(type: valueType) else { return nil }
        }
        return nil
    }

    /// Reads `general.architecture` from the loaded GGUF model's metadata.
    ///
    /// Returns `nil` when the key is absent or the metadata read fails —
    /// callers treat that as "unknown, assume supported" to avoid false
    /// positives on exotic-but-legitimate LM GGUFs. `llama_model_meta_val_str`
    /// writes a null-terminated C string into `buf` and returns the byte
    /// length; a negative return value indicates the key was not found.
    static func readArchitectureMetadata(model: OpaquePointer) -> String? {
        let key = "general.architecture"
        // 256 bytes is ample — real values are short strings like "llama",
        // "qwen2", "mistral". The C API writes the length-prefixed string.
        var buffer = [CChar](repeating: 0, count: 256)
        let written = buffer.withUnsafeMutableBufferPointer { ptr in
            llama_model_meta_val_str(model, key, ptr.baseAddress, ptr.count)
        }
        guard written > 0 else { return nil }
        return buffer.withUnsafeBytes { ptr in String(decoding: ptr.prefix(Int(written)), as: UTF8.self) }
    }
}
