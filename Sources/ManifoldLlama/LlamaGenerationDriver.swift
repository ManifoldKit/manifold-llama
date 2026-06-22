import Foundation
import LlamaSwift
import os
import ManifoldInference
import ManifoldHardware

/// Owns the token-generation loop for a single `LlamaBackend.generate()` call.
///
/// `LlamaGenerationDriver` is stateless — every dependency it needs is passed
/// as an explicit parameter to `run()`. This keeps it free of any reference to
/// `LlamaBackend` and makes the generation logic independently testable.
///
/// Conforms to `LocalInferenceAdapter` so cross-backend drift guards (e.g.
/// `LocalBackendRealDriverCoverageTest`) can introspect the composed
/// witnesses without instantiating `LlamaBackend`. Sendable explicitly —
/// the struct has no stored mutable state.
// @_spi(Testing): published only for backend test targets (companion-package split, #1749).
@_spi(Testing) public struct LlamaGenerationDriver: LocalInferenceAdapter {

    public init() {}

    // MARK: - LocalInferenceAdapter conformance

    public let adapterName: String = "llama.generation"
    public let toolCallShape: any LocalToolCallShape = InlineXMLToolCallMarkers()
    public let thinkingMarkerStrategy: LocalThinkingMarkerStrategy = .eagerWhenMarkersPresent
    /// Llama's static capability shape published for drift-guard probing.
    /// Mirrors the payload `LlamaBackend.capabilities` returns once a model
    /// is loaded; conformance to `LocalInferenceAdapter` requires a
    /// driver-level snapshot so coverage tests do not have to boot the
    /// backend.
    public let declaredCapabilities: BackendCapabilities = BackendCapabilities(
        supportedParameters: [
            .temperature, .topP, .topK, .repeatPenalty,
            .minP, .repetitionPenalty, .presencePenalty, .frequencyPenalty,
        ],
        maxContextTokens: 8192,
        requiresPromptTemplate: true,
        supportsSystemPrompt: true,
        supportsToolCalling: true,
        supportsStructuredOutput: false,
        supportsNativeJSONMode: false,
        cancellationStyle: .cooperative,
        supportsTokenCounting: true,
        memoryStrategy: .resident,
        maxOutputTokens: 4096,
        supportsStreaming: true,
        isRemote: false,
        supportsThinking: true
    )


    private static let logger = Logger(
        subsystem: ManifoldConfiguration.shared.logSubsystem,
        category: "inference"
    )

    /// Consecutive identical decoded-token run length that triggers an early exit.
    ///
    /// Small models (e.g. smollm2-135m) can enter visible repetition loops where the same
    /// token string is emitted hundreds of times. The existing `LoopingDetector` catches this
    /// after the fact; this constant lets the generation loop break out as soon as the
    /// repetition is unambiguous, saving KV-cache cycles and wall time.
    private static let maxRepeatWindow = 20

    /// Maximum phrase length (in tokens) to scan for repeated sequences.
    private static let maxPhraseLen = 20
    /// Minimum consecutive phrase repetitions before early exit.
    private static let minPhraseRepeats = 3
    /// Capacity of the phrase-detection token buffer (maxPhraseLen × minPhraseRepeats + 1).
    private static let phraseWindowCap = maxPhraseLen * minPhraseRepeats + 1

    public struct DRYSamplerDescriptor: Equatable {
        public let nCtxTrain: Int32
        public let options: LlamaDRYSamplerOptions

        public init?(config: GenerationConfig, nCtxTrain: Int32) {
            guard let options = config.llamaDRY else { return nil }
            self.nCtxTrain = nCtxTrain
            self.options = options
        }
    }

    public struct XTCSamplerDescriptor: Equatable {
        public let options: LlamaXTCSamplerOptions
        public let resolvedSeed: UInt32

        public init?(config: GenerationConfig, fallbackSeed: UInt32) {
            guard let options = config.llamaXTC else { return nil }
            self.options = options
            self.resolvedSeed = options.seed ?? fallbackSeed
        }
    }

    public struct MirostatV2SamplerDescriptor: Equatable {
        public let options: LlamaMirostatV2SamplerOptions
        public let resolvedSeed: UInt32

        public init?(config: GenerationConfig, fallbackSeed: UInt32) {
            guard let options = config.llamaMirostatV2 else { return nil }
            self.options = options
            self.resolvedSeed = options.seed ?? fallbackSeed
        }
    }

    // MARK: - Run

    /// Executes the generation loop: clears the KV cache, builds the sampler
    /// chain, decodes the prompt in `n_batch`-sized chunks, and runs the
    /// token-generation loop until `maxTokens` visible output tokens are produced,
    /// an EOG token is sampled, or `isCancelled()` returns `true`.
    ///
    /// Thinking tokens (`.thinkingToken` events) do **not** count toward `maxTokens`.
    /// They are governed separately by `config.maxThinkingTokens`. The total loop
    /// iteration count is `maxTokens + effectiveThinkingBudget`, capped to the
    /// remaining KV context space.
    ///
    /// Yields `.token` events into `continuation` and drives `generationStream`
    /// phase transitions (`.streaming`, `.done`, `.failed`). On any error the
    /// continuation is finished with a thrown `InferenceError` and the stream
    /// phase is set to `.failed`.
    ///
    /// - Parameters:
    ///   - context: Live `llama_context *` snapshot captured under `stateLock`.
    ///   - vocab: Live `llama_vocab *` snapshot captured under `stateLock`.
    ///   - tokens: Tokenized prompt (including BOS) — computed before the Task.
    ///   - reuseLen: Number of leading prompt tokens that *matched* the previous
    ///     turn's KV state (the detected shared-prefix length). When > 0 the driver
    ///     emits `.kvCacheReuse(promptTokensReused:)` as an observability signal.
    ///     The driver reuses the prefix's KV cells but only up to a `batchSize`
    ///     boundary (`alignedReuseLen`): it resumes the re-decode at that aligned
    ///     position so the chunk producing the sampling-position (N-1) logits has a
    ///     batch shape bit-identical to the cold turn's, preserving greedy
    ///     determinism (ManifoldKit#1677) while keeping genuine prefix reuse.
    ///   - maxTokens: Maximum number of new tokens to generate.
    ///   - config: Sampling parameters (temperature, topP, repeatPenalty).
    ///   - markers: Thinking markers for the active template, or nil to disable ThinkingTransform.
    ///     When non-nil, `.thinkingToken` / `.thinkingCompleted` events are emitted for reasoning
    ///     content and `config.maxThinkingTokens` is enforced. When nil, every decoded chunk
    ///     surfaces as a plain `.token` event — there is no longer a sniff-mode fallback that
    ///     retroactively engages the parser on raw `<think>` substrings; auto-detection at
    ///     load time replaced it.
    ///   - isCancelled: Closure that returns `true` when the caller has requested
    ///     cancellation (combines `Task.isCancelled` and the backend's `Atomic<Bool>`).
    ///   - generationStream: Stream whose phase is updated on the main actor.
    ///   - continuation: Raw stream continuation for yielding events.
    /// Returns `true` when the KV cache is in a coherent state after the call
    /// (success or clean cancellation), `false` when a C decode error occurred
    /// and the KV state should be treated as undefined. Callers must clear their
    /// `sessionKVState` when this returns `false`.
    @discardableResult
    func run(
        context: OpaquePointer,
        vocab: OpaquePointer,
        tokens: [llama_token],
        reuseLen: Int,
        maxTokens: Int,
        config: GenerationConfig,
        markers: ThinkingMarkers?,
        isCancelled: () -> Bool,
        generationStream: GenerationStream,
        continuation: AsyncThrowingStream<GenerationEvent, Error>.Continuation,
        // Adaptive prefill headroom (issue #1592). Defaults wire the real Mach
        // footprint sampler and free-memory query; tests inject synthetic readings.
        // `prefillFootprintSampler` returns this process's resident bytes; the
        // chunk-to-chunk delta feeds the EWMA. `prefillHeadroomSampler` returns
        // remaining free bytes, sampled fresh each chunk so the guard tracks live
        // pressure. `onPrefillEstimate` reports the learned per-token cost back to
        // the backend so a future load's `ModelLoadPlan` can use a measured budget.
        prefillFootprintSampler: @Sendable () -> UInt64? = { AppMemoryUsage.currentBytes() },
        prefillHeadroomSampler: @Sendable () -> UInt64? = { DeviceCapabilityService.queryAvailableMemory() },
        prefillSafetyFactor: Double = 1.5,
        onPrefillEstimate: (@Sendable (UInt64) -> Void)? = nil,
        onUsage: (@Sendable (Int, Int) -> Void)? = nil
    ) async -> Bool {
        Self.logger.debug("LlamaGenerationDriver run started")

        // MARK: Batch size

        // `n_batch` caps how many tokens can flow through a single
        // `llama_decode` call. llama.cpp asserts
        // `GGML_ASSERT(n_tokens_all <= cparams.n_batch)` in
        // `llama-context.cpp`, so prompts longer than this must be decoded
        // in chunks. We never set `n_batch` on `ctxParams`, so it inherits
        // llama.cpp's default (2048 at the time of writing).
        let batchSize = max(1, Int(llama_n_batch(context)))

        // MARK: KV cache clear / reuse

        // Batch-shape determinism (ManifoldKit#1677) WITHOUT abandoning KV reuse.
        //
        // The hazard: a partial re-decode of a KV-reuse tail can be a *different
        // batch shape* than the first turn's full-prompt decode. Metal attention
        // kernels pick their parallel-reduction strategy from the batch token
        // count, so if the chunk that contains the sampling position (N-1) has a
        // different start/size than it did on the cold turn, the FP-accumulation
        // order at N-1 differs and the greedy argmax can flip on near-tied logits.
        // The old -2 tail cap (PR #966) only happened to align the reduction path
        // for Qwen-family models; it did not generalise.
        //
        // The fix: keep genuine prefix reuse, but only reuse a prefix that ends on
        // a `batchSize` boundary. The prompt is decoded in `[0,B), [B,2B), …`
        // chunks; the chunk containing N-1 starts at `floor((N-1)/B)*B`. If we
        // resume the re-decode at *any* batch-aligned position ≤ that boundary,
        // every subsequent chunk — including the final one that produces the N-1
        // logits — has byte-identical start, size, and positions to the cold path,
        // so the Metal reduction path is bit-identical every turn.
        //
        // Aligning the reused prefix down to a batch boundary is also what makes
        // the *cached* KV bit-identical: those cells were produced on a previous
        // turn by the same batched `[0,B), [B,2B), …` calls at the same positions
        // as the cold path would issue, so reusing whole aligned chunks is exact.
        // (A reuse boundary mid-chunk would leave the next chunk a partial-shape
        // decode, reintroducing the hazard — hence the floor to `batchSize`.)
        //
        // `alignedReuseLen` drives the actual decode start; `reuseLen` (the full
        // detected shared-prefix length) is still surfaced as
        // `.kvCacheReuse(promptTokensReused:)` as the observability signal.
        //
        // We also cap the reuse below the *final* chunk's start. The sampling
        // position N-1 lives in the chunk beginning at `finalChunkStart`; that
        // chunk MUST be re-decoded so its last token carries `logits = 1`. If the
        // whole prompt matched the previous turn and `tokens.count` is a multiple
        // of `batchSize`, an un-capped floor would equal `tokens.count`, the decode
        // loop would never run, and there would be no logits to sample. Capping at
        // `finalChunkStart` guarantees we always re-decode the N-1 chunk (which is
        // also where the determinism fix lives).
        let alignedReuseLen = Self.alignedKVReuseLength(
            tokenCount: tokens.count, reuseLen: reuseLen, batchSize: batchSize)

        if let memory = llama_get_memory(context) {
            if alignedReuseLen > 0 {
                // Keep the batch-aligned reused prefix's KV cells; trim only the
                // tail beyond it. Running this here (inside the generation Task) is
                // lifecycle-safe: all context-touching work in this driver is
                // serialized with unloadModel() via the task install (see the
                // pointer re-read in LlamaBackend.generate), exactly as the full
                // `llama_memory_clear` path is.
                llama_memory_seq_rm(memory, 0, Int32(alignedReuseLen), -1)
            } else {
                llama_memory_clear(memory, false)
            }
        }

        if reuseLen > 0 {
            continuation.yield(.kvCacheReuse(promptTokensReused: reuseLen))
        }

        // MARK: Sampler chain setup

        // Sampler chain order matters. Grammar (when present) must run BEFORE the
        // probability filters (top_k / top_p / min_p) so it can prune invalid
        // tokens to -inf while every candidate is still in play. If grammar runs
        // after min_p, the filters can shrink the candidate pool to a set that
        // contains no grammar-valid tokens; the grammar then masks all remaining
        // logits to -inf, dist samples a numerical fallback (e.g. token 365 `(`),
        // and the chain's automatic accept step inside `llama_sampler_sample`
        // calls `llama_grammar_accept_token`, which throws
        // `std::runtime_error: Unexpected empty grammar stack` across the C ABI
        // and aborts the process with libc++abi (see prior crash logs from
        // test_grammar_cancelCleansTeardown). Final order:
        //   penalties → grammar → dry → top_k → top_p → min_p → temp → xtc → dist
        // When mirostat v2 is active it replaces the (temp, xtc, dist) tail with
        // a single `mirostat_v2` step that handles both temperature and final
        // selection.

        // Shared sampler parameters, computed once so the permissive (no-grammar)
        // and strict (grammar) chains used by the thinking-phase gate (issue #1595)
        // are byte-identical apart from the grammar stage.

        // Prefer the explicit `repetitionPenalty` knob when callers supplied it; fall
        // back to the legacy `repeatPenalty` field otherwise. The chain is added when
        // ANY of the three penalties is non-no-op; presence and frequency are additive
        // so 0.0 is the no-op value, while repetition is multiplicative so 1.0 is no-op.
        let effectiveRepetitionPenalty = config.repetitionPenalty ?? config.repeatPenalty
        let effectivePresencePenalty = config.presencePenalty ?? 0.0
        let effectiveFrequencyPenalty = config.frequencyPenalty ?? 0.0
        // llama.cpp uses one shared window for all three penalties; default 64 matches
        // pre-existing behaviour. MLX exposes per-penalty windows; llama does not.
        let effectivePenaltyWindow = Int32(config.repetitionContextSize ?? 64)
        let penaltiesActive = effectiveRepetitionPenalty > 1.0
            || effectivePresencePenalty != 0.0
            || effectiveFrequencyPenalty != 0.0

        // Use the caller-supplied seed when available so consecutive runs with the same
        // prompt + config produce identical token streams. `llama_sampler_init_dist`
        // takes `uint32_t`, so we truncate the GenerationConfig's `UInt64` seed —
        // collisions across the truncation boundary are not a correctness issue, only
        // a slight loss of seed-space entropy. Falls back to a fresh random seed when
        // the caller didn't request determinism. Computed once and shared by both
        // chains so the seeded RNG sequence is consistent across a phase switch.
        let samplerSeed: UInt32
        if let seed = config.seed {
            samplerSeed = UInt32(truncatingIfNeeded: seed)
        } else {
            samplerSeed = UInt32.random(in: 0...UInt32.max)
        }

        // Thinking-marker / grammar gating decision (issue #1595).
        //
        // `useParser` mirrors the value recomputed below for the generation loop:
        // a thinking parser runs only when markers are present AND thinking is not
        // explicitly disabled (`maxThinkingTokens == 0`). When a grammar AND a
        // thinking parser are both active we must NOT let the grammar constrain the
        // reasoning block, so we build two chains and gate which one samples each
        // iteration. All other cases (no grammar, or grammar without thinking) keep
        // the single-chain behaviour unchanged.
        let thinkingDisabled = config.maxThinkingTokens == 0
        let useParser = !thinkingDisabled && markers != nil
        let hasGrammar = config.grammar != nil
        let gateGrammarOnThinking = hasGrammar && useParser

        // Outcome of building one sampler chain. `chainInitFailed` and
        // `grammarParseFailed` map to the two distinct error paths the original
        // single-chain code surfaced (different message + different KV-coherence
        // return value), so collapsing them would lose that fidelity.
        enum SamplerBuildOutcome {
            case success(UnsafeMutablePointer<llama_sampler>)
            case chainInitFailed
            case grammarParseFailed
        }

        func makeSampler(includeGrammar: Bool) -> SamplerBuildOutcome {
            let sparams = llama_sampler_chain_default_params()
            guard let sampler = llama_sampler_chain_init(sparams) else {
                return .chainInitFailed
            }
            if penaltiesActive {
                llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
                    effectivePenaltyWindow,        // last_n tokens to penalize (shared window)
                    effectiveRepetitionPenalty,    // repeat penalty (multiplicative; 1.0 = no-op)
                    effectiveFrequencyPenalty,     // frequency penalty (additive; 0.0 = no-op)
                    effectivePresencePenalty       // presence penalty (additive; 0.0 = no-op)
                ))
            }

            // Grammar-constrained sampling: GBNF grammar from config, inserted at the
            // front of the chain (right after penalties) so it prunes the logit
            // distribution before any probability-based filter narrows the candidate
            // set. Parse failure (invalid GBNF) is surfaced as an error — silent
            // fallback to unconstrained sampling would produce output that violates
            // the caller's grammar contract.
            if includeGrammar, let grammarString = config.grammar {
                var grammarSamplerCreated = false
                grammarString.withCString { grammarCStr in
                    "root".withCString { rootCStr in
                        if let gs = llama_sampler_init_grammar(vocab, grammarCStr, rootCStr) {
                            llama_sampler_chain_add(sampler, gs)
                            grammarSamplerCreated = true
                        }
                    }
                }
                if !grammarSamplerCreated {
                    llama_sampler_free(sampler)
                    return .grammarParseFailed
                }
            }

            if let model = llama_get_model(context),
               let dry = DRYSamplerDescriptor(config: config, nCtxTrain: llama_model_n_ctx_train(model)) {
                let drySampler = Self.withCStringArray(dry.options.sequenceBreakers) { breakers in
                    var mutableBreakers = breakers
                    return mutableBreakers.withUnsafeMutableBufferPointer { breakerBuffer in
                        llama_sampler_init_dry(
                            vocab,
                            dry.nCtxTrain,
                            dry.options.multiplier,
                            dry.options.base,
                            dry.options.allowedLength,
                            dry.options.penaltyLastN,
                            breakerBuffer.baseAddress,
                            breakerBuffer.count
                        )
                    }
                }
                llama_sampler_chain_add(sampler, drySampler)
            }

            // temperature == 0.0 means true greedy decoding: always pick the argmax token.
            // The stochastic `dist` sampler introduces seed-dependent tie-breaking that can
            // produce non-deterministic output when two logits are numerically equal (a
            // realistic occurrence when the KV cache re-decode path uses a different Metal
            // accumulation order than the original full-batch decode). Using the dedicated
            // greedy sampler eliminates that randomness entirely.
            if config.temperature <= 0.0 {
                llama_sampler_chain_add(sampler, llama_sampler_init_greedy())
            } else {
                // Surface `config.topK` to the sampler chain. Historical default of 40 is
                // preserved when the caller leaves it nil, so existing behaviour is unchanged
                // for callers that never set the field (which previously had no effect).
                let effectiveTopK = config.topK.map { Int32($0) } ?? 40
                llama_sampler_chain_add(sampler, llama_sampler_init_top_k(effectiveTopK))
                if config.topP < 1.0 {
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(config.topP, 1))
                }
                // Honour `config.minP` when supplied; default to 0.05 for parity with prior behaviour.
                let effectiveMinP = config.minP ?? 0.05
                llama_sampler_chain_add(sampler, llama_sampler_init_min_p(effectiveMinP, 1))

                // Mirostat v2 owns both the temperature step and the final token selection
                // (it samples internally), so when it is active we skip temp/xtc/dist
                // entirely. When inactive we keep the historical chain tail.
                if let mirostat = MirostatV2SamplerDescriptor(config: config, fallbackSeed: samplerSeed) {
                    llama_sampler_chain_add(sampler, llama_sampler_init_mirostat_v2(
                        mirostat.resolvedSeed,
                        mirostat.options.tau,
                        mirostat.options.eta
                    ))
                } else {
                    llama_sampler_chain_add(sampler, llama_sampler_init_temp(config.temperature))
                    if let xtc = XTCSamplerDescriptor(config: config, fallbackSeed: samplerSeed) {
                        llama_sampler_chain_add(sampler, llama_sampler_init_xtc(
                            xtc.options.probability,
                            xtc.options.threshold,
                            xtc.options.minKeep,
                            xtc.resolvedSeed
                        ))
                    }
                    llama_sampler_chain_add(sampler, llama_sampler_init_dist(samplerSeed))
                }
            }
            return .success(sampler)
        }

        // Strict chain: carries the grammar when the caller supplied one. This is the
        // only chain used for non-thinking models and for thinking-disabled requests,
        // so its construction path is identical to pre-#1595.
        let outputSampler: UnsafeMutablePointer<llama_sampler>
        switch makeSampler(includeGrammar: hasGrammar) {
        case .success(let s):
            outputSampler = s
        case .chainInitFailed:
            await MainActor.run { generationStream.setPhase(.failed("Failed to create sampler")) }
            continuation.finish(throwing: InferenceError.inferenceFailure("Failed to create sampler"))
            return false
        case .grammarParseFailed:
            await MainActor.run { generationStream.setPhase(.failed("Failed to parse GBNF grammar")) }
            continuation.finish(throwing: InferenceError.inferenceFailure("Failed to parse GBNF grammar string"))
            // KV cache state is untouched at this point — no decode has run yet —
            // so the cache is still coherent and the caller can keep `sessionKVState`.
            return true
        }
        defer { llama_sampler_free(outputSampler) }

        // Permissive chain: identical to the strict chain minus the grammar stage.
        // Built only when gating is required (grammar + thinking both active) and
        // used while the model is inside its reasoning block so the schema cannot
        // clamp `<think>…</think>` tokens. Grammar parse failure cannot occur here
        // (includeGrammar == false). If the chain fails to initialise we fall back
        // to the strict chain — grammar would then (incorrectly) constrain reasoning,
        // but that is strictly better than aborting the generation outright.
        let thinkingSampler: UnsafeMutablePointer<llama_sampler>? = {
            guard gateGrammarOnThinking else { return nil }
            switch makeSampler(includeGrammar: false) {
            case .success(let s):
                return s
            case .chainInitFailed, .grammarParseFailed:
                Self.logger.error("Failed to build thinking-phase sampler; grammar will apply during reasoning")
                return nil
            }
        }()
        defer { if let thinkingSampler { llama_sampler_free(thinkingSampler) } }

        // Phase gate: starts permissive when gating, flips to strict on the first
        // `.thinkingCompleted`. A no-op (always strict) when not gating.
        var grammarGate = GrammarPhaseGate(gateOnThinking: gateGrammarOnThinking)

        // MARK: Chunked prompt decode

        // Process prompt in `n_batch`-sized chunks. A single `llama_decode`
        // call cannot exceed `n_batch` tokens, so we stride through the
        // prompt and decode each chunk separately. Only the last token of
        // the final chunk has `logits = 1` — that's the one we sample from
        // to kick off generation.
        //
        // Start at `alignedReuseLen` (the reused prefix floored to a `batchSize`
        // boundary), NOT at the raw `reuseLen`. Because the resume position is a
        // multiple of `batchSize`, the chunk boundaries from here on coincide
        // exactly with the cold path's `[0,B), [B,2B), …` chunking, so the chunk
        // that produces the sampling-position (N-1) logits has bit-identical
        // start/size/positions every turn (ManifoldKit#1677). A mid-chunk resume
        // (raw `reuseLen`) would be a different batch shape and flip the greedy
        // argmax on near-tied logits for non-Qwen architectures. When
        // `alignedReuseLen == 0` this is equivalent to a full re-decode from 0.
        // Adaptive per-model footprint, learned across this prompt's chunks.
        // Stays dormant (guard returns false, estimate nil) until the first
        // accepted sample, so a single-chunk prompt behaves exactly as before.
        var footprintEstimator = PrefillFootprintEstimator()
        var prefillAborted = false

        var promptDecodeFailed = false
        var prefillStart: ContinuousClock.Instant? = nil
        var promptPos = alignedReuseLen
        while promptPos < tokens.count {
            if isCancelled() { break }

            let chunkSize = min(batchSize, tokens.count - promptPos)
            let isLastChunk = (promptPos + chunkSize) == tokens.count

            // Pre-chunk abort guard: if the learned per-token cost predicts this
            // chunk's transient growth (× safety factor) would overrun the free
            // memory remaining right now, decline it and surface
            // `.memoryInsufficient` rather than decoding into a jetsam/Metal kill.
            // No-op until an estimate exists.
            if let remaining = prefillHeadroomSampler(),
               footprintEstimator.wouldExceedHeadroom(
                   remainingBytes: remaining,
                   nextChunkTokens: chunkSize,
                   safetyFactor: prefillSafetyFactor
               ) {
                let required = footprintEstimator.predictedTransientBytes(
                    forTokens: chunkSize,
                    safetyFactor: prefillSafetyFactor
                ) ?? remaining
                Self.logger.error(
                    "Llama prefill aborted: predicted \(required) bytes for next chunk exceeds \(remaining) free"
                )
                prefillAborted = true
                llama_synchronize(context)
                await MainActor.run {
                    generationStream.setPhase(.failed("Insufficient memory for prefill"))
                }
                continuation.finish(
                    throwing: InferenceError.memoryInsufficient(required: required, available: remaining)
                )
                break
            }

            let footprintBefore = prefillFootprintSampler()

            var promptBatch = llama_batch_init(Int32(chunkSize), 0, 1)
            for i in 0..<chunkSize {
                promptBatch.token[i] = tokens[promptPos + i]
                promptBatch.pos[i] = Int32(promptPos + i)
                promptBatch.n_seq_id[i] = 1
                promptBatch.seq_id[i]?[0] = 0
                promptBatch.logits[i] = (isLastChunk && i == chunkSize - 1) ? 1 : 0
            }
            promptBatch.n_tokens = Int32(chunkSize)

            let decodeResult = llama_decode(context, promptBatch)
            llama_batch_free(promptBatch)

            if decodeResult != 0 {
                promptDecodeFailed = true
                break
            }

            // Sample resident footprint after the decode and fold the delta into
            // the EWMA. Negative deltas (allocator/cache reclaim) are rejected by
            // the estimator so they cannot bias the next prediction toward zero.
            let footprintAfter = prefillFootprintSampler()
            footprintEstimator.record(
                beforeBytes: footprintBefore,
                afterBytes: footprintAfter,
                tokensProcessed: chunkSize
            )

            promptPos += chunkSize

            // Emit a prefill-progress heartbeat so the orchestrator can surface
            // progress bars and compute time-to-first-token estimates.
            if prefillStart == nil { prefillStart = ContinuousClock.now }
            let elapsed = ContinuousClock.now - prefillStart!
            let elapsedSeconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
            let tps = elapsedSeconds > 0 ? Double(promptPos) / elapsedSeconds : 0
            continuation.yield(.prefillProgress(
                tokensProcessed: promptPos,
                tokensTotal: tokens.count,
                tokensPerSecond: tps
            ))
        }

        // A pre-chunk abort already finished the continuation with a thrown
        // error; the KV cache holds a partially-decoded prefix, so report it
        // incoherent to force a clean clear on the next turn.
        if prefillAborted {
            return false
        }

        // Surface the learned per-token cost so the backend can retain it and a
        // subsequent load can recompute its `ModelLoadPlan` against a measured
        // budget. Reported only when a real estimate exists.
        if let measured = footprintEstimator.estimatedBytesPerToken {
            onPrefillEstimate?(measured)
        }

        if promptDecodeFailed {
            Self.logger.error("Llama prompt decode failed")
            // Synchronize before returning so any partially-committed Metal
            // command buffers from the prompt chunks that did succeed are
            // flushed.  The next generate() call will clear the KV cache;
            // without this fence that clear can race with in-flight GPU work.
            return await Self.finishDecodeFailure(
                message: "Failed to decode prompt",
                synchronize: { llama_synchronize(context) },
                generationStream: generationStream,
                continuation: continuation
            )
        }

        // Honour cancellation that fired mid-prompt before entering the
        // generation loop.
        if isCancelled() {
            // Flush any Metal ops from prompt chunks already decoded.
            llama_synchronize(context)
            await MainActor.run { generationStream.setPhase(.done) }
            continuation.finish()
            return true
        }

        // MARK: Token generation loop

        // Generation loop uses a fresh 1-capacity batch — the prompt loop
        // allocated and freed a batch per chunk, so there's nothing to
        // reuse here.
        var genBatch = llama_batch_init(1, 0, 1)
        defer { llama_batch_free(genBatch) }

        // The chunked prompt loop placed tokens at positions
        // [0, tokens.count - 1]; the next decoded token goes at
        // `tokens.count`.
        var nCur = tokens.count
        var invalidUTF8: [CChar] = []
        var isFirstToken = true

        // Thinking-marker handling has two modes:
        //
        // 1. Eager — `markers != nil` (caller passed explicit markers, or the
        //    backend auto-detected them from the GGUF chat template). Every
        //    decoded token flows through `ThinkingTransform` from the first byte.
        //
        // 2. Disabled — `markers == nil`. The model does not advertise reasoning
        //    blocks, so every token yields `.token` with no tag scanning. Raw
        //    `<think>` substrings (if the model emits them anyway) surface as
        //    visible text rather than `.thinkingToken` events. This matches
        //    non-reasoning models' fast path. Removed in this change: an
        //    earlier "sniff" mode that probed the first 64 bytes for `<think>`
        //    and retroactively engaged the parser. Auto-detection at load time
        //    (LlamaModelLoader.readChatTemplateMetadata) replaced it.
        //
        // Special case: `config.maxThinkingTokens == 0` disables thinking entirely
        // (issue #597). Even when `markers` is non-nil, the parser stays off and
        // every decoded token flows straight to `.token`. The model may still
        // emit raw `<think>` / `</think>` substrings, but the driver routes
        // them as visible text rather than `.thinkingToken` events.
        // `thinkingDisabled` / `useParser` were computed once during sampler setup
        // above (the grammar gate needs them before the chains are built); reuse
        // those values here rather than re-deriving them.
        var thinkingTokenCount = 0
        // Flag set when maxThinkingTokens is reached so we can break the outer loop cleanly.
        var thinkingLimitReached = false
        // Flag set when the generation loop exits due to cancellation. Usage must NOT
        // be emitted for cancelled turns — no complete turn happened.
        var exitedDueToCancellation = false
        // Visible-output token counter. Thinking tokens do NOT count toward maxTokens —
        // maxOutputTokens governs visible output budget, and maxThinkingTokens governs the
        // reasoning budget separately. Without this split, a reasoning model that thinks for
        // N tokens before answering consumes the entire maxOutputTokens budget in the thinking
        // phase and never emits any visible output (see issue #519 regression: Qwen3-0.6B
        // exhausted a 256-token budget in <think>…</think> and never reached EOG).
        var visibleTokenCount = 0
        // Total loop iterations = visible budget + thinking budget. When thinking is
        // disabled (useParser == false) the thinking budget is 0, so the loop cap equals
        // maxTokens exactly — identical to the previous behaviour for non-thinking models.
        //
        // The thinking budget is capped to the remaining context slots so the driver
        // never runs llama_decode past the context window even when maxThinkingTokens
        // is nil (defaulting to maxTokens). With a small context (e.g. 512 tokens in
        // tests) and a large prompt, the effective budget shrinks accordingly.
        let contextCapacity = Int(llama_n_ctx(context))
        let usedSlots = tokens.count  // prompt occupies this many KV slots already
        let totalLoopBudget = Self.thinkingLoopBudget(
            contextCapacity: contextCapacity,
            usedSlots: usedSlots,
            useParser: useParser,
            maxThinkingTokens: config.maxThinkingTokens,
            maxTokens: maxTokens
        ).totalLoopBudget

        // Tool-call parser: active when the config carries at least one tool definition.
        // Processes `.token` events from the thinking layer (or raw decoded text) and
        // re-routes `<|tool_call>…<|end_of_turn>` / `<tool_call>…</tool_call>` blocks
        // into `.toolCall` events. When no tools are configured the parser is skipped
        // entirely — zero overhead on non-tool-calling generation paths.
        let useToolParser = !config.tools.isEmpty

        // Build the unified output-parsing chain. Order is `[thinking, tool]`:
        // thinking is stripped first, then the tool stage re-scans the remaining
        // visible `.token` text — tool calls never appear inside a thinking
        // block, so the tool stage only ever sees post-thinking text. Preserves
        // Llama's historical two-stage order exactly.
        var stages: [Stage] = []
        if useParser {
            stages.append(.thinking(ThinkingTransform(markers: markers ?? .qwen3)))
        }
        if useToolParser {
            stages.append(.tool(ToolCallTransform(markers: LlamaToolMarkers.markers(), surfaceTruncatedToolBody: true)))
        }
        var session = OutputParserSession(stages)

        // Stop-sequence enforcement (feature-completeness fix). `config.stopSequences`
        // was plumbed end-to-end but this driver previously dropped it: generation
        // only ever halted on an EOG *token ID*, never on generated *text*. Non-ChatML
        // small instruct models leak ChatML/other-family control strings as plain
        // text (no matching EOG token), so without this they run on and fabricate
        // fake turns. We union the caller's stops with a default cross-family
        // control-marker set (`resolveStopSequences`) and run the streamed visible
        // and thinking text through a hold-back matcher so a marker spanning multiple
        // tokens is never emitted and the output is truncated exactly at the boundary.
        // Separate matchers per channel: a marker does not meaningfully straddle a
        // thinking→visible boundary, and visible text is where leaked markers appear.
        let resolvedStops = Self.resolveStopSequences(config.stopSequences)
        var tokenStopMatcher = StopSequenceMatcher(stops: resolvedStops)
        var thinkingStopMatcher = StopSequenceMatcher(stops: resolvedStops)
        // Set once a stop sequence matched so the loop breaks after flushing the
        // pending emit for the current chunk. Termination is treated exactly like
        // EOG: break out, fall through to the normal `.done` phase + usage emission.
        var stopSequenceHit = false

        // Repetition-window state: track the last decoded token string and how
        // many times it has appeared consecutively. Exceeding `maxRepeatWindow`
        // triggers an early exit — no need to run the loop all the way to maxTokens.
        var repeatWindow = RepeatWindow(limit: Self.maxRepeatWindow)

        // Phrase-level repetition state: a bounded sliding window (Array, evicted
        // via removeFirst — O(n) but cap=61 so cost is negligible) of the last
        // `phraseWindowCap` decoded token strings. After each token is appended,
        // the tail is scanned for back-to-back phrase repetitions of lengths 2–20.
        var phraseWindow: [String] = []
        phraseWindow.reserveCapacity(Self.phraseWindowCap + 1)

        generationLoop: for iteration in 0..<totalLoopBudget {
            if isCancelled() { exitedDueToCancellation = true; break }

            // First iteration samples from the final prompt chunk's logits,
            // which llama.cpp exposes at index -1 ("last available").
            // Subsequent iterations sample from the 1-token gen batch
            // decoded at the end of the previous iteration, at index 0.
            let logitIndex: Int32 = iteration == 0 ? -1 : 0
            // Phase-aware grammar (issue #1595): sample with the permissive chain
            // while the model is reasoning, and with the strict (grammar) chain once
            // `.thinkingCompleted` has flipped the gate. `thinkingSampler` is non-nil
            // only when gating is active and grammar is inactive, so the fallback to
            // `outputSampler` covers every other case (no gating, or already strict).
            let sampler = grammarGate.isGrammarActive ? outputSampler : (thinkingSampler ?? outputSampler)
            let token = llama_sampler_sample(sampler, context, logitIndex)

            if llama_vocab_is_eog(vocab, token) { break }

            // Decode token to text and route through ThinkingTransform when active.
            if let text = LlamaTokenization.tokenToString(token, vocab: vocab, invalidUTF8Buffer: &invalidUTF8) {
                // Single-token repetition guard: identical-token run of ≥maxRepeatWindow
                // terminates the loop. Catches small-model repetition loops (e.g.
                // smollm2-135m emitting " " hundreds of times) before the post-hoc
                // LoopingDetector has to clean them up.
                if repeatWindow.observe(text) {
                    break generationLoop
                }

                // Phrase-level repetition guard: catch multi-token loops (2–20 tokens)
                // that the single-token window misses. Live fuzz runs on smollm2-135m
                // surfaced loops with repeating units such as ASCII-art phrases, HTML
                // timestamp blocks, and RTL override sequences.
                phraseWindow.append(text)
                if phraseWindow.count > Self.phraseWindowCap {
                    phraseWindow.removeFirst()
                }
                let maxScanLen = min(Self.maxPhraseLen, phraseWindow.count / Self.minPhraseRepeats)
                if maxScanLen >= 2 {
                    for phraseLen in 2...maxScanLen {
                        if Self.tailRepeats(phraseWindow, phraseLen: phraseLen, minRepeats: Self.minPhraseRepeats) {
                            break generationLoop
                        }
                    }
                }

                // Route the decoded token through the unified chain. The session
                // runs the thinking transform then the tool transform in order;
                // tool calls never appear inside thinking blocks, so the tool
                // stage only sees post-thinking visible text.
                let events = session.ingest(text)

                // Advance the grammar phase gate: once the reasoning block closes
                // (`.thinkingCompleted`), the strict chain takes over for the next
                // sampled token — the first real output token. No-op when not gating.
                grammarGate.observe(events)

                var visibleBudgetExceeded = false
                for event in events {
                    // Route text-bearing events through the stop-sequence matcher
                    // (no-op when no stops are configured — `isEmpty` short-circuits
                    // before any allocation). The matcher holds back a tail so a
                    // marker spanning multiple tokens is never streamed, and reports
                    // when a stop boundary was crossed; the matched marker and all
                    // trailing text are dropped. Non-text events pass through
                    // unchanged. We still account the *source* token against the
                    // thinking/visible budgets so decode-loop accounting is unchanged
                    // even when the matcher releases empty/partial text.
                    let yielded: GenerationEvent?
                    switch event {
                    case .token(let s) where !tokenStopMatcher.isEmpty:
                        let (emit, stopped) = tokenStopMatcher.push(s)
                        if stopped { stopSequenceHit = true }
                        yielded = emit.isEmpty ? nil : .token(emit)
                    case .thinkingToken(let s) where !thinkingStopMatcher.isEmpty:
                        let (emit, stopped) = thinkingStopMatcher.push(s)
                        if stopped { stopSequenceHit = true }
                        yielded = emit.isEmpty ? nil : .thinkingToken(emit)
                    default:
                        yielded = event
                    }

                    if isFirstToken, let yielded {
                        switch yielded {
                        case .token, .thinkingToken:
                            // Trigger .streaming on first reasoning token too — models can think
                            // for 30s before any visible output; staying in .connecting is poor UX.
                            await MainActor.run { generationStream.setPhase(.streaming) }
                            isFirstToken = false
                        default: break
                        }
                    }
                    if let yielded { continuation.yield(yielded) }
                    switch event {
                    case .thinkingToken:
                        thinkingTokenCount += 1
                        if let limit = config.maxThinkingTokens, thinkingTokenCount >= limit {
                            thinkingLimitReached = true
                        }
                    case .token:
                        visibleTokenCount += 1
                        if visibleTokenCount >= maxTokens {
                            visibleBudgetExceeded = true
                        }
                    default: break
                    }
                    if stopSequenceHit || thinkingLimitReached || visibleBudgetExceeded { break }
                }
                if stopSequenceHit || thinkingLimitReached || visibleBudgetExceeded { break generationLoop }
            }

            // Prepare next batch
            genBatch.n_tokens = 0
            genBatch.token[0] = token
            genBatch.pos[0] = Int32(nCur)
            genBatch.n_seq_id[0] = 1
            genBatch.seq_id[0]?[0] = 0
            genBatch.logits[0] = 1
            genBatch.n_tokens = 1
            nCur += 1

            if isCancelled() { exitedDueToCancellation = true; break }

            if llama_decode(context, genBatch) != 0 {
                // Synchronize before surfacing the error so the GPU drains any
                // work that *did* commit before the failure.  Without this, a
                // subsequent KV-cache clear from a retry can race Metal ops.
                return await Self.finishDecodeFailure(
                    message: "Decode failed during generation",
                    synchronize: { llama_synchronize(context) },
                    generationStream: generationStream,
                    continuation: continuation
                )
            }
        }

        // Flush any bytes held back by the chain's tag-boundary buffers. The
        // session cascades each stage's finalize output through the stages
        // downstream of it (thinking → tool), and is a no-op when the chain is
        // empty (neither thinking nor tool parsing was engaged).
        //
        // These finalize-flushed events must still pass through the stop-sequence
        // matchers: a marker could sit entirely inside the chain's held-back tail.
        // When a stop already matched (`stopSequenceHit`) the matchers have latched
        // and release nothing, so no post-stop text leaks here either.
        for event in session.finalize() {
            switch event {
            case .token(let s) where !tokenStopMatcher.isEmpty:
                let (emit, _) = tokenStopMatcher.push(s)
                if !emit.isEmpty { continuation.yield(.token(emit)) }
            case .thinkingToken(let s) where !thinkingStopMatcher.isEmpty:
                let (emit, _) = thinkingStopMatcher.push(s)
                if !emit.isEmpty { continuation.yield(.thinkingToken(emit)) }
            default:
                continuation.yield(event)
            }
        }

        // Release each matcher's held-back tail. By construction the tail is only
        // ever a *partial* stop prefix (a full match would have latched + dropped
        // it), so at end of stream it is genuine output and is emitted verbatim.
        // No-op when stopped, when no stops are configured, or when nothing is held.
        let flushedToken = tokenStopMatcher.flush()
        if !flushedToken.isEmpty { continuation.yield(.token(flushedToken)) }
        let flushedThinking = thinkingStopMatcher.flush()
        if !flushedThinking.isEmpty { continuation.yield(.thinkingToken(flushedThinking)) }

        // Flush all pending Metal command buffers before returning.
        //
        // `llama_decode` enqueues Metal compute passes asynchronously — by the
        // time the generation loop exits (EOG, cancellation, or token budget),
        // the GPU may still be executing the last batch.  If the caller
        // immediately starts a new `generate()` call, the KV-cache clear
        // (`llama_memory_clear` / `llama_memory_seq_rm`) at the top of the next
        // run enqueues *new* Metal operations before the previous command buffers
        // have committed.  That ordering violation trips llama.cpp's internal
        // render-set assertion:
        //
        //   GGML_ASSERT([rsets->data count] == 0)   (ggml-metal-device.m:618)
        //
        // which leads to corrupted KV state and, on the very next `llama_decode`,
        // a Swift array bounds crash (ContiguousArrayBuffer.swift:692).
        // `llama_synchronize` blocks the calling thread until the GPU is idle,
        // making the context safe for the next caller.  The call is cheap when
        // the GPU is already done (typically sub-millisecond) and is the
        // authoritative fix recommended by the llama.cpp contract (see
        // `docs/LLAMA_CONTRACT.md` and `llama.h` line 972).
        llama_synchronize(context)

        await MainActor.run { generationStream.setPhase(.done) }
        Self.logger.debug("LlamaGenerationDriver run finished")
        // Emit usage only on a complete turn — not when the loop exited due to
        // cancellation (user-initiated stop mid-stream is not a finished turn).
        if !exitedDueToCancellation {
            onUsage?(tokens.count, visibleTokenCount)
            continuation.yield(.usage(TokenUsage(promptTokens: tokens.count, completionTokens: visibleTokenCount)))
        }
        continuation.finish()
        return true
    }

    // MARK: - Decode-failure contract (model-free, seam-tested)

    /// Centralises the decode-failure teardown sequence shared by the prompt-chunk
    /// and generation-loop decode-error sites. The ordering is load-bearing and is
    /// the whole reason this is one function instead of two inlined blocks:
    ///
    ///   1. `synchronize()` FIRST — drains any Metal command buffers that *did*
    ///      commit before the failing `llama_decode`. Skipping this lets the next
    ///      turn's KV-cache clear enqueue new Metal ops while old buffers are still
    ///      in flight, tripping llama.cpp's render-set assertion and a downstream
    ///      Swift array bounds crash (see the synchronize comment in `run`).
    ///   2. set the stream phase to `.failed`.
    ///   3. finish the continuation with `.inferenceFailure` so the consumer sees
    ///      a thrown error rather than a silent end-of-stream.
    ///   4. return `false` so the caller clears `sessionKVState` — the KV cache is
    ///      undefined after a failed decode and the prefix must not be reused.
    ///
    /// `synchronize` is injected so the contract can be exercised headlessly: the
    /// real call sites pass `{ llama_synchronize(context) }`, tests pass a recorder
    /// that captures call order without a live `llama_context`.
    @_spi(Testing) public static func finishDecodeFailure(
        message: String,
        synchronize: () -> Void,
        generationStream: GenerationStream,
        continuation: AsyncThrowingStream<GenerationEvent, Error>.Continuation
    ) async -> Bool {
        synchronize()
        await MainActor.run { generationStream.setPhase(.failed(message)) }
        continuation.finish(throwing: InferenceError.inferenceFailure(message))
        return false
    }

    // MARK: - Pure decision helpers (model-free, unit-tested)

    /// Batch-aligned KV-reuse floor. The reused prefix is floored to a `batchSize`
    /// multiple and capped below the final chunk's start so the N-1 chunk is always
    /// re-decoded (it must carry `logits = 1` for sampling). See the call site for
    /// the full rationale.
    @_spi(Testing) public static func alignedKVReuseLength(
        tokenCount: Int, reuseLen: Int, batchSize: Int
    ) -> Int {
        let safeBatch = max(1, batchSize)
        let finalChunkStart = tokenCount == 0 ? 0 : ((tokenCount - 1) / safeBatch) * safeBatch
        return min((reuseLen / safeBatch) * safeBatch, finalChunkStart)
    }

    /// Computes the thinking-token budget and the total decode-loop budget, clamped
    /// so the loop never runs `llama_decode` past the remaining context window even
    /// when `maxThinkingTokens` is nil (defaulting to `maxTokens`).
    @_spi(Testing) public static func thinkingLoopBudget(
        contextCapacity: Int, usedSlots: Int, useParser: Bool,
        maxThinkingTokens: Int?, maxTokens: Int
    ) -> (effectiveThinkingBudget: Int, totalLoopBudget: Int) {
        let remainingSlots = max(0, contextCapacity - usedSlots)
        let rawThinkingBudget = useParser ? (maxThinkingTokens ?? maxTokens) : 0
        let effectiveThinkingBudget = min(rawThinkingBudget, max(0, remainingSlots - maxTokens))
        return (effectiveThinkingBudget, maxTokens + effectiveThinkingBudget)
    }

    /// Single-token repetition guard. Tracks the last decoded token string and how
    /// many times it has appeared consecutively; `observe` returns `true` once the
    /// run reaches `limit`, signalling the decode loop to break.
    @_spi(Testing) public struct RepeatWindow {
        public let limit: Int
        private var last = ""
        private var count = 0

        public init(limit: Int) { self.limit = limit }

        public mutating func observe(_ text: String) -> Bool {
            if text == last {
                count += 1
                return count >= limit
            } else {
                last = text
                count = 1
                return false
            }
        }
    }

    // MARK: - Phrase Detection

    /// Returns true when the tail of `window` contains `minRepeats` consecutive
    /// identical phrases of length `phraseLen`.
    @_spi(Testing) public static func tailRepeats(_ window: [String], phraseLen: Int, minRepeats: Int) -> Bool {
        let needed = phraseLen * minRepeats
        guard window.count >= needed else { return false }
        let n = window.count
        let phrase = window[(n - phraseLen)...]
        for rep in 1..<minRepeats {
            let start = n - phraseLen * (rep + 1)
            let end   = n - phraseLen * rep
            guard start >= 0 else { return false }
            if window[start..<end].elementsEqual(phrase) == false { return false }
        }
        return true
    }

    // MARK: - Stop sequences

    /// Default cross-family turn/control terminator strings, enforced as stop
    /// sequences regardless of the active prompt template.
    ///
    /// Rationale (issue: stop sequences never enforced): the template-derived stop
    /// set only contains the loaded model's *native* turn markers. ChatML training
    /// contamination means non-ChatML small instruct models (Mistral `[INST]`,
    /// Gemma `<end_of_turn>`, Llama-3 `<|eot_id|>`) frequently emit *foreign* control
    /// strings — e.g. `<|im_end|>`, `<|im_start|>`, `[tool_call]` — as PLAIN TEXT
    /// rather than as real EOG token IDs. `llama_vocab_is_eog` never fires on those,
    /// so without a text-level stop the model runs on and fabricates entire fake
    /// multi-turn conversations and bogus tool calls.
    ///
    /// Membership criteria: every entry is a literal control/turn-delimiter marker
    /// that should never appear in legitimate assistant prose. We deliberately
    /// EXCLUDE bare common words and anything that could occur in normal output. The
    /// set spans the major instruct families:
    ///   - ChatML:   `<|im_end|>`, `<|im_start|>`
    ///   - Gemma:    `<end_of_turn>`
    ///   - Llama-3:  `<|eot_id|>`, `<|end_of_text|>`
    ///   - Mistral/Llama-2: `</s>`, `[/INST]`
    ///   - leaked tool-call open marker: `[tool_call]`
    @_spi(Testing) public static let defaultControlMarkerStops: [String] = [
        "<|im_end|>",
        "<|im_start|>",
        "<end_of_turn>",
        "<|eot_id|>",
        "<|end_of_text|>",
        "</s>",
        "[/INST]",
        "[tool_call]",
    ]

    /// Resolves the effective stop-sequence set: the union of the caller's
    /// `config.stopSequences` and the default cross-family control markers, with
    /// empties removed and duplicates collapsed. Order is not significant — the
    /// matcher always prefers the longest match at a given position.
    @_spi(Testing) public static func resolveStopSequences(_ configured: [String]) -> [String] {
        var seen = Set<String>()
        var result: [String] = []
        for s in configured + defaultControlMarkerStops where !s.isEmpty {
            if seen.insert(s).inserted { result.append(s) }
        }
        return result
    }

    /// Streaming stop-sequence matcher with multi-token-spanning hold-back.
    ///
    /// A stop sequence can straddle several decoded tokens, so this type buffers a
    /// tail of pending characters and only releases the portion that provably
    /// cannot become the prefix of any stop string. The hold-back length is
    /// bounded by `longest stop − 1` characters: anything older than that cannot be
    /// the start of an in-progress match.
    ///
    /// Matching operates on Swift `Character`s (extended grapheme clusters), the
    /// same unit `String` indexing and the streamed text use, so a marker is never
    /// split mid-grapheme. When a stop is found the matched marker and everything
    /// after it are dropped — they are never released to the consumer — and
    /// `isStopped` latches `true`.
    ///
    /// Pure and model-free: all decode-loop callers feed it decoded text chunks and
    /// emit only what `push`/`flush` release.
    @_spi(Testing) public struct StopSequenceMatcher {
        /// Stop strings as character arrays (precomputed for prefix scanning).
        private let stops: [[Character]]
        /// `longestStop − 1`; the max number of trailing chars we must retain to
        /// detect a match that completes on a future chunk. Zero when no stops.
        private let maxHoldback: Int
        /// Pending un-released characters (kept ≤ maxHoldback after each `push`,
        /// except transiently while we scan a freshly appended chunk).
        private var buffer: [Character] = []
        /// Latches once a stop has been matched; further `push`es release nothing.
        public private(set) var isStopped = false

        /// `true` when there are no stop strings to enforce — callers can take a
        /// zero-overhead passthrough path.
        public var isEmpty: Bool { stops.isEmpty }

        public init(stops: [String]) {
            let nonEmpty = stops.filter { !$0.isEmpty }
            self.stops = nonEmpty.map { Array($0) }
            self.maxHoldback = max(0, (nonEmpty.map { $0.count }.max() ?? 0) - 1)
        }

        /// Feeds a decoded text chunk. Returns the text that is safe to emit now —
        /// i.e. everything up to (but excluding) any matched stop, minus a held-back
        /// tail that could still be the start of a stop on the next chunk. Once a
        /// stop has matched (or already matched on a prior chunk) the returned text
        /// stops exactly at the boundary and `isStopped` is `true`.
        public mutating func push(_ chunk: String) -> (emit: String, stopped: Bool) {
            if stops.isEmpty { return (chunk, false) }
            if isStopped { return ("", true) }
            buffer.append(contentsOf: chunk)

            // Scan for the earliest position at which any stop matches in full.
            // Prefer the LONGEST match at that earliest position so overlapping
            // stops (e.g. "</s>" vs "</s>!") resolve deterministically to the
            // longest marker; the trailing chars are dropped either way.
            var matchStart: Int? = nil
            var matchLen = 0
            var i = 0
            while i < buffer.count {
                var bestLenHere = 0
                for stop in stops where matches(buffer, at: i, stop: stop) {
                    bestLenHere = max(bestLenHere, stop.count)
                }
                if bestLenHere > 0 {
                    matchStart = i
                    matchLen = bestLenHere
                    break
                }
                i += 1
            }

            if let start = matchStart {
                // Emit everything before the marker; discard the marker and all
                // trailing buffered text (it is part of / past the stop boundary).
                let emit = String(buffer[0..<start])
                buffer.removeAll(keepingCapacity: false)
                isStopped = true
                _ = matchLen  // marker length is irrelevant once we cut at `start`
                return (emit, true)
            }

            // No full match. Release everything except a tail short enough to be a
            // partial stop prefix. We only need to retain a suffix that is a prefix
            // of some stop; the maximum such suffix length is `maxHoldback`, but we
            // can release more aggressively by retaining only as many trailing chars
            // as actually form a stop prefix.
            let retain = longestStopPrefixSuffixLength()
            let releaseCount = buffer.count - retain
            let emit = String(buffer[0..<releaseCount])
            buffer.removeFirst(releaseCount)
            return (emit, false)
        }

        /// End-of-stream flush. Whatever remains in the buffer is, by construction,
        /// only ever a *partial* stop prefix (a full match would have stopped us),
        /// and no more text is coming — so it is genuine output and is released
        /// verbatim. Returns "" once stopped or when nothing is held back.
        public mutating func flush() -> String {
            if isStopped { return "" }
            let remaining = String(buffer)
            buffer.removeAll(keepingCapacity: false)
            return remaining
        }

        /// Returns true when `stop` occurs in `buffer` starting exactly at `index`.
        private func matches(_ buffer: [Character], at index: Int, stop: [Character]) -> Bool {
            guard index + stop.count <= buffer.count else { return false }
            for k in 0..<stop.count where buffer[index + k] != stop[k] {
                return false
            }
            return true
        }

        /// Length of the longest buffer suffix that is a proper prefix of some stop
        /// string (and therefore must be held back in case it completes next chunk).
        /// Bounded by `maxHoldback`. Returns 0 when no suffix can extend into a stop.
        private func longestStopPrefixSuffixLength() -> Int {
            let cap = min(maxHoldback, buffer.count)
            // Try the longest candidate suffix first so we hold back the minimum
            // necessary while never releasing a char that could start a match.
            var len = cap
            while len > 0 {
                let suffixStart = buffer.count - len
                for stop in stops where len < stop.count {
                    var isPrefix = true
                    for k in 0..<len where buffer[suffixStart + k] != stop[k] {
                        isPrefix = false
                        break
                    }
                    if isPrefix { return len }
                }
                len -= 1
            }
            return 0
        }
    }

    /// Builds a transient `[UnsafePointer<CChar>?]` for `strings` and hands it to
    /// `body`, guaranteeing every pointer stays valid for the duration of the call
    /// via nested `withCString` scopes. Exposed for testing because its
    /// pointer-lifetime/`defer`-unwind semantics are subtle and otherwise only
    /// reachable through the live DRY-sampler path (which needs a loaded model).
    @_spi(Testing) public static func withCStringArray<Result>(
        _ strings: [String],
        _ body: ([UnsafePointer<CChar>?]) -> Result
    ) -> Result {
        var pointers: [UnsafePointer<CChar>?] = []

        func append(_ index: Int) -> Result {
            guard index < strings.count else {
                return body(pointers)
            }

            return strings[index].withCString { pointer in
                pointers.append(pointer)
                defer { pointers.removeLast() }
                return append(index + 1)
            }
        }

        return append(0)
    }
}
