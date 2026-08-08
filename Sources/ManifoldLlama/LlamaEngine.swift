import Foundation
import LlamaSwift
import ManifoldContract
import ManifoldInference

/// Opaque handle to a llama.cpp sampler chain.
///
/// The real engine wraps a `UnsafeMutablePointer<llama_sampler>`; a scripted
/// test engine identifies its chains by `id` alone and leaves `rawPointer` nil.
/// Keeping the driver's sampler references behind this box is what lets the
/// generation loop run with no llama.cpp state at all.
// @_spi(Testing): published only for backend test targets (companion-package split, #1749).
@_spi(Testing) public struct LlamaSamplerHandle: @unchecked Sendable {
  public let rawPointer: UnsafeMutableRawPointer?

  /// Scripted-engine identity only. `LlamaCAPIEngine` never sets this (it
  /// identifies chains by `rawPointer`), so it is always `0` in production —
  /// it exists so a test double can tell its chains apart and assert that
  /// every chain it handed out was freed.
  public let id: Int

  public init(rawPointer: UnsafeMutableRawPointer?, id: Int = 0) {
    self.rawPointer = rawPointer
    self.id = id
  }
}

/// Result of building one sampler chain.
///
/// `chainInitFailed` and `grammarParseFailed` map to the two distinct error
/// paths ``LlamaGenerationDriver/run(engine:tokens:reuseLen:maxTokens:config:markers:isCancelled:generationStream:continuation:prefillFootprintSampler:prefillHeadroomSampler:prefillSafetyFactor:onPrefillEstimate:onUsage:onToken:onError:)``
/// surfaces (different message, different KV-coherence return value), so
/// collapsing them would lose that fidelity. Both are unreachable with a real
/// model on a healthy machine — a scripted engine is the only way to drive
/// them (#165).
@_spi(Testing) public enum LlamaSamplerBuildOutcome: Sendable {
  case success(LlamaSamplerHandle)
  case chainInitFailed
  case grammarParseFailed
}

/// The llama.cpp C-API surface ``LlamaGenerationDriver`` needs to run one
/// generation.
///
/// This is the seam that makes `LlamaBackend.generate()`'s Task body — and the
/// driver's whole decode/sample loop — exercisable with no `.gguf` and no Metal
/// (#165). It is the protocol-witness counterpart to the injected-closure
/// approach the driver already uses for `prefillFootprintSampler` /
/// `onToken` / `onError`, and mirrors ManifoldKit's
/// `SSEGenerationTaskRunner`, whose stream parser is injected for exactly the
/// same reason.
///
/// ``LlamaCAPIEngine`` is the one production implementation; every method on it
/// is a verbatim transcription of the C calls that used to be inline in the
/// driver, so the text-only generation path is behaviourally unchanged (same
/// batch shapes, same KV reuse, same sampler chain order).
///
/// Ownership: the engine owns any C resources it allocates. The driver calls
/// ``freeSampler(_:)`` for every handle ``makeSampler(config:seed:includeGrammar:)``
/// returned, and ``releaseDecodeResources()`` exactly once before returning.
@_spi(Testing) public protocol LlamaEngine: AnyObject, Sendable {

  /// `llama_n_batch` — the maximum number of tokens a single decode may carry.
  /// Also the alignment unit for KV-prefix reuse (ManifoldKit#1677).
  var batchSize: Int { get }

  /// `llama_n_ctx` — total KV slots available in this context.
  var contextCapacity: Int { get }

  /// Applies the driver's KV-reuse decision: keep the first
  /// `alignedPrefixLength` cells and trim the tail (`llama_memory_seq_rm`)
  /// when positive, otherwise clear the whole cache (`llama_memory_clear`).
  func applyKVReuse(alignedPrefixLength: Int)

  /// Builds one sampler chain for `config`. `includeGrammar` is `false` for
  /// the permissive chain used during a gated thinking phase (#1595).
  func makeSampler(config: GenerationConfig, seed: UInt32, includeGrammar: Bool)
    -> LlamaSamplerBuildOutcome

  /// Releases a chain previously returned by ``makeSampler(config:seed:includeGrammar:)``.
  func freeSampler(_ handle: LlamaSamplerHandle)

  /// `llama_sampler_sample` at `logitIndex` (`-1` = last available logits).
  func sample(_ handle: LlamaSamplerHandle, logitIndex: Int32) -> llama_token

  /// `llama_vocab_is_eog`.
  func isEndOfGeneration(_ token: llama_token) -> Bool

  /// Detokenizes one token, accumulating any incomplete UTF-8 bytes in
  /// `invalidUTF8Buffer` across calls (see ``LlamaTokenization/tokenToString(_:vocab:invalidUTF8Buffer:)``).
  func tokenToString(_ token: llama_token, invalidUTF8Buffer: inout [CChar]) -> String?

  /// Decodes one prompt chunk of at most ``batchSize`` tokens starting at
  /// `startPosition`. `logitsOnLastToken` marks the final token of the final
  /// chunk so the sampler has logits to read. Returns `llama_decode`'s status
  /// (`0` = success).
  func decodePromptChunk(
    tokens: ArraySlice<llama_token>, startPosition: Int, logitsOnLastToken: Bool
  ) -> Int32

  /// Decodes the single freshly sampled token at `position`, always with
  /// `logits = 1`. Returns `llama_decode`'s status (`0` = success).
  func decodeGeneratedToken(_ token: llama_token, position: Int) -> Int32

  /// `llama_synchronize` — blocks until the GPU has drained.
  func synchronize()

  /// Frees the reusable generation batch. Called exactly once per `run(...)`.
  func releaseDecodeResources()
}

/// The pair of llama.cpp-touching steps ``LlamaBackend/generate(prompt:systemPrompt:config:hints:)``
/// delegates so its orchestration can be driven headlessly (#165).
///
/// Production never constructs one of these — `generate()` falls back to
/// ``LlamaTokenization/tokenize(_:vocab:addBos:parseSpecial:)`` and
/// ``LlamaCAPIEngine`` when no seam is installed. See
/// ``LlamaBackend/installGenerationSeamForTesting(_:)`` for what deliberately
/// stays on the real path in between.
@_spi(Testing) public struct LlamaGenerationSeam: Sendable {

  /// Stands in for the `LlamaTokenization.tokenize(prompt, vocab:, addBos: true)`
  /// call `generate()` makes before the Task is created. Receives the same
  /// vocab pointer snapshot the production call would, which a scripted seam
  /// must not dereference (it is the addr-1 sentinel from
  /// ``LlamaBackend/armFakeLoadedStateForTesting()``).
  public var tokenize: @Sendable (String, OpaquePointer) -> [llama_token]

  /// Builds the engine handed to ``LlamaGenerationDriver``, from the same
  /// `(context, vocab)` pointer pair re-read under `stateLock` inside the
  /// generation Task. Again: sentinel pointers, never dereference them.
  public var makeEngine: @Sendable (OpaquePointer, OpaquePointer) -> any LlamaEngine

  public init(
    tokenize: @escaping @Sendable (String, OpaquePointer) -> [llama_token],
    makeEngine: @escaping @Sendable (OpaquePointer, OpaquePointer) -> any LlamaEngine
  ) {
    self.tokenize = tokenize
    self.makeEngine = makeEngine
  }
}

/// The production ``LlamaEngine``: a direct, stateless-apart-from-the-batch
/// wrapper over the llama.cpp C API.
///
/// Every method body is the code that previously lived inline in
/// ``LlamaGenerationDriver/run(engine:tokens:reuseLen:maxTokens:config:markers:isCancelled:generationStream:continuation:prefillFootprintSampler:prefillHeadroomSampler:prefillSafetyFactor:onPrefillEstimate:onUsage:onToken:onError:)``,
/// moved without modification. The `context` / `vocab` pointers are borrowed —
/// they are snapshotted under `LlamaBackend`'s `stateLock` and stay valid for
/// the lifetime of the generation Task, which `unloadModel()` awaits before
/// freeing them.
@_spi(Testing) public final class LlamaCAPIEngine: LlamaEngine, @unchecked Sendable {

  private let context: OpaquePointer
  private let vocab: OpaquePointer

  /// Reusable 1-capacity batch for the generation loop, allocated on first
  /// use and freed by ``releaseDecodeResources()``. The prompt loop allocates
  /// and frees a batch per chunk (chunk sizes vary), so there is nothing to
  /// share with it.
  private var generationBatch: llama_batch?

  public init(context: OpaquePointer, vocab: OpaquePointer) {
    self.context = context
    self.vocab = vocab
  }

  /// Safety net for ``generationBatch``. The contract is that
  /// ``LlamaGenerationDriver/run(engine:...)`` calls
  /// ``releaseDecodeResources()`` on every exit path via a `defer` — but that
  /// is a convention the type cannot enforce, and an engine constructed
  /// outside `run(...)` would otherwise leak the `llama_batch`. Freeing here
  /// too is safe: ``releaseDecodeResources()`` nils the field, so the normal
  /// path leaves nothing for this to double-free.
  ///
  /// Only the batch is owned. `context` and `vocab` are borrowed — they are
  /// freed by `LlamaBackend.unloadModel()`, which awaits the generation task
  /// first.
  deinit {
    if let batch = generationBatch {
      llama_batch_free(batch)
    }
  }

  public var batchSize: Int { max(1, Int(llama_n_batch(context))) }

  public var contextCapacity: Int { Int(llama_n_ctx(context)) }

  public func applyKVReuse(alignedPrefixLength: Int) {
    guard let memory = llama_get_memory(context) else { return }
    if alignedPrefixLength > 0 {
      // Keep the batch-aligned reused prefix's KV cells; trim only the
      // tail beyond it. Running this inside the generation Task is
      // lifecycle-safe: all context-touching work is serialized with
      // unloadModel() via the task install (see the pointer re-read in
      // LlamaBackend.generate), exactly as the full clear path is.
      llama_memory_seq_rm(memory, 0, Int32(alignedPrefixLength), -1)
    } else {
      llama_memory_clear(memory, false)
    }
  }

  public func makeSampler(
    config: GenerationConfig, seed: UInt32, includeGrammar: Bool
  ) -> LlamaSamplerBuildOutcome {
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
    let penaltiesActive =
      effectiveRepetitionPenalty > 1.0
      || effectivePresencePenalty != 0.0
      || effectiveFrequencyPenalty != 0.0

    let sparams = llama_sampler_chain_default_params()
    guard let sampler = llama_sampler_chain_init(sparams) else {
      return .chainInitFailed
    }
    if penaltiesActive {
      llama_sampler_chain_add(
        sampler,
        llama_sampler_init_penalties(
          effectivePenaltyWindow,  // last_n tokens to penalize (shared window)
          effectiveRepetitionPenalty,  // repeat penalty (multiplicative; 1.0 = no-op)
          effectiveFrequencyPenalty,  // frequency penalty (additive; 0.0 = no-op)
          effectivePresencePenalty  // presence penalty (additive; 0.0 = no-op)
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
      let dry = LlamaGenerationDriver.DRYSamplerDescriptor(
        config: config, nCtxTrain: llama_model_n_ctx_train(model))
    {
      let drySampler = LlamaGenerationDriver.withCStringArray(dry.options.sequenceBreakers) {
        breakers in
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
      if let mirostat = LlamaGenerationDriver.MirostatV2SamplerDescriptor(
        config: config, fallbackSeed: seed)
      {
        llama_sampler_chain_add(
          sampler,
          llama_sampler_init_mirostat_v2(
            mirostat.resolvedSeed,
            mirostat.options.tau,
            mirostat.options.eta
          ))
      } else {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(config.temperature))
        if let xtc = LlamaGenerationDriver.XTCSamplerDescriptor(config: config, fallbackSeed: seed)
        {
          llama_sampler_chain_add(
            sampler,
            llama_sampler_init_xtc(
              xtc.options.probability,
              xtc.options.threshold,
              xtc.options.minKeep,
              xtc.resolvedSeed
            ))
        }
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed))
      }
    }
    return .success(LlamaSamplerHandle(rawPointer: UnsafeMutableRawPointer(sampler)))
  }

  public func freeSampler(_ handle: LlamaSamplerHandle) {
    guard let raw = handle.rawPointer else { return }
    llama_sampler_free(raw.assumingMemoryBound(to: llama_sampler.self))
  }

  public func sample(_ handle: LlamaSamplerHandle, logitIndex: Int32) -> llama_token {
    guard let raw = handle.rawPointer else { return 0 }
    return llama_sampler_sample(
      raw.assumingMemoryBound(to: llama_sampler.self), context, logitIndex)
  }

  public func isEndOfGeneration(_ token: llama_token) -> Bool {
    llama_vocab_is_eog(vocab, token)
  }

  public func tokenToString(_ token: llama_token, invalidUTF8Buffer: inout [CChar]) -> String? {
    LlamaTokenization.tokenToString(token, vocab: vocab, invalidUTF8Buffer: &invalidUTF8Buffer)
  }

  public func decodePromptChunk(
    tokens: ArraySlice<llama_token>, startPosition: Int, logitsOnLastToken: Bool
  ) -> Int32 {
    let chunkSize = tokens.count
    var promptBatch = llama_batch_init(Int32(chunkSize), 0, 1)
    let base = tokens.startIndex
    for i in 0..<chunkSize {
      promptBatch.token[i] = tokens[base + i]
      promptBatch.pos[i] = Int32(startPosition + i)
      promptBatch.n_seq_id[i] = 1
      promptBatch.seq_id[i]?[0] = 0
      promptBatch.logits[i] = (logitsOnLastToken && i == chunkSize - 1) ? 1 : 0
    }
    promptBatch.n_tokens = Int32(chunkSize)

    let decodeResult = llama_decode(context, promptBatch)
    llama_batch_free(promptBatch)
    return decodeResult
  }

  public func decodeGeneratedToken(_ token: llama_token, position: Int) -> Int32 {
    // Generation loop uses a fresh 1-capacity batch — the prompt loop
    // allocated and freed a batch per chunk, so there's nothing to
    // reuse here.
    if generationBatch == nil {
      generationBatch = llama_batch_init(1, 0, 1)
    }
    guard var batch = generationBatch else { return -1 }
    batch.n_tokens = 0
    batch.token[0] = token
    batch.pos[0] = Int32(position)
    batch.n_seq_id[0] = 1
    batch.seq_id[0]?[0] = 0
    batch.logits[0] = 1
    batch.n_tokens = 1
    generationBatch = batch
    return llama_decode(context, batch)
  }

  public func synchronize() {
    llama_synchronize(context)
  }

  public func releaseDecodeResources() {
    if let batch = generationBatch {
      llama_batch_free(batch)
      generationBatch = nil
    }
  }
}
