import XCTest
import Foundation
import LlamaSwift
import ManifoldInference
import ManifoldTestSupport
// BackendInternals SPI: HeuristicTokenizer seam published for the companion split (#1749).
@_spi(BackendInternals) import ManifoldContract
@_spi(Testing) import ManifoldLlama

/// Headless coverage of the **production** tokenization callsites
/// (`LlamaBackend.tokenCount` / `LlamaBackend.countTokens`) against a synthetic
/// **fixture vocabulary** — no model download (issue #29).
///
/// ## Why a fixture vocab
///
/// `LlamaBackend.tokenCount` calls `LlamaTokenization.tokenize(..., parseSpecial: false)`
/// while `countTokens` calls `llama_tokenize(..., parse_special: true)`. The
/// pre-existing `LlamaTokenizationTests` only exercises `tokenize` with
/// `parseSpecial` passed *explicitly*; the real callsites' hardcoded values
/// (`tokenCount` → `false`, `countTokens` → `true`) were never executed in CI
/// because doing so required a real GGUF. A regression that flipped either
/// hardcoded argument would not be caught.
///
/// This builds a tiny SPM vocabulary in memory (byte-fallback tokens + one
/// multi-character CONTROL token `<|im_start|>`), writes it as a `vocab_only`
/// GGUF, loads just the vocab (no tensors, no weights), and injects the vocab
/// pointer into a `LlamaBackend` via the `@_spi(Testing)` seam. The production
/// `tokenCount` / `countTokens` methods then run against it.
///
/// `<|im_start|>` is a single CONTROL token in the fixture, so the two
/// production callsites *must* diverge: `tokenCount` (parseSpecial:false)
/// fragments it into per-byte tokens; `countTokens` (parseSpecial:true) resolves
/// it to one token. That divergence is the regression oracle — if `tokenCount`
/// were changed to `parseSpecial: true`, its count would collapse to match
/// `countTokens` and the assertion fires.
///
/// Runs in the default CI lane (`ci.yml`, macos-15 = Apple Silicon, physical) —
/// `vocab_only` load still initializes Metal, so the test is gated on the same
/// hardware predicates as `LlamaTokenizationTests`. It does NOT require a model
/// on disk.
final class LlamaTokenCountFixtureVocabTests: XCTestCase {

    private var fixture: FixtureVocab!

    override func setUp() async throws {
        try await super.setUp()
        try XCTSkipUnless(HardwareRequirements.isPhysicalDevice,
                          "LlamaBackend / llama vocab load requires Metal (unavailable in simulator)")
        try XCTSkipUnless(HardwareRequirements.isAppleSilicon,
                          "LlamaBackend requires Apple Silicon")
        fixture = try FixtureVocab.make()
    }

    override func tearDown() async throws {
        fixture = nil
        try await super.tearDown()
    }

    // MARK: - tokenCount production callsite (parseSpecial: false)

    /// `tokenCount` hardcodes `parseSpecial: false`. For the multi-character
    /// CONTROL token `<|im_start|>` that means byte-fragmentation, not
    /// single-token resolution.
    ///
    /// Sabotage check: change the `parseSpecial:` argument in
    /// `LlamaBackend.tokenCount` from `false` to `true` — the count collapses
    /// from the byte-fragment count to 1 and `XCTAssertGreaterThan` fails.
    func test_tokenCount_usesParseSpecialFalse_fragmentsControlToken() async throws {
        let backend = LlamaBackend()
        backend.injectVocabForTesting(fixture.vocab)
        addTeardownBlock { backend.injectVocabForTesting(nil) }

        let imStart = "<|im_start|>"
        let count = backend.tokenCount(imStart)

        // parseSpecial:false fragments the 12-byte control string into its byte
        // pieces. With special-token resolution (parseSpecial:true) it would be 1.
        XCTAssertGreaterThan(
            count, 1,
            "tokenCount must use parseSpecial:false, fragmenting <|im_start|> into multiple "
            + "byte tokens; got \(count). A count of 1 means the production callsite was "
            + "changed to parseSpecial:true."
        )
        XCTAssertEqual(
            count, imStart.utf8.count,
            "With byte-fallback and parseSpecial:false, <|im_start|> fragments to one token "
            + "per UTF-8 byte (\(imStart.utf8.count)); got \(count)."
        )
    }

    /// The two production callsites disagree on `<|im_start|>` precisely because
    /// they pass different `parse_special` values. Pinning the *relationship*
    /// (tokenCount > countTokens) is robust to vocab tweaks: it only holds while
    /// `tokenCount` is parseSpecial:false AND `countTokens` is parseSpecial:true.
    ///
    /// Sabotage check: align either callsite's `parse_special` argument with the
    /// other and the strict inequality fails.
    func test_tokenCount_vs_countTokens_divergeOnControlToken() async throws {
        let backend = LlamaBackend()
        backend.injectVocabForTesting(fixture.vocab)
        addTeardownBlock { backend.injectVocabForTesting(nil) }

        let imStart = "<|im_start|>"
        let heuristicCount = backend.tokenCount(imStart)          // parseSpecial: false
        let exactCount = try backend.countTokens(imStart)         // parse_special: true (+ BOS)

        // countTokens resolves <|im_start|> to a single special token (+1 for the
        // BOS it always prepends) = 2; tokenCount fragments to one-per-byte = 12.
        XCTAssertGreaterThan(
            heuristicCount, exactCount,
            "tokenCount(parseSpecial:false)=\(heuristicCount) must exceed "
            + "countTokens(parse_special:true)=\(exactCount) on a control token; "
            + "if not, one callsite's parse_special argument regressed."
        )
        XCTAssertEqual(exactCount, 2,
                       "countTokens resolves <|im_start|> to one special token plus the BOS it prepends")
    }

    // MARK: - tokenCount on plain text (byte-fallback path)

    /// Plain text has no special tokens, so `parseSpecial` does not change the
    /// result — both callsites agree. This pins the byte-fallback path and proves
    /// the fixture vocab tokenizes ordinary input deterministically.
    func test_tokenCount_plainText_matchesByteFallback() async throws {
        let backend = LlamaBackend()
        backend.injectVocabForTesting(fixture.vocab)
        addTeardownBlock { backend.injectVocabForTesting(nil) }

        // "helo" -> 4 byte tokens (no normal-word entries in the fixture vocab).
        let plain = "helo"
        XCTAssertEqual(backend.tokenCount(plain), plain.utf8.count,
                       "byte-fallback should produce one token per UTF-8 byte for plain text")
    }

    // MARK: - tokenCount fallback when no vocab is loaded

    /// With no vocab injected, `tokenCount` falls back to the 4-chars-per-token
    /// heuristic rather than crashing. Covers the `guard ... else` arm of the
    /// production method.
    func test_tokenCount_withoutVocab_usesHeuristic() async throws {
        let backend = LlamaBackend()
        // Deliberately do not inject a vocab.
        let text = "the quick brown fox"
        XCTAssertEqual(backend.tokenCount(text), HeuristicTokenizer.tokenCount(text),
                       "tokenCount must fall back to the heuristic when no vocab is loaded")
    }
}

// MARK: - Fixture vocab builder

/// A minimal `vocab_only` GGUF fixture loaded into a borrowed `llama_vocab`
/// pointer. Owns the underlying `llama_model`; frees it (and removes the temp
/// file) on `deinit`.
private final class FixtureVocab {
    let vocab: OpaquePointer
    private let model: OpaquePointer
    private let fileURL: URL

    private init(vocab: OpaquePointer, model: OpaquePointer, fileURL: URL) {
        self.vocab = vocab
        self.model = model
        self.fileURL = fileURL
    }

    deinit {
        llama_model_free(model)
        do {
            try FileManager.default.removeItem(at: fileURL)
        } catch {
            // Best-effort temp cleanup; a leaked fixture file is harmless.
        }
    }

    /// Builds the SPM byte-fallback vocab, writes it as a `vocab_only` GGUF, and
    /// loads just the vocabulary (no tensors).
    static func make() throws -> FixtureVocab {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("manifold-llama-fixture-vocab-\(UUID().uuidString).gguf")

        // SPM vocabulary. Order: unk, bos, eos, then byte-fallback tokens
        // <0x00>..<0xFF>, then the multi-char CONTROL token. Byte fallback lets
        // any string tokenize without normal-word entries.
        var tokens: [String] = []
        var types: [Int32] = []   // 1=NORMAL 2=UNKNOWN 3=CONTROL 6=BYTE
        tokens.append("<unk>"); types.append(2)
        tokens.append("<s>");   types.append(3)
        tokens.append("</s>");  types.append(3)
        for byte in 0...255 {
            tokens.append(String(format: "<0x%02X>", byte))
            types.append(6)
        }
        tokens.append("<|im_start|>"); types.append(3)   // multi-char CONTROL token
        let scores = [Float](repeating: 0, count: tokens.count)

        guard let ctx = gguf_init_empty() else {
            throw FixtureError.ggufInitFailed
        }
        defer { gguf_free(ctx) }

        gguf_set_val_str(ctx, "general.architecture", "llama")
        gguf_set_val_str(ctx, "general.name", "manifold-llama-fixture-vocab")
        // Minimal llama hparams. vocab_only skips tensors, but some metadata
        // readers still peek at these.
        gguf_set_val_u32(ctx, "llama.context_length", 512)
        gguf_set_val_u32(ctx, "llama.embedding_length", 8)
        gguf_set_val_u32(ctx, "llama.block_count", 1)
        gguf_set_val_u32(ctx, "llama.feed_forward_length", 8)
        gguf_set_val_u32(ctx, "llama.attention.head_count", 1)

        gguf_set_val_str(ctx, "tokenizer.ggml.model", "llama")
        // Disable the SPM space prefix so byte-fallback expansion stays at one
        // token per UTF-8 byte (keeps the production helper's buffer sizing,
        // `utf8.count + addBos`, large enough — see LlamaTokenization.tokenize).
        gguf_set_val_bool(ctx, "tokenizer.ggml.add_space_prefix", false)
        gguf_set_val_u32(ctx, "tokenizer.ggml.unknown_token_id", 0)
        gguf_set_val_u32(ctx, "tokenizer.ggml.bos_token_id", 1)
        gguf_set_val_u32(ctx, "tokenizer.ggml.eos_token_id", 2)

        tokens.withUnsafeStrArray { ptr in
            gguf_set_arr_str(ctx, "tokenizer.ggml.tokens", ptr, tokens.count)
        }
        scores.withUnsafeBufferPointer { buf in
            gguf_set_arr_data(ctx, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32,
                              buf.baseAddress, tokens.count)
        }
        types.withUnsafeBufferPointer { buf in
            gguf_set_arr_data(ctx, "tokenizer.ggml.token_type", GGUF_TYPE_INT32,
                              buf.baseAddress, tokens.count)
        }

        guard gguf_write_to_file(ctx, url.path, false) else {
            throw FixtureError.ggufWriteFailed
        }

        var params = llama_model_default_params()
        params.vocab_only = true
        params.n_gpu_layers = 0
        guard let model = llama_model_load_from_file(url.path, params) else {
            try? FileManager.default.removeItem(at: url)
            throw FixtureError.modelLoadFailed
        }
        guard let vocab = llama_model_get_vocab(model) else {
            llama_model_free(model)
            try? FileManager.default.removeItem(at: url)
            throw FixtureError.vocabUnavailable
        }
        return FixtureVocab(vocab: vocab, model: model, fileURL: url)
    }

    enum FixtureError: Error {
        case ggufInitFailed
        case ggufWriteFailed
        case modelLoadFailed
        case vocabUnavailable
    }
}

private extension Array where Element == String {
    /// Builds a contiguous, NUL-terminated C array of `char *` valid for the
    /// duration of `body`.
    func withUnsafeStrArray<R>(_ body: (UnsafeMutablePointer<UnsafePointer<CChar>?>) -> R) -> R {
        let cStrings: [UnsafeMutablePointer<CChar>] = map { strdup($0) }
        defer { cStrings.forEach { free($0) } }
        var ptrs: [UnsafePointer<CChar>?] = cStrings.map { UnsafePointer($0) }
        return ptrs.withUnsafeMutableBufferPointer { body($0.baseAddress!) }
    }
}
