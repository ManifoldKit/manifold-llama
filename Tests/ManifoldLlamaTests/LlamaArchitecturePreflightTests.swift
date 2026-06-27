import XCTest
import ManifoldInference
import ManifoldLlama
@_spi(Testing) import ManifoldLlama
import ManifoldTestSupport

/// Preflight-check tests for `LlamaModelLoader` — unsupported GGUF architectures
/// (vision encoders, embedding-only models, speech/diffusion) must throw
/// `InferenceError.unsupportedModelArchitecture` before `generate()` can reach
/// `llama_decode` and crash on a non-LM model. See bundled plan item P2.
final class LlamaArchitecturePreflightTests: XCTestCase {

    // MARK: - Denylist Contents (no hardware required — pure logic)

    /// Vision encoders are the canonical case: a CLIP-L weight dump loaded as a
    /// GGUF has no decode path and will crash inside `llama_decode`.
    ///
    /// Sabotage check: remove `"clip"` from `unsupportedArchitectures` and this
    /// assertion fails — `isUnsupportedArchitecture("clip")` returns false and
    /// the preflight would silently accept a non-LM GGUF.
    func test_denylist_rejectsClipVisionEncoder() {
        XCTAssertTrue(LlamaModelLoader.isUnsupportedArchitecture("clip"),
                      "CLIP vision encoders must be rejected — they have no causal-LM decode path")
    }

    /// Embedding-only BERT variants expose no generation path.
    func test_denylist_rejectsBertEmbedders() {
        XCTAssertTrue(LlamaModelLoader.isUnsupportedArchitecture("bert"))
        XCTAssertTrue(LlamaModelLoader.isUnsupportedArchitecture("nomic-bert"))
        XCTAssertTrue(LlamaModelLoader.isUnsupportedArchitecture("jina-bert-v2"))
    }

    /// Multimodal LLaVA / mllama checkpoints require a projector + mm path that
    /// llama.cpp's standard decode loop doesn't provide.
    func test_denylist_rejectsMultimodalWrappers() {
        XCTAssertTrue(LlamaModelLoader.isUnsupportedArchitecture("llava"))
        XCTAssertTrue(LlamaModelLoader.isUnsupportedArchitecture("mllama"))
    }

    /// Speech and diffusion weight dumps leak into user Models/ folders as .gguf
    /// files often enough to warrant an explicit deny.
    func test_denylist_rejectsSpeechAndDiffusion() {
        XCTAssertTrue(LlamaModelLoader.isUnsupportedArchitecture("whisper"))
        XCTAssertTrue(LlamaModelLoader.isUnsupportedArchitecture("stablediffusion"))
        XCTAssertTrue(LlamaModelLoader.isUnsupportedArchitecture("sd3"))
    }

    /// Fused-multimodal Gemma 3n ships text + audio + vision + mmproj tensors in
    /// one GGUF. The pinned llama.cpp build recognizes the arch but its
    /// text-model loader only claims the text tower and aborts in
    /// `done_getting_tensors` ("wrong number of tensors"). Deny it so the user
    /// gets a typed error instead of a cryptic nil-load failure. See issue #62.
    ///
    /// Note: `gemma4` was previously denylisted alongside `gemma3n`, but
    /// text-only gemma4 GGUFs load fine on b9744, so it was removed (see
    /// `test_denylist_stillAcceptsTextOnlyGemma`). `gemma3n` stays — it has not
    /// been verified as loadable in a text-only repack.
    ///
    /// Sabotage check: remove `"gemma3n"` from `unsupportedArchitectures` and
    /// this assertion fails.
    func test_denylist_rejectsFusedMultimodalGemma() {
        XCTAssertTrue(LlamaModelLoader.isUnsupportedArchitecture("gemma3n"))
        // Casing is normalized, matching how HF/converter tooling emits these.
        XCTAssertTrue(LlamaModelLoader.isUnsupportedArchitecture("Gemma3n"))
    }

    /// The fused-multimodal deny must not spill onto the text-only Gemma
    /// architectures the backend loads fine today. `gemma4` joined this set once
    /// text-only gemma4 GGUFs were confirmed loadable on b9744 (issue #62) — a
    /// FUSED gemma4 file now fails later in llama.cpp rather than via our typed
    /// error, an accepted tradeoff since fused single-GGUFs aren't a supported
    /// text-inference input.
    func test_denylist_stillAcceptsTextOnlyGemma() {
        for arch in ["gemma", "gemma2", "gemma3", "gemma4", "Gemma4"] {
            XCTAssertFalse(
                LlamaModelLoader.isUnsupportedArchitecture(arch),
                "Text-only Gemma architecture '\(arch)' must remain loadable"
            )
        }
    }

    // MARK: - Pre-load Header Architecture Reader (issue #62)

    /// The pre-load header reader must extract `general.architecture` from a
    /// minimal hand-built GGUF header so denylisted architectures that abort
    /// inside `llama_model_load_from_file` (returning nil before any model
    /// pointer exists) still surface a typed `unsupportedModelArchitecture`.
    ///
    /// Sabotage check: have `readArchitectureFromHeader` always return nil and
    /// this assertion fails — the cryptic generic load failure returns.
    func test_headerReader_extractsArchitectureFromMinimalGGUF() throws {
        // Uses `gemma3n` (still denylisted) — `gemma4` is now an accepted
        // text-only arch and would not throw at preflight (see issue #62).
        let url = try Self.writeMinimalGGUF(architecture: "gemma3n")
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertEqual(LlamaModelLoader.readArchitectureFromHeader(at: url), "gemma3n")
        XCTAssertThrowsError(try LlamaModelLoader.preflightArchitecture(
            LlamaModelLoader.readArchitectureFromHeader(at: url)
        )) { error in
            guard case InferenceError.unsupportedModelArchitecture(let arch) = error else {
                return XCTFail("Expected unsupportedModelArchitecture, got \(error)")
            }
            XCTAssertEqual(arch, "gemma3n")
        }
    }

    /// A non-GGUF / truncated file must read as nil ("unknown, assume
    /// supported") so the preflight never blocks a legitimate load on a parse
    /// hiccup — the real load path remains the authority.
    func test_headerReader_returnsNilForNonGGUF() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("not-a-gguf-\(UUID().uuidString).bin")
        try Data([0x00, 0x01, 0x02, 0x03, 0x04, 0x05]).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertNil(LlamaModelLoader.readArchitectureFromHeader(at: url))
    }

    /// Builds a valid-enough GGUF v3 header carrying a single
    /// `general.architecture` string KV (plus one preceding scalar KV to prove
    /// the reader skips intervening entries). No tensor data — the reader never
    /// reaches it.
    private static func writeMinimalGGUF(architecture: String) throws -> URL {
        var data = Data()
        func appendU32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        func appendU64(_ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        func appendStr(_ s: String) {
            let bytes = Array(s.utf8)
            appendU64(UInt64(bytes.count))
            data.append(contentsOf: bytes)
        }

        data.append(contentsOf: [0x47, 0x47, 0x55, 0x46]) // "GGUF"
        appendU32(3)            // version
        appendU64(0)            // tensor count
        appendU64(2)            // kv count

        // KV 0: a u32 scalar the reader must skip over.
        appendStr("general.quantization_version")
        appendU32(4)            // value type = uint32
        appendU32(2)

        // KV 1: the architecture string.
        appendStr("general.architecture")
        appendU32(8)            // value type = string
        appendStr(architecture)

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("minimal-\(UUID().uuidString).gguf")
        try data.write(to: url)
        return url
    }

    // MARK: - Case-insensitivity

    /// Case-insensitivity: GGUF authors are inconsistent about casing; `CLIP`
    /// and `clip` must both match.
    func test_denylist_isCaseInsensitive() {
        XCTAssertTrue(LlamaModelLoader.isUnsupportedArchitecture("CLIP"))
        XCTAssertTrue(LlamaModelLoader.isUnsupportedArchitecture("Bert"))
    }

    // MARK: - Allowlist (things the denylist must NOT reject)

    /// The denylist must not reject legitimate causal-LM architectures — a false
    /// positive here breaks every current and future chat model loaded through
    /// `LlamaBackend`.
    ///
    /// Sabotage check: add `"llama"` to `unsupportedArchitectures` and this
    /// assertion fails — the preflight would refuse to load every Llama family
    /// model.
    func test_denylist_acceptsCausalLMArchitectures() {
        let legitimate = [
            "llama", "llama2", "llama3",
            "qwen", "qwen2", "qwen3",
            "mistral", "mixtral",
            "gemma", "gemma2", "gemma3",
            "phi", "phi3",
            "falcon", "mamba", "gptneox", "gpt2",
            // Even architectures we haven't explicitly tested must default to allow —
            // the denylist-not-allowlist decision hinges on this behaviour.
            "brand-new-arch-that-doesnt-exist-yet",
        ]
        for arch in legitimate {
            XCTAssertFalse(
                LlamaModelLoader.isUnsupportedArchitecture(arch),
                "Legitimate LM architecture '\(arch)' must NOT be on the denylist"
            )
        }
    }

    // MARK: - Real GGUF Load (hardware-gated)

    /// When a real chat GGUF is available on disk, loading it through
    /// `LlamaBackend.loadModel` must succeed — proving the preflight doesn't
    /// reject legitimate chat models.
    ///
    /// This is the positive half of the preflight contract. The negative half
    /// (non-LM GGUF throws `unsupportedModelArchitecture`) cannot be exercised
    /// in CI without bundling a ~50 MB vision-encoder fixture; we rely on the
    /// pure-logic denylist tests above for that coverage.
    ///
    /// Sabotage check: change `isUnsupportedArchitecture` to return `true`
    /// unconditionally. This test then throws `.unsupportedModelArchitecture`
    /// and fails.
    func test_preflight_acceptsRealChatGGUF() async throws {
        try XCTSkipUnless(HardwareRequirements.isAppleSilicon, "LlamaBackend requires Apple Silicon")
        try XCTSkipUnless(HardwareRequirements.isPhysicalDevice, "LlamaBackend requires Metal (unavailable in simulator)")
        guard let modelURL = HardwareRequirements.findGGUFModel() else {
            throw XCTSkip("No chat GGUF available. Set LLAMA_TEST_MODEL=<path> or place a `.gguf` in ~/Documents/Models/ to run the preflight happy-path test.")
        }

        let backend = LlamaBackend()
        addTeardownBlock { await backend.unloadAndWait() }

        do {
            try await backend.loadModel(from: modelURL, plan: .testStub(effectiveContextSize: 512))
        } catch InferenceError.unsupportedModelArchitecture(let arch) {
            XCTFail("Real chat GGUF was rejected by the preflight as architecture '\(arch)' — the denylist is too aggressive")
            return
        }

        XCTAssertTrue(backend.isModelLoaded,
                      "A chat GGUF must pass the preflight and load successfully")
    }

    // MARK: - Error Description

    /// The error's `errorDescription` must name the architecture so users can
    /// diagnose which file they need to replace.
    func test_unsupportedArchitectureError_descriptionNamesTheArchitecture() {
        let error = InferenceError.unsupportedModelArchitecture("clip")
        let message = error.errorDescription ?? ""
        XCTAssertTrue(message.contains("clip"),
                      "errorDescription must include the architecture string; got: \(message)")
        XCTAssertFalse(error.isRetryable,
                       "Architecture mismatch is permanent — retry can never help")
    }
}
