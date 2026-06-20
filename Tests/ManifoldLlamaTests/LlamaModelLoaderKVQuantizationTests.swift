import XCTest
import ManifoldInference
import ManifoldLlama
@_spi(Testing) import ManifoldLlama
import ManifoldTestSupport

/// Model-bearing coverage of the three `LlamaModelLoader` arms that the headless
/// suite (`LlamaModelLoaderHeadlessFailurePathTests`) could not reach without a
/// real/failing `llama_*` call (issue #28):
///
///   1. KV-cache quantization `.f16` mapping — the library-default branch that
///      leaves `ctxParams.type_k` / `type_v` at `GGML_TYPE_F16`.
///   2. KV-cache quantization `.q4` mapping — the `GGML_TYPE_Q4_0` branch.
///      (`.q8` is the default already exercised by every other gated suite;
///      we loop it in here for completeness so all three enum arms are pinned.)
///   3. Context-creation nil failure (code -2) — `llama_init_from_model`
///      returning nil while `llama_model_load_from_file` succeeded.
///
/// Arms 1–2 require a real GGUF: the only observable contract is "the configured
/// KV-cache dtype actually loads a working context without tripping a GGML assert
/// or a type mismatch". They are gated on a fixture model and skip cleanly when
/// absent (run for real in the nightly model lane).
///
/// Arm 3 cannot be triggered reliably from a config — forcing
/// `llama_init_from_model` to return nil while the model loaded requires an
/// allocator fault we can't manufacture portably — so it is asserted through the
/// `contextCreationFailure` seam (mirroring the `finishDecodeFailure` seam used
/// for decode-error teardown). That arm needs no model and runs everywhere.
final class LlamaModelLoaderKVQuantizationTests: XCTestCase {

    // MARK: - KV-cache quantization mapping (model-gated)

    /// Each KV-cache quantization option must produce a loadable context. `.f16`
    /// leaves `type_k`/`type_v` at the library default; `.q8` and `.q4` set
    /// `GGML_TYPE_Q8_0` / `GGML_TYPE_Q4_0`. The observable contract is that all
    /// three dtypes load without a GGML assert or a KV-cache type mismatch — a
    /// wrong mapping (e.g. an unsupported dtype, or `type_k`/`type_v` left
    /// mismatched) surfaces here as a failed load or a process-level abort.
    ///
    /// Sabotage check: point the `.q4` arm at an unsupported GGML type, or set
    /// `type_k` without `type_v`, and the corresponding load throws/aborts.
    func test_loadModel_eachKVQuantization_loadsWorkingContext() async throws {
        try XCTSkipUnless(HardwareRequirements.isAppleSilicon, "LlamaBackend requires Apple Silicon")
        try XCTSkipUnless(HardwareRequirements.isPhysicalDevice, "LlamaBackend requires Metal (unavailable in simulator)")
        guard let modelURL = HardwareRequirements.findGGUFModel() else {
            throw XCTSkip("No chat GGUF available. Set LLAMA_TEST_MODEL=<path> or place a `.gguf` in ~/Documents/Models/ to run the KV-quantization load tests.")
        }

        // All three enum arms. Looping keeps the gate (and the all-skipped guard
        // accounting in model-tests.yml) on a single test body while still
        // exercising every `kvCacheQuantization` switch case in `initializeModel`.
        let quantizations: [BackendLoadOptions.KVCacheQuantization] = [.f16, .q8, .q4]

        for quant in quantizations {
            let backend = LlamaBackend()
            backend.setLoadOptions(BackendLoadOptions(kvCacheQuantization: quant))
            addTeardownBlock { await backend.unloadAndWait() }

            // A GGML assert or KV-cache dtype mismatch on the configured
            // quantization surfaces as a thrown error or a process abort here.
            try await backend.loadModel(from: modelURL, plan: .testStub(effectiveContextSize: 512))

            XCTAssertTrue(
                backend.isModelLoaded,
                "kvCacheQuantization=\(quant.rawValue) must produce a loadable context"
            )
            XCTAssertEqual(
                backend.loadOptionsForTesting.kvCacheQuantization, quant,
                "The load must have used the configured KV-cache quantization \(quant.rawValue)"
            )

            await backend.unloadAndWait()
        }
    }

    // MARK: - Context-creation nil failure, code -2 (seam, model-free)

    /// `llama_init_from_model` returning nil — an allocator failure at the
    /// requested context size while the model itself loaded — must surface as a
    /// typed `.modelLoadFailed` with the `LlamaBackend` domain and code **-2**,
    /// distinct from the model-load failure (code -1). The nil-context guard in
    /// `initializeModel` throws exactly `contextCreationFailure(...)`, so
    /// asserting that seam pins the wired contract without an unforceable
    /// allocator fault.
    ///
    /// Sabotage check: change the guard's code to -1 (collapsing it into the
    /// model-load failure) and this assertion fails.
    func test_contextCreationFailure_throwsModelLoadFailedCodeMinusTwo() {
        let error = LlamaModelLoader.contextCreationFailure(effectiveContextSize: 4096)

        guard case InferenceError.modelLoadFailed(let underlying) = error else {
            return XCTFail("Expected .modelLoadFailed, got \(error)")
        }
        let nsError = underlying as NSError
        XCTAssertEqual(nsError.domain, "LlamaBackend",
                       "Context-creation failure must carry the LlamaBackend domain")
        XCTAssertEqual(nsError.code, -2,
                       "Context-creation failure must be code -2, distinct from the -1 model-load failure")
    }

    /// The context-creation (-2) and model-load (-1) failures must stay distinct
    /// codes so callers can tell "the file didn't load" from "the file loaded but
    /// no context could be allocated at this size" (the latter is retry-with-a-
    /// smaller-context actionable; the former is not).
    func test_contextCreationFailure_codeIsDistinctFromModelLoadFailure() {
        guard case InferenceError.modelLoadFailed(let underlying) =
                LlamaModelLoader.contextCreationFailure(effectiveContextSize: 8192) else {
            return XCTFail("Expected .modelLoadFailed")
        }
        XCTAssertNotEqual((underlying as NSError).code, -1,
                          "Context-creation failure (-2) must not collide with the model-load failure code (-1)")
    }

    /// The error message must name the requested context size so a caller can see
    /// which size to back off from when retrying with a smaller plan.
    func test_contextCreationFailure_messageNamesRequestedContextSize() {
        guard case InferenceError.modelLoadFailed(let underlying) =
                LlamaModelLoader.contextCreationFailure(effectiveContextSize: 12345) else {
            return XCTFail("Expected .modelLoadFailed")
        }
        let message = (underlying as NSError).localizedDescription
        XCTAssertTrue(message.contains("12345"),
                      "The failure message must name the requested context size; got: \(message)")
    }
}
