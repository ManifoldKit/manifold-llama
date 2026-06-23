import XCTest
import ManifoldInference
import ManifoldLlama
@_spi(Testing) import ManifoldLlama
@_spi(BackendInternals) import ManifoldHardware

/// Unit tests for `LlamaBackend.capabilities.toolDialect`.
///
/// Verifies the arch-string → `ToolCallDialect` mapping without loading a model —
/// uses `injectArchitectureForTesting(_:)` to set `_architecture` under the lock,
/// then asserts on `backend.capabilities.toolDialect`.
final class LlamaToolCallDialectTests: XCTestCase {

    // MARK: - Pre-load state

    /// Before any model loads, `_architecture` is nil, so `toolDialect` must be nil
    /// rather than `.unknown` — nil means "not yet determined", not "known-unknown".
    func test_preLoad_toolDialect_isNil() {
        let backend = LlamaBackend()
        XCTAssertNil(backend.capabilities.toolDialect,
                     "toolDialect must be nil before a model is loaded")
    }

    // MARK: - Gemma family

    func test_gemma_toolDialect_family() {
        let backend = LlamaBackend()
        backend.injectArchitectureForTesting("gemma")
        let dialect = backend.capabilities.toolDialect
        XCTAssertEqual(dialect?.family, .gemma)
        XCTAssertEqual(dialect?.openDelimiter, "<|tool_call|>")
        XCTAssertEqual(dialect?.closeDelimiter, "<|end_of_turn>")
        XCTAssertEqual(dialect?.argEncoding, .json)
        XCTAssertEqual(dialect?.extractability, .buried)
    }

    func test_gemma2_toolDialect_family() {
        let backend = LlamaBackend()
        backend.injectArchitectureForTesting("gemma2")
        XCTAssertEqual(backend.capabilities.toolDialect?.family, .gemma)
    }

    func test_gemma3_toolDialect_family() {
        let backend = LlamaBackend()
        backend.injectArchitectureForTesting("gemma3")
        XCTAssertEqual(backend.capabilities.toolDialect?.family, .gemma)
    }

    // MARK: - Qwen family

    func test_qwen_toolDialect_family() {
        let backend = LlamaBackend()
        backend.injectArchitectureForTesting("qwen2")
        let dialect = backend.capabilities.toolDialect
        XCTAssertEqual(dialect?.family, .qwen)
        XCTAssertEqual(dialect?.openDelimiter, "<tool_call>")
        XCTAssertEqual(dialect?.closeDelimiter, "</tool_call>")
        XCTAssertEqual(dialect?.argEncoding, .json)
        XCTAssertEqual(dialect?.extractability, .clean)
    }

    func test_qwen3_toolDialect_family() {
        let backend = LlamaBackend()
        backend.injectArchitectureForTesting("qwen3")
        XCTAssertEqual(backend.capabilities.toolDialect?.family, .qwen)
    }

    // MARK: - Llama family

    func test_llama_toolDialect_family() {
        let backend = LlamaBackend()
        backend.injectArchitectureForTesting("llama")
        let dialect = backend.capabilities.toolDialect
        XCTAssertEqual(dialect?.family, .llamaPythonTag)
        XCTAssertNil(dialect?.openDelimiter,
                     "Llama bare-JSON has no opening delimiter")
        XCTAssertNil(dialect?.closeDelimiter,
                     "Llama bare-JSON has no closing delimiter")
        XCTAssertEqual(dialect?.argEncoding, .json)
        XCTAssertEqual(dialect?.extractability, .buried)
    }

    func test_llama3_toolDialect_family() {
        let backend = LlamaBackend()
        backend.injectArchitectureForTesting("llama3")
        XCTAssertEqual(backend.capabilities.toolDialect?.family, .llamaPythonTag)
    }

    // MARK: - Mistral family

    func test_mistral_toolDialect_family() {
        let backend = LlamaBackend()
        backend.injectArchitectureForTesting("mistral")
        let dialect = backend.capabilities.toolDialect
        XCTAssertEqual(dialect?.family, .mistral)
        XCTAssertEqual(dialect?.openDelimiter, "[TOOL_CALLS]")
        XCTAssertNil(dialect?.closeDelimiter,
                     "Mistral has no closing delimiter — body ends at EOS")
        XCTAssertEqual(dialect?.argEncoding, .json)
        XCTAssertEqual(dialect?.extractability, .buried)
    }

    func test_mixtral_toolDialect_family() {
        let backend = LlamaBackend()
        backend.injectArchitectureForTesting("mixtral")
        XCTAssertEqual(backend.capabilities.toolDialect?.family, .mistral)
    }

    // MARK: - Unknown architecture

    func test_unknownArch_toolDialect_family() {
        let backend = LlamaBackend()
        backend.injectArchitectureForTesting("phi3")
        let dialect = backend.capabilities.toolDialect
        XCTAssertEqual(dialect?.family, .unknown)
        XCTAssertEqual(dialect?.extractability, .toolLess)
    }

    func test_brandNewArch_toolDialect_family() {
        let backend = LlamaBackend()
        backend.injectArchitectureForTesting("brand-new-arch-not-yet-supported")
        XCTAssertEqual(backend.capabilities.toolDialect?.family, .unknown)
    }

    // MARK: - Case-insensitivity (GGUF authors use mixed case)

    func test_upperCaseGemma_toolDialect_family() {
        let backend = LlamaBackend()
        backend.injectArchitectureForTesting("Gemma3")
        XCTAssertEqual(backend.capabilities.toolDialect?.family, .gemma,
                       "Architecture matching must be case-insensitive")
    }

    func test_upperCaseLlama_toolDialect_family() {
        let backend = LlamaBackend()
        backend.injectArchitectureForTesting("LLaMA")
        XCTAssertEqual(backend.capabilities.toolDialect?.family, .llamaPythonTag,
                       "Architecture matching must be case-insensitive")
    }
}
