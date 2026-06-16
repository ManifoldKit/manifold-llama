import XCTest
import ManifoldLlama
@_spi(Testing) import ManifoldLlama
import ManifoldInference

// Tests for LlamaBackend.secureWipe().
// Real KV-zeroing requires an active llama.cpp context with a loaded model,
// which needs Llama hardware/binary and a GGUF file.  All tests that touch
// a live context are skipped unless the Llama trait is present.
final class LlamaKVCacheSecureWipeTests: XCTestCase {

    func testSecureWipeDoesNotCrashWithNoContextLoaded() {
        // secureWipe() on an unloaded backend should be a safe no-op — no
        // context or memory pointer to dereference.
        let backend = LlamaBackend()
        backend.secureWipe()  // Must not crash
    }

    func testSecureWipeClearsSessionKVState() {
        // Seed a synthetic cached prefix (via the @_spi(Testing) seam), then prove
        // secureWipe() actually nils sessionKVState. The `sessionKVState = nil`
        // line runs regardless of whether a context is loaded, so this catches a
        // regression that drops it — without needing a real model.
        let backend = LlamaBackend()
        backend.seedSessionKVStateForTesting(tokenCount: 12)
        XCTAssertEqual(backend.sessionKVTokenCountForTesting, 12,
            "Precondition: seeded session KV state must be observable")

        backend.secureWipe()
        XCTAssertNil(backend.sessionKVTokenCountForTesting,
            "secureWipe() must clear sessionKVState so the next turn won't reuse a stale prefix")
    }

    func testResetConversationClearsSessionKVState() {
        // resetConversation() shares the same clearing contract as secureWipe().
        let backend = LlamaBackend()
        backend.seedSessionKVStateForTesting(tokenCount: 7)
        XCTAssertEqual(backend.sessionKVTokenCountForTesting, 7)

        backend.resetConversation()
        XCTAssertNil(backend.sessionKVTokenCountForTesting,
            "resetConversation() must clear the cached prefix")
    }
}
