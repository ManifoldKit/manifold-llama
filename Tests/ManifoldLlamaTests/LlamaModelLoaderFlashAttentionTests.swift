import XCTest
import LlamaSwift
@_spi(Testing) import ManifoldLlama

/// Model-free coverage of the `LlamaModelLoader.resolvedFlashAttnType` helper
/// that was introduced to fix issue #86:
///
///   - `flashAttention == false` was previously a silent no-op, leaving
///     `flash_attn_type` at `LLAMA_FLASH_ATTN_TYPE_AUTO` (-1) — which enables
///     FA on capable hardware. A caller explicitly opting out therefore got FA
///     anyway.
///   - The simulator path omitted the assignment entirely, also landing on AUTO
///     rather than DISABLED as the comment claimed.
///
/// `resolvedFlashAttnType(flashAttentionRequested:isSimulator:)` captures the
/// correct mapping. These assertions pin all four arms of that mapping so a
/// future regression is caught headlessly, without a GGUF model.
///
/// Sabotage check: change the `false` arm to return AUTO and the
/// `(requested: false, isSimulator: false) → DISABLED` assertion fails.
final class LlamaModelLoaderFlashAttentionTests: XCTestCase {

    // MARK: - resolvedFlashAttnType — all four arms

    /// Non-simulator + requested=true → ENABLED.
    func test_resolvedFlashAttnType_nonSimulator_requestedTrue_returnsEnabled() {
        let result = LlamaModelLoader.resolvedFlashAttnType(
            flashAttentionRequested: true,
            isSimulator: false
        )
        XCTAssertEqual(
            result, LLAMA_FLASH_ATTN_TYPE_ENABLED,
            "Non-simulator + requested=true must resolve to ENABLED, got \(result.rawValue)"
        )
    }

    /// Non-simulator + requested=false → DISABLED (not AUTO).
    ///
    /// This is the primary regression arm from issue #86. Before the fix,
    /// `flash_attn_type` was left at its default value of AUTO (-1), enabling
    /// FA on capable hardware even though the caller passed `false`.
    func test_resolvedFlashAttnType_nonSimulator_requestedFalse_returnsDisabled() {
        let result = LlamaModelLoader.resolvedFlashAttnType(
            flashAttentionRequested: false,
            isSimulator: false
        )
        XCTAssertEqual(
            result, LLAMA_FLASH_ATTN_TYPE_DISABLED,
            "Non-simulator + requested=false must resolve to DISABLED (not AUTO); got \(result.rawValue)"
        )
        // Explicit guard: AUTO (-1) is the wrong value.
        XCTAssertNotEqual(
            result, LLAMA_FLASH_ATTN_TYPE_AUTO,
            "DISABLED must not collapse to AUTO; that would silently enable FA on capable hardware"
        )
    }

    /// Simulator + requested=true → DISABLED (simulator Metal does not
    /// reliably support FA kernels; the requested value is overridden).
    func test_resolvedFlashAttnType_simulator_requestedTrue_returnsDisabled() {
        let result = LlamaModelLoader.resolvedFlashAttnType(
            flashAttentionRequested: true,
            isSimulator: true
        )
        XCTAssertEqual(
            result, LLAMA_FLASH_ATTN_TYPE_DISABLED,
            "Simulator + requested=true must resolve to DISABLED; got \(result.rawValue)"
        )
    }

    /// Simulator + requested=false → DISABLED. Both paths through the
    /// simulator guard must yield DISABLED regardless of the requested value.
    func test_resolvedFlashAttnType_simulator_requestedFalse_returnsDisabled() {
        let result = LlamaModelLoader.resolvedFlashAttnType(
            flashAttentionRequested: false,
            isSimulator: true
        )
        XCTAssertEqual(
            result, LLAMA_FLASH_ATTN_TYPE_DISABLED,
            "Simulator + requested=false must resolve to DISABLED; got \(result.rawValue)"
        )
    }

    // MARK: - Enum value sanity

    /// Confirm the three type constants carry the C-ABI values the rest of
    /// llama.cpp expects. If the vendored header ever diverges, a loader that
    /// passes the wrong int to `llama_init_from_model` will misbehave silently.
    func test_flashAttnTypeConstants_haveExpectedRawValues() {
        XCTAssertEqual(LLAMA_FLASH_ATTN_TYPE_AUTO.rawValue,     -1, "AUTO must be -1")
        XCTAssertEqual(LLAMA_FLASH_ATTN_TYPE_DISABLED.rawValue,  0, "DISABLED must be 0")
        XCTAssertEqual(LLAMA_FLASH_ATTN_TYPE_ENABLED.rawValue,   1, "ENABLED must be 1")
    }
}
