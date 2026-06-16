import XCTest
@_spi(Testing) import ManifoldLlama

/// Model-free unit tests for the `contextExhausted` preflight predicate extracted
/// out of `LlamaBackend.generate()` (issue #27).
///
/// The full guard (`tokens.count + maxTokens <= contextSize else throw
/// .contextExhausted`) was previously reachable only with a loaded vocab/context —
/// `generate()` tokenizes the prompt first, so the boundary arithmetic could not be
/// exercised in CI. Callers retry on `.contextExhausted`, so a `< vs <=` slip is a
/// silent correctness bug (it would spuriously reject an exact-fit prompt). These
/// tests pin the boundary headlessly via the extracted `contextWindowFits` static.
///
/// The live `generate()` wiring that throws `.contextExhausted` from this predicate
/// is still covered (model-bound) by
/// `test_fixture_nBatchBoundary_oversizedPromptThrowsContextExhausted_scaffold520`.
final class LlamaBackendContextPreflightTests: XCTestCase {

    /// Exact fit must be accepted: every prompt token and every requested output
    /// token has a KV slot. This is the `<=` half of the boundary — the case a `<`
    /// slip would wrongly reject.
    func test_contextWindowFits_exactFit_isAccepted() {
        XCTAssertTrue(LlamaBackend.contextWindowFits(
            promptTokens: 100, maxOutputTokens: 412, contextSize: 512),
            "promptTokens + maxOutputTokens == contextSize must fit (<=, not <)")
    }

    /// One token over the window must be rejected — the immediate neighbor of the
    /// exact-fit case, so the two together fence the boundary on both sides.
    func test_contextWindowFits_oneOver_isRejected() {
        XCTAssertFalse(LlamaBackend.contextWindowFits(
            promptTokens: 100, maxOutputTokens: 413, contextSize: 512),
            "one token past the window must not fit")
    }

    /// Comfortably under the window fits.
    func test_contextWindowFits_underBudget_isAccepted() {
        XCTAssertTrue(LlamaBackend.contextWindowFits(
            promptTokens: 10, maxOutputTokens: 16, contextSize: 512))
    }

    /// A prompt alone larger than the whole context (zero output headroom) is
    /// rejected — the oversized-prompt path the live scaffold520 fixture drives.
    func test_contextWindowFits_promptAloneOverflows_isRejected() {
        XCTAssertFalse(LlamaBackend.contextWindowFits(
            promptTokens: 600, maxOutputTokens: 0, contextSize: 512))
    }

    /// Degenerate but well-defined: empty prompt, zero output, zero context fits
    /// (0 <= 0). Pins that the relation is non-strict at the origin too.
    func test_contextWindowFits_allZero_isAccepted() {
        XCTAssertTrue(LlamaBackend.contextWindowFits(
            promptTokens: 0, maxOutputTokens: 0, contextSize: 0))
    }
}
