import XCTest
@_spi(Testing) import ManifoldLlama

/// Re-homed from core's cross-family `MLXMemoryPressureMissingTests`
/// (Tests/ManifoldBackendsTests, retired in core PR C2 — ManifoldKit#1749).
/// This is the positive Llama guard; the negative MLX guard lives in
/// manifold-mlx. If the handler stops being wired up in `LlamaBackend`,
/// the audit asymmetry the perf-audit plan relies on no longer exists and
/// the eventual fix-up issue must be re-scoped.
///
/// Previously this asserted only that the *source text* contained the substring
/// "MemoryPressureHandler" — a grep that passed even for a comment and exercised
/// no runtime behavior. It now drives the wired callback behaviorally: an idle
/// `LlamaBackend` must respond to a `.warning` pressure event by cancelling
/// generation. If the handler is no longer registered in `init`, the event is a
/// no-op and `isCancelledForTesting` stays false.
final class LlamaMemoryPressureHandlerAuditTests: XCTestCase {

    func test_llamaBackend_respondsToMemoryPressure() {
        let backend = LlamaBackend()
        XCTAssertFalse(backend.isCancelledForTesting,
            "Precondition: a fresh backend is not cancelled")

        backend.simulateMemoryPressure(.warning)

        XCTAssertTrue(backend.isCancelledForTesting,
            "LlamaBackend must register a memory-pressure callback in init (#415) that stops generation on .warning — a no-op here means the handler is no longer wired")
    }
}
