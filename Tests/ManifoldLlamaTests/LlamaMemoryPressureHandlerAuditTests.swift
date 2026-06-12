import XCTest

/// Re-homed from core's cross-family `MLXMemoryPressureMissingTests`
/// (Tests/ManifoldBackendsTests, retired in core PR C2 — ManifoldKit#1749).
/// This is the positive Llama guard; the negative MLX guard lives in
/// manifold-mlx. If the handler stops being wired up in `LlamaBackend`,
/// the audit asymmetry the perf-audit plan relies on no longer exists and
/// the eventual fix-up issue must be re-scoped.
final class LlamaMemoryPressureHandlerAuditTests: XCTestCase {

    private func sourcePath(fileName: String) -> String {
        let testFileURL = URL(fileURLWithPath: #filePath)
        let packageRoot = testFileURL
            .deletingLastPathComponent()  // Tests/ManifoldLlamaTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // package root
        return packageRoot
            .appendingPathComponent("Sources")
            .appendingPathComponent("ManifoldLlama")
            .appendingPathComponent(fileName)
            .path
    }

    func test_llamaBackend_hasMemoryPressureHandler() throws {
        let source = try String(
            contentsOfFile: sourcePath(fileName: "LlamaBackend.swift"),
            encoding: .utf8
        )
        XCTAssertTrue(
            source.contains("MemoryPressureHandler"),
            "LlamaBackend.swift no longer references 'MemoryPressureHandler' (#415) — the audit asymmetry has dissolved; revisit the perf-audit plan."
        )
    }
}
