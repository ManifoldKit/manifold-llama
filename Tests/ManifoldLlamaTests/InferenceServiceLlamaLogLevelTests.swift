import XCTest
import ManifoldInference
@_spi(Testing) import ManifoldLlama

/// Covers the public `InferenceService.llamaLogLevel` get/set bridge (previously
/// 0% covered). The underlying `LlamaBackendProcessLifecycle` round-trip is
/// proven elsewhere; this verifies the `InferenceService` extension actually
/// forwards through to it rather than dropping the value.
@MainActor
final class InferenceServiceLlamaLogLevelTests: XCTestCase {

    override func tearDown() {
        // Restore the process-global default so we don't perturb other suites.
        LlamaBackendProcessLifecycle.setLogLevel(.info)
        super.tearDown()
    }

    func test_set_forwardsToProcessLifecycle() {
        let service = InferenceService()
        service.llamaLogLevel = .silent
        assertSame(LlamaBackendProcessLifecycle.currentLogLevel, .silent,
            "Setting llamaLogLevel must forward to LlamaBackendProcessLifecycle")
    }

    func test_get_reflectsProcessLifecycle() {
        let service = InferenceService()
        LlamaBackendProcessLifecycle.setLogLevel(.verbose)
        assertSame(service.llamaLogLevel, .verbose,
            "Reading llamaLogLevel must reflect the current process log level")
    }

    func test_roundTrip_throughInferenceServiceOnly() {
        let service = InferenceService()
        for level in [LlamaLogLevel.silent, .warning, .info, .verbose] {
            service.llamaLogLevel = level
            assertSame(service.llamaLogLevel, level,
                "llamaLogLevel must round-trip through the InferenceService property")
        }
    }

    // LlamaLogLevel is Sendable but not Equatable; compare by case.
    private func assertSame(_ a: LlamaLogLevel, _ b: LlamaLogLevel,
                            _ message: String,
                            file: StaticString = #file, line: UInt = #line) {
        let same: Bool
        switch (a, b) {
        case (.silent, .silent), (.warning, .warning), (.info, .info), (.verbose, .verbose):
            same = true
        default:
            same = false
        }
        XCTAssertTrue(same, "\(message) (got \(a), expected \(b))", file: file, line: line)
    }
}
