#if DEBUG
import XCTest
@_spi(Testing) import ManifoldLlamaKit

/// Tests for ``LlamaBackendProcessLifecycle`` log-level get/set.
///
/// ``InferenceService/llamaLogLevel`` is a thin `@MainActor` wrapper around
/// ``LlamaBackendProcessLifecycle/setLogLevel(_:)`` and ``currentLogLevel``.
/// These tests exercise the underlying lifecycle methods directly so the full
/// get/set contract is covered without a MainActor hop. The `_currentLogLevelForTesting`
/// accessor confirms the stored value; `currentLogLevel` confirms the public read path.
final class LlamaLogLevelTests: XCTestCase {

    override func tearDown() {
        super.tearDown()
        // Restore .info (the default) so log-state changes don't bleed into
        // other tests running in the same process.
        LlamaBackendProcessLifecycle.setLogLevel(.info)
    }

    func test_setLogLevel_silent_roundtrips() {
        LlamaBackendProcessLifecycle.setLogLevel(.silent)
        XCTAssertEqual(LlamaBackendProcessLifecycle.currentLogLevel, .silent)
        XCTAssertEqual(LlamaBackendProcessLifecycle._currentLogLevelForTesting, .silent)
    }

    func test_setLogLevel_warning_roundtrips() {
        LlamaBackendProcessLifecycle.setLogLevel(.warning)
        XCTAssertEqual(LlamaBackendProcessLifecycle.currentLogLevel, .warning)
        XCTAssertEqual(LlamaBackendProcessLifecycle._currentLogLevelForTesting, .warning)
    }

    func test_setLogLevel_info_roundtrips() {
        LlamaBackendProcessLifecycle.setLogLevel(.info)
        XCTAssertEqual(LlamaBackendProcessLifecycle.currentLogLevel, .info)
        XCTAssertEqual(LlamaBackendProcessLifecycle._currentLogLevelForTesting, .info)
    }

    func test_setLogLevel_verbose_roundtrips() {
        LlamaBackendProcessLifecycle.setLogLevel(.verbose)
        XCTAssertEqual(LlamaBackendProcessLifecycle.currentLogLevel, .verbose)
        XCTAssertEqual(LlamaBackendProcessLifecycle._currentLogLevelForTesting, .verbose)
    }

    func test_setLogLevel_allLevelsInSequence_eachPersistsUntilNextSet() {
        let sequence: [LlamaLogLevel] = [.silent, .warning, .verbose, .info, .silent]
        for level in sequence {
            LlamaBackendProcessLifecycle.setLogLevel(level)
            XCTAssertEqual(LlamaBackendProcessLifecycle.currentLogLevel, level,
                           "currentLogLevel must equal \(level) immediately after setLogLevel")
        }
    }

    func test_setLogLevel_currentLogLevel_matchesTestingAccessor() {
        for level: LlamaLogLevel in [.silent, .warning, .info, .verbose] {
            LlamaBackendProcessLifecycle.setLogLevel(level)
            XCTAssertEqual(
                LlamaBackendProcessLifecycle.currentLogLevel,
                LlamaBackendProcessLifecycle._currentLogLevelForTesting,
                "currentLogLevel and _currentLogLevelForTesting must agree for \(level)"
            )
        }
    }
}
#endif
