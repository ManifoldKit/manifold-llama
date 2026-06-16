import XCTest
import ManifoldInference
import ManifoldHardware
import ManifoldLlama

/// Covers the public `LlamaBackends.register(with:)` entry point — the registrar
/// used by `ManifoldKit.quickStart(backends: [LlamaBackends.self])`. Previously
/// 0% covered: a regression that registered the factory for the wrong
/// `ModelType`, dropped the `default: return nil` arm, or skipped
/// `declareSupport(for: .gguf)` would have shipped undetected.
@MainActor
final class LlamaBackendRegistrarTests: XCTestCase {

    func test_register_declaresGGUFSupport() {
        let service = InferenceService()
        // Sanity: GGUF is not supported before registration.
        XCTAssertFalse(service.registeredBackendSnapshot().supportsGGUF,
            "GGUF must not be supported before LlamaBackends.register runs")

        LlamaBackends.register(with: service)

        let snapshot = service.registeredBackendSnapshot()
        XCTAssertTrue(snapshot.supportsGGUF,
            "register(with:) must declareSupport(for: .gguf)")
    }

    func test_register_doesNotDeclareUnrelatedBackends() {
        let service = InferenceService()
        LlamaBackends.register(with: service)

        let snapshot = service.registeredBackendSnapshot()
        XCTAssertFalse(snapshot.supportsMLX,
            "The llama registrar must not claim MLX support")
        XCTAssertFalse(snapshot.supportsFoundation,
            "The llama registrar must not claim Foundation support")
        XCTAssertFalse(snapshot.supportsCloudInference,
            "The llama registrar must not register any cloud provider")
    }

    func test_register_reportsGGUFCompatible_otherTypesNot() {
        let service = InferenceService()
        LlamaBackends.register(with: service)

        XCTAssertTrue(service.compatibility(for: .gguf).isSupported,
            "A registered .gguf factory must report the type as supported")
        XCTAssertFalse(service.compatibility(for: .mlx).isSupported,
            "No MLX factory was registered — .mlx must not be supported")
    }
}
