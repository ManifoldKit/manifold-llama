import ManifoldInference

/// The llama.cpp (GGUF) family registrar.
///
/// Moved in from core's `ManifoldBackendsUmbrella` in the v0.48 companion
/// split (core PR C2, ManifoldKit#1749) and de-`#if`'d — this package always
/// compiles the backend, so registration is unconditional.
///
/// ```swift
/// import ManifoldKit
/// import ManifoldLlama
///
/// let kit = try await ManifoldKit.quickStart(backends: [LlamaBackends.self])
/// ```
public enum LlamaBackends: BackendRegistrar {
    @MainActor
    public static func register(with service: InferenceService) {
        service.registerBackendFactory { modelType in
            switch modelType {
            case .gguf: return LlamaBackend()
            default:    return nil
            }
        }
        service.declareSupport(for: .gguf)
    }
}
