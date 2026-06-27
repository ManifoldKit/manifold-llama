// manifold-tools-llama bfcl — run ManifoldKit's BFCL argument-level tool-call
// eval against a real llama.cpp / GGUF model.
//
// Reuses the shared, backend-agnostic `BFCLRunner` from the ManifoldTools library
// (scoring loop, output format, capture records) and the vendored BFCL fixtures;
// the only Llama-specific wiring is building the `InferenceService` on a
// `LlamaBackend` via the production load path — identical to the scenario harness
// in main.swift, so the model's embedded chat template renders the native tool
// block (#69).
import Foundation
import ManifoldInference
import ManifoldModelCatalog
import ManifoldTools
import ManifoldLlama

@MainActor
enum BFCLLlamaCLI {

    static func run(_ args: [String]) async -> Int32 {
        var modelPath: String?
        var category = "multiple"
        var dumpPath: String?

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--model":
                if i + 1 < args.count { modelPath = args[i + 1]; i += 1 }
            case "--category":
                if i + 1 < args.count { category = args[i + 1]; i += 1 }
            case "--dump":
                if i + 1 < args.count { dumpPath = args[i + 1]; i += 1 }
            case "-h", "--help":
                print("usage: manifold-tools-llama bfcl --model <path.gguf> [--category simple|multiple] [--dump PATH.jsonl]")
                return 0
            default:
                FileHandle.standardError.write(Data("unknown flag '\(args[i])'\n".utf8))
                return 2
            }
            i += 1
        }

        guard let modelPath else {
            FileHandle.standardError.write(Data("manifold-tools-llama bfcl: --model <path.gguf> is required\n".utf8))
            return 2
        }
        let modelURL = URL(fileURLWithPath: modelPath)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            FileHandle.standardError.write(Data("model file not found: \(modelURL.path)\n".utf8))
            return 1
        }

        let cases: [BFCLLoadedCase]
        do {
            cases = try BFCLCaseLoader.loadBundled(category: category)
        } catch {
            FileHandle.standardError.write(Data("failed to load BFCL '\(category)' cases: \(error)\n".utf8))
            return 1
        }
        print("BFCL category: \(category)")

        // Empty registry: BFCLRunner captures the model's first tool call and
        // scores it; we never dispatch. Tools are advertised via GenerationConfig.
        let service = InferenceService(toolRegistry: ToolRegistry())
        // Register the GGUF factory so loadModel(from:plan:) constructs the backend
        // through the coordinator — the path that captures the embedded chat
        // template and renders the native tool block (see main.swift #69 note).
        let backendBox = BackendBox()
        service.registerBackendFactory { modelType in
            guard modelType == .gguf else { return nil }
            let backend = LlamaBackend()
            backendBox.backend = backend
            return backend
        }
        service.declareSupport(for: .gguf)

        let modelInfo: ModelInfo
        do {
            modelInfo = try ModelInfo.load(ggufURL: modelURL)
        } catch {
            FileHandle.standardError.write(Data("failed to read GGUF metadata: \(error)\n".utf8))
            return 1
        }
        if modelInfo.chatTemplateRaw == nil {
            FileHandle.standardError.write(Data(
                "WARNING — \(modelURL.lastPathComponent) has no embedded tokenizer.chat_template; tool rendering falls back to the detected enum\n".utf8))
        }

        do {
            print("Loading model: \(modelURL.lastPathComponent)  (chat_template: \(modelInfo.chatTemplateRaw != nil ? "present" : "ABSENT"))")
            try await service.loadModel(from: modelInfo, plan: .systemManaged(requestedContextSize: 4096))
        } catch {
            let detail = (error as? LocalizedError)?.errorDescription ?? "\(error)"
            FileHandle.standardError.write(Data("LOAD FAILED: \(modelURL.lastPathComponent): \(detail)\n".utf8))
            if let backend = backendBox.backend { await backend.unloadAndWait() }
            return 3
        }

        let outcome = await BFCLRunner().run(
            cases: cases,
            service: service,
            modelLabel: "llama/\(modelURL.deletingPathExtension().lastPathComponent)"
        )

        var exitCode: Int32 = 0
        if let dumpPath {
            do {
                let body = try outcome.records.map { try $0.jsonLine() }.joined(separator: "\n")
                try (body + "\n").write(toFile: dumpPath, atomically: true, encoding: .utf8)
                print("\nWrote \(outcome.records.count) case record(s) → \(dumpPath)")
            } catch {
                FileHandle.standardError.write(Data("failed to write dump to \(dumpPath): \(error)\n".utf8))
                exitCode = 1
            }
        }

        // Deterministic teardown before process exit (Metal residency-set SIGABRT
        // guard — the coordinator's own unloadModel() is fire-and-forget).
        if let backend = backendBox.backend {
            await backend.unloadAndWait()
        } else {
            service.unloadModel()
        }
        return exitCode
    }
}
