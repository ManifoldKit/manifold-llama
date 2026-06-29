import XCTest
import ManifoldInference
import ManifoldTestSupport
@testable import ManifoldLlamaEvalKit

/// Tests for the raw-prompt eval runner (`manifold-llama-eval`), the llama.cpp
/// leg of the manifold-eval same-GGUF cross-backend differential.
///
/// Two tiers:
/// - **CI (no model):** `RawRun` JSON encoding + metadata helpers — pure,
///   deterministic, always run.
/// - **Local (model-gated):** drives `EvalRunner.run` against a small on-disk
///   GGUF when one is discoverable (`HardwareRequirements.findGGUFModel`,
///   opt-in via `MANIFOLD_DISCOVER_LOCAL_MODELS=1` / `LLAMA_TEST_MODEL`).
///   XCTSkips cleanly on CI / hosts without a model.
final class RawPromptEvalRunnerTests: XCTestCase {

    // MARK: - CI: pure RawRun encoding (no model)

    /// The `RawRun` contract encodes to JSON with the exact field names
    /// manifold-eval parses — including the dotted `"llama.cpp"` tooling key —
    /// and round-trips losslessly.
    func test_rawRun_encodesContractShape_andRoundTrips() throws {
        let run = RawRun(
            backend: "llama.cpp",
            model: "Qwen3-0.6B-Q4_K_M",
            quant: "Q4_K_M",
            promptSha256: "abc123",
            inputTokenIds: [1, 2, 3],
            output: "hello",
            outputTokenIds: [],
            sampler: RawRun.Sampler(
                temperature: 0.0, seed: 42, topK: 0, repeatPenalty: 1.0, maxTokens: 16),
            coreCommit: "deadbeef",
            toolingVersions: RawRun.ToolingVersions(llamaCpp: "b9744"),
            repeatIndex: 0)

        let line = try run.encodedJSONLine()

        // Single line (one JSON object per invocation).
        XCTAssertFalse(line.contains("\n"), "RawRun JSON must be a single line")

        // The dotted tooling key and the prompt-hash key are present verbatim.
        XCTAssertTrue(line.contains("\"llama.cpp\":\"b9744\""),
                      "tooling key must be the dotted \"llama.cpp\"; got \(line)")
        XCTAssertTrue(line.contains("\"promptSha256\":\"abc123\""))
        XCTAssertTrue(line.contains("\"backend\":\"llama.cpp\""))

        // Round-trips losslessly through the same contract type.
        let data = Data(line.utf8)
        let decoded = try JSONDecoder().decode(RawRun.self, from: data)
        XCTAssertEqual(decoded, run)
        XCTAssertEqual(decoded.toolingVersions.llamaCpp, "b9744")
        XCTAssertEqual(decoded.sampler.temperature, 0.0)
    }

    /// Quant extraction handles the common GGUF file-name conventions.
    func test_parseQuant_recognizesCommonTags() {
        XCTAssertEqual(EvalMetadata.parseQuant(fromFileName: "Qwen3-0.6B-Q4_K_M.gguf"), "Q4_K_M")
        XCTAssertEqual(EvalMetadata.parseQuant(fromFileName: "model-IQ3_XXS.gguf"), "IQ3_XXS")
        XCTAssertEqual(EvalMetadata.parseQuant(fromFileName: "tiny-f16.gguf"), "F16")
        XCTAssertEqual(EvalMetadata.parseQuant(fromFileName: "no-quant-here.gguf"), "unknown")
    }

    /// The prompt hash is the SHA-256 of the exact bytes (a known vector).
    func test_sha256Hex_matchesKnownVector() {
        // SHA-256("abc") — canonical NIST vector.
        let digest = EvalMetadata.sha256Hex(Data("abc".utf8))
        XCTAssertEqual(
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    }

    /// Core-commit resolution reads a ManifoldKit pin's revision out of a
    /// `Package.resolved` (preferring revision over version).
    func test_resolveCoreCommit_readsManifoldKitPinRevision() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("rawrun-coremeta-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let resolved = dir.appendingPathComponent("Package.resolved")
        let json = """
        {
          "pins": [
            {
              "identity": "manifoldkit",
              "kind": "remoteSourceControl",
              "location": "https://github.com/ManifoldKit/ManifoldKit",
              "state": { "revision": "454cb8893a77da25160efc112aa5ca299096f02e", "version": "0.63.0" }
            }
          ],
          "version": 3
        }
        """
        try Data(json.utf8).write(to: resolved)

        let commit = EvalMetadata.resolveCoreCommit(searchStartDirectories: [dir])
        XCTAssertEqual(commit, "454cb8893a77da25160efc112aa5ca299096f02e")
    }

    func test_resolveCoreCommit_returnsUnknownWhenAbsent() {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("rawrun-empty-\(UUID().uuidString)", isDirectory: true)
        // Directory does not exist / has no Package.resolved up the chain we care
        // about — resolution must degrade to "unknown" rather than throwing.
        let commit = EvalMetadata.resolveCoreCommit(searchStartDirectories: [dir])
        XCTAssertEqual(commit, "unknown")
    }

    // MARK: - Local: model-gated end-to-end run

    /// Drives `EvalRunner.run` against a small on-disk GGUF and asserts the
    /// produced `RawRun` is well-formed: non-empty output, non-empty input token
    /// ids, a set prompt hash, and temperature 0 reflected in the sampler.
    ///
    /// Gated on Apple Silicon + Metal + a discoverable GGUF — XCTSkips on CI.
    func test_evalRunner_producesWellFormedRawRun() async throws {
        try XCTSkipUnless(HardwareRequirements.isPhysicalDevice,
                          "LlamaBackend requires Metal (unavailable in simulator)")
        try XCTSkipUnless(HardwareRequirements.isAppleSilicon,
                          "LlamaBackend requires Apple Silicon")
        guard let modelURL = HardwareRequirements.findGGUFModel() else {
            throw XCTSkip("No GGUF on disk. Set LLAMA_TEST_MODEL=<path> or "
                        + "MANIFOLD_DISCOVER_LOCAL_MODELS=1 with a `.gguf` in ~/Documents/Models/.")
        }

        let promptURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("rawrun-prompt-\(UUID().uuidString).txt")
        let promptText = "The capital of France is"
        try Data(promptText.utf8).write(to: promptURL)
        defer { try? FileManager.default.removeItem(at: promptURL) }

        let options = EvalOptions(
            modelPath: modelURL.path,
            promptFile: promptURL.path,
            temperature: 0.0,
            seed: 42,
            maxTokens: 16,
            repeatIndex: 0,
            requestedContextSize: 512)

        let run = try await EvalRunner.run(options)

        XCTAssertEqual(run.backend, "llama.cpp")
        XCTAssertFalse(run.output.isEmpty, "greedy generation must produce output")
        XCTAssertFalse(run.inputTokenIds.isEmpty, "prompt must tokenize to >= 1 token")
        XCTAssertFalse(run.promptSha256.isEmpty, "prompt hash must be set")
        XCTAssertEqual(run.promptSha256, EvalMetadata.sha256Hex(Data(promptText.utf8)),
                       "prompt hash must be SHA-256 of the verbatim prompt bytes")
        XCTAssertEqual(run.sampler.temperature, 0.0)
        XCTAssertEqual(run.sampler.maxTokens, 16)
        XCTAssertEqual(run.sampler.seed, 42)
        XCTAssertEqual(run.toolingVersions.llamaCpp, EvalMetadata.llamaCppBuild)

        // The record must encode to a single parseable JSON line.
        let line = try run.encodedJSONLine()
        XCTAssertFalse(line.contains("\n"))
        _ = try JSONDecoder().decode(RawRun.self, from: Data(line.utf8))
    }
}
