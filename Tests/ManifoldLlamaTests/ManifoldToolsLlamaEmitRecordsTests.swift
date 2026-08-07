import XCTest
import ManifoldTestSupport
import ManifoldTools

/// Coverage for `manifold-tools-llama --emit-records` (manifold-llama#178): the
/// llama.cpp leg previously had no way to emit the normalized `ConformanceRecord`
/// schema, unlike its Ollama (core) and MLX siblings, so it contributed nothing
/// to the cross-runtime tool-selection matrix.
///
/// **Why subprocess, not `@testable import`:** `manifold-tools-llama` is an
/// executable target whose `main.swift` runs top-level code (including a bare
/// `exit(exitCode)`) at module-load time — importing it in-process would run
/// the CLI (and terminate the whole XCTest process) as a side effect of the
/// `import` statement. This mirrors the documented rationale in
/// `EvalCLIArgumentParsingTests`: the process boundary is the seam.
final class ManifoldToolsLlamaEmitRecordsTests: XCTestCase {

    /// Locates the directory `swift test` places build products in (the same
    /// directory the `.xctest` bundle itself lives in) — the standard SwiftPM
    /// pattern for tests that shell out to a sibling executable product.
    private func productsDirectory() throws -> URL {
        for bundle in Bundle.allBundles where bundle.bundlePath.hasSuffix(".xctest") {
            return bundle.bundleURL.deletingLastPathComponent()
        }
        throw XCTSkip("could not locate the .xctest bundle to find sibling build products")
    }

    private func binaryURL() throws -> URL {
        let binary = try productsDirectory().appendingPathComponent("manifold-tools-llama")
        guard FileManager.default.fileExists(atPath: binary.path) else {
            throw XCTSkip("manifold-tools-llama binary not found at \(binary.path) — expected `swift build` to have produced it before `swift test` (CI always builds before testing).")
        }
        return binary
    }

    @discardableResult
    private func runCLI(_ arguments: [String]) throws -> (exitCode: Int32, stdout: String, stderr: String) {
        let process = Process()
        process.executableURL = try binaryURL()
        process.arguments = arguments
        // Run from a scratch cwd so the CLI's own relative-path defaults (e.g.
        // its default --output under tmp/manifold-tools-llama/) never collide
        // with a concurrent invocation or leave litter in the repo checkout.
        let cwd = FileManager.default.temporaryDirectory
            .appendingPathComponent("manifold-tools-llama-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: cwd, withIntermediateDirectories: true)
        process.currentDirectoryURL = cwd
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        try process.run()
        process.waitUntilExit()

        let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
        let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
        return (
            process.terminationStatus,
            String(decoding: stdoutData, as: UTF8.self),
            String(decoding: stderrData, as: UTF8.self)
        )
    }

    private func decodeRecords(at url: URL) throws -> [ConformanceRecord] {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode([ConformanceRecord].self, from: data)
    }

    // MARK: - Absence: a missing GGUF must read as notMeasured, never fail

    /// Load-bearing assertion (this is the exact confusion `CellStatus` exists to
    /// prevent, per `ConformanceRecord.swift`'s doc comment): a GGUF path that
    /// does not exist on disk must produce `.notMeasured` records — never
    /// `.measured` with a `fail` verdict, and never silently zero records. A
    /// no-op emitter (one that writes `[]` or skips the file) fails this test;
    /// so does one that mislabels the hole as `.loadFail` instead of
    /// `.notMeasured` (sabotage-verified: flipping the `main.swift` call site
    /// from `.notMeasured(...)` to `.loadFail(...)` turns this test red).
    func test_missingGGUF_emitsNotMeasuredNotFail() throws {
        let tmp = FileManager.default.temporaryDirectory
        let recordsURL = tmp.appendingPathComponent("records-\(UUID().uuidString).json")
        let transcriptURL = tmp.appendingPathComponent("transcript-\(UUID().uuidString).jsonl")
        let missingModel = tmp.appendingPathComponent("does-not-exist-\(UUID().uuidString).gguf")
        defer {
            try? FileManager.default.removeItem(at: recordsURL)
            try? FileManager.default.removeItem(at: transcriptURL)
        }

        let result = try runCLI([
            "--model", missingModel.path,
            "--scenario", "01-now",
            "--output", transcriptURL.path,
            "--emit-records", recordsURL.path,
        ])

        XCTAssertEqual(result.exitCode, 1, "a missing model file is exit 1, not a load failure (3) or a scenario failure")
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: recordsURL.path),
            "--emit-records must write a file even when the model never loaded — that is exactly the absence case CellStatus exists for"
        )

        let records = try decodeRecords(at: recordsURL)
        XCTAssertEqual(records.count, 1, "one hole for the one requested scenario")
        let record = try XCTUnwrap(records.first)

        XCTAssertEqual(record.scenario, "01-now")
        XCTAssertEqual(record.backend, "llama.cpp")
        XCTAssertEqual(record.model, missingModel.lastPathComponent)

        switch record.status {
        case .notMeasured:
            break // expected
        case .measured:
            XCTFail("a missing GGUF must never read as measured")
        case .loadFail, .renderFail:
            XCTFail("a missing GGUF is an absence, not a load attempt that failed — got \(record.status)")
        }

        XCTAssertNil(record.verdict, "an un-measured hole carries no verdict — verdict == .fail would be exactly the false-failure this schema prevents")
        XCTAssertNil(record.toolSelection, "no measurement means no tool-selection scores")
    }

    /// A missing-GGUF hole must be emitted for EVERY scenario the invocation
    /// requested (`--scenario all`), not just the first — otherwise a sweep that
    /// asks for the full corpus silently loses coverage for every scenario past
    /// the first.
    func test_missingGGUF_emitsHoleForEveryRequestedScenario() throws {
        let tmp = FileManager.default.temporaryDirectory
        let listURL = tmp.appendingPathComponent("list-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: listURL) }
        let recordsURL = tmp.appendingPathComponent("records-\(UUID().uuidString).json")
        let transcriptURL = tmp.appendingPathComponent("transcript-\(UUID().uuidString).jsonl")
        let missingModel = tmp.appendingPathComponent("does-not-exist-\(UUID().uuidString).gguf")
        defer {
            try? FileManager.default.removeItem(at: recordsURL)
            try? FileManager.default.removeItem(at: transcriptURL)
        }

        let listing = try runCLI(["--list"])
        // "  <id> — <description>" per line, per printUsage's --list branch.
        let expectedCount = listing.stdout
            .split(separator: "\n")
            .filter { $0.hasPrefix("  ") }
            .count
        XCTAssertGreaterThan(expectedCount, 1, "sanity: the bundled corpus has more than one scenario")

        let result = try runCLI([
            "--model", missingModel.path,
            "--scenario", "all",
            "--output", transcriptURL.path,
            "--emit-records", recordsURL.path,
        ])
        XCTAssertEqual(result.exitCode, 1)

        let records = try decodeRecords(at: recordsURL)
        XCTAssertEqual(records.count, expectedCount, "a hole per scenario in the requested (here: full) corpus")
        XCTAssertTrue(records.allSatisfy {
            if case .notMeasured = $0.status { return true }
            return false
        })
    }

    // MARK: - Real run: measured records carry the real cell identity

    /// Model-gated end-to-end check that the wiring is actually live: a real run
    /// against a real GGUF must emit `.measured` records whose backend/model/quant
    /// are the REAL cell — not "unknown"/"unknown"/"unknown", which is what every
    /// record would carry if `TranscriptLogger` weren't stamped with attribution
    /// (it previously wasn't; see the source change in this PR). Skips cleanly
    /// when no GGUF is available (CI has none).
    func test_realModel_measuredRecordsCarryRealCellIdentity() throws {
        guard let modelURL = HardwareRequirements.findGGUFModel(nameContains: "Qwen3.5-2B") else {
            throw XCTSkip("No matching GGUF on disk. Set LLAMA_TEST_MODEL=<path> or place a Qwen3.5-2B GGUF under ~/Documents/Models/ to run this test.")
        }

        let tmp = FileManager.default.temporaryDirectory
        let recordsURL = tmp.appendingPathComponent("records-\(UUID().uuidString).json")
        let transcriptURL = tmp.appendingPathComponent("transcript-\(UUID().uuidString).jsonl")
        defer {
            try? FileManager.default.removeItem(at: recordsURL)
            try? FileManager.default.removeItem(at: transcriptURL)
        }

        let result = try runCLI([
            "--model", modelURL.path,
            "--scenario", "01-now",
            "--output", transcriptURL.path,
            "--emit-records", recordsURL.path,
        ])

        XCTAssertNotEqual(result.exitCode, 3, "model load must have succeeded for this assertion to be meaningful: \(result.stderr)")
        XCTAssertTrue(FileManager.default.fileExists(atPath: recordsURL.path), "stderr: \(result.stderr)")

        let records = try decodeRecords(at: recordsURL)
        let record = try XCTUnwrap(records.first { $0.scenario == "01-now" })

        XCTAssertEqual(record.status, .measured)
        XCTAssertEqual(record.backend, "llama.cpp")
        XCTAssertEqual(record.model, modelURL.lastPathComponent, "must be the real GGUF file name, not a placeholder")
        XCTAssertNotEqual(record.quant, "unknown", "the Qwen3.5-2B fixture on disk carries a real Qxx quant token in its filename")
        XCTAssertEqual(record.renderer, "jinja-prompt")
        XCTAssertNotNil(record.verdict, "a measured record always carries a verdict")
    }
}
