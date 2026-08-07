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
        // Exact equality, not XCTAssertNotEqual(quant, "unknown") — the original
        // version of this assertion passed on the truncated "Q4" the pre-fix
        // parser actually returned (rev-183 finding on #183): it proved the
        // capability didn't TOTALLY fail, not that it worked. The fixture's
        // real filename carries "Q4_K_M"; assert the whole label survives.
        XCTAssertEqual(record.quant, "Q4_K_M", "the fixture's real quant label must survive whole, not truncated to its leading \"Q4\"")
        XCTAssertEqual(record.renderer, "jinja-prompt")
        XCTAssertNotNil(record.verdict, "a measured record always carries a verdict")
    }

    // MARK: - quantLabel(fromFileName:) precision (rev-183 blocker 2 on #183)

    /// Drives the CLI's missing-GGUF absence path (no real weights needed —
    /// `quantLabel(fromFileName:)` runs on the file NAME before the
    /// existence check's error path returns) purely to exercise quant-label
    /// parsing against a crafted file name, and returns the emitted record's
    /// `quant` field.
    private func quantFromMissingModel(named fileName: String) throws -> String {
        let tmp = FileManager.default.temporaryDirectory
        let recordsURL = tmp.appendingPathComponent("records-\(UUID().uuidString).json")
        let transcriptURL = tmp.appendingPathComponent("transcript-\(UUID().uuidString).jsonl")
        let missingModel = tmp.appendingPathComponent("quant-probe-\(UUID().uuidString)", isDirectory: true)
            .appendingPathComponent(fileName)
        defer {
            try? FileManager.default.removeItem(at: recordsURL)
            try? FileManager.default.removeItem(at: transcriptURL)
        }

        _ = try runCLI([
            "--model", missingModel.path,
            "--scenario", "01-now",
            "--output", transcriptURL.path,
            "--emit-records", recordsURL.path,
        ])
        let records = try decodeRecords(at: recordsURL)
        return try XCTUnwrap(records.first).quant
    }

    /// The defect this fixes truncated "Q4_K_M" to "Q4": splitting on "_"
    /// broke the K/M sub-label into its own tokens before the detector ever
    /// saw the full marker (13 of 14 real GGUFs on rev-183's machine
    /// truncated this way).
    func test_quantLabel_fullKQuantSuffix_notTruncated() throws {
        XCTAssertEqual(try quantFromMissingModel(named: "model-Q4_K_M.gguf"), "Q4_K_M")
    }

    /// The defect this fixes returned "unknown" for the entire I-quant family
    /// — the old detector's `hasPrefix("q")` check never matched a leading
    /// "I" (IQ4_XS, IQ2_XXS, ...) at all: total loss, not truncation.
    func test_quantLabel_iQuant_notUnknown() throws {
        XCTAssertEqual(try quantFromMissingModel(named: "model-IQ4_XS.gguf"), "IQ4_XS")
    }

    /// The defect this fixes could return the WRONG label, not just a
    /// truncated one: a reversed token scan with no ordering guard between
    /// the quant and precision branches returned whichever pattern the scan
    /// hit last, so "m-Q4_K_M-f16.gguf" returned "f16" — a 4-bit K-quant cell
    /// mislabelled full-precision. A quant match must always win over a
    /// precision match, regardless of which sits later in the name.
    func test_quantLabel_prefersQuantOverTrailingPrecisionToken() throws {
        XCTAssertEqual(try quantFromMissingModel(named: "m-Q4_K_M-f16.gguf"), "Q4_K_M")
    }

    // MARK: - Blocker 1 (rev-183): a mid-stream scenario failure must not read as measured/fail

    private func context(transcriptRef: String = "probe") -> ConformanceScorer.RecordContext {
        ConformanceScorer.RecordContext(renderer: "jinja-prompt", coreCommit: "test", transcriptRef: transcriptRef)
    }

    /// Validates the CORE mechanism `main.swift`'s scenario-loop catch block
    /// now relies on (appending `.error(scenarioId:message:)` on any thrown
    /// error), rather than driving the real CLI: forcing llama.cpp to throw
    /// mid-generation deterministically (a decode failure, context overflow)
    /// is not something a test can trigger against a real model on demand.
    /// What CAN be driven directly is `ConformanceScorer`'s behavior on the
    /// two transcript shapes the fix distinguishes — the exact demonstration
    /// rev-183 used to find the defect (a hand-built transcript group of
    /// "partial turn, zero assertions, no error event").
    func test_partialTurnWithoutErrorEvent_documentsPreFixDefectShape_andFixRoutesToHole() throws {
        // BEFORE this PR's fix: the scenario streamed one token and then threw,
        // so `token_delta` proves `producedModelTurn == true`, but nothing ever
        // logged the failure — `errored` stays false with zero assertions.
        let unfixedTranscript = """
        {"kind":"prompt","scenario":"partial","user":"x","requiredTools":[],"backend":"llama.cpp","model":"m.gguf","quant":"Q4_K_M"}
        {"kind":"token_delta","scenario":"partial","text":"partial output","backend":"llama.cpp","model":"m.gguf","quant":"Q4_K_M"}
        """
        let unfixedRecords = ConformanceScorer.records(jsonl: unfixedTranscript, context: context())
        let unfixedRecord = try XCTUnwrap(unfixedRecords.first)
        // Documents the pre-fix defect this blocker exists to close: a cell
        // that was never actually measured reads as a fabricated MEASURED
        // FAIL — precisely the "absence read as failure" confusion
        // CellStatus exists to prevent.
        XCTAssertEqual(unfixedRecord.status, .measured, "documents the pre-fix defect: no .error event, so the hole is invisible to the scorer")
        XCTAssertEqual(unfixedRecord.verdict, .fail, "documents the pre-fix defect: a fabricated failure, not a real measurement")

        // AFTER the fix: `main.swift`'s catch block now appends this event
        // before falling through to the next scenario.
        let fixedTranscript = unfixedTranscript + "\n"
            + #"{"kind":"error","scenario":"partial","message":"scenario run failed: decode failure","backend":"llama.cpp","model":"m.gguf","quant":"Q4_K_M"}"#
        let fixedRecords = ConformanceScorer.records(jsonl: fixedTranscript, context: context())
        let fixedRecord = try XCTUnwrap(fixedRecords.first)
        XCTAssertNotEqual(fixedRecord.status, .measured, "the fix must route a mid-stream failure to a hole, never a measured record")
        guard case .loadFail(let reason) = fixedRecord.status else {
            return XCTFail("expected .loadFail, got \(fixedRecord.status)")
        }
        XCTAssertTrue(reason.contains("decode failure"), "the real error detail must survive into the hole's reason")
        XCTAssertNil(fixedRecord.verdict, "a hole carries no verdict — this is what prevents the fabricated .fail above")
    }
}
