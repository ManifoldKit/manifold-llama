import XCTest

/// Coverage for `manifold-llama-eval`'s hand-rolled `--top-k` / `--repeat-penalty`
/// argument parsing and validation.
///
/// **Why subprocess, not `@testable import`:** `CLI.parse`'s `fail(_:)` helper
/// calls the real `exit(2)` on invalid input (see `main.swift`) — importing the
/// executable target and calling `CLI.parse` in-process with a deliberately bad
/// value would terminate the whole XCTest process, not just fail one test. These
/// tests instead spawn the real built `manifold-llama-eval` binary and assert on
/// its exit code / stderr — the same black-box contract a real caller observes.
/// This mirrors the "unit-testable ... without a model" split already
/// established for `ManifoldLlamaEvalKit` (see its target's doc comment in
/// `Package.swift`): the executable's argument parsing has no separate library
/// seam, so the process boundary is the seam here.
///
/// No Metal / Apple Silicon / GGUF model is required: `--top-k` and
/// `--repeat-penalty` are validated during argument parsing, before the runner
/// ever checks for `--model` / `--prompt-file` or touches `LlamaBackend`. These
/// tests run unconditionally in CI.
final class EvalCLIArgumentParsingTests: XCTestCase {

    /// Locates the directory `swift test` places build products in (the same
    /// directory the `.xctest` bundle itself lives in) — the standard SwiftPM
    /// pattern for tests that need to shell out to a sibling executable product.
    private func productsDirectory() throws -> URL {
        for bundle in Bundle.allBundles where bundle.bundlePath.hasSuffix(".xctest") {
            return bundle.bundleURL.deletingLastPathComponent()
        }
        throw XCTSkip("could not locate the .xctest bundle to find sibling build products")
    }

    private func runCLI(_ arguments: [String]) throws -> (exitCode: Int32, stdout: String, stderr: String) {
        let binary = try productsDirectory().appendingPathComponent("manifold-llama-eval")
        guard FileManager.default.fileExists(atPath: binary.path) else {
            throw XCTSkip("manifold-llama-eval binary not found at \(binary.path) — expected `swift build` to have produced it before `swift test` (CI always builds before testing).")
        }

        let process = Process()
        process.executableURL = binary
        process.arguments = arguments
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

    // MARK: - --top-k

    func test_topK_rejectsNegativeValue() throws {
        let result = try runCLI(["--top-k", "-1"])
        XCTAssertEqual(result.exitCode, 2, "negative --top-k must be a bad-argument exit (2); stderr: \(result.stderr)")
        XCTAssertTrue(result.stderr.contains("--top-k"), "stderr should name the offending flag; got: \(result.stderr)")
    }

    func test_topK_rejectsNonInteger() throws {
        let result = try runCLI(["--top-k", "not-a-number"])
        XCTAssertEqual(result.exitCode, 2)
        XCTAssertTrue(result.stderr.contains("--top-k"), "got: \(result.stderr)")
    }

    func test_topK_acceptsZeroAndPositiveValues() throws {
        // `0` and positive values must NOT be rejected at the parsing stage —
        // only the (missing) --model/--prompt-file requirement should fire next,
        // which exits 2 with a DIFFERENT message ("--model ... is required").
        // This distinguishes "flag rejected" from "flag accepted, something else
        // is missing".
        for value in ["0", "40", "1"] {
            let result = try runCLI(["--top-k", value])
            XCTAssertEqual(result.exitCode, 2)
            XCTAssertFalse(result.stderr.contains("--top-k requires"),
                            "--top-k \(value) should parse cleanly; got: \(result.stderr)")
            XCTAssertTrue(result.stderr.contains("--model"),
                          "expected the run to fail on the missing --model requirement next; got: \(result.stderr)")
        }
    }

    // MARK: - --repeat-penalty

    func test_repeatPenalty_rejectsZero() throws {
        let result = try runCLI(["--repeat-penalty", "0"])
        XCTAssertEqual(result.exitCode, 2, "non-positive --repeat-penalty must be a bad-argument exit (2); stderr: \(result.stderr)")
        XCTAssertTrue(result.stderr.contains("--repeat-penalty"), "got: \(result.stderr)")
    }

    func test_repeatPenalty_rejectsNegativeValue() throws {
        let result = try runCLI(["--repeat-penalty", "-1.1"])
        XCTAssertEqual(result.exitCode, 2)
        XCTAssertTrue(result.stderr.contains("--repeat-penalty"), "got: \(result.stderr)")
    }

    func test_repeatPenalty_rejectsNonNumeric() throws {
        let result = try runCLI(["--repeat-penalty", "not-a-number"])
        XCTAssertEqual(result.exitCode, 2)
        XCTAssertTrue(result.stderr.contains("--repeat-penalty"), "got: \(result.stderr)")
    }

    func test_repeatPenalty_acceptsPositiveValues() throws {
        for value in ["1.0", "1.1", "0.5"] {
            let result = try runCLI(["--repeat-penalty", value])
            XCTAssertEqual(result.exitCode, 2)
            XCTAssertFalse(result.stderr.contains("--repeat-penalty requires"),
                            "--repeat-penalty \(value) should parse cleanly; got: \(result.stderr)")
            XCTAssertTrue(result.stderr.contains("--model"),
                          "expected the run to fail on the missing --model requirement next; got: \(result.stderr)")
        }
    }

    // MARK: - --help documents the new flags

    func test_help_documentsNewFlags() throws {
        let result = try runCLI(["--help"])
        XCTAssertEqual(result.exitCode, 0)
        XCTAssertTrue(result.stdout.contains("--top-k"), "usage text must document --top-k")
        XCTAssertTrue(result.stdout.contains("--repeat-penalty"), "usage text must document --repeat-penalty")
    }
}
