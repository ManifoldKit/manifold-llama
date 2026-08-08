import ManifoldTestSupport
import ManifoldTools
import XCTest

/// Coverage for the decoy-padding scoring fix (manifold-llama, `--extra-tools`):
/// decoys must be ADVERTISED to the model without being counted as REQUIRED for
/// scoring. Before this fix, `padScenario(_:advertisingAlso:)` spliced decoy
/// names directly into `scenario.requiredTools` — the same list the transcript
/// logs as ground truth and `ConformanceScorer` uses as the expected-tool set —
/// so a model that correctly declined every decoy scored a false negative per
/// decoy. Measured on real weights (see PR body): mean F1 0.700 -> 0.245 across
/// d0 -> +5 decoys before the fix, a scoring artifact masquerading as a real
/// backend/model degradation.
///
/// **Why subprocess:** `manifold-tools-llama` is an executable target whose
/// `main.swift` runs top-level code at module-load time — see
/// `ManifoldToolsLlamaEmitRecordsTests`'s doc comment for the full rationale.
final class ManifoldToolsLlamaDecoyScoringTests: XCTestCase {

  private func productsDirectory() throws -> URL {
    for bundle in Bundle.allBundles where bundle.bundlePath.hasSuffix(".xctest") {
      return bundle.bundleURL.deletingLastPathComponent()
    }
    throw XCTSkip("could not locate the .xctest bundle to find sibling build products")
  }

  private func binaryURL() throws -> URL {
    let binary = try productsDirectory().appendingPathComponent("manifold-tools-llama")
    guard FileManager.default.fileExists(atPath: binary.path) else {
      throw XCTSkip(
        "manifold-tools-llama binary not found at \(binary.path) — expected `swift build` to have produced it before `swift test` (CI always builds before testing)."
      )
    }
    return binary
  }

  @discardableResult
  private func runCLI(_ arguments: [String]) throws -> (
    exitCode: Int32, stdout: String, stderr: String
  ) {
    let process = Process()
    process.executableURL = try binaryURL()
    process.arguments = arguments
    let cwd = FileManager.default.temporaryDirectory
      .appendingPathComponent(
        "manifold-tools-llama-decoy-tests-\(UUID().uuidString)", isDirectory: true)
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

  /// Reads the `.prompt` event for `scenario` out of a transcript JSONL —
  /// the event `ScenarioRunner` logs before generation starts, carrying
  /// `requiredTools` (the scored expected set) and `advertisedTools` (every
  /// tool actually offered to the model).
  ///
  /// Fails loudly (not `XCTSkip`) when the event is absent: a missing
  /// `.prompt` event almost always means the run itself failed upstream
  /// (e.g. the model stopped loading after a llama.cpp bump) — exactly the
  /// class of failure this test exists to catch. `XCTSkip` here would let
  /// the suite's skip count quietly absorb a real regression instead of
  /// reddening for it (the guard-defeated-upstream shape this run has hit
  /// repeatedly elsewhere tonight).
  private func promptEvent(scenario: String, in transcriptURL: URL) throws -> (
    requiredTools: [String], advertisedTools: [String]
  ) {
    let text = try String(contentsOf: transcriptURL, encoding: .utf8)
    for line in text.split(separator: "\n") {
      guard let data = line.data(using: .utf8),
        let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
        obj["kind"] as? String == "prompt",
        obj["scenario"] as? String == scenario
      else { continue }
      let required = obj["requiredTools"] as? [String] ?? []
      let advertised = obj["advertisedTools"] as? [String] ?? []
      return (required, advertised)
    }
    XCTFail(
      "no .prompt event for scenario '\(scenario)' found in \(transcriptURL.path) — the run likely failed before logging one"
    )
    struct MissingPromptEvent: Error {}
    throw MissingPromptEvent()
  }

  private func decodeRecords(at url: URL) throws -> [ConformanceRecord] {
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode([ConformanceRecord].self, from: data)
  }

  /// Model-gated end-to-end check, real weights: `--extra-tools 5` must
  /// advertise the 5 decoys to the model WITHOUT counting any of them as
  /// required, and a model that (correctly) calls only the real tool must
  /// score a clean recall — not a penalty for the decoys it rightly ignored.
  ///
  /// A no-op fix (e.g. one that advertises decoys but forgets to exclude
  /// them from `requiredTools`, or vice versa) fails this test: the
  /// `requiredTools` assertion alone would pass if decoys were silently
  /// dropped from BOTH lists, so the `advertisedTools` superset check is
  /// load-bearing — it proves the decoys really were offered to the model,
  /// not just kept out of scoring by omission.
  func test_extraTools_advertisesDecoysWithoutInflatingRequiredTools() throws {
    guard let modelURL = HardwareRequirements.findGGUFModel(nameContains: "Qwen3.5-2B") else {
      throw XCTSkip(
        "No matching GGUF on disk. Set LLAMA_TEST_MODEL=<path> or place a Qwen3.5-2B GGUF under ~/Documents/Models/ to run this test."
      )
    }

    let tmp = FileManager.default.temporaryDirectory
    let transcriptURL = tmp.appendingPathComponent("transcript-\(UUID().uuidString).jsonl")
    let recordsURL = tmp.appendingPathComponent("records-\(UUID().uuidString).json")
    defer {
      try? FileManager.default.removeItem(at: transcriptURL)
      try? FileManager.default.removeItem(at: recordsURL)
    }

    let result = try runCLI([
      "--model", modelURL.path,
      "--scenario", "01-now",
      "--extra-tools", "5",
      "--output", transcriptURL.path,
      "--emit-records", recordsURL.path,
    ])
    // A `guard`, not a recorded-but-non-aborting `XCTAssertNotEqual`: if
    // the model failed to load, every assertion below is meaningless
    // against an empty/absent transcript. Failing here loudly, instead of
    // falling through into `promptEvent`'s "not found" path, keeps the
    // failure attributable to the real cause (a load failure) rather than
    // a downstream symptom.
    guard result.exitCode != 3 else {
      XCTFail("model failed to load (exit 3): \(result.stderr)")
      return
    }

    let (required, advertised) = try promptEvent(scenario: "01-now", in: transcriptURL)
    let decoyNames = Set(DecoyTools.names(5))
    XCTAssertFalse(decoyNames.isEmpty, "sanity: the decoy pool must actually have entries")

    // The load-bearing assertion: no decoy name ever appears in the SCORED
    // expected-tool set.
    XCTAssertEqual(
      Set(required), ["now"],
      "requiredTools must be exactly the scenario's own declared tool — never a decoy name")
    for decoy in decoyNames {
      XCTAssertFalse(
        required.contains(decoy),
        "decoy '\(decoy)' leaked into requiredTools — this is exactly the false-negative bug this fix closes"
      )
    }

    // The decoys must still reach the model, or "not scored" is
    // indistinguishable from "not tested" — advertisedTools is where a
    // decoy SHOULD show up.
    XCTAssertTrue(
      decoyNames.isSubset(of: Set(advertised)),
      "all 5 decoys must be advertised to the model even though none are required")
    XCTAssertTrue(
      advertised.contains("now"), "the real tool must still be advertised alongside the decoys")

    // Scoring: a model that only calls the real tool and ignores every
    // decoy must score a clean recall, not a penalty. (Precision/recall
    // over the correct single-tool dispatch — verified against the real
    // scored record, not assumed from the transcript alone.)
    let records = try decodeRecords(at: recordsURL)
    let record = try XCTUnwrap(records.first { $0.scenario == "01-now" })
    XCTAssertEqual(record.status, .measured, "stderr: \(result.stderr)")
    let scores = try XCTUnwrap(record.toolSelection, "a tool-bearing scenario must carry Scores")
    XCTAssertEqual(
      scores.recall, 1.0, accuracy: 1e-9,
      "declining every decoy must not cost recall — the real tool was correctly dispatched")
  }
}
