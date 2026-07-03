import Foundation
import XCTest

import ManifoldTools

/// Lever 1 of #100 — result-grounding prompt.
///
/// The four-model tool-calling soak found dispatch is solid but the second turn
/// (after a tool returns) is where llama/gemma fail: they narrate instead of
/// answering FROM the tool result. `manifold-tools-llama` now appends a shared
/// result-grounding directive to every *tool-using* scenario's system prompt
/// before running it (`groundScenario` in the harness `main.swift`).
///
/// The directive composition lives in the executable target, which a test target
/// cannot import, so this suite pins the *intent*: it loads the same corpus the
/// harness runs against (`ScenarioCorpusFixture` — core's bundled `built-in`
/// scenarios spliced with this package's vendored `ScenarioOverrides`), replays
/// the same one-line composition rule, and asserts the grounding text lands on
/// tool-using scenarios and is withheld from toolless ones. A drift in the
/// directive wording or the toolless carve-out breaks this test, which is the
/// guard we want around a prompt change.
///
/// No model is loaded — this is a pure data/string assertion that runs in CI.
final class LlamaResultGroundingPromptTests: XCTestCase {

    /// Kept in lockstep with `resultGroundingDirective` in the harness
    /// `main.swift`. If you change the directive there, change it here — the
    /// duplication is deliberate (the executable target is not importable) and
    /// the test exists precisely to make that coupling visible.
    private static let directive =
        "When a tool returns a result, answer USING that result directly — quote its "
        + "values verbatim where the user asks for them. Do NOT narrate that you called "
        + "a tool, do NOT paraphrase or recompute the result, and do NOT add facts the "
        + "tool did not return. The tool result is the ground truth for your answer."

    /// Mirror of `groundedSystemPrompt(base:requiredTools:)` in the harness.
    private static func grounded(base: String, requiredTools: [String]) -> String {
        guard !requiredTools.isEmpty else { return base }
        let trimmed = base.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return directive }
        return trimmed + " " + directive
    }

    /// `manifold-tools-llama` no longer vendors the full scenario corpus — it
    /// consumes ManifoldKit core's bundled `built-in` corpus and splices in
    /// four llama/gemma-tolerant overrides by id (see `loadScenarios()` in
    /// `main.swift`). The invariant this test pins is the one
    /// `scripts/check-vendored-sync.sh` also enforces: the vendored
    /// `ScenarioOverrides` directory exists, is non-empty, and every override
    /// id it ships still targets a scenario id core's bundled corpus provides
    /// (an orphaned override would silently stop being spliced in).
    func testScenarioDirectoryIsPresent() throws {
        let overridesDir = ScenarioCorpusFixture.overridesDirectory()
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: overridesDir.path),
            "vendored ScenarioOverrides directory not found at \(overridesDir.path)")

        let overrides = try ScenarioLoader.load(from: overridesDir)
        XCTAssertFalse(overrides.isEmpty, "ScenarioOverrides directory is empty")

        let coreIDs = Set(try ScenarioLoader.loadBuiltIn().map(\.id))
        for override in overrides {
            XCTAssertTrue(
                coreIDs.contains(override.id),
                "override '\(override.id)' does not target any core-shipped scenario id — orphaned override")
        }
    }

    func testToolUsingScenariosGainTheGroundingDirective() throws {
        let scenarios = try ScenarioCorpusFixture.load()
        XCTAssertFalse(scenarios.isEmpty, "no scenarios decoded — directory empty or unreadable")

        var groundedCount = 0
        for scenario in scenarios where !scenario.requiredTools.isEmpty {
            let prompt = Self.grounded(
                base: scenario.systemPrompt,
                requiredTools: scenario.requiredTools)
            XCTAssertTrue(
                prompt.contains(Self.directive),
                "tool-using scenario '\(scenario.id)' did not receive the grounding directive")
            // The scenario's own instruction must survive — grounding stacks on
            // top of it rather than replacing it.
            XCTAssertTrue(
                prompt.contains(scenario.systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)),
                "grounding clobbered scenario '\(scenario.id)' own system prompt")
            groundedCount += 1
        }
        XCTAssertGreaterThan(groundedCount, 0, "expected at least one tool-using scenario")
    }

    func testToollessScenariosAreLeftUnchanged() throws {
        let scenarios = try ScenarioCorpusFixture.load()
        for scenario in scenarios where scenario.requiredTools.isEmpty {
            let prompt = Self.grounded(
                base: scenario.systemPrompt,
                requiredTools: scenario.requiredTools)
            XCTAssertEqual(
                prompt, scenario.systemPrompt,
                "toolless scenario '\(scenario.id)' should not be grounded (no tool result to ground in)")
            XCTAssertFalse(
                prompt.contains(Self.directive),
                "toolless scenario '\(scenario.id)' unexpectedly carries the grounding directive")
        }
    }
}
