import Foundation
import ManifoldTools

/// Test-only mirror of `loadScenarios()` in the `manifold-tools-llama`
/// executable's `main.swift`.
///
/// SwiftPM does not support `@testable import` of an executable target, so
/// `LlamaResultGroundingPromptTests` and `LlamaStructuredJsonGrammarTests`
/// cannot import the production `loadScenarios()` directly. This composes the
/// same two calls the harness does — ManifoldKit core's bundled `built-in`
/// corpus (`ScenarioLoader.loadBuiltIn()`) spliced with this package's four
/// llama/gemma-tolerant overrides by id — reading `ScenarioOverrides` from the
/// source tree (relative to this file) rather than a resource bundle, since
/// the test target has no bundled copy of it. Keep this in lockstep with
/// `loadScenarios()` in `Sources/manifold-tools-llama/main.swift`.
enum ScenarioCorpusFixture {

    /// Resolves `Sources/manifold-tools-llama/ScenarioOverrides` from this
    /// file's location so the suite does not depend on the executable's
    /// resource bundle.
    static func overridesDirectory() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // ManifoldLlamaTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // package root
            .appendingPathComponent("Sources/manifold-tools-llama/ScenarioOverrides", isDirectory: true)
    }

    /// Core's bundled `built-in` scenarios with this package's overrides
    /// spliced in by id — the same corpus `manifold-tools-llama` runs against
    /// a real model.
    static func load() throws -> [Scenario] {
        let base = try ScenarioLoader.loadBuiltIn()
        let overrides = try ScenarioLoader.load(from: overridesDirectory())
        let overridesByID = Dictionary(uniqueKeysWithValues: overrides.map { ($0.id, $0) })
        return base.map { overridesByID[$0.id] ?? $0 }
    }
}
