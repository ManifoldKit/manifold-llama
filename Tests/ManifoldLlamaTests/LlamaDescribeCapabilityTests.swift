import XCTest
import ManifoldTestSupport
import ManifoldInference
import ManifoldModelCatalog

/// Integration coverage for the pipeline behind `manifold-tools-llama --describe`
/// (issue #2005 layers 1+2): `ModelInfo.load` → `ChatTemplateToolDescriptor`
/// (static dialect/negative-gate) → `RenderConsistencyChecker` (static
/// render-consistency). These are MK 0.59 types; this test pins the *wiring*
/// (metadata → descriptor → consistency) against a real GGUF so an upstream API
/// change or a metadata-read regression surfaces here.
///
/// Model-gated: reads GGUF metadata only — no weights, no Metal, no generation —
/// so it would run even in the simulator, but it still needs a `.gguf` on disk.
/// Skips cleanly when none is found (CI has no model). Assertions are
/// dialect-agnostic invariants, since `findGGUFModel()` may return any model.
final class LlamaDescribeCapabilityTests: XCTestCase {

    func test_describePipeline_realModel_invariantsHold() throws {
        guard let modelURL = HardwareRequirements.findGGUFModel() else {
            throw XCTSkip(
                "No GGUF model found on disk. Set LLAMA_TEST_MODEL=<path> or place a `.gguf` file in ~/Documents/Models/ to run this test."
            )
        }

        // Metadata read must succeed (no weights mapped).
        let modelInfo = try ModelInfo.load(ggufURL: modelURL)

        let descriptor = ChatTemplateToolDescriptor(parsingChatTemplate: modelInfo.chatTemplateRaw)
        let consistency = RenderConsistencyChecker.check(chatTemplateRaw: modelInfo.chatTemplateRaw)

        // --- Layer-1 invariants (dialect-agnostic) ---

        // toolless is exactly the not-expressible case, and vice-versa: the
        // parser only emits `.toolless` when no tools guard is present, which is
        // also the only path that sets toolsExpressible == false.
        XCTAssertEqual(
            descriptor.toolsExpressible,
            descriptor.extractability != .toolless,
            "toolsExpressible must agree with extractability != .toolless"
        )

        // A decoded dialect is only ever reported for an expressible template.
        if descriptor.declaredDialect != nil {
            XCTAssertTrue(
                descriptor.toolsExpressible,
                "A declaredDialect implies the template expresses tools"
            )
        }

        // --- Layer-2 invariant: nothing to round-trip when not expressible ---
        if !descriptor.toolsExpressible {
            XCTAssertEqual(
                consistency.status, .notApplicable,
                "A template that cannot express tools has no dialect to render-check"
            )
            XCTAssertFalse(
                consistency.toolDefinitionRendered,
                "A toolless template cannot render a tool definition"
            )
        }
    }
}
