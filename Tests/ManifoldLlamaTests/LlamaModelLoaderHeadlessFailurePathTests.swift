import XCTest
import ManifoldInference
@_spi(Testing) import ManifoldLlama

/// Headless (no-GGUF, no `llama_*` call) coverage of two `LlamaModelLoader`
/// failure-path arms previously reachable only through a real/failing model
/// load (issue #28):
///
///  1. The progress-callback `@convention(c)` ABI bridge — the
///     `passRetained`/`fromOpaque`/`release` round-trip whose retain/release
///     balance was latently untested.
///  2. The architecture-denylist *wired throw* in `initializeModel` — the
///     metadata→predicate→`throw .unsupportedModelArchitecture` wiring (the
///     predicate and denylist set are already covered by
///     `LlamaArchitecturePreflightTests`).
///
/// Deferred to the model-bearing lane (need a real/failing `llama_*` call):
/// context-creation-nil throw (code -2) and KV-quant `.f16`/`.q4` ctxParams
/// mapping arms. See the PR body for the checklist.
final class LlamaModelLoaderHeadlessFailurePathTests: XCTestCase {

    // MARK: - Progress-callback ABI round-trip

    /// The boxed context must survive the opaque-pointer round-trip: what goes
    /// in via `passRetained`/`toOpaque` must come back out, identical, via
    /// `fromOpaque`/`takeUnretainedValue`.
    ///
    /// Sabotage check: change `progressContext(fromOpaque:)` to construct a new
    /// `ProgressCallbackContext` instead of recovering the boxed one and the
    /// identity assertion fails.
    func test_progressCallbackBox_survivesOpaqueRoundTrip() {
        let context = LlamaModelLoader.ProgressCallbackContext { _ in }

        let recovered: LlamaModelLoader.ProgressCallbackContext =
            LlamaModelLoader.withProgressCallbackBox(context) { opaque in
                LlamaModelLoader.progressContext(fromOpaque: opaque)
            }

        XCTAssertTrue(recovered === context,
                      "The boxed context must be the same object after the opaque-pointer round-trip")
    }

    /// The recovered handler must still be callable — proving the box kept the
    /// closure (and its captured state) alive across the C boundary.
    func test_progressCallbackBox_recoveredHandlerStillFires() async {
        let recorder = ProgressRecorder()
        let context = LlamaModelLoader.ProgressCallbackContext { value in
            await recorder.record(value)
        }

        let handler: @Sendable (Double) async -> Void =
            LlamaModelLoader.withProgressCallbackBox(context) { opaque in
                LlamaModelLoader.progressContext(fromOpaque: opaque).handler
            }

        await handler(0.5)
        let recorded = await recorder.values
        XCTAssertEqual(recorded, [0.5],
                       "The handler recovered through the ABI bridge must fire with the forwarded progress value")
    }

    /// The retain/release contract must be balanced: exactly one `passRetained`
    /// balanced by exactly one `release`. We prove there is no leak (the box is
    /// eventually deallocated) and no over-release (no crash / use-after-free
    /// while `body` holds the opaque pointer).
    ///
    /// A retained-but-never-released box would keep `weakBox` alive past the
    /// helper; an over-released box would crash inside the helper. The middle
    /// ground — exactly-once — is the only way this passes.
    ///
    /// Sabotage check: add a second `ref.release()` and the process traps
    /// (over-release); drop the `defer { ref.release() }` and the
    /// `XCTAssertNil` below fails (leak).
    func test_progressCallbackBox_retainReleaseIsBalanced() {
        weak var weakBox: LlamaModelLoader.ProgressCallbackContext?

        do {
            let context = LlamaModelLoader.ProgressCallbackContext { _ in }
            weakBox = context

            // While body runs, the box is alive via both the strong `context`
            // local and the retained Unmanaged reference. Touching the
            // recovered object here would crash on an over-release.
            let identityOK = LlamaModelLoader.withProgressCallbackBox(context) { opaque in
                LlamaModelLoader.progressContext(fromOpaque: opaque) === context
            }
            XCTAssertTrue(identityOK)

            // `context` is still alive here (the helper released only its own
            // retain), so the box must still exist.
            XCTAssertNotNil(weakBox,
                            "The helper must release only its own retain — the caller's strong reference keeps the box alive")
        }

        // Strong reference gone + helper's retain released exactly once ⇒ the
        // box must now be deallocated. A leaked extra retain would keep it.
        XCTAssertNil(weakBox,
                     "After the caller's reference drops, the box must deallocate — proving no leaked retain from the ABI bridge")
    }

    // MARK: - Architecture-denylist wired throw

    /// The wired preflight throw must fire for a denylisted architecture — this
    /// is the `initializeModel` wiring (metadata→predicate→throw), exercised
    /// without loading a GGUF.
    ///
    /// Sabotage check: change `preflightArchitecture` to `return` unconditionally
    /// and this assertion fails (no throw).
    func test_preflightArchitecture_throwsForDenylistedArchitecture() {
        XCTAssertThrowsError(try LlamaModelLoader.preflightArchitecture("clip")) { error in
            guard case InferenceError.unsupportedModelArchitecture(let arch) = error else {
                return XCTFail("Expected .unsupportedModelArchitecture, got \(error)")
            }
            XCTAssertEqual(arch, "clip",
                           "The thrown error must carry the offending architecture string")
        }
    }

    /// Casing must not let a denylisted architecture slip past the wired throw.
    func test_preflightArchitecture_throwsCaseInsensitively() {
        XCTAssertThrowsError(try LlamaModelLoader.preflightArchitecture("CLIP"))
        XCTAssertThrowsError(try LlamaModelLoader.preflightArchitecture("Bert"))
    }

    /// A legitimate causal-LM architecture must pass the wired throw.
    ///
    /// Sabotage check: add `"llama"` to the denylist and this throws / fails.
    func test_preflightArchitecture_passesForCausalLM() {
        XCTAssertNoThrow(try LlamaModelLoader.preflightArchitecture("llama"))
        XCTAssertNoThrow(try LlamaModelLoader.preflightArchitecture("qwen3"))
    }

    /// A `nil` architecture (key absent from GGUF metadata) must be treated as
    /// "unknown, assume supported" — the wired throw must NOT fire, so exotic
    /// but legitimate LM GGUFs without the key still load.
    ///
    /// Sabotage check: make `preflightArchitecture` throw on `nil` and this
    /// fails.
    func test_preflightArchitecture_passesForMissingMetadata() {
        XCTAssertNoThrow(try LlamaModelLoader.preflightArchitecture(nil))
    }
}

/// Actor recorder so the async progress handler's side effect can be asserted
/// without a `@unchecked Sendable` mutable box.
private actor ProgressRecorder {
    private(set) var values: [Double] = []
    func record(_ value: Double) { values.append(value) }
}
