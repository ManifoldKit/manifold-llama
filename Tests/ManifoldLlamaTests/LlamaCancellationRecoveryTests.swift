import ManifoldInference
import ManifoldTestSupport
import XCTest

@_spi(Testing) import ManifoldLlama

final class LlamaCancellationRecoveryTests: XCTestCase {
  /// Live counterpart to the held-engine regressions. Never reuse or free the
  /// context until the cancelled task has released native decode resources.
  func test_cancelJoin_recoverThenUnloadReload() async throws {
    guard let model = HardwareRequirements.findGGUFModel() else {
      throw XCTSkip("Set LLAMA_TEST_MODEL to a cached GGUF for native cancellation/reload coverage")
    }
    let backend = LlamaBackend()
    addTeardownBlock { await backend.unloadAndWait() }
    try await backend.loadModel(from: model, plan: .testStub(effectiveContextSize: 512))
    let config = GenerationConfig(temperature: 0, maxOutputTokens: 64)
    let stream = try backend.generate(
      prompt: "Count the numbers from one to twenty:", systemPrompt: nil, config: config)
    var tokensBeforeStop = 0
    var stoppedWhileGenerating = false
    var cancelledUsage: TokenUsage?
    for try await event in stream.events {
      if case .token = event {
        tokensBeforeStop += 1
        if tokensBeforeStop == 1 {
          stoppedWhileGenerating = backend.isGenerating
          backend.stopGeneration()
        }
      }
      if case .usage(let usage) = event { cancelledUsage = usage }
    }
    await backend.awaitGenerationSettled()
    XCTAssertGreaterThan(tokensBeforeStop, 0, "fixture must cancel an active decode")
    XCTAssertTrue(stoppedWhileGenerating)
    XCTAssertNil(cancelledUsage, "fixture must interrupt a turn before completed-turn usage")
    XCTAssertFalse(backend.isGenerating)
    try await assertRecovery(backend)
    await backend.unloadAndWait()
    XCTAssertFalse(backend.isModelLoaded)
    try await backend.loadModel(from: model, plan: .testStub(effectiveContextSize: 512))
    try await assertRecovery(backend)
  }

  private func assertRecovery(_ backend: LlamaBackend) async throws {
    let stream = try backend.generate(
      prompt: "The capital of France is", systemPrompt: nil,
      config: .init(temperature: 0, maxOutputTokens: 8))
    var usage: TokenUsage?
    for try await event in stream.events {
      if case .usage(let value) = event { usage = value }
    }
    await backend.awaitGenerationSettled()
    XCTAssertFalse(backend.isGenerating)
    XCTAssertGreaterThan(try XCTUnwrap(usage).completionTokens, 0)
  }
}
