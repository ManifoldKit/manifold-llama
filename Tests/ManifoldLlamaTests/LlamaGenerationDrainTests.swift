import Foundation
import ManifoldInference
import XCTest
import os

@_spi(Testing) import ManifoldLlama

/// A stopped stream is not proof that the engine has returned from native work.
/// Hold prefill until the test releases it, then verify every join owns that work.
final class LlamaGenerationDrainTests: XCTestCase {
  func test_settleWithoutStop_waitsForEngine_control() async throws {
    try await checkDrain(.settle)
  }

  func test_stopThenSettle_waitsForEngine() async throws {
    try await checkDrain(.stop)
  }

  func test_memoryWarningThenSettle_waitsForEngine() async throws {
    try await checkDrain(.warning)
  }

  func test_stopThenConcurrentUnloadWaiters_bothWaitForEngine() async throws {
    try await checkDrain(.unload)
  }

  func test_memoryCriticalThenUnload_waitsForEngine() async throws {
    try await checkDrain(.critical)
  }

  func test_settleAfterUnload_joinsTransferredGeneration() async throws {
    try await checkDrain(.settleAfterUnload)
  }

  func test_prefillTokenization_holdsModelOwnershipUntilLookupReturns() async throws {
    try await checkVocabularyLookup(inputTokenIds: false)
  }

  func test_evalInputTokenIds_holdsModelOwnershipUntilLookupReturns() async throws {
    try await checkVocabularyLookup(inputTokenIds: true)
  }

  private func checkVocabularyLookup(inputTokenIds: Bool) async throws {
    let backend = LlamaBackend()
    let entered = DispatchSemaphore(value: 0)
    let release = DispatchSemaphore(value: 0)
    let detached = DispatchSemaphore(value: 0)
    backend.armFakeLoadedStateForTesting()
    defer { backend.disarmFakeLoadedStateForTesting() }
    backend.installGenerationSeamForTesting(.init(tokenize: { _, _ in
      entered.signal()
      _ = waitForSignal(release, seconds: 10)
      return [1]
    }, makeEngine: { _, _ in ScriptedLlamaEngine(script: []) }))
    let generation = Task.detached { () throws -> GenerationStream? in
      if inputTokenIds {
        XCTAssertEqual(backend.inputTokenIds(forPrompt: "held lookup"), [1])
        return nil
      }
      return try backend.generate(
        prompt: "held lookup", systemPrompt: nil, config: .init(maxOutputTokens: 1))
    }
    let lookupEntered = await signalArrived(entered, seconds: 3)
    XCTAssertTrue(lookupEntered)
    let unload = Task.detached {
      // This takes the same state lock as unload's pointer detach, but removes
      // sentinel pointers without passing address 1 to native free.
      backend.disarmFakeLoadedStateForTesting()
      detached.signal()
      await backend.unloadAndWait()
    }
    let detachedEarly = await signalArrived(detached, seconds: 0.1)
    XCTAssertFalse(detachedEarly, "model ownership detached during vocabulary lookup")
    release.signal()
    if let stream = try await generation.value {
      for try await _ in stream.events {}
    }
    await unload.value
    await backend.awaitGenerationSettled()
  }

  private enum Operation { case settle, stop, warning, unload, settleAfterUnload, critical }

  private func checkDrain(_ operation: Operation) async throws {
    let backend = LlamaBackend()
    let engine = HeldPrefillEngine()
    backend.armFakeLoadedStateForTesting()
    defer {
      engine.release.signal()
      backend.disarmFakeLoadedStateForTesting()
    }
    backend.installGenerationSeamForTesting(
      .init(tokenize: { _, _ in [1] }, makeEngine: { _, _ in engine }))
    let stream = try backend.generate(
      prompt: "held", systemPrompt: nil, config: .init(maxOutputTokens: 1))
    let entered = await signalArrived(engine.entered, seconds: 3)
    XCTAssertTrue(entered, "control must reach the held engine before cancellation")
    switch operation {
    case .settle: break
    case .warning: backend.simulateMemoryPressure(.warning)
    case .stop, .unload, .settleAfterUnload, .critical: backend.stopGeneration()
    }
    if operation == .unload || operation == .settleAfterUnload || operation == .critical {
      // The engine owns no native allocations. Remove sentinel pointers before
      // unloading, preserving the real generation-task/cleanup ownership path.
      backend.disarmFakeLoadedStateForTesting()
      if operation == .critical { backend.simulateMemoryPressure(.critical) }
      backend.unloadModel()
    }
    let joined = DispatchSemaphore(value: 0)
    let first = Task {
      if operation == .unload || operation == .critical { await backend.unloadAndWait() }
      else { await backend.awaitGenerationSettled() }
      let released = engine.resourcesReleased.withLock { $0 }
      joined.signal()
      return released
    }
    let second = Task {
      if operation == .unload || operation == .critical { await backend.unloadAndWait() }
      else { await backend.awaitGenerationSettled() }
      let released = engine.resourcesReleased.withLock { $0 }
      joined.signal()
      return released
    }
    let returnedEarly = await signalArrived(joined, seconds: 0.1)
    XCTAssertFalse(returnedEarly, "join returned while the engine was held")
    XCTAssertFalse(engine.resourcesReleased.withLock { $0 })
    engine.release.signal()
    for try await _ in stream.events {}
    let firstDrained = await first.value
    let secondDrained = await second.value
    XCTAssertTrue(firstDrained, "first waiter must observe native resource release")
    XCTAssertTrue(secondDrained, "one waiter must not consume the other's task handle")
    XCTAssertFalse(backend.isGenerating)
    await backend.awaitGenerationSettled()
  }
}

private func signalArrived(_ signal: DispatchSemaphore, seconds: Double) async -> Bool {
  await Task.detached { waitForSignal(signal, seconds: seconds) }.value
}

private func waitForSignal(_ signal: DispatchSemaphore, seconds: Double) -> Bool {
  signal.wait(timeout: .now() + seconds) == .success
}

private final class HeldPrefillEngine: LlamaEngine, Sendable {
  let batchSize = 128
  let contextCapacity = 4096
  let entered = DispatchSemaphore(value: 0)
  let release = DispatchSemaphore(value: 0)
  let resourcesReleased = OSAllocatedUnfairLock(initialState: false)
  func applyKVReuse(alignedPrefixLength: Int) {}
  func makeSampler(config: GenerationConfig, seed: UInt32, includeGrammar: Bool)
    -> LlamaSamplerBuildOutcome { .success(.init(rawPointer: nil)) }
  func freeSampler(_ handle: LlamaSamplerHandle) {}
  func sample(_ handle: LlamaSamplerHandle, logitIndex: Int32) -> Int32 { -1 }
  func isEndOfGeneration(_ token: Int32) -> Bool { true }
  func tokenToString(_ token: Int32, invalidUTF8Buffer: inout [CChar]) -> String? { nil }
  func decodePromptChunk(tokens: ArraySlice<Int32>, startPosition: Int,
                         logitsOnLastToken: Bool) -> Int32 {
    entered.signal()
    return release.wait(timeout: .now() + 10) == .success ? 0 : -1
  }
  func decodeGeneratedToken(_ token: Int32, position: Int) -> Int32 { 0 }
  func synchronize() {}
  func releaseDecodeResources() { resourcesReleased.withLock { $0 = true } }
}
