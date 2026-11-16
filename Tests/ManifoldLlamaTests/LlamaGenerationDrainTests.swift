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

  func test_immediateResend_waitsForStoppedDecodeWithoutReusingAnIncompletePrompt() async throws {
    let backend = LlamaBackend()
    let stopped = HeldPrefillEngine(batchSize: 1)
    let replacement = HeldPrefillEngine(reply: "replacement")
    let engineStarts = OSAllocatedUnfairLock(initialState: 0)
    backend.armFakeLoadedStateForTesting()
    defer {
      stopped.release.signal()
      replacement.release.signal()
      backend.disarmFakeLoadedStateForTesting()
    }
    backend.installGenerationSeamForTesting(.init(tokenize: { _, _ in [1, 2] }, makeEngine: { _, _ in
      engineStarts.withLock { count in
        count += 1
        return count == 1 ? stopped : replacement
      }
    }))

    let first = try backend.generate(prompt: "first", systemPrompt: nil, config: .init(maxOutputTokens: 1))
    let originalEntered = await signalArrived(stopped.entered, seconds: 3)
    XCTAssertTrue(originalEntered)
    let firstConsumer = Task { for try await _ in first.events {} }
    backend.stopGeneration()
    XCTAssertFalse(backend.isGenerating, "Stop should release admission while decode is still held")
    let next = try backend.generate(prompt: "second", systemPrompt: nil, config: .init(maxOutputTokens: 1))
    firstConsumer.cancel()

    let prematureDecode = await signalArrived(replacement.entered, seconds: 0.1)
    XCTAssertFalse(prematureDecode, "replacement touched C-context work before its predecessor released it")
    XCTAssertEqual(engineStarts.withLock { $0 }, 1)
    stopped.release.signal()
    let replacementEntered = await signalArrived(replacement.entered, seconds: 3)
    XCTAssertTrue(replacementEntered)
    XCTAssertTrue(stopped.resourcesReleased.withLock { $0 })
    XCTAssertTrue(backend.isGenerating, "old stream teardown must not clear the replacement's state")

    replacement.release.signal()
    var text = ""
    for try await event in next.events {
      if case .token(let chunk) = event { text += chunk }
      if case .kvCacheReuse = event { XCTFail("cancelled partial prefill was incorrectly reused") }
    }
    try await firstConsumer.value
    await backend.awaitGenerationSettled()
    XCTAssertEqual(text, "replacement")
    XCTAssertFalse(backend.isGenerating)
  }

  func test_repeatedStop_skipsWaitingReplacementAndOrdersWipeBeforeNextDecode() async throws {
    let backend = LlamaBackend()
    let stopped = HeldPrefillEngine()
    let last = HeldPrefillEngine(reply: "after wipe")
    let steps = OSAllocatedUnfairLock(initialState: [String]())
    backend.armFakeLoadedStateForTesting()
    defer {
      stopped.release.signal()
      last.release.signal()
      backend.disarmFakeLoadedStateForTesting()
    }
    backend.installGenerationSeamForTesting(.init(
      tokenize: { _, _ in [1] },
      makeEngine: { _, _ in
        steps.withLock { values in
          let isFirst = values.isEmpty
          values.append(isFirst ? "old" : "next")
          return isFirst ? stopped : last
        }
      },
      clearCache: { zeroData in
        XCTAssertTrue(stopped.resourcesReleased.withLock { $0 })
        steps.withLock { $0.append(zeroData ? "wipe" : "reset") }
      }
    ))

    let first = try backend.generate(prompt: "first", systemPrompt: nil, config: .init(maxOutputTokens: 1))
    let originalEntered = await signalArrived(stopped.entered, seconds: 3)
    XCTAssertTrue(originalEntered)
    backend.stopGeneration()
    let cancelled = try backend.generate(prompt: "discard", systemPrompt: nil, config: .init(maxOutputTokens: 1))
    backend.stopGeneration()
    backend.stopGeneration()
    backend.resetConversation()
    backend.secureWipe()
    let next = try backend.generate(prompt: "last", systemPrompt: nil, config: .init(maxOutputTokens: 1))

    let prematureDecode = await signalArrived(last.entered, seconds: 0.1)
    XCTAssertFalse(prematureDecode)
    XCTAssertEqual(steps.withLock { $0 }, ["old"])
    stopped.release.signal()

    let lastEntered = await signalArrived(last.entered, seconds: 3)
    XCTAssertTrue(lastEntered)
    XCTAssertTrue(backend.isGenerating)
    XCTAssertEqual(steps.withLock { $0 }, ["old", "reset", "wipe", "next"])
    last.release.signal()

    for try await _ in first.events {}
    for try await event in cancelled.events {
      XCTFail("cancelled waiting turn unexpectedly produced an event: \(event)")
    }
    var text = ""
    for try await event in next.events {
      if case .token(let chunk) = event { text += chunk }
      if case .kvCacheReuse = event { XCTFail("wiped KV state was incorrectly reused") }
    }
    await backend.awaitGenerationSettled()
    XCTAssertEqual(text, "after wipe")
  }

  func test_unload_joinsStoppedDecodeAndWipeWithReplacementPending() async throws {
    let backend = LlamaBackend()
    let stopped = HeldPrefillEngine()
    let steps = OSAllocatedUnfairLock(initialState: [String]())
    backend.armFakeLoadedStateForTesting()
    defer {
      stopped.release.signal()
      backend.disarmFakeLoadedStateForTesting()
    }
    backend.installGenerationSeamForTesting(.init(
      tokenize: { _, _ in [1] },
      makeEngine: { _, _ in
        steps.withLock { $0.append("decode") }
        return stopped
      },
      clearCache: { zeroData in
        XCTAssertTrue(zeroData)
        XCTAssertTrue(stopped.resourcesReleased.withLock { $0 })
        steps.withLock { $0.append("wipe") }
      }
    ))

    let first = try backend.generate(prompt: "first", systemPrompt: nil, config: .init(maxOutputTokens: 1))
    let originalEntered = await signalArrived(stopped.entered, seconds: 3)
    XCTAssertTrue(originalEntered)
    backend.stopGeneration()
    backend.secureWipe()
    let pending = try backend.generate(prompt: "pending", systemPrompt: nil, config: .init(maxOutputTokens: 1))
    backend.disarmFakeLoadedStateForTesting()

    let joined = DispatchSemaphore(value: 0)
    let unload = Task {
      await backend.unloadAndWait()
      joined.signal()
    }
    let returnedEarly = await signalArrived(joined, seconds: 0.1)
    XCTAssertFalse(returnedEarly, "unload cannot finish ahead of the deferred wipe")
    XCTAssertEqual(steps.withLock { $0 }, ["decode"])
    stopped.release.signal()
    await unload.value
    for try await _ in first.events {}
    for try await event in pending.events {
      XCTFail("unloaded waiting turn unexpectedly produced an event: \(event)")
    }
    XCTAssertEqual(steps.withLock { $0 }, ["decode", "wipe"])
    XCTAssertFalse(backend.isGenerating)
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
  let batchSize: Int
  let contextCapacity = 4096
  let reply: String?

  init(batchSize: Int = 128, reply: String? = nil) {
    self.batchSize = batchSize
    self.reply = reply
  }
  let entered = DispatchSemaphore(value: 0)
  let release = DispatchSemaphore(value: 0)
  let resourcesReleased = OSAllocatedUnfairLock(initialState: false)
  func applyKVReuse(alignedPrefixLength: Int) {}
  func makeSampler(config: GenerationConfig, seed: UInt32, includeGrammar: Bool)
    -> LlamaSamplerBuildOutcome { .success(.init(rawPointer: nil)) }
  func freeSampler(_ handle: LlamaSamplerHandle) {}
  func sample(_ handle: LlamaSamplerHandle, logitIndex: Int32) -> Int32 { reply == nil ? -1 : 42 }
  func isEndOfGeneration(_ token: Int32) -> Bool { token == -1 }
  func tokenToString(_ token: Int32, invalidUTF8Buffer: inout [CChar]) -> String? { reply }
  func decodePromptChunk(tokens: ArraySlice<Int32>, startPosition: Int,
                         logitsOnLastToken: Bool) -> Int32 {
    entered.signal()
    return release.wait(timeout: .now() + 10) == .success ? 0 : -1
  }
  func decodeGeneratedToken(_ token: Int32, position: Int) -> Int32 { 0 }
  func synchronize() {}
  func releaseDecodeResources() { resourcesReleased.withLock { $0 = true } }
}
