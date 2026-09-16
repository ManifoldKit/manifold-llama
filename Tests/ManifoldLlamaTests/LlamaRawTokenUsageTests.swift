import ManifoldInference
import XCTest

@_spi(Testing) @testable import ManifoldLlama

final class LlamaRawTokenUsageTests: XCTestCase {
  func test_toolOnly_countsRawSamplesDespiteNoVisibleTokens() async throws {
    try await checkUsage(
      script: ["<tool_call>", #"{"name":"echo","arguments":{"value":"ok"}}"#, "</tool_call>"],
      tools: true, expectedVisible: 0, expectedTools: 1)
  }

  func test_reasoningAndMarkers_countAsCompletionTokens() async throws {
    try await checkUsage(
      script: ["<think>", "reason", "</think>", "answer"],
      thinking: true, expectedVisible: 1)
  }

  func test_bufferedUTF8_countsSamplesWithNoString() async throws {
    try await checkUsage(script: ["partial", "€"], buffered: [0], expectedVisible: 1)
  }

  func test_oneSampleSplitByParser_isNotCountedTwice() async throws {
    try await checkUsage(
      script: [#"before<tool_call>{"name":"echo","arguments":{"value":"ok"}}</tool_call>after"#],
      tools: true, expectedVisible: 2, expectedTools: 1)
  }

  func test_emptyOutput_excludesEndOfGenerationSample() async throws {
    try await checkUsage(script: [], expectedVisible: 0)
  }

  private func checkUsage(script: [String], buffered: Set<Int> = [], tools: Bool = false,
                          thinking: Bool = false, expectedVisible: Int,
                          expectedTools: Int = 0) async throws {
    let backend = LlamaBackend()
    let engine = ScriptedLlamaEngine(script: script, bufferedTokenIndices: buffered)
    backend.armFakeLoadedStateForTesting()
    defer { backend.disarmFakeLoadedStateForTesting() }
    backend.installGenerationSeamForTesting(
      .init(tokenize: { _, _ in [1, 2] }, makeEngine: { _, _ in engine }))
    let sink = RawUsageMetricSink()
    backend.metricSink = sink
    var config = GenerationConfig(maxOutputTokens: 32)
    if tools {
      config.tools = [.init(name: "echo", description: "Echo",
                           parameters: .object(["type": .string("object")]))]
    }
    var hints = GenerationRuntimeHints()
    if thinking { hints.thinkingMarkers = .qwen3 }
    let stream = try backend.generate(prompt: "test", systemPrompt: nil, config: config, hints: hints)
    var visible = 0
    var calls = 0
    var usage: TokenUsage?
    for try await event in stream.events {
      switch event {
      case .token: visible += 1
      case .toolCall: calls += 1
      case .usage(let value): usage = value
      default: break
      }
    }
    await backend.awaitGenerationSettled()
    XCTAssertEqual(visible, expectedVisible, "fixture must exercise the intended parser shape")
    XCTAssertEqual(calls, expectedTools)
    XCTAssertEqual(try XCTUnwrap(usage).completionTokens, script.count)
    XCTAssertEqual(usage?.promptTokens, 2)
    XCTAssertEqual(backend.lastUsage?.completionTokens, script.count)
    let metric = await sink.firstRecord()
    XCTAssertEqual(try XCTUnwrap(metric).completionTokens, script.count)
    if expectedVisible == 0 { XCTAssertEqual(metric?.timeToFirstToken, .zero) }
  }
}

private actor RawUsageMetricSink: InferenceMetricSink {
  private var metric: InferenceMetric?
  func record(_ metric: InferenceMetric) async { self.metric = metric }
  func firstRecord() async -> InferenceMetric? {
    let deadline = ContinuousClock.now + .seconds(3)
    while metric == nil && ContinuousClock.now < deadline { await Task.yield() }
    return metric
  }
}
