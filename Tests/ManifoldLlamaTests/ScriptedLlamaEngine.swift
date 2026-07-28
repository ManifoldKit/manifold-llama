import Foundation
import LlamaSwift
import ManifoldContract
@_spi(Testing) @testable import ManifoldLlama

/// A fully scripted ``LlamaEngine`` — the test double that makes
/// `LlamaBackend.generate()`'s Task body and `LlamaGenerationDriver`'s
/// decode/sample loop runnable with no `.gguf`, no `llama_context`, and no
/// Metal (#165).
///
/// It answers every C-API question the driver asks from plain Swift values and
/// records what it was asked, so a test can assert on the *shape* of the
/// generation — chunk boundaries, KV-reuse prefix, decode positions,
/// synchronize ordering — not just on the text that came out.
///
/// Token encoding: the engine hands the driver token id `i` for the `i`-th
/// sample, and ``tokenToString(_:invalidUTF8Buffer:)`` maps that id straight
/// back to `script[i]`. Once the script is exhausted it returns
/// ``endOfGenerationToken``, which is the only id
/// ``isEndOfGeneration(_:)`` reports true for.
final class ScriptedLlamaEngine: LlamaEngine, @unchecked Sendable {

    /// What ``makeSampler(config:seed:includeGrammar:)`` should do. Lets a test
    /// reach the driver's two sampler-failure branches, which are otherwise
    /// unreachable without provoking a real allocation failure or feeding a
    /// real vocab an invalid GBNF string.
    enum SamplerBehavior: Sendable {
        case succeed
        case chainInitFailed
        case grammarParseFailed
    }

    struct PromptChunk: Equatable, Sendable {
        let tokens: [llama_token]
        let startPosition: Int
        let logitsOnLastToken: Bool
    }

    struct GeneratedDecode: Equatable, Sendable {
        let token: llama_token
        let position: Int
    }

    /// The id handed to the driver once the script runs out.
    static let endOfGenerationToken: llama_token = -99

    // MARK: - Script

    private let script: [String]
    private let samplerBehavior: SamplerBehavior
    /// Status codes returned by successive ``decodePromptChunk(tokens:startPosition:logitsOnLastToken:)``
    /// calls; a short array is padded with `0` (success).
    private let promptDecodeStatuses: [Int32]
    /// Status returned by every ``decodeGeneratedToken(_:position:)`` call.
    private let generatedDecodeStatus: Int32

    let batchSize: Int
    let contextCapacity: Int

    init(
        script: [String] = [],
        batchSize: Int = 2048,
        contextCapacity: Int = 4096,
        samplerBehavior: SamplerBehavior = .succeed,
        promptDecodeStatuses: [Int32] = [],
        generatedDecodeStatus: Int32 = 0
    ) {
        self.script = script
        self.batchSize = batchSize
        self.contextCapacity = contextCapacity
        self.samplerBehavior = samplerBehavior
        self.promptDecodeStatuses = promptDecodeStatuses
        self.generatedDecodeStatus = generatedDecodeStatus
    }

    // MARK: - Recorded interactions

    private let lock = NSLock()
    private var _kvReusePrefixes: [Int] = []
    private var _samplerRequests: [Bool] = []
    private var _freedSamplerIDs: [Int] = []
    private var _promptChunks: [PromptChunk] = []
    private var _generatedDecodes: [GeneratedDecode] = []
    private var _synchronizeCount = 0
    private var _releaseDecodeResourcesCount = 0
    private var _sampleCount = 0
    private var _nextSamplerID = 1

    private func withLock<T>(_ body: () -> T) -> T {
        lock.lock(); defer { lock.unlock() }
        return body()
    }

    /// The `alignedPrefixLength` of every ``applyKVReuse(alignedPrefixLength:)``
    /// call, in order. Exactly one per `run(...)`.
    var kvReusePrefixes: [Int] { withLock { _kvReusePrefixes } }
    /// `includeGrammar` for each chain the driver asked for, in order.
    var samplerRequests: [Bool] { withLock { _samplerRequests } }
    /// Ids of every chain the driver freed. Must match what it was handed.
    var freedSamplerIDs: [Int] { withLock { _freedSamplerIDs } }
    var promptChunks: [PromptChunk] { withLock { _promptChunks } }
    var generatedDecodes: [GeneratedDecode] { withLock { _generatedDecodes } }
    var synchronizeCount: Int { withLock { _synchronizeCount } }
    var releaseDecodeResourcesCount: Int { withLock { _releaseDecodeResourcesCount } }

    // MARK: - LlamaEngine

    func applyKVReuse(alignedPrefixLength: Int) {
        withLock { _kvReusePrefixes.append(alignedPrefixLength) }
    }

    func makeSampler(config: GenerationConfig, seed: UInt32, includeGrammar: Bool) -> LlamaSamplerBuildOutcome {
        withLock { _samplerRequests.append(includeGrammar) }
        switch samplerBehavior {
        case .chainInitFailed:
            return .chainInitFailed
        case .grammarParseFailed:
            // Mirror the real engine: a grammar parse failure only happens on a
            // chain that was asked to carry the grammar.
            return includeGrammar ? .grammarParseFailed : .success(nextHandle())
        case .succeed:
            return .success(nextHandle())
        }
    }

    private func nextHandle() -> LlamaSamplerHandle {
        withLock {
            let id = _nextSamplerID
            _nextSamplerID += 1
            return LlamaSamplerHandle(rawPointer: nil, id: id)
        }
    }

    func freeSampler(_ handle: LlamaSamplerHandle) {
        withLock { _freedSamplerIDs.append(handle.id) }
    }

    func sample(_ handle: LlamaSamplerHandle, logitIndex: Int32) -> llama_token {
        withLock {
            let index = _sampleCount
            _sampleCount += 1
            return index < script.count ? llama_token(index) : Self.endOfGenerationToken
        }
    }

    func isEndOfGeneration(_ token: llama_token) -> Bool {
        token == Self.endOfGenerationToken
    }

    func tokenToString(_ token: llama_token, invalidUTF8Buffer: inout [CChar]) -> String? {
        let index = Int(token)
        guard index >= 0, index < script.count else { return nil }
        return script[index]
    }

    func decodePromptChunk(tokens: ArraySlice<llama_token>, startPosition: Int, logitsOnLastToken: Bool) -> Int32 {
        withLock {
            let ordinal = _promptChunks.count
            _promptChunks.append(PromptChunk(
                tokens: Array(tokens),
                startPosition: startPosition,
                logitsOnLastToken: logitsOnLastToken
            ))
            return ordinal < promptDecodeStatuses.count ? promptDecodeStatuses[ordinal] : 0
        }
    }

    func decodeGeneratedToken(_ token: llama_token, position: Int) -> Int32 {
        withLock {
            _generatedDecodes.append(GeneratedDecode(token: token, position: position))
            return generatedDecodeStatus
        }
    }

    func synchronize() {
        withLock { _synchronizeCount += 1 }
    }

    func releaseDecodeResources() {
        withLock { _releaseDecodeResourcesCount += 1 }
    }
}
