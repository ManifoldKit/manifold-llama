import Foundation

/// One raw-prompt generation record, the fixed wire contract the manifold-eval
/// differential harness consumes from the llama.cpp leg.
///
/// **The shape is a fixed contract — do not redesign.** manifold-eval shells the
/// `manifold-llama-eval` runner and parses exactly one `RawRun` JSON object off
/// stdout, then diffs it against the same record produced by the Ollama leg for
/// the *same GGUF*. The field names (including the dotted `"llama.cpp"` tooling
/// key) and nesting are part of that contract.
///
/// See ManifoldKit `docs/plans/manifold-eval-repo-v2-override.md` §13b.
public struct RawRun: Codable, Equatable, Sendable {
    /// Always `"llama.cpp"` for this leg.
    public let backend: String
    /// The model identifier (the GGUF file's name stem, e.g. `Qwen3-0.6B-Q4_K_M`).
    public let model: String
    /// The quantization tag parsed from the file name (e.g. `Q4_K_M`), or
    /// `"unknown"` when no recognizable tag is present.
    public let quant: String
    /// SHA-256 hex digest of the exact prompt **string bytes** fed to the
    /// tokenizer (the verbatim `--prompt-file` contents — no trimming, no
    /// re-templating).
    public let promptSha256: String
    /// The token ids llama.cpp tokenizes the prompt into, with BOS included
    /// (the backend adds it). This is the exact sequence generation decodes from.
    public let inputTokenIds: [Int]
    /// The generated visible text.
    public let output: String
    /// The generated token ids. May be `[]`: the streaming backend surfaces
    /// decoded text deltas, not the underlying generated ids, so they are not
    /// readily available without re-tokenizing (which would not reproduce the
    /// actual sampled ids). Left empty rather than reporting a misleading
    /// re-tokenization — the contract explicitly permits `[]`.
    public let outputTokenIds: [Int]
    /// The sampler settings actually used for this run.
    public let sampler: Sampler
    /// The ManifoldKit core commit/version this runner was built against (the
    /// resolved `Package.resolved` pin revision; falls back to the version
    /// string, then `"unknown"`).
    public let coreCommit: String
    /// External tooling versions. Carries the pinned llama.cpp xcframework build.
    public let toolingVersions: ToolingVersions
    /// Which repeat this record is, within a multi-run sweep (0-based).
    public let repeatIndex: Int

    public init(
        backend: String,
        model: String,
        quant: String,
        promptSha256: String,
        inputTokenIds: [Int],
        output: String,
        outputTokenIds: [Int],
        sampler: Sampler,
        coreCommit: String,
        toolingVersions: ToolingVersions,
        repeatIndex: Int
    ) {
        self.backend = backend
        self.model = model
        self.quant = quant
        self.promptSha256 = promptSha256
        self.inputTokenIds = inputTokenIds
        self.output = output
        self.outputTokenIds = outputTokenIds
        self.sampler = sampler
        self.coreCommit = coreCommit
        self.toolingVersions = toolingVersions
        self.repeatIndex = repeatIndex
    }

    /// Sampler settings as carried in the contract.
    public struct Sampler: Codable, Equatable, Sendable {
        public let temperature: Double
        public let seed: Int
        public let topK: Int
        public let repeatPenalty: Double
        public let maxTokens: Int

        public init(
            temperature: Double,
            seed: Int,
            topK: Int,
            repeatPenalty: Double,
            maxTokens: Int
        ) {
            self.temperature = temperature
            self.seed = seed
            self.topK = topK
            self.repeatPenalty = repeatPenalty
            self.maxTokens = maxTokens
        }
    }

    /// External tooling versions. The `llama.cpp` key is dotted, so it needs an
    /// explicit `CodingKey` (Swift identifiers cannot contain `.`).
    public struct ToolingVersions: Codable, Equatable, Sendable {
        public let llamaCpp: String

        public init(llamaCpp: String) {
            self.llamaCpp = llamaCpp
        }

        enum CodingKeys: String, CodingKey {
            case llamaCpp = "llama.cpp"
        }
    }

    /// Encodes the record as a single-line JSON object (stable key order) — the
    /// exact form the runner writes to stdout for manifold-eval to parse.
    public func encodedJSONLine() throws -> String {
        let encoder = JSONEncoder()
        // Sorted keys keep stdout byte-stable across runs; no pretty-printing so
        // the whole record is one line (one JSON object per invocation).
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(self)
        return String(decoding: data, as: UTF8.self)
    }
}
