import Foundation
import ManifoldInference
@_spi(Testing) import ManifoldLlama

/// Inputs for one raw-prompt eval run. These map 1:1 to the runner's CLI flags
/// (manifold-eval invokes `manifold-llama-eval` with exactly these).
public struct EvalOptions: Sendable {
    /// Path to the `.gguf` model file (`--model`).
    public var modelPath: String
    /// Path to the prompt file, read verbatim (`--prompt-file`).
    public var promptFile: String
    /// Sampling temperature (`--temperature`). `0` selects greedy decoding.
    public var temperature: Double
    /// Sampling seed (`--seed`). Negative means "unseeded".
    public var seed: Int
    /// Max tokens to generate (`--max-tokens`).
    public var maxTokens: Int
    /// 0-based index within a repeat sweep (`--repeat-index`).
    public var repeatIndex: Int
    /// Requested context size for the load plan. Not a CLI flag; the planner
    /// clamps it to the model's trained context.
    public var requestedContextSize: Int
    /// Top-k sampling cutoff (`--top-k`). `0` (the default) means "not applied" —
    /// the neutral default that keeps the differential's default mode a clean
    /// function of the model rather than sampler luck. Positive values are
    /// forwarded to `GenerationConfig.topK`; note that at `temperature == 0`
    /// (the only supported diff mode) the driver decodes greedily and top-k has
    /// no effect regardless of this value.
    public var topK: Int
    /// Repetition penalty (`--repeat-penalty`). `1.0` (the default) is
    /// llama.cpp's no-op value — the neutral default. Overriding it lets an
    /// operator force-match samplers across the Ollama/llama.cpp differential
    /// legs when debugging a divergence.
    public var repeatPenalty: Double

    public init(
        modelPath: String,
        promptFile: String,
        temperature: Double,
        seed: Int,
        maxTokens: Int,
        repeatIndex: Int,
        requestedContextSize: Int = 8192,
        topK: Int = 0,
        repeatPenalty: Double = 1.0
    ) {
        self.modelPath = modelPath
        self.promptFile = promptFile
        self.temperature = temperature
        self.seed = seed
        self.maxTokens = maxTokens
        self.repeatIndex = repeatIndex
        self.requestedContextSize = requestedContextSize
        self.topK = topK
        self.repeatPenalty = repeatPenalty
    }
}

/// Errors surfaced by ``EvalRunner``. Each carries an actionable message so the
/// CLI can print it to stderr and exit non-zero without a stack trace.
public enum EvalError: Error, LocalizedError {
    case promptFileUnreadable(path: String, underlying: Error)
    case modelNotFound(path: String)
    case tokenizationFailed
    case emptyOutput

    public var errorDescription: String? {
        switch self {
        case let .promptFileUnreadable(path, underlying):
            return "could not read prompt file at \(path): \(underlying)"
        case let .modelNotFound(path):
            return "model file not found: \(path)"
        case .tokenizationFailed:
            return "prompt tokenized to zero tokens (empty prompt or vocab not loaded)"
        case .emptyOutput:
            return "generation produced no output tokens"
        }
    }
}

/// Drives a single raw-prompt generation through ``LlamaBackend`` and produces a
/// ``RawRun`` record — the llama.cpp leg of the manifold-eval cross-backend
/// differential.
///
/// **Raw prompt, no chat template.** The prompt file's bytes are fed verbatim to
/// `LlamaBackend.generate(prompt:systemPrompt:config:)`, which tokenizes the
/// string directly with `addBos: true` and applies no chat template — exactly
/// the same string + tokenization both the recorded `promptSha256` and
/// `inputTokenIds` describe.
///
/// **Neutral by default, overridable sampler — greedy (temperature 0) is the
/// supported mode.** To keep the cross-backend diff a function of the model
/// (not sampler luck) *by default*, `EvalOptions.repeatPenalty` defaults to
/// `1.0` (llama.cpp's no-op value) and `EvalOptions.topK` defaults to `0`
/// ("not applied"; `temperature == 0` also selects true greedy/argmax decoding
/// in the driver, which short-circuits top-k entirely regardless of the
/// configured value). `topP` stays fixed at `1.0` and is not exposed as a flag.
/// Both `repeatPenalty` and `topK` are overridable (`--repeat-penalty` /
/// `--top-k`) so an operator can force-match samplers across the
/// Ollama/llama.cpp differential legs when debugging a divergence — the
/// recorded `sampler` always reflects the values actually passed to
/// `GenerationConfig`, never a hardcoded stand-in. The seed is plumbed for
/// completeness.
///
/// > Note: `temperature > 0` is NOT a supported deterministic-diff mode. Above 0
/// > the driver applies min-p (0.05) by default, and falls back to a top-k of 40
/// > when `topK` is left at its neutral default — neither is guaranteed to match
/// > the recorded `sampler` when `temperature > 0`; the differential only ever
/// > runs greedy.
public enum EvalRunner {

    public static func run(_ options: EvalOptions) async throws -> RawRun {
        let modelURL = URL(fileURLWithPath: options.modelPath)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw EvalError.modelNotFound(path: modelURL.path)
        }

        // Read the prompt file's raw bytes. The SHA-256 is computed over these
        // exact bytes (the string we feed the tokenizer), before any decoding —
        // no trimming, no re-templating.
        let promptURL = URL(fileURLWithPath: options.promptFile)
        let promptData: Data
        do {
            promptData = try Data(contentsOf: promptURL)
        } catch {
            throw EvalError.promptFileUnreadable(path: promptURL.path, underlying: error)
        }
        let promptString = String(decoding: promptData, as: UTF8.self)
        let promptSha256 = EvalMetadata.sha256Hex(promptData)

        let backend = LlamaBackend()
        do {
            try await backend.loadModel(
                from: modelURL,
                plan: .systemManaged(requestedContextSize: options.requestedContextSize))

            // The exact input token ids generation will decode from (BOS included).
            let inputTokenIds = backend.inputTokenIds(forPrompt: promptString)
            guard !inputTokenIds.isEmpty else {
                throw EvalError.tokenizationFailed
            }

            var config = GenerationConfig(
                temperature: Float(options.temperature),
                topP: 1.0,
                repeatPenalty: Float(options.repeatPenalty),
                topK: options.topK > 0 ? Int32(options.topK) : nil,
                maxOutputTokens: options.maxTokens)
            config.seed = options.seed >= 0 ? UInt64(options.seed) : nil

            var output = ""
            let stream = try backend.generate(
                prompt: promptString, systemPrompt: nil, config: config)
            for try await event in stream.events {
                if case let .token(text) = event {
                    output += text
                }
            }
            // Without awaiting settle the generation task's `defer` (which clears
            // `isGenerating`) has no happens-before with the consumer loop exit —
            // see LlamaSeedDeterminismTests. Harmless here (single run) but keeps
            // the backend in a clean state before teardown.
            await backend.awaitGenerationSettled()

            await backend.unloadAndWait()

            let sampler = RawRun.Sampler(
                temperature: options.temperature,
                seed: options.seed,
                // Record the values actually used, not a hardcoded stand-in —
                // at the neutral defaults this is topK: 0 ("not applied") and
                // repeatPenalty: 1.0 (no-op), unchanged from before this was
                // overridable.
                topK: options.topK,
                repeatPenalty: options.repeatPenalty,
                maxTokens: options.maxTokens)

            return RawRun(
                backend: "llama.cpp",
                model: EvalMetadata.modelIdentifier(fromFileName: modelURL.lastPathComponent),
                quant: EvalMetadata.parseQuant(fromFileName: modelURL.lastPathComponent),
                promptSha256: promptSha256,
                inputTokenIds: inputTokenIds,
                output: output,
                // The streaming backend surfaces decoded text, not generated ids;
                // see RawRun.outputTokenIds. Empty rather than misleading.
                outputTokenIds: [],
                sampler: sampler,
                coreCommit: EvalMetadata.resolveCoreCommit(),
                toolingVersions: RawRun.ToolingVersions(llamaCpp: EvalMetadata.llamaCppBuild),
                repeatIndex: options.repeatIndex)
        } catch {
            // Ensure the C/Metal resources are released before propagating — the
            // backend holds a llama_context/model that must be torn down.
            await backend.unloadAndWait()
            throw error
        }
    }
}
