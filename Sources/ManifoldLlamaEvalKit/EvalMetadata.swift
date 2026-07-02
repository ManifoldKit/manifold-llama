import Foundation
import CryptoKit

/// Pure metadata helpers for the raw-prompt eval runner: prompt hashing, quant
/// extraction, the pinned llama.cpp build, and core-commit resolution. Kept free
/// of any model/Metal dependency so they unit-test in CI without a GGUF.
public enum EvalMetadata {

    /// The pinned llama.cpp xcframework build, reported as
    /// `toolingVersions["llama.cpp"]`.
    ///
    /// **Keep in sync with `Package.swift`'s `llama-cpp` binary target URL**
    /// (`vendor-llama-b9859/llama-b9859-slim.xcframework.zip`) and
    /// `docs/LLAMA_CONTRACT.md`. There is no public llama.cpp API that reports
    /// its own build string, so this is sourced from the pin, not queried.
    public static let llamaCppBuild = "b9859"

    /// SHA-256 hex digest of `data` (the exact prompt string bytes).
    public static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// The model identifier reported in the record: the GGUF file name with its
    /// `.gguf` extension stripped (e.g. `Qwen3-0.6B-Q4_K_M`).
    public static func modelIdentifier(fromFileName name: String) -> String {
        (name as NSString).deletingPathExtension
    }

    /// Extracts a quantization tag from a GGUF file name, or `"unknown"`.
    ///
    /// GGUF file names conventionally end with the quant, e.g.
    /// `Qwen3-0.6B-Q4_K_M.gguf` → `Q4_K_M`, `model-IQ3_XXS.gguf` → `IQ3_XXS`,
    /// `model-f16.gguf` → `F16`. The hyphen-delimited segments are scanned from
    /// the end (where the quant lives) for the first one that looks like a quant
    /// tag: `Q<digit>…` / `IQ<digit>…`, or a float type (`F16`/`F32`/`BF16`).
    public static func parseQuant(fromFileName name: String) -> String {
        let stem = (name as NSString).deletingPathExtension
        let floatTags: Set<String> = ["F16", "F32", "BF16", "FP16"]
        for segment in stem.split(separator: "-").reversed() {
            let upper = String(segment).uppercased()
            if floatTags.contains(upper) {
                return upper
            }
            if upper.range(of: #"^I?Q[0-9]"#, options: .regularExpression) != nil {
                return upper
            }
        }
        return "unknown"
    }

    /// Resolves the ManifoldKit core commit by walking up from each start
    /// directory looking for a `Package.resolved`, parsing it, and returning the
    /// ManifoldKit pin's `revision` (preferred) or `version`. Returns `"unknown"`
    /// when no resolved file with a ManifoldKit pin is found.
    ///
    /// Defaults search from the running binary's directory and the current
    /// working directory — for the built `manifold-llama-eval` (in
    /// `.build/<config>/`) and for `swift run`, an ancestor is the package root
    /// that holds `Package.resolved`.
    public static func resolveCoreCommit(
        searchStartDirectories: [URL] = EvalMetadata.defaultCoreCommitSearchDirectories()
    ) -> String {
        let fileManager = FileManager.default
        for start in searchStartDirectories {
            var dir = start.standardizedFileURL
            while true {
                let resolved = dir.appendingPathComponent("Package.resolved")
                if fileManager.fileExists(atPath: resolved.path),
                   let commit = manifoldKitPin(inResolvedFileAt: resolved) {
                    return commit
                }
                let parent = dir.deletingLastPathComponent().standardizedFileURL
                if parent.path == dir.path { break } // reached filesystem root
                dir = parent
            }
        }
        return "unknown"
    }

    /// Default search roots for ``resolveCoreCommit(searchStartDirectories:)``.
    public static func defaultCoreCommitSearchDirectories() -> [URL] {
        var dirs: [URL] = []
        let executable = URL(fileURLWithPath: CommandLine.arguments.first ?? "")
            .resolvingSymlinksInPath()
        dirs.append(executable.deletingLastPathComponent())
        dirs.append(URL(fileURLWithPath: FileManager.default.currentDirectoryPath))
        return dirs
    }

    /// Parses a `Package.resolved` file and returns the ManifoldKit pin's
    /// revision (preferred) or version, or `nil` if absent/unparseable.
    static func manifoldKitPin(inResolvedFileAt url: URL) -> String? {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            return nil
        }
        let parsed: Any
        do {
            parsed = try JSONSerialization.jsonObject(with: data)
        } catch {
            return nil
        }
        guard let root = parsed as? [String: Any],
              let pins = root["pins"] as? [[String: Any]] else {
            return nil
        }
        for pin in pins {
            let identity = (pin["identity"] as? String)?.lowercased() ?? ""
            let location = (pin["location"] as? String)?.lowercased() ?? ""
            let isManifoldKit = identity == "manifoldkit"
                || location.contains("manifoldkit/manifoldkit")
            guard isManifoldKit, let state = pin["state"] as? [String: Any] else { continue }
            if let revision = state["revision"] as? String, !revision.isEmpty {
                return revision
            }
            if let version = state["version"] as? String, !version.isEmpty {
                return version
            }
        }
        return nil
    }
}
