import Foundation
import ManifoldInference
@_spi(Testing) import ManifoldLlama
import ManifoldTestSupport
import XCTest

/// File-backed fixtures exercise the production preflight without loading weights.
/// Sabotage-evidence: restoring the unchecked fixed-array skip traps the isolated
/// test runner on UInt64.max and on Int.max / 8 + 1 (conversion and multiplication).
final class LlamaHeaderBoundsTests: XCTestCase {
  func test_fixedArrayRejectsUnrepresentableCount() throws {
    try assertRejected(array(type: 0, count: UInt64.max))
  }

  func test_fixedArrayRejectsByteCountOverflow() throws {
    try assertRejected(array(type: 10, count: UInt64(Int.max / 8) + 1))
  }

  func test_fixedArrayRejectsTruncatedPayload() throws {
    try assertRejected(array(type: 10, count: 8, payload: Data(repeating: 0, count: 7)))
  }

  func test_stringArrayRejectsImpossibleCount() throws {
    try assertRejected(array(type: 8, count: UInt64.max))
  }

  func test_stringArrayRejectsTruncatedString() throws {
    try assertRejected(array(type: 8, count: 1, payload: u64(64) + Data([1])))
  }

  func test_stringsRejectOversizedOrTruncatedLengths() throws {
    for length in [UInt64.max, 65_537, 64] {
      let bytes = header(count: 1) + string("general.architecture") + u32(8) + u64(length)
      try withFile(bytes) { XCTAssertNil(LlamaModelLoader.readArchitectureFromHeader(at: $0)) }
    }
    let truncatedKey = header(count: 1) + u64(UInt64.max) + Data(repeating: 0, count: 12)
    try withFile(truncatedKey) { XCTAssertNil(LlamaModelLoader.readArchitectureFromHeader(at: $0)) }
  }

  func test_rejectsImpossibleMetadataCountAndUnsupportedTypes() throws {
    try withFile(header(count: UInt64.max)) {
      XCTAssertNil(LlamaModelLoader.readArchitectureFromHeader(at: $0))
    }
    try assertRejected(array(type: 9, count: 1)) // nested arrays are not GGUF scalar elements
    try assertRejected(array(type: 13, count: 1))
    try withFile(header(count: 1) + string("general.architecture") + u32(4) + u32(0)) {
      XCTAssertNil(LlamaModelLoader.readArchitectureFromHeader(at: $0))
    }
  }

  func test_validScalarArraysAndEmptyArraysPreserveArchitectureInV2AndV3() throws {
    for version: UInt32 in [2, 3] {
      for (type, width): (UInt32, Int) in [(0, 1), (1, 1), (7, 1), (2, 2), (3, 2),
                                         (4, 4), (5, 4), (6, 4), (10, 8), (11, 8), (12, 8)] {
        for count: UInt64 in [0, 3] {
          let value = array(type: type, count: count, payload: Data(repeating: 0, count: Int(count) * width))
          try withFile(fixture(value: value, version: version)) {
            XCTAssertEqual(LlamaModelLoader.readArchitectureFromHeader(at: $0), "llama")
          }
        }
      }
    }
  }

  func test_validStringsAndStringArraysPreserveArchitecture() throws {
    let values = [u32(8) + string("ignored"), array(type: 8, count: 0),
                  array(type: 8, count: 3, payload: string("") + string("token") + string("é"))]
    for value in values {
      try withFile(fixture(value: value)) {
        XCTAssertEqual(LlamaModelLoader.readArchitectureFromHeader(at: $0), "llama")
      }
    }
  }

  func test_sparseLargeArrayPreservesArchitecture() throws {
    // A large valid tokenizer array must remain skippable without an arbitrary
    // scan cap or an allocation proportional to its payload. The hole is sparse.
    let payloadSize: UInt64 = 128 * 1024 * 1024
    let prefix = header(count: 2) + string("ignored") + array(type: 0, count: payloadSize)
    let url = try writeFile(prefix)
    defer { removeFile(url) }
    let handle = try FileHandle(forWritingTo: url)
    do {
      try handle.seek(toOffset: UInt64(prefix.count) + payloadSize)
      try handle.write(contentsOf: string("general.architecture") + u32(8) + string("llama"))
      try handle.close()
    } catch {
      do { try handle.close() }
      catch { XCTFail("Could not close sparse GGUF fixture: \(error)") }
      throw error
    }
    XCTAssertEqual(LlamaModelLoader.readArchitectureFromHeader(at: url), "llama")
  }

  func test_publicLoadRejectsHeaderArchitectureBeforeNativeModelLoad() async throws {
    let bytes = fixture(value: array(type: 10, count: 3, payload: Data(repeating: 0, count: 24)),
                        architecture: "clip")
    let url = try writeFile(bytes)
    defer { removeFile(url) }
    let backend = LlamaBackend()
    addTeardownBlock { await backend.unloadAndWait() }
    do {
      try await backend.loadModel(from: url, plan: .testStub(effectiveContextSize: 512))
      XCTFail("A header-only CLIP fixture must fail architecture preflight")
    } catch InferenceError.unsupportedModelArchitecture(let architecture) {
      XCTAssertEqual(architecture, "clip")
      XCTAssertFalse(backend.isModelLoaded)
    }
  }

  private func assertRejected(_ value: Data, file: StaticString = #filePath, line: UInt = #line) throws {
    try withFile(fixture(value: value)) {
      XCTAssertNil(LlamaModelLoader.readArchitectureFromHeader(at: $0), file: file, line: line)
    }
  }

  private func fixture(value: Data, version: UInt32 = 3, architecture: String = "llama") -> Data {
    header(count: 2, version: version) + string("ignored") + value
      + string("general.architecture") + u32(8) + string(architecture)
  }

  private func header(count: UInt64, version: UInt32 = 3) -> Data {
    Data([0x47, 0x47, 0x55, 0x46]) + u32(version) + u64(0) + u64(count)
  }

  private func array(type: UInt32, count: UInt64, payload: Data = Data()) -> Data {
    u32(9) + u32(type) + u64(count) + payload
  }

  private func u32(_ value: UInt32) -> Data {
    withUnsafeBytes(of: value.littleEndian) { Data($0) }
  }

  private func u64(_ value: UInt64) -> Data {
    withUnsafeBytes(of: value.littleEndian) { Data($0) }
  }

  private func string(_ value: String) -> Data {
    let bytes = Data(value.utf8)
    return u64(UInt64(bytes.count)) + bytes
  }

  private func writeFile(_ bytes: Data) throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("header-bounds-\(UUID()).gguf")
    try bytes.write(to: url)
    return url
  }

  private func withFile(_ bytes: Data, body: (URL) throws -> Void) throws {
    let url = try writeFile(bytes)
    defer { removeFile(url) }
    try body(url)
  }

  private func removeFile(_ url: URL) {
    do { try FileManager.default.removeItem(at: url) }
    catch { XCTFail("Could not remove GGUF fixture: \(error)") }
  }
}
