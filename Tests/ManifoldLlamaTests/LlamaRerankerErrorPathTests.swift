import XCTest
import ManifoldInference
import ManifoldLlamaKit
@_spi(Testing) import ManifoldLlamaKit

/// Error-path and boundary-condition tests for ``LlamaReranker``.
///
/// All tests here are model-free and run without any GGUF present. They
/// exercise the load-failure contract, `isReady` state transitions, and
/// candidate-list boundary conditions that the existing happy-path suite
/// does not cover.
final class LlamaRerankerErrorPathTests: XCTestCase {

    // MARK: - Load failure

    func test_loadModel_nonexistentPath_throwsModelLoadFailed() async {
        let reranker = LlamaReranker()
        let badURL = URL(filePath: "/nonexistent/path/reranker.gguf")
        do {
            try await reranker.loadModel(from: badURL)
            XCTFail("loadModel must throw for a nonexistent path")
        } catch let error as RerankerError {
            guard case .modelLoadFailed = error else {
                XCTFail("Expected RerankerError.modelLoadFailed, got \(error)")
                return
            }
        } catch {
            XCTFail("Expected RerankerError, got \(type(of: error)): \(error)")
        }
    }

    func test_isReady_remainsFalse_afterFailedLoad() async {
        let reranker = LlamaReranker()
        _ = try? await reranker.loadModel(from: URL(filePath: "/nonexistent/reranker.gguf"))
        XCTAssertFalse(reranker.isReady,
                       "isReady must stay false when the load attempt fails")
    }

    // MARK: - unloadModel on never-loaded reranker

    func test_unloadModel_whenNeverLoaded_isNoOpAndDoesNotCrash() {
        let reranker = LlamaReranker()
        XCTAssertFalse(reranker.isReady)
        reranker.unloadModel()
        XCTAssertFalse(reranker.isReady,
                       "isReady must remain false after unloading a never-loaded reranker")
    }

    // MARK: - Candidate boundary conditions (not-ready passthrough)

    func test_rerank_emptyCandidates_returnsEmpty() async throws {
        let reranker = LlamaReranker()
        let result = try await reranker.rerank(query: "query", candidates: [], limit: 10)
        XCTAssertTrue(result.isEmpty,
                      "rerank with empty candidates must return an empty array")
    }

    func test_rerank_zeroLimit_returnsEmpty() async throws {
        let reranker = LlamaReranker()
        let docID = UUID()
        let candidates = [
            VectorSearchHit(
                chunk: DocumentChunk(documentID: docID, text: "some passage", chunkIndex: 0),
                documentTitle: "Doc",
                score: 0.9
            ),
        ]
        let result = try await reranker.rerank(query: "query", candidates: candidates, limit: 0)
        XCTAssertTrue(result.isEmpty,
                      "rerank with limit=0 must return an empty array")
    }

    func test_rerank_limitExceedsCandidates_returnsAllCandidates() async throws {
        let reranker = LlamaReranker()  // not loaded → passthrough
        let docID = UUID()
        let candidates = (0..<3).map {
            VectorSearchHit(
                chunk: DocumentChunk(documentID: docID, text: "p\($0)", chunkIndex: $0),
                documentTitle: "Doc",
                score: Float($0)
            )
        }
        let result = try await reranker.rerank(query: "q", candidates: candidates, limit: 100)
        XCTAssertEqual(result.count, 3,
                       "when limit > candidates.count, all candidates must be returned")
    }

    // MARK: - RerankerError description

    func test_rerankerError_modelNotLoaded_hasDescription() {
        let error = RerankerError.modelNotLoaded
        XCTAssertNotNil(error.errorDescription)
        XCTAssertFalse(error.errorDescription!.isEmpty)
    }

    func test_rerankerError_modelLoadFailed_includesUnderlyingDescription() {
        let underlying = NSError(domain: "Test", code: 42, userInfo: [
            NSLocalizedDescriptionKey: "sentinel-message",
        ])
        let error = RerankerError.modelLoadFailed(underlying: underlying)
        XCTAssertTrue(error.errorDescription?.contains("sentinel-message") == true,
                      "modelLoadFailed description must include the underlying error message")
    }
}
