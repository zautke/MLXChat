import XCTest
@testable import MLXChatCore

final class MLXChatCoreTests: XCTestCase {
    func testDefaultModelIsCatalogDefault() {
        XCTAssertEqual(ModelCatalog.default.huggingFaceRepo, "mlx-community/Qwen3.5-9B-MLX-4bit")
        XCTAssertTrue(ModelCatalog.all.contains(ModelCatalog.default))
    }

    func testCatalogIDsAreUnique() {
        let ids = ModelCatalog.all.map(\.id)
        XCTAssertEqual(Set(ids).count, ids.count)
    }

    func testGGUFModelsRequireFilename() {
        for model in ModelCatalog.all where model.backend == .llamaCpp {
            XCTAssertNotNil(model.filename, "\(model.displayName) must declare a GGUF filename")
        }
    }

    func testModelDescriptorCodableRoundTrip() throws {
        let data = try JSONEncoder().encode(ModelCatalog.default)
        let decoded = try JSONDecoder().decode(ModelDescriptor.self, from: data)
        XCTAssertEqual(decoded, ModelCatalog.default)
    }

    func testChatMessageCodableRoundTrip() throws {
        let message = ChatMessage(role: .user, content: "hello")
        let data = try JSONEncoder().encode(message)
        let decoded = try JSONDecoder().decode(ChatMessage.self, from: data)
        XCTAssertEqual(decoded.content, "hello")
        XCTAssertEqual(decoded.role, .user)
    }
}
