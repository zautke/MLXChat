import Foundation
import MLXChatCore

public protocol Retriever: Actor {
    func retrieve(query: String, k: Int) async throws -> [RetrievedChunk]
}

public struct RetrievedChunk: Sendable, Codable, Identifiable {
    public let id: String
    public let content: String
    public let score: Float
    public let source: String?

    public init(id: String, content: String, score: Float, source: String?) {
        self.id = id
        self.content = content
        self.score = score
        self.source = source
    }
}

/// Placeholder retriever — wire up Basic Memory MCP or a BM25 index here.
public actor StubRetriever: Retriever {
    public init() {}
    public func retrieve(query: String, k: Int) async throws -> [RetrievedChunk] {
        []
    }
}

public extension Array where Element == ChatMessage {
    /// Augment the final user message with retrieved context before generation.
    func withRetrievedContext(_ chunks: [RetrievedChunk]) -> [ChatMessage] {
        guard !chunks.isEmpty else { return self }
        let ctx = chunks.map { "- \($0.content)" }.joined(separator: "\n")
        let augmented = "Use this context if relevant:\n\(ctx)\n\n"
            + (last(where: { $0.role == .user })?.content ?? "")
        var copy = self
        if let idx = copy.lastIndex(where: { $0.role == .user }) {
            copy[idx].content = augmented
        }
        return copy
    }
}
