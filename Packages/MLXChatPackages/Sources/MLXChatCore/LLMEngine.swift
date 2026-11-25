import Foundation

/// The single inference boundary. The app, the HTTP server, and the voice
/// pipeline all talk to this protocol — never to a concrete engine.
public protocol LLMEngine: Actor {
    nonisolated var descriptor: ModelDescriptor { get }
    var isLoaded: Bool { get }

    /// Download (if needed) and load the model into memory.
    /// `progress` reports download progress 0.0...1.0; not called during compile/load.
    func load(progress: @Sendable @escaping (Double) -> Void) async throws

    /// Free model + KV cache. Idempotent.
    func unload() async

    /// Non-streaming generation. Returns the full response and stats.
    func respond(messages: [ChatMessage], options: GenerateOptions) async throws -> (text: String, stats: EngineStats)

    /// Streaming generation. The stream emits tokens as they arrive and a final
    /// `isFinal: true` token whose `text` is empty; stats are then available via `lastStats`.
    nonisolated func stream(messages: [ChatMessage], options: GenerateOptions) -> AsyncThrowingStream<GenerationToken, Error>

    /// Stats from the most recent `respond` or `stream` call.
    var lastStats: EngineStats? { get }
}
