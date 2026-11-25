import Foundation
import LocalLLMClient
import LocalLLMClientLlama
import MLXChatCore

/// llama.cpp / GGUF backend via LocalLLMClient (pinned 0.5.0 — API unstable upstream).
public actor LlamaEngine: LLMEngine {
    public nonisolated let descriptor: ModelDescriptor
    public private(set) var isLoaded: Bool = false
    public private(set) var lastStats: EngineStats?

    private var session: LLMSession?

    public init(descriptor: ModelDescriptor) {
        precondition(descriptor.backend == .llamaCpp, "LlamaEngine requires .llamaCpp backend")
        precondition(descriptor.filename != nil, "GGUF requires an explicit filename")
        self.descriptor = descriptor
    }

    public func load(progress: @Sendable @escaping (Double) -> Void) async throws {
        guard !isLoaded else { return }
        guard let filename = descriptor.filename else {
            throw EngineError.modelLoadFailed(underlying: "missing GGUF filename")
        }

        let model = LLMSession.DownloadModel.llama(
            id: descriptor.huggingFaceRepo,
            model: filename,
            parameter: .init(
                context: descriptor.recommendedContextWindow,
                temperature: 0.7,
                topK: 40,
                topP: 0.9
            )
        )

        do {
            try await model.downloadModel { fraction in
                progress(fraction)
            }
            self.session = LLMSession(model: model)
            self.isLoaded = true
        } catch {
            throw EngineError.modelLoadFailed(underlying: String(describing: error))
        }
    }

    public func unload() async {
        session = nil
        isLoaded = false
    }

    public func respond(
        messages: [ChatMessage],
        options: GenerateOptions
    ) async throws -> (text: String, stats: EngineStats) {
        var collected = ""
        for try await token in stream(messages: messages, options: options) {
            collected += token.text
        }
        guard let stats = lastStats else {
            throw EngineError.generationFailed(underlying: "no stats produced")
        }
        return (collected, stats)
    }

    public nonisolated func stream(
        messages: [ChatMessage],
        options: GenerateOptions
    ) -> AsyncThrowingStream<GenerationToken, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self._stream(messages: messages, options: options, continuation: continuation)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func _stream(
        messages: [ChatMessage],
        options: GenerateOptions,
        continuation: AsyncThrowingStream<GenerationToken, Error>.Continuation
    ) async throws {
        guard let session else { throw EngineError.modelNotLoaded }

        // History goes into the session; the final user turn is the prompt.
        let history = messages.dropLast()
        session.messages = history.map(Self.toLLMMessage)
        let prompt = messages.last(where: { $0.role == .user })?.content ?? ""

        var index = 0
        let start = ContinuousClock.now
        for try await chunk in session.streamResponse(to: prompt) {
            continuation.yield(GenerationToken(text: chunk, tokenIndex: index, isFinal: false))
            index += 1
        }
        let duration = start.duration(to: .now)
        self.lastStats = EngineStats(
            promptTokens: 0, // LocalLLMClient does not surface token counts yet
            completionTokens: index,
            promptTokensPerSecond: 0,
            completionTokensPerSecond: Self.tps(index, duration),
            peakResidentMemoryBytes: 0
        )
        continuation.yield(GenerationToken(text: "", tokenIndex: index, isFinal: true))
        continuation.finish()
    }

    private static func toLLMMessage(_ msg: ChatMessage) -> LLMInput.Message {
        switch msg.role {
        case .system: return .system(msg.content)
        case .user: return .user(msg.content)
        case .assistant: return .assistant(msg.content)
        }
    }

    private static func tps(_ tokens: Int, _ duration: Duration) -> Double {
        let seconds = Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1e18
        return seconds > 0 ? Double(tokens) / seconds : 0
    }
}
