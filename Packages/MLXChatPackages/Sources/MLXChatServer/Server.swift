import Foundation
import Hummingbird
import MLXChatCore

/// Embedded OpenAI-compatible HTTP server. Defaults to loopback only.
/// iPadOS suspends the app in the background — the server lives only while the app does.
public actor ChatServer {
    private let engineProvider: @Sendable () async -> any LLMEngine
    private var task: Task<Void, Never>?

    public init(engineProvider: @Sendable @escaping () async -> any LLMEngine) {
        self.engineProvider = engineProvider
    }

    public func start(host: String = "127.0.0.1", port: Int = 8765) {
        guard task == nil else { return }
        task = Task { [engineProvider] in
            let router = Router()

            router.get("/healthz") { _, _ in
                "ok"
            }

            router.get("/v1/models") { _, _ in
                ModelsResponse(data: ModelCatalog.all.map(ModelObject.init))
            }

            router.post("/v1/chat/completions") { [engineProvider] req, ctx in
                let body = try await req.decode(as: ChatCompletionRequest.self, context: ctx)
                let engine = await engineProvider()
                let messages = body.messages.map {
                    ChatMessage(role: ChatMessage.Role(rawValue: $0.role) ?? .user, content: $0.content)
                }
                let options = GenerateOptions()

                if body.stream == true {
                    let stream = await engine.stream(messages: messages, options: options)
                    let bytes = AsyncStream<ByteBuffer> { continuation in
                        Task {
                            let encoder = JSONEncoder()
                            do {
                                for try await token in stream {
                                    let chunk = ChatCompletionChunk(content: token.text, isFinal: token.isFinal)
                                    if let data = try? encoder.encode(chunk),
                                       let line = String(data: data, encoding: .utf8) {
                                        continuation.yield(ByteBuffer(string: "data: \(line)\n\n"))
                                    }
                                }
                            } catch {
                                let err = ChatCompletionChunk(content: "error: \(error.localizedDescription)", isFinal: true)
                                if let data = try? encoder.encode(err),
                                   let line = String(data: data, encoding: .utf8) {
                                    continuation.yield(ByteBuffer(string: "data: \(line)\n\n"))
                                }
                            }
                            continuation.yield(ByteBuffer(string: "data: [DONE]\n\n"))
                            continuation.finish()
                        }
                    }
                    return Response(
                        status: .ok,
                        headers: [.contentType: "text/event-stream", .cacheControl: "no-cache"],
                        body: ResponseBody(asyncSequence: bytes)
                    )
                } else {
                    let (text, _) = try await engine.respond(messages: messages, options: options)
                    let payload = ChatCompletionResponse(content: text)
                    let data = try JSONEncoder().encode(payload)
                    return Response(
                        status: .ok,
                        headers: [.contentType: "application/json"],
                        body: ResponseBody(byteBuffer: ByteBuffer(data: data))
                    )
                }
            }

            let app = Application(
                router: router,
                configuration: .init(address: .hostname(host, port: port))
            )
            do {
                try await app.runService()
            } catch {
                // Server stopped or failed; nothing actionable from inside the task.
            }
        }
    }

    public func stop() {
        task?.cancel()
        task = nil
    }
}

// MARK: - DTOs

struct ChatCompletionRequest: Codable {
    struct Msg: Codable {
        let role: String
        let content: String
    }

    let model: String?
    let messages: [Msg]
    let stream: Bool?
}

struct ChatCompletionResponse: Codable, ResponseEncodable {
    let content: String
}

struct ChatCompletionChunk: Codable {
    let content: String
    let isFinal: Bool
}

struct ModelObject: Codable {
    let id: String
    let owned_by: String

    init(_ descriptor: ModelDescriptor) {
        self.id = descriptor.huggingFaceRepo
        self.owned_by = "mlxchat"
    }
}

struct ModelsResponse: Codable, ResponseEncodable {
    let data: [ModelObject]
}
