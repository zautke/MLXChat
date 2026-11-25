import Foundation
import Observation
import MLXChatCore
import MLXChatServer

/// Owns the embedded Hummingbird server lifecycle. Off by default.
@MainActor
@Observable
final class ServerController {
    private(set) var isRunning = false
    private(set) var port: Int = 8765
    private var server: ChatServer?
    private let store: EngineStore

    init(store: EngineStore) {
        self.store = store
    }

    func toggle() {
        isRunning ? stop() : start()
    }

    func start(port: Int = 8765) {
        guard !isRunning else { return }
        self.port = port
        let store = self.store
        let server = ChatServer(engineProvider: { @Sendable in
            await store.currentEngine()
        })
        Task { await server.start(host: "127.0.0.1", port: port) }
        self.server = server
        isRunning = true
    }

    func stop() {
        let server = self.server
        Task { await server?.stop() }
        self.server = nil
        isRunning = false
    }
}
