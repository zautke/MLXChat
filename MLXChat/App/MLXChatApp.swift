import SwiftUI

@main
struct MLXChatApp: App {
    @State private var store: EngineStore
    @State private var serverController: ServerController

    init() {
        let store = EngineStore()
        _store = State(initialValue: store)
        _serverController = State(initialValue: ServerController(store: store))
    }

    var body: some Scene {
        WindowGroup {
            ChatView(store: store, serverController: serverController)
        }
    }
}
