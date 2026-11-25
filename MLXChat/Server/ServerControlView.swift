import SwiftUI

struct ServerControlView: View {
    let controller: ServerController
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Toggle("Local server", isOn: Binding(
                        get: { controller.isRunning },
                        set: { _ in controller.toggle() }
                    ))
                    LabeledContent("Address", value: "127.0.0.1:\(controller.port)")
                } header: {
                    Text("OpenAI-compatible API")
                } footer: {
                    Text("Serves POST /v1/chat/completions, GET /v1/models and GET /healthz on this device only. The server stops when the app is suspended. Not reachable from other devices.")
                }

                Section("Try it") {
                    Text("curl http://127.0.0.1:\(controller.port)/v1/chat/completions -H 'content-type: application/json' -d '{\"model\":\"mlxchat\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}'")
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                }
            }
            .navigationTitle("Local Server")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
