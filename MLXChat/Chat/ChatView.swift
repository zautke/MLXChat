import SwiftUI
import MLXChatCore

struct ChatView: View {
    @State private var store: EngineStore
    @State private var serverController: ServerController
    @State private var viewModel: ChatViewModel
    @State private var showModelPicker = false
    @State private var showServerControl = false
    @State private var memoryGovernor = MemoryGovernor()

    init(store: EngineStore, serverController: ServerController) {
        _store = State(initialValue: store)
        _serverController = State(initialValue: serverController)
        _viewModel = State(initialValue: ChatViewModel(store: store))
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                switch viewModel.loadState {
                case .idle, .loading, .downloading, .failed:
                    LoadingView(state: viewModel.loadState, retry: {
                        Task { await viewModel.ensureLoaded() }
                    })
                case .ready:
                    MessageListView(messages: viewModel.messages, streamingText: viewModel.streamingText)
                    Divider()
                    if let stats = viewModel.stats {
                        StatsBar(stats: stats)
                    }
                    ComposerView(
                        input: $viewModel.input,
                        isGenerating: viewModel.isGenerating,
                        onSend: viewModel.send,
                        onCancel: viewModel.cancelGeneration
                    )
                }
            }
            .navigationTitle(viewModel.selectedModel.displayName)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        showServerControl = true
                    } label: {
                        Image(systemName: serverController.isRunning ? "server.rack" : "server.rack")
                            .foregroundStyle(serverController.isRunning ? .green : .secondary)
                    }
                    .accessibilityLabel("Local server")
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Models", systemImage: "cpu") { showModelPicker = true }
                }
            }
            .sheet(isPresented: $showModelPicker) {
                ModelPickerView(store: store)
            }
            .sheet(isPresented: $showServerControl) {
                ServerControlView(controller: serverController)
            }
            .task {
                memoryGovernor.start()
                await viewModel.ensureLoaded()
            }
        }
    }
}

private struct LoadingView: View {
    let state: EngineStore.LoadState
    let retry: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            switch state {
            case .idle, .loading:
                ProgressView("Loading model…")
            case .downloading(let fraction):
                ProgressView(value: fraction) {
                    Text("Downloading model… \(Int(fraction * 100))%")
                }
                .padding(.horizontal, 48)
            case .failed(let message):
                Image(systemName: "exclamationmark.triangle")
                    .font(.largeTitle)
                    .foregroundStyle(.orange)
                Text("Model failed to load")
                    .font(.headline)
                Text(message)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
                Button("Retry", action: retry)
                    .buttonStyle(.borderedProminent)
            case .ready:
                EmptyView()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
