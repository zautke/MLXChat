import Foundation
import Observation
import MLXChatCore

@MainActor
@Observable
final class ChatViewModel {
    var messages: [ChatMessage] = [
        .init(role: .system, content: "You are a helpful, concise assistant.")
    ]
    var input = ""
    var isGenerating = false
    var stats: EngineStats?
    private(set) var streamingText = ""

    private let store: EngineStore
    private var generationTask: Task<Void, Never>?

    init(store: EngineStore) {
        self.store = store
    }

    var loadState: EngineStore.LoadState { store.loadState }
    var selectedModel: ModelDescriptor { store.selectedModel }

    func ensureLoaded() async {
        await store.ensureLoaded()
    }

    func switchModel(to descriptor: ModelDescriptor) async {
        cancelGeneration()
        await store.switchModel(to: descriptor)
    }

    func send() {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isGenerating else { return }

        messages.append(.init(role: .user, content: trimmed))
        input = ""
        isGenerating = true
        streamingText = ""

        let history = messages
        generationTask = Task { [weak self] in
            guard let self else { return }
            await self.runGeneration(history: history)
        }
    }

    func cancelGeneration() {
        generationTask?.cancel()
        generationTask = nil
        if isGenerating {
            if !streamingText.isEmpty {
                messages.append(.init(role: .assistant, content: streamingText + " ▌"))
            }
            streamingText = ""
            isGenerating = false
        }
    }

    private func runGeneration(history: [ChatMessage]) async {
        defer { isGenerating = false }
        guard let engine = await store.currentEngineIfReady() else {
            messages.append(.init(role: .assistant, content: "⚠️ Model is not ready."))
            return
        }

        do {
            for try await token in engine.stream(messages: history, options: GenerateOptions()) {
                if Task.isCancelled { break }
                if token.isFinal { break }
                streamingText += token.text
            }
            if !streamingText.isEmpty {
                messages.append(.init(role: .assistant, content: streamingText))
            }
            streamingText = ""
            stats = await engine.lastStats
        } catch {
            if !streamingText.isEmpty {
                messages.append(.init(role: .assistant, content: streamingText))
                streamingText = ""
            }
            messages.append(.init(role: .assistant, content: "⚠️ \(error.localizedDescription)"))
        }
    }
}
