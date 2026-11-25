import Foundation
import Observation
import MLXChatCore
import MLXChatEngineMLX
import MLXChatEngineLlama

/// Owns the currently selected model and its engine. Shared by the chat UI,
/// the HTTP server, and the memory governor.
@MainActor
@Observable
final class EngineStore {
    enum LoadState: Equatable {
        case idle
        case downloading(Double)
        case loading
        case ready
        case failed(String)
    }

    var selectedModel: ModelDescriptor = ModelCatalog.default
    var loadState: LoadState = .idle

    private(set) var engine: (any LLMEngine)?

    func makeEngine(for descriptor: ModelDescriptor) -> any LLMEngine {
        switch descriptor.backend {
        case .mlx: return MLXEngine(descriptor: descriptor)
        case .llamaCpp: return LlamaEngine(descriptor: descriptor)
        }
    }

    func ensureLoaded() async {
        if case .ready = loadState, engine?.descriptor == selectedModel { return }
        await load(selectedModel)
    }

    func switchModel(to descriptor: ModelDescriptor) async {
        guard descriptor != selectedModel else { return }
        await engine?.unload()
        engine = nil
        selectedModel = descriptor
        loadState = .idle
        await load(descriptor)
    }

    func unload() async {
        await engine?.unload()
        engine = nil
        loadState = .idle
    }

    /// Engine only if already loaded; nil otherwise (never triggers a load).
    func currentEngineIfReady() -> (any LLMEngine)? {
        guard case .ready = loadState else { return nil }
        return engine
    }

    /// Provider for the HTTP server: returns the current engine, loading it if needed.
    func currentEngine() async -> any LLMEngine {
        await ensureLoaded()
        // ensureLoaded either sets engine or sets .failed; fall back to a fresh engine so the server never force-unwraps.
        if let engine { return engine }
        return makeEngine(for: selectedModel)
    }

    private func load(_ descriptor: ModelDescriptor) async {
        loadState = .loading
        let newEngine = makeEngine(for: descriptor)
        do {
            try await newEngine.load { [weak self] fraction in
                Task { @MainActor in
                    self?.loadState = .downloading(fraction)
                }
            }
            self.engine = newEngine
            self.loadState = .ready
        } catch {
            self.loadState = .failed(String(describing: error))
        }
    }
}
