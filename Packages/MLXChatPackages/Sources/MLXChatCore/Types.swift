import Foundation

public struct ChatMessage: Codable, Sendable, Identifiable, Hashable {
    public enum Role: String, Codable, Sendable, CaseIterable {
        case system, user, assistant
    }

    public let id: UUID
    public let role: Role
    public var content: String
    public let createdAt: Date

    public init(id: UUID = UUID(), role: Role, content: String, createdAt: Date = .now) {
        self.id = id
        self.role = role
        self.content = content
        self.createdAt = createdAt
    }
}

public struct GenerateOptions: Sendable {
    public var maxTokens: Int
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var repetitionPenalty: Float
    public var stopTokens: [String]
    /// Pass `false` to disable Qwen3.5-style thinking per request. Nil = model default.
    public var enableThinking: Bool?
    /// TurboQuant KV-cache scheme (e.g. "turbo8v3") for long contexts. Nil = default (fp16/affine8).
    public var kvScheme: String?

    public init(
        maxTokens: Int = 1024,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        topK: Int = 40,
        repetitionPenalty: Float = 1.1,
        stopTokens: [String] = [],
        enableThinking: Bool? = nil,
        kvScheme: String? = nil
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.stopTokens = stopTokens
        self.enableThinking = enableThinking
        self.kvScheme = kvScheme
    }
}

public struct GenerationToken: Sendable {
    public let text: String
    public let tokenIndex: Int
    public let isFinal: Bool

    public init(text: String, tokenIndex: Int, isFinal: Bool) {
        self.text = text
        self.tokenIndex = tokenIndex
        self.isFinal = isFinal
    }
}

public struct EngineStats: Sendable {
    public let promptTokens: Int
    public let completionTokens: Int
    public let promptTokensPerSecond: Double
    public let completionTokensPerSecond: Double
    public let peakResidentMemoryBytes: UInt64

    public init(promptTokens: Int, completionTokens: Int, promptTokensPerSecond: Double, completionTokensPerSecond: Double, peakResidentMemoryBytes: UInt64) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.promptTokensPerSecond = promptTokensPerSecond
        self.completionTokensPerSecond = completionTokensPerSecond
        self.peakResidentMemoryBytes = peakResidentMemoryBytes
    }
}

public enum EngineError: Error, Sendable, LocalizedError {
    case modelNotLoaded
    case modelLoadFailed(underlying: String)
    case downloadFailed(underlying: String)
    case generationFailed(underlying: String)
    case insufficientMemory(availableBytes: UInt64, requiredBytes: UInt64)
    case backendUnavailable(reason: String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Model is not loaded."
        case .modelLoadFailed(let underlying):
            return "Model failed to load: \(underlying)"
        case .downloadFailed(let underlying):
            return "Model download failed: \(underlying)"
        case .generationFailed(let underlying):
            return "Generation failed: \(underlying)"
        case .insufficientMemory(let available, let required):
            let a = ByteCountFormatter.string(fromByteCount: Int64(available), countStyle: .memory)
            let r = ByteCountFormatter.string(fromByteCount: Int64(required), countStyle: .memory)
            return "Not enough memory (available \(a), need ~\(r)). Try a smaller model."
        case .backendUnavailable(let reason):
            return "Backend unavailable: \(reason)"
        }
    }
}

public enum BackendKind: String, Codable, Sendable, CaseIterable {
    case mlx
    case llamaCpp
}

public struct ModelDescriptor: Codable, Sendable, Identifiable, Hashable {
    public var id: String { huggingFaceRepo + "#" + (filename ?? "") }
    public let displayName: String
    public let huggingFaceRepo: String
    /// Required only for GGUF; nil for MLX (whole repo is downloaded).
    public let filename: String?
    public let backend: BackendKind
    public let approximateBytesOnDisk: UInt64
    public let recommendedContextWindow: Int
    public let supportsThinking: Bool

    public init(
        displayName: String,
        huggingFaceRepo: String,
        filename: String? = nil,
        backend: BackendKind,
        approximateBytesOnDisk: UInt64,
        recommendedContextWindow: Int = 32_768,
        supportsThinking: Bool = false
    ) {
        self.displayName = displayName
        self.huggingFaceRepo = huggingFaceRepo
        self.filename = filename
        self.backend = backend
        self.approximateBytesOnDisk = approximateBytesOnDisk
        self.recommendedContextWindow = recommendedContextWindow
        self.supportsThinking = supportsThinking
    }
}
