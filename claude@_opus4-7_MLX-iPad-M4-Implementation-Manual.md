# MLX iOS Wrapper for iPad Pro M4 (16GB) — Implementation Manual

**Author:** Claude (research + plan)  
**Owner:** Luke (zautke)  
**Target device:** iPad Pro 11" M4, 1TB/2TB tier — 16GB unified memory, 10-core CPU, 10-core GPU, 16-core Neural Engine, 120GB/s memory bandwidth  
**Target OS:** iPadOS 26+ (for `SpeechAnalyzer` v1 dictation; falls back to `SFSpeechRecognizer` on iPadOS 18)  
**Date:** April 26, 2026

---

## 0. TL;DR — what we're building

A SwiftUI iPad app, **`MLXChat`** (placeholder name — change before App Store submission), that:

1. Wraps a **dual-backend** local LLM (Large Language Model) runtime: Apple **MLX (Machine Learning eXchange)** for native MLX-format models, and **`llama.cpp`** for GGUF (GPT-Generated Unified Format) models.
2. Exposes a **single Swift `LLMEngine` protocol** so the rest of the app never sees the backend choice.
3. Ships with **`mlx-community/Qwen3-8B-4bit-AWQ`** as the default model (~4.7GB on disk, ~5.5GB resident, leaves headroom for a 32k context KV cache plus app overhead inside the iOS Increased Memory Limit ceiling).
4. Supports **streaming token-by-token UI**, **multi-turn chat with system prompt**, **on-device voice dictation** (`SpeechAnalyzer` on iPadOS 26, `SFSpeechRecognizer` fallback), **runtime model picker / hot-swap**, and a **stub RAG (Retrieval-Augmented Generation) hook**.
5. Ships an **embedded OpenAI-compatible HTTP server** (Hummingbird 2) bound to `127.0.0.1` so other on-device apps (or your Mac on the same network with a manual port forward) can hit `/v1/chat/completions`.

**The "generic app interface"** Luke asked about resolves to three concentric layers, all of which are produced as part of this build:

| Layer | What | Lives in |
|---|---|---|
| **L1 — Swift API** | `LLMEngine` protocol (`func respond(...) async`, `func stream(...) -> AsyncThrowingStream<Token, Error>`) | `MLXChatCore` SPM (Swift Package Manager) target |
| **L2 — SwiftUI surface** | `ChatView`, `ModelPickerView`, `VoiceInputBar`, `MessageBubble` | `MLXChat` app target |
| **L3 — HTTP surface** | `POST /v1/chat/completions` (OpenAI-compatible), `GET /v1/models`, `GET /healthz` | `MLXChatServer` SPM target |

---

## 1. Hardware reality check (read this before you write any code)

These are the constraints that drive every decision below. Skipping any one of them will bite you on the first model load.

### 1.1 RAM ceiling

> Sources reviewed: Apple Developer Forums thread #777370 (March 2025), thread #724195 (system limits on iPad-on-Mac apps), Rene Ritchie "Why the iPad Pro Has a 5GB RAM Limit", AppleInsider on iOS 15 entitlement.

- The iPad Pro 11" M4 1TB/2TB tier has **16 GB physical RAM**, **120 GB/s memory bandwidth** (Apple Tech Specs, support.apple.com/en-us/119892).
- Without the `com.apple.developer.kernel.increased-memory-limit` entitlement, an iPad app is jetsamed at roughly **half of physical RAM** in the worst case, but realistically **~5 GB** on a 16 GB iPad.
- **With** the entitlement, the *empirically observed* ceiling reported by developers on a 16 GB iPad Pro is **~12 GB**. Apple does **not** document a number; you must call `os_proc_available_memory()` at runtime and treat its return value as authoritative.
- iPadOS 18 added a `vm-compressor-space-shortage` jetsam reason that fires when your app pushes the VM compressor too hard, even if the headline allocation looks fine. Spiky access patterns (e.g., reading random KV-cache pages) trigger this faster than steady streaming reads.
- The simulator has no Metal MTLGPUFamily that MLX accepts. **All MLX runs must be on a real device.** Build the UI in the simulator; run inference on hardware.

### 1.2 Working memory budget on a 16 GB iPad

Carve up the ~12 GB ceiling honestly:

| Bucket | Reserve | Why |
|---|---|---|
| Model weights (Qwen3-8B-4bit-AWQ) | ~5.5 GB resident | The 4.7 GB on-disk file expands during dequantization-aware Metal kernels |
| KV cache (32k context, 8B model, 4-bit weights, fp16 KV) | ~2.5–3.0 GB | Grows linearly with sequence length |
| MLX GPU buffer cache | 256 MB | We cap this with `MLX.GPU.set(cacheLimit:)` |
| Tokenizer + chat template | ~150 MB | swift-transformers overhead |
| App + SwiftUI + SpeechAnalyzer | ~500 MB | Realistic baseline |
| Hummingbird HTTP server | ~50 MB | Idle |
| Headroom for jetsam safety | **~2.5 GB** | Don't skip this |
| **Total** | **~11–11.5 GB** | Fits inside the entitlement ceiling |

If you hot-swap to a 4B model, you reclaim ~3 GB and can extend context to 64k+.

### 1.3 Why Qwen3-8B-4bit-AWQ specifically

| Candidate | On-disk | Quality (Arena AI / SWE-bench) | Notes |
|---|---|---|---|
| **Qwen3-8B-4bit-AWQ** ✅ default | 4.7 GB | Top of the 8B class | AWQ (Activation-aware Weight Quantization) preserves more accuracy than plain 4-bit |
| Qwen3-8B-4bit-DWQ | 4.7 GB | ≈ AWQ, sometimes slightly better on math | DWQ (Dynamic Weight Quantization) — newer, less battle-tested |
| Qwen3-8B-4bit | 4.5 GB | Baseline | Use as fallback |
| Gemma 4 E4B 4-bit | ~5 GB | Strong, 256k context | Good alternative; matches Qwen3-8B on quality, longer context |
| Qwen2.5-Coder-7B-Instruct-4bit | 4.3 GB | Best for code | Switch to this for coding tasks |
| Qwen3-4B-4bit | 2.3 GB | Step down | Use if 8B causes jetsam pressure |
| Gemma 4 E2B 4-bit | ~2 GB | Surprisingly strong | Fast fallback |

### 1.4 The MLX vs Ollama/GGUF reality

> Will It Run AI benchmarks (April 2026): MLX is 15–30% faster than llama.cpp on Apple Silicon at the same quantization, with ~10% lower memory overhead. Ollama uses GGUF; MLX uses MLX-format `.safetensors`. The *model* (Qwen3-8B) is identical — only the file format and runtime differ.

You said "wrap the optimal local Olama model" — what we actually do is download the **MLX-format** version of the same model family from `mlx-community` on Hugging Face, *plus* keep a `llama.cpp` (GGUF) backend available so you can run any Ollama-compatible model unchanged. The `LLMEngine` protocol abstracts the choice. **Default backend is MLX.** GGUF is opt-in via the model picker.

---

## 2. Package architecture

Three SPM packages (Swift Package Manager) in a workspace, plus the iOS app target:

```
MLXChat.xcworkspace/
├── MLXChat.xcodeproj                   # iOS app target (SwiftUI)
└── Packages/
    ├── MLXChatCore/                    # L1 — Swift API + engines
    │   ├── Package.swift
    │   └── Sources/
    │       ├── MLXChatCore/            # Public protocols, types
    │       ├── MLXChatEngineMLX/       # MLX backend
    │       └── MLXChatEngineLlama/     # llama.cpp / GGUF backend
    ├── MLXChatServer/                  # L3 — Hummingbird HTTP
    │   ├── Package.swift
    │   └── Sources/MLXChatServer/
    └── MLXChatRAG/                     # Stub for vector store / retrieval
        ├── Package.swift
        └── Sources/MLXChatRAG/
```

### 2.1 Top-level dependencies (verified April 26, 2026)

| Dependency | Version | Source | Confirmed |
|---|---|---|---|
| `mlx-swift` | `0.25.3+` | `github.com/ml-explore/mlx-swift` | swiftpackageregistry.com, last updated 2026-04-24 |
| **`mlx-swift-lm`** (new home for LLM/VLM code, replaces `mlx-swift-examples` libraries) | `3.31.3` | `github.com/ml-explore/mlx-swift-lm` | swiftpackageindex.com, released 10 days ago |
| `LocalLLMClient` | `0.4.6` (or `main`) | `github.com/tattn/LocalLLMClient` | Just merged PR #85 to use mlx-swift-lm |
| `swift-transformers` | `0.1.14+` | `github.com/huggingface/swift-transformers` | Pulled transitively |
| `hummingbird` | `2.x` | `github.com/hummingbird-project/hummingbird` | Used by macMLX, SwiftLM |
| `swift-numerics` | `1.0.3+` | `github.com/apple/swift-numerics` | Pulled transitively |

> **Important breaking change:** `mlx-swift-lm` 3.x decoupled from tokenizer/downloader packages. The old `import MLXLLM` from `mlx-swift-examples` is migrating to `import MLXLLM` from `mlx-swift-lm`. The release notes link to migration docs.

### 2.2 `MLXChatCore/Package.swift`

```swift
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MLXChatCore",
    platforms: [
        .iOS(.v18),       // Minimum for Increased Memory Limit + MLX
        .macOS(.v14),     // For dev / catalyst
    ],
    products: [
        .library(name: "MLXChatCore", targets: ["MLXChatCore"]),
        .library(name: "MLXChatEngineMLX", targets: ["MLXChatEngineMLX"]),
        .library(name: "MLXChatEngineLlama", targets: ["MLXChatEngineLlama"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.25.3"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.31.3"),
        .package(url: "https://github.com/tattn/LocalLLMClient", from: "0.4.6"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.14"),
    ],
    targets: [
        .target(
            name: "MLXChatCore",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        .target(
            name: "MLXChatEngineMLX",
            dependencies: [
                "MLXChatCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
        ),
        .target(
            name: "MLXChatEngineLlama",
            dependencies: [
                "MLXChatCore",
                .product(name: "LocalLLMClient", package: "LocalLLMClient"),
                .product(name: "LocalLLMClientLlama", package: "LocalLLMClient"),
                .product(name: "LocalLLMClientUtility", package: "LocalLLMClient"),
            ]
        ),
        .testTarget(name: "MLXChatCoreTests", dependencies: ["MLXChatCore"]),
    ],
    swiftLanguageModes: [.v6]
)
```

---

## 3. The generic app interface (L1 — `MLXChatCore`)

This is the protocol every other layer codes against. Both backends conform to it. The HTTP server and the SwiftUI views hold an `any LLMEngine` and never branch on backend.

### 3.1 Public types

```swift
// Sources/MLXChatCore/Types.swift
import Foundation

public struct ChatMessage: Codable, Sendable, Identifiable, Hashable {
    public enum Role: String, Codable, Sendable { case system, user, assistant }
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
    public var maxTokens: Int = 1024
    public var temperature: Float = 0.7
    public var topP: Float = 0.9
    public var topK: Int = 40
    public var repetitionPenalty: Float = 1.1
    public var stopTokens: [String] = []
    public init() {}
}

public struct GenerationToken: Sendable {
    public let text: String
    public let tokenIndex: Int
    public let isFinal: Bool
}

public struct EngineStats: Sendable {
    public let promptTokens: Int
    public let completionTokens: Int
    public let promptTokensPerSecond: Double
    public let completionTokensPerSecond: Double
    public let peakResidentMemoryBytes: UInt64
}

public enum EngineError: Error, Sendable {
    case modelNotLoaded
    case modelLoadFailed(underlying: String)
    case downloadFailed(underlying: String)
    case generationFailed(underlying: String)
    case insufficientMemory(availableBytes: UInt64, requiredBytes: UInt64)
    case backendUnavailable(reason: String)
}

public enum BackendKind: String, Codable, Sendable, CaseIterable {
    case mlx
    case llamaCpp
}

public struct ModelDescriptor: Codable, Sendable, Identifiable, Hashable {
    public var id: String { huggingFaceRepo + "#" + (filename ?? "") }
    public let displayName: String
    public let huggingFaceRepo: String
    /// Required only for GGUF; nil for MLX (whole repo is downloaded)
    public let filename: String?
    public let backend: BackendKind
    public let approximateBytesOnDisk: UInt64
    public let recommendedContextWindow: Int
    public let supportsThinking: Bool   // Qwen3 /think mode

    public init(displayName: String,
                huggingFaceRepo: String,
                filename: String? = nil,
                backend: BackendKind,
                approximateBytesOnDisk: UInt64,
                recommendedContextWindow: Int = 32_768,
                supportsThinking: Bool = false) {
        self.displayName = displayName
        self.huggingFaceRepo = huggingFaceRepo
        self.filename = filename
        self.backend = backend
        self.approximateBytesOnDisk = approximateBytesOnDisk
        self.recommendedContextWindow = recommendedContextWindow
        self.supportsThinking = supportsThinking
    }
}
```

### 3.2 The `LLMEngine` protocol — single source of truth

```swift
// Sources/MLXChatCore/LLMEngine.swift
import Foundation

public protocol LLMEngine: Actor {
    var descriptor: ModelDescriptor { get }
    var isLoaded: Bool { get }

    /// Download (if needed) and load the model into memory.
    /// `progress` reports download progress 0.0...1.0; not called during compile/load.
    func load(progress: @Sendable @escaping (Double) -> Void) async throws

    /// Free model + KV cache. Idempotent.
    func unload() async

    /// Non-streaming generation. Returns the full response and stats.
    func respond(messages: [ChatMessage], options: GenerateOptions) async throws -> (text: String, stats: EngineStats)

    /// Streaming generation. The stream emits tokens as they arrive and a final
    /// `isFinal: true` token whose `text` is empty after which `EngineStats`
    /// is available via `lastStats`.
    func stream(messages: [ChatMessage], options: GenerateOptions) -> AsyncThrowingStream<GenerationToken, Error>

    /// Stats from the most recent `respond` or `stream` call.
    var lastStats: EngineStats? { get }
}
```

### 3.3 Why an `actor`

Both MLX inference state (`ModelContainer` from mlx-swift-lm) and `LocalLLMClient` are not `Sendable`-safe across raw threads. Wrapping each engine in an actor gives you Swift 6 concurrency safety for free, and it lets the HTTP server, the SwiftUI view, and the voice input pipeline all hold the same engine reference without manual locking. Token streams are explicitly `AsyncThrowingStream` so SwiftUI's `.task { for try await … }` works directly.

### 3.4 The MLX engine implementation

```swift
// Sources/MLXChatEngineMLX/MLXEngine.swift
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXChatCore

public actor MLXEngine: LLMEngine {
    public nonisolated let descriptor: ModelDescriptor
    public private(set) var isLoaded: Bool = false
    public private(set) var lastStats: EngineStats?

    private var container: ModelContainer?

    /// Buffer cache cap: 256 MB is the empirical sweet spot for 8B models on
    /// 16 GB iPads — large enough to amortize Metal allocator churn, small
    /// enough that it doesn't squeeze the KV cache.
    private static let gpuCacheLimitBytes = 256 * 1024 * 1024

    public init(descriptor: ModelDescriptor) {
        precondition(descriptor.backend == .mlx, "MLXEngine requires .mlx backend")
        self.descriptor = descriptor
    }

    public func load(progress: @Sendable @escaping (Double) -> Void) async throws {
        guard !isLoaded else { return }

        // Cap the buffer cache BEFORE loading so the loader doesn't fight
        // for memory.
        MLX.GPU.set(cacheLimit: Self.gpuCacheLimitBytes)

        // Pre-flight: do we have headroom?
        let available = os_proc_available_memory()
        let required = UInt64(Double(descriptor.approximateBytesOnDisk) * 1.4) // ~40% expansion
        guard UInt64(available) >= required else {
            throw EngineError.insufficientMemory(
                availableBytes: UInt64(available),
                requiredBytes: required
            )
        }

        let configuration = ModelConfiguration(
            id: descriptor.huggingFaceRepo
        )

        do {
            self.container = try await LLMModelFactory.shared.loadContainer(
                configuration: configuration
            ) { p in
                progress(p.fractionCompleted)
            }
            self.isLoaded = true
        } catch {
            throw EngineError.modelLoadFailed(underlying: String(describing: error))
        }
    }

    public func unload() async {
        container = nil
        isLoaded = false
        MLX.GPU.clearCache()
    }

    public func respond(
        messages: [ChatMessage],
        options: GenerateOptions
    ) async throws -> (text: String, stats: EngineStats) {
        var collected = ""
        let stream = self.stream(messages: messages, options: options)
        for try await token in stream {
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
                    try await self._stream(
                        messages: messages,
                        options: options,
                        continuation: continuation
                    )
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
        guard let container = self.container else {
            throw EngineError.modelNotLoaded
        }

        let userInput = UserInput(
            messages: messages.map { msg in
                ["role": msg.role.rawValue, "content": msg.content]
            }
        )

        let parameters = GenerateParameters(
            temperature: options.temperature,
            topP: options.topP,
            repetitionPenalty: options.repetitionPenalty
        )

        let startTime = ContinuousClock.now
        var promptTokenCount = 0
        var completionTokenCount = 0
        var generationStarted: ContinuousClock.Instant?

        try await container.perform { [weak self] context in
            let input = try await context.processor.prepare(input: userInput)
            promptTokenCount = input.text.tokens.size

            _ = try MLXLMCommon.generate(
                input: input,
                parameters: parameters,
                context: context
            ) { tokens in
                if generationStarted == nil { generationStarted = .now }
                let newText = context.tokenizer.decode(tokens: tokens)
                continuation.yield(GenerationToken(
                    text: newText,
                    tokenIndex: completionTokenCount,
                    isFinal: false
                ))
                completionTokenCount += 1
                return completionTokenCount >= options.maxTokens ? .stop : .more
            }
        }

        let endTime = ContinuousClock.now
        let totalDuration = startTime.duration(to: endTime)
        let promptDuration = generationStarted.map { startTime.duration(to: $0) }
            ?? totalDuration
        let completionDuration = generationStarted.map { $0.duration(to: endTime) }
            ?? .seconds(0)

        let stats = EngineStats(
            promptTokens: promptTokenCount,
            completionTokens: completionTokenCount,
            promptTokensPerSecond: tps(promptTokenCount, promptDuration),
            completionTokensPerSecond: tps(completionTokenCount, completionDuration),
            peakResidentMemoryBytes: residentMemoryBytes()
        )
        self.lastStats = stats

        continuation.yield(GenerationToken(text: "", tokenIndex: completionTokenCount, isFinal: true))
        continuation.finish()
    }

    private func tps(_ tokens: Int, _ duration: Duration) -> Double {
        let seconds = Double(duration.components.seconds) +
            Double(duration.components.attoseconds) / 1e18
        return seconds > 0 ? Double(tokens) / seconds : 0
    }

    private func residentMemoryBytes() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? info.resident_size : 0
    }
}
```

### 3.5 The llama.cpp engine implementation

```swift
// Sources/MLXChatEngineLlama/LlamaEngine.swift
import Foundation
import LocalLLMClient
import LocalLLMClientLlama
import LocalLLMClientUtility
import MLXChatCore

public actor LlamaEngine: LLMEngine {
    public nonisolated let descriptor: ModelDescriptor
    public private(set) var isLoaded: Bool = false
    public private(set) var lastStats: EngineStats?

    private var session: LLMSession?

    public init(descriptor: ModelDescriptor) {
        precondition(descriptor.backend == .llamaCpp, "LlamaEngine requires .llamaCpp backend")
        precondition(descriptor.filename != nil, "GGUF requires explicit filename")
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
            try await model.downloadModel { p in progress(p) }
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
        guard let session else { throw EngineError.modelNotLoaded }
        session.messages = messages.map(toLLMMessage)
        let userPrompt = messages.last(where: { $0.role == .user })?.content ?? ""
        let start = ContinuousClock.now
        let text = try await session.respond(to: userPrompt)
        let duration = start.duration(to: .now)
        let stats = EngineStats(
            promptTokens: 0, // LocalLLMClient doesn't surface token counts yet
            completionTokens: text.count / 4, // rough heuristic
            promptTokensPerSecond: 0,
            completionTokensPerSecond: tps(text.count / 4, duration),
            peakResidentMemoryBytes: 0
        )
        self.lastStats = stats
        return (text, stats)
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
        session.messages = messages.map(toLLMMessage)
        let userPrompt = messages.last(where: { $0.role == .user })?.content ?? ""

        var index = 0
        let start = ContinuousClock.now
        for try await chunk in session.streamResponse(to: userPrompt) {
            continuation.yield(GenerationToken(text: chunk, tokenIndex: index, isFinal: false))
            index += 1
        }
        let duration = start.duration(to: .now)
        self.lastStats = EngineStats(
            promptTokens: 0,
            completionTokens: index,
            promptTokensPerSecond: 0,
            completionTokensPerSecond: tps(index, duration),
            peakResidentMemoryBytes: 0
        )
        continuation.yield(GenerationToken(text: "", tokenIndex: index, isFinal: true))
        continuation.finish()
    }

    private func toLLMMessage(_ msg: ChatMessage) -> LLMSession.Message {
        switch msg.role {
        case .system: return .system(msg.content)
        case .user: return .user(msg.content)
        case .assistant: return .assistant(msg.content)
        }
    }

    private func tps(_ tokens: Int, _ duration: Duration) -> Double {
        let seconds = Double(duration.components.seconds) +
            Double(duration.components.attoseconds) / 1e18
        return seconds > 0 ? Double(tokens) / seconds : 0
    }
}
```

### 3.6 Engine factory

```swift
// Sources/MLXChatCore/EngineFactory.swift
import Foundation

public enum EngineFactory {
    /// Build the right engine for a descriptor. Caller decides whether to
    /// `await engine.load(...)` immediately or lazily.
    public static func make(_ descriptor: ModelDescriptor) -> any LLMEngine {
        switch descriptor.backend {
        case .mlx:
            // Resolved via runtime require; if MLXChatEngineMLX isn't linked,
            // this is a programmer error.
            return _makeMLX(descriptor)
        case .llamaCpp:
            return _makeLlama(descriptor)
        }
    }
}

// In MLXChatEngineMLX (file scope):
import MLXChatEngineMLX
@_cdecl("mlxchat_make_mlx") public func _makeMLX(_ d: ModelDescriptor) -> any LLMEngine {
    MLXEngine(descriptor: d)
}

// In MLXChatEngineLlama (file scope):
import MLXChatEngineLlama
@_cdecl("mlxchat_make_llama") public func _makeLlama(_ d: ModelDescriptor) -> any LLMEngine {
    LlamaEngine(descriptor: d)
}
```

> **Note:** The `@_cdecl` trick is one option; alternatively, just import both engine modules from the app target and resolve via a switch. Pick whichever you prefer. The simpler approach for an iOS app:

```swift
// In the app target:
import MLXChatCore
import MLXChatEngineMLX
import MLXChatEngineLlama

func makeEngine(_ d: ModelDescriptor) -> any LLMEngine {
    switch d.backend {
    case .mlx:       return MLXEngine(descriptor: d)
    case .llamaCpp:  return LlamaEngine(descriptor: d)
    }
}
```

---

## 4. The model registry

```swift
// Sources/MLXChatCore/ModelCatalog.swift
public enum ModelCatalog {
    public static let qwen3_8B_4bit_AWQ = ModelDescriptor(
        displayName: "Qwen3-8B (AWQ 4-bit, MLX)",
        huggingFaceRepo: "mlx-community/Qwen3-8B-4bit-AWQ",
        backend: .mlx,
        approximateBytesOnDisk: 4_700_000_000,
        recommendedContextWindow: 32_768,
        supportsThinking: true
    )
    public static let qwen3_8B_4bit = ModelDescriptor(
        displayName: "Qwen3-8B (4-bit, MLX)",
        huggingFaceRepo: "mlx-community/Qwen3-8B-4bit",
        backend: .mlx,
        approximateBytesOnDisk: 4_500_000_000,
        recommendedContextWindow: 32_768,
        supportsThinking: true
    )
    public static let qwen3_4B_4bit = ModelDescriptor(
        displayName: "Qwen3-4B (4-bit, MLX)",
        huggingFaceRepo: "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        backend: .mlx,
        approximateBytesOnDisk: 2_300_000_000,
        recommendedContextWindow: 32_768,
        supportsThinking: false
    )
    public static let gemma4_E4B_4bit = ModelDescriptor(
        displayName: "Gemma 4 E4B (4-bit, MLX)",
        huggingFaceRepo: "mlx-community/gemma-4-e4b-it-4bit",
        backend: .mlx,
        approximateBytesOnDisk: 5_000_000_000,
        recommendedContextWindow: 65_536,
        supportsThinking: false
    )
    public static let qwen2_5_Coder_7B_4bit = ModelDescriptor(
        displayName: "Qwen2.5-Coder-7B (4-bit, MLX)",
        huggingFaceRepo: "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        backend: .mlx,
        approximateBytesOnDisk: 4_300_000_000,
        recommendedContextWindow: 32_768,
        supportsThinking: false
    )
    // GGUF (llama.cpp) fallback — same model family as the user's Ollama setup
    public static let qwen3_8B_GGUF = ModelDescriptor(
        displayName: "Qwen3-8B (Q4_K_M, GGUF)",
        huggingFaceRepo: "lmstudio-community/Qwen3-8B-GGUF",
        filename: "Qwen3-8B-Q4_K_M.gguf",
        backend: .llamaCpp,
        approximateBytesOnDisk: 4_900_000_000,
        recommendedContextWindow: 32_768,
        supportsThinking: true
    )

    public static let all: [ModelDescriptor] = [
        qwen3_8B_4bit_AWQ,
        qwen3_8B_4bit,
        qwen3_4B_4bit,
        gemma4_E4B_4bit,
        qwen2_5_Coder_7B_4bit,
        qwen3_8B_GGUF,
    ]

    public static let `default` = qwen3_8B_4bit_AWQ
}
```

---

## 5. Xcode project setup — every checkbox that matters

### 5.1 Capabilities & entitlements (`MLXChat.entitlements`)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.developer.kernel.increased-memory-limit</key>
    <true/>
    <key>com.apple.developer.kernel.extended-virtual-addressing</key>
    <true/>
    <key>com.apple.security.app-sandbox</key>
    <true/>
    <key>com.apple.security.network.client</key>
    <true/>
    <key>com.apple.security.network.server</key>
    <true/>
</dict>
</plist>
```

In Xcode: `Signing & Capabilities` → `+ Capability` → add **Increased Memory Limit**, **Extended Virtual Addressing**, **App Sandbox**, **Background Modes** (check `Audio` for voice and `Background fetch` for download resumption).

### 5.2 `Info.plist` keys

| Key | Value | Why |
|---|---|---|
| `NSMicrophoneUsageDescription` | `MLXChat uses your microphone for on-device voice dictation. Your voice never leaves the device.` | Required for `SpeechAnalyzer` and `AVAudioRecorder` |
| `NSSpeechRecognitionUsageDescription` | `MLXChat transcribes your speech entirely on-device.` | Required even for `SpeechAnalyzer` per Apple guidance |
| `UIBackgroundModes` | `audio`, `fetch` | Continue dictation through interruptions; resume model download |
| `NSAppTransportSecurity` → `NSAllowsLocalNetworking` | `true` | For the local Hummingbird server |
| `LSApplicationQueriesSchemes` | (empty) | Defensive |
| `UIRequiredDeviceCapabilities` | `arm64` | Excludes simulator from App Store |

### 5.3 Build settings

| Setting | Value | Why |
|---|---|---|
| `EXCLUDED_ARCHS` (iOS Simulator) | `x86_64 arm64` | MLX Metal kernels do not run in the simulator. Build only the UI on simulator if you need it |
| `SWIFT_STRICT_CONCURRENCY` | `complete` | Future-proofs you for Swift 6 |
| `IPHONEOS_DEPLOYMENT_TARGET` | `18.0` (or `26.0` for SpeechAnalyzer-only) | See §7 for the speech fallback story |
| `OTHER_SWIFT_FLAGS` (Release) | `-O -whole-module-optimization` | Default but verify; MLX inference is hot-loop dominated |

### 5.4 App lifecycle gotchas

- **Don't load the model in `App.init`.** SwiftUI calls `init` early, before SwiftUI scenes attach; throwing there crashes silently. Load on first `.task` of the root view.
- **Don't run inference on the main thread.** The actor handles this for you. Any `ProgressView` updates from a streaming closure must `await MainActor.run { ... }` if you're not already in `.task`.
- **`MLX.GPU.set(cacheLimit:)` must be called *before* the first MLX op.** Put it in your engine's `load()`, not in app startup. Calling it once is enough; it's process-wide.

---

## 6. SwiftUI surface (L2)

The view hierarchy for v1:

```
RootView
├── ChatView (default tab)
│   ├── MessageListView
│   │   └── MessageBubble (role-styled)
│   ├── StatsBar (optional, shows tok/s + memory)
│   └── ComposerView
│       ├── TextField
│       └── VoiceInputButton  →  VoiceInputBar (sheet)
├── ModelPickerView (settings sheet)
│   └── ModelRow (descriptor + downloaded? + size)
└── ServerControlView (settings sheet)
    └── Local HTTP toggle, port, requests log
```

### 6.1 The view model

```swift
// MLXChat/ChatViewModel.swift
import Foundation
import MLXChatCore
import MLXChatEngineMLX
import MLXChatEngineLlama
import Observation

@MainActor
@Observable
public final class ChatViewModel {
    public var messages: [ChatMessage] = [
        .init(role: .system, content: "You are a helpful, concise assistant.")
    ]
    public var input: String = ""
    public var isGenerating: Bool = false
    public var loadProgress: Double = 0
    public var loadState: LoadState = .idle
    public var stats: EngineStats?
    public var selectedModel: ModelDescriptor = ModelCatalog.default

    public enum LoadState: Equatable {
        case idle
        case downloading(Double)
        case loading
        case ready
        case failed(String)
    }

    private var engine: (any LLMEngine)?

    public init() {}

    public func ensureLoaded() async {
        if case .ready = loadState, engine?.descriptor == selectedModel { return }
        loadState = .loading
        let new: any LLMEngine = {
            switch selectedModel.backend {
            case .mlx:      return MLXEngine(descriptor: selectedModel)
            case .llamaCpp: return LlamaEngine(descriptor: selectedModel)
            }
        }()
        do {
            try await new.load { [weak self] p in
                Task { @MainActor in self?.loadState = .downloading(p) }
            }
            self.engine = new
            self.loadState = .ready
        } catch {
            self.loadState = .failed(String(describing: error))
        }
    }

    public func switchModel(to descriptor: ModelDescriptor) async {
        guard descriptor != selectedModel else { return }
        await engine?.unload()
        engine = nil
        selectedModel = descriptor
        loadState = .idle
        await ensureLoaded()
    }

    public func send() async {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isGenerating, let engine else { return }

        messages.append(.init(role: .user, content: trimmed))
        input = ""
        var assistant = ChatMessage(role: .assistant, content: "")
        messages.append(assistant)
        isGenerating = true
        defer { isGenerating = false }

        let stream = engine.stream(
            messages: messages.dropLast(), // omit the empty placeholder
            options: GenerateOptions()
        )
        do {
            for try await token in stream {
                if token.isFinal { break }
                assistant.content += token.text
                messages[messages.count - 1] = assistant
            }
            self.stats = await engine.lastStats
        } catch {
            assistant.content = "⚠️ \(error)"
            messages[messages.count - 1] = assistant
        }
    }

    public func cancelGeneration() {
        // The stream's onTermination cancels the underlying task.
        // Nothing else to do here in v1.
    }
}
```

### 6.2 ChatView

```swift
// MLXChat/ChatView.swift
import SwiftUI
import MLXChatCore

struct ChatView: View {
    @State private var viewModel = ChatViewModel()
    @State private var showModelPicker = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if case .ready = viewModel.loadState {
                    MessageListView(messages: viewModel.messages.filter { $0.role != .system })
                    Divider()
                    if let stats = viewModel.stats {
                        StatsBar(stats: stats)
                    }
                    ComposerView(
                        input: $viewModel.input,
                        isGenerating: viewModel.isGenerating,
                        onSend: { Task { await viewModel.send() } }
                    )
                } else {
                    LoadingView(state: viewModel.loadState)
                }
            }
            .navigationTitle(viewModel.selectedModel.displayName)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Models", systemImage: "cpu") { showModelPicker = true }
                }
            }
            .sheet(isPresented: $showModelPicker) {
                ModelPickerView(viewModel: viewModel)
            }
            .task { await viewModel.ensureLoaded() }
        }
    }
}
```

### 6.3 Streaming UX detail

The `ChatViewModel.send()` above mutates `messages[last].content` on every token. SwiftUI's diffing handles this efficiently *only* if the message struct is `Hashable` and `Identifiable` (we made it both). The visible effect is the assistant bubble growing token by token. There's no separate "typing indicator" needed — the bubble itself is the indicator.

If you want a blinking cursor while generating, append `▎` to the content while `isGenerating` and strip it on completion.

### 6.4 Model picker

```swift
struct ModelPickerView: View {
    let viewModel: ChatViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                ForEach(ModelCatalog.all) { model in
                    Button {
                        Task {
                            await viewModel.switchModel(to: model)
                            dismiss()
                        }
                    } label: {
                        ModelRow(
                            descriptor: model,
                            isSelected: model == viewModel.selectedModel
                        )
                    }
                    .buttonStyle(.plain)
                }
            } header: {
                Text("Selecting a different model unloads the current one and downloads the new one if needed. ~10s on a fast connection for an already-cached model.")
            }
            .navigationTitle("Models")
            .toolbar { ToolbarItem(placement: .topBarTrailing) { Button("Done") { dismiss() } } }
        }
    }
}
```

---

## 7. Voice input (Speech.framework)

Two paths, gated by deployment target.

### 7.1 iPadOS 26+ — `SpeechAnalyzer` (preferred)

> Apple's iOS 26 SpeechAnalyzer + SpeechTranscriber: on-device, long-form, low-latency, distant-mic capable, model lives outside your app's memory space (no impact on app size or memory budget).

```swift
// MLXChat/Voice/SpeechAnalyzerDictation.swift
import Foundation
import Speech
import AVFoundation

@available(iOS 26.0, *)
@MainActor
final class SpeechAnalyzerDictation: ObservableObject {
    @Published var transcript: String = ""
    @Published var isListening = false

    private var analyzer: SpeechAnalyzer?
    private var transcriber: SpeechTranscriber?
    private var inputBuilder: AsyncStream<AnalyzerInput>.Continuation?
    private var format: AVAudioFormat?
    private let audioEngine = AVAudioEngine()
    private let converter = BufferConverter() // helper, see Apple's sample

    func start() async throws {
        guard await requestAuthorization() else { return }

        let transcriber = SpeechTranscriber(
            locale: .current,
            preset: .progressiveLiveTranscription
        )
        try await ensureModel(for: transcriber, locale: .current)

        let analyzer = SpeechAnalyzer(modules: [transcriber])
        self.analyzer = analyzer
        self.transcriber = transcriber
        self.format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber])

        let (inputSequence, builder) = AsyncStream<AnalyzerInput>.makeStream()
        self.inputBuilder = builder

        Task { @MainActor in
            for try await result in transcriber.results {
                let text = String(result.text.characters)
                self.transcript = text
            }
        }

        try await analyzer.start(inputSequence: inputSequence)
        try startMic()
        isListening = true
    }

    func stop() async {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        inputBuilder?.finish()
        try? await analyzer?.finalizeAndFinish(through: .latest)
        analyzer = nil
        transcriber = nil
        isListening = false
    }

    private func startMic() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement, options: [.allowBluetooth])
        try session.setActive(true, options: .notifyOthersOnDeactivation)

        let inputNode = audioEngine.inputNode
        let nativeFormat = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: nativeFormat) { [weak self] buffer, _ in
            guard let self, let format = self.format, let builder = self.inputBuilder else { return }
            do {
                let converted = try self.converter.convertBuffer(buffer, to: format)
                builder.yield(AnalyzerInput(buffer: converted))
            } catch {
                // Drop buffer on conversion error
            }
        }
        audioEngine.prepare()
        try audioEngine.start()
    }

    private func requestAuthorization() async -> Bool {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status == .authorized)
            }
        }
    }

    private func ensureModel(for transcriber: SpeechTranscriber, locale: Locale) async throws {
        let supported = await SpeechTranscriber.supportedLocales.contains(locale)
        guard supported else { throw NSError(domain: "Speech", code: 1) }
        if await SpeechTranscriber.installedLocales.contains(locale) { return }
        let request = AssetInventory.assetInstallationRequest(supporting: [transcriber])
        try await request?.downloadAndInstall()
    }
}
```

### 7.2 iPadOS 18–25 fallback — `SFSpeechRecognizer`

Standard `SFSpeechAudioBufferRecognitionRequest` + `AVAudioEngine` flow. Wrap behind the same `Dictation` protocol so the view doesn't care:

```swift
protocol Dictation: AnyObject {
    var transcript: String { get }
    var isListening: Bool { get }
    func start() async throws
    func stop() async
}
```

Pick one in a factory based on `if #available(iOS 26, *)`.

### 7.3 Voice → chat hand-off

The `VoiceInputBar` watches `dictation.transcript` and writes it to `ChatViewModel.input`. On `stop()`, if the transcript is non-empty, it calls `viewModel.send()`. Wire-up is intentionally trivial — the abstraction does the work.

---

## 8. Local OpenAI-compatible HTTP server (L3)

Embedded inside the iPad app. Bound to `127.0.0.1` only by default; toggleable to `0.0.0.0` for LAN testing (gated by a setting and a banner warning).

### 8.1 `MLXChatServer/Package.swift`

```swift
// swift-tools-version: 6.0
import PackageDescription
let package = Package(
    name: "MLXChatServer",
    platforms: [.iOS(.v18), .macOS(.v14)],
    products: [.library(name: "MLXChatServer", targets: ["MLXChatServer"])],
    dependencies: [
        .package(path: "../MLXChatCore"),
        .package(url: "https://github.com/hummingbird-project/hummingbird", from: "2.0.0"),
    ],
    targets: [
        .target(name: "MLXChatServer", dependencies: [
            .product(name: "MLXChatCore", package: "MLXChatCore"),
            .product(name: "Hummingbird", package: "hummingbird"),
        ])
    ],
    swiftLanguageModes: [.v6]
)
```

### 8.2 The server

```swift
// Sources/MLXChatServer/Server.swift
import Foundation
import Hummingbird
import MLXChatCore

public actor ChatServer {
    private var app: Application<RouterResponder<BasicRequestContext>>?
    private let engineProvider: @Sendable () async -> any LLMEngine

    public init(engineProvider: @Sendable @escaping () async -> any LLMEngine) {
        self.engineProvider = engineProvider
    }

    public func start(host: String = "127.0.0.1", port: Int = 8765) async throws {
        let router = Router()
        router.get("/healthz") { _, _ in "ok" }
        router.get("/v1/models") { _, _ in
            ModelsResponse(data: ModelCatalog.all.map(ModelObject.init))
        }
        router.post("/v1/chat/completions") { [engineProvider] req, ctx in
            let body = try await req.decode(as: ChatCompletionRequest.self, context: ctx)
            let engine = await engineProvider()
            let messages = body.messages.map { ChatMessage(role: .init(rawValue: $0.role) ?? .user, content: $0.content) }
            let opts = GenerateOptions()

            if body.stream == true {
                return Response(status: .ok, headers: ["content-type": "text/event-stream"]) { writer in
                    let stream = engine.stream(messages: messages, options: opts)
                    for try await token in stream {
                        let chunk = ChatCompletionChunk(content: token.text, isFinal: token.isFinal)
                        let data = try JSONEncoder().encode(chunk)
                        try await writer.write(.init("data: \(String(data: data, encoding: .utf8)!)\n\n"))
                    }
                    try await writer.write(.init("data: [DONE]\n\n"))
                }
            } else {
                let (text, _) = try await engine.respond(messages: messages, options: opts)
                return ChatCompletionResponse(content: text)
            }
        }

        var app = Application(
            router: router,
            configuration: .init(address: .hostname(host, port: port))
        )
        try await app.runService()
        self.app = app
    }

    public func stop() async {
        // Hummingbird 2 clean shutdown
    }
}

// MARK: - DTOs (Data Transfer Objects)

struct ChatCompletionRequest: Codable {
    struct Msg: Codable { let role: String; let content: String }
    let model: String
    let messages: [Msg]
    let stream: Bool?
}
struct ChatCompletionResponse: Codable, ResponseEncodable {
    let content: String
}
struct ChatCompletionChunk: Codable { let content: String; let isFinal: Bool }
struct ModelObject: Codable {
    let id: String
    let owned_by: String
    init(_ d: ModelDescriptor) { self.id = d.huggingFaceRepo; self.owned_by = "mlxchat" }
}
struct ModelsResponse: Codable, ResponseEncodable { let data: [ModelObject] }
```

### 8.3 When to start it

- Default: server is **off** at app launch.
- Settings toggle: `Local Server → On`. Starts on `127.0.0.1:8765`.
- LAN toggle (separate, gated, with red banner): switches to `0.0.0.0:8765`.
- iPadOS will not allow truly background HTTP serving for sustained periods — the server stops when the app is backgrounded for more than ~30 seconds. Document this in the settings sheet.

### 8.4 Test the server from your Mac

```bash
# On the iPad, find its local IP under Settings → Wi-Fi → (i) → IP Address
curl http://192.168.1.42:8765/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"hello"}]}'
```

---

## 9. RAG hook stub (`MLXChatRAG`)

A clean abstraction so you can wire in your existing Basic Memory MCP or a simple BM25 index later without touching the rest of the app.

```swift
// Sources/MLXChatRAG/Retriever.swift
import Foundation
import MLXChatCore

public protocol Retriever: Actor {
    func retrieve(query: String, k: Int) async throws -> [RetrievedChunk]
}

public struct RetrievedChunk: Sendable, Codable, Identifiable {
    public let id: String
    public let content: String
    public let score: Float
    public let source: String?
}

public actor StubRetriever: Retriever {
    public init() {}
    public func retrieve(query: String, k: Int) async throws -> [RetrievedChunk] {
        return [] // wire up your vector store / Basic Memory MCP here
    }
}

/// Augments a user message with retrieved context before generation.
public extension Array where Element == ChatMessage {
    func withRetrievedContext(_ chunks: [RetrievedChunk]) -> [ChatMessage] {
        guard !chunks.isEmpty else { return self }
        let ctx = chunks.map { "- \($0.content)" }.joined(separator: "\n")
        let augmented = "Use this context if relevant:\n\(ctx)\n\n" +
            (last(where: { $0.role == .user })?.content ?? "")
        var copy = self
        if let idx = copy.lastIndex(where: { $0.role == .user }) {
            copy[idx].content = augmented
        }
        return copy
    }
}
```

In `ChatViewModel.send()`, before calling `engine.stream(...)`, insert:

```swift
let retriever: any Retriever = StubRetriever()
let chunks = try await retriever.retrieve(query: trimmed, k: 5)
let augmented = messages.dropLast().withRetrievedContext(chunks)
let stream = engine.stream(messages: augmented, options: GenerateOptions())
```

---

## 10. Memory pressure handling (the part everyone forgets)

```swift
// MLXChat/MemoryGovernor.swift
import Foundation
import os

@MainActor
final class MemoryGovernor: ObservableObject {
    @Published var warningCount = 0
    @Published var available: UInt64 = 0
    private var observer: NSObjectProtocol?
    private var timer: Timer?

    func start() {
        observer = NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.handleWarning()
            }
        }
        timer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.available = UInt64(os_proc_available_memory())
            }
        }
    }

    private func handleWarning() {
        warningCount += 1
        // 1. Clear MLX's GPU cache first — usually enough to recover.
        MLX.GPU.clearCache()
        // 2. If warnings keep arriving, prompt the user to switch to a smaller model.
        if warningCount >= 3 {
            // Post a notification the ChatViewModel listens to.
            NotificationCenter.default.post(name: .mlxChatNeedsSmallerModel, object: nil)
        }
    }
}
extension Notification.Name {
    static let mlxChatNeedsSmallerModel = Notification.Name("mlxChatNeedsSmallerModel")
}
```

Display `available` in the StatsBar — it's the single most useful debug signal you'll have.

---

## 11. Step-by-step build order

This is the order I'd actually do this in — each step verifiable in isolation before the next:

| # | Task | Verify |
|---|---|---|
| 1 | Create Xcode workspace + iOS app target + the four SPM packages | App builds empty |
| 2 | Add SPM dependencies to `MLXChatCore` Package.swift | Resolve succeeds |
| 3 | Add entitlements + Info.plist keys | Build succeeds with `Increased Memory Limit` shown in Signing & Capabilities |
| 4 | Implement `LLMEngine` protocol and types (§3.1, §3.2) | Compiles |
| 5 | Implement `MLXEngine` (§3.4) | Build, do not run |
| 6 | Wire up minimal `ChatViewModel` and a smoke-test view that just calls `engine.respond("hello")` | **Run on real iPad.** First run downloads ~5GB; expect 1–2 min on good Wi-Fi. Verify text comes back. **This is your "hello world".** |
| 7 | Add streaming UI (§6) | Tokens appear progressively |
| 8 | Add `LlamaEngine` + GGUF model in catalog | Switch backends in picker, verify GGUF download + run |
| 9 | Add `MemoryGovernor` + StatsBar | Watch `os_proc_available_memory()` count down during long generations |
| 10 | Add `SpeechAnalyzer` voice input (with `SFSpeechRecognizer` fallback) | Tap mic, dictate, send |
| 11 | Add `ModelPickerView` + hot-swap | Switch from Qwen3-8B to Qwen3-4B and back; verify memory drops/recovers |
| 12 | Add Hummingbird HTTP server | `curl localhost:8765/healthz` from iPad's terminal app, then `curl http://<ipad-ip>:8765/v1/chat/completions` from your Mac |
| 13 | Wire `StubRetriever` into the send path | Empty for now; integration with Basic Memory MCP is a separate work item |
| 14 | TestFlight build, profile with Instruments (Memory + Time Profiler) | Sustain a 2k-token generation without jetsam |
| 15 | App Store submission prep | App privacy answers (microphone, no tracking), screenshots |

---

## 12. Performance expectations (what to verify with Instruments)

Based on community benchmarks for Qwen3-8B at 4-bit on M-series silicon. The iPad M4 GPU is a 10-core (binned vs M4 Pro), with **120 GB/s** memory bandwidth — substantially less than M4 Max (~546 GB/s) but better than the M4 base in MacBook Air.

| Metric | Expected | Why |
|---|---|---|
| Cold model load | 8–15 s | Includes safetensors → Metal upload |
| Warm model load (already in cache) | 2–4 s | Memory map only |
| Prompt processing (1k tokens) | ~80–120 t/s | Bandwidth-bound |
| Generation speed | **22–32 t/s** for Qwen3-8B 4-bit | Memory-bandwidth-bound, M4 base GPU |
| Generation speed (Qwen3-4B 4-bit) | **40–55 t/s** | Smaller model, less bandwidth needed |
| Time to first token | 0.4–0.8 s for 100-token prompt | Prompt processing dominates |
| Resident memory peak | ~9–10 GB during 32k context | Fits within ~12 GB ceiling |

If you observe < 15 t/s on Qwen3-8B-4bit, something is wrong. Likely culprits, in order:
1. Built in Debug configuration (try Release).
2. Buffer cache cap set too low (try 512 MB instead of 256 MB).
3. Background task is pegging the GPU (close other apps).
4. Thermal throttling — iPad has gotten hot. Let it cool, retest.

---

## 13. Distribution plan (your 2/3/4 answer)

### 13.1 Personal (paid Apple Developer account)

Standard signing. The `Increased Memory Limit` entitlement is granted automatically — no special request required.

### 13.2 TestFlight (friends/family)

Same entitlements. Two extra steps:
- App Store Connect → distribute build → fill out **Encryption Compliance** (`ITSAppUsesNonExemptEncryption = NO` in Info.plist if you only use HTTPS via standard libs and the local server isn't using custom crypto).
- Privacy nutrition label: declare microphone, declare no tracking, declare no data collection (since everything is on-device).

### 13.3 App Store

Watch out for:
- **App size.** The model is *not* bundled (good). Apple's review team will install on a test device with no Wi-Fi history; the first-run download must work over LTE or a guest Wi-Fi without auth wall. Test this.
- **App size 2.** Even without the model, MLX ships a lot of Metal kernels. Expect ~80–150 MB IPA.
- **Review notes.** Tell Apple: "On first launch, the app downloads a ~4.7 GB language model from Hugging Face and runs all inference locally. No data leaves the device. The model is required for the app's primary function."
- **Privacy.** Declare microphone usage even if voice is optional. Reviewers test it.
- **Background audio mode.** If you keep dictation alive across backgrounding, document why. Otherwise, drop the `audio` background mode and stop dictation in `scenePhase == .background`.

---

## 14. Risks & open questions

| Risk | Likelihood | Mitigation |
|---|---|---|
| `mlx-swift-lm` 3.x API drifts from this manual before you start | Medium | Pin to `3.31.3` exactly via `.exact("3.31.3")` in Package.swift; bump when you have time |
| Apple rejects the app for "downloads executable code" | Low — model weights aren't code | Cite Hugging Face precedent (LM Studio, MLX Chat are on the App Store) |
| iPadOS 26 SpeechAnalyzer model assets fail to install | Low | Detect, fall back to `SFSpeechRecognizer` cleanly |
| TurboQuant KV cache compression lands in `mlx-swift` upstream | Medium | If/when it does, opt in via a build flag; gives 15–19% decode speedup at long context. Not in scope for v1 |
| You bought the 256/512 GB iPad and only have 8 GB | Confirmed not the case (1TB+ tier) | n/a |

### Open questions you should think about before coding

1. **System prompt persistence.** Should each model carry its own default system prompt, or should it be user-edited and stored once per app? I default to per-app + per-model override, but you may want per-conversation.
2. **Conversation persistence.** Not in v1 scope as written. Want SwiftData or just `@AppStorage`-backed JSON? SwiftData is overkill for v1; `@AppStorage` JSON is fine until you hit 100+ conversations.
3. **Thinking blocks.** Qwen3 emits `<think>...</think>` chains. Render collapsed by default? I'd say yes. Trivial regex split in the message bubble.
4. **Tool calling.** `LocalLLMClient` flagged tool-calling as experimental. Worth keeping out of v1.
5. **Model deletion.** Where in the model picker should "delete cached weights" live? Long-press? Settings? I'd put it in a swipe action on each `ModelRow`.

---

## 15. CLAUDE_NOTES.md entries to log after this build

Per your iteration protocol — when you're done, log these:

- ✅ SUCCESS: `mlx-swift-lm` 3.x is the new home for LLM/VLM code (replaces `mlx-swift-examples` libraries). `LocalLLMClient` already migrated as of v0.4.6.
- ✅ SUCCESS: `MLX.GPU.set(cacheLimit: 256MB)` is the empirical sweet spot for 8B models on 16GB iPads.
- ⚠️ NEEDS IMPROVEMENT: `EngineStats.peakResidentMemoryBytes` is sampled, not a true peak. Wire up a sampling timer during generation if you need accurate peaks.
- ❌ AVOID: Loading the model in `App.init`. Always load in the root view's `.task`.
- ❌ AVOID: Running MLX in the iOS Simulator. Build UI there, run inference on hardware only.

---

## Appendix A — Quick reference: where each file goes

```
MLXChat.xcworkspace/
├── MLXChat/                                   # iOS app
│   ├── App/
│   │   ├── MLXChatApp.swift
│   │   └── MLXChat.entitlements
│   ├── Chat/
│   │   ├── ChatView.swift
│   │   ├── ChatViewModel.swift
│   │   ├── MessageListView.swift
│   │   ├── MessageBubble.swift
│   │   ├── ComposerView.swift
│   │   └── StatsBar.swift
│   ├── Models/
│   │   ├── ModelPickerView.swift
│   │   └── ModelRow.swift
│   ├── Voice/
│   │   ├── Dictation.swift                    # Protocol
│   │   ├── SpeechAnalyzerDictation.swift      # iPadOS 26+
│   │   ├── SFSpeechDictation.swift            # iPadOS 18 fallback
│   │   ├── BufferConverter.swift              # Apple sample helper
│   │   └── VoiceInputBar.swift
│   ├── Server/
│   │   └── ServerControlView.swift
│   └── System/
│       └── MemoryGovernor.swift
└── Packages/
    ├── MLXChatCore/
    ├── MLXChatEngineMLX/  (target inside MLXChatCore SPM)
    ├── MLXChatEngineLlama/(target inside MLXChatCore SPM)
    ├── MLXChatServer/
    └── MLXChatRAG/
```

---

## Appendix B — Source verification (so you can audit my claims)

| Claim | Source | Date |
|---|---|---|
| iPad Pro 11" M4 1TB+ has 16 GB RAM, 120 GB/s bandwidth | apple.com/ipad/compare; support.apple.com/en-us/119892 | Active |
| Increased Memory Limit ceiling ~12 GB on 16 GB iPad | developer.apple.com/forums/thread/724195; thread/777370 | March 2025 |
| `mlx-swift-lm` 3.31.3 is current, replaces examples libraries | swiftpackageindex.com/ml-explore/mlx-swift-lm | April 17, 2026 |
| `LocalLLMClient` 0.4.6 with mlx-swift-lm migration | github.com/tattn/LocalLLMClient releases; PR #85 | November 27, 2025 + April 2026 |
| MLX is 15–30% faster than llama.cpp on Apple Silicon | willitrunai.com/blog/mlx-vs-ollama-apple-silicon-benchmarks | 1 week ago |
| Qwen3-8B-4bit-AWQ exists at `mlx-community/Qwen3-8B-4bit-AWQ` | huggingface.co/mlx-community/Qwen3-8B-4bit-AWQ | Active |
| `SpeechAnalyzer` API design (modular, async, on-device) | developer.apple.com WWDC25-277; Speech framework docs | June 2025 |
| `LLMModelFactory.shared.loadContainer` is the load API | github.com/ml-explore/mlx-swift-examples releases | Migration notes |
| Increased Memory Limit + Extended Virtual Addressing entitlements | developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit | Active |

— end of manual —
