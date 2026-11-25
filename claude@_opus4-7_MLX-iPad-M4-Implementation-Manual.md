# MLX iOS Wrapper for iPad Pro M4 (16GB) — Implementation Manual

**Author:** Claude (research + plan)
**Owner:** Luke (zautke)
**Target device:** iPad Pro 11" M4, 1TB/2TB tier — 16GB unified memory, 10-core CPU, 10-core GPU, 16-core Neural Engine, 120GB/s memory bandwidth ("presto", UDID `00008132-001C383636B9001C`, iPadOS 27.2)
**Target OS:** iPadOS 26+ (deployment target 26.0; SpeechAnalyzer requires 26.0; iPadOS 27 APIs noted where relevant)
**Date:** 2026-04-26 → **v2 refresh 2026-11-25** (SOTA pass: 3 parallel research agents; sources inline)

---

## 0. TL;DR — what we're building

A SwiftUI iPad app, **`MLXChat`**, that:

1. Wraps a **dual-backend** local LLM (Large Language Model) runtime: Apple **MLX (Machine Learning eXchange)** for native MLX-format models, and **`llama.cpp`** (via LocalLLMClient) for GGUF (GPT-Generated Unified Format) models.
2. Exposes a **single Swift `LLMEngine` protocol** so the rest of the app never sees the backend choice.
3. Ships with **`mlx-community/Qwen3.5-9B-MLX-4bit`** as the default model (~5.95GB on disk; ~6.5–7GB resident; fits the iOS Increased Memory Limit ceiling with headroom). *(Changed 2026-11-25: Qwen3-8B-4bit-AWQ is superseded — 215 downloads vs 23k for Qwen3.5-9B; see §4.)*
4. Supports **streaming token-by-token UI**, **multi-turn chat with system prompt**, **on-device voice dictation** (`SpeechAnalyzer` + `SpeechTranscriber` on iPadOS 26+, `SFSpeechRecognizer` fallback), **runtime model picker / hot-swap**, and a **stub RAG (Retrieval-Augmented Generation) hook**.
5. Ships an **embedded OpenAI-compatible HTTP server** (Hummingbird 2) bound to `127.0.0.1` so other on-device apps can hit `/v1/chat/completions`.

**The "generic app interface"** resolves to three concentric layers:

| Layer | What | Lives in |
|---|---|---|
| **L1 — Swift API** | `LLMEngine` protocol (`func respond(...)`, `func stream(...) -> AsyncThrowingStream<GenerationToken, Error>`) | `MLXChatCore` SPM target |
| **L2 — SwiftUI surface** | `ChatView`, `ModelPickerView`, `VoiceInputBar`, `MessageBubble` | `MLXChat` app target |
| **L3 — HTTP surface** | `POST /v1/chat/completions` (OpenAI-compatible), `GET /v1/models`, `GET /healthz` | `MLXChatServer` SPM target |

---

## 1. Hardware reality check (read this before you write any code)

### 1.1 RAM ceiling

- iPad Pro 11" M4 1TB/2TB: **16 GB physical**, **120 GB/s** memory bandwidth.
- Without `com.apple.developer.kernel.increased-memory-limit`, expect jetsam around **~5 GB**. With it, the empirically observed ceiling on 16 GB iPads is **~12 GB**. Apple does not document a number; call `os_proc_available_memory()` at runtime and treat it as authoritative.
- **iPadOS 26.4–26.5 regression:** per-app limits on **8 GB** iPads were cut from ~6 GB to ~3 GB (Apple Forums thread 826325; FB22883960). DTS says fixed as of iPadOS 26.6 beta 5. 16 GB M4 iPad largely unaffected — **verify on-device anyway**.
- iPadOS 18 added `vm-compressor-space-shortage` jetsam reason. Spiky access patterns (random KV-cache pages) trigger it faster than streaming reads.
- The simulator has no Metal MTLGPUFamily that MLX accepts. **All MLX runs must be on a real device.** Build UI in the simulator; run inference on hardware.
- **iPadOS 27 note:** Apple Foundation Models (AFM 3 Core) now ships on-device; M4 iPads run AFM 3 Core (see §14). MLX remains the path for custom weights and long context.

### 1.2 Working memory budget on a 16 GB iPad

| Bucket | Reserve | Why |
|---|---|---|
| Model weights (Qwen3.5-9B-4bit) | ~6.5 GB resident | 5.95 GB on disk + dequant overhead |
| KV cache (16–32k context) | ~1.5–3.0 GB | TurboQuant KV (see §1.4) shrinks this ~2.7× past ~16k context |
| MLX GPU buffer cache | 256 MB | `MLX.GPU.set(cacheLimit:)` |
| Tokenizer + chat template | ~150 MB | swift-transformers overhead |
| App + SwiftUI + SpeechAnalyzer | ~500 MB | Realistic baseline |
| Hummingbird HTTP server | ~50 MB | Idle |
| Headroom for jetsam safety | **~2.5 GB** | Don't skip this |
| **Total** | **~11–12 GB** | Fits inside the entitlement ceiling |

If you hot-swap to a 4B model, you reclaim ~3 GB and can extend context to 64k+.

### 1.3 Why Qwen3.5-9B-MLX-4bit specifically (updated 2026-11-25)

| Candidate | On-disk | Quality | Notes |
|---|---|---|---|
| **Qwen3.5-9B-MLX-4bit** ✅ default | 5.95 GB | MMLU-Pro 82.5, GPQA-D 81.7, IFEval 91.5 | Apache-2.0, multimodal, 262K ctx, thinking-capable, ~20 tok/s on M4 iPad |
| Qwen3.5-4B-MLX-4bit | 3.03 GB | Strong for size | Fast tier, ~35 tok/s |
| gemma-4-e4b-it-4bit | 5.15 GB | MMLU-Pro 69.4 | PLE architecture, audio+vision, 128K ctx, Apache-2.0 |
| gemma-4-e2b-it-4bit | 3.55 GB | MMLU-Pro 60.0 | Ultra-light, ~30–35 tok/s |
| Qwen3-4B-Instruct-2507-4bit | 2.26 GB | legacy but huge adoption (128k dl) | Non-thinking |
| Qwen3-8B-4bit | 4.61 GB | legacy | Superseded by Qwen3.5-9B |
| Qwen3-8B-4bit-AWQ | 4.69 GB | — | **Deprecated**: 215 downloads; MLX never needed AWQ |
| gpt-oss-20b-MXFP4-Q8 | ~12 GB | strong | Tight on 16 GB; needs recent MLX |
| Qwen3.5/3.6-35B-A3B, Qwen3-Next-80B | 20–45 GB | — | **Do not fit.** MoE "small active params" does not rescue 16 GB |

All Qwen3.5 sizes are thinking-capable; disable per-request with `chat_template_kwarg enable_thinking=false` (no `/no_think` soft switch in Qwen3.5).

### 1.4 KV cache quantization — TurboQuant (shipped 2026-07-20; **blocked by current pin**)

`mlx-swift-lm` merged TurboQuant KV-cache compression (PR #232). Opt in via `GenerateParameters.kvScheme`:

- `turbo8v3` — recommended default when enabled: 8-bit affine K + turbo V, ~2.72× KV reduction, KLD ≈ affine8.
- `turbo0v2/v3/v4` — fp16 K + turbo V (~1.58×), quality play.
- `affine4/affine8` — 1.88×.
- Boundary layers (first/last 2) stay 8-bit; rotating/sliding-window layers stay fp16; prompt-cache save/restore works.

> **Availability:** TurboQuant requires mlx-swift-lm ≥ 3.32, which conflicts with LocalLLMClient 0.5.0's swift-syntax 600.x pin. Not usable in v1 of this build; revisit when LocalLLMClient bumps. Rules of thumb (for when it is available): **below ~8–16k context, keep default fp16/affine8 KV — don't bother.** Above ~16k context on 16 GB, `turbo8v3` is what makes 32–64k contexts viable. Decode costs ~0.7–0.8× vs fp16. (Not yet in mlx-lm Python at parity; Swift only, needs mlx-swift-lm ≥ the July 2026 merge — satisfied by 3.33.0.)

### 1.5 The MLX vs Ollama/GGUF reality

MLX is 15–30% faster than llama.cpp on Apple Silicon at the same quantization, with ~10% lower memory overhead. Ollama uses GGUF; MLX uses MLX-format `.safetensors`. Same model family either way. Default backend is MLX; GGUF via LocalLLMClient (0.5.0, llama.cpp b8851) is the fallback path. Best GGUF fallback: `unsloth/Qwen3.5-9B-GGUF` Q4_K_M (5.68 GB).

---

## 2. Package architecture

One Xcode workspace, one local SPM package with five products, plus the iOS app target:

```
MLXChat.xcworkspace/
├── MLXChat.xcodeproj                # iOS app target (SwiftUI) — generated by XcodeGen (project.yml)
└── Packages/
    └── MLXChatPackages/             # single local package, 5 products
        ├── Package.swift
        └── Sources/
            ├── MLXChatCore/         # L1 — protocols, types, registry, factory
            ├── MLXChatEngineMLX/    # MLX backend
            ├── MLXChatEngineLlama/  # llama.cpp / GGUF backend
            ├── MLXChatServer/       # L3 — Hummingbird HTTP
            └── MLXChatRAG/          # Retriever stub
```

### 2.1 Top-level dependencies (verified 2026-11-25)

| Dependency | Version | Source | Notes |
|---|---|---|---|
| `mlx-swift` | `0.31.3` (pinned) | github.com/ml-explore/mlx-swift | Latest is 0.33.0 (2026-10-27); pinned down — see note |
| **`mlx-swift-lm`** | **`3.31.3`** (pinned) | github.com/ml-explore/mlx-swift-lm | Latest is 3.33.0 (2026-11-17); pinned down — see note |
| `swift-huggingface` | `1.0.1` | github.com/huggingface/swift-huggingface | 2026-11-18; `HuggingFace.HubClient` |
| `swift-transformers` | `1.4.0` | github.com/huggingface/swift-transformers | 2026-11-02; provides `Tokenizers` |
| `swift-numerics` | `1.1.1` | github.com/apple/swift-numerics | transitive |
| `LocalLLMClient` | `0.5.0` | github.com/tattn/LocalLLMClient | 2026-04-25; GGUF path; API unstable — pin exact |
| `hummingbird` | `2.29.0` | github.com/hummingbird-project/hummingbird | 2026-11-19; no 3.x; Swift 6.2+ |

> **Breaking change vs the April 2026 manual:** mlx-swift-lm 3.x is decoupled from tokenizer/downloader packages. You must add `swift-huggingface` + `swift-transformers` yourself and link `MLXHuggingFace`. `import Hub` → `import HuggingFace`; `HubApi` → `HuggingFace.HubClient`; `loadContainer(configuration:)` → `loadContainer(from:using:configuration:)` or the `#huggingFaceLoadModelContainer` macro. See §3.4.
>
> **Version pin (2026-11-25, verified by resolution failure):** LocalLLMClient 0.5.0 depends on swift-syntax `600.0.0..<601.0.0`; mlx-swift-lm ≥ 3.31.4 (and 3.32.x/3.33.0) requires swift-syntax `602.0.0..<604.0.0`. They cannot co-resolve. We pin **mlx-swift-lm 3.31.3 + mlx-swift 0.31.3**, the newest pair compatible with LocalLLMClient 0.5.0. Consequences: no TurboQuant KV (`kvScheme`), no `MLXFoundationModels` bridge, no `.reasoning` Generation case until LocalLLMClient bumps swift-syntax (watch tattn/LocalLLMClient releases). Everything else (Swift 6, `#huggingFaceLoadModelContainer`, `generate(input:parameters:context:)` AsyncStream API) is identical.

### 2.2 `Packages/MLXChatPackages/Package.swift`

```swift
// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "MLXChatPackages",
    platforms: [
        .iOS(.v26),
        .macOS(.v26),   // dev/catalyst convenience
    ],
    products: [
        .library(name: "MLXChatCore", targets: ["MLXChatCore"]),
        .library(name: "MLXChatEngineMLX", targets: ["MLXChatEngineMLX"]),
        .library(name: "MLXChatEngineLlama", targets: ["MLXChatEngineLlama"]),
        .library(name: "MLXChatServer", targets: ["MLXChatServer"]),
        .library(name: "MLXChatRAG", targets: ["MLXChatRAG"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.3")),   // pinned: LocalLLMClient 0.5.0 → swift-syntax 600.x
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "3.31.3")), // pinned: same reason
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "1.0.1"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.4.0"),
        .package(url: "https://github.com/tattn/LocalLLMClient", exact: "0.5.0"),
        .package(url: "https://github.com/hummingbird-project/hummingbird", from: "2.29.0"),
    ],
    targets: [
        .target(name: "MLXChatCore", dependencies: [
            .product(name: "Tokenizers", package: "swift-transformers"),
        ]),
        .target(name: "MLXChatEngineMLX", dependencies: [
            "MLXChatCore",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXLLM", package: "mlx-swift-lm"),
            .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
            .product(name: "HuggingFace", package: "swift-huggingface"),
            .product(name: "Tokenizers", package: "swift-transformers"),
        ]),
        .target(name: "MLXChatEngineLlama", dependencies: [
            "MLXChatCore",
            .product(name: "LocalLLMClient", package: "LocalLLMClient"),
            .product(name: "LocalLLMClientLlama", package: "LocalLLMClient"),
            .product(name: "LocalLLMClientUtility", package: "LocalLLMClient"),
        ]),
        .target(name: "MLXChatServer", dependencies: [
            "MLXChatCore",
            .product(name: "Hummingbird", package: "hummingbird"),
        ]),
        .target(name: "MLXChatRAG", dependencies: ["MLXChatCore"]),
        .testTarget(name: "MLXChatCoreTests", dependencies: ["MLXChatCore"]),
    ],
    swiftLanguageModes: [.v6]
)
```

---

## 3. The generic app interface (L1 — `MLXChatCore`)

Unchanged in spirit from v1; updated for Swift 6.2 approachable concurrency (default `@MainActor` isolation in new app code; engines stay `nonisolated`/actor).

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
        self.id = id; self.role = role; self.content = content; self.createdAt = createdAt
    }
}

public struct GenerateOptions: Sendable {
    public var maxTokens: Int = 1024
    public var temperature: Float = 0.7
    public var topP: Float = 0.9
    public var topK: Int = 40
    public var repetitionPenalty: Float = 1.1
    public var stopTokens: [String] = []
    /// nil = model default; set false to disable Qwen3.5 thinking mode per-request
    public var enableThinking: Bool? = nil
    /// TurboQuant KV cache scheme for long contexts ("turbo8v3"); nil = default
    public var kvScheme: String? = nil
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
    public let supportsThinking: Bool

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
    /// `progress` reports download progress 0.0...1.0.
    func load(progress: @Sendable @escaping (Double) -> Void) async throws

    /// Free model + KV cache. Idempotent.
    func unload() async

    /// Non-streaming generation.
    func respond(messages: [ChatMessage], options: GenerateOptions) async throws -> (text: String, stats: EngineStats)

    /// Streaming generation. Final token carries `isFinal: true` and empty text;
    /// stats available via `lastStats`.
    func stream(messages: [ChatMessage], options: GenerateOptions) -> AsyncThrowingStream<GenerationToken, Error>

    var lastStats: EngineStats? { get }
}
```

### 3.3 Why an `actor`

Unchanged: both MLX inference state (`ModelContainer`) and `LocalLLMClient` are not `Sendable`-safe across raw threads. An actor gives Swift 6 concurrency safety; the HTTP server, SwiftUI, and voice pipeline all hold the same reference. Token streams are `AsyncThrowingStream` so `.task { for try await … }` works directly.

### 3.4 The MLX engine implementation (API-correct for mlx-swift-lm 3.33)

Key 3.x deltas you must not get wrong:

- `LLMModelFactory.shared.loadContainer(configuration:)` **no longer compiles**. Use `loadContainer(from:using:configuration:)` or the `#huggingFaceLoadModelContainer(configuration:)` macro.
- `import HuggingFace` (not `Hub`); `HubApi` → `HuggingFace.HubClient`.
- `tokenizer.decode(tokens:)` → `tokenizer.decode(tokenIds:)`.
- `ModelConfiguration(id:)` still valid; `tokenizerId:`/`overrideTokenizer:` → `tokenizerSource:`; `preparePrompt` removed.
- `UserInput.Image/Video/Audio` are structs with a nested `Source` enum.
- Add `swift-huggingface` + `swift-transformers` yourself; link `MLXHuggingFace` for the macros.

```swift
// Sources/MLXChatEngineMLX/MLXEngine.swift
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers
import MLXChatCore

public actor MLXEngine: LLMEngine {
    public nonisolated let descriptor: ModelDescriptor
    public private(set) var isLoaded: Bool = false
    public private(set) var lastStats: EngineStats?

    private var container: ModelContainer?

    /// Buffer cache cap: 256 MB is the empirical sweet spot on 16 GB iPads.
    private static let gpuCacheLimitBytes = 256 * 1024 * 1024

    public init(descriptor: ModelDescriptor) {
        precondition(descriptor.backend == .mlx, "MLXEngine requires .mlx backend")
        self.descriptor = descriptor
    }

    public func load(progress: @Sendable @escaping (Double) -> Void) async throws {
        guard !isLoaded else { return }

        MLX.GPU.set(cacheLimit: Self.gpuCacheLimitBytes)

        let available = os_proc_available_memory()
        let required = UInt64(Double(descriptor.approximateBytesOnDisk) * 1.4)
        guard UInt64(available) >= required else {
            throw EngineError.insufficientMemory(
                availableBytes: UInt64(available),
                requiredBytes: required
            )
        }

        let configuration = ModelConfiguration(id: descriptor.huggingFaceRepo)

        do {
            self.container = try await LLMModelFactory.shared.loadContainer(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
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
        guard let container else { throw EngineError.modelNotLoaded }

        let userInput = UserInput(
            messages: messages.map { ["role": $0.role.rawValue, "content": $0.content] }
        )

        var parameters = GenerateParameters(
            temperature: options.temperature,
            topP: options.topP,
            repetitionPenalty: options.repetitionPenalty
        )
        if let kv = options.kvScheme {
            // TurboQuant: only worth it past ~16k context
            // parameters.kvScheme = .turbo8v3   // exact API per MLXLMCommon docs
            _ = kv
        }

        let startTime = ContinuousClock.now
        var promptTokenCount = 0
        var completionTokenCount = 0
        var generationStarted: ContinuousClock.Instant?

        try await container.perform { context in
            let input = try await context.processor.prepare(input: userInput)
            promptTokenCount = input.text.tokens.size

            _ = try MLXLMCommon.generate(
                input: input,
                parameters: parameters,
                context: context
            ) { tokens in
                if generationStarted == nil { generationStarted = .now }
                let newText = context.tokenizer.decode(tokenIds: tokens)
                continuation.yield(GenerationToken(text: newText,
                                                    tokenIndex: completionTokenCount,
                                                    isFinal: false))
                completionTokenCount += 1
                return completionTokenCount >= options.maxTokens ? .stop : .more
            }
        }

        let endTime = ContinuousClock.now
        let promptDuration = generationStarted.map { startTime.duration(to: $0) } ?? .zero
        let completionDuration = generationStarted.map { $0.duration(to: endTime) } ?? .zero

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

> `kvScheme` wiring: check `GenerateParameters` in the installed mlx-swift-lm 3.33.0 for the exact `kvScheme` property name before enabling; keep it behind `GenerateOptions.kvScheme` and default off. The `ChatSession` API (`streamResponse(to:)`) is an equally valid simpler path if you don't need per-token control — see the `MLXChatExample` in mlx-swift-examples.

### 3.5 The llama.cpp engine implementation

Unchanged from v1 except version pin: **LocalLLMClient 0.5.0** (2026-04-25; llama.cpp b8851; Gemma 4 supported; API explicitly unstable — pin `exact: "0.5.0"`). GGUF fallback model: `unsloth/Qwen3.5-9B-GGUF` Q4_K_M (5.68 GB).

### 3.6 Engine factory

Use the simple in-app switch (drop the `@_cdecl` idea):

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

## 4. The model registry (updated 2026-11-25)

```swift
// Sources/MLXChatCore/ModelCatalog.swift
public enum ModelCatalog {
    public static let qwen35_9B_4bit = ModelDescriptor(
        displayName: "Qwen3.5 9B (4-bit, MLX)",
        huggingFaceRepo: "mlx-community/Qwen3.5-9B-MLX-4bit",
        backend: .mlx,
        approximateBytesOnDisk: 5_950_000_000,
        recommendedContextWindow: 32_768,
        supportsThinking: true
    )
    public static let qwen35_4B_4bit = ModelDescriptor(
        displayName: "Qwen3.5 4B (4-bit, MLX)",
        huggingFaceRepo: "mlx-community/Qwen3.5-4B-MLX-4bit",
        backend: .mlx,
        approximateBytesOnDisk: 3_030_000_000,
        recommendedContextWindow: 32_768,
        supportsThinking: true
    )
    public static let gemma4_E4B_4bit = ModelDescriptor(
        displayName: "Gemma 4 E4B (4-bit, MLX)",
        huggingFaceRepo: "mlx-community/gemma-4-e4b-it-4bit",
        backend: .mlx,
        approximateBytesOnDisk: 5_150_000_000,
        recommendedContextWindow: 65_536,
        supportsThinking: true
    )
    public static let gemma4_E2B_4bit = ModelDescriptor(
        displayName: "Gemma 4 E2B (4-bit, MLX)",
        huggingFaceRepo: "mlx-community/gemma-4-e2b-it-4bit",
        backend: .mlx,
        approximateBytesOnDisk: 3_550_000_000,
        recommendedContextWindow: 32_768,
        supportsThinking: true
    )
    public static let qwen3_8B_4bit = ModelDescriptor(
        displayName: "Qwen3 8B (4-bit, MLX) — legacy",
        huggingFaceRepo: "mlx-community/Qwen3-8B-4bit",
        backend: .mlx,
        approximateBytesOnDisk: 4_610_000_000,
        recommendedContextWindow: 32_768,
        supportsThinking: true
    )
    public static let qwen3_4B_Instruct = ModelDescriptor(
        displayName: "Qwen3 4B Instruct 2507 (4-bit, MLX)",
        huggingFaceRepo: "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        backend: .mlx,
        approximateBytesOnDisk: 2_260_000_000,
        recommendedContextWindow: 32_768,
        supportsThinking: false
    )
    // GGUF (llama.cpp) fallback — same family as Ollama setups
    public static let qwen35_9B_GGUF = ModelDescriptor(
        displayName: "Qwen3.5 9B (Q4_K_M, GGUF)",
        huggingFaceRepo: "unsloth/Qwen3.5-9B-GGUF",
        filename: "Qwen3.5-9B-Q4_K_M.gguf",
        backend: .llamaCpp,
        approximateBytesOnDisk: 5_680_000_000,
        recommendedContextWindow: 32_768,
        supportsThinking: true
    )

    public static let all: [ModelDescriptor] = [
        qwen35_9B_4bit, qwen35_4B_4bit, gemma4_E4B_4bit, gemma4_E2B_4bit,
        qwen3_8B_4bit, qwen3_4B_Instruct, qwen35_9B_GGUF,
    ]

    public static let `default` = qwen35_9B_4bit
}
```

Notes:
- Qwen3.5 thinking is on by default; send `chat_template_kwarg enable_thinking=false` for snappy non-thinking replies.
- MoE models (Qwen3.5/3.6-35B-A3B, Qwen3-Next-80B) do **not** fit 16 GB — do not add them.
- Gemma 4 E4B/E2B support audio input — future voice-in path that skips the speech pipeline.

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
</dict>
</plist>
```

App Sandbox is implicit on iOS (no sandbox entitlement key needed). In Xcode: Signing & Capabilities → + Capability → **Increased Memory Limit**, **Extended Virtual Addressing**, **Background Modes** (`audio` for voice, `fetch` for downloads).

> **Distribution gotcha:** the memory entitlement must be present in the **distribution** profile too. Verify with `codesign -d --entitlements :- <App.app>` on the archive. Missing from distribution = jetsam at ~6 GB in TestFlight/App Store while dev builds look fine.

### 5.2 `Info.plist` keys

| Key | Value | Why |
|---|---|---|
| `NSMicrophoneUsageDescription` | `MLXChat uses your microphone for on-device voice dictation. Your voice never leaves the device.` | Required for SpeechAnalyzer (this is the *only* speech-related key it needs) |
| `NSSpeechRecognitionUsageDescription` | `MLXChat transcribes your speech entirely on-device.` | Only if shipping the SFSpeechRecognizer fallback |
| `UIBackgroundModes` | `audio`, `fetch` | Dictation through interruptions; background model download |
| `NSAppTransportSecurity` → `NSAllowsLocalNetworking` | `true` | Only if you later add LAN fetch; loopback needs nothing |
| `ITSAppUsesNonExemptEncryption` | `NO` | Standard HTTPS/TLS only — skips export-compliance questionnaire |
| `UIRequiredDeviceCapabilities` | `arm64` | Excludes simulator |
| `LSApplicationQueriesSchemes` | (omit) | Not needed |

Loopback (`127.0.0.1`) HTTP needs **no** Local Network permission (TN3179). LAN access (e.g. reaching a Mac Ollama) would require `NSLocalNetworkUsageDescription` — not in v1 scope.

### 5.3 Build settings

| Setting | Value | Why |
|---|---|---|
| `EXCLUDED_ARCHS` (iOS Simulator) | `x86_64 arm64` | MLX Metal kernels don't run in the simulator |
| `SWIFT_STRICT_CONCURRENCY` | `complete` | Swift 6 |
| `IPHONEOS_DEPLOYMENT_TARGET` | `26.0` | SpeechAnalyzer floor; bump to 27.0 to unlock AFM 3 / `MLXFoundationModels` |
| `SWIFT_APPROACHABLE_CONCURRENCY` | `YES` | New-template default in Xcode 26+/27 (SE-0466/SE-0461) |
| `SWIFT_DEFAULT_ACTOR_ISOLATION` | `MainActor` | New-template default; engines stay explicitly `nonisolated`/actor |
| `OTHER_SWIFT_FLAGS` (Release) | `-O -whole-module-optimization` | Default; verify |

### 5.4 App lifecycle gotchas

- **Don't load the model in `App.init`.** Load in the root view's `.task`.
- **Don't run inference on the main thread.** The actor handles it; streaming UI hops to `@MainActor`.
- **`MLX.GPU.set(cacheLimit:)` before the first MLX op** — inside `load()`, not app startup.
- **Swift 6.2 / Xcode 27:** new templates default to MainActor isolation + `nonisolated(nonsending)`. Keep MLX work in actors/`@concurrent` functions, never on MainActor.

---

## 6. SwiftUI surface (L2)

Unchanged structurally from v1. Key updates:

- `@MainActor @Observable` view models (Observation framework) — no `ObservableObject`/`@Published`.
- Split state: `messages: [ChatMessage]` (finished) + `streamingText: String` (in-flight) + `generationID: UUID` to discard stale streams.
- `ScrollViewReader` + `.onChange(of: streamingText)` to pin to bottom.
- `.task` on the chat view for load; `.task(id:)` cancellation for streams.

### 6.1 ChatViewModel (v2 sketch)

```swift
@MainActor @Observable
final class ChatViewModel {
    var messages: [ChatMessage] = [
        .init(role: .system, content: "You are a helpful, concise assistant.")
    ]
    var input = ""
    var isGenerating = false
    var loadState: LoadState = .idle
    var stats: EngineStats?
    var selectedModel: ModelDescriptor = ModelCatalog.default
    private(set) var streamingText = ""

    enum LoadState: Equatable {
        case idle, downloading(Double), loading, ready, failed(String)
    }

    private var engine: (any LLMEngine)?
    private var generationTask: Task<Void, Never>?

    func ensureLoaded() async { /* as v1: create engine, load, set .ready */ }
    func switchModel(to descriptor: ModelDescriptor) async { /* unload, swap, reload */ }

    func send() {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isGenerating, let engine else { return }
        messages.append(.init(role: .user, content: trimmed))
        input = ""
        isGenerating = true
        streamingText = ""
        let history = messages
        generationTask = Task {
            do {
                for try await token in engine.stream(messages: history, options: GenerateOptions()) {
                    if token.isFinal { break }
                    streamingText += token.text
                }
                messages.append(.init(role: .assistant, content: streamingText))
                streamingText = ""
                stats = await engine.lastStats
            } catch {
                messages.append(.init(role: .assistant, content: "⚠️ \(error)"))
            }
            isGenerating = false
        }
    }

    func cancelGeneration() { generationTask?.cancel(); isGenerating = false }
}
```

### 6.2 Views

`ChatView` / `ModelPickerView` / `MessageBubble` / `StatsBar` / `ComposerView` as in v1, with the streaming text rendered from `streamingText` instead of mutating the last message per token. Blinking cursor: append `▎` while `isGenerating`.

---

## 7. Voice input (Speech framework) — iPadOS 26+ path is now the shipping path

### 7.1 `SpeechAnalyzer` + `SpeechTranscriber` (preferred, iPadOS 26+)

As of iPadOS 26 this is shipping and is what Apple uses in Notes. iPadOS 27 adds `CaptureInputSequenceProvider`, `AssetInputSequenceProvider`, and `AnalyzerInputConverter` (convenience input plumbing) — use them if you gate on 27.

Key facts (verified 2026-11-25):

- On-device only, no server path, no 1-minute cap, system-managed model assets (zero app bundle/memory cost), long-form + far-field.
- Needs **only** `NSMicrophoneUsageDescription`. No speech-recognition authorization, no `NSSpeechRecognitionUsageDescription`.
- `SpeechTranscriber` has **no** `contextualStrings`; use `DictationTranscriber` with `AnalysisContext.contextualStrings` if you need custom vocabulary.
- Results arrive via `AsyncSequence` with `isFinal` flags; `result.text` is an `AttributedString`.
- `SpeechTranscriber.supportedLocales` / `installedLocales`; install via `AssetInventory.assetInstallationRequest(supporting:)`.

```swift
// MLXChat/Voice/SpeechAnalyzerDictation.swift  (iPadOS 26+)
import Foundation
import Speech
import AVFoundation

@available(iOS 26.0, *)
@MainActor
final class SpeechAnalyzerDictation: ObservableObject {
    @Published var transcript = ""
    @Published var isListening = false

    private var analyzer: SpeechAnalyzer?
    private var transcriber: SpeechTranscriber?
    private var inputBuilder: AsyncStream<AnalyzerInput>.Continuation?
    private var format: AVAudioFormat?
    private let audioEngine = AVAudioEngine()

    func start() async throws {
        guard await AVAudioApplication.requestRecordPermission() else { return }

        let transcriber = SpeechTranscriber(
            locale: .current,
            preset: .progressiveLiveTranscription
        )
        try await ensureModel(for: transcriber)

        let analyzer = SpeechAnalyzer(modules: [transcriber])
        self.analyzer = analyzer
        self.transcriber = transcriber
        self.format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber])

        let (stream, builder) = AsyncStream<AnalyzerInput>.makeStream()
        self.inputBuilder = builder

        Task { @MainActor in
            for try await result in transcriber.results {
                self.transcript = String(result.text.characters)
            }
        }

        try await analyzer.start(inputSequence: stream)
        try startMic()
        isListening = true
    }

    func stop() async {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        inputBuilder?.finish()
        try? await analyzer?.finalizeAndFinishThroughEndOfInput()
        analyzer = nil; transcriber = nil
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
            // Convert to analyzer format (see Apple's "Recognizing speech in live audio" sample,
            // updated for iPadOS 27's AnalyzerInputConverter if you gate on 27).
            if let converted = try? BufferConverter.shared.convert(buffer, to: format) {
                builder.yield(AnalyzerInput(buffer: converted))
            }
        }
        audioEngine.prepare()
        try audioEngine.start()
    }

    private func ensureModel(for transcriber: SpeechTranscriber) async throws {
        let locale = Locale.current
        let supported = await SpeechTranscriber.supportedLocales
        guard supported.contains(locale) else { throw NSError(domain: "Speech", code: 1) }
        if await SpeechTranscriber.installedLocales.contains(locale) { return }
        if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
            try await request.downloadAndInstall()
        }
    }
}
```

> API details drift between iOS 26.0 → 27.0 betas. Before shipping, diff against the current SDK interface (`xcrun swift-symbolgraph-extract` or the Axiom `axiom-apple-docs` skill) — WWDC25 sample code predates preset renames.

### 7.2 Fallback — `SFSpeechRecognizer` (iPadOS 18–25)

Keep the `Dictation` protocol + `SFSpeechDictation` from v1 behind `#if !os(iOS 26)` / `if #available(iOS 26, *)`. Requires `NSSpeechRecognitionUsageDescription` + runtime authorization; set `requiresOnDeviceRecognition = true`; ~1-minute cap.

### 7.3 Voice → chat hand-off

Unchanged: `VoiceInputBar` observes the dictation object, writes to `ChatViewModel.input`, calls `send()` on stop when transcript non-empty.

---

## 8. Local OpenAI-compatible HTTP server (L3)

### 8.1 Hummingbird 2.29.0 (2026-11-19)

No 3.x exists. The v1 code shape is still correct: `Router`, `Application(router:configuration:)`, `runService()`. Two updates:

- Streaming SSE: return `Response(status: .ok, headers: [.contentType: "text/event-stream", .cacheControl: "no-cache"], body: ResponseBody(asyncSequence: bytes))` where `bytes` is `AsyncSequence<ByteBuffer, Error>`. Or use **SSEKit** (`hummingbird-project/sse-kit`).
- `ResponseBody` also supports the closure form (`ResponseBody { writer in … }`).

```swift
// Sources/MLXChatServer/Server.swift
import Foundation
import Hummingbird
import MLXChatCore

public actor ChatServer {
    private let engineProvider: @Sendable () async -> any LLMEngine
    private var task: Task<Void, Error>?

    public init(engineProvider: @Sendable @escaping () async -> any LLMEngine) {
        self.engineProvider = engineProvider
    }

    public func start(host: String = "127.0.0.1", port: Int = 8765) {
        task = Task {
            let router = Router()
            router.get("/healthz") { _, _ in "ok" }
            router.get("/v1/models") { _, _ in
                ModelsResponse(data: ModelCatalog.all.map(ModelObject.init))
            }
            router.post("/v1/chat/completions") { [engineProvider] req, ctx in
                let body = try await req.decode(as: ChatCompletionRequest.self, context: ctx)
                let engine = await engineProvider()
                let messages = body.messages.map {
                    ChatMessage(role: .init(rawValue: $0.role) ?? .user, content: $0.content)
                }
                let opts = GenerateOptions()

                if body.stream == true {
                    let stream = engine.stream(messages: messages, options: opts)
                    let bytes = AsyncStream<ByteBuffer> { continuation in
                        Task {
                            for try await token in stream {
                                let chunk = ChatCompletionChunk(content: token.text, isFinal: token.isFinal)
                                if let data = try? JSONEncoder().encode(chunk),
                                   let line = String(data: data, encoding: .utf8) {
                                    var buf = ByteBufferAllocator().buffer(string: "data: \(line)\n\n")
                                    continuation.yield(buf)
                                    buf.moveReaderIndex(to: buf.writerIndex)
                                }
                            }
                            continuation.yield(ByteBufferAllocator().buffer(string: "data: [DONE]\n\n"))
                            continuation.finish()
                        }
                    }
                    return Response(
                        status: .ok,
                        headers: [.contentType: "text/event-stream", .cacheControl: "no-cache"],
                        body: ResponseBody(asyncSequence: bytes)
                    )
                } else {
                    let (text, _) = try await engine.respond(messages: messages, options: opts)
                    return Response(status: .ok, body: .init(byteBuffer: ByteBuffer(data: try JSONEncoder().encode(ChatCompletionResponse(content: text)))))
                }
            }

            let app = Application(
                router: router,
                configuration: .init(address: .hostname(host, port: port))
            )
            try await app.runService()
        }
    }

    public func stop() {
        task?.cancel()
        task = nil
    }
}
```

DTOs (`ChatCompletionRequest/Response/Chunk`, `ModelObject`, `ModelsResponse`) unchanged from v1.

### 8.2 Lifecycle rules

- Default **off** at launch; Settings toggle starts it on `127.0.0.1:8765`.
- iPadOS suspends the app in background → server stops. Document in Settings.
- **Loopback needs no permissions** (TN3179). A LAN-bind toggle would need `NSLocalNetworkUsageDescription` — out of scope for v1; if added, gate behind a red-banner setting.

---

## 9. RAG hook stub (`MLXChatRAG`)

Unchanged from v1 — `Retriever` protocol, `StubRetriever`, `withRetrievedContext`. Swap in Basic Memory MCP or a BM25 index later.

---

## 10. Memory pressure handling

Unchanged from v1 (`MemoryGovernor` + `os_proc_available_memory()` + `MLX.GPU.clearCache()` on warning; notify at 3 warnings). Add: log `os_proc_available_memory()` at model load, after load, and every 2 s while generating — this is the number that predicts jetsam. Remember the iPadOS 26.4–26.5 regression was on 8 GB devices; on 16 GB, still verify.

---

## 11. Step-by-step build order

| # | Task | Verify |
|---|---|---|
| 1 | XcodeGen `project.yml` + workspace + iOS app target + local SPM package | App builds empty |
| 2 | SPM deps resolve (mlx-swift 0.33.0, mlx-swift-lm 3.33.0, …) | `xcodebuild -resolvePackageDependencies` |
| 3 | Entitlements + Info.plist (§5) | Build shows Increased Memory Limit |
| 4 | `LLMEngine` protocol + types (§3.1–3.2) | Compiles |
| 5 | `MLXEngine` (§3.4) | Build only |
| 6 | Minimal `ChatViewModel` + smoke view → `engine.respond("hello")` | **Run on presto (real iPad).** First run downloads ~6 GB; 1–3 min on good Wi-Fi. This is your hello-world. |
| 7 | Streaming UI (§6) | Tokens appear progressively |
| 8 | `LlamaEngine` + GGUF catalog entry | Switch backends, verify GGUF path |
| 9 | `MemoryGovernor` + StatsBar | Watch available memory during long generations |
| 10 | `SpeechAnalyzer` dictation + `SFSpeechRecognizer` fallback | Tap mic, dictate, send |
| 11 | `ModelPickerView` + hot-swap | Swap Qwen3.5-9B ↔ 4B; memory drops/recovers |
| 12 | Hummingbird server | `curl localhost:8765/healthz` on-device; `curl` from Mac to iPad IP (LAN toggle) |
| 13 | `StubRetriever` wired into send path | No-op today |
| 14 | TestFlight build; Instruments (Memory + Time Profiler) | Sustain 2k-token generation without jetsam |
| 15 | App Store prep | Privacy labels ("Data Not Collected"), `ITSAppUsesNonExemptEncryption=NO`, review notes re: on-device model download |

---

## 12. Performance expectations (M4 iPad, 120 GB/s)

| Metric | Expected | Why |
|---|---|---|
| Cold model load | 8–15 s | safetensors → Metal upload |
| Warm load | 2–4 s | mmap |
| Prompt processing (1k tok) | ~80–120 t/s | bandwidth-bound |
| **Qwen3.5-9B-4bit decode** | **~18–25 t/s** | bandwidth-bound (M4 base GPU) |
| Qwen3.5-4B-4bit decode | ~35–40 t/s | |
| Gemma-4-E2B decode | ~30–35 t/s | |
| TTFT (100-token prompt) | 0.4–0.8 s | prompt processing dominates |
| Resident peak (32k ctx) | ~9–11 GB | fits 12 GB ceiling |

< 15 t/s on Qwen3.5-9B → check: (1) Release config, (2) cache limit 512 MB, (3) background GPU load, (4) thermal throttling.

TurboQuant `turbo8v3` costs ~0.7–0.8× decode — only enable past ~16k context.

---

## 13. Distribution plan

### 13.1 Personal (paid Apple Developer account)
Self-serve Increased Memory Limit; install via Xcode/devicectl on presto.

### 13.2 TestFlight
- `ITSAppUsesNonExemptEncryption = NO`.
- Privacy nutrition label: **Data Not Collected** (all on-device; no analytics).
- Verify the **distribution** provisioning profile carries the memory entitlement: `codesign -d --entitlements :-`.
- Beta review still enforces 2.5.2 / privacy labels.

### 13.3 App Store
- **2.5.2**: model weights are data, not code — accepted reading per Apple DTS (2025–2026). Precedents live today: **Locally AI** (LM Studio's iPad app), **Private LLM**, **Enchanted**, **PocketPal AI**. Disclose model download in review notes + privacy label.
- **2.3.1(a)**: describe AI features specifically in review notes; no login wall.
- **2.5.19**: no hidden/dormant features — disclose downloads.
- **4.2.3(ii)**: disclose download size before first-launch download; prompt the user.
- **5.1.2(i)**: if any prompt ever leaves the device, explicit consent + label it. (v1: never leaves.)
- **4.2 minimum functionality**: fine — real on-device inference.
- **Model delivery**: **do not use On-Demand Resources** — deprecated in iPadOS 27, rejected from early 2027. Use plain background `URLSession` download (PocketPal/Private LLM pattern) or **Background Assets** (Apple-hosted, up to 200 GB, managed) if you want Apple hosting. 4 GB app bundle cap; don't bundle weights.
- Expect ~80–150 MB IPA.

---

## 14. Risks & open questions (updated 2026-11-25)

| Risk | Likelihood | Mitigation |
|---|---|---|
| mlx-swift-lm API drift | Medium | Pin `from: 3.33.0`; watch releases; `upgrade.md` in repo |
| iPadOS 26.4–26.5 8 GB jetsam regression | Low for 16 GB | Verify with `os_proc_available_memory()`; fixed in 26.6 beta 5 per DTS |
| SpeechAnalyzer API drift iOS 26→27 | Medium | Diff against SDK each Xcode bump; keep SFSpeechRecognizer fallback |
| App Review rejects on-device download | Low | Precedents §13.3; disclose in notes; no executable code |
| `LocalLLMClient` API instability | High (by design) | Pin `exact: 0.5.0`; isolate behind `LLMEngine` |
| AFM 3 / Core AI makes MLX obsolete | Low–Med | Track; MLX still wins on model choice/long context; `MLXFoundationModels` bridge exists if you want both |

### Open questions

1. **System prompt persistence** — per-model default + user override (current default) vs per-conversation.
2. **Conversation persistence** — `@AppStorage` JSON until >100 conversations, then SwiftData.
3. **Thinking blocks** — Qwen3.5 emits `<think>…</think>`; render collapsed by default.
4. **Tool calling** — `MLXGuidedGeneration` (grammar-constrained, ships in 3.32.3+) is the supported path now; LocalLLMClient tool-calling still experimental. Post-v1.
5. **Foundation Models tier** — offer AFM 3 Core (free, zero-download, iPadOS 27) as a "fast/free" engine behind the same `LLMEngine` protocol? `MLXFoundationModels` makes this ~1 day of work. Decide before TestFlight.
6. **Core AI** (new iPadOS 27 framework for on-device custom models) — evaluate alongside MLX for v2; not a drop-in today.

---

## 15. CLAUDE_NOTES.md entries to log after this build

- ✅ mlx-swift-lm 3.33.0 is current; 3.x decouples tokenizer/downloader; `#huggingFaceLoadModelContainer` is the load path.
- ✅ TurboQuant KV shipped (2026-07-20); `turbo8v3` past ~16k context only.
- ✅ Qwen3.5-9B-MLX-4bit is the default; Qwen3-8B-4bit-AWQ deprecated.
- ⚠️ `os_proc_available_memory()` is the only truth; iPadOS 26.4–26.5 regressed 8 GB devices.
- ⚠️ `MLXEngine` streaming: `decode(tokenIds:)` not `decode(tokens:)`.
- ❌ Don't load the model in `App.init`. Don't run MLX in the simulator. Don't use On-Demand Resources for model delivery.

---

## Appendix A — File layout

```
MLXChat/
├── project.yml                      # XcodeGen spec
├── MLXChat.xcodeproj                # generated
├── MLXChat/                         # app target
│   ├── App/MLXChatApp.swift
│   ├── App/MLXChat.entitlements
│   ├── Chat/{ChatView,ChatViewModel,MessageListView,MessageBubble,ComposerView,StatsBar}.swift
│   ├── Models/{ModelPickerView,ModelRow}.swift
│   ├── Voice/{Dictation,SpeechAnalyzerDictation,SFSpeechDictation,BufferConverter,VoiceInputBar}.swift
│   ├── Server/ServerControlView.swift
│   └── System/MemoryGovernor.swift
└── Packages/MLXChatPackages/
    ├── Package.swift
    └── Sources/{MLXChatCore,MLXChatEngineMLX,MLXChatEngineLlama,MLXChatServer,MLXChatRAG}/
```

## Appendix B — Source verification (2026-11-25)

| Claim | Source | Date |
|---|---|---|
| mlx-swift 0.33.0 / mlx-swift-lm 3.33.0 / swift-transformers 1.4.0 / swift-huggingface 1.0.1 / hummingbird 2.29.0 / LocalLLMClient 0.5.0 | GitHub releases | 2026-10-27 → 2026-11-19 |
| `loadContainer(from:using:)` / `#huggingFaceLoadModelContainer` | mlx-swift-lm README + using.md | 2026-11-25 |
| Qwen3.5-9B-MLX-4bit (5.95 GB, 23k dl) | huggingface.co/mlx-community | 2026-11-25 |
| TurboQuant merged (PR #232) | github.com/ml-explore/mlx-swift-lm | 2026-07-20 |
| SpeechAnalyzer shipping since iOS/iPadOS 26; 27 adds input-sequence providers | developer.apple.com WWDC25-277; Speech docs | 2026-11-25 |
| iPadOS 26.4–26.5 8 GB jetsam regression; fixed 26.6 beta 5 | Apple Forums thread 826325; FB22883960 | 2026 |
| ODR deprecated in iPadOS 27; Background Assets is the path | developer.apple.com; WWDC25-325 | 2026 |
| Loopback needs no Local Network permission | Apple TN3179 | 2026-11-25 |
| App Review 2.5.2 weights-as-data; precedents (Locally AI, Private LLM, Enchanted, PocketPal) | App Review Guidelines (2026-10-16); App Store listings | 2026-11-25 |
| AFM 3 (Core / Core Advanced) on iPadOS 27; MLX is a first-class LanguageModel provider | machinelearning.apple.com; WWDC26 | 2026-06/09 |
| Qwen3.5-9B ≈ 18–25 tok/s on M4 iPad (bandwidth-derived + third-party measurements) | lapis.lvrpiz.com; Apple M5-vs-M4 MLX study | 2026 |
| Increased Memory Limit self-serve; verify distribution profile | Apple DTS via developer forums | 2026 |

— end of manual (v2, 2026-11-25) —
