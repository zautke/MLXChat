import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers
import MLXChatCore

/// MLX backend. Requires a physical iOS/iPadOS device — MLX Metal kernels do
/// not run in the simulator.
public actor MLXEngine: LLMEngine {
    public nonisolated let descriptor: ModelDescriptor
    public private(set) var isLoaded: Bool = false
    public private(set) var lastStats: EngineStats?

    private var container: ModelContainer?

    /// Buffer cache cap: 256 MB is the empirical sweet spot on 16 GB iPads —
    /// large enough to amortize Metal allocator churn, small enough to leave
    /// room for the KV cache.
    private static let gpuCacheLimitBytes = 256 * 1024 * 1024

    /// Drop the Metal buffer cache. Safe to call any time; used by the memory governor.
    public static func clearGPUCache() {
        MLX.GPU.clearCache()
    }

    public init(descriptor: ModelDescriptor) {
        precondition(descriptor.backend == .mlx, "MLXEngine requires .mlx backend")
        self.descriptor = descriptor
    }

    public func load(progress: @Sendable @escaping (Double) -> Void) async throws {
        guard !isLoaded else { return }

        // Cap the buffer cache BEFORE the first MLX op.
        MLX.GPU.set(cacheLimit: Self.gpuCacheLimitBytes)

        // os_proc_available_memory() is iOS/tvOS only; on macOS fall back to physical memory.
        #if os(iOS) || os(tvOS) || os(visionOS)
        let available = UInt64(os_proc_available_memory())
        #else
        let available = ProcessInfo.processInfo.physicalMemory
        #endif
        let required = UInt64(Double(descriptor.approximateBytesOnDisk) * 1.4)
        guard available >= required else {
            throw EngineError.insufficientMemory(
                availableBytes: available,
                requiredBytes: required
            )
        }

        let configuration = ModelConfiguration(id: descriptor.huggingFaceRepo)

        do {
            self.container = try await LLMModelFactory.shared.loadContainer(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: configuration,
                progressHandler: { p in progress(p.fractionCompleted) }
            )
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

        let parameters = GenerateParameters(
            maxTokens: options.maxTokens,
            temperature: options.temperature,
            topP: options.topP,
            topK: options.topK,
            repetitionPenalty: options.repetitionPenalty
        )
        // Note: kvScheme (TurboQuant) requires mlx-swift-lm >= 3.32, which is blocked
        // by LocalLLMClient 0.5.0's swift-syntax 600.x pin. See Package.swift.

        let startTime = ContinuousClock.now

        let (promptTokenCount, completionTokenCount, generationStarted) = try await container.perform { context -> (Int, Int, ContinuousClock.Instant?) in
            var userInput = UserInput(
                messages: messages.map { ["role": $0.role.rawValue, "content": $0.content] }
            )
            if let enableThinking = options.enableThinking {
                userInput.additionalContext = ["enable_thinking": enableThinking]
            }
            let input = try await context.processor.prepare(input: userInput)
            let promptTokens = input.text.tokens.size

            let stream = try MLXLMCommon.generate(
                input: input,
                parameters: parameters,
                context: context
            )

            var completionTokens = 0
            var generationStarted: ContinuousClock.Instant? = nil

            for await generation in stream {
                switch generation {
                case .chunk(let text):
                    if generationStarted == nil { generationStarted = .now }
                    continuation.yield(GenerationToken(
                        text: text,
                        tokenIndex: completionTokens,
                        isFinal: false
                    ))
                    completionTokens += 1
                case .info(let info):
                    return (info.promptTokenCount, info.generationTokenCount, generationStarted)
                case .toolCall:
                    break
                }
            }
            return (promptTokens, completionTokens, generationStarted)
        }

        let endTime = ContinuousClock.now
        let promptDuration = generationStarted.map { startTime.duration(to: $0) } ?? .zero
        let completionDuration = generationStarted.map { $0.duration(to: endTime) } ?? .zero

        let stats = EngineStats(
            promptTokens: promptTokenCount,
            completionTokens: completionTokenCount,
            promptTokensPerSecond: Self.tps(promptTokenCount, promptDuration),
            completionTokensPerSecond: Self.tps(completionTokenCount, completionDuration),
            peakResidentMemoryBytes: Self.residentMemoryBytes()
        )
        self.lastStats = stats

        continuation.yield(GenerationToken(text: "", tokenIndex: completionTokenCount, isFinal: true))
        continuation.finish()
    }

    private static func tps(_ tokens: Int, _ duration: Duration) -> Double {
        let seconds = Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1e18
        return seconds > 0 ? Double(tokens) / seconds : 0
    }

    private static func residentMemoryBytes() -> UInt64 {
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
