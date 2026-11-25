import Foundation

/// Curated model catalog. Verified against Hugging Face on 2026-11-25.
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
        qwen35_9B_4bit,
        qwen35_4B_4bit,
        gemma4_E4B_4bit,
        gemma4_E2B_4bit,
        qwen3_8B_4bit,
        qwen3_4B_Instruct,
        qwen35_9B_GGUF,
    ]

    public static let `default` = qwen35_9B_4bit
}
