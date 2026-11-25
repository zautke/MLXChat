// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "MLXChatPackages",
    platforms: [
        .iOS(.v26),
        .macOS(.v26),
    ],
    products: [
        .library(name: "MLXChatCore", targets: ["MLXChatCore"]),
        .library(name: "MLXChatEngineMLX", targets: ["MLXChatEngineMLX"]),
        .library(name: "MLXChatEngineLlama", targets: ["MLXChatEngineLlama"]),
        .library(name: "MLXChatServer", targets: ["MLXChatServer"]),
        .library(name: "MLXChatRAG", targets: ["MLXChatRAG"]),
    ],
    dependencies: [
        // NOTE: pinned to 3.31.3 / 0.31.3 — LocalLLMClient 0.5.0 requires swift-syntax 600.x,
        // which conflicts with mlx-swift-lm >= 3.32 (requires 602+). TurboQuant KV and
        // MLXFoundationModels are therefore unavailable until LocalLLMClient bumps.
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.3")),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "3.31.3")),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "1.0.1"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.4.0"),
        // Vendored (path dependency) because LocalLLMClientLlamaC uses unsafeFlags,
        // which SPM only permits in path-based packages. See Vendor/LocalLLMClient/VENDORED.md.
        .package(path: "../../Vendor/LocalLLMClient"),
        .package(url: "https://github.com/hummingbird-project/hummingbird", from: "2.29.0"),
    ],
    targets: [
        .target(
            name: "MLXChatCore",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .target(
            name: "MLXChatEngineMLX",
            dependencies: [
                "MLXChatCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .target(
            name: "MLXChatEngineLlama",
            dependencies: [
                "MLXChatCore",
                .product(name: "LocalLLMClient", package: "LocalLLMClient"),
                .product(name: "LocalLLMClientLlama", package: "LocalLLMClient"),
            ],
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),
        .target(
            name: "MLXChatServer",
            dependencies: [
                "MLXChatCore",
                .product(name: "Hummingbird", package: "hummingbird"),
            ]
        ),
        .target(
            name: "MLXChatRAG",
            dependencies: ["MLXChatCore"]
        ),
        .testTarget(
            name: "MLXChatCoreTests",
            dependencies: ["MLXChatCore"]
        ),
    ],
    swiftLanguageModes: [.v6]
)
