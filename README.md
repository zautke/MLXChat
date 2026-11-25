# MLXChat

On-device LLM chat for iPad Pro M4 (16 GB). SwiftUI + MLX (Apple Silicon) with a llama.cpp/GGUF fallback, an embedded OpenAI-compatible HTTP server, and on-device voice dictation.

**Design doc:** [`claude@_opus4-7_MLX-iPad-M4-Implementation-Manual.md`](claude@_opus4-7_MLX-iPad-M4-Implementation-Manual.md) (v2, 2026-11-25)

## Layout

- `Packages/MLXChatPackages/` — SPM package: `MLXChatCore` (protocol + catalog), `MLXChatEngineMLX`, `MLXChatEngineLlama`, `MLXChatServer` (Hummingbird 2), `MLXChatRAG`
- `MLXChat/` — SwiftUI app (chat UI, model picker, voice dictation, memory governor, server control)
- `project.yml` — XcodeGen spec (`xcodegen generate`)

## Build & run

```sh
git submodule update --init --recursive   # LocalLLMClient + llama.cpp
./Scripts/patch-vendor.sh                 # required for Xcode 27 clang module scanning (idempotent)
xcodegen generate
open MLXChat.xcodeproj
```

Requires Xcode 27, iPadOS 26+ deployment target, a physical iPad (MLX does not run in the simulator). Default model: `mlx-community/Qwen3.5-9B-MLX-4bit` (~6 GB download on first launch).

Note: the app's `Increased Memory Limit` entitlement must be present in the App ID's provisioning profile, or iOS caps the process at ~5 GB and the 9B model will not load. Enable the "Increased Memory Limit" capability for `com.zautke.mlxchat` in the Apple Developer portal, then rebuild with `-allowProvisioningUpdates`.

## Notes

- `mlx-swift-lm` is pinned to 3.31.x because `LocalLLMClient` 0.5.0 requires swift-syntax 600.x; see the manual §2.1 for the full rationale.
- The local HTTP server is loopback-only (`127.0.0.1:8765`) and off by default.
