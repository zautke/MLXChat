# Vendored LocalLLMClient

Source: https://github.com/tattn/LocalLLMClient @ `0.5.0` (edc39ef2ffc1cef9cf856b0788de8d331f776c2e), MIT license (see LICENSE).

Vendored (not a remote SPM dependency) for two reasons:

1. Its `LocalLLMClientLlamaC` target uses `unsafeFlags`, which SPM forbids in remote dependencies.
2. Xcode 27's clang dependency scanner requires the C++ umbrella module to declare `requires cplusplus`; the stock module map does not.

Local modifications vs upstream 0.5.0 (keep minimal; re-apply on upgrade):

- `Sources/LocalLLMClientLlamaC/include/module.modulemap`: added `requires cplusplus`
- `Package.swift`: `cxxSettings` gained `.unsafeFlags(["-std=c++17"])` on the `LocalLLMClientLlamaC` target

The llama.cpp sources under `Sources/LocalLLMClientLlamaC/exclude/llama.cpp` are a git submodule of THIS repo pointing at https://github.com/ggml-org/llama.cpp @ `e365e658f07b63371489570dfde597f199b26c23` (as pinned by upstream 0.5.0).

Clone with: `git submodule update --init --recursive`
