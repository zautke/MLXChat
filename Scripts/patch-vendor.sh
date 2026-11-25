#!/bin/bash
# Apply required patches to the vendored LocalLLMClient submodule.
# Required for Xcode 27 / clang module scanning of the C++ umbrella module.
# Idempotent: safe to re-run.
set -euo pipefail

cd "$(dirname "$0")/.."
SUBMODULE="Vendor/LocalLLMClient"
PATCH="Vendor/localllmclient-xcode27-cxx-interop.patch"

if [ ! -d "$SUBMODULE/.git" ] && [ ! -f "$SUBMODULE/.git" ]; then
  echo "error: $SUBMODULE is not checked out. Run: git submodule update --init --recursive" >&2
  exit 1
fi

cd "$SUBMODULE"
if git apply --check --reverse "../$(basename "$PATCH")" 2>/dev/null; then
  echo "patch already applied"
else
  git apply "../$(basename "$PATCH")"
  echo "patch applied"
fi
