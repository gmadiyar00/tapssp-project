#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_TMP="$PROJECT_ROOT/.tmp_llama_build"

echo "Project root: $PROJECT_ROOT"

command -v cmake >/dev/null 2>&1 || { echo "cmake not found. Install CMake (brew install cmake) and rerun."; exit 1; }
command -v make >/dev/null 2>&1 || { echo "make not found. Install build tools and rerun."; exit 1; }

rm -rf "$BUILD_TMP"
mkdir -p "$BUILD_TMP"

echo "Cloning llama.cpp into $BUILD_TMP/llama.cpp..."
git clone https://github.com/ggerganov/llama.cpp.git "$BUILD_TMP/llama.cpp"

echo "Building llama.cpp (this may take a while)..."
cd "$BUILD_TMP/llama.cpp"
mkdir -p build && cd build
cmake -DLLAMA_METAL=ON ..

if ! make -j"$(sysctl -n hw.ncpu)"; then
  echo "Parallel build failed, retrying single-threaded..."
  make
fi

mkdir -p "$PROJECT_ROOT/llama.cpp"
cp -f ./main "$PROJECT_ROOT/llama.cpp/main"
chmod +x "$PROJECT_ROOT/llama.cpp/main"

echo "Built binary copied to $PROJECT_ROOT/llama.cpp/main"

echo "Cleaning up build temp..."
rm -rf "$BUILD_TMP"

echo "Done. You can now run: cargo run"