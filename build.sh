#!/bin/bash
set -e

git submodule update --init --recursive
mkdir -p build
cd build
cmake ..
cmake --build .

echo "Build complete. Executable is ready"
