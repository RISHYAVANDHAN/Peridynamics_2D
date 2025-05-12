#!/bin/bash
set -e

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure CMake and build
cmake ..
cmake --build .

echo "Build complete. Executable is in build/peridynamics_2d"
