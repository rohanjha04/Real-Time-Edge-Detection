#!/usr/bin/env bash
set -euo pipefail

# Usage: ./rebuild.sh [build-dir]
BUILD_DIR=${1:-build}

# 1. Ensure build directory exists
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# 2. Configure with CMake (pointing to your OpenCV build)
cmake .. 
# -DOpenCV_DIR="C:/OpenCV_cpp/opencv/build"

# 3. Build
cmake --build .
