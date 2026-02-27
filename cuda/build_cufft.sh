#!/bin/bash
# Build cuFFT wrapper as shared library
# Run from janus-sim directory

set -e

CUDA_PATH=${CUDA_PATH:-/usr/local/cuda}
OUT_DIR=${OUT_DIR:-target/release}

echo "Building cuFFT wrapper..."
echo "  CUDA_PATH: $CUDA_PATH"
echo "  OUT_DIR: $OUT_DIR"

mkdir -p "$OUT_DIR"

# Compile to shared library
nvcc -shared -o "$OUT_DIR/libcufft_wrapper.so" \
    cuda/cufft_wrapper.cu \
    -Xcompiler -fPIC \
    -lcufft \
    -arch=sm_86 \
    -O3 \
    --std=c++11

echo "Built: $OUT_DIR/libcufft_wrapper.so"
ls -la "$OUT_DIR/libcufft_wrapper.so"
