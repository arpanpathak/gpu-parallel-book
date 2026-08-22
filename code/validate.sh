#!/usr/bin/env bash
# Validate every runnable example in the book's code companion.
#
# Usage:
#   ./validate.sh                  # portable compute_60 PTX (any CUDA 12.x GPU)
#   ARCH=sm_87 ./validate.sh       # native SASS on Jetson Orin
#   ARCH=sm_80 ./validate.sh       # native SASS on A100
#   ARCH=sm_90 ./validate.sh       # native SASS on H100
#
# Rust examples (ch13/ch14) are validated only if their toolchains exist.
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${ARCH:-compute_60}"
NVCC=(nvcc -arch="$ARCH")
PASS=0
FAIL=0

run() {
    local name="$1"; shift
    echo "=== $name ==="
    if "$@"; then
        PASS=$((PASS + 1))
    else
        echo "FAILED: $name" >&2
        FAIL=$((FAIL + 1))
    fi
}

# --- CUDA C++ examples -------------------------------------------------------
run "ch03_vector_add" \
    bash -c 'cd ch03_vector_add && nvcc -arch="$1" vector_add.cu -o vector_add && ./vector_add' _ "$ARCH"

run "ch05_histogram" \
    bash -c 'cd ch05_histogram && nvcc -arch="$1" histogram.cu main.cu -o histogram && ./histogram' _ "$ARCH"

run "ch08_reduction" \
    bash -c 'cd ch08_reduction && nvcc -rdc=true -arch="$1" reduction.cu main.cu -o reduction && ./reduction' _ "$ARCH"

run "ch09_sgemm" \
    bash -c 'cd ch09_sgemm && nvcc -arch="$1" sgemm.cu main.cu -o sgemm && ./sgemm' _ "$ARCH"

run "ch10_device_buffer" \
    bash -c 'cd ch10_device_buffer && nvcc -std=c++20 -arch="$1" test.cu -o test && ./test' _ "$ARCH"

run "ch11_thrust" \
    bash -c 'cd ch11_library_examples && nvcc --extended-lambda -arch="$1" thrust_example.cu -o thrust_example && ./thrust_example' _ "$ARCH"

run "ch11_cub" \
    bash -c 'cd ch11_library_examples && nvcc -arch="$1" cub_reduce.cu -o cub_reduce && ./cub_reduce' _ "$ARCH"

run "ch11_cublas" \
    bash -c 'cd ch11_library_examples && nvcc -arch="$1" cublas_sgemm.cu -o cublas_sgemm -lcublas && ./cublas_sgemm' _ "$ARCH"

run "ch12_nvrtc" \
    bash -c 'cd ch12_nvrtc && nvcc -arch="$1" main.cu -o nvrtc_example -lnvrtc -lcuda && ./nvrtc_example' _ "$ARCH"

run "ch15_capstone" \
    bash -c 'cd ch15_capstone && nvcc -arch="$1" pipeline.cu -o pipeline && ./pipeline 96 64 2' _ "$ARCH"

# --- Rust examples (optional toolchains) -------------------------------------
if command -v cargo >/dev/null 2>&1; then
    run "ch13_rust_vector_add" \
        bash -c 'cd ch13_rust_vector_add && cargo build --release && ./target/release/vector-add'
else
    echo "SKIP ch13_rust_vector_add (cargo not found)"
fi

if command -v cargo-oxide >/dev/null 2>&1; then
    run "ch14_cuda_oxide" \
        bash -c 'cd ch14_cuda_oxide && cargo oxide run host_closure'
else
    echo "SKIP ch14_cuda_oxide (cargo-oxide not found)"
fi

echo
echo "PASSED=$PASS FAILED=$FAIL"
[ "$FAIL" -eq 0 ]
