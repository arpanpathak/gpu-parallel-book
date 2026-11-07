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
    bash -c 'cd ch08_reduction && nvcc -arch="$1" reduce.cu -o reduce && ./reduce' _ "$ARCH"

run "ch09_sgemm" \
    bash -c 'cd ch09_sgemm && nvcc -arch="$1" sgemm.cu -o sgemm -lcublas && ./sgemm' _ "$ARCH"

run "ch10_raw" \
    bash -c 'cd ch10_modern && nvcc -arch="$1" raw.cu -o raw && ./raw' _ "$ARCH"

run "ch10_modern" \
    bash -c 'cd ch10_modern && nvcc -std=c++20 --extended-lambda -arch="$1" modern.cu -o modern && ./modern' _ "$ARCH"

run "ch11_reduce_scan" \
    bash -c 'cd ch11_libraries && nvcc -arch="$1" -O2 reduce_scan.cu -o reduce_scan && ./reduce_scan' _ "$ARCH"

run "ch11_sort" \
    bash -c 'cd ch11_libraries && nvcc -arch="$1" -O2 sort.cu -o sort && ./sort' _ "$ARCH"

run "ch11_gemm" \
    bash -c 'cd ch11_libraries && nvcc -arch="$1" -O2 gemm.cu -o gemm -lcublas && ./gemm' _ "$ARCH"

run "ch11_fft_rand" \
    bash -c 'cd ch11_libraries && nvcc -arch="$1" -O2 fft_rand.cu -o fft_rand -lcurand -lcufft && ./fft_rand' _ "$ARCH"

run "ch12_nvrtc" \
    bash -c 'cd ch12_nvrtc && nvcc -arch="$1" nvrtc.cu -o nvrtc -lnvrtc -lcuda && ./nvrtc' _ "$ARCH"

run "ch15_capstone" \
    bash -c 'cd ch15_capstone && nvcc -arch="$1" pipeline.cu -o pipeline && ./pipeline 96 64 2' _ "$ARCH"

# bugs.cu is left out on purpose: its kernels are wrong, and 1 of them can hang the GPU.
for prog in measure latency verify; do
    run "ch16_$prog" \
        bash -c 'cd ch16_profiling && nvcc -arch="$1" "$2.cu" -o "$2" && "./$2"' _ "$ARCH" "$prog"
done

run "ch17_systems" \
    bash -c 'cd ch17_systems && nvcc -arch="$1" -O3 systems.cu -o systems && ./systems' _ "$ARCH"

run "ch18_peer" \
    bash -c 'cd ch18_peer && nvcc -arch="$1" peer.cu -o peer && ./peer' _ "$ARCH"

run "ch19_collectives" \
    bash -c 'cd ch19_collectives && nvcc -arch="$1" allreduce.cu -o allreduce && ./allreduce' _ "$ARCH"

# --- Part V, CUDA C++ (code/cuda): each file states its build line ----------
for src in cuda/ch*/*.cu; do
    dir=$(dirname "$src")
    build=$(sed -n 's|^// Build: ||p' "$src" | head -n 1)
    build=${build//-arch=native/-arch=$ARCH}
    bin=${build##* -o }

    run "$dir" bash -c 'cd "$1" && $2 && "./$3"' _ "$dir" "$build" "$bin"
done

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
    run "oxide/ch19_collectives" \
        bash -c 'cd oxide/ch19_collectives && cargo oxide run'

    for crate in oxide/ch1[4-9]_* oxide/ch2[0-4]_*; do
        case "$crate" in
            oxide/ch16_latency|oxide/ch17_systems|oxide/ch18_peer|oxide/ch19_collectives) continue ;;
        esac

        run "$crate" bash -c 'cd "$1" && cargo oxide run' _ "$crate"
    done
else
    echo "SKIP ch14_cuda_oxide and the oxide/ crates (cargo-oxide not found)"
fi

echo
echo "PASSED=$PASS FAILED=$FAIL"
[ "$FAIL" -eq 0 ]
