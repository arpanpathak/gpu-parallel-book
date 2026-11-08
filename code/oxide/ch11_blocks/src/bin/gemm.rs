//! Chapter 11 in Rust: cuBLAS called through its C interface, with row-major matrices,
//! then SGEMM in FP32 and on tensor cores.
//!
//!     cargo oxide run --bin gemm

use std::ffi::{c_int, c_void};
use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer};

const N: usize = 4096; // the matrix size of Chapter 9

type Error = Box<dyn std::error::Error>;

/// A cuBLAS handle: a pointer to state that only cuBLAS reads.
type Handle = *mut c_void;

const CUBLAS_OP_N: c_int = 0;
const CUBLAS_TF32_TENSOR_OP_MATH: c_int = 3;

#[link(name = "cublas")]
unsafe extern "C" {
    fn cublasCreate_v2(handle: *mut Handle) -> c_int;
    fn cublasSetStream_v2(handle: Handle, stream: *mut c_void) -> c_int;
    fn cublasSetMathMode(handle: Handle, mode: c_int) -> c_int;
    fn cublasDestroy_v2(handle: Handle) -> c_int;
    fn cublasSgemm_v2(
        handle: Handle,
        op_a: c_int,
        op_b: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: *const f32,
        c: *mut f32,
        ldc: c_int,
    ) -> c_int;
}

/// cuBLAS returns 0 for success and an error number otherwise.
fn ok(result: c_int, call: &str) -> Result<(), Error> {
    if result == 0 {
        Ok(())
    } else {
        Err(format!("{call} failed with cuBLAS error {result}").into())
    }
}

/// The device address of a buffer, as the pointer type cuBLAS takes.
fn ptr(buffer: &DeviceBuffer<f32>) -> *mut f32 {
    buffer.cu_deviceptr() as *mut f32
}

/// Average time of `runs` runs of `launch`, in milliseconds, after 1 warm-up run.
fn time_ms<F>(stream: &CudaStream, runs: u32, mut launch: F) -> Result<f64, Error>
where
    F: FnMut() -> Result<(), Error>,
{
    launch()?;
    stream.synchronize()?;

    let start = Instant::now();
    for _ in 0..runs {
        launch()?;
    }
    stream.synchronize()?;

    Ok(start.elapsed().as_secs_f64() * 1e3 / runs as f64)
}

fn print(name: &str, m: &[f32], rows: usize, cols: usize) {
    let rows: Vec<String> = m
        .chunks(cols)
        .take(rows)
        .map(|row| row.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" "))
        .collect();
    println!("  {name} = [{}]", rows.join("; "));
}

/// C = A B for a 2 x 3 matrix A and a 3 x 2 matrix B, all stored row by row.
fn small_product(handle: Handle, stream: &Arc<CudaStream>) -> Result<bool, Error> {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let expected = [58.0, 64.0, 139.0, 154.0];

    let a_dev = DeviceBuffer::from_host(stream, &a)?;
    let b_dev = DeviceBuffer::from_host(stream, &b)?;
    let c_dev = DeviceBuffer::<f32>::zeroed(stream, 4)?;
    let (pa, pb, pc) = (ptr(&a_dev), ptr(&b_dev), ptr(&c_dev));

    let (one, zero) = (1.0f32, 0.0f32);
    let op = CUBLAS_OP_N;
    println!("C = A B with A 2 x 3 and B 3 x 2, stored row by row");
    print("A", &a, 2, 3);
    print("B", &b, 3, 2);

    // Wrong: cuBLAS reads each array column by column.
    // SAFETY: the 3 pointers address live buffers of 6, 6 and 4 floats, enough for the
    // sizes and leading dimensions given.
    let result = unsafe {
        cublasSgemm_v2(handle, op, op, 2, 2, 3, &one, pa, 2, pb, 3, &zero, pc, 2)
    };
    ok(result, "cublasSgemm")?;
    print("C, operands as given  ", &c_dev.to_host_vec(stream)?, 2, 2);

    // Right: in column-major terms our arrays hold A^T and B^T. Ask for B^T A^T = C^T,
    // whose column-major storage is C row by row.
    // SAFETY: as above.
    let result = unsafe {
        cublasSgemm_v2(handle, op, op, 2, 2, 3, &one, pb, 2, pa, 3, &zero, pc, 2)
    };
    ok(result, "cublasSgemm")?;
    let c = c_dev.to_host_vec(stream)?;
    print("C, operands swapped   ", &c, 2, 2);

    Ok(c == expected)
}

/// Values in [-0.5, 0.5) from a small linear congruential generator, as in Chapter 9.
fn random_matrix(state: &mut u32) -> Vec<f32> {
    (0..N * N)
        .map(|_| {
            *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (*state >> 8) as f32 / 16_777_216.0 - 0.5
        })
        .collect()
}

/// The largest difference from a double-precision product, over 64 elements of C.
fn max_error(a: &[f32], b: &[f32], c: &[f32]) -> f64 {
    let mut worst: f64 = 0.0;

    for s in 0..64 {
        let i = s * 977 % N;
        let j = (s * 1231 + 7) % N;
        let mut exact = 0.0;
        for k in 0..N {
            exact += a[i * N + k] as f64 * b[k * N + j] as f64;
        }
        worst = worst.max((exact - c[i * N + j] as f64).abs());
    }
    worst
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    println!("GPU: {}", ctx.device_name()?);

    let start = Instant::now();
    let mut handle: Handle = std::ptr::null_mut();
    // SAFETY: `handle` is a valid place for cuBLAS to store its handle.
    ok(unsafe { cublasCreate_v2(&mut handle) }, "cublasCreate")?;
    println!("cublasCreate took {:.1} ms", start.elapsed().as_secs_f64() * 1e3);

    // SAFETY: the handle is live, and the stream lives until the program ends.
    let result = unsafe { cublasSetStream_v2(handle, stream.cu_stream().cast()) };
    ok(result, "cublasSetStream")?;

    let mut passed = small_product(handle, &stream)?;

    let mut state = 1;
    let a = random_matrix(&mut state);
    let b = random_matrix(&mut state);
    let a_dev = DeviceBuffer::from_host(&stream, &a)?;
    let b_dev = DeviceBuffer::from_host(&stream, &b)?;
    let c_dev = DeviceBuffer::<f32>::zeroed(&stream, N * N)?;
    let (pa, pb, pc) = (ptr(&a_dev), ptr(&b_dev), ptr(&c_dev));

    let (one, zero) = (1.0f32, 0.0f32);
    let op = CUBLAS_OP_N;
    let n = N as c_int;
    let mut gemm = || -> Result<(), Error> {
        // SAFETY: the 3 pointers address live buffers of N x N floats.
        let result = unsafe {
            cublasSgemm_v2(handle, op, op, n, n, n, &one, pb, n, pa, n, &zero, pc, n)
        };
        ok(result, "cublasSgemm")
    };

    let report = |name: &str, ms: f64| -> Result<f64, Error> {
        let c = c_dev.to_host_vec(&stream)?;
        let gflops = 2.0 * (N * N * N) as f64 / ms / 1e6;
        let error = max_error(&a, &b, &c);
        println!("  {name:<22} {ms:8.2} ms  {gflops:7.1} GFLOP/s  max error {error:.1e}");
        Ok(error)
    };

    for _ in 0..10 {
        time_ms(&stream, 5, &mut gemm)?; // warm up: clocks rise under sustained load
    }

    println!("C = A B, {N} x {N} floats, average of 10 runs");
    let fp32 = report("cuBLAS, FP32", time_ms(&stream, 10, &mut gemm)?)?;

    // SAFETY: the handle is live.
    let result = unsafe { cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) };
    ok(result, "cublasSetMathMode")?;
    let tf32 = report("cuBLAS, TF32 tensor", time_ms(&stream, 10, &mut gemm)?)?;

    passed &= fp32 < 1e-3 && tf32 < 1e-2;
    println!("{}", if passed { "PASSED" } else { "FAILED" });

    // SAFETY: the handle is live and is not used after this call.
    ok(unsafe { cublasDestroy_v2(handle) }, "cublasDestroy")?;
    Ok(())
}
