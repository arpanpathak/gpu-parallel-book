//! Chapter 16: matrix multiplication in Rust, 1 thread per element of C.
//!
//!     cargo oxide run

use std::ops::{Add, Mul};
use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy, LaunchConfig2D};
use cuda_device::thread::{Index2D, Runtime2DIndex};
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_contract, thread};
use cuda_host::RowWidth;

const BLOCK: u32 = 16; // a block is 16 x 16 threads
const FIXED: usize = 1024; // the width of C in the compile-time kernel

type Error = Box<dyn std::error::Error>;

#[cuda_module]
mod kernels {
    use super::*;

    /// C = A B for FIXED x FIXED matrices: the width is part of the type of `c`.
    #[kernel]
    #[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1))]
    pub fn matmul_fixed(a: &[f32], b: &[f32], mut c: DisjointSlice<f32, Index2D<FIXED>>) {
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();

        if let Some(idx) = thread::index_2d::<FIXED>()
            && let Some(out) = c.get_mut(idx)
        {
            let mut sum = 0.0;
            let mut i = 0;
            #[unroll(4)]
            while i < FIXED {
                sum += a[row * FIXED + i] * b[i * FIXED + col];
                i += 1;
            }
            *out = sum;
        }
    }

    /// C (m x n) = A (m x k) B (k x n), all row by row. The width of `c` is bound
    /// on the host with `RowWidth`; the contract checks the 3 lengths before launch.
    #[kernel]
    #[launch_contract(
        domain = 2,
        coordinates = u32,
        block = (16, 16, 1),
        requires = (a.len() >= m * k, b.len() >= k * n, c.len() >= m * n)
    )]
    pub fn matmul<T>(
        m: u32,
        n: u32,
        k: u32,
        a: &[T],
        b: &[T],
        mut c: DisjointSlice<T, Runtime2DIndex>,
    ) where
        T: Copy + Default + Add<Output = T> + Mul<Output = T>,
    {
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();
        let (m, n, k) = (m as usize, n as usize, k as usize);

        if row < m
            && col < n
            && let Some(idx) = thread::index_2d_runtime(&c)
            && let Some(out) = c.get_mut(idx)
        {
            let mut sum = T::default();
            let mut i = 0;
            #[unroll(4)]
            while i < k {
                sum = sum + a[row * k + i] * b[i * n + col];
                i += 1;
            }
            *out = sum;
        }
    }

    /// The f32 body of `matmul` without bounds checks on `a` and `b`. The guard
    /// row < m, col < n and the contract's lengths keep every index in bounds.
    #[kernel(unchecked_indexing)]
    #[launch_contract(
        domain = 2,
        coordinates = u32,
        block = (16, 16, 1),
        requires = (a.len() >= m * k, b.len() >= k * n, c.len() >= m * n)
    )]
    pub fn matmul_no_bounds(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, Runtime2DIndex>,
    ) {
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();
        let (m, n, k) = (m as usize, n as usize, k as usize);

        if row < m
            && col < n
            && let Some(idx) = thread::index_2d_runtime(&c)
            && let Some(out) = c.get_mut(idx)
        {
            let mut sum = 0.0;
            let mut i = 0;
            #[unroll(4)]
            while i < k {
                sum += a[row * k + i] * b[i * n + col];
                i += 1;
            }
            *out = sum;
        }
    }
}

/// Average time of `runs` launches in milliseconds, after 1 launch that is not timed.
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

/// Values in [-0.5, 0.5) from a small linear congruential generator.
fn random<T: From<f32>>(len: usize, state: &mut u32) -> Vec<T> {
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let x = (*state >> 8) as f32 / 16_777_216.0 - 0.5;
        values.push(T::from(x));
    }
    values
}

/// The largest difference from a double-precision product, over 64 elements of C.
fn max_error<T>(a: &[T], b: &[T], c: &[T], n: usize) -> f64
where
    T: Copy + Into<f64>,
{
    let mut worst: f64 = 0.0;
    for s in 0..64 {
        let i = s * 977 % n;
        let j = (s * 1231 + 7) % n;

        let mut exact = 0.0;
        for k in 0..n {
            exact += a[i * n + k].into() * b[k * n + j].into();
        }

        let got: f64 = c[i * n + j].into();
        worst = worst.max((exact - got).abs());
    }
    worst
}

/// GFLOP/s of an n x n product that took `ms` milliseconds: 2 n^3 operations.
fn gflops(n: usize, ms: f64) -> f64 {
    2.0 * (n * n * n) as f64 / ms / 1e6
}

/// Launches that keep n^3 x runs near 2^33, so each size runs for a similar time.
fn runs_for(n: usize) -> u32 {
    ((1usize << 33) / (n * n * n)).clamp(1, 50) as u32
}

/// A grid of 16 x 16 blocks that covers an n x n matrix C.
fn config(n: usize) -> LaunchConfig2D {
    let blocks = (n as u32).div_ceil(BLOCK);
    LaunchConfig2D::new((blocks, blocks), (BLOCK, BLOCK), 0)
}

/// Time matmul::<T> on n x n matrices, print a line, and return the largest error.
fn run_generic<T>(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    name: &str,
    n: usize,
) -> Result<f64, Error>
where
    T: DeviceCopy + Copy + Default + Add<Output = T> + Mul<Output = T>,
    T: From<f32> + Into<f64>,
{
    let mut state = 1;
    let a_host: Vec<T> = random(n * n, &mut state);
    let b_host: Vec<T> = random(n * n, &mut state);
    let a = DeviceBuffer::from_host(stream, &a_host)?;
    let b = DeviceBuffer::from_host(stream, &b_host)?;
    let mut c = DeviceBuffer::<T>::zeroed(stream, n * n)?;

    let side = n as u32;
    let prepared = module.prepare_matmul::<T>(config(n))?;
    let mut launch = || {
        let width = RowWidth::new(&mut c, side);
        Ok(module.matmul(stream, &prepared, side, side, side, &a, &b, width)?)
    };
    let ms = time_ms(stream, runs_for(n), &mut launch)?;

    let error = max_error(&a_host, &b_host, &c.to_host_vec(stream)?, n);
    println!("  {n:6}  {name:<14} {ms:9.3} {:9.1} {error:10.1e}", gflops(n, ms));
    Ok(error)
}

/// The same for matmul_no_bounds, f32 only.
fn run_unchecked(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    n: usize,
) -> Result<f64, Error> {
    let mut state = 1;
    let a_host: Vec<f32> = random(n * n, &mut state);
    let b_host: Vec<f32> = random(n * n, &mut state);
    let a = DeviceBuffer::from_host(stream, &a_host)?;
    let b = DeviceBuffer::from_host(stream, &b_host)?;
    let mut c = DeviceBuffer::<f32>::zeroed(stream, n * n)?;

    let side = n as u32;
    let prepared = module.prepare_matmul_no_bounds(config(n))?;
    let mut launch = || {
        let width = RowWidth::new(&mut c, side);
        Ok(module.matmul_no_bounds(stream, &prepared, side, side, side, &a, &b, width)?)
    };
    let ms = time_ms(stream, runs_for(n), &mut launch)?;

    let error = max_error(&a_host, &b_host, &c.to_host_vec(stream)?, n);
    let name = "f32 unchecked";
    println!("  {n:6}  {name:<14} {ms:9.3} {:9.1} {error:10.1e}", gflops(n, ms));
    Ok(error)
}

/// The same for matmul_fixed, whose size is FIXED x FIXED.
fn run_fixed(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
) -> Result<f64, Error> {
    let n = FIXED;
    let mut state = 1;
    let a_host: Vec<f32> = random(n * n, &mut state);
    let b_host: Vec<f32> = random(n * n, &mut state);
    let a = DeviceBuffer::from_host(stream, &a_host)?;
    let b = DeviceBuffer::from_host(stream, &b_host)?;
    let mut c = DeviceBuffer::<f32>::zeroed(stream, n * n)?;

    let prepared = module.prepare_matmul_fixed(config(n))?;
    let mut launch = || Ok(module.matmul_fixed(stream, &prepared, &a, &b, &mut c)?);
    let ms = time_ms(stream, runs_for(n), &mut launch)?;

    let error = max_error(&a_host, &b_host, &c.to_host_vec(stream)?, n);
    let name = "f32 fixed width";
    println!("  {n:6}  {name:<14} {ms:9.3} {:9.1} {error:10.1e}", gflops(n, ms));
    Ok(error)
}

/// A launch that claims k = 2 n for buffers of n x n: the contract refuses it.
fn launch_too_long_k(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    n: usize,
) -> Result<(), Error> {
    let a = DeviceBuffer::<f32>::zeroed(stream, n * n)?;
    let b = DeviceBuffer::<f32>::zeroed(stream, n * n)?;
    let mut c = DeviceBuffer::<f32>::zeroed(stream, n * n)?;

    let side = n as u32;
    let prepared = module.prepare_matmul::<f32>(config(n))?;
    let width = RowWidth::new(&mut c, side);
    module.matmul(stream, &prepared, side, side, 2 * side, &a, &b, width)?;
    Ok(())
}

/// Launch 1,024 x 1,024 products for 0.5 s: the clocks rise under sustained load.
fn warm_up(module: &kernels::LoadedModule, stream: &Arc<CudaStream>) -> Result<(), Error> {
    let n = 1024;
    let a = DeviceBuffer::<f32>::zeroed(stream, n * n)?;
    let b = DeviceBuffer::<f32>::zeroed(stream, n * n)?;
    let mut c = DeviceBuffer::<f32>::zeroed(stream, n * n)?;

    let side = n as u32;
    let prepared = module.prepare_matmul::<f32>(config(n))?;
    let start = Instant::now();
    while start.elapsed().as_millis() < 500 {
        let width = RowWidth::new(&mut c, side);
        module.matmul(stream, &prepared, side, side, side, &a, &b, width)?;
        stream.synchronize()?;
    }
    Ok(())
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };

    warm_up(&module, &stream)?;

    println!("C = A B, n x n, 1 thread per element of C, blocks of 16 x 16 threads");
    println!("       n  kernel                ms   GFLOP/s  max error");

    let mut f32_errors = Vec::new();
    let mut f64_errors = Vec::new();

    f32_errors.push(run_fixed(&module, &stream)?);
    for n in [512, 1000, 1024, 2048, 4096] {
        f32_errors.push(run_generic::<f32>(&module, &stream, "f32", n)?);
        f32_errors.push(run_unchecked(&module, &stream, n)?);
    }
    for n in [512, 1000, 1024, 2048] {
        f64_errors.push(run_generic::<f64>(&module, &stream, "f64", n)?);
    }

    let refused = match launch_too_long_k(&module, &stream, 256) {
        Err(error) => {
            println!("k = 2 n refused before launch: {error}");
            true
        }
        Ok(()) => false,
    };

    let f32_ok = f32_errors.iter().all(|&e| e < 1e-3);
    let f64_ok = f64_errors.iter().all(|&e| e < 1e-9);
    let passed = f32_ok && f64_ok && refused;
    println!("{}", if passed { "PASSED" } else { "FAILED" });
    Ok(())
}
