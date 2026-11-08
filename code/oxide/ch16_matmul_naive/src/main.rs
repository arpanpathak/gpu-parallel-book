//! Chapter 16: matrix multiplication in Rust, 1 thread per entry of C.
//!
//!     cargo oxide run

use std::ops::{Add, Mul};
use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy, LaunchConfig2D};
use cuda_device::thread::{Index2D, Runtime2DIndex};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
use cuda_device::{launch_bounds, launch_contract};
use cuda_host::RowWidth;

const FIXED: usize = 1024; // the width of C in the kernel with a compile-time width

type Error = Box<dyn std::error::Error>;
type Module = kernels::LoadedModule;

#[cuda_module]
mod kernels {
    use super::*;

    /// C (m x n) = A (m x k) B (k x n), all row-major. 1 thread computes 1 entry of C.
    /// The width of `c` is bound on the host with `RowWidth`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 2,
        coordinates = u32,
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
            let mut inner = 0;
            #[unroll(4)]
            while inner < k {
                sum = sum + a[row * k + inner] * b[inner * n + col];
                inner += 1;
            }
            *out = sum;
        }
    }

    /// The same product for FIXED x FIXED matrices: the width is part of the type of `c`.
    #[kernel]
    #[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1))]
    pub fn matmul_fixed(a: &[f32], b: &[f32], mut c: DisjointSlice<f32, Index2D<FIXED>>) {
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();

        if let Some(idx) = thread::index_2d::<FIXED>()
            && let Some(out) = c.get_mut(idx)
        {
            let mut sum = 0.0;
            let mut inner = 0;
            #[unroll(4)]
            while inner < FIXED {
                sum += a[row * FIXED + inner] * b[inner * FIXED + col];
                inner += 1;
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

/// Integers in [-5, 5] from a small linear congruential generator: every partial sum
/// of a product with k <= 4096 is an integer below 2^24, so f32 results are exact.
fn integers<T: From<f32>>(len: usize, state: &mut u32) -> Vec<T> {
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let x = ((*state >> 8) % 11) as f32 - 5.0;
        values.push(T::from(x));
    }
    values
}

/// The number of 64 sampled entries of C that equal the exact product.
fn exact_entries<T: Copy + Into<f64>>(a: &[T], b: &[T], c: &[T], n: usize) -> usize {
    let mut exact = 0;
    for s in 0..64 {
        let i = s * 977 % n;
        let j = (s * 1231 + 7) % n;

        let mut sum = 0.0;
        for p in 0..n {
            sum += a[i * n + p].into() * b[p * n + j].into();
        }

        let got: f64 = c[i * n + j].into();
        if got == sum {
            exact += 1;
        }
    }
    exact
}

/// Launches that keep n^3 x runs near 2^33, so each size runs for a similar time.
fn runs_for(n: usize) -> u32 {
    ((1usize << 33) / (n * n * n)).clamp(1, 50) as u32
}

/// A grid of side x side blocks that covers an m x n matrix C.
fn config(m: usize, n: usize, side: u32) -> LaunchConfig2D {
    let grid = ((n as u32).div_ceil(side), (m as u32).div_ceil(side));
    LaunchConfig2D::new(grid, (side, side), 0)
}

/// The 4 x 4 example: blocks of 2 x 2 threads, printed, and compared with the hand result.
fn run_example(module: &Module, stream: &Arc<CudaStream>) -> Result<bool, Error> {
    let a_int = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let b_int = [1, 0, 2, 1, 0, 1, 1, 2, 2, 1, 0, 1, 1, 2, 1, 0];
    let want = [11, 13, 8, 8, 27, 29, 24, 24, 43, 45, 40, 40, 59, 61, 56, 56];
    let a_host: Vec<f32> = a_int.iter().map(|&x| x as f32).collect();
    let b_host: Vec<f32> = b_int.iter().map(|&x| x as f32).collect();

    let a = DeviceBuffer::from_host(stream, &a_host)?;
    let b = DeviceBuffer::from_host(stream, &b_host)?;
    let mut c = DeviceBuffer::<f32>::zeroed(stream, 16)?;

    let prepared = module.prepare_matmul::<f32>(config(4, 4, 2))?;
    let width = RowWidth::new(&mut c, 4);
    module.matmul(stream, &prepared, 4, 4, 4, &a, &b, width)?;
    let result = c.to_host_vec(stream)?;

    println!("C = A B on the 4 x 4 example, a grid of 2 x 2 blocks of 2 x 2 threads");
    for row in result.chunks(4) {
        println!("  {:4} {:4} {:4} {:4}", row[0], row[1], row[2], row[3]);
    }
    let same = result.iter().zip(want).all(|(&got, w)| got == w as f32);
    let answer = if same { "yes" } else { "no" };
    println!("equal to the product computed by hand: {answer}\n");
    Ok(same)
}

/// Print 1 line of the table for an n x n product of type T that took `ms` milliseconds.
fn report<T>(name: &str, n: usize, ms: f64, exact: usize) {
    let flop = 2.0 * (n * n * n) as f64;
    let gflops = flop / ms / 1e6;
    let fetched = flop * size_of::<T>() as f64 / ms / 1e6;
    println!("  {n:5}  {name:<10} {ms:9.3} {gflops:9.1} {fetched:10.1} {exact:6} of 64");
}

/// Time matmul::<T> on n x n matrices, print a line, and return true if the check passed.
fn run_generic<T>(
    module: &Module,
    stream: &Arc<CudaStream>,
    name: &str,
    n: usize,
) -> Result<bool, Error>
where
    T: DeviceCopy + Copy + Default + Add<Output = T> + Mul<Output = T>,
    T: From<f32> + Into<f64>,
{
    let mut state = 1;
    let a_host: Vec<T> = integers(n * n, &mut state);
    let b_host: Vec<T> = integers(n * n, &mut state);
    let a = DeviceBuffer::from_host(stream, &a_host)?;
    let b = DeviceBuffer::from_host(stream, &b_host)?;
    let mut c = DeviceBuffer::<T>::zeroed(stream, n * n)?;

    let side = n as u32;
    let prepared = module.prepare_matmul::<T>(config(n, n, 16))?;
    let mut launch = || {
        let width = RowWidth::new(&mut c, side);
        Ok(module.matmul(stream, &prepared, side, side, side, &a, &b, width)?)
    };
    let ms = time_ms(stream, runs_for(n), &mut launch)?;

    let exact = exact_entries(&a_host, &b_host, &c.to_host_vec(stream)?, n);
    report::<T>(name, n, ms, exact);
    Ok(exact == 64)
}

/// The same for matmul_fixed, whose size is FIXED x FIXED.
fn run_fixed(module: &Module, stream: &Arc<CudaStream>) -> Result<bool, Error> {
    let n = FIXED;
    let mut state = 1;
    let a_host: Vec<f32> = integers(n * n, &mut state);
    let b_host: Vec<f32> = integers(n * n, &mut state);
    let a = DeviceBuffer::from_host(stream, &a_host)?;
    let b = DeviceBuffer::from_host(stream, &b_host)?;
    let mut c = DeviceBuffer::<f32>::zeroed(stream, n * n)?;

    let prepared = module.prepare_matmul_fixed(config(n, n, 16))?;
    let mut launch = || Ok(module.matmul_fixed(stream, &prepared, &a, &b, &mut c)?);
    let ms = time_ms(stream, runs_for(n), &mut launch)?;

    let exact = exact_entries(&a_host, &b_host, &c.to_host_vec(stream)?, n);
    report::<f32>("f32 fixed", n, ms, exact);
    Ok(exact == 64)
}

/// A launch that claims k = 2 n for buffers of n x n: the contract refuses it.
fn launch_too_long_k(
    module: &Module,
    stream: &Arc<CudaStream>,
    n: usize,
) -> Result<(), Error> {
    let a = DeviceBuffer::<f32>::zeroed(stream, n * n)?;
    let b = DeviceBuffer::<f32>::zeroed(stream, n * n)?;
    let mut c = DeviceBuffer::<f32>::zeroed(stream, n * n)?;

    let side = n as u32;
    let prepared = module.prepare_matmul::<f32>(config(n, n, 16))?;
    let width = RowWidth::new(&mut c, side);
    module.matmul(stream, &prepared, side, side, 2 * side, &a, &b, width)?;
    Ok(())
}

/// About 0.5 s of 1024 x 1024 products: the GPU raises its clocks under sustained load.
fn warm_up(module: &Module, stream: &Arc<CudaStream>) -> Result<(), Error> {
    let n = 1024;
    let a = DeviceBuffer::<f32>::zeroed(stream, n * n)?;
    let b = DeviceBuffer::<f32>::zeroed(stream, n * n)?;
    let mut c = DeviceBuffer::<f32>::zeroed(stream, n * n)?;

    let side = n as u32;
    let prepared = module.prepare_matmul::<f32>(config(n, n, 16))?;
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

    let mut passed = run_example(&module, &stream)?;
    warm_up(&module, &stream)?;

    println!("C = A B, n x n, 1 thread per entry of C, blocks of 16 x 16 threads");
    println!("      n  type              ms   GFLOP/s  fetched GB/s   exact");
    for n in [512, 1024, 2048, 4096] {
        passed &= run_generic::<f32>(&module, &stream, "f32", n)?;
    }
    passed &= run_fixed(&module, &stream)?;
    for n in [512, 1024, 2048] {
        passed &= run_generic::<f64>(&module, &stream, "f64", n)?;
    }

    match launch_too_long_k(&module, &stream, 256) {
        Err(error) => println!("k = 2 n refused before launch: {error}"),
        Ok(()) => passed = false,
    }

    println!("{}", if passed { "PASSED" } else { "FAILED" });
    Ok(())
}
