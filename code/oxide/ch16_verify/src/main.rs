//! Chapter 16 in Rust: GPU results against the CPU, with a tolerance worked out in advance.
//!
//!     cargo oxide run

use std::f32::consts::LOG2_E;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, float, kernel, launch_contract};
use cuda_device::thread;

type Error = Box<dyn std::error::Error>;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn exp_accurate(x: &[f32], mut y: DisjointSlice<f32>) {
        let i = thread::index_1d();
        let k = i.get();
        if let Some(v) = y.get_mut(i) {
            *v = x[k].exp();
        }
    }

    /// exp(x) as 2^(x log2 e), with the hardware's approximate 2^x, as __expf computes it.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn exp_fast(x: &[f32], mut y: DisjointSlice<f32>) {
        let i = thread::index_1d();
        let k = i.get();
        if let Some(v) = y.get_mut(i) {
            *v = float::ex2_approx_ftz_f32(x[k] * LOG2_E);
        }
    }

    /// Each thread sums a strided share of x; the block adds the 256 sums as a tree.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn sum(x: &[f32], n: u32, mut block_sums: DisjointSlice<f32>) {
        static mut S: SharedArray<f32, 256> = SharedArray::UNINIT;
        let t = thread::threadIdx_x() as usize;
        let block = thread::blockIdx_x() as usize;
        let grid = thread::gridDim_x() as usize * 256;

        let mut v = 0.0;
        let mut i = block * 256 + t;
        while i < n as usize {
            v += x[i];
            i += grid;
        }

        // SAFETY: each thread writes only its own cell, before the barrier.
        unsafe { S[t] = v };
        thread::sync_threads();

        let mut step: usize = 0;
        #[unroll]
        while step < 8 {
            let stride: usize = 128 >> step;
            if t < stride {
                // SAFETY: in this step thread t alone writes cell t, and it reads cell
                // t + stride, which no thread writes in this step.
                unsafe { S[t] += S[t + stride] };
            }
            thread::sync_threads();
            step += 1;
        }

        if t == 0 {
            // SAFETY: only thread 0 of each block writes, at its own block number.
            unsafe { *block_sums.get_unchecked_mut(block) = S[0] };
        }
    }
}

/// The number of floats between a and b, for 2 floats of the same sign.
fn ulps(a: f32, b: f32) -> i64 {
    let ia = a.to_bits() as i32 as i64;
    let ib = b.to_bits() as i32 as i64;
    (ia - ib).abs()
}

/// The largest distance in ULPs between y and the nearest float to exp(x).
fn exp_error(x: &[f32], y: &[f32]) -> i64 {
    let mut worst = 0;
    for (&x, &y) in x.iter().zip(y) {
        let exact = (x as f64).exp() as f32;
        worst = worst.max(ulps(y, exact));
    }
    worst
}

/// 2 launches: 1,024 blocks write 1,024 partial sums, then 1 block adds them.
fn gpu_sum(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
    x: &DeviceBuffer<f32>,
    n: u32,
    partial: &mut DeviceBuffer<f32>,
    total: &mut DeviceBuffer<f32>,
) -> Result<f32, Error> {
    let many = module.prepare_sum(LaunchConfig1D::new(1024, 256, 0))?;
    let one = module.prepare_sum(LaunchConfig1D::new(1, 256, 0))?;
    module.sum(stream, &many, x, n, partial)?;
    module.sum(stream, &one, partial, 1024, total)?;

    Ok(total.to_host_vec(stream)?[0])
}

fn exp_test(stream: &CudaStream, module: &kernels::LoadedModule) -> Result<(), Error> {
    let n = 1 << 20;
    let x: Vec<f32> = (0..n).map(|i| -10.0 + 20.0 * i as f32 / n as f32).collect();

    let dx = DeviceBuffer::from_host(stream, &x)?;
    let mut dy = DeviceBuffer::<f32>::zeroed(stream, n)?;
    let blocks = n.div_ceil(256) as u32;

    println!("y = exp(x) for 1,048,576 values of x in [-10, 10)");

    let accurate = module.prepare_exp_accurate(LaunchConfig1D::new(blocks, 256, 0))?;
    module.exp_accurate(stream, &accurate, &dx, &mut dy)?;
    let error = exp_error(&x, &dy.to_host_vec(stream)?);
    println!("  exp           largest error {error:3} ULPs");

    let fast = module.prepare_exp_fast(LaunchConfig1D::new(blocks, 256, 0))?;
    module.exp_fast(stream, &fast, &dx, &mut dy)?;
    let error = exp_error(&x, &dy.to_host_vec(stream)?);
    println!("  ex2_approx    largest error {error:3} ULPs");
    Ok(())
}

fn verdict(error: f64, tolerance: f64) -> &'static str {
    if error <= tolerance { "PASSED" } else { "FAILED" }
}

/// Sum x on the GPU twice: correctly, and as a caller passing 1 element too few.
fn sum_test(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
    title: &str,
    x: &[f32],
    tolerance: f64,
) -> Result<(), Error> {
    let n = x.len() as u32;
    let dx = DeviceBuffer::from_host(stream, x)?;
    let mut partial = DeviceBuffer::<f32>::zeroed(stream, 1024)?;
    let mut total = DeviceBuffer::<f32>::zeroed(stream, 1)?;

    let exact: f64 = x.iter().map(|&v| v as f64).sum();

    let good_sum = gpu_sum(stream, module, &dx, n, &mut partial, &mut total)?;
    let bad_sum = gpu_sum(stream, module, &dx, n - 1, &mut partial, &mut total)?;
    let good = (good_sum as f64 - exact).abs();
    let bad = (bad_sum as f64 - exact).abs();

    println!("{title}");
    println!("  exact sum       {exact:12.2}");
    println!("  tolerance       {tolerance:12.2}");
    println!("  n elements      error {good:6.2}  {}", verdict(good, tolerance));
    println!("  n - 1 elements  error {bad:6.2}  {}", verdict(bad, tolerance));
    Ok(())
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };

    exp_test(&stream, &module)?;

    let n = 1 << 24;
    let mut state: u32 = 1;
    let mut magnitude = 0.0;
    let mut random = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let v = (state >> 8) as f32 / 16_777_216.0;
        magnitude += v as f64;
        random.push(v);
    }

    // Each element passes through at most 84 additions on the way to the total:
    // 64 + 8 in launch 1, 4 + 8 in launch 2. Each rounds by 2^-24 of its sum at most.
    let bound = 84.0 * 2f64.powi(-24) * magnitude;
    let title = "sum of 16,777,216 random floats in [0, 1)";
    sum_test(&stream, &module, title, &random, bound)?;

    // Partial sums of ones are whole numbers up to 2^24, which floats hold exactly.
    let ones = vec![1.0f32; n];
    sum_test(&stream, &module, "sum of 16,777,216 ones", &ones, 0.0)?;
    Ok(())
}
