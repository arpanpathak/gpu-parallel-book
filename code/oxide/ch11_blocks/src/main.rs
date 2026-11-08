//! Chapter 11 in Rust: block_reduce and block_scan, and closures in place of Thrust calls.
//!
//!     cargo oxide run

use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::cooperative_groups::ops::Sum;
use cuda_device::cooperative_groups::{block_reduce, block_scan, this_thread_block};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, device, kernel, launch_contract};
use cuda_device::{thread, warp};

const BLOCK: u32 = 256;
const GRID: u32 = 1024;

type Error = Box<dyn std::error::Error>;

#[cuda_module]
mod kernels {
    use super::*;

    #[device]
    fn warp_sum(mut v: f32) -> f32 {
        for offset in [16, 8, 4, 2, 1] {
            v += warp::shuffle_down_f32(v, offset);
        }
        v
    }

    /// Chapter 8's sum: each warp shuffles, 1 barrier per block, written by hand.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn sum_hand(input: &[f32], mut partial: DisjointSlice<f32>) {
        static mut WARPS: SharedArray<f32, 8> = SharedArray::UNINIT;
        let stride = (thread::gridDim_x() * BLOCK) as usize;
        let lane = warp::lane_id() as usize;
        let warp = thread::threadIdx_x() as usize / 32;

        let mut sum = 0.0;
        let mut i = thread::index_1d().get();
        while i < input.len() {
            sum += input[i];
            i += stride;
        }

        sum = warp_sum(sum);
        if lane == 0 {
            // SAFETY: lane 0 of each warp writes its own slot, before the barrier.
            unsafe { WARPS[warp] = sum };
        }
        thread::sync_threads();

        if warp == 0 {
            // SAFETY: after the barrier the slots are only read.
            sum = warp_sum(if lane < 8 { unsafe { WARPS[lane] } } else { 0.0 });
            if lane == 0 {
                // SAFETY: only thread 0 of each block writes, at its own block number.
                unsafe { *partial.get_unchecked_mut(thread::blockIdx_x() as usize) = sum };
            }
        }
    }

    /// The same sum, with cuda-oxide's block_reduce in place of warp_sum and the slots.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn sum_blocks(input: &[f32], mut partial: DisjointSlice<f32>) {
        static mut WARPS: SharedArray<f32, 8> = SharedArray::UNINIT;
        let stride = (thread::gridDim_x() * BLOCK) as usize;
        let block = this_thread_block();

        let mut sum = 0.0;
        let mut i = thread::index_1d().get();
        while i < input.len() {
            sum += input[i];
            i += stride;
        }

        sum = block_reduce::<f32, Sum, 8>(&block, sum, &raw mut WARPS);
        if thread::threadIdx_x() == 0 {
            // SAFETY: only thread 0 of each block writes, at its own block number.
            unsafe { *partial.get_unchecked_mut(thread::blockIdx_x() as usize) = sum };
        }
    }

    /// An inclusive scan of each block of 256 values, 1 value per thread.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn scan_blocks(input: &[u32], mut out: DisjointSlice<u32>) {
        static mut WARPS: SharedArray<u32, 8> = SharedArray::UNINIT;
        let block = this_thread_block();
        let i = thread::index_1d();

        let v = if i.get() < input.len() { input[i.get()] } else { 0 };
        let scanned = block_scan::<u32, Sum, 8>(&block, v, &raw mut WARPS);
        if let Some(slot) = out.get_mut(i) {
            *slot = scanned;
        }
    }

    /// out[i] = f(input[i]), 1 element per thread: the work of thrust::transform.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn map<F>(f: F, input: &[f32], mut out: DisjointSlice<f32>)
    where
        F: Fn(f32) -> f32 + Copy,
    {
        let idx = thread::index_1d();
        let i = idx.get();

        if let Some(slot) = out.get_mut(idx) {
            *slot = f(input[i]);
        }
    }

    /// 1 partial sum of f(input[i]) per block, each f(x) kept in a register: the work
    /// of thrust::transform_reduce.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn map_sum<F>(f: F, input: &[f32], mut partial: DisjointSlice<f32>)
    where
        F: Fn(f32) -> f32 + Copy,
    {
        static mut WARPS: SharedArray<f32, 8> = SharedArray::UNINIT;
        let stride = (thread::gridDim_x() * BLOCK) as usize;
        let block = this_thread_block();

        let mut sum = 0.0;
        let mut i = thread::index_1d().get();
        while i < input.len() {
            sum += f(input[i]);
            i += stride;
        }

        sum = block_reduce::<f32, Sum, 8>(&block, sum, &raw mut WARPS);

        if thread::threadIdx_x() == 0 {
            let b = thread::blockIdx_x() as usize;
            // SAFETY: only thread 0 of each block writes, at its own block number.
            unsafe { *partial.get_unchecked_mut(b) = sum };
        }
    }
}

/// Average time of 20 runs of `launch`, in milliseconds, after 1 warm-up run.
fn time_ms(
    stream: &Arc<CudaStream>,
    mut launch: impl FnMut() -> Result<(), Error>,
) -> Result<f64, Error> {
    launch()?;
    stream.synchronize()?;

    let start = Instant::now();
    for _ in 0..20 {
        launch()?;
    }
    stream.synchronize()?;

    Ok(start.elapsed().as_secs_f64() * 1e3 / 20.0)
}

/// Numbers from a small linear congruential generator, as in Chapter 8.
fn random(n: usize, seed: u32) -> Vec<u32> {
    let mut state = seed;

    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        })
        .collect()
}

/// The sum of squares of 64 Mi floats in 2 passes and in 1.
fn sums_of_squares(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    values: &[f32],
) -> Result<bool, Error> {
    let n = values.len();
    let exact: f64 = values.iter().map(|&v| v as f64 * v as f64).sum();

    let input = DeviceBuffer::from_host(stream, values)?;
    let mut squares = DeviceBuffer::<f32>::zeroed(stream, n)?;
    let mut partial = DeviceBuffer::<f32>::zeroed(stream, GRID as usize)?;
    let mut total = DeviceBuffer::<f32>::zeroed(stream, 1)?;

    let each = module.prepare_map(LaunchConfig1D::new(n as u32 / BLOCK, BLOCK, 0))?;
    let grid = module.prepare_map_sum(LaunchConfig1D::new(GRID, BLOCK, 0))?;
    let fused = module.prepare_map_sum(LaunchConfig1D::new(GRID, BLOCK, 0))?;
    let one = module.prepare_map_sum(LaunchConfig1D::new(1, BLOCK, 0))?;

    let square = |v: f32| v * v;
    let same = |v: f32| v;

    let gb = (n * 4) as f64 / 1e9;
    println!("sum of squares, GB/s of input read");

    let ms = time_ms(stream, || {
        module.map(stream, &each, square, &input, &mut squares)?;
        module.map_sum(stream, &grid, same, &squares, &mut partial)?;
        Ok(module.map_sum(stream, &one, same, &partial, &mut total)?)
    })?;
    let two_passes = total.to_host_vec(stream)?[0];
    let rate = gb / ms * 1e3;
    println!("  map, then map_sum    {ms:6.2} ms  {rate:6.1} GB/s  sum {two_passes:.1}");

    let ms = time_ms(stream, || {
        module.map_sum(stream, &fused, square, &input, &mut partial)?;
        Ok(module.map_sum(stream, &one, same, &partial, &mut total)?)
    })?;
    let one_pass = total.to_host_vec(stream)?[0];
    let rate = gb / ms * 1e3;
    println!("  map_sum, fused       {ms:6.2} ms  {rate:6.1} GB/s  sum {one_pass:.1}");
    println!("  exact (f64)                               sum {exact:.1}");

    let close = |s: f32| (s as f64 - exact).abs() < 1e-5 * exact;
    Ok(close(two_passes) && close(one_pass))
}

/// A sum of 4,096 floats in 3 ways: storage allocated for each sum and the result
/// copied to the host, as thrust::reduce does; storage allocated once and the result
/// copied; storage allocated once and the result kept on the GPU.
fn small_sums(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    values: &[f32],
) -> Result<(), Error> {
    let n = 4096;
    let blocks = 16;
    let input = DeviceBuffer::from_host(stream, &values[..n])?;

    let grid = module.prepare_map_sum(LaunchConfig1D::new(blocks, BLOCK, 0))?;
    let one = module.prepare_map_sum(LaunchConfig1D::new(1, BLOCK, 0))?;
    let same = |v: f32| v;

    println!("sum of {n} floats, microseconds per sum");

    let ms = time_ms(stream, || {
        let mut partial = DeviceBuffer::<f32>::zeroed(stream, blocks as usize)?;
        let mut total = DeviceBuffer::<f32>::zeroed(stream, 1)?;
        module.map_sum(stream, &grid, same, &input, &mut partial)?;
        module.map_sum(stream, &one, same, &partial, &mut total)?;
        total.to_host_vec(stream)?;
        Ok(())
    })?;
    println!("  allocated each time, result copied  {:6.1} us", ms * 1e3);

    let mut partial = DeviceBuffer::<f32>::zeroed(stream, blocks as usize)?;
    let mut total = DeviceBuffer::<f32>::zeroed(stream, 1)?;

    let ms = time_ms(stream, || {
        module.map_sum(stream, &grid, same, &input, &mut partial)?;
        module.map_sum(stream, &one, same, &partial, &mut total)?;
        total.to_host_vec(stream)?;
        Ok(())
    })?;
    println!("  allocated once, result copied       {:6.1} us", ms * 1e3);

    let ms = time_ms(stream, || {
        module.map_sum(stream, &grid, same, &input, &mut partial)?;
        Ok(module.map_sum(stream, &one, same, &partial, &mut total)?)
    })?;
    println!("  allocated once, result kept         {:6.1} us", ms * 1e3);
    Ok(())
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };
    let n = 64 << 20;

    // 1. A sum of 67,108,864 floats in [0, 1), by hand and with block_reduce.
    let values: Vec<f32> = random(n, 1)
        .iter()
        .map(|&r| (r >> 8) as f32 / 16_777_216.0)
        .collect();
    let exact: f64 = values.iter().map(|&v| v as f64).sum();

    let input = DeviceBuffer::from_host(&stream, &values)?;
    let mut partial = DeviceBuffer::<f32>::zeroed(&stream, GRID as usize)?;
    let mut total = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let grid = LaunchConfig1D::new(GRID, BLOCK, 0);
    let one = LaunchConfig1D::new(1, BLOCK, 0);
    let hand = (module.prepare_sum_hand(grid)?, module.prepare_sum_hand(one)?);
    let blocks = (module.prepare_sum_blocks(grid)?, module.prepare_sum_blocks(one)?);

    let mut by_hand = || -> Result<(), Error> {
        module.sum_hand(&stream, &hand.0, &input, &mut partial)?;
        Ok(module.sum_hand(&stream, &hand.1, &partial, &mut total)?)
    };
    for _ in 0..10 {
        time_ms(&stream, &mut by_hand)?; // warm up the clocks
    }

    let gb = (n * 4) as f64 / 1e9;
    println!("sum of {n} floats, average of 20 runs");

    let ms = time_ms(&stream, &mut by_hand)?;
    let rate = gb / ms * 1e3;
    let sum_hand = total.to_host_vec(&stream)?[0];
    println!("  Chapter 8, by hand  {ms:6.2} ms  {rate:6.1} GB/s  sum {sum_hand:.1}");

    let ms = time_ms(&stream, || {
        module.sum_blocks(&stream, &blocks.0, &input, &mut partial)?;
        Ok(module.sum_blocks(&stream, &blocks.1, &partial, &mut total)?)
    })?;
    let rate = gb / ms * 1e3;
    let sum_blocks = total.to_host_vec(&stream)?[0];
    println!("  block_reduce        {ms:6.2} ms  {rate:6.1} GB/s  sum {sum_blocks:.1}");
    println!("  exact (f64)                               sum {exact:.1}");

    // 2. Each block of 256 values 0 to 3, scanned with block_scan.
    let counts: Vec<u32> = random(n, 2).iter().map(|&r| r >> 30).collect();
    let input = DeviceBuffer::from_host(&stream, &counts)?;
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, n)?;
    let config = LaunchConfig1D::new((n as u32).div_ceil(BLOCK), BLOCK, 0);
    let scan = module.prepare_scan_blocks(config)?;
    module.scan_blocks(&stream, &scan, &input, &mut out)?;

    let result = out.to_host_vec(&stream)?;
    let mut running = 0;
    let mut scan_ok = true;
    for (i, (&c, &r)) in counts.iter().zip(&result).enumerate() {
        if i % BLOCK as usize == 0 {
            running = 0;
        }
        running += c;
        scan_ok &= r == running;
    }
    println!("block_scan of {n} values, 256 per block");
    println!("  first block: {:?} ... {}", &result[..8], result[255]);

    // 3. The sum of squares in 2 passes and in 1, and 3 ways to sum 4,096 floats.
    let squares_ok = sums_of_squares(&module, &stream, &values)?;
    small_sums(&module, &stream, &values)?;

    let close = |s: f32| (s as f64 - exact).abs() < 1e-5 * exact;
    let ok = close(sum_hand) && close(sum_blocks) && scan_ok && squares_ok;
    println!("{}", if ok { "PASSED" } else { "FAILED" });
    Ok(())
}
