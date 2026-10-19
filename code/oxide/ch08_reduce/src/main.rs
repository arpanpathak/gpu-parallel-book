//! Chapter 8 in Rust: a sum by warp shuffles, a scan, and a histogram of 4-byte words.
//!
//!     cargo oxide run

use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, BlockAtomicU32, DeviceAtomicU32};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, device, kernel, launch_contract};
use cuda_device::{thread, warp};

const BLOCK: u32 = 256;
const GRID: u32 = 1024;

type Error = Box<dyn std::error::Error>;
type Levels = Vec<(DeviceBuffer<u32>, DeviceBuffer<u32>)>;

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

    #[device]
    fn warp_scan(mut v: u32) -> u32 {
        let lane = warp::lane_id();
        for offset in [1, 2, 4, 8, 16] {
            let left = warp::shuffle_up_sync(u32::MAX, v, offset);
            if lane >= offset {
                v += left;
            }
        }
        v
    }

    /// Each thread sums many elements, each warp shuffles, 1 barrier per block.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn sum_shuffle(input: &[f32], mut partial: DisjointSlice<f32>) {
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

    /// An inclusive scan of 1,024 values per block, 4 per thread.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn scan_blocks(
        input: &[u32],
        mut out: DisjointSlice<u32>,
        mut sums: DisjointSlice<u32>,
    ) {
        static mut WARPS: SharedArray<u32, 8> = SharedArray::UNINIT;
        let first = thread::index_1d().get() * 4;
        let lane = warp::lane_id() as usize;
        let warp = thread::threadIdx_x() as usize / 32;

        let mut a = [0u32; 4];
        let mut own = 0;
        for k in 0..4 {
            if first + k < input.len() {
                own += input[first + k];
            }
            a[k] = own;
        }

        let v = warp_scan(own);
        if lane == 31 {
            // SAFETY: lane 31 of each warp writes its own slot, before the barrier.
            unsafe { WARPS[warp] = v };
        }
        thread::sync_threads();

        if warp == 0 {
            // SAFETY: only warp 0 touches the slots between the 2 barriers.
            let w = warp_scan(if lane < 8 { unsafe { WARPS[lane] } } else { 0 });
            if lane < 8 {
                unsafe { WARPS[lane] = w };
            }
        }
        thread::sync_threads();

        // SAFETY: after the second barrier the slots are only read.
        let before = v - own + if warp > 0 { unsafe { WARPS[warp - 1] } } else { 0 };
        for k in 0..4 {
            if first + k < out.len() {
                // SAFETY: thread t alone writes elements 4t to 4t + 3.
                unsafe { *out.get_unchecked_mut(first + k) = before + a[k] };
            }
        }
        if thread::threadIdx_x() == BLOCK - 1 {
            let block = thread::blockIdx_x() as usize;
            // SAFETY: only the last thread of each block writes, at its own block number.
            unsafe { *sums.get_unchecked_mut(block) = before + a[3] };
        }
    }

    /// Adds to every value of block b the total of blocks 0 to b - 1.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn add_offsets(mut out: DisjointSlice<u32>, scanned: &[u32]) {
        let block = thread::blockIdx_x() as usize;
        let first = thread::index_1d().get() * 4;

        if block > 0 {
            for k in 0..4 {
                if first + k < out.len() {
                    // SAFETY: thread t alone writes elements 4t to 4t + 3.
                    unsafe { *out.get_unchecked_mut(first + k) += scanned[block - 1] };
                }
            }
        }
    }

    /// A histogram of bytes in shared bins, reading 4 bytes per load.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn hist_words(words: &[u32], bins: &[DeviceAtomicU32]) {
        static mut LOCAL: SharedArray<u32, 256> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;

        // SAFETY: each thread zeroes only its own bin, before the barrier.
        unsafe { LOCAL[tid] = 0 };
        thread::sync_threads();

        let stride = (thread::gridDim_x() * BLOCK) as usize;
        let mut i = thread::index_1d().get();
        while i < words.len() {
            for k in 0..4 {
                let byte = (words[i] >> (8 * k)) & 255;
                // SAFETY: between the 2 barriers every access to LOCAL is atomic.
                let bin = unsafe {
                    let base = (*&raw mut LOCAL).as_mut_ptr();
                    BlockAtomicU32::from_ptr(base.add(byte as usize))
                };
                bin.fetch_add(1, AtomicOrdering::Relaxed);
            }
            i += stride;
        }
        thread::sync_threads();

        // SAFETY: after the barrier no thread writes LOCAL again.
        let local = unsafe { LOCAL[tid] };
        bins[tid].fetch_add(local, AtomicOrdering::Relaxed);
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

/// Numbers from a small linear congruential generator, so the run needs no extra crate.
fn random(n: usize, seed: u32) -> Vec<u32> {
    let mut state = seed;

    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        })
        .collect()
}

/// Buffers for the block totals and their scan, per level: n / 1,024, / 1,024 again, ... 1.
fn levels(stream: &Arc<CudaStream>, mut n: usize) -> Result<Levels, Error> {
    let mut levels = Vec::new();
    while n > 1 {
        n = n.div_ceil(1024);
        let totals = DeviceBuffer::<u32>::zeroed(stream, n)?;
        let scanned = DeviceBuffer::<u32>::zeroed(stream, n)?;
        levels.push((totals, scanned));
    }
    Ok(levels)
}

/// Scans each block, then the block totals (by the same function), then adds them.
fn scan(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    input: &DeviceBuffer<u32>,
    out: &mut DeviceBuffer<u32>,
    levels: &mut [(DeviceBuffer<u32>, DeviceBuffer<u32>)],
) -> Result<(), Error> {
    let blocks = input.len().div_ceil(1024) as u32;
    let config = LaunchConfig1D::new(blocks, BLOCK, 0);
    let ((totals, scanned), rest) = levels.split_first_mut().ok_or("no level left")?;
    module.scan_blocks(stream, &module.prepare_scan_blocks(config)?, input, out, totals)?;

    if blocks > 1 {
        scan(module, stream, totals, scanned, rest)?;
        module.add_offsets(stream, &module.prepare_add_offsets(config)?, out, scanned)?;
    }
    Ok(())
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };
    let n = 64 << 20;

    // 1. A sum of 67,108,864 floats in [0, 1).
    let values: Vec<f32> = random(n, 1)
        .iter()
        .map(|&r| (r >> 8) as f32 / 16_777_216.0)
        .collect();
    let input = DeviceBuffer::from_host(&stream, &values)?;
    let mut partial = DeviceBuffer::<f32>::zeroed(&stream, GRID as usize)?;
    let mut total = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let blocks = module.prepare_sum_shuffle(LaunchConfig1D::new(GRID, BLOCK, 0))?;
    let last = module.prepare_sum_shuffle(LaunchConfig1D::new(1, BLOCK, 0))?;

    let mut sum = || -> Result<(), Error> {
        module.sum_shuffle(&stream, &blocks, &input, &mut partial)?;
        Ok(module.sum_shuffle(&stream, &last, &partial, &mut total)?)
    };
    for _ in 0..10 {
        time_ms(&stream, &mut sum)?; // warm up the clocks
    }
    let ms = time_ms(&stream, &mut sum)?;

    let gpu = total.to_host_vec(&stream)?[0];
    let exact: f64 = values.iter().map(|&v| v as f64).sum();
    let gb = (n * 4) as f64 / 1e9 / (ms / 1e3);
    println!("sum of {n} floats, average of 20 runs");
    println!("  shuffles      {ms:6.2} ms  {gb:6.1} GB/s  sum {gpu:.1}");
    println!("  exact (f64)                         sum {exact:.1}");

    // 2. An inclusive scan of values 0 to 3.
    let counts: Vec<u32> = random(n, 2).iter().map(|&r| r >> 30).collect();
    let input = DeviceBuffer::from_host(&stream, &counts)?;
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, n)?;
    let mut levels = levels(&stream, n)?;

    let ms = time_ms(&stream, || scan(&module, &stream, &input, &mut out, &mut levels))?;

    let result = out.to_host_vec(&stream)?;
    let mut running = 0u32;
    let correct = counts.iter().zip(&result).all(|(&c, &r)| {
        running += c;
        r == running
    });
    let gb = (2 * n * 4) as f64 / 1e9 / (ms / 1e3);
    let verdict = if correct { "yes" } else { "no" };
    println!("inclusive scan of {n} values");
    println!("  {ms:6.2} ms  {gb:6.1} GB/s  last {}  correct: {verdict}", result[n - 1]);

    // 3. A histogram of 64 MiB of random bytes, read as 4-byte words.
    let words = DeviceBuffer::from_host(&stream, &random(n / 4, 3))?;
    let hist = module.prepare_hist_words(LaunchConfig1D::new(GRID, BLOCK, 0))?;
    let scratch = DeviceBuffer::<u32>::zeroed(&stream, 256)?.cast_elem::<DeviceAtomicU32>();

    let ms = time_ms(&stream, || Ok(module.hist_words(&stream, &hist, &words, &scratch)?))?;

    let bins = DeviceBuffer::<u32>::zeroed(&stream, 256)?.cast_elem::<DeviceAtomicU32>();
    module.hist_words(&stream, &hist, &words, &bins)?;
    let counts = bins.cast_elem::<u32>().to_host_vec(&stream)?;
    let total: u64 = counts.iter().map(|&c| c as u64).sum();
    let gb = n as f64 / 1e9 / (ms / 1e3);
    println!("histogram of 64 MiB, 4 bytes per load");
    println!("  {ms:6.2} ms  {gb:6.1} GB/s  total {total}");

    Ok(())
}
