//! Chapter 5 in Rust: an atomic counter, 2 histograms, a block sum with and without
//! barriers, a branch on alternate lanes or warps, and a float sum by atomics.
//!
//!     cargo oxide run

use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, BlockAtomicU32, DeviceAtomicF32, DeviceAtomicU32};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, device, kernel, launch_contract};
use cuda_device::thread;

#[cuda_module]
mod kernels {
    use super::*;

    // A lost update on purpose. The load and the store are 2 separate operations,
    // so another thread can store between them, exactly like the plain C++ add.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn count_racy(counter: &[DeviceAtomicU32]) {
        let seen = counter[0].load(AtomicOrdering::Relaxed);

        counter[0].store(seen + 1, AtomicOrdering::Relaxed);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn count(counter: &[DeviceAtomicU32]) {
        counter[0].fetch_add(1, AtomicOrdering::Relaxed);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn hist_global(data: &[u8], bins: &[DeviceAtomicU32]) {
        let stride = (thread::gridDim_x() * thread::blockDim_x()) as usize;
        let mut i = thread::index_1d().get();

        while i < data.len() {
            bins[data[i] as usize].fetch_add(1, AtomicOrdering::Relaxed);
            i += stride;
        }
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn hist_shared(data: &[u8], bins: &[DeviceAtomicU32]) {
        static mut LOCAL: SharedArray<u32, 256> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;

        // SAFETY: each thread zeroes only its own bin, before the barrier.
        unsafe { LOCAL[tid] = 0 };
        thread::sync_threads();

        let stride = (thread::gridDim_x() * thread::blockDim_x()) as usize;
        let mut i = thread::index_1d().get();

        while i < data.len() {
            // SAFETY: between the 2 barriers every access to LOCAL is atomic.
            let bin = unsafe {
                let base = (*&raw mut LOCAL).as_mut_ptr();
                BlockAtomicU32::from_ptr(base.add(data[i] as usize))
            };
            bin.fetch_add(1, AtomicOrdering::Relaxed);
            i += stride;
        }
        thread::sync_threads();

        // SAFETY: after the barrier no thread writes LOCAL again.
        let local = unsafe { LOCAL[tid] };
        bins[tid].fetch_add(local, AtomicOrdering::Relaxed);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn block_sum(input: &[f32], mut out: DisjointSlice<f32>, barrier: bool) {
        static mut S: SharedArray<f32, 256> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let block = thread::blockIdx_x() as usize;

        // SAFETY: each thread writes only its own element.
        unsafe { S[tid] = input[block * 256 + tid] };
        if barrier {
            thread::sync_threads();
        }

        let mut stride = 128;
        while stride > 0 {
            if tid < stride {
                // SAFETY: true only with the barriers. Thread tid alone writes S[tid] here,
                // and S[tid + stride] was written before the last barrier.
                unsafe { S[tid] += S[tid + stride] };
            }
            if barrier {
                thread::sync_threads();
            }
            stride /= 2;
        }

        if tid == 0 {
            // SAFETY: only thread 0 of each block writes, at its own block number.
            unsafe { *out.get_unchecked_mut(block) = S[0] };
        }
    }

    /// The loop of the odd threads: 64 steps of v = sin(v) + 1.
    #[device]
    fn sin_loop(mut v: f32) -> f32 {
        let mut k = 0;
        #[unroll]
        while k < 64 {
            v = v.sin() + 1.0;
            k += 1;
        }
        v
    }

    /// The loop of the even threads: 64 steps of v = sqrt(v) / 2 + 1 / 4.
    #[device]
    fn sqrt_loop(mut v: f32) -> f32 {
        let mut k = 0;
        #[unroll]
        while k < 64 {
            v = v.sqrt() * 0.5 + 0.25;
            k += 1;
        }
        v
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn branches(mut x: DisjointSlice<f32>, by_warp: bool) {
        let tid = thread::threadIdx_x();
        let odd = if by_warp { (tid / 32) % 2 == 1 } else { tid % 2 == 1 };

        let Some(x_i) = x.get_mut(thread::index_1d()) else {
            return;
        };

        let v = *x_i;
        *x_i = if odd { sin_loop(v) } else { sqrt_loop(v) };
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn sum_atomic(values: &[f32], total: &[DeviceAtomicF32]) {
        let i = thread::index_1d().get();

        if i < values.len() {
            total[0].fetch_add(values[i], AtomicOrdering::Relaxed);
        }
    }
}

/// Bytes from a small linear congruential generator, so the run needs no extra crate.
fn random_bytes(n: usize) -> Vec<u8> {
    let mut state: u32 = 1;

    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        })
        .collect()
}

/// `n` zeroed counters that kernels update with atomic operations.
fn atomic_u32(
    stream: &Arc<CudaStream>,
    n: usize,
) -> Result<DeviceBuffer<DeviceAtomicU32>, DriverError> {
    Ok(DeviceBuffer::<u32>::zeroed(stream, n)?.cast_elem::<DeviceAtomicU32>())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };

    // 1. 1,048,576 threads each add 1 to 1 counter, with a race and with an atomic.
    let grid = LaunchConfig1D::new(4096, 256, 0);

    let racy = atomic_u32(&stream, 1)?;
    let launch = module.prepare_count_racy(grid)?;
    module.count_racy(&stream, &launch, &racy)?;
    let lost = racy.cast_elem::<u32>().to_host_vec(&stream)?;

    let counter = atomic_u32(&stream, 1)?;
    let launch = module.prepare_count(grid)?;
    module.count(&stream, &launch, &counter)?;
    let result = counter.cast_elem::<u32>().to_host_vec(&stream)?;

    println!("1,048,576 threads each add 1 to 1 counter");
    println!("  load, store  counter = {}", lost[0]);
    println!("  fetch_add    counter = {}", result[0]);

    // 2. A histogram of 64 MiB of bytes, with global bins and with shared bins.
    let n = 64 << 20;
    let inputs = [("random bytes", random_bytes(n)), ("all 7", vec![7u8; n])];
    let global = module.prepare_hist_global(LaunchConfig1D::new(1024, 256, 0))?;
    let shared = module.prepare_hist_shared(LaunchConfig1D::new(1024, 256, 0))?;

    println!("histogram of 64 MiB, average of 10 runs");
    for (name, bytes) in &inputs {
        let data = DeviceBuffer::from_host(&stream, bytes)?;

        for kind in ["global bins", "shared bins"] {
            let mut ms = 0.0;
            let mut total = 0;

            for run in 0..11 {
                let bins = atomic_u32(&stream, 256)?;
                stream.synchronize()?;

                let start = Instant::now();
                if kind == "global bins" {
                    module.hist_global(&stream, &global, &data, &bins)?;
                } else {
                    module.hist_shared(&stream, &shared, &data, &bins)?;
                }
                stream.synchronize()?;

                if run > 0 {
                    ms += start.elapsed().as_secs_f64() * 1e3;
                }

                let counts = bins.cast_elem::<u32>().to_host_vec(&stream)?;
                total = counts.iter().map(|&c| c as u64).sum();
            }

            println!("  {name:<12} {kind:<13} {:7.2} ms  total {total}", ms / 10.0);
        }
    }

    // 3. 4,096 blocks each sum 256 ones in shared memory, with and without barriers.
    let blocks = 4096;
    let ones = DeviceBuffer::from_host(&stream, &vec![1.0f32; blocks * 256])?;
    let mut sums = DeviceBuffer::<f32>::zeroed(&stream, blocks)?;
    let block_sum = module.prepare_block_sum(LaunchConfig1D::new(blocks as u32, 256, 0))?;

    println!("4,096 blocks each sum 256 ones; the right answer is 256");
    for barrier in [true, false] {
        module.block_sum(&stream, &block_sum, &ones, &mut sums, barrier)?;

        let s = sums.to_host_vec(&stream)?;
        let wrong = s.iter().filter(|&&sum| sum != 256.0).count();
        let name = if barrier { "with barriers" } else { "without barriers" };
        println!("  {name:<18} wrong sums: {wrong:4} of {blocks}   (block 0: {})", s[0]);
    }

    // 4. Half the threads take each branch: alternate lanes, or alternate warps.
    let n = 16 << 20;
    let mut x = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    let branch = module.prepare_branches(LaunchConfig1D::new((n / 256) as u32, 256, 0))?;

    println!("branch on every other thread, {n} threads, average of 10 runs");
    for by_warp in [true, false] {
        module.branches(&stream, &branch, &mut x, by_warp)?; // warm up
        stream.synchronize()?;

        let start = Instant::now();
        for _ in 0..10 {
            module.branches(&stream, &branch, &mut x, by_warp)?;
        }
        stream.synchronize()?;

        let ms = start.elapsed().as_secs_f64() * 1e3 / 10.0;
        let name = if by_warp { "whole warps agree" } else { "lanes alternate" };
        println!("  {name:<22} {ms:6.2} ms");
    }

    // 5. A float sum by fetch_add, in whatever order the threads arrive.
    let values: Vec<f32> = (0..1 << 20).map(|i| 1.0 / (1 + i % 1000) as f32).collect();
    let exact: f64 = values.iter().map(|&v| v as f64).sum();
    let values_dev = DeviceBuffer::from_host(&stream, &values)?;
    let sum = module.prepare_sum_atomic(LaunchConfig1D::new(4096, 256, 0))?;

    println!("sum of 1,048,576 floats by fetch_add (exact: {exact:.4})");
    for run in 1..=4 {
        let total = DeviceBuffer::<f32>::zeroed(&stream, 1)?.cast_elem::<DeviceAtomicF32>();
        module.sum_atomic(&stream, &sum, &values_dev, &total)?;

        let result = total.cast_elem::<f32>().to_host_vec(&stream)?;
        println!("  run {run}   {:.4}", result[0]);
    }

    Ok(())
}
