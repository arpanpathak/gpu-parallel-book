//! Chapter 5 in Rust: an atomic counter, 2 histograms, and a float sum by atomics.
//!
//!     cargo oxide run

use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, BlockAtomicU32, DeviceAtomicF32, DeviceAtomicU32};
use cuda_device::{SharedArray, cuda_module, kernel, launch_contract, thread};

#[cuda_module]
mod kernels {
    use super::*;

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

    // 1. 1,048,576 threads each add 1 to 1 counter.
    let counter = atomic_u32(&stream, 1)?;
    let launch = module.prepare_count(LaunchConfig1D::new(4096, 256, 0))?;
    module.count(&stream, &launch, &counter)?;

    let result = counter.cast_elem::<u32>().to_host_vec(&stream)?;
    println!("1,048,576 threads each add 1 to 1 counter");
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

    // 3. A float sum by fetch_add, in whatever order the threads arrive.
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
