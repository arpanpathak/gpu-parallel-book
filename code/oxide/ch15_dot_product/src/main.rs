//! Chapter 15: the dot product of 2 vectors, 4 ways.
//!
//!     cargo oxide run

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, device, kernel, launch_contract};
use cuda_device::{thread, warp};

const BLOCK: u32 = 256;
const GRID: u32 = 1024;
const N: usize = 16 << 20;

type Error = Box<dyn std::error::Error>;

#[cuda_module]
mod kernels {
    use super::*;

    /// The sum of `v` over the 32 lanes of a warp, left in lane 0.
    #[device]
    fn warp_sum(mut v: f32) -> f32 {
        let mut step: u32 = 0;

        #[unroll]
        while step < 5 {
            v += warp::shuffle_down_f32(v, 16u32 >> step);
            step += 1;
        }
        v
    }

    /// The sum of `v` over the 256 threads of a block, returned to thread 0.
    #[device]
    fn block_sum(v: f32) -> f32 {
        static mut WARPS: SharedArray<f32, 8> = SharedArray::UNINIT;
        let lane = warp::lane_id() as usize;
        let warp = thread::threadIdx_x() as usize / 32;

        let v = warp_sum(v);
        if lane == 0 {
            // SAFETY: lane 0 of each warp writes its own slot, before the barrier.
            unsafe { WARPS[warp] = v };
        }
        thread::sync_threads();

        // SAFETY: after the barrier the slots are only read.
        let w = if lane < 8 { unsafe { WARPS[lane] } } else { 0.0 };
        if warp == 0 { warp_sum(w) } else { 0.0 }
    }

    /// The products a[i] * b[i] of this thread's elements, read with a grid-stride loop.
    #[device]
    fn products(a: &[f32], b: &[f32]) -> f32 {
        let stride = (thread::gridDim_x() * BLOCK) as usize;
        let mut i = thread::index_1d().get();

        let mut sum = 0.0;
        while i < a.len() {
            sum += a[i] * b[i];
            i += stride;
        }
        sum
    }

    /// Map: p[i] = a[i] * b[i], 1 element per thread.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn multiply(a: &[f32], b: &[f32], mut p: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();

        if let Some(out) = p.get_mut(idx) {
            *out = a[i] * b[i];
        }
    }

    /// Reduce: 1 partial sum of `input` per block.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn sum(input: &[f32], mut partial: DisjointSlice<f32>) {
        let stride = (thread::gridDim_x() * BLOCK) as usize;
        let mut i = thread::index_1d().get();

        let mut v = 0.0;
        while i < input.len() {
            v += input[i];
            i += stride;
        }

        let total = block_sum(v);
        if thread::threadIdx_x() == 0 {
            // SAFETY: only thread 0 of each block writes, at its own block number.
            unsafe { *partial.get_unchecked_mut(thread::blockIdx_x() as usize) = total };
        }
    }

    /// Fused: products and a tree in shared memory, 1 partial sum per block.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn dot_tree(a: &[f32], b: &[f32], mut partial: DisjointSlice<f32>) {
        static mut CELLS: SharedArray<f32, 256> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;

        // SAFETY: each thread writes only its own cell, before the barrier.
        unsafe { CELLS[tid] = products(a, b) };
        thread::sync_threads();

        let mut step: usize = 0;
        #[unroll]
        while step < 8 {
            let half: usize = 128 >> step;
            if tid < half {
                // SAFETY: in this step thread t alone writes cell t, and reads cell
                // t + half, which no thread writes in the same step.
                unsafe { CELLS[tid] += CELLS[tid + half] };
            }
            thread::sync_threads();
            step += 1;
        }

        if tid == 0 {
            // SAFETY: only thread 0 of each block writes, at its own block number.
            unsafe { *partial.get_unchecked_mut(thread::blockIdx_x() as usize) = CELLS[0] };
        }
    }

    /// Fused: products and warp shuffles, 1 partial sum per block.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn dot_shuffle(a: &[f32], b: &[f32], mut partial: DisjointSlice<f32>) {
        let total = block_sum(products(a, b));

        if thread::threadIdx_x() == 0 {
            // SAFETY: only thread 0 of each block writes, at its own block number.
            unsafe { *partial.get_unchecked_mut(thread::blockIdx_x() as usize) = total };
        }
    }

    /// Sets the atomic result to 0 before a run of `dot_atomic`.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn clear(result: &[DeviceAtomicF32]) {
        if thread::index_1d().get() == 0 {
            result[0].store(0.0, AtomicOrdering::Relaxed);
        }
    }

    /// Copies the atomic result into an ordinary float that the host can read.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn publish(result: &[DeviceAtomicF32], mut out: DisjointSlice<f32>) {
        if let Some(value) = out.get_mut(thread::index_1d()) {
            *value = result[0].load(AtomicOrdering::Relaxed);
        }
    }

    /// Fused: products, warp shuffles, and 1 atomic addition per block.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn dot_atomic(a: &[f32], b: &[f32], result: &[DeviceAtomicF32]) {
        let total = block_sum(products(a, b));

        if thread::threadIdx_x() == 0 {
            result[0].fetch_add(total, AtomicOrdering::Relaxed);
        }
    }
}

/// The 4 ways of computing the dot product that the program compares.
#[derive(Clone, Copy)]
enum Method {
    MapReduce,
    Tree,
    Shuffle,
    Atomic,
}

/// The device memory of the program.
struct Buffers {
    a: DeviceBuffer<f32>,
    b: DeviceBuffer<f32>,
    p: DeviceBuffer<f32>,
    partial: DeviceBuffer<f32>,
    total: DeviceBuffer<f32>,
    atomic: DeviceBuffer<DeviceAtomicF32>,
}

/// Launches 1 dot product; the result ends in `buf.total`.
fn run(
    method: Method,
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    buf: &mut Buffers,
) -> Result<(), Error> {
    let grid = LaunchConfig1D::new(GRID, BLOCK, 0);
    let one = LaunchConfig1D::new(1, BLOCK, 0);
    let map = LaunchConfig1D::new(N as u32 / BLOCK, BLOCK, 0);
    let last = module.prepare_sum(one)?;

    match method {
        Method::MapReduce => {
            let multiply = module.prepare_multiply(map)?;
            let reduce = module.prepare_sum(grid)?;
            module.multiply(stream, &multiply, &buf.a, &buf.b, &mut buf.p)?;
            module.sum(stream, &reduce, &buf.p, &mut buf.partial)?;
            module.sum(stream, &last, &buf.partial, &mut buf.total)?;
        }
        Method::Tree => {
            let tree = module.prepare_dot_tree(grid)?;
            module.dot_tree(stream, &tree, &buf.a, &buf.b, &mut buf.partial)?;
            module.sum(stream, &last, &buf.partial, &mut buf.total)?;
        }
        Method::Shuffle => {
            let shuffle = module.prepare_dot_shuffle(grid)?;
            module.dot_shuffle(stream, &shuffle, &buf.a, &buf.b, &mut buf.partial)?;
            module.sum(stream, &last, &buf.partial, &mut buf.total)?;
        }
        Method::Atomic => {
            let clear = module.prepare_clear(one)?;
            let fused = module.prepare_dot_atomic(grid)?;
            let publish = module.prepare_publish(one)?;
            module.clear(stream, &clear, &buf.atomic)?;
            module.dot_atomic(stream, &fused, &buf.a, &buf.b, &buf.atomic)?;
            module.publish(stream, &publish, &buf.atomic, &mut buf.total)?;
        }
    }
    Ok(())
}

/// Average time of 20 runs of `method`, in milliseconds, after 10 warm-up runs.
fn time_ms(
    method: Method,
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    buf: &mut Buffers,
) -> Result<f64, Error> {
    for _ in 0..10 {
        run(method, module, stream, buf)?;
    }
    stream.synchronize()?;

    let start = Instant::now();
    for _ in 0..20 {
        run(method, module, stream, buf)?;
    }
    stream.synchronize()?;

    Ok(start.elapsed().as_secs_f64() * 1e3 / 20.0)
}

/// Floats in [0, 1) from a small linear congruential generator.
fn random(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;

    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / 16_777_216.0
        })
        .collect()
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };

    let a = random(N, 1);
    let b = random(N, 2);
    let exact: f64 = a.iter().zip(&b).map(|(&x, &y)| x as f64 * y as f64).sum();

    let mut buf = Buffers {
        a: DeviceBuffer::from_host(&stream, &a)?,
        b: DeviceBuffer::from_host(&stream, &b)?,
        p: DeviceBuffer::zeroed(&stream, N)?,
        partial: DeviceBuffer::zeroed(&stream, GRID as usize)?,
        total: DeviceBuffer::zeroed(&stream, 1)?,
        atomic: DeviceBuffer::<f32>::zeroed(&stream, 1)?.cast_elem(),
    };

    let error = |r: f32| ((r as f64 - exact) / exact).abs();
    let mut worst = 0.0f64;

    println!("dot product of {N} pairs of floats, average of 20 runs");
    let methods = [
        ("map, then reduce", Method::MapReduce, 16),
        ("fused, shared tree", Method::Tree, 8),
        ("fused, shuffles", Method::Shuffle, 8),
        ("fused, atomic combine", Method::Atomic, 8),
    ];

    for (name, method, bytes_per_pair) in methods {
        let ms = time_ms(method, &module, &stream, &mut buf)?;
        let result = buf.total.to_host_vec(&stream)?[0];
        worst = worst.max(error(result));

        let gb = (N * bytes_per_pair) as f64 / 1e9 / (ms / 1e3);
        let per_pair = format!("{bytes_per_pair} B/pair");
        println!("  {name:<22} {ms:6.3} ms  {per_pair:>9}  {gb:6.1} GB/s  {result:.1}");
    }
    println!("  {:<46} {exact:.1}", "exact (f64)");

    println!("distinct results in 100 runs");
    for (name, method) in [("second launch", Method::Shuffle), ("atomic", Method::Atomic)] {
        let mut seen = BTreeSet::new();
        for _ in 0..100 {
            run(method, &module, &stream, &mut buf)?;
            seen.insert(buf.total.to_host_vec(&stream)?[0].to_bits());
        }

        let low = f32::from_bits(*seen.first().ok_or("no result")?);
        let high = f32::from_bits(*seen.last().ok_or("no result")?);
        println!("  {name:<14} {:3}  from {low:.1} to {high:.1}", seen.len());

        worst = worst.max(error(low)).max(error(high));
    }

    let verdict = if worst < 1e-5 { "PASSED" } else { "FAILED" };
    println!("largest relative error {worst:.2e}: {verdict}");

    Ok(())
}
