//! Chapter 15: the dot product of 2 vectors as a reduction, 4 ways: map then reduce,
//! a fused kernel with a tree in shared memory, a fused kernel with warp shuffles, and
//! a fused kernel that combines its block results with atomic additions.
//!
//!     cargo oxide run

use std::collections::BTreeSet;
use std::sync::Arc;

use cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT as TIMING;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, device, kernel, launch_contract};
use cuda_device::{thread, warp};

const BLOCK: u32 = 256;
const GRID: u32 = 1024;
const N: usize = 1 << 24;
const RUNS: u32 = 20;

/// The worked example of the chapter: 16 pairs whose products add up to 63.
const A16: [f32; 16] = [1., 2., 3., 1., 5., 1., 2., 3., 3., 1., 2., 3., 1., 3., 2., 1.];
const B16: [f32; 16] = [2., 3., 1., 4., 1., 1., 4., 1., 1., 2., 2., 2., 1., 3., 1., 4.];

type Error = Box<dyn std::error::Error>;

#[cuda_module]
mod kernels {
    use super::*;

    /// The sum of `value` over the 32 lanes of a warp, left in lane 0.
    #[device]
    fn warp_sum(mut value: f32) -> f32 {
        let mut step: u32 = 0;

        #[unroll]
        while step < 5 {
            value += warp::shuffle_down_f32(value, 16u32 >> step);
            step += 1;
        }
        value
    }

    /// The sum of `value` over the 256 threads of a block, returned to thread 0.
    #[device]
    fn block_sum(value: f32) -> f32 {
        static mut WARP_SUMS: SharedArray<f32, 8> = SharedArray::UNINIT;
        let lane = warp::lane_id() as usize;
        let warp_index = thread::threadIdx_x() as usize / 32;

        let value = warp_sum(value);
        if lane == 0 {
            // SAFETY: lane 0 of each warp writes its own slot, before the barrier.
            unsafe { WARP_SUMS[warp_index] = value };
        }
        thread::sync_threads();

        // SAFETY: after the barrier the slots are only read.
        let warp_total = if lane < 8 { unsafe { WARP_SUMS[lane] } } else { 0.0 };
        if warp_index == 0 { warp_sum(warp_total) } else { 0.0 }
    }

    /// The sum of a[index] * b[index] over this thread's elements, a grid apart.
    #[device]
    fn products(a: &[f32], b: &[f32]) -> f32 {
        let grid_size = (thread::gridDim_x() * BLOCK) as usize;
        let mut index = thread::index_1d().get();

        let mut sum = 0.0;
        while index < a.len() {
            sum += a[index] * b[index];
            index += grid_size;
        }
        sum
    }

    /// Map: products[index] = a[index] * b[index], one element per thread.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn multiply(a: &[f32], b: &[f32], mut element_products: DisjointSlice<f32>) {
        let thread_id = thread::index_1d();
        let index = thread_id.get();

        if let Some(out) = element_products.get_mut(thread_id) {
            *out = a[index] * b[index];
        }
    }

    /// Reduce: 1 partial sum of `input` per block.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn sum(input: &[f32], mut partial: DisjointSlice<f32>) {
        let grid_size = (thread::gridDim_x() * BLOCK) as usize;
        let mut index = thread::index_1d().get();

        let mut my_sum = 0.0;
        while index < input.len() {
            my_sum += input[index];
            index += grid_size;
        }

        let total = block_sum(my_sum);
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
        let thread_index = thread::threadIdx_x() as usize;

        // SAFETY: each thread writes only its own cell, before the barrier.
        unsafe { CELLS[thread_index] = products(a, b) };
        thread::sync_threads();

        let mut step: usize = 0;
        #[unroll]
        while step < 8 {
            let stride: usize = 128 >> step;
            if thread_index < stride {
                // SAFETY: in this step, this thread alone writes its cell and reads cell
                // thread_index + stride, which no thread writes in this step.
                unsafe { CELLS[thread_index] += CELLS[thread_index + stride] };
            }
            thread::sync_threads();
            step += 1;
        }

        if thread_index == 0 {
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

const METHODS: [(&str, Method, usize); 4] = [
    ("map, then reduce", Method::MapReduce, 16),
    ("fused, shared tree", Method::Tree, 8),
    ("fused, shuffles", Method::Shuffle, 8),
    ("fused, atomic combine", Method::Atomic, 8),
];

/// The device memory of 1 problem.
struct Buffers {
    a: DeviceBuffer<f32>,
    b: DeviceBuffer<f32>,
    products: DeviceBuffer<f32>,
    partial: DeviceBuffer<f32>,
    total: DeviceBuffer<f32>,
    atomic: DeviceBuffer<DeviceAtomicF32>,
}

impl Buffers {
    fn new(stream: &Arc<CudaStream>, a: &[f32], b: &[f32]) -> Result<Self, Error> {
        Ok(Buffers {
            a: DeviceBuffer::from_host(stream, a)?,
            b: DeviceBuffer::from_host(stream, b)?,
            products: DeviceBuffer::zeroed(stream, a.len())?,
            partial: DeviceBuffer::zeroed(stream, GRID as usize)?,
            total: DeviceBuffer::zeroed(stream, 1)?,
            atomic: DeviceBuffer::<f32>::zeroed(stream, 1)?.cast_elem(),
        })
    }
}

/// Runs 1 method once and returns its result.
fn result<R>(
    run: &R,
    method: Method,
    buf: &mut Buffers,
    stream: &CudaStream,
) -> Result<f32, Error>
where
    R: Fn(Method, &mut Buffers) -> Result<(), Error>,
{
    run(method, buf)?;
    Ok(buf.total.to_host_vec(stream)?[0])
}

/// Milliseconds per run of 1 method: 10 warm-up runs, then RUNS runs timed with events.
fn time_ms<R>(
    run: &R,
    method: Method,
    buf: &mut Buffers,
    stream: &CudaStream,
) -> Result<f64, Error>
where
    R: Fn(Method, &mut Buffers) -> Result<(), Error>,
{
    for _ in 0..10 {
        run(method, buf)?;
    }

    let start = stream.record_event(Some(TIMING))?;
    for _ in 0..RUNS {
        run(method, buf)?;
    }
    let stop = stream.record_event(Some(TIMING))?;
    stop.synchronize()?;

    Ok(start.elapsed_ms(&stop)? as f64 / RUNS as f64)
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

/// Every method on 1 input whose dot product is an integer that a float holds exactly.
/// Returns the number of methods whose result differs from `expect`.
fn exact_test<R>(
    run: &R,
    stream: &Arc<CudaStream>,
    label: &str,
    a: &[f32],
    b: &[f32],
) -> Result<usize, Error>
where
    R: Fn(Method, &mut Buffers) -> Result<(), Error>,
{
    let expect: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let mut buf = Buffers::new(stream, a, b)?;
    let mut bad = 0;

    print!("  {label:<26}");
    for (_, method, _) in METHODS {
        let got = result(run, method, &mut buf, stream)?;
        print!(" {got:>9}");
        bad += usize::from(got != expect);
    }
    println!("   expected {expect}");
    Ok(bad)
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };

    let name = ctx.device_name()?;
    let sms = ctx.multiprocessor_count()?;
    println!("GPU: {name}, {sms} SMs");

    let grid = LaunchConfig1D::new(GRID, BLOCK, 0);
    let one = LaunchConfig1D::new(1, BLOCK, 0);
    let map = LaunchConfig1D::new(N as u32 / BLOCK, BLOCK, 0);

    let multiply = module.prepare_multiply(map)?;
    let reduce = module.prepare_sum(grid)?;
    let last = module.prepare_sum(one)?;
    let tree = module.prepare_dot_tree(grid)?;
    let shuffle = module.prepare_dot_shuffle(grid)?;
    let clear = module.prepare_clear(one)?;
    let atomic = module.prepare_dot_atomic(grid)?;
    let publish = module.prepare_publish(one)?;

    // Launches 1 dot product; the result ends in `buf.total`.
    let run = |method: Method, buf: &mut Buffers| -> Result<(), Error> {
        let (a, b, stream) = (&buf.a, &buf.b, &stream);
        match method {
            Method::MapReduce => {
                module.multiply(stream, &multiply, a, b, &mut buf.products)?;
                module.sum(stream, &reduce, &buf.products, &mut buf.partial)?;
                module.sum(stream, &last, &buf.partial, &mut buf.total)?;
            }
            Method::Tree => {
                module.dot_tree(stream, &tree, a, b, &mut buf.partial)?;
                module.sum(stream, &last, &buf.partial, &mut buf.total)?;
            }
            Method::Shuffle => {
                module.dot_shuffle(stream, &shuffle, a, b, &mut buf.partial)?;
                module.sum(stream, &last, &buf.partial, &mut buf.total)?;
            }
            Method::Atomic => {
                module.clear(stream, &clear, &buf.atomic)?;
                module.dot_atomic(stream, &atomic, a, b, &buf.atomic)?;
                module.publish(stream, &publish, &buf.atomic, &mut buf.total)?;
            }
        }
        Ok(())
    };

    println!("exact tests                   map+red      tree   shuffle    atomic");
    let mut failed = exact_test(&run, &stream, "worked example, 16 pairs", &A16, &B16)?;

    let ones_a: Vec<f32> = (0..N).map(|index| f32::from(index % 3 == 0)).collect();
    let ones_b: Vec<f32> = (0..N).map(|index| f32::from(index % 2 == 0)).collect();
    failed += exact_test(&run, &stream, "0s and 1s, 16777216 pairs", &ones_a, &ones_b)?;

    let a = random(N, 1);
    let b = random(N, 2);
    let exact: f64 = a.iter().zip(&b).map(|(&x, &y)| x as f64 * y as f64).sum();
    let mut buf = Buffers::new(&stream, &a, &b)?;
    let error = |r: f32| ((r as f64 - exact) / exact).abs();
    let mut worst = 0.0f64;

    println!("{N} pairs of floats in [0, 1), {RUNS} runs");
    println!("  method                    time   bytes/pair      rate      result");
    for (name, method, per_pair) in METHODS {
        let ms = time_ms(&run, method, &mut buf, &stream)?;
        let got = result(&run, method, &mut buf, &stream)?;
        worst = worst.max(error(got));

        let gbs = (N * per_pair) as f64 / ms / 1e6;
        print!("  {name:<22} {ms:6.3} ms {per_pair:>8} B");
        println!("  {gbs:6.1} GB/s  {got:.1}");
    }
    println!("  {:<54} {exact:.1}", "exact, in double precision");

    println!("distinct results in 100 runs");
    for (name, method) in [("second launch", Method::Shuffle), ("atomic", Method::Atomic)] {
        let mut seen = BTreeSet::new();
        for _ in 0..100 {
            seen.insert(result(&run, method, &mut buf, &stream)?.to_bits());
        }

        let low = f32::from_bits(*seen.first().ok_or("no result")?);
        let high = f32::from_bits(*seen.last().ok_or("no result")?);
        println!("  {name:<14} {:3}  from {low:.1} to {high:.1}", seen.len());
        worst = worst.max(error(low)).max(error(high));
    }

    println!("largest relative error {worst:.2e}");
    if failed == 0 && worst < 1e-5 {
        println!("PASSED");
    } else {
        println!("FAILED");
    }
    Ok(())
}
