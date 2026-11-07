//! Chapter 14: SAXPY as a kernel, as a closure passed to a generic map, and as a
//! grid-stride loop launched with fewer threads than elements.
//!
//!     cargo oxide run

use std::sync::Arc;
use std::time::Instant;

use cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT as TIMING;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_contract, thread};

const BLOCK: u32 = 256;
const N: usize = 1 << 24;
const A: f32 = 2.5;
const RUNS: u32 = 50;

type Error = Box<dyn std::error::Error>;

#[cuda_module]
mod kernels {
    use super::*;

    /// y[i] = a * x[i] + y[i], 1 thread per element.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn saxpy(a: f32, x: &[f32], mut y: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();

        if let Some(yi) = y.get_mut(idx) {
            *yi = a * x[i] + *yi;
        }
    }

    /// y[i] = f(x[i], y[i]), 1 thread per element, for any function f.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn map<F>(f: F, x: &[f32], mut y: DisjointSlice<f32>)
    where
        F: Fn(f32, f32) -> f32 + Copy,
    {
        let idx = thread::index_1d();
        let i = idx.get();

        if let Some(yi) = y.get_mut(idx) {
            *yi = f(x[i], *yi);
        }
    }

    /// y[i] = f(x[i], y[i]) for i = t, t + stride, t + 2 stride, ...
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn map_stride<F>(f: F, x: &[f32], mut y: DisjointSlice<f32>)
    where
        F: Fn(f32, f32) -> f32 + Copy,
    {
        let stride = (thread::gridDim_x() * BLOCK) as usize;
        let mut i = thread::index_1d().get();

        while i < y.len() {
            // SAFETY: thread t writes only the elements i with i % stride == t, and
            // every thread of the grid has a different t, so no 2 threads share an i.
            let yi = unsafe { y.get_unchecked_mut(i) };
            *yi = f(x[i], *yi);
            i += stride;
        }
    }
}

/// Milliseconds per launch: 500 ms of warm-up, then RUNS launches timed with events.
fn time_ms<F>(stream: &CudaStream, mut launch: F) -> Result<f64, Error>
where
    F: FnMut() -> Result<(), Error>,
{
    let warm = Instant::now();
    while warm.elapsed().as_millis() < 500 {
        launch()?;
        stream.synchronize()?;
    }

    let start = stream.record_event(Some(TIMING))?;
    for _ in 0..RUNS {
        launch()?;
    }
    let stop = stream.record_event(Some(TIMING))?;
    stop.synchronize()?;

    Ok(start.elapsed_ms(&stop)? as f64 / RUNS as f64)
}

/// Reset y, run `launch` once, and count the elements that differ from a x + y.
fn wrong<F>(
    stream: &Arc<CudaStream>,
    y: &mut DeviceBuffer<f32>,
    x0: &[f32],
    y0: &[f32],
    mut launch: F,
) -> Result<usize, Error>
where
    F: FnMut(&mut DeviceBuffer<f32>) -> Result<(), Error>,
{
    y.copy_from_host(stream, y0)?;
    launch(y)?;
    let got = y.to_host_vec(stream)?;

    let expect = x0.iter().zip(y0).map(|(&x, &y)| A * x + y);
    Ok(got.iter().zip(expect).filter(|(g, e)| **g != *e).count())
}

fn report(name: &str, threads: u32, ms: f64, wrong: usize) {
    let gbs = (3 * 4 * N) as f64 / ms / 1e6;
    let each = N as f64 / threads as f64;
    print!("  {name:<18} {threads:>9} {each:>7.1}");
    println!(" {ms:>7.3} ms {gbs:>6.1} GB/s {wrong:>5}");
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };

    let name = ctx.device_name()?;
    let sms = ctx.multiprocessor_count()?;
    let per_sm = ctx.max_threads_per_multiprocessor()?;
    println!("GPU: {name}, {sms} SMs, at most {per_sm} threads per SM");
    println!("SAXPY on {N} floats, a = {A}: 12 bytes and 2 FLOPs per element");

    let x0: Vec<f32> = (0..N).map(|i| (i % 1024) as f32 * 0.5).collect();
    let y0: Vec<f32> = (0..N).map(|i| (i % 7) as f32).collect();
    let x = DeviceBuffer::from_host(&stream, &x0)?;
    let mut y = DeviceBuffer::from_host(&stream, &y0)?;

    let a = A;
    let axpy = move |x: f32, y: f32| a * x + y;

    let blocks = (N as u32).div_ceil(BLOCK);
    let one_each = LaunchConfig1D::new(blocks, BLOCK, 0);
    let mut failed = 0;

    println!("  kernel               threads per thr     time        rate wrong");

    let saxpy = module.prepare_saxpy(one_each)?;
    let mut launch = |y: &mut DeviceBuffer<f32>| {
        Ok(module.saxpy(&stream, &saxpy, A, &x, y)?)
    };
    let bad = wrong(&stream, &mut y, &x0, &y0, &mut launch)?;
    let ms = time_ms(&stream, || launch(&mut y))?;
    report("saxpy", blocks * BLOCK, ms, bad);
    failed += bad;

    let map = module.prepare_map(one_each)?;
    let mut launch = |y: &mut DeviceBuffer<f32>| {
        Ok(module.map(&stream, &map, axpy, &x, y)?)
    };
    let bad = wrong(&stream, &mut y, &x0, &y0, &mut launch)?;
    let ms = time_ms(&stream, || launch(&mut y))?;
    report("map(closure)", blocks * BLOCK, ms, bad);
    failed += bad;

    let larger = |x: f32, y: f32| x.max(y);
    let max = module.prepare_map(one_each)?;
    y.copy_from_host(&stream, &y0)?;
    module.map(&stream, &max, larger, &x, &mut y)?;

    let got = y.to_host_vec(&stream)?;
    let expect = x0.iter().zip(&y0).map(|(x, y)| x.max(*y));
    let bad = got.iter().zip(expect).filter(|(g, e)| **g != *e).count();
    println!("  map(x.max(y)): {bad} wrong");
    failed += bad;

    for k in [1, 2, 3, 6, 12, 24, 48] {
        let grid = LaunchConfig1D::new(sms * k, BLOCK, 0);
        let stride = module.prepare_map_stride(grid)?;
        let mut launch = |y: &mut DeviceBuffer<f32>| {
            Ok(module.map_stride(&stream, &stride, axpy, &x, y)?)
        };

        let bad = wrong(&stream, &mut y, &x0, &y0, &mut launch)?;
        let ms = time_ms(&stream, || launch(&mut y))?;
        report(&format!("stride, {k} blk/SM"), sms * k * BLOCK, ms, bad);
        failed += bad;
    }

    if failed == 0 {
        println!("PASSED");
    } else {
        println!("FAILED: {failed} wrong elements");
    }
    Ok(())
}
