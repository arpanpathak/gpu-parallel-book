//! Chapter 14: SAXPY, y = a x + y, with 1 thread per element, as a generic map that
//! takes a closure, and with a grid-stride loop that uses fewer threads than elements.
//!
//!     cargo oxide run

use std::sync::Arc;
use std::time::Instant;

use cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT as TIMING;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
use cuda_device::{launch_bounds, launch_contract};

const BLOCK: u32 = 256;
const N: usize = 1 << 24;
const A: f32 = 2.5;
const RUNS: u32 = 50;

/// The worked example of the chapter: 12 elements, a = 2.
const X12: [f32; 12] = [2.0, 7.0, 1.0, 8.0, 2.0, 8.0, 1.0, 8.0, 2.0, 8.0, 4.0, 5.0];
const Y12: [f32; 12] = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0];

type Error = Box<dyn std::error::Error>;

#[cuda_module]
mod kernels {
    use super::*;

    /// y[i] = a * x[i] + y[i], 1 thread per element.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1)]
    pub fn saxpy(a: f32, x: &[f32], mut y: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();

        if let Some(yi) = y.get_mut(idx) {
            *yi = a * x[i] + *yi;
        }
    }

    /// y[i] = f(x[i], y[i]), 1 thread per element, for any function f.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1)]
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

    /// y[i] = f(x[i], y[i]) for i = t, t + P, t + 2P, ..., where P is the number of
    /// threads in the grid and t the index of this thread.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1)]
    pub fn map_stride<F>(f: F, x: &[f32], mut y: DisjointSlice<f32>)
    where
        F: Fn(f32, f32) -> f32 + Copy,
    {
        let threads = (thread::gridDim_x() * thread::blockDim_x()) as usize;
        let mut i = thread::index_1d().get();

        while i < y.len() {
            // SAFETY: thread t writes only the elements i with i % P == t. The P threads
            // of the grid have P different values of t, so no 2 threads share an i.
            let yi = unsafe { y.get_unchecked_mut(i) };
            *yi = f(x[i], *yi);
            i += threads;
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

/// Resets y, runs `launch` once, and counts the elements that differ from a x + y.
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

fn report(name: &str, threads: u32, bytes: usize, ms: f64, wrong: usize) {
    let gbs = bytes as f64 / ms / 1e6;
    let each = N as f64 / threads as f64;
    print!("  {name:<20} {threads:>9} {each:>7.1}");
    println!(" {ms:>7.3} ms {gbs:>6.1} GB/s {wrong:>5}");
}

fn show(name: &str, values: &[f32]) {
    let row: Vec<String> = values.iter().map(|v| format!("{v:3}")).collect();
    println!("  {name:<20}{}", row.join(""));
}

/// Runs SAXPY with a = 2 on the 12 elements of the worked example, first with 2 blocks
/// of 8 threads, then with a grid-stride loop on 2 blocks of 2 threads. Returns the
/// number of wrong elements.
fn worked_example(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
) -> Result<usize, Error> {
    let x = DeviceBuffer::from_host(stream, &X12)?;
    let mut y = DeviceBuffer::from_host(stream, &Y12)?;
    let expect: Vec<f32> = X12.iter().zip(&Y12).map(|(x, y)| 2.0 * x + y).collect();

    println!("worked example: n = 12, a = 2");
    show("x", &X12);
    show("y", &Y12);

    let cover = module.prepare_saxpy(LaunchConfig1D::new(2, 8, 0))?;
    module.saxpy(stream, &cover, 2.0, &x, &mut y)?;
    let first = y.to_host_vec(stream)?;
    show("2 blocks of 8", &first);

    let a = 2.0f32;
    let axpy = move |x: f32, y: f32| a * x + y;
    y.copy_from_host(stream, &Y12)?;
    let stride = module.prepare_map_stride(LaunchConfig1D::new(2, 2, 0))?;
    module.map_stride(stream, &stride, axpy, &x, &mut y)?;
    let second = y.to_host_vec(stream)?;
    show("2 blocks of 2, loop", &second);

    let bad_first = first.iter().zip(&expect).filter(|(g, e)| g != e).count();
    let bad_second = second.iter().zip(&expect).filter(|(g, e)| g != e).count();
    Ok(bad_first + bad_second)
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

    let mut failed = worked_example(&module, &stream)?;

    println!("SAXPY on {N} floats, a = {A}: 12 bytes and 2 FLOP per element");
    let x0: Vec<f32> = (0..N).map(|i| (i % 1024) as f32 * 0.5).collect();
    let y0: Vec<f32> = (0..N).map(|i| (i % 7) as f32).collect();
    let x = DeviceBuffer::from_host(&stream, &x0)?;
    let mut y = DeviceBuffer::from_host(&stream, &y0)?;

    let a = A;
    let axpy = move |x: f32, y: f32| a * x + y;
    let blocks = (N as u32).div_ceil(BLOCK);
    let one_each = LaunchConfig1D::new(blocks, BLOCK, 0);
    let bytes = 12 * N;

    println!("  kernel                 threads per thr     time        rate wrong");

    let saxpy = module.prepare_saxpy(one_each)?;
    let mut launch = |y: &mut DeviceBuffer<f32>| {
        Ok(module.saxpy(&stream, &saxpy, A, &x, y)?)
    };
    let bad = wrong(&stream, &mut y, &x0, &y0, &mut launch)?;
    let ms = time_ms(&stream, || launch(&mut y))?;
    report("saxpy", blocks * BLOCK, bytes, ms, bad);
    failed += bad;

    let map = module.prepare_map(one_each)?;
    let mut launch = |y: &mut DeviceBuffer<f32>| {
        Ok(module.map(&stream, &map, axpy, &x, y)?)
    };
    let bad = wrong(&stream, &mut y, &x0, &y0, &mut launch)?;
    let ms = time_ms(&stream, || launch(&mut y))?;
    report("map, closure axpy", blocks * BLOCK, bytes, ms, bad);
    failed += bad;

    let larger = |x: f32, y: f32| x.max(y);
    let max = module.prepare_map(one_each)?;
    y.copy_from_host(&stream, &y0)?;
    module.map(&stream, &max, larger, &x, &mut y)?;

    let got = y.to_host_vec(&stream)?;
    let expect = x0.iter().zip(&y0).map(|(x, y)| x.max(*y));
    let bad = got.iter().zip(expect).filter(|(g, e)| **g != *e).count();
    println!("  map, closure max: {bad} wrong");
    failed += bad;

    for k in [1, 2, 3, 6, 12, 48] {
        let grid = LaunchConfig1D::new(sms * k, BLOCK, 0);
        let stride = module.prepare_map_stride(grid)?;
        let mut launch = |y: &mut DeviceBuffer<f32>| {
            Ok(module.map_stride(&stream, &stride, axpy, &x, y)?)
        };

        let bad = wrong(&stream, &mut y, &x0, &y0, &mut launch)?;
        let ms = time_ms(&stream, || launch(&mut y))?;
        report(&format!("loop, {k} per SM"), sms * k * BLOCK, bytes, ms, bad);
        failed += bad;
    }

    let mut copy = DeviceBuffer::<f32>::zeroed(&stream, N)?;
    let ms = time_ms(&stream, || Ok(copy.copy_from_device_async(&x, &stream)?))?;
    let gbs = (8 * N) as f64 / ms / 1e6;
    println!("  device-to-device copy, 8 bytes per element  {ms:>7.3} ms {gbs:>6.1} GB/s");

    if failed == 0 {
        println!("PASSED");
    } else {
        println!("FAILED: {failed} wrong elements");
    }
    Ok(())
}
