//! Chapter 19: Jacobi iteration for the Laplace equation, 1 thread per cell.
//!
//!     cargo oxide run

use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D, LaunchConfig2D};
use cuda_device::thread::Runtime2DIndex;
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_contract, thread};
use cuda_host::RowWidth;

const BX: usize = 32; // block width: 1 warp along a row of the grid
const BY: usize = 8; // block height: 8 warps, 256 threads
const HOT: f32 = 100.0; // the top edge; the other edges are held at 0

type Error = Box<dyn std::error::Error>;

#[cuda_module]
mod kernels {
    use super::*;

    /// 1 Jacobi step: each interior cell of `new` gets the mean of 4 neighbors in `old`.
    #[kernel]
    #[launch_contract(domain = 2, block = (32, 8, 1))]
    pub fn jacobi(old: &[f32], mut new: DisjointSlice<f32, Runtime2DIndex>) {
        let width = new.row_width() as usize;
        let height = new.len() / width;

        if let Some(idx) = thread::index_2d_runtime(&new) {
            let (row, col) = (idx.row(), idx.col());
            let interior = row >= 1 && row + 1 < height && col >= 1 && col + 1 < width;

            if interior {
                let c = row * width + col;
                let sum = old[c - width] + old[c + width] + old[c - 1] + old[c + 1];

                if let Some(cell) = new.get_mut(idx) {
                    *cell = 0.25 * sum;
                }
            }
        }
    }

    /// A copy: 1 read and 1 write per cell, the least traffic a step can have.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn copy(input: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();

        if let Some(o) = out.get_mut(idx) {
            *o = input[i];
        }
    }
}

/// A plate of `width` x `height` cells: the top row at HOT, every other cell at 0.
fn initial_grid(width: usize, height: usize) -> Vec<f32> {
    let mut grid = vec![0.0; width * height];
    grid[..width].fill(HOT);
    grid
}

/// 1 Jacobi step on the CPU, the reference for the GPU kernel.
fn cpu_step(old: &[f32], new: &mut [f32], width: usize) {
    let height = old.len() / width;

    for row in 1..height - 1 {
        for col in 1..width - 1 {
            let c = row * width + col;
            let sum = old[c - width] + old[c + width] + old[c - 1] + old[c + 1];
            new[c] = 0.25 * sum;
        }
    }
}

/// The root-mean-square difference between 2 grids, over all cells.
fn rms_change(new: &[f32], old: &[f32]) -> f64 {
    let squares: f64 = new.iter().zip(old).map(|(a, b)| f64::from(a - b).powi(2)).sum();
    (squares / new.len() as f64).sqrt()
}

/// A number in the style of C's %.2e, so that both programs print the same text.
fn sci(v: f64) -> String {
    let exponent = if v == 0.0 { 0 } else { v.abs().log10().floor() as i32 };
    format!("{:.2}e{exponent:+03}", v / 10f64.powi(exponent))
}

fn print_grid(label: &str, grid: &[f32], width: usize) {
    println!("  {label}");
    for row in grid.chunks(width) {
        let cells: Vec<String> = row.iter().map(|v| format!("{v:7.2}")).collect();
        println!("    {}", cells.join(" "));
    }
}

/// Enough 32 x 8 blocks to cover a grid of `width` x `height` cells, 1 thread per cell.
fn grid_2d(width: usize, height: usize) -> LaunchConfig2D {
    let blocks_x = width.div_ceil(BX) as u32;
    let blocks_y = height.div_ceil(BY) as u32;
    LaunchConfig2D::new((blocks_x, blocks_y), (BX as u32, BY as u32), 0)
}

/// Average time of `runs` calls of `launch`, in milliseconds, after 0.4 s of warm-up.
fn time_ms(
    stream: &Arc<CudaStream>,
    runs: u32,
    mut launch: impl FnMut() -> Result<(), Error>,
) -> Result<f64, Error> {
    let warm = Instant::now();
    while warm.elapsed().as_secs_f64() < 0.4 {
        launch()?;
        stream.synchronize()?;
    }

    let start = Instant::now();
    for _ in 0..runs {
        launch()?;
    }
    stream.synchronize()?;

    Ok(start.elapsed().as_secs_f64() * 1e3 / runs as f64)
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };

    let name = ctx.device_name()?;
    let sms = ctx.multiprocessor_count()?;
    println!("GPU: {name}, {sms} SMs");

    // 1. The 6 x 6 plate of the text, small enough to check by hand.
    let (w, h) = (6, 6);
    let start = initial_grid(w, h);
    let mut a = DeviceBuffer::from_host(&stream, &start)?;
    let mut b = DeviceBuffer::from_host(&stream, &start)?;
    let small = module.prepare_jacobi(grid_2d(w, h))?;

    println!("1. 6 x 6 plate, top edge at {HOT}");
    let mut changes = Vec::new();
    for step in 1..=100 {
        module.jacobi(&stream, &small, &a, RowWidth::new(&mut b, w as u32))?;
        std::mem::swap(&mut a, &mut b);

        let (new, old) = (a.to_host_vec(&stream)?, b.to_host_vec(&stream)?);
        changes.push(rms_change(&new, &old));
        if step <= 2 || step == 100 {
            print_grid(&format!("after step {step}"), &new, w);
        }
    }

    println!("  RMS change per step, and its ratio to the step before");
    println!("    step {:3}   {:>9}", 1, sci(changes[0]));
    for step in [2, 3, 4, 10, 20, 40] {
        let ratio = changes[step - 1] / changes[step - 2];
        println!("    step {step:3}   {:>9}   ratio {ratio:.4}", sci(changes[step - 1]));
    }

    let mut cpu_a = start.clone();
    let mut cpu_b = start.clone();
    for _ in 0..100 {
        cpu_step(&cpu_a, &mut cpu_b, w);
        std::mem::swap(&mut cpu_a, &mut cpu_b);
    }
    let small_same = a.to_host_vec(&stream)? == cpu_a;
    println!("  same bits as the CPU after 100 steps: {small_same}");

    // 2. A 64 x 64 plate, iterated until the RMS change per step is below 1e-6.
    let (w, h) = (64, 64);
    let start = initial_grid(w, h);
    let mut a = DeviceBuffer::from_host(&stream, &start)?;
    let mut b = DeviceBuffer::from_host(&stream, &start)?;
    let step = module.prepare_jacobi(grid_2d(w, h))?;

    println!("2. 64 x 64 plate until the RMS change per step is below 1e-6");
    let mut steps = 0;
    loop {
        module.jacobi(&stream, &step, &a, RowWidth::new(&mut b, w as u32))?;
        std::mem::swap(&mut a, &mut b);
        steps += 1;

        if steps % 1000 == 0 {
            let rms = rms_change(&a.to_host_vec(&stream)?, &b.to_host_vec(&stream)?);
            println!("  step {steps:5}   RMS change {}", sci(rms));
            if rms < 1e-6 {
                break;
            }
        }
    }

    let field = a.to_host_vec(&stream)?;
    let mut cpu_a = start.clone();
    let mut cpu_b = start.clone();
    for _ in 0..steps {
        cpu_step(&cpu_a, &mut cpu_b, w);
        std::mem::swap(&mut cpu_a, &mut cpu_b);
    }
    let diff = field.iter().zip(&cpu_a).map(|(g, c)| (g - c).abs()).fold(0.0, f32::max);
    let shown = sci(f64::from(diff));
    println!("  largest difference from the CPU after {steps} steps: {shown}");

    // 3. A 4096 x 4096 plate: time per step and effective bandwidth.
    let (w, h) = (4096, 4096);
    let n = w * h;
    let start = initial_grid(w, h);
    let mut a = DeviceBuffer::from_host(&stream, &start)?;
    let mut b = DeviceBuffer::from_host(&stream, &start)?;
    let big = module.prepare_jacobi(grid_2d(w, h))?;
    let copy = module.prepare_copy(LaunchConfig1D::new((n / 256) as u32, 256, 0))?;

    let ms_copy = time_ms(&stream, 50, || Ok(module.copy(&stream, &copy, &a, &mut b)?))?;

    let ms_jacobi = time_ms(&stream, 50, || {
        module.jacobi(&stream, &big, &a, RowWidth::new(&mut b, w as u32))?;
        module.jacobi(&stream, &big, &b, RowWidth::new(&mut a, w as u32))?;
        Ok(())
    })? / 2.0;

    let gb = |ms: f64| (2 * n * 4) as f64 / 1e9 / (ms / 1e3);
    println!("3. 4096 x 4096 plate, 8 bytes per cell per step (1 read, 1 write)");
    println!("  copy kernel        {ms_copy:6.3} ms   {:6.1} GB/s", gb(ms_copy));
    println!("  jacobi             {ms_jacobi:6.3} ms   {:6.1} GB/s", gb(ms_jacobi));

    let passed = small_same && diff == 0.0;
    println!("{}", if passed { "PASSED" } else { "FAILED" });
    Ok(())
}
