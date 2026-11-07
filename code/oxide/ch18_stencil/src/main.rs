//! Chapter 18 in Rust: Jacobi iteration for the Laplace equation with a 5-point stencil.
//!
//!     cargo oxide run

use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D, LaunchConfig2D};
use cuda_device::thread::Runtime2DIndex;
use cuda_device::{DisjointSlice, SharedArray, cuda_module, device, kernel, launch_contract};
use cuda_device::{thread, warp};
use cuda_host::RowWidth;

const BX: usize = 32;
const BY: usize = 8;
const TW: usize = BX + 2;
const TH: usize = BY + 2;
const SUM_BLOCKS: u32 = 64;
const HOT: f32 = 100.0;

type Error = Box<dyn std::error::Error>;

#[cuda_module]
mod kernels {
    use super::*;

    /// 1 Jacobi step: every interior cell becomes the mean of its 4 neighbours.
    #[kernel]
    #[launch_contract(domain = 2, block = (32, 8, 1))]
    pub fn jacobi(old: &[f32], mut new: DisjointSlice<f32, Runtime2DIndex>) {
        let width = new.row_width() as usize;
        let height = new.len() / width;

        if let Some(idx) = thread::index_2d_runtime(&new) {
            let (row, col) = (idx.row(), idx.col());

            if row >= 1 && row + 1 < height && col >= 1 && col + 1 < width {
                let c = row * width + col;
                let sum = old[c - width] + old[c + width] + old[c - 1] + old[c + 1];

                if let Some(cell) = new.get_mut(idx) {
                    *cell = 0.25 * sum;
                }
            }
        }
    }

    /// The same step, with the block's cells and a 1-cell halo staged in shared memory.
    #[kernel]
    #[launch_contract(domain = 2, block = (32, 8, 1))]
    pub fn jacobi_tiled(old: &[f32], mut new: DisjointSlice<f32, Runtime2DIndex>) {
        static mut TILE: SharedArray<f32, { TW * TH }> = SharedArray::UNINIT;

        let width = new.row_width() as usize;
        let height = new.len() / width;
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let left = thread::blockIdx_x() as usize * BX;
        let top = thread::blockIdx_y() as usize * BY;

        let mut k = ty * BX + tx;
        while k < TW * TH {
            let r = (top + k / TW).wrapping_sub(1);
            let c = (left + k % TW).wrapping_sub(1);
            let value = if r < height && c < width { old[r * width + c] } else { 0.0 };

            // SAFETY: slot k is written by thread k mod 256 alone, before the barrier.
            unsafe { TILE[k] = value };
            k += BX * BY;
        }
        thread::sync_threads();

        if let Some(idx) = thread::index_2d_runtime(&new) {
            let (row, col) = (idx.row(), idx.col());

            if row >= 1 && row + 1 < height && col >= 1 && col + 1 < width {
                let s = (ty + 1) * TW + tx + 1;
                // SAFETY: after the barrier the tile is only read.
                let (up, down) = unsafe { (TILE[s - TW], TILE[s + TW]) };
                // SAFETY: as above.
                let (west, east) = unsafe { (TILE[s - 1], TILE[s + 1]) };
                let sum = up + down + west + east;

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

    #[device]
    fn warp_sum(mut v: f32) -> f32 {
        for offset in [16, 8, 4, 2, 1] {
            v += warp::shuffle_down_f32(v, offset);
        }
        v
    }

    /// Per block: the sum of (a[i] - b[i])^2 over a grid-stride share of the cells.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn change(a: &[f32], b: &[f32], mut partial: DisjointSlice<f32>) {
        static mut WARPS: SharedArray<f32, 8> = SharedArray::UNINIT;

        let stride = (thread::gridDim_x() * 256) as usize;
        let lane = warp::lane_id() as usize;
        let warp = thread::threadIdx_x() as usize / 32;

        let mut sum = 0.0;
        let mut i = thread::index_1d().get();
        while i < a.len() {
            let d = a[i] - b[i];
            sum += d * d;
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
}

/// A grid of `width` x `height` cells, top row at HOT, every other cell at 0.
fn initial_grid(width: usize, height: usize) -> Vec<f32> {
    let mut grid = vec![0.0; width * height];
    grid[..width].fill(HOT);
    grid
}

/// 1 Jacobi step on the CPU, the reference for the GPU kernels.
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

fn print_grid(label: &str, grid: &[f32], width: usize) {
    println!("  {label}");
    for row in grid.chunks(width) {
        let cells: Vec<String> = row.iter().map(|v| format!("{v:7.2}")).collect();
        println!("    {}", cells.join(" "));
    }
}

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

    // 1. A 5 x 5 grid, small enough to check by hand.
    let (w, h) = (5, 5);
    let start = initial_grid(w, h);
    let mut a = DeviceBuffer::from_host(&stream, &start)?;
    let mut b = DeviceBuffer::from_host(&stream, &start)?;
    let small = module.prepare_jacobi(grid_2d(w, h))?;

    println!("1. 5 x 5 grid, top edge at {HOT}");
    for step in 1..=200 {
        module.jacobi(&stream, &small, &a, RowWidth::new(&mut b, w as u32))?;
        std::mem::swap(&mut a, &mut b);

        if step <= 2 || step == 200 {
            print_grid(&format!("after step {step}"), &a.to_host_vec(&stream)?, w);
        }
    }

    // 2. A 64 x 64 grid, iterated until the root-mean-square change per step < 1e-6.
    let (w, h) = (64, 64);
    let start = initial_grid(w, h);
    let mut a = DeviceBuffer::from_host(&stream, &start)?;
    let mut b = DeviceBuffer::from_host(&stream, &start)?;
    let mut partial = DeviceBuffer::<f32>::zeroed(&stream, SUM_BLOCKS as usize)?;
    let step = module.prepare_jacobi(grid_2d(w, h))?;
    let change = module.prepare_change(LaunchConfig1D::new(SUM_BLOCKS, 256, 0))?;

    println!("2. 64 x 64 grid until the RMS change per step is below 1e-6");
    let mut steps = 0;
    loop {
        module.jacobi(&stream, &step, &a, RowWidth::new(&mut b, w as u32))?;
        steps += 1;

        if steps % 1000 == 0 {
            module.change(&stream, &change, &a, &b, &mut partial)?;
            let total: f32 = partial.to_host_vec(&stream)?.iter().sum();
            let rms = (total / (w * h) as f32).sqrt();
            println!("  step {steps:5}   RMS change {rms:.2e}");

            if rms < 1e-6 {
                std::mem::swap(&mut a, &mut b);
                break;
            }
        }
        std::mem::swap(&mut a, &mut b);
    }

    let field = a.to_host_vec(&stream)?;
    println!("  every 9th cell of every 9th row (8 x 8 of the 64 x 64 cells)");
    for row in (0..h).step_by(9) {
        let mut line = String::new();
        for col in (0..w).step_by(9) {
            line += &format!("{:7.2}", field[row * w + col]);
        }
        println!("   {line}");
    }

    let mut cpu_a = start.clone();
    let mut cpu_b = start.clone();
    for _ in 0..steps {
        cpu_step(&cpu_a, &mut cpu_b, w);
        std::mem::swap(&mut cpu_a, &mut cpu_b);
    }
    let diff = field.iter().zip(&cpu_a).map(|(g, c)| (g - c).abs()).fold(0.0, f32::max);
    println!("  largest difference from the CPU after {steps} steps: {diff:.2e}");

    // 3. A 4096 x 4096 grid: time per step and effective bandwidth.
    let (w, h) = (4096, 4096);
    let n = w * h;
    let start = initial_grid(w, h);
    let mut a = DeviceBuffer::from_host(&stream, &start)?;
    let mut b = DeviceBuffer::from_host(&stream, &start)?;
    let global = module.prepare_jacobi(grid_2d(w, h))?;
    let tiled = module.prepare_jacobi_tiled(grid_2d(w, h))?;
    let copy = module.prepare_copy(LaunchConfig1D::new((n / 256) as u32, 256, 0))?;

    let ms_copy = time_ms(&stream, 50, || Ok(module.copy(&stream, &copy, &a, &mut b)?))?;

    let ms_global = time_ms(&stream, 50, || {
        module.jacobi(&stream, &global, &a, RowWidth::new(&mut b, w as u32))?;
        module.jacobi(&stream, &global, &b, RowWidth::new(&mut a, w as u32))?;
        Ok(())
    })? / 2.0;

    let ms_tiled = time_ms(&stream, 50, || {
        module.jacobi_tiled(&stream, &tiled, &a, RowWidth::new(&mut b, w as u32))?;
        module.jacobi_tiled(&stream, &tiled, &b, RowWidth::new(&mut a, w as u32))?;
        Ok(())
    })? / 2.0;

    let gb = |ms: f64| (2 * n * 4) as f64 / 1e9 / (ms / 1e3);
    println!("3. 4096 x 4096 grid, 8 bytes per cell per step (1 read, 1 write)");
    println!("  copy kernel        {ms_copy:6.3} ms   {:6.1} GB/s", gb(ms_copy));
    println!("  jacobi             {ms_global:6.3} ms   {:6.1} GB/s", gb(ms_global));
    println!("  jacobi, tiled      {ms_tiled:6.3} ms   {:6.1} GB/s", gb(ms_tiled));

    let mut results = Vec::new();
    for use_tiles in [false, true] {
        a.copy_from_host(&stream, &start)?;
        b.copy_from_host(&stream, &start)?;

        for _ in 0..10 {
            if use_tiles {
                module.jacobi_tiled(&stream, &tiled, &a, RowWidth::new(&mut b, w as u32))?;
            } else {
                module.jacobi(&stream, &global, &a, RowWidth::new(&mut b, w as u32))?;
            }
            std::mem::swap(&mut a, &mut b);
        }
        results.push(a.to_host_vec(&stream)?);
    }

    let same = results[0] == results[1];
    println!("  both kernels give the same bits after 10 steps: {same}");

    let passed = diff < 1e-4 && same;
    println!("{}", if passed { "PASSED" } else { "FAILED" });
    Ok(())
}
