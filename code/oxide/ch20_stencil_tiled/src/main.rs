//! Chapter 20: tiled Jacobi steps with halo cells in shared memory, in Rust.
//!
//!     cargo oxide run

use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D, LaunchConfig2D};
use cuda_device::thread::Runtime2DIndex;
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, launch_contract, thread};
use cuda_host::RowWidth;

const BX: usize = 32; // cells computed by a block, across
const BY: usize = 8; // cells computed by a block, down
const TW: usize = BX + 2; // tile width: the block's cells and a 1-cell halo
const TH: usize = BY + 2; // tile height
const TT: usize = 32; // temporal blocking: a 32 x 32 tile, 1 thread per tile cell
const STEPS: usize = 4; // temporal blocking: Jacobi steps per load
const OUT: usize = TT - 2 * STEPS; // temporal blocking: cells written per block, across
const HOT: f32 = 100.0; // the top edge; the other edges are held at 0

type Error = Box<dyn std::error::Error>;

#[cuda_module]
mod kernels {
    use super::*;

    /// 1 Jacobi step, 1 thread per cell, every neighbour read from global memory.
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

    /// 1 Jacobi step from a 34 x 10 tile; 32 x 8 threads load it, some 2 or 4 cells each.
    #[kernel]
    #[launch_contract(domain = 2, block = (32, 8, 1))]
    pub fn jacobi_tiled(old: &[f32], mut new: DisjointSlice<f32, Runtime2DIndex>) {
        static mut TILE: SharedArray<f32, { TW * TH }> = SharedArray::UNINIT;

        let width = new.row_width() as usize;
        let height = new.len() / width;
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let top = thread::blockIdx_y() as usize * BY;
        let left = thread::blockIdx_x() as usize * BX;

        let mut i = ty;
        while i < TH {
            let mut j = tx;
            while j < TW {
                let r = (top + i).wrapping_sub(1);
                let c = (left + j).wrapping_sub(1);
                let value = if r < height && c < width { old[r * width + c] } else { 0.0 };

                // SAFETY: slot (i, j) is written by thread (i mod 8, j mod 32) alone.
                unsafe { TILE[i * TW + j] = value };
                j += BX;
            }
            i += BY;
        }
        thread::sync_threads();

        if let Some(idx) = thread::index_2d_runtime(&new) {
            let (row, col) = (idx.row(), idx.col());
            let interior = row >= 1 && row + 1 < height && col >= 1 && col + 1 < width;

            if interior {
                let s = (ty + 1) * TW + tx + 1;
                // SAFETY: after the barrier no thread writes the tile.
                let (up, down) = unsafe { (TILE[s - TW], TILE[s + TW]) };
                // SAFETY: as above.
                let (left, right) = unsafe { (TILE[s - 1], TILE[s + 1]) };
                let sum = up + down + left + right;

                if let Some(cell) = new.get_mut(idx) {
                    *cell = 0.25 * sum;
                }
            }
        }
    }

    /// 1 Jacobi step from a 34 x 10 tile; 34 x 10 threads load 1 cell each, 32 x 8 compute.
    #[kernel]
    #[launch_contract(domain = 2, block = (34, 10, 1))]
    pub fn jacobi_halo_threads(old: &[f32], mut new: DisjointSlice<f32, Runtime2DIndex>) {
        static mut TILE: SharedArray<f32, { TW * TH }> = SharedArray::UNINIT;

        let width = new.row_width() as usize;
        let height = new.len() / width;
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let r = (thread::blockIdx_y() as usize * BY + ty).wrapping_sub(1);
        let c = (thread::blockIdx_x() as usize * BX + tx).wrapping_sub(1);
        let s = ty * TW + tx;

        let value = if r < height && c < width { old[r * width + c] } else { 0.0 };
        // SAFETY: slot s belongs to thread (ty, tx) alone.
        unsafe { TILE[s] = value };
        thread::sync_threads();

        let inner = tx >= 1 && tx <= BX && ty >= 1 && ty <= BY;
        let interior = r >= 1 && r + 1 < height && c >= 1 && c + 1 < width;

        if inner && interior {
            // SAFETY: after the barrier no thread writes the tile.
            let (up, down) = unsafe { (TILE[s - TW], TILE[s + TW]) };
            // SAFETY: as above.
            let (left, right) = unsafe { (TILE[s - 1], TILE[s + 1]) };
            let sum = up + down + left + right;

            // SAFETY: the inner threads of the blocks cover each cell of the grid once,
            // so no other thread writes cell (r, c), and r * width + c lies inside `new`.
            unsafe { *new.get_unchecked_mut(r * width + c) = 0.25 * sum };
        }
    }

    /// STEPS Jacobi steps from 1 load: a 32 x 32 tile, ping-pong in shared memory.
    #[kernel]
    #[launch_contract(domain = 2, block = (32, 32, 1))]
    pub fn jacobi_temporal(old: &[f32], mut new: DisjointSlice<f32, Runtime2DIndex>) {
        static mut TILES: SharedArray<f32, { 2 * TT * TT }> = SharedArray::UNINIT;

        let width = new.row_width() as usize;
        let height = new.len() / width;
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let r = (thread::blockIdx_y() as usize * OUT + ty).wrapping_sub(STEPS);
        let c = (thread::blockIdx_x() as usize * OUT + tx).wrapping_sub(STEPS);
        let s = ty * TT + tx;
        let inside = r < height && c < width;
        let interior = r >= 1 && r + 1 < height && c >= 1 && c + 1 < width;

        let value = if inside { old[r * width + c] } else { 0.0 };
        // SAFETY: slot s of the first tile belongs to thread (ty, tx) alone.
        unsafe { TILES[s] = value };
        thread::sync_threads();

        let mut step = 1;
        #[unroll]
        while step <= STEPS {
            let src = (step - 1) % 2 * TT * TT + s;
            let dst = step % 2 * TT * TT + s;
            let valid = tx >= step && tx < TT - step && ty >= step && ty < TT - step;

            if valid {
                // SAFETY: in this step the tile at src is only read, slot dst belongs to
                // thread (ty, tx) alone, and the barrier below separates the steps.
                unsafe {
                    let (up, down) = (TILES[src - TT], TILES[src + TT]);
                    let (left, right) = (TILES[src - 1], TILES[src + 1]);
                    let sum = up + down + left + right;
                    TILES[dst] = if interior { 0.25 * sum } else { TILES[src] };
                }
            }
            thread::sync_threads();
            step += 1;
        }

        let owned = tx >= STEPS && tx < TT - STEPS && ty >= STEPS && ty < TT - STEPS;
        if owned && interior {
            let last = STEPS % 2 * TT * TT + s;
            // SAFETY: the owned squares of the blocks cover each cell of the grid once,
            // so no other thread writes cell (r, c), and r * width + c lies inside `new`.
            unsafe { *new.get_unchecked_mut(r * width + c) = TILES[last] };
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

/// The 4 ways to advance the plate: 1 step per launch, or STEPS steps per launch.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Plain,
    Tiled,
    HaloThreads,
    Temporal,
}

impl Kind {
    fn steps_per_launch(self) -> usize {
        if self == Kind::Temporal { STEPS } else { 1 }
    }
}

type Buffer = DeviceBuffer<f32>;

/// The 4 kernels prepared for a plate of `width` x `height` cells, as 1 launch function.
fn stepper<'a>(
    module: &'a kernels::LoadedModule,
    stream: &'a Arc<CudaStream>,
    width: usize,
    height: usize,
) -> Result<impl Fn(Kind, &Buffer, &mut Buffer) -> Result<(), Error> + 'a, Error> {
    let (bx, by) = (width.div_ceil(BX) as u32, height.div_ceil(BY) as u32);
    let (ox, oy) = (width.div_ceil(OUT) as u32, height.div_ceil(OUT) as u32);
    let tt = TT as u32;

    let one_per_cell = LaunchConfig2D::new((bx, by), (BX as u32, BY as u32), 0);
    let one_per_tile_cell = LaunchConfig2D::new((bx, by), (TW as u32, TH as u32), 0);
    let several_steps = LaunchConfig2D::new((ox, oy), (tt, tt), 0);

    let plain = module.prepare_jacobi(one_per_cell)?;
    let tiled = module.prepare_jacobi_tiled(one_per_cell)?;
    let halo = module.prepare_jacobi_halo_threads(one_per_tile_cell)?;
    let temporal = module.prepare_jacobi_temporal(several_steps)?;

    Ok(move |kind: Kind, old: &Buffer, new: &mut Buffer| {
        let new = RowWidth::new(new, width as u32);
        match kind {
            Kind::Plain => module.jacobi(stream, &plain, old, new)?,
            Kind::Tiled => module.jacobi_tiled(stream, &tiled, old, new)?,
            Kind::HaloThreads => module.jacobi_halo_threads(stream, &halo, old, new)?,
            Kind::Temporal => module.jacobi_temporal(stream, &temporal, old, new)?,
        }
        Ok(())
    })
}

/// The plate after `steps` Jacobi steps of `kind`, starting from `start`.
fn run(
    stream: &Arc<CudaStream>,
    launch: &impl Fn(Kind, &Buffer, &mut Buffer) -> Result<(), Error>,
    kind: Kind,
    start: &[f32],
    steps: usize,
) -> Result<Vec<f32>, Error> {
    let mut a = DeviceBuffer::from_host(stream, start)?;
    let mut b = DeviceBuffer::from_host(stream, start)?;

    for _ in 0..steps / kind.steps_per_launch() {
        launch(kind, &a, &mut b)?;
        std::mem::swap(&mut a, &mut b);
    }
    Ok(a.to_host_vec(stream)?)
}

/// A plate of `width` x `height` cells: the top row at HOT, every other cell at 0.
fn initial_grid(width: usize, height: usize) -> Vec<f32> {
    let mut grid = vec![0.0; width * height];
    grid[..width].fill(HOT);
    grid
}

fn print_grid(label: &str, grid: &[f32], width: usize) {
    println!("  {label}");
    for row in grid.chunks(width) {
        let cells: Vec<String> = row.iter().map(|v| format!("{v:7.2}")).collect();
        println!("    {}", cells.join(" "));
    }
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

    let kinds = [Kind::Tiled, Kind::HaloThreads, Kind::Temporal];
    let names = ["loop loader", "extra threads", "4 steps per load"];

    // 1. The 6 x 6 plate of the text: every tiled kernel against the plain kernel.
    let (w, h) = (6, 6);
    let start = initial_grid(w, h);
    let small = stepper(&module, &stream, w, h)?;

    println!("1. 6 x 6 plate, top edge at {HOT}");
    print_grid("after step 2, tiled", &run(&stream, &small, Kind::Tiled, &start, 2)?, w);

    let mut small_same = true;
    let reference = run(&stream, &small, Kind::Plain, &start, 4)?;
    for (kind, name) in kinds.iter().zip(names) {
        let same = run(&stream, &small, *kind, &start, 4)? == reference;
        println!("  {name:24} same bits as the plain kernel after 4 steps: {same}");
        small_same &= same;
    }

    // 2. A 4096 x 4096 plate: time per step and effective bandwidth.
    let (w, h) = (4096, 4096);
    let n = w * h;
    let start = initial_grid(w, h);
    let big = stepper(&module, &stream, w, h)?;
    let mut a = DeviceBuffer::from_host(&stream, &start)?;
    let mut b = DeviceBuffer::from_host(&stream, &start)?;
    let copy = module.prepare_copy(LaunchConfig1D::new((n / 256) as u32, 256, 0))?;

    let gb = |ms: f64| (2 * n * 4) as f64 / 1e9 / (ms / 1e3);
    println!("2. 4096 x 4096 plate, 8 bytes per cell per step (1 read, 1 write)");

    let ms = time_ms(&stream, 50, || Ok(module.copy(&stream, &copy, &a, &mut b)?))?;
    println!("  {:24} {ms:6.3} ms   {:6.1} GB/s", "copy kernel", gb(ms));

    let all_kinds = [Kind::Plain, Kind::Tiled, Kind::HaloThreads, Kind::Temporal];
    let all_names = ["plain", names[0], names[1], names[2]];
    for (kind, name) in all_kinds.iter().zip(all_names) {
        let two_launches = time_ms(&stream, 50, || {
            big(*kind, &a, &mut b)?;
            big(*kind, &b, &mut a)
        })?;
        let ms = two_launches / (2 * kind.steps_per_launch()) as f64;
        println!("  {name:24} {ms:6.3} ms   {:6.1} GB/s", gb(ms));
    }

    let mut big_same = true;
    let reference = run(&stream, &big, Kind::Plain, &start, 8)?;
    for (kind, name) in kinds.iter().zip(names) {
        let same = run(&stream, &big, *kind, &start, 8)? == reference;
        println!("  {name:24} same bits as the plain kernel after 8 steps: {same}");
        big_same &= same;
    }

    let passed = small_same && big_same;
    println!("{}", if passed { "PASSED" } else { "FAILED" });
    Ok(())
}
