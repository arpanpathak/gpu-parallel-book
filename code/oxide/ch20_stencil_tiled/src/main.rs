//! Chapter 20: tiled Jacobi steps with halo cells in shared memory, in Rust.
//!
//!     cargo oxide run

use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D, LaunchConfig2D};
use cuda_device::thread::Runtime2DIndex;
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, launch_contract, thread};
use cuda_host::RowWidth;

const BLOCK_WIDTH: usize = 32; // cells computed by a block, across
const BLOCK_HEIGHT: usize = 8; // cells computed by a block, down
const TILE_WIDTH: usize = BLOCK_WIDTH + 2; // the block's cells and a 1-cell halo
const TILE_HEIGHT: usize = BLOCK_HEIGHT + 2;
const TEMPORAL_TILE: usize = 32; // temporal blocking: a 32 x 32 tile, 1 thread per cell
const STEPS: usize = 4; // temporal blocking: Jacobi steps per load
const OWN_WIDTH: usize = TEMPORAL_TILE - 2 * STEPS; // temporal blocking: cells written
const HOT: f32 = 100.0; // the top edge; the other edges are held at 0

type Error = Box<dyn std::error::Error>;

#[cuda_module]
mod kernels {
    use super::*;

    /// 1 Jacobi step, 1 thread per cell, every neighbor read from global memory.
    #[kernel]
    #[launch_contract(domain = 2, block = (32, 8, 1))]
    pub fn jacobi(old: &[f32], mut new: DisjointSlice<f32, Runtime2DIndex>) {
        let width = new.row_width() as usize;
        let height = new.len() / width;

        if let Some(idx) = thread::index_2d_runtime(&new) {
            let (row, col) = (idx.row(), idx.col());
            let interior = row >= 1 && row + 1 < height && col >= 1 && col + 1 < width;

            if interior {
                let center = row * width + col;
                let (up, down) = (old[center - width], old[center + width]);
                let (to_left, to_right) = (old[center - 1], old[center + 1]);
                let sum = up + down + to_left + to_right;

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
        const AREA: usize = TILE_WIDTH * TILE_HEIGHT;
        static mut TILE: SharedArray<f32, AREA> = SharedArray::UNINIT;

        let width = new.row_width() as usize;
        let height = new.len() / width;
        let thread_x = thread::threadIdx_x() as usize;
        let thread_y = thread::threadIdx_y() as usize;
        let top = thread::blockIdx_y() as usize * BLOCK_HEIGHT;
        let left = thread::blockIdx_x() as usize * BLOCK_WIDTH;

        let mut slot_row = thread_y;
        while slot_row < TILE_HEIGHT {
            let mut slot_col = thread_x;
            while slot_col < TILE_WIDTH {
                let cell_row = (top + slot_row).wrapping_sub(1);
                let cell_col = (left + slot_col).wrapping_sub(1);
                let inside = cell_row < height && cell_col < width;
                let value = if inside { old[cell_row * width + cell_col] } else { 0.0 };

                // SAFETY: slot (slot_row, slot_col) is written by 1 thread alone,
                // the thread (slot_row mod 8, slot_col mod 32).
                unsafe { TILE[slot_row * TILE_WIDTH + slot_col] = value };
                slot_col += BLOCK_WIDTH;
            }
            slot_row += BLOCK_HEIGHT;
        }
        thread::sync_threads();

        if let Some(idx) = thread::index_2d_runtime(&new) {
            let (row, col) = (idx.row(), idx.col());
            let interior = row >= 1 && row + 1 < height && col >= 1 && col + 1 < width;

            if interior {
                let center = (thread_y + 1) * TILE_WIDTH + thread_x + 1;
                // SAFETY: after the barrier no thread writes the tile.
                let up = unsafe { TILE[center - TILE_WIDTH] };
                // SAFETY: as above.
                let down = unsafe { TILE[center + TILE_WIDTH] };
                // SAFETY: as above.
                let (to_left, to_right) = unsafe { (TILE[center - 1], TILE[center + 1]) };
                let sum = up + down + to_left + to_right;

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
        const AREA: usize = TILE_WIDTH * TILE_HEIGHT;
        static mut TILE: SharedArray<f32, AREA> = SharedArray::UNINIT;

        let width = new.row_width() as usize;
        let height = new.len() / width;
        let slot_col = thread::threadIdx_x() as usize;
        let slot_row = thread::threadIdx_y() as usize;
        let row = (thread::blockIdx_y() as usize * BLOCK_HEIGHT + slot_row).wrapping_sub(1);
        let col = (thread::blockIdx_x() as usize * BLOCK_WIDTH + slot_col).wrapping_sub(1);
        let slot = slot_row * TILE_WIDTH + slot_col;

        let value = if row < height && col < width { old[row * width + col] } else { 0.0 };
        // SAFETY: the slot belongs to this thread alone.
        unsafe { TILE[slot] = value };
        thread::sync_threads();

        let inner_row = slot_row >= 1 && slot_row <= BLOCK_HEIGHT;
        let inner_col = slot_col >= 1 && slot_col <= BLOCK_WIDTH;
        let interior = row >= 1 && row + 1 < height && col >= 1 && col + 1 < width;

        if inner_row && inner_col && interior {
            // SAFETY: after the barrier no thread writes the tile.
            let (up, down) = unsafe { (TILE[slot - TILE_WIDTH], TILE[slot + TILE_WIDTH]) };
            // SAFETY: as above.
            let (to_left, to_right) = unsafe { (TILE[slot - 1], TILE[slot + 1]) };
            let sum = up + down + to_left + to_right;

            // SAFETY: the inner threads of the blocks cover each cell of the grid once,
            // so no other thread writes cell (row, col), which lies inside `new`.
            unsafe { *new.get_unchecked_mut(row * width + col) = 0.25 * sum };
        }
    }

    /// STEPS Jacobi steps from 1 load: a 32 x 32 tile, ping-pong in shared memory.
    #[kernel]
    #[launch_contract(domain = 2, block = (32, 32, 1))]
    pub fn jacobi_temporal(old: &[f32], mut new: DisjointSlice<f32, Runtime2DIndex>) {
        const AREA: usize = TEMPORAL_TILE * TEMPORAL_TILE;
        static mut TILES: SharedArray<f32, { 2 * AREA }> = SharedArray::UNINIT;

        let width = new.row_width() as usize;
        let height = new.len() / width;
        let slot_col = thread::threadIdx_x() as usize;
        let slot_row = thread::threadIdx_y() as usize;
        let first_row = thread::blockIdx_y() as usize * OWN_WIDTH;
        let first_col = thread::blockIdx_x() as usize * OWN_WIDTH;
        let row = (first_row + slot_row).wrapping_sub(STEPS);
        let col = (first_col + slot_col).wrapping_sub(STEPS);
        let slot = slot_row * TEMPORAL_TILE + slot_col;
        let inside = row < height && col < width;
        let interior = row >= 1 && row + 1 < height && col >= 1 && col + 1 < width;

        let value = if inside { old[row * width + col] } else { 0.0 };
        // SAFETY: this slot of the first tile belongs to this thread alone.
        unsafe { TILES[slot] = value };
        thread::sync_threads();

        let mut step = 1;
        #[unroll]
        while step <= STEPS {
            let src = (step - 1) % 2 * AREA + slot;
            let dst = step % 2 * AREA + slot;
            let last = TEMPORAL_TILE - step;
            let valid_row = slot_row >= step && slot_row < last;
            let valid_col = slot_col >= step && slot_col < last;

            if valid_row && valid_col {
                // SAFETY: in this step the tile at src is only read, slot dst belongs to
                // this thread alone, and the barrier below separates the steps.
                unsafe {
                    let up = TILES[src - TEMPORAL_TILE];
                    let down = TILES[src + TEMPORAL_TILE];
                    let (to_left, to_right) = (TILES[src - 1], TILES[src + 1]);
                    let sum = up + down + to_left + to_right;
                    TILES[dst] = if interior { 0.25 * sum } else { TILES[src] };
                }
            }
            thread::sync_threads();
            step += 1;
        }

        let end = TEMPORAL_TILE - STEPS;
        let owned_row = slot_row >= STEPS && slot_row < end;
        let owned_col = slot_col >= STEPS && slot_col < end;
        if owned_row && owned_col && interior {
            let last = STEPS % 2 * AREA + slot;
            // SAFETY: the owned squares of the blocks cover each cell of the grid once,
            // so no other thread writes cell (row, col), which lies inside `new`.
            unsafe { *new.get_unchecked_mut(row * width + col) = TILES[last] };
        }
    }

    /// A copy: 1 read and 1 write per cell, the least traffic a step can have.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn copy(input: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let index = idx.get();

        if let Some(out_cell) = out.get_mut(idx) {
            *out_cell = input[index];
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
    let block_columns = width.div_ceil(BLOCK_WIDTH) as u32;
    let block_rows = height.div_ceil(BLOCK_HEIGHT) as u32;
    let blocks = (block_columns, block_rows);
    let squares = (width.div_ceil(OWN_WIDTH) as u32, height.div_ceil(OWN_WIDTH) as u32);

    let block = (BLOCK_WIDTH as u32, BLOCK_HEIGHT as u32);
    let tile_block = (TILE_WIDTH as u32, TILE_HEIGHT as u32);
    let square = (TEMPORAL_TILE as u32, TEMPORAL_TILE as u32);

    let one_per_cell = LaunchConfig2D::new(blocks, block, 0);
    let one_per_tile_cell = LaunchConfig2D::new(blocks, tile_block, 0);
    let several_steps = LaunchConfig2D::new(squares, square, 0);

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
    let mut current = DeviceBuffer::from_host(stream, start)?;
    let mut next = DeviceBuffer::from_host(stream, start)?;

    for _ in 0..steps / kind.steps_per_launch() {
        launch(kind, &current, &mut next)?;
        std::mem::swap(&mut current, &mut next);
    }
    Ok(current.to_host_vec(stream)?)
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
        let cells: Vec<String> = row.iter().map(|value| format!("{value:7.2}")).collect();
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
    let (width, height) = (6, 6);
    let start = initial_grid(width, height);
    let small = stepper(&module, &stream, width, height)?;

    println!("1. 6 x 6 plate, top edge at {HOT}");
    let after_two = run(&stream, &small, Kind::Tiled, &start, 2)?;
    print_grid("after step 2, tiled", &after_two, width);

    let mut small_same = true;
    let reference = run(&stream, &small, Kind::Plain, &start, 4)?;
    for (kind, name) in kinds.iter().zip(names) {
        let same = run(&stream, &small, *kind, &start, 4)? == reference;
        println!("  {name:24} same bits as the plain kernel after 4 steps: {same}");
        small_same &= same;
    }

    // 2. A 4096 x 4096 plate: time per step and effective bandwidth.
    let (width, height) = (4096, 4096);
    let cells = width * height;
    let start = initial_grid(width, height);
    let big = stepper(&module, &stream, width, height)?;
    let mut current = DeviceBuffer::from_host(&stream, &start)?;
    let mut next = DeviceBuffer::from_host(&stream, &start)?;
    let copy = module.prepare_copy(LaunchConfig1D::new((cells / 256) as u32, 256, 0))?;

    let gb = |ms: f64| (2 * cells * 4) as f64 / 1e9 / (ms / 1e3);
    println!("2. 4096 x 4096 plate, 8 bytes per cell per step (1 read, 1 write)");

    let copy_once = || Ok(module.copy(&stream, &copy, &current, &mut next)?);
    let ms = time_ms(&stream, 50, copy_once)?;
    println!("  {:24} {ms:6.3} ms   {:6.1} GB/s", "copy kernel", gb(ms));

    let all_kinds = [Kind::Plain, Kind::Tiled, Kind::HaloThreads, Kind::Temporal];
    let all_names = ["plain", names[0], names[1], names[2]];
    for (kind, name) in all_kinds.iter().zip(all_names) {
        let two_launches = time_ms(&stream, 50, || {
            big(*kind, &current, &mut next)?;
            big(*kind, &next, &mut current)
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
