//! Chapter 7 in Rust: coalescing, and a matrix transpose through shared memory.
//!
//!     cargo oxide run

use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D, LaunchConfig2D};
use cuda_device::thread::Index2D;
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, launch_contract, thread};

const WIDTH: usize = 4096;
const N: usize = WIDTH * WIDTH; // 16,777,216 floats, 64 MiB

type Error = Box<dyn std::error::Error>;

#[cuda_module]
mod kernels {
    use super::*;

    /// Thread i reads element i * stride.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn strided(input: &[f32], mut out: DisjointSlice<f32>, stride: u32) {
        let idx = thread::index_1d();
        let i = idx.get();

        if let Some(o) = out.get_mut(idx) {
            *o = input[i * stride as usize];
        }
    }

    #[kernel]
    #[launch_contract(domain = 2, coordinates = u32, block = (32, 32, 1))]
    pub fn transpose_naive(input: &[f32], mut out: DisjointSlice<f32, Index2D<WIDTH>>) {
        let x = (thread::blockIdx_x() * 32 + thread::threadIdx_x()) as usize;
        let y = (thread::blockIdx_y() * 32 + thread::threadIdx_y()) as usize;

        // SAFETY: (x, y) -> x * WIDTH + y is a permutation of the matrix, so no 2 threads
        // write the same element, and every index is below WIDTH * WIDTH.
        unsafe { *out.get_unchecked_mut(x * WIDTH + y) = input[y * WIDTH + x] };
    }

    #[kernel]
    #[launch_contract(domain = 2, coordinates = u32, block = (32, 32, 1))]
    pub fn transpose_tiled(input: &[f32], mut out: DisjointSlice<f32, Index2D<WIDTH>>) {
        static mut TILE: SharedArray<f32, { 32 * 33 }> = SharedArray::UNINIT;

        let (tx, ty) = (thread::threadIdx_x() as usize, thread::threadIdx_y() as usize);
        let bx = thread::blockIdx_x() as usize * 32;
        let by = thread::blockIdx_y() as usize * 32;

        // SAFETY: each thread writes only its own slot of the tile, before the barrier.
        unsafe { TILE[ty * 33 + tx] = input[(by + ty) * WIDTH + bx + tx] };
        thread::sync_threads();

        // SAFETY: the tile is only read after the barrier. The map
        // (tx, ty) -> (bx + ty) * WIDTH + by + tx is a permutation of the matrix,
        // so no 2 threads write the same element.
        unsafe { *out.get_unchecked_mut((bx + ty) * WIDTH + by + tx) = TILE[tx * 33 + ty] };
    }
}

/// Average time of 20 runs of `launch`, in milliseconds, after 1 warm-up run.
fn time_ms(
    stream: &Arc<CudaStream>,
    mut launch: impl FnMut() -> Result<(), Error>,
) -> Result<f64, Error> {
    launch()?;
    stream.synchronize()?;

    let start = Instant::now();
    for _ in 0..20 {
        launch()?;
    }
    stream.synchronize()?;

    Ok(start.elapsed().as_secs_f64() * 1e3 / 20.0)
}

fn gb_per_s(bytes: usize, ms: f64) -> f64 {
    bytes as f64 / 1e9 / (ms / 1e3)
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };

    // 1. Strided reads, from 2 GiB: room for stride 32, too big for any L2.
    let big = DeviceBuffer::<f32>::zeroed(&stream, 32 * N)?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, N)?;
    let launch = module.prepare_strided(LaunchConfig1D::new((N / 256) as u32, 256, 0))?;
    let mut read = |stride| -> Result<(), Error> {
        Ok(module.strided(&stream, &launch, &big, &mut out, stride)?)
    };

    time_ms(&stream, || read(1))?; // warm up the clocks

    println!("each thread reads 1 float, neighbours `stride` floats apart");
    for stride in [1u32, 2, 4, 8, 16, 32] {
        let ms = time_ms(&stream, || read(stride))?;
        let useful = gb_per_s(2 * N * 4, ms);
        println!("  stride {stride:2}   {ms:7.3} ms   {useful:6.1} GB/s useful");
    }

    // 2. Transposes of a 4096 x 4096 matrix.
    let values: Vec<f32> = (0..N).map(|i| (i % 1000) as f32).collect();
    let input = DeviceBuffer::from_host(&stream, &values)?;
    let blocks = (WIDTH / 32) as u32;
    let grid = LaunchConfig2D::new((blocks, blocks), (32, 32), 0);
    let naive = module.prepare_transpose_naive(grid)?;
    let tiled = module.prepare_transpose_tiled(grid)?;

    let ms_naive = time_ms(&stream, || {
        Ok(module.transpose_naive(&stream, &naive, &input, &mut out)?)
    })?;
    let ms_tiled = time_ms(&stream, || {
        Ok(module.transpose_tiled(&stream, &tiled, &input, &mut out)?)
    })?;

    let (gb_naive, gb_tiled) = (gb_per_s(2 * N * 4, ms_naive), gb_per_s(2 * N * 4, ms_tiled));
    println!("transpose of a {WIDTH} x {WIDTH} matrix");
    println!("  naive                 {ms_naive:6.2} ms   {gb_naive:6.1} GB/s");
    println!("  tiled and padded      {ms_tiled:6.2} ms   {gb_tiled:6.1} GB/s");

    let result = out.to_host_vec(&stream)?;
    let correct = (0..WIDTH)
        .all(|y| (0..WIDTH).all(|x| result[x * WIDTH + y] == values[y * WIDTH + x]));
    println!("tiled transpose correct: {}", if correct { "yes" } else { "no" });

    Ok(())
}
