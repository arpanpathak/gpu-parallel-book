//! Chapter 9 in Rust: matrix multiplication by rows, by shared tiles, by 4 x 4 patches.
//!
//!     cargo oxide run

use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig2D};
use cuda_device::thread::Index2D;
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, launch_contract, thread};

const N: usize = 4096; // C = A B, all N x N, stored row by row

type Error = Box<dyn std::error::Error>;

#[cuda_module]
mod kernels {
    use super::*;

    /// 1 thread per element of C, threadIdx.x along a row of C.
    #[kernel]
    #[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1))]
    pub fn by_row(a: &[f32], b: &[f32], mut c: DisjointSlice<f32, Index2D<N>>) {
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();

        let mut sum = 0.0;
        let mut k = 0;
        #[unroll(4)]
        while k < N {
            sum += a[row * N + k] * b[k * N + col];
            k += 1;
        }

        if let Some(idx) = thread::index_2d::<N>()
            && let Some(out) = c.get_mut(idx)
        {
            *out = sum;
        }
    }

    /// A block of 16 x 16 threads computes a 16 x 16 tile of C, from shared memory.
    #[kernel]
    #[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1))]
    pub fn tiled(a: &[f32], b: &[f32], mut c: DisjointSlice<f32, Index2D<N>>) {
        static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;
        let x = thread::threadIdx_x() as usize;
        let y = thread::threadIdx_y() as usize;
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();

        let mut sum = 0.0;
        let mut k0 = 0;
        while k0 < N {
            // SAFETY: each thread writes only its own slot of each tile.
            unsafe {
                TILE_A[y * 16 + x] = a[row * N + k0 + x];
                TILE_B[y * 16 + x] = b[(k0 + y) * N + col];
            }
            thread::sync_threads();

            let mut k = 0;
            #[unroll]
            while k < 16 {
                // SAFETY: between the 2 barriers the tiles are only read.
                sum += unsafe { TILE_A[y * 16 + k] * TILE_B[k * 16 + x] };
                k += 1;
            }
            thread::sync_threads();

            k0 += 16;
        }

        if let Some(idx) = thread::index_2d::<N>()
            && let Some(out) = c.get_mut(idx)
        {
            *out = sum;
        }
    }

    /// A block of 256 threads computes a 64 x 64 tile of C, each thread a 4 x 4 patch.
    #[kernel]
    #[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1))]
    pub fn patches(a: &[f32], b: &[f32], mut c: DisjointSlice<f32, Index2D<N>>) {
        static mut TILE_A: SharedArray<f32, { 64 * 17 }> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, { 16 * 64 }> = SharedArray::UNINIT;
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let row0 = thread::blockIdx_y() as usize * 64;
        let col0 = thread::blockIdx_x() as usize * 64;

        let mut acc = [[0.0f32; 4]; 4];
        let mut k0 = 0;
        while k0 < N {
            let mut e = ty * 16 + tx;
            while e < 1024 {
                // SAFETY: element e of each tile is written by 1 thread.
                unsafe {
                    TILE_A[e / 16 * 17 + e % 16] = a[(row0 + e / 16) * N + k0 + e % 16];
                    TILE_B[e] = b[(k0 + e / 64) * N + col0 + e % 64];
                }
                e += 256;
            }
            thread::sync_threads();

            let mut k = 0;
            #[unroll]
            while k < 16 {
                let mut av = [0.0f32; 4];
                let mut bv = [0.0f32; 4];
                let mut i = 0;
                #[unroll]
                while i < 4 {
                    // SAFETY: between the 2 barriers the tiles are only read.
                    unsafe {
                        av[i] = TILE_A[(ty * 4 + i) * 17 + k];
                        bv[i] = TILE_B[k * 64 + tx * 4 + i];
                    }
                    i += 1;
                }

                let mut ij = 0;
                #[unroll]
                while ij < 16 {
                    acc[ij / 4][ij % 4] += av[ij / 4] * bv[ij % 4];
                    ij += 1;
                }

                k += 1;
            }
            thread::sync_threads();

            k0 += 16;
        }

        for i in 0..4 {
            for j in 0..4 {
                let at = (row0 + ty * 4 + i) * N + col0 + tx * 4 + j;
                // SAFETY: the 4 x 4 patches of all threads cover C once, without overlap.
                unsafe { *c.get_unchecked_mut(at) = acc[i][j] };
            }
        }
    }
}

/// Average time of `runs` runs of `launch`, in milliseconds, after 1 warm-up run.
fn time_ms<F>(stream: &CudaStream, runs: u32, mut launch: F) -> Result<f64, Error>
where
    F: FnMut() -> Result<(), Error>,
{
    launch()?;
    stream.synchronize()?;

    let start = Instant::now();
    for _ in 0..runs {
        launch()?;
    }
    stream.synchronize()?;

    Ok(start.elapsed().as_secs_f64() * 1e3 / runs as f64)
}

/// Values in [-0.5, 0.5) from a small linear congruential generator.
fn random(n: usize, state: &mut u32) -> Vec<f32> {
    let mut values = Vec::with_capacity(n);
    for _ in 0..n {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        values.push((*state >> 8) as f32 / 16_777_216.0 - 0.5);
    }
    values
}

/// The largest difference from a double-precision product, over 64 elements of C.
fn max_error(a: &[f32], b: &[f32], c: &[f32]) -> f64 {
    let mut worst: f64 = 0.0;
    for s in 0..64 {
        let i = s * 977 % N;
        let j = (s * 1231 + 7) % N;

        let mut exact = 0.0;
        for k in 0..N {
            exact += a[i * N + k] as f64 * b[k * N + j] as f64;
        }
        worst = worst.max((exact - c[i * N + j] as f64).abs());
    }
    worst
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };

    let mut state = 1;
    let a_host = random(N * N, &mut state);
    let b_host = random(N * N, &mut state);
    let a = DeviceBuffer::from_host(&stream, &a_host)?;
    let b = DeviceBuffer::from_host(&stream, &b_host)?;
    let mut c = DeviceBuffer::<f32>::zeroed(&stream, N * N)?;

    let g16 = (N / 16) as u32;
    let g64 = (N / 64) as u32;
    let rows = module.prepare_by_row(LaunchConfig2D::new((g16, g16), (16, 16), 0))?;
    let tiles = module.prepare_tiled(LaunchConfig2D::new((g16, g16), (16, 16), 0))?;
    let patch = module.prepare_patches(LaunchConfig2D::new((g64, g64), (16, 16), 0))?;

    // Warm up: the clocks rise under sustained load.
    for _ in 0..10 {
        time_ms(&stream, 5, || Ok(module.patches(&stream, &patch, &a, &b, &mut c)?))?;
    }

    let report = |name: &str, ms: f64, c: &DeviceBuffer<f32>| -> Result<(), Error> {
        let gflops = 2.0 * (N * N * N) as f64 / ms / 1e6;
        let error = max_error(&a_host, &b_host, &c.to_host_vec(&stream)?);
        println!("  {name:<20} {ms:8.2} ms  {gflops:7.1} GFLOP/s  max error {error:.1e}");
        Ok(())
    };

    println!("C = A B, {N} x {N} floats");
    let ms = time_ms(&stream, 3, || Ok(module.by_row(&stream, &rows, &a, &b, &mut c)?))?;
    report("1 per thread", ms, &c)?;

    let ms = time_ms(&stream, 5, || Ok(module.tiled(&stream, &tiles, &a, &b, &mut c)?))?;
    report("tiles of 16 x 16", ms, &c)?;

    let ms = time_ms(&stream, 10, || Ok(module.patches(&stream, &patch, &a, &b, &mut c)?))?;
    report("4 x 4 per thread", ms, &c)?;

    Ok(())
}
