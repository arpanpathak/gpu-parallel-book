//! Chapter 17: tiled matrix multiplication in shared memory, in Rust.
//!
//!     cargo oxide run

use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig2D};
use cuda_device::thread::Index2D;
use cuda_device::{DisjointSlice, DynamicSharedArray, SharedArray};
use cuda_device::{cuda_module, device, kernel, launch_bounds, launch_contract, thread};

const N: usize = 4096; // C = A B, all N x N, stored row by row

type Error = Box<dyn std::error::Error>;

#[cuda_module]
mod kernels {
    use super::*;

    /// 1 thread per element of C, reading A and B from global memory.
    #[kernel]
    #[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1))]
    pub fn naive(a: &[f32], b: &[f32], mut c: DisjointSlice<f32, Index2D<N>>) {
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

    /// Tiles of 16 x 16 in static shared memory; the inner loop is unrolled.
    #[kernel]
    #[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1))]
    pub fn tiled16(a: &[f32], b: &[f32], mut c: DisjointSlice<f32, Index2D<N>>) {
        static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;

        let x = thread::threadIdx_x() as usize;
        let y = thread::threadIdx_y() as usize;
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();

        let mut sum = 0.0;
        let mut k0 = 0;
        while k0 < N {
            // SAFETY: x, y < 16: slot y * 16 + x is below 256 and is this thread's alone.
            unsafe {
                TILE_A[y * 16 + x] = a[row * N + k0 + x];
                TILE_B[y * 16 + x] = b[(k0 + y) * N + col];
            }
            thread::sync_threads();

            let mut k = 0;
            #[unroll]
            while k < 16 {
                // SAFETY: between the 2 barriers no thread writes either tile.
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

    /// The same kernel with the inner loop written as a `for` loop.
    #[kernel]
    #[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1))]
    pub fn tiled16_for(a: &[f32], b: &[f32], mut c: DisjointSlice<f32, Index2D<N>>) {
        static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;

        let x = thread::threadIdx_x() as usize;
        let y = thread::threadIdx_y() as usize;
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();

        let mut sum = 0.0;
        let mut k0 = 0;
        while k0 < N {
            // SAFETY: x, y < 16: slot y * 16 + x is below 256 and is this thread's alone.
            unsafe {
                TILE_A[y * 16 + x] = a[row * N + k0 + x];
                TILE_B[y * 16 + x] = b[(k0 + y) * N + col];
            }
            thread::sync_threads();

            for k in 0..16 {
                // SAFETY: between the 2 barriers no thread writes either tile.
                sum += unsafe { TILE_A[y * 16 + k] * TILE_B[k * 16 + x] };
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

    /// Tiles of T x T in dynamic shared memory: 1 source, 1 kernel for each T.
    #[kernel]
    #[launch_bounds(1024)]
    #[launch_contract(domain = 2, coordinates = u32, dynamic_shared_range = (512, 8192))]
    pub fn tiled<const T: usize>(
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, Index2D<N>>,
    ) {
        let tile_a: *mut f32 = DynamicSharedArray::<f32>::get();
        let tile_b: *mut f32 = DynamicSharedArray::<f32>::offset(T * T * 4);

        let x = thread::threadIdx_x() as usize;
        let y = thread::threadIdx_y() as usize;
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();

        let mut sum = 0.0;
        let mut k0 = 0;
        while k0 < N {
            // SAFETY: the launch gives T x T threads and 2 T^2 floats of shared memory,
            // so slot y * T + x lies inside each tile and is this thread's alone.
            // tiled_config::<T>() builds that launch.
            unsafe {
                *tile_a.add(y * T + x) = a[row * N + k0 + x];
                *tile_b.add(y * T + x) = b[(k0 + y) * N + col];
            }
            thread::sync_threads();

            let mut part = 0.0;
            let mut k = 0;
            #[unroll]
            while k < T {
                // SAFETY: between the 2 barriers no thread writes either tile.
                part += unsafe { *tile_a.add(y * T + k) * *tile_b.add(k * T + x) };
                k += 1;
            }
            sum += part;
            thread::sync_threads();

            k0 += T;
        }

        if let Some(idx) = thread::index_2d::<N>()
            && let Some(out) = c.get_mut(idx)
        {
            *out = sum;
        }
    }

    /// The generic kernel with its inner product in a separate device function.
    #[kernel]
    #[launch_bounds(1024)]
    #[launch_contract(domain = 2, coordinates = u32, dynamic_shared_range = (512, 8192))]
    pub fn tiled_fn<const T: usize>(
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, Index2D<N>>,
    ) {
        let tile_a: *mut f32 = DynamicSharedArray::<f32>::get();
        let tile_b: *mut f32 = DynamicSharedArray::<f32>::offset(T * T * 4);

        let x = thread::threadIdx_x() as usize;
        let y = thread::threadIdx_y() as usize;
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();

        let mut sum = 0.0;
        let mut k0 = 0;
        while k0 < N {
            // SAFETY: as in `tiled`.
            unsafe {
                *tile_a.add(y * T + x) = a[row * N + k0 + x];
                *tile_b.add(y * T + x) = b[(k0 + y) * N + col];
            }
            thread::sync_threads();

            // SAFETY: between the 2 barriers no thread writes either tile.
            sum += unsafe { tile_dot::<T>(tile_a, tile_b, x, y) };
            thread::sync_threads();

            k0 += T;
        }

        if let Some(idx) = thread::index_2d::<N>()
            && let Some(out) = c.get_mut(idx)
        {
            *out = sum;
        }
    }

    /// Row y of tile A times column x of tile B: T products, added in order.
    ///
    /// # Safety
    /// Both pointers address T x T floats that no thread writes during the call.
    #[device]
    unsafe fn tile_dot<const T: usize>(
        tile_a: *const f32,
        tile_b: *const f32,
        x: usize,
        y: usize,
    ) -> f32 {
        let mut sum = 0.0;
        let mut k = 0;
        #[unroll]
        while k < T {
            // SAFETY: y * T + k and k * T + x are both below T * T.
            sum += unsafe { *tile_a.add(y * T + k) * *tile_b.add(k * T + x) };
            k += 1;
        }
        sum
    }
}

/// The launch of tiled::<T>: T x T threads and 2 tiles of T x T floats per block.
fn tiled_config<const T: usize>() -> LaunchConfig2D {
    let blocks = (N / T) as u32;
    let side = T as u32;
    let shared = (2 * T * T * size_of::<f32>()) as u32;
    LaunchConfig2D::new((blocks, blocks), (side, side), shared)
}

/// Average time of `runs` runs of `launch`, in milliseconds, after 1 warm-up run.
fn time_ms<F, E>(stream: &CudaStream, runs: u32, mut launch: F) -> Result<f64, Error>
where
    F: FnMut() -> Result<(), E>,
    E: std::error::Error + 'static,
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

    let blocks = (N / 16) as u32;
    let static16 = LaunchConfig2D::new((blocks, blocks), (16, 16), 0);
    let naive = module.prepare_naive(static16)?;
    let t16 = module.prepare_tiled16(static16)?;
    let t16_for = module.prepare_tiled16_for(static16)?;
    let g8 = module.prepare_tiled::<8>(tiled_config::<8>())?;
    let g16 = module.prepare_tiled::<16>(tiled_config::<16>())?;
    let g32 = module.prepare_tiled::<32>(tiled_config::<32>())?;
    let f16 = module.prepare_tiled_fn::<16>(tiled_config::<16>())?;

    // Warm up: the GPU raises its clocks under sustained load.
    time_ms(&stream, 4, || module.tiled16(&stream, &t16, &a, &b, &mut c))?;

    let mut passed = true;
    let mut report = |name: &str, ms: f64, c: &DeviceBuffer<f32>| -> Result<(), Error> {
        let gflops = 2.0 * (N * N * N) as f64 / ms / 1e6;
        let error = max_error(&a_host, &b_host, &c.to_host_vec(&stream)?);
        passed &= error < 1e-3;

        println!("  {name:<16} {ms:8.2} ms  {gflops:7.1} GFLOP/s  max error {error:.1e}");
        Ok(())
    };

    println!("C = A B, {N} x {N} floats");

    let ms = time_ms(&stream, 2, || module.naive(&stream, &naive, &a, &b, &mut c))?;
    report("naive", ms, &c)?;

    let ms = time_ms(&stream, 3, || module.tiled16(&stream, &t16, &a, &b, &mut c))?;
    report("tiled16, while", ms, &c)?;

    let ms = time_ms(&stream, 3, || module.tiled16_for(&stream, &t16_for, &a, &b, &mut c))?;
    report("tiled16, for", ms, &c)?;

    let ms = time_ms(&stream, 3, || module.tiled::<8>(&stream, &g8, &a, &b, &mut c))?;
    report("tiled::<8>", ms, &c)?;

    let ms = time_ms(&stream, 3, || module.tiled::<16>(&stream, &g16, &a, &b, &mut c))?;
    report("tiled::<16>", ms, &c)?;

    let ms = time_ms(&stream, 3, || module.tiled::<32>(&stream, &g32, &a, &b, &mut c))?;
    report("tiled::<32>", ms, &c)?;

    let ms = time_ms(&stream, 3, || module.tiled_fn::<16>(&stream, &f16, &a, &b, &mut c))?;
    report("tiled_fn::<16>", ms, &c)?;

    println!("{}", if passed { "PASSED" } else { "FAILED" });
    Ok(())
}
