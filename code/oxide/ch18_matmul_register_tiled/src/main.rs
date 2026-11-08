//! Chapter 18: register tiling, a TM x TN block of C per thread.
//!
//!     cargo oxide run

use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig2D};
use cuda_device::thread::Runtime2DIndex;
use cuda_device::{DisjointSlice, DynamicSharedArray, SharedArray};
use cuda_device::{LocalIndex32, RuntimeRowMajorTiles};
use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract, thread};
use cuda_host::RowWidth;

type Error = Box<dyn std::error::Error>;
type Module = kernels::LoadedModule;

#[cuda_module]
mod kernels {
    use super::*;

    /// The tiled kernel of Chapter 17 with T = 16, for comparison: a block of 16 x 16
    /// threads computes a 16 x 16 square of C, with 2 tiles in static shared memory.
    #[kernel]
    #[launch_contract(
        domain = 2,
        coordinates = u32,
        block = (16, 16, 1),
        requires = (a.len() >= m * k, b.len() >= k * n, c.len() >= m * n)
    )]
    pub fn matmul_tiled(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, Runtime2DIndex>,
    ) {
        static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();
        let (m, n, k) = (m as usize, n as usize, k as usize);

        let mut sum = 0.0;
        let mut t = 0;
        while t < k.div_ceil(16) {
            let (a_col, b_row) = (t * 16 + tx, t * 16 + ty);
            let a_value = if row < m && a_col < k { a[row * k + a_col] } else { 0.0 };
            let b_value = if b_row < k && col < n { b[b_row * n + col] } else { 0.0 };
            // SAFETY: the contract fixes the block at 16 x 16, so slot ty * 16 + tx is
            // below 256 and is this thread's alone.
            unsafe {
                TILE_A[ty * 16 + tx] = a_value;
                TILE_B[ty * 16 + tx] = b_value;
            }
            thread::sync_threads();

            let mut u = 0;
            #[unroll]
            while u < 16 {
                // SAFETY: between the 2 barriers no thread writes either tile.
                sum += unsafe { TILE_A[ty * 16 + u] * TILE_B[u * 16 + tx] };
                u += 1;
            }
            thread::sync_threads();

            t += 1;
        }

        if row < m
            && col < n
            && let Some(idx) = thread::index_2d_runtime(&c)
            && let Some(out) = c.get_mut(idx)
        {
            *out = sum;
        }
    }

    /// C (m x n) = A (m x k) B (k x n), all row-major. A block computes a BM x BN tile of
    /// C in phases of BK; thread (ty, tx) computes the TM x TN block of that tile that
    /// starts at row ty * TM and column tx * TN, and keeps its TM * TN sums in registers.
    #[kernel(launch_context = lc)]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 2,
        coordinates = u32,
        dynamic_shared_range = (64, 16384),
        requires = (a.len() >= m * k, b.len() >= k * n, c.len() >= m * n)
    )]
    pub fn matmul_register<
        const BM: usize,
        const BN: usize,
        const BK: usize,
        const TM: usize,
        const TN: usize,
    >(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, RuntimeRowMajorTiles<TM, TN>>,
    ) {
        let tile_a: *mut f32 = DynamicSharedArray::<f32>::get();
        let tile_b: *mut f32 = DynamicSharedArray::<f32>::offset(BM * (BK + 1) * 4);

        let threads = (BM / TM) * (BN / TN);
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let id = ty * (BN / TN) + tx;
        let row0 = thread::blockIdx_y() as usize * BM;
        let col0 = thread::blockIdx_x() as usize * BN;
        let (m, n, k) = (m as usize, n as usize, k as usize);

        let mut acc = [[0.0f32; TN]; TM];
        let mut t = 0;
        while t < k.div_ceil(BK) {
            let mut q = 0;
            #[unroll]
            while q < BM * BK / threads {
                let e = q * threads + id;
                let (r, s) = (row0 + e / BK, t * BK + e % BK);
                let value = if r < m && s < k { a[r * k + s] } else { 0.0 };
                // SAFETY: regtile_config launches `threads` threads, so slot e of tile A
                // has 1 writer; the slot lies inside the BM x (BK + 1) tile.
                unsafe { *tile_a.add(e / BK * (BK + 1) + e % BK) = value };
                q += 1;
            }

            let mut q = 0;
            #[unroll]
            while q < BK * BN / threads {
                let e = q * threads + id;
                let (s, col) = (t * BK + e / BN, col0 + e % BN);
                let value = if s < k && col < n { b[s * n + col] } else { 0.0 };
                // SAFETY: as for tile A, with BK x BN slots.
                unsafe { *tile_b.add(e) = value };
                q += 1;
            }
            thread::sync_threads();

            let mut u = 0;
            #[unroll]
            while u < BK {
                let mut a_reg = [0.0f32; TM];
                let mut b_reg = [0.0f32; TN];

                let mut i = 0;
                #[unroll]
                while i < TM {
                    // SAFETY: between the 2 barriers no thread writes either tile.
                    a_reg[i] = unsafe { *tile_a.add((ty * TM + i) * (BK + 1) + u) };
                    i += 1;
                }

                let mut j = 0;
                #[unroll]
                while j < TN {
                    // SAFETY: between the 2 barriers no thread writes either tile.
                    b_reg[j] = unsafe { *tile_b.add(u * BN + tx * TN + j) };
                    j += 1;
                }

                let mut i = 0;
                #[unroll]
                while i < TM {
                    let mut j = 0;
                    #[unroll]
                    while j < TN {
                        acc[i][j] += a_reg[i] * b_reg[j];
                        j += 1;
                    }
                    i += 1;
                }

                u += 1;
            }
            thread::sync_threads();

            t += 1;
        }

        if let Some(mut tile) = c.tile_2d32_rt(thread::coord_2d_u32(lc)) {
            let mut i = 0;
            #[unroll]
            while i < TM {
                let mut j = 0;
                #[unroll]
                while j < TN {
                    if let Some(r) = LocalIndex32::new(i as u32)
                        && let Some(q) = LocalIndex32::new(j as u32)
                    {
                        tile.at(r, q).write(acc[i][j]);
                    }
                    j += 1;
                }
                i += 1;
            }
        }
    }
}

/// The launch of matmul_register for an m x n matrix C: (BN / TN) x (BM / TM) threads
/// per block, enough blocks to cover C, and tiles of BM x (BK + 1) and BK x BN floats.
/// Every thread's TM x TN block must lie wholly inside C or wholly outside it.
fn regtile_config<
    const BM: usize,
    const BN: usize,
    const BK: usize,
    const TM: usize,
    const TN: usize,
>(
    m: usize,
    n: usize,
) -> Result<LaunchConfig2D, Error> {
    if m % TM != 0 || n % TN != 0 {
        let message = format!("{m} x {n} is not a multiple of the thread tile {TM} x {TN}");
        return Err(message.into());
    }

    let grid = ((n as u32).div_ceil(BN as u32), (m as u32).div_ceil(BM as u32));
    let block = ((BN / TN) as u32, (BM / TM) as u32);
    let shared = ((BM * (BK + 1) + BK * BN) * size_of::<f32>()) as u32;
    Ok(LaunchConfig2D::new(grid, block, shared))
}

/// Average time of `runs` launches in milliseconds, after 1 launch that is not timed.
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

/// Integers in [-5, 5] from a small linear congruential generator: every partial sum
/// of a product with k <= 4096 is an integer below 2^24, so f32 results are exact.
fn integers(len: usize, state: &mut u32) -> Vec<f32> {
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        values.push(((*state >> 8) % 11) as f32 - 5.0);
    }
    values
}

/// An m x k by k x n product: the inputs on the host and all 3 matrices on the device.
struct Problem {
    m: usize,
    n: usize,
    k: usize,
    a_host: Vec<f32>,
    b_host: Vec<f32>,
    a: DeviceBuffer<f32>,
    b: DeviceBuffer<f32>,
    c: DeviceBuffer<f32>,
}

impl Problem {
    fn new(stream: &Arc<CudaStream>, m: usize, n: usize, k: usize) -> Result<Self, Error> {
        let mut state = if m == n && n == k { 1 } else { 7 };
        let a_host = integers(m * k, &mut state);
        let b_host = integers(k * n, &mut state);
        let a = DeviceBuffer::from_host(stream, &a_host)?;
        let b = DeviceBuffer::from_host(stream, &b_host)?;
        let c = DeviceBuffer::<f32>::zeroed(stream, m * n)?;
        Ok(Problem { m, n, k, a_host, b_host, a, b, c })
    }

    /// Launch matmul_register::<BM, BN, BK, TM, TN> on this problem.
    fn run<
        const BM: usize,
        const BN: usize,
        const BK: usize,
        const TM: usize,
        const TN: usize,
    >(
        &mut self,
        module: &Module,
        stream: &Arc<CudaStream>,
    ) -> Result<(), Error> {
        let config = regtile_config::<BM, BN, BK, TM, TN>(self.m, self.n)?;
        let ready = module.prepare_matmul_register::<BM, BN, BK, TM, TN>(config)?;
        let (m, n, k) = (self.m as u32, self.n as u32, self.k as u32);
        let w = RowWidth::new(&mut self.c, n);
        let (a, b) = (&self.a, &self.b);
        module.matmul_register::<BM, BN, BK, TM, TN>(stream, &ready, m, n, k, a, b, w)?;
        Ok(())
    }

    /// Launch matmul_tiled, the kernel of Chapter 17, on this problem.
    fn tiled16(&mut self, module: &Module, stream: &Arc<CudaStream>) -> Result<(), Error> {
        let side = 16;
        let grid = ((self.n as u32).div_ceil(side), (self.m as u32).div_ceil(side));
        let config = LaunchConfig2D::new(grid, (side, side), 0);
        let ready = module.prepare_matmul_tiled(config)?;
        let (m, n, k) = (self.m as u32, self.n as u32, self.k as u32);
        let w = RowWidth::new(&mut self.c, n);
        module.matmul_tiled(stream, &ready, m, n, k, &self.a, &self.b, w)?;
        Ok(())
    }

    /// The number of `samples` entries of C (all of them if C is smaller) that are exact.
    fn exact_entries(&self, stream: &CudaStream, samples: usize) -> Result<usize, Error> {
        let (m, n, k) = (self.m, self.n, self.k);
        let c = self.c.to_host_vec(stream)?;
        let count = samples.min(m * n);

        let mut exact = 0;
        for s in 0..count {
            let i = if count == m * n { s / n } else { s * 977 % m };
            let j = if count == m * n { s % n } else { (s * 1231 + 7) % n };

            let mut sum = 0.0;
            for p in 0..k {
                sum += self.a_host[i * k + p] as f64 * self.b_host[p * n + j] as f64;
            }
            if c[i * n + j] as f64 == sum {
                exact += 1;
            }
        }
        Ok(exact)
    }
}

/// The 4 x 4 example: 1 block of 2 x 2 threads, each thread a 2 x 2 block of C.
fn run_example(module: &Module, stream: &Arc<CudaStream>) -> Result<bool, Error> {
    let a_int = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let b_int = [1, 0, 2, 1, 0, 1, 1, 2, 2, 1, 0, 1, 1, 2, 1, 0];
    let want = [11, 13, 8, 8, 27, 29, 24, 24, 43, 45, 40, 40, 59, 61, 56, 56];

    let mut p = Problem::new(stream, 4, 4, 4)?;
    p.a_host = a_int.iter().map(|&x| x as f32).collect();
    p.b_host = b_int.iter().map(|&x| x as f32).collect();
    p.a = DeviceBuffer::from_host(stream, &p.a_host)?;
    p.b = DeviceBuffer::from_host(stream, &p.b_host)?;

    p.run::<4, 4, 2, 2, 2>(module, stream)?;
    let c = p.c.to_host_vec(stream)?;

    print!("C = A B on the 4 x 4 example, 1 block of 2 x 2 threads,");
    println!(" 2 x 2 entries per thread");
    for row in c.chunks(4) {
        println!("  {:4} {:4} {:4} {:4}", row[0], row[1], row[2], row[3]);
    }
    let same = c.iter().zip(want).all(|(&got, w)| got == w as f32);
    let answer = if same { "yes" } else { "no" };
    println!("equal to the product computed by hand: {answer}\n");
    Ok(same)
}

/// A shape that is not a multiple of any block tile: every entry of C is checked.
fn run_shape(module: &Module, stream: &Arc<CudaStream>) -> Result<bool, Error> {
    let (m, n, k) = (1000, 1000, 1000);
    let mut p = Problem::new(stream, m, n, k)?;
    p.run::<64, 64, 16, 4, 4>(module, stream)?;

    let exact = p.exact_entries(stream, m * n)?;
    let all = m * n;
    print!("  m = {m:4}, k = {k:4}, n = {n:4}, 4 x 4 per thread:");
    println!(" {exact:7} of {all:7} entries exact\n");
    Ok(exact == all)
}

/// Time 1 kernel on n x n matrices and print 1 line of the table. The kernel computes
/// bm x bn tiles of C, tm x tn entries per thread.
fn run_timed<F>(
    stream: &Arc<CudaStream>,
    n: usize,
    name: &str,
    (bm, bn, tm, tn): (usize, usize, usize, usize),
    launch: F,
) -> Result<bool, Error>
where
    F: Fn(&mut Problem) -> Result<(), Error>,
{
    let mut p = Problem::new(stream, n, n, n)?;
    let cube = (n * n * n) as f64;
    let runs = (1e11 / (cube * 4.0)) as u32 + 2;
    let ms = time_ms(stream, runs.min(50), || launch(&mut p))?;
    let exact = p.exact_entries(stream, 64)?;

    let gflops = 2.0 * cube / ms / 1e6;
    let reads = (tm + tn) as f64 / (tm * tn) as f64;
    let fetched = cube * (1.0 / bm as f64 + 1.0 / bn as f64) * 4.0 / (1 << 20) as f64;

    print!("  {n:5}  {name:<9} {bm:3} x {bn:3} {ms:9.3} {gflops:9.1}");
    println!(" {reads:10.2} {fetched:12.1} {exact:6} of 64");
    Ok(exact == 64)
}

/// run_timed for matmul_register::<BM, BN, BK, TM, TN>.
fn run_config<
    const BM: usize,
    const BN: usize,
    const BK: usize,
    const TM: usize,
    const TN: usize,
>(
    module: &Module,
    stream: &Arc<CudaStream>,
    n: usize,
    name: &str,
) -> Result<bool, Error> {
    let shape = (BM, BN, TM, TN);
    run_timed(stream, n, name, shape, |p| p.run::<BM, BN, BK, TM, TN>(module, stream))
}

/// About 0.5 s of 1024 x 1024 products: the GPU raises its clocks under sustained load.
fn warm_up(module: &Module, stream: &Arc<CudaStream>) -> Result<(), Error> {
    let mut p = Problem::new(stream, 1024, 1024, 1024)?;
    let start = Instant::now();
    while start.elapsed().as_millis() < 500 {
        p.run::<64, 64, 16, 4, 4>(module, stream)?;
        stream.synchronize()?;
    }
    Ok(())
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };
    let (module, stream) = (&module, &stream);

    let mut passed = run_example(module, stream)?;
    passed &= run_shape(module, stream)?;
    warm_up(module, stream)?;

    println!("C = A B, n x n, 256 threads per block");
    print!("      n  thread        block        ms   GFLOP/s");
    println!("  reads/FMA  fetched MiB   exact");
    for n in [1024, 2048, 4096] {
        let tiled = |p: &mut Problem| p.tiled16(module, stream);
        passed &= run_timed(stream, n, "Ch. 17", (16, 16, 1, 1), tiled)?;
        passed &= run_config::<16, 16, 16, 1, 1>(module, stream, n, "1 x 1")?;
        passed &= run_config::<32, 16, 16, 2, 1>(module, stream, n, "2 x 1")?;
        passed &= run_config::<32, 32, 16, 2, 2>(module, stream, n, "2 x 2")?;
        passed &= run_config::<64, 64, 16, 4, 4>(module, stream, n, "4 x 4")?;
        passed &= run_config::<128, 128, 8, 8, 8>(module, stream, n, "8 x 8")?;
    }

    println!("{}", if passed { "PASSED" } else { "FAILED" });
    Ok(())
}
