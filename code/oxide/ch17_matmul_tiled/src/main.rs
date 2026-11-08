//! Chapter 17: tiled matrix multiplication in shared memory, against the direct kernel.
//!
//!     cargo oxide run

use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig2D};
use cuda_device::thread::Runtime2DIndex;
use cuda_device::{DisjointSlice, DynamicSharedArray, cuda_module, kernel, thread};
use cuda_device::{launch_bounds, launch_contract};
use cuda_host::RowWidth;

type Error = Box<dyn std::error::Error>;
type Module = kernels::LoadedModule;

#[cuda_module]
mod kernels {
    use super::*;

    /// C (m x n) = A (m x k) B (k x n), all row-major. 1 thread computes 1 entry of C and
    /// fetches all its operands from global memory.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 2,
        coordinates = u32,
        requires = (a.len() >= m * k, b.len() >= k * n, c.len() >= m * n)
    )]
    pub fn matmul_direct(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, Runtime2DIndex>,
    ) {
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();
        let (m, n, k) = (m as usize, n as usize, k as usize);

        if row < m
            && col < n
            && let Some(idx) = thread::index_2d_runtime(&c)
            && let Some(out) = c.get_mut(idx)
        {
            let mut sum = 0.0;
            let mut s = 0;
            #[unroll(4)]
            while s < k {
                sum += a[row * k + s] * b[s * n + col];
                s += 1;
            }
            *out = sum;
        }
    }

    /// The same product with T x T tiles in shared memory. A block of T x T threads
    /// computes a T x T square of C in ceil(k / T) phases.
    #[kernel]
    #[launch_bounds(1024)]
    #[launch_contract(
        domain = 2,
        coordinates = u32,
        dynamic_shared_range = (32, 8192),
        requires = (a.len() >= m * k, b.len() >= k * n, c.len() >= m * n)
    )]
    pub fn matmul_tiled<const T: usize>(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, Runtime2DIndex>,
    ) {
        let tile_a: *mut f32 = DynamicSharedArray::<f32>::get();
        let tile_b: *mut f32 = DynamicSharedArray::<f32>::offset(T * T * 4);

        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();
        let (m, n, k) = (m as usize, n as usize, k as usize);

        let mut sum = 0.0;
        let mut t = 0;
        while t < k.div_ceil(T) {
            let a_col = t * T + tx;
            let b_row = t * T + ty;
            let a_value = if row < m && a_col < k { a[row * k + a_col] } else { 0.0 };
            let b_value = if b_row < k && col < n { b[b_row * n + col] } else { 0.0 };
            // SAFETY: tiled_config::<T> launches T x T threads with 2 T^2 floats
            // of shared memory, so slot ty * T + tx lies in each tile and belongs
            // to this thread alone; the barrier below completes these writes
            // before any thread reads them.
            unsafe {
                *tile_a.add(ty * T + tx) = a_value;
                *tile_b.add(ty * T + tx) = b_value;
            }
            thread::sync_threads();

            let mut u = 0;
            #[unroll]
            while u < T {
                // SAFETY: between the 2 barriers no thread writes either tile.
                sum += unsafe { *tile_a.add(ty * T + u) * *tile_b.add(u * T + tx) };
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
}

/// The launch of matmul_tiled::<T> for an m x n matrix C: T x T threads per block,
/// enough blocks to cover C, and 2 tiles of T x T floats in dynamic shared memory.
fn tiled_config<const T: usize>(m: usize, n: usize) -> LaunchConfig2D {
    let side = T as u32;
    let grid = ((n as u32).div_ceil(side), (m as u32).div_ceil(side));
    let shared = (2 * T * T * size_of::<f32>()) as u32;
    LaunchConfig2D::new(grid, (side, side), shared)
}

/// The launch of matmul_direct: blocks of 16 x 16 threads that cover C.
fn direct_config(m: usize, n: usize) -> LaunchConfig2D {
    let grid = ((n as u32).div_ceil(16), (m as u32).div_ceil(16));
    LaunchConfig2D::new(grid, (16, 16), 0)
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

    /// Launch matmul_direct on this problem.
    fn direct(&mut self, module: &Module, stream: &Arc<CudaStream>) -> Result<(), Error> {
        let prepared = module.prepare_matmul_direct(direct_config(self.m, self.n))?;
        let (m, n, k) = (self.m as u32, self.n as u32, self.k as u32);
        let width = RowWidth::new(&mut self.c, n);
        module.matmul_direct(stream, &prepared, m, n, k, &self.a, &self.b, width)?;
        Ok(())
    }

    /// Launch matmul_tiled::<T> on this problem.
    fn tiled<const T: usize>(
        &mut self,
        module: &Module,
        stream: &Arc<CudaStream>,
    ) -> Result<(), Error> {
        let prepared = module.prepare_matmul_tiled::<T>(tiled_config::<T>(self.m, self.n))?;
        let (m, n, k) = (self.m as u32, self.n as u32, self.k as u32);
        let width = RowWidth::new(&mut self.c, n);
        module.matmul_tiled::<T>(stream, &prepared, m, n, k, &self.a, &self.b, width)?;
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

/// The 4 x 4 example with T = 2: 4 blocks of 2 x 2 threads, 2 phases each.
fn run_example(module: &Module, stream: &Arc<CudaStream>) -> Result<bool, Error> {
    let a_int = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let b_int = [1, 0, 2, 1, 0, 1, 1, 2, 2, 1, 0, 1, 1, 2, 1, 0];
    let want = [11, 13, 8, 8, 27, 29, 24, 24, 43, 45, 40, 40, 59, 61, 56, 56];

    let mut p = Problem::new(stream, 4, 4, 4)?;
    p.a_host = a_int.iter().map(|&x| x as f32).collect();
    p.b_host = b_int.iter().map(|&x| x as f32).collect();
    p.a = DeviceBuffer::from_host(stream, &p.a_host)?;
    p.b = DeviceBuffer::from_host(stream, &p.b_host)?;

    p.tiled::<2>(module, stream)?;
    let c = p.c.to_host_vec(stream)?;

    println!("C = A B on the 4 x 4 example, tiles of 2 x 2, 2 phases");
    for row in c.chunks(4) {
        println!("  {:4} {:4} {:4} {:4}", row[0], row[1], row[2], row[3]);
    }
    let same = c.iter().zip(want).all(|(&got, w)| got == w as f32);
    let answer = if same { "yes" } else { "no" };
    println!("equal to the product computed by hand: {answer}\n");
    Ok(same)
}

/// Sizes that are not multiples of T: every entry of C is checked.
fn run_shape<const T: usize>(
    module: &Module,
    stream: &Arc<CudaStream>,
    (m, n, k): (usize, usize, usize),
) -> Result<bool, Error> {
    let mut p = Problem::new(stream, m, n, k)?;
    p.tiled::<T>(module, stream)?;

    let exact = p.exact_entries(stream, m * n)?;
    let all = m * n;
    print!("  m = {m:4}, k = {k:4}, n = {n:4}, T = {T:2}:");
    println!(" {exact:7} of {all:7} entries exact");
    Ok(exact == all)
}

/// Time 1 kernel on n x n matrices and print 1 line of the table. `tile` is 1 for the
/// direct kernel, which fetches every operand from global memory.
fn run_timed<F>(
    stream: &Arc<CudaStream>,
    name: &str,
    n: usize,
    tile: usize,
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
    let fetched = 2.0 * cube / tile as f64 * 4.0;
    let mib = fetched / (1 << 20) as f64;
    let rate = fetched / ms / 1e6;

    print!("  {n:5}  {name:<10} {ms:9.3} {gflops:9.1} {mib:12.1} {rate:11.1}");
    println!(" {exact:6} of 64");
    Ok(exact == 64)
}

/// About 0.5 s of 1024 x 1024 products: the GPU raises its clocks under sustained load.
fn warm_up(module: &Module, stream: &Arc<CudaStream>) -> Result<(), Error> {
    let mut p = Problem::new(stream, 1024, 1024, 1024)?;
    let start = Instant::now();
    while start.elapsed().as_millis() < 500 {
        p.tiled::<16>(module, stream)?;
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

    println!("Sizes that are not multiples of the tile width");
    passed &= run_shape::<2>(module, stream, (3, 3, 3))?;
    passed &= run_shape::<16>(module, stream, (128, 96, 64))?;
    passed &= run_shape::<16>(module, stream, (1000, 1000, 1000))?;
    println!();

    warm_up(module, stream)?;

    println!("C = A B, n x n; fetched: the modelled reads from global memory");
    print!("      n  kernel            ms   GFLOP/s");
    println!("  fetched MiB  fetch GB/s   exact");
    for n in [512, 1024, 2048, 4096] {
        passed &= run_timed(stream, "direct", n, 1, |p| p.direct(module, stream))?;
        passed &= run_timed(stream, "T = 8", n, 8, |p| p.tiled::<8>(module, stream))?;
        passed &= run_timed(stream, "T = 16", n, 16, |p| p.tiled::<16>(module, stream))?;
        passed &= run_timed(stream, "T = 32", n, 32, |p| p.tiled::<32>(module, stream))?;
    }

    println!("{}", if passed { "PASSED" } else { "FAILED" });
    Ok(())
}
