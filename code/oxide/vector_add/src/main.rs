//! Vector addition on the GPU with cuda-oxide: c[i] = a[i] + b[i].
//!
//! Build and run from this directory:
//!
//!     cargo oxide run

use std::time::Instant;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_bounds, launch_contract, thread};

/// Threads per block. The kernel's launch contract names the same number.
const BLOCK: u32 = 256;

/// The number of elements. It is not a multiple of `BLOCK`, so the last
/// block has threads with no element to add.
const N: usize = 1_000_003;

#[cuda_module]
mod kernels {
    use super::*;

    /// One thread adds one pair of elements.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn vector_add(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();

        if let Some(out) = c.get_mut(idx) {
            *out = a[i] + b[i];
        }
    }
}

/// The same computation on one CPU core, used as the reference answer.
fn vector_add_cpu(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter()
        .zip(b)
        .map(|(x, y)| x + y)
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Inputs on the host.
    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| 2.0 * i as f32).collect();

    // 2. A context on GPU 0, and the stream that orders our work on it.
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // 3. Copy the inputs to device memory and allocate the output.
    let a_dev = DeviceBuffer::from_host(&stream, &a)?;
    let b_dev = DeviceBuffer::from_host(&stream, &b)?;
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

    // 4. Load the GPU code that the compiler embedded in this binary.
    // SAFETY: the bundle was built from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };

    // 5. Enough blocks to cover N: ceil(N / BLOCK).
    let blocks = (N as u32).div_ceil(BLOCK);
    let prepared = module.prepare_vector_add(LaunchConfig1D::new(blocks, BLOCK, 0))?;

    // 6. Launch, then copy the result back. The copy waits for the kernel.
    let start = Instant::now();
    module.vector_add(&stream, &prepared, &a_dev, &b_dev, &mut c_dev)?;
    let c = c_dev.to_host_vec(&stream)?;
    let gpu_time = start.elapsed();

    // 7. Check every element against the CPU.
    let start = Instant::now();
    let expected = vector_add_cpu(&a, &b);
    let cpu_time = start.elapsed();

    let wrong = c
        .iter()
        .zip(&expected)
        .filter(|(got, want)| got != want)
        .count();

    println!("elements      {N}");
    println!("blocks        {blocks} x {BLOCK} threads = {} threads", blocks * BLOCK);
    println!("idle threads  {}", blocks as usize * BLOCK as usize - N);
    println!("c[0..4]       {:?}", &c[..4]);
    println!("c[N-1]        {}", c[N - 1]);
    println!("gpu + copy    {gpu_time:?}");
    println!("cpu           {cpu_time:?}");

    if wrong == 0 {
        println!("PASSED: all {N} elements match the CPU");
        Ok(())
    } else {
        Err(format!("FAILED: {wrong} elements differ from the CPU").into())
    }
}
