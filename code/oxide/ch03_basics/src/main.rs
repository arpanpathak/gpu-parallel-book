//! Chapter 3 in Rust: every thread prints who it is, then vector addition.
//!
//!     cargo oxide run

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, cuda_module, gpu_printf, kernel, launch_contract, thread};

const N: usize = 1 << 20;
const BLOCK: u32 = 256;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_contract(domain = 1, block = (4, 1, 1))]
    pub fn who_am_i() {
        let block = thread::blockIdx_x();
        let position = thread::threadIdx_x();
        let index = block * thread::blockDim_x() + position;

        gpu_printf!("block {}, position {} -> index {}\n", block, position, index);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn add_vectors(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out) = c.get_mut(idx) {
            *out = a[i] + b[i];
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };

    // 2 blocks of 4 threads, each printing its block, position and index.
    let who = module.prepare_who_am_i(LaunchConfig1D::new(2, 4, 0))?;
    module.who_am_i(&stream, &who)?;
    stream.synchronize()?;

    // Vector addition: a[i] = i, b[i] = 2i, so c[i] should be 3i.
    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| 2.0 * i as f32).collect();

    let a_dev = DeviceBuffer::from_host(&stream, &a)?;
    let b_dev = DeviceBuffer::from_host(&stream, &b)?;
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

    let blocks = (N as u32).div_ceil(BLOCK);
    let add = module.prepare_add_vectors(LaunchConfig1D::new(blocks, BLOCK, 0))?;

    module.add_vectors(&stream, &add, &a_dev, &b_dev, &mut c_dev)?;
    let c = c_dev.to_host_vec(&stream)?;

    let max_err = (0..N).map(|i| (c[i] as f64 - 3.0 * i as f64).abs()).fold(0.0, f64::max);

    println!("n = {N}, {blocks} blocks of {BLOCK} threads");
    println!("c[0] = {}, c[1] = {}, c[{}] = {}", c[0], c[1], N - 1, c[N - 1]);
    println!("max error = {max_err}");

    Ok(())
}
