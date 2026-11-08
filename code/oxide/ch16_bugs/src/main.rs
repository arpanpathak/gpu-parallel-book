//! Chapter 16 in Rust: the bugs of bugs.cu, for Compute Sanitizer to find.
//!
//!     cargo oxide run -- race | barrier | uninit
//!     cargo oxide build --features no_guard     # the bounds bug: does not compile

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, launch_contract, thread};

type Error = Box<dyn std::error::Error>;

#[cuda_module]
mod kernels {
    use super::*;

    /// 1. Chapter 3's vector addition without its guard `if let Some(..)`.
    #[cfg(feature = "no_guard")]
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn add_no_guard(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let i = thread::index_1d();
        let k = i.get();
        *c.get_mut(i) = a[k] + b[k];
    }

    /// 2. Chapter 5's block sum with its barriers left out.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn sum_no_barrier(input: &[f32], mut out: DisjointSlice<f32>) {
        static mut S: SharedArray<f32, 256> = SharedArray::UNINIT;
        let t = thread::threadIdx_x() as usize;
        let block = thread::blockIdx_x() as usize;

        // SAFETY: none. No barrier orders these stores before the loads below: the bug.
        unsafe { S[t] = input[block * 256 + t] };

        let mut step: usize = 0;
        #[unroll]
        while step < 8 {
            let stride: usize = 128 >> step;
            if t < stride {
                // SAFETY: none, as above.
                unsafe { S[t] += S[t + stride] };
            }
            step += 1;
        }

        if t == 0 {
            // SAFETY: only thread 0 of each block writes, at its own block number.
            unsafe { *out.get_unchecked_mut(block) = S[0] };
        }
    }

    /// 3. Even and odd threads of each warp wait at 2 different barriers.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn barrier_in_branch(mut out: DisjointSlice<f32>) {
        static mut S: SharedArray<f32, 256> = SharedArray::UNINIT;
        let t = thread::threadIdx_x() as usize;

        // SAFETY: each thread writes only its own cell, before the barrier.
        unsafe { S[t] = t as f32 };

        if t % 2 == 0 {
            thread::sync_threads();
            if let Some(o) = out.get_mut(thread::index_1d()) {
                // SAFETY: after the barrier, no thread writes S.
                *o = unsafe { S[255 - t] };
            }
        } else {
            thread::sync_threads();
            if let Some(o) = out.get_mut(thread::index_1d()) {
                // SAFETY: as above.
                *o = unsafe { S[255 - t] };
            }
        }
    }

    /// 4. A sum over memory that nothing has written.
    #[kernel]
    #[launch_contract(domain = 1, block = (32, 1, 1))]
    pub fn sum_unwritten(input: &[f32], total: &[DeviceAtomicF32]) {
        let mut sum = 0.0;
        let mut i = thread::threadIdx_x() as usize;
        while i < input.len() {
            sum += input[i];
            i += 32;
        }
        total[0].fetch_add(sum, AtomicOrdering::Relaxed);
    }
}

fn race(stream: &Arc<CudaStream>, module: &kernels::LoadedModule) -> Result<(), Error> {
    let blocks = 64;
    let input = DeviceBuffer::from_host(stream, &vec![1.0f32; blocks * 256])?;
    let mut out = DeviceBuffer::<f32>::zeroed(stream, blocks)?;

    let launch = module.prepare_sum_no_barrier(LaunchConfig1D::new(blocks as u32, 256, 0))?;
    module.sum_no_barrier(stream, &launch, &input, &mut out)?;

    let sums = out.to_host_vec(stream)?;
    let wrong = sums.iter().filter(|&&sum| sum != 256.0).count();
    println!("{blocks} blocks of 256 ones: {wrong} wrong block sums");
    Ok(())
}

fn barrier(stream: &Arc<CudaStream>, module: &kernels::LoadedModule) -> Result<(), Error> {
    let mut out = DeviceBuffer::<f32>::zeroed(stream, 256)?;

    let launch = module.prepare_barrier_in_branch(LaunchConfig1D::new(1, 256, 0))?;
    module.barrier_in_branch(stream, &launch, &mut out)?;

    let values = out.to_host_vec(stream)?;
    let wrong = (0..256).filter(|&t| values[t] != (255 - t) as f32).count();
    println!("2 barriers, 1 per branch: {wrong} wrong elements of 256");
    Ok(())
}

fn uninit(stream: &Arc<CudaStream>, module: &kernels::LoadedModule) -> Result<(), Error> {
    let n = 1024;
    // SAFETY: none. Nothing writes `input` before the kernel reads it: the bug.
    let input = unsafe { DeviceBuffer::<f32>::uninitialized_async(stream, n)? };
    let total = DeviceBuffer::<f32>::zeroed(stream, 1)?.cast_elem::<DeviceAtomicF32>();

    let launch = module.prepare_sum_unwritten(LaunchConfig1D::new(1, 32, 0))?;
    module.sum_unwritten(stream, &launch, &input, &total)?;

    let sum = total.cast_elem::<f32>().to_host_vec(stream)?[0];
    println!("sum of 1,024 floats nobody wrote: {sum}");
    Ok(())
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };

    match std::env::args().nth(1).as_deref() {
        Some("race") => race(&stream, &module)?,
        Some("barrier") => barrier(&stream, &module)?,
        Some("uninit") => uninit(&stream, &module)?,
        _ => return Err("usage: cargo oxide run -- race | barrier | uninit".into()),
    }

    ctx.synchronize()?;
    Ok(())
}
