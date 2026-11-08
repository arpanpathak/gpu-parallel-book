//! Chapter 10 in Rust: ownership frees device memory, generic kernels, 1 function, 2 sides.
//!
//!     cargo oxide run

use std::ffi::c_void;
use std::ops::Mul;
use std::sync::Arc;
use std::time::Instant;

use cuda_core::sys;
use cuda_core::sys::CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_USED_MEM_CURRENT as USED_MEM;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy, DriverError, IntoResult};
use cuda_core::LaunchConfig1D;
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_contract, thread};

const BLOCK: u32 = 256;
const _: () = assert!(BLOCK % 32 == 0, "a block is a whole number of warps");
const _: () = assert!(BLOCK <= 1024, "a block holds at most 1,024 threads");

const N: usize = 1 << 21;
const BIG: usize = 1 << 24;

type Error = Box<dyn std::error::Error>;

/// 1 + x + x^2/2 + x^3/6, the first 4 terms of e^x, by Horner's rule.
fn exp_poly(x: f32) -> f32 {
    1.0 + x * (1.0 + x * (0.5 + x * (1.0 / 6.0)))
}

/// The same polynomial, with every multiply-add rounded once.
fn exp_poly_fma(x: f32) -> f32 {
    let p = x.mul_add(1.0 / 6.0, 0.5);
    let p = x.mul_add(p, 1.0);
    x.mul_add(p, 1.0)
}

#[cuda_module]
mod kernels {
    use super::*;

    /// out[i] = factor * input[i], for any T that can be multiplied.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn scale<T>(factor: T, input: &[T], mut out: DisjointSlice<T>)
    where
        T: Copy + Mul<Output = T>,
    {
        let idx = thread::index_1d();
        let i = idx.get();

        if let Some(slot) = out.get_mut(idx) {
            *slot = factor * input[i];
        }
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn poly(x: &[f32], mut y: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();

        if let Some(slot) = y.get_mut(idx) {
            *slot = exp_poly(x[i]);
        }
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn poly_fma(x: &[f32], mut y: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();

        if let Some(slot) = y.get_mut(idx) {
            *slot = exp_poly_fma(x[i]);
        }
    }
}

/// Bytes in use from the pool that stream-ordered allocations draw on, in MiB.
fn pool_mib(ctx: &CudaContext) -> Result<f64, DriverError> {
    ctx.synchronize()?;

    let mut pool = std::ptr::null_mut();
    let mut used: u64 = 0;
    let used_ptr = (&mut used as *mut u64).cast::<c_void>();

    // SAFETY: both out-pointers are valid for the length of each call.
    unsafe {
        sys::cuDeviceGetDefaultMemPool(&mut pool, ctx.cu_device()).result()?;
        sys::cuMemPoolGetAttribute(pool, USED_MEM, used_ptr).result()?;
    }
    Ok(used as f64 / 1_048_576.0)
}

/// A copy of `host` in device memory from the pool, freed when the buffer is dropped.
fn pool_copy<T: DeviceCopy>(
    stream: &Arc<CudaStream>,
    host: &[T],
) -> Result<DeviceBuffer<T>, DriverError> {
    // SAFETY: the copy below writes every element before anything reads one.
    let mut buffer = unsafe { DeviceBuffer::uninitialized_async(stream, host.len())? };
    buffer.copy_from_host(stream, host)?;
    Ok(buffer)
}

/// The function of raw.cu. Every return, early or not, drops both buffers.
fn scale_on_gpu(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    host: &[f32],
    factor: f32,
) -> Result<Option<Vec<f32>>, Error> {
    let input = pool_copy(stream, host)?;
    let mut out = pool_copy(stream, host)?;

    if factor == 0.0 {
        return Ok(None);
    }

    let blocks = (host.len() as u32).div_ceil(BLOCK);
    let prepared = module.prepare_scale::<f32>(LaunchConfig1D::new(blocks, BLOCK, 0))?;
    module.scale(stream, &prepared, factor, &input, &mut out)?;
    Ok(Some(out.to_host_vec(stream)?))
}

/// The launch of 2,048 threads per block. The `?` returns before any kernel runs.
fn launch_too_wide(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    host: &[f32],
) -> Result<(), Error> {
    let input = pool_copy(stream, host)?;
    let mut out = pool_copy(stream, host)?;

    let blocks = (host.len() as u32) / 2048;
    let prepared = module.prepare_scale::<f32>(LaunchConfig1D::new(blocks, 2048, 0))?;
    module.scale(stream, &prepared, 2.0, &input, &mut out)?;
    Ok(())
}

/// The same function with `std::mem::forget`: the only way to skip a drop.
fn forget_on_gpu(stream: &Arc<CudaStream>, host: &[f32]) -> Result<(), DriverError> {
    let input = pool_copy(stream, host)?;
    std::mem::forget(input);
    Ok(())
}

/// Average time of `runs` runs of `launch`, in milliseconds, after 500 ms of warm-up:
/// the clocks of the GPU and of its memory rise under sustained load.
fn time_ms<F>(stream: &CudaStream, runs: u32, mut launch: F) -> Result<f64, Error>
where
    F: FnMut() -> Result<(), Error>,
{
    let start = Instant::now();
    while start.elapsed().as_millis() < 500 {
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

/// Time scale::<T> on `data`, print the rate, and check every result.
fn time_scale<T>(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    name: &str,
    data: &[T],
    factor: T,
) -> Result<bool, Error>
where
    T: DeviceCopy + Mul<Output = T> + PartialEq,
{
    let input = DeviceBuffer::from_host(stream, data)?;
    let mut out = DeviceBuffer::<T>::zeroed(stream, data.len())?;

    let blocks = (data.len() as u32).div_ceil(BLOCK);
    let prepared = module.prepare_scale::<T>(LaunchConfig1D::new(blocks, BLOCK, 0))?;
    let mut launch = || Ok(module.scale(stream, &prepared, factor, &input, &mut out)?);

    let ms = time_ms(stream, 20, &mut launch)?;
    let bytes = 2 * data.len() * size_of::<T>();
    println!("  {name:<14} {ms:7.2} ms  {:6.1} GB/s", bytes as f64 / ms / 1e6);

    let result = out.to_host_vec(stream)?;
    Ok(result.iter().zip(data).all(|(&y, &x)| y == factor * x))
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };

    let host = vec![1.0f32; N];
    let mut passed = true;

    println!("ownership");
    let before = pool_mib(&ctx)?;
    for _ in 0..8 {
        scale_on_gpu(&module, &stream, &host, 0.0)?;
    }
    let lost = pool_mib(&ctx)? - before;
    println!("  8 calls that return early: {lost:.1} MiB lost");
    passed &= lost == 0.0;

    match launch_too_wide(&module, &stream, &host) {
        Ok(()) => passed = false,
        Err(error) => println!("  2048 threads per block returned: {error}"),
    }
    println!("  after the error: {:.1} MiB in use", pool_mib(&ctx)?);

    let result = scale_on_gpu(&module, &stream, &host, 2.0)?.unwrap_or_default();
    let correct = result.iter().filter(|&&x| x == 2.0).count();
    println!("  copies sized by the type: {correct} of {N} correct");
    passed &= correct == N;

    for _ in 0..8 {
        forget_on_gpu(&stream, &host)?;
    }
    println!("  8 calls with mem::forget: {:.1} MiB lost", pool_mib(&ctx)? - before);

    println!("1 generic kernel, 3 types, {BIG} values");
    let floats: Vec<f32> = (0..BIG).map(|i| (i % 1000) as f32).collect();
    let doubles: Vec<f64> = (0..BIG).map(|i| (i % 1000) as f64).collect();
    let ints: Vec<i32> = (0..BIG).map(|i| (i % 1000) as i32).collect();

    passed &= time_scale(&module, &stream, "scale::<f32>", &floats, 3.0)?;
    passed &= time_scale(&module, &stream, "scale::<f64>", &doubles, 3.0)?;
    passed &= time_scale(&module, &stream, "scale::<i32>", &ints, 3)?;

    println!("1 function on both sides, {N} values in [-1, 1)");
    let xs: Vec<f32> = (0..N).map(|i| -1.0 + 2.0 * i as f32 / N as f32).collect();
    let x = DeviceBuffer::from_host(&stream, &xs)?;
    let mut y = DeviceBuffer::<f32>::zeroed(&stream, N)?;
    let config = LaunchConfig1D::new((N as u32).div_ceil(BLOCK), BLOCK, 0);

    let prepared = module.prepare_poly(config)?;
    module.poly(&stream, &prepared, &x, &mut y)?;
    let gpu = y.to_host_vec(&stream)?;
    let plain_differ = xs.iter().zip(&gpu).filter(|&(&x, &y)| exp_poly(x) != y).count();
    println!("  x * y + z: {plain_differ} of {N} differ from the CPU");

    let prepared = module.prepare_poly_fma(config)?;
    module.poly_fma(&stream, &prepared, &x, &mut y)?;
    let gpu = y.to_host_vec(&stream)?;
    let fused_differ = xs.iter().zip(&gpu).filter(|&(&x, &y)| exp_poly_fma(x) != y).count();
    println!("  mul_add:   {fused_differ} of {N} differ from the CPU");
    passed &= fused_differ == 0;

    if passed {
        println!("PASSED");
        Ok(())
    } else {
        Err("FAILED".into())
    }
}
