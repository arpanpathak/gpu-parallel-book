//! Chapter 10 in Rust: ownership frees device memory, generic kernels, 1 function, 2 sides.
//!
//!     cargo oxide run

use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::Mul;
use std::panic::Location;
use std::sync::Arc;
use std::time::Instant;

use cuda_core::sys;
use cuda_core::sys::CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_USED_MEM_CURRENT as USED_MEM;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy, DriverError, IntoResult};
use cuda_core::LaunchConfig1D;
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_contract, thread};

const BLOCK: u32 = 256;
const _: () = assert!(BLOCK % 32 == 0, "a block is a whole number of warps");
const _: () = assert!(BLOCK <= 1024, "a block holds at most 1,024 threads");

const fn blocks_for(n: usize) -> u32 {
    n.div_ceil(BLOCK as usize) as u32
}

const _: () = assert!(blocks_for(1000) == 4);

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

    /// out[i] = f(input[i]), for any T and any operation f from T to T, such as a closure.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn transform<T, F>(f: F, input: &[T], mut out: DisjointSlice<T>)
    where
        T: Copy,
        F: Fn(T) -> T + Copy,
    {
        let idx = thread::index_1d();
        let i = idx.get();

        if let Some(slot) = out.get_mut(idx) {
            *slot = f(input[i]);
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

    /// Count the values above a threshold. DeviceAtomicU32 adds atomically, device-wide.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn count_above(input: &[f32], threshold: f32, count: &[DeviceAtomicU32]) {
        let i = thread::index_1d().get();
        if i < input.len() && input[i] > threshold {
            count[0].fetch_add(1, AtomicOrdering::Relaxed);
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

/// Mistake 1 of raw.cu: ManuallyDrop turns the drops off, so the frees are written by
/// hand, and the early return skips them.
fn scale_by_hand(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    host: &[f32],
    factor: f32,
) -> Result<Option<Vec<f32>>, Error> {
    let mut input = ManuallyDrop::new(pool_copy(stream, host)?);
    let mut out = ManuallyDrop::new(pool_copy(stream, host)?);

    if factor == 0.0 {
        return Ok(None); // nothing to scale
    }

    let config = LaunchConfig1D::new(blocks_for(host.len()), BLOCK, 0);
    let prepared = module.prepare_scale::<f32>(config)?;
    module.scale(stream, &prepared, factor, &input, &mut out)?;
    let result = out.to_host_vec(stream)?;

    // SAFETY: neither buffer is used after its drop.
    unsafe {
        ManuallyDrop::drop(&mut input);
        ManuallyDrop::drop(&mut out);
    }
    Ok(Some(result))
}

/// Mistake 3 of raw.cu: the driver's copy counts bytes, but `host.len()` counts floats.
fn copy_count_as_bytes(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    host: &[f32],
) -> Result<usize, Error> {
    let input = DeviceBuffer::<f32>::zeroed(stream, host.len())?;
    let mut out = DeviceBuffer::<f32>::zeroed(stream, host.len())?;

    let src = host.as_ptr().cast::<c_void>();
    // SAFETY: both sides hold 4 bytes per float, so host.len() bytes fit in each.
    check(unsafe { sys::cuMemcpyHtoD_v2(input.cu_deviceptr(), src, host.len()) })?;

    let config = LaunchConfig1D::new(blocks_for(host.len()), BLOCK, 0);
    let prepared = module.prepare_scale::<f32>(config)?;
    module.scale(stream, &prepared, 2.0, &input, &mut out)?;

    let result = out.to_host_vec(stream)?;
    Ok(result.iter().filter(|&&x| x == 2.0).count())
}

/// Turn a driver result into an error that names the caller's file and line.
#[track_caller]
fn check(result: sys::CUresult) -> Result<(), Error> {
    let place = Location::caller();
    match result.result() {
        Ok(()) => Ok(()),
        Err(error) => Err(format!("{}:{}: {error}", place.file(), place.line()).into()),
    }
}

/// `count` values of T in device memory from the pool, owned by exactly 1 variable.
struct PoolBuffer<T> {
    ptr: sys::CUdeviceptr,
    count: usize,
    stream: Arc<CudaStream>,
    elements: PhantomData<T>,
}

impl<T: DeviceCopy> PoolBuffer<T> {
    #[track_caller]
    fn new(stream: &Arc<CudaStream>, count: usize) -> Result<Self, Error> {
        let mut ptr = 0;
        let bytes = count * size_of::<T>();
        // SAFETY: `ptr` is a valid out-pointer, and the stream belongs to this context.
        check(unsafe { sys::cuMemAllocAsync(&mut ptr, bytes, stream.cu_stream()) })?;

        let elements = PhantomData;
        Ok(Self { ptr, count, stream: stream.clone(), elements })
    }

    #[track_caller]
    fn from_host(stream: &Arc<CudaStream>, host: &[T]) -> Result<Self, Error> {
        let buffer = Self::new(stream, host.len())?;
        let src = host.as_ptr().cast::<c_void>();
        // SAFETY: the buffer and `host` both hold buffer.bytes() bytes.
        check(unsafe { sys::cuMemcpyHtoD_v2(buffer.ptr, src, buffer.bytes()) })?;
        Ok(buffer)
    }
}

/// A move: `b` takes the allocation, and the name `a` can no longer be used.
fn move_once(stream: &Arc<CudaStream>, host: &[f32]) -> Result<usize, Error> {
    let a = PoolBuffer::from_host(stream, host)?;
    let b = a;
    Ok(b.len())
}

impl<T> PoolBuffer<T> {
    fn len(&self) -> usize {
        self.count
    }

    fn bytes(&self) -> usize {
        self.count * size_of::<T>()
    }
}

impl<T> Drop for PoolBuffer<T> {
    fn drop(&mut self) {
        // A drop cannot report an error, so the result is ignored.
        // SAFETY: `ptr` came from cuMemAllocAsync, and only this drop frees it.
        let _ = unsafe { sys::cuMemFreeAsync(self.ptr, self.stream.cu_stream()) };
    }
}

/// The early return of raw.cu with PoolBuffer: both drops run on the return.
fn early_return_pool(
    stream: &Arc<CudaStream>,
    host: &[f32],
    factor: f32,
) -> Result<bool, Error> {
    let input = PoolBuffer::from_host(stream, host)?;
    let out = PoolBuffer::<f32>::new(stream, input.len())?;

    if factor == 0.0 {
        return Ok(false);
    }
    Ok(out.len() == input.len())
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

    let blocks = blocks_for(host.len());
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

    let blocks = blocks_for(data.len());
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
    let config = LaunchConfig1D::new(blocks_for(N), BLOCK, 0);

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

    println!("the mistakes of raw.cu, in Rust");
    let before = pool_mib(&ctx)?;
    for _ in 0..8 {
        scale_by_hand(&module, &stream, &host, 0.0)?;
    }
    let lost = pool_mib(&ctx)? - before;
    println!("  1. 8 calls with ManuallyDrop that return early: {lost:.1} MiB lost");

    let correct = copy_count_as_bytes(&module, &stream, &host)?;
    println!("  3. a copy of N bytes, not N floats: {correct} of {N} correct");

    println!("PoolBuffer, check and closures");
    let before = pool_mib(&ctx)?;
    for _ in 0..8 {
        early_return_pool(&stream, &host, 0.0)?;
    }
    let lost = pool_mib(&ctx)? - before;
    println!("  8 calls that return early: {lost:.1} MiB lost");
    passed &= lost == 0.0;

    let moved = move_once(&stream, &host)?;
    let in_use = pool_mib(&ctx)? - before;
    println!("  after a move: {moved} values, {in_use:.1} MiB in use");
    passed &= moved == N;

    if let Err(error) = PoolBuffer::<f32>::new(&stream, 1 << 45) {
        println!("  128 TiB returned: {error}");
    }

    let factor = 3.0f32;
    let triple = move |v: f32| factor * v;
    let input = DeviceBuffer::from_host(&stream, &floats)?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, BIG)?;

    let config = LaunchConfig1D::new(blocks_for(BIG), BLOCK, 0);
    let prepared = module.prepare_transform_for(&factor, &triple, config)?;
    module.transform(&stream, &prepared, triple, &input, &mut out)?;

    let result = out.to_host_vec(&stream)?;
    let correct = result.iter().zip(&floats).filter(|&(&y, &x)| y == triple(x)).count();
    println!("  transform with a closure: {correct} of {BIG} correct");
    passed &= correct == BIG;

    let count = DeviceBuffer::<u32>::zeroed(&stream, 1)?.cast_elem::<DeviceAtomicU32>();
    let config = LaunchConfig1D::new(blocks_for(N), BLOCK, 0);
    let prepared = module.prepare_count_above(config)?;
    module.count_above(&stream, &prepared, &y, 1.5, &count)?;

    let gpu_count = count.cast_elem::<u32>().to_host_vec(&stream)?[0] as usize;
    let cpu_count = gpu.iter().filter(|&&v| v > 1.5).count();
    println!("slice and DeviceAtomicU32: {gpu_count} values above 1.5, CPU {cpu_count}");
    passed &= gpu_count == cpu_count;

    if passed {
        println!("PASSED");
        Ok(())
    } else {
        Err("FAILED".into())
    }
}
