//! Chapter 4 in Rust: copy speed from pageable and from pinned host memory, copy time
//! against copy size, unified memory, and zero-copy memory.
//!
//!     cargo oxide run

use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, IntoResult};
use cuda_core::{LaunchConfig1D, PinnedHostBuffer, sys};
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_contract, thread};

type Error = Box<dyn std::error::Error>;

const N: usize = 16 << 20; // 16,777,216 floats, 64 MiB

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn scale(mut x: DisjointSlice<f32>) {
        if let Some(v) = x.get_mut(thread::index_1d()) {
            *v = 2.0 * *v;
        }
    }
}

/// Runs `copy` once to warm up, then `reps` times; returns the average in milliseconds.
fn time_copy(
    reps: u32,
    mut copy: impl FnMut() -> Result<(), DriverError>,
) -> Result<f64, DriverError> {
    copy()?;

    let start = Instant::now();
    for _ in 0..reps {
        copy()?;
    }

    Ok(start.elapsed().as_secs_f64() * 1e3 / reps as f64)
}

fn gb_per_s(bytes: usize, ms: f64) -> f64 {
    bytes as f64 / 1e9 / (ms / 1e3)
}

/// Average time of `reps` launches of `scale` over `x`, in milliseconds.
fn time_scale(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    x: &mut DeviceBuffer<f32>,
    reps: u32,
) -> Result<f64, Error> {
    let blocks = x.len().div_ceil(256) as u32;
    let launch = module.prepare_scale(LaunchConfig1D::new(blocks, 256, 0))?;

    let start = Instant::now();
    for _ in 0..reps {
        module.scale(stream, &launch, x)?;
    }

    stream.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1e3 / reps as f64)
}

/// 3. Unified memory: 1 pointer for both sides. The host writes it first.
fn unified_memory(
    module: &kernels::LoadedModule,
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
) -> Result<(), Error> {
    let mut ptr: sys::CUdeviceptr = 0;
    let global = sys::CUmemAttach_flags_enum_CU_MEM_ATTACH_GLOBAL;
    // SAFETY: cuMemAllocManaged stores the address of N new floats in `ptr`.
    unsafe { sys::cuMemAllocManaged(&mut ptr, N * size_of::<f32>(), global) }.result()?;

    // SAFETY: `ptr` holds N floats; the buffer frees them with cuMemFree, as CUDA requires.
    let mut managed = unsafe { DeviceBuffer::<f32>::from_raw_parts(ptr, N, ctx.clone()) };

    // SAFETY: managed memory is valid on the host, and no kernel uses it yet.
    let host = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f32, N) };
    host.fill(1.0);

    println!("unified memory, kernel over {} MiB", managed.num_bytes() >> 20);
    let first = time_scale(module, stream, &mut managed, 1)?;
    let next = time_scale(module, stream, &mut managed, 10)?;

    println!("  1st launch, after the host wrote it  {first:6.2} ms");
    println!("  next 10 launches, average            {next:6.2} ms");

    // SAFETY: time_scale waited for the kernels, so the host may read the memory again.
    let value = unsafe { *(ptr as *const f32) };
    println!("  host reads managed[0] = {value}");

    Ok(())
}

/// 4. Zero-copy: pinned host memory that a kernel reads and writes in place.
fn zero_copy(
    module: &kernels::LoadedModule,
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    device: &mut DeviceBuffer<f32>,
) -> Result<(), Error> {
    let bytes = N * size_of::<f32>();
    let mut host: *mut c_void = std::ptr::null_mut();
    let mut ptr: sys::CUdeviceptr = 0;
    let mapped_flag = sys::CU_MEMHOSTALLOC_DEVICEMAP;

    // SAFETY: cuMemHostAlloc stores the address of `bytes` new pinned bytes in `host`.
    unsafe { sys::cuMemHostAlloc(&mut host, bytes, mapped_flag) }.result()?;
    // SAFETY: `host` is mapped memory; its address for kernels is stored in `ptr`.
    unsafe { sys::cuMemHostGetDevicePointer_v2(&mut ptr, host, 0) }.result()?;

    // SAFETY: `host` holds `bytes` bytes, and no kernel uses them yet.
    unsafe { std::ptr::write_bytes(host.cast::<u8>(), 0, bytes) };

    // SAFETY: `ptr` addresses N floats; into_raw_parts below stops the buffer freeing them.
    let mut mapped = unsafe { DeviceBuffer::<f32>::from_raw_parts(ptr, N, ctx.clone()) };

    time_scale(module, stream, device, 1)?; // warm up the kernel itself
    println!("kernel over {} MiB, average of 10 launches", bytes >> 20);

    let in_device = time_scale(module, stream, device, 10)?;
    println!("  in device memory      {in_device:6.2} ms");

    let in_mapped = time_scale(module, stream, &mut mapped, 10)?;
    println!("  in zero-copy memory   {in_mapped:6.2} ms");

    // Pinned host memory is freed with cuMemFreeHost, not with the buffer's cuMemFree.
    let _ = mapped.into_raw_parts();

    // SAFETY: time_scale waited for the kernels, so no kernel uses the memory any more.
    unsafe { sys::cuMemFreeHost(host) }.result()?;

    Ok(())
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let mut pageable = vec![0.0f32; N];
    let mut pinned = PinnedHostBuffer::<f32>::zeroed(&ctx, N)?;
    let mut device = DeviceBuffer::<f32>::zeroed(&stream, N)?;
    let bytes = N * size_of::<f32>();

    println!("copies of {} MiB", bytes >> 20);

    let ms = time_copy(20, || device.copy_from_host(&stream, &pageable))?;
    println!("  pageable host -> device  {ms:6.2} ms  {:5.1} GB/s", gb_per_s(bytes, ms));

    let ms = time_copy(20, || device.copy_from_host(&stream, pinned.as_slice()))?;
    println!("  pinned   host -> device  {ms:6.2} ms  {:5.1} GB/s", gb_per_s(bytes, ms));

    let ms = time_copy(20, || device.copy_to_host(&stream, &mut pageable))?;
    println!("  device -> pageable host  {ms:6.2} ms  {:5.1} GB/s", gb_per_s(bytes, ms));

    let ms = time_copy(20, || device.copy_to_pinned_host(&stream, &mut pinned))?;
    println!("  device -> pinned   host  {ms:6.2} ms  {:5.1} GB/s", gb_per_s(bytes, ms));

    println!("pinned host -> device, by size");

    let mut len = 1 << 10; // 1,024 floats, 4 KiB
    while len <= N {
        let mut part = DeviceBuffer::<f32>::zeroed(&stream, len)?;
        let src = &pinned.as_slice()[..len];

        let ms = time_copy(200, || part.copy_from_host(&stream, src))?;
        let size = len * size_of::<f32>();
        let (kib, us) = (size >> 10, ms * 1e3);
        println!("  {kib:8} KiB  {us:8.1} us  {:5.1} GB/s", gb_per_s(size, ms));

        len *= 4;
    }

    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };

    unified_memory(&module, &ctx, &stream)?;
    zero_copy(&module, &ctx, &stream, &mut device)?;

    Ok(())
}
