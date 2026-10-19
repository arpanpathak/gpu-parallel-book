//! Chapter 4 in Rust: copy speed from pageable and from pinned host memory,
//! and copy time against copy size.
//!
//!     cargo oxide run

use std::time::Instant;

use cuda_core::{CudaContext, DeviceBuffer, DriverError, PinnedHostBuffer};

const N: usize = 16 << 20; // 16,777,216 floats, 64 MiB

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    Ok(())
}
