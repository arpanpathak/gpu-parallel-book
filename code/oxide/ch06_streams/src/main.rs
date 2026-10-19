//! Chapter 6 in Rust: a launch timed with events, and a pipeline on 2 streams.
//!
//!     cargo oxide run

use std::sync::Arc;
use std::time::Instant;

use cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT as TIMING;
use cuda_core::{
    CudaContext, CudaStream, DeviceBuffer, DriverError, LaunchConfig1D, PinnedHostBuffer,
};
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_contract, thread};

const N: usize = 32 << 20; // 33,554,432 floats, 128 MiB
const STEPS: u32 = 256;

#[cuda_module]
mod kernels {
    use super::*;

    /// Some arithmetic per element, so that a kernel takes about as long as its copies.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn work(mut x: DisjointSlice<f32>, steps: u32) {
        if let Some(v) = x.get_mut(thread::index_1d()) {
            let mut a = *v;
            for _ in 0..steps {
                a = a * 0.999 + 0.5;
            }
            *v = a;
        }
    }
}

/// 1 chunk of the data: its pinned host buffer and its device buffer.
struct Chunk {
    host: PinnedHostBuffer<f32>,
    device: DeviceBuffer<f32>,
}

/// The data split into `count` chunks of 1.0s.
fn chunks(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    count: usize,
) -> Result<Vec<Chunk>, DriverError> {
    let len = N / count;

    (0..count)
        .map(|_| {
            let host = PinnedHostBuffer::from_slice(ctx, &vec![1.0f32; len])?;
            let device = DeviceBuffer::<f32>::zeroed(stream, len)?;
            Ok(Chunk { host, device })
        })
        .collect()
}

/// Copy in, compute and copy out every chunk, chunk c on stream c % streams.len().
/// Returns the average of 5 runs, in milliseconds.
fn pipeline(
    module: &kernels::LoadedModule,
    streams: &[Arc<CudaStream>],
    parts: &mut [Chunk],
) -> Result<f64, Box<dyn std::error::Error>> {
    let len = parts[0].host.len();
    let blocks = len.div_ceil(256) as u32;
    let launch = module.prepare_work(LaunchConfig1D::new(blocks, 256, 0))?;
    let start = Instant::now();

    for _ in 0..5 {
        for (c, part) in parts.iter_mut().enumerate() {
            let stream = &streams[c % streams.len()];

            // SAFETY: `part.host` is neither read, written nor dropped until every
            // stream is synchronized below, after both copies have completed.
            unsafe { part.device.copy_from_pinned_host_async(stream, &part.host)? };
            module.work(stream, &launch, &mut part.device, STEPS)?;
            unsafe { part.device.copy_to_pinned_host_async(stream, &mut part.host)? };
        }
    }

    for stream in streams {
        stream.synchronize()?;
    }

    Ok(start.elapsed().as_secs_f64() * 1e3 / 5.0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let streams = [ctx.new_stream()?, ctx.new_stream()?];
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };

    let mut whole = chunks(&ctx, &streams[0], 1)?;
    let mut eighths = chunks(&ctx, &streams[0], 8)?;
    pipeline(&module, &streams, &mut eighths)?; // warm up: the clocks rise under load

    // 1. A launch returns at once; events time the kernel on the GPU.
    let stream = &streams[0];
    let launch = module.prepare_work(LaunchConfig1D::new((N / 256) as u32, 256, 0))?;
    let device = &mut whole[0].device;

    let t0 = Instant::now();
    let start = stream.record_event(Some(TIMING))?;
    module.work(stream, &launch, device, STEPS)?;
    let stop = stream.record_event(Some(TIMING))?;
    let launched = t0.elapsed().as_secs_f64() * 1e3;

    stop.synchronize()?;
    let finished = t0.elapsed().as_secs_f64() * 1e3;

    println!("a launch of a kernel over 128 MiB");
    println!("  launch call returned after   {launched:8.3} ms");
    println!("  kernel finished after        {finished:8.3} ms");
    println!("  kernel time by events        {:8.3} ms", start.elapsed_ms(&stop)?);

    // 2. The same work in 1 chunk on 1 stream, and in 8 chunks on 2 streams.
    println!("copy in, kernel, copy out over 128 MiB");
    let one = pipeline(&module, &streams[..1], &mut whole)?;
    let eight = pipeline(&module, &streams, &mut eighths)?;
    println!("  1 stream, 1 chunk        {one:7.2} ms");
    println!("  2 streams, 8 chunks      {eight:7.2} ms");

    Ok(())
}
