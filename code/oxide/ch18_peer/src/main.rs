//! Chapter 18 in Rust: which GPUs can reach each other's memory, and how fast bytes move.
//!
//!     cargo oxide run
//!
//! With 2 or more GPUs that can reach each other, the copies go from GPU 0 to GPU 1.
//! With 1 GPU, they stay inside GPU 0 and measure its own memory instead.

use std::sync::Arc;
use std::time::{Duration, Instant};

use cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT as TIMING;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, IntoResult};
use cuda_core::{LaunchConfig1D, peer};
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_contract, thread};

const N: usize = 32 << 20; // 33,554,432 floats, 128 MiB
const BYTES: usize = N * 4;
const REPS: u32 = 50;

type Error = Box<dyn std::error::Error>;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn copy(src: &[f32], mut dst: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out) = dst.get_mut(idx) {
            *out = src[i];
        }
    }
}

fn device_count() -> Result<usize, DriverError> {
    let mut count = 0;
    // SAFETY: `count` is a valid place for the driver to write 1 integer, and the driver
    // was initialised by `CudaContext::new` before this call.
    unsafe { cuda_core::sys::cuDeviceGetCount(&mut count).result()? };
    Ok(count as usize)
}

/// The peer matrix: can the GPU of each row read and write the memory of each column?
fn print_peer_matrix(gpus: &[Arc<CudaContext>]) -> Result<(), DriverError> {
    print!("peer access (row can reach column):\n      ");
    for j in 0..gpus.len() {
        print!("  GPU{j}");
    }
    println!();

    for (i, from) in gpus.iter().enumerate() {
        print!("  GPU{i}");

        for (j, to) in gpus.iter().enumerate() {
            let cell = if i == j {
                "X"
            } else if peer::can_access_peer(from, to)? {
                "yes"
            } else {
                "no"
            };
            print!("  {cell:>4}");
        }
        println!();
    }
    Ok(())
}

/// Average time of 1 run of `run`, in milliseconds, after 0.4 s of warm-up runs.
fn average_ms(
    stream: &CudaStream,
    mut run: impl FnMut() -> Result<(), Error>,
) -> Result<f32, Error> {
    let begin = Instant::now();
    while begin.elapsed() < Duration::from_millis(400) {
        run()?;
        stream.synchronize()?;
    }

    let start = stream.record_event(Some(TIMING))?;
    for _ in 0..REPS {
        run()?;
    }
    let stop = stream.record_event(Some(TIMING))?;
    stop.synchronize()?;

    Ok(start.elapsed_ms(&stop)? / REPS as f32)
}

fn report(how: &str, ms: f32, ok: bool) {
    let gb_per_s = BYTES as f64 / 1e9 / (ms as f64 / 1e3);
    let verdict = if ok { "PASSED" } else { "FAILED" };
    println!("  {how:<26} {ms:7.3} ms {gb_per_s:7.1} GB/s  {verdict}");
}

fn main() -> Result<(), Error> {
    let first = CudaContext::new(0)?;
    let count = device_count()?;
    println!("GPUs: {count}");

    let mut gpus = vec![first];
    for d in 1..count {
        gpus.push(CudaContext::new(d)?);
    }

    for (d, gpu) in gpus.iter().enumerate() {
        let name = gpu.device_name()?;
        let sms = gpu.multiprocessor_count()?;
        println!("  GPU{d}  {name}, {sms} SMs");
    }
    print_peer_matrix(&gpus)?;

    // Copy from GPU 0 to GPU 1 if GPU 0 can reach it, otherwise within GPU 0.
    let reachable = count > 1 && peer::can_access_peer(&gpus[0], &gpus[1])?;
    let (src, dst) = if reachable { (0, 1) } else { (0, 0) };

    if reachable {
        peer::enable_peer_access(&gpus[src], &gpus[dst])?;
    } else {
        println!("no peer to reach: copying within GPU0");
    }

    let host: Vec<f32> = (0..N).map(|i| (i % 1000) as f32).collect();
    let stream = gpus[src].new_stream()?;
    let dst_stream = gpus[dst].new_stream()?;

    let a = DeviceBuffer::from_host(&stream, &host)?;
    let mut b = DeviceBuffer::<f32>::zeroed(&dst_stream, N)?;

    gpus[src].bind_to_thread()?;
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&gpus[src])? };
    let launch = module.prepare_copy(LaunchConfig1D::new((N / 256) as u32, 256, 0))?;

    println!("{} MiB from GPU{src} to GPU{dst}, average of {REPS} copies", BYTES >> 20);

    let ms_peer = average_ms(&stream, || Ok(b.copy_from_device_async(&a, &stream)?))?;
    let ok_peer = b.to_host_vec(&dst_stream)? == host;
    report("copy_from_device_async", ms_peer, ok_peer);

    b = DeviceBuffer::<f32>::zeroed(&dst_stream, N)?;
    gpus[src].bind_to_thread()?;

    let ms_kernel = average_ms(&stream, || Ok(module.copy(&stream, &launch, &a, &mut b)?))?;
    let ok_kernel = b.to_host_vec(&dst_stream)? == host;
    report("copy kernel on GPU0", ms_kernel, ok_kernel);

    if !(ok_peer && ok_kernel) {
        return Err("a copy did not match its source".into());
    }
    Ok(())
}
