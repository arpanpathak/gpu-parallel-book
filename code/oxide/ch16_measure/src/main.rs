//! Chapter 16 in Rust: the GPU's limits, the warm-up effect, and a kernel report by hand.
//!
//!     cargo oxide run

use cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT as TIMING;
use cuda_core::sys::{CUdevice, CUdevice_attribute, cuDeviceGetAttribute};
use cuda_core::sys::{
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CLOCK_RATE as SM_CLOCK,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH as BUS_BITS,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE as L2_BYTES,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR as REGS,
    // shared memory per SM, in bytes
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR as SHM,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE as MEM_CLOCK,
};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, IntoResult};
use cuda_core::LaunchConfig1D;
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_contract, thread};

type Error = Box<dyn std::error::Error>;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn fill(mut x: DisjointSlice<f32>) {
        let i = thread::index_1d();
        let k = i.get();
        if let Some(v) = x.get_mut(i) {
            *v = (k % 1000) as f32;
        }
    }

    /// y = 2 x: reads 4 bytes and writes 4 bytes per element.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn scale(x: &[f32], mut y: DisjointSlice<f32>) {
        let i = thread::index_1d();
        let k = i.get();
        if let Some(v) = y.get_mut(i) {
            *v = 2.0 * x[k];
        }
    }
}

/// The time of 1 launch of `scale`, in milliseconds, measured with 2 events.
fn time_scale(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
    x: &DeviceBuffer<f32>,
    y: &mut DeviceBuffer<f32>,
) -> Result<f32, Error> {
    let blocks = x.len().div_ceil(256) as u32;
    let launch = module.prepare_scale(LaunchConfig1D::new(blocks, 256, 0))?;

    let start = stream.record_event(Some(TIMING))?;
    module.scale(stream, &launch, x, y)?;
    let stop = stream.record_event(Some(TIMING))?;
    stop.synchronize()?;

    Ok(start.elapsed_ms(&stop)?)
}

/// Asks the driver for 1 integer attribute of `device`.
fn attribute(device: CUdevice, which: CUdevice_attribute) -> Result<i32, DriverError> {
    let mut value = 0;

    // SAFETY: `value` is a valid place for 1 integer; `device` came from a live context.
    unsafe { cuDeviceGetAttribute(&mut value, which, device).result()? };
    Ok(value)
}

/// Print the limits the GPU reports, and the peaks they imply.
fn limits(ctx: &CudaContext) -> Result<(), DriverError> {
    let device = ctx.cu_device();
    let (major, minor) = ctx.compute_capability()?;
    let sms = ctx.multiprocessor_count()?;

    let sm_clock_khz = attribute(device, SM_CLOCK)?;
    let mem_clock_khz = attribute(device, MEM_CLOCK)?;
    let bus_bits = attribute(device, BUS_BITS)?;
    let shared_kib = attribute(device, SHM)? / 1024;

    println!("{}, compute capability {major}.{minor}", ctx.device_name()?);
    println!("  SMs                       {sms:8}");
    println!("  threads per SM, at most   {:8}", ctx.max_threads_per_multiprocessor()?);
    println!("  registers per SM          {:8}", attribute(device, REGS)?);
    println!("  shared memory per SM      {shared_kib:8} KiB");
    println!("  L2 cache                  {:8} KiB", attribute(device, L2_BYTES)? / 1024);
    println!("  SM clock                  {:8} MHz", sm_clock_khz / 1000);
    println!("  memory clock              {:8} MHz", mem_clock_khz / 1000);
    println!("  memory bus                {bus_bits:8} bits");

    // 128 FP32 units per SM from compute capability 8.6; 2 FLOP per multiply-add.
    let peak_gflops = sms as f64 * 128.0 * 2.0 * sm_clock_khz as f64 / 1e6;

    // 2 transfers per memory clock (double data rate) of bus_bits / 8 bytes each.
    let peak_gbs = 2.0 * mem_clock_khz as f64 * 1e3 * (bus_bits / 8) as f64 / 1e9;

    println!("  FP32 peak from the clock  {peak_gflops:8.0} GFLOP/s");
    println!("  bandwidth from the clock  {peak_gbs:8.1} GB/s");
    Ok(())
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };

    limits(&ctx)?;

    let n: usize = 64 << 20; // 67,108,864 floats, 256 MiB per array
    let bytes = 2.0 * (n * size_of::<f32>()) as f64;
    let mut x = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    let mut y = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    let fill = module.prepare_fill(LaunchConfig1D::new(n.div_ceil(256) as u32, 256, 0))?;
    module.fill(&stream, &fill, &mut x)?;
    stream.synchronize()?;

    // The first 40 runs of y = 2 x.
    let mut ms = Vec::with_capacity(40);
    for _ in 0..40 {
        ms.push(time_scale(&stream, &module, &x, &mut y)?);
    }

    println!("y = 2 x over 256 MiB, the first 40 runs in ms");
    for (r, t) in ms.iter().enumerate() {
        let end = if r % 10 == 9 { "\n" } else { "" };
        print!(" {t:5.2}{end}");
    }

    let mean = ms.iter().map(|&t| t as f64).sum::<f64>() / ms.len() as f64;

    let mut last = ms[20..].to_vec();
    last.sort_by(f32::total_cmp);
    let median = (last[9] + last[10]) / 2.0;

    println!("mean of all 40 runs     {mean:7.2} ms");
    println!("median of runs 21 to 40 {median:7.2} ms");

    // The report by hand: bytes moved / time.
    let gbs = bytes / (median as f64 * 1e-3) / 1e9;
    println!("bytes moved             {:7.0} MiB", bytes / (1 << 20) as f64);
    println!("achieved bandwidth      {gbs:7.1} GB/s");

    let scale = module.prepare_scale(LaunchConfig1D::new(1, 256, 0))?;
    let blocks = scale.function().max_active_blocks_per_multiprocessor(256, 0)?;
    println!("blocks of 256 per SM    {blocks:7} = {} threads", blocks * 256);

    let result = y.to_host_vec(&stream)?;
    let wrong = (0..n).filter(|&i| result[i] != 2.0 * (i % 1000) as f32).count();
    let verdict = if wrong == 0 { "PASSED" } else { "FAILED" };
    println!("{verdict}: {wrong} of {n} elements wrong");

    if wrong == 0 { Ok(()) } else { Err("wrong elements".into()) }
}
