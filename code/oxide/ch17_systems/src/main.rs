//! Chapter 17 in Rust: what the driver reports about the GPU, CPU access to pinned
//! memory, and a work queue with a doorbell between 2 threads.
//!
//!     cargo oxide run

use std::hint::{black_box, spin_loop};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::thread;
use std::time::Instant;

use cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_INTEGRATED as INTEGRATED;
use cuda_core::sys::{CUdevice, CUdevice_attribute, cuDeviceGetAttribute};
use cuda_core::{CudaContext, DriverError, IntoResult, PinnedHostBuffer};

/// Asks the driver for 1 integer attribute of `device`.
fn attribute(device: CUdevice, which: CUdevice_attribute) -> Result<i32, DriverError> {
    let mut value = 0;

    // SAFETY: `value` is a valid place for 1 integer, and `device` came from a live context.
    unsafe { cuDeviceGetAttribute(&mut value, which, device).result()? };
    Ok(value)
}

/// Sums every word, so that the reads cannot be skipped.
fn cpu_read(x: &[u32]) -> u32 {
    x.iter().fold(0u32, |s, &v| s.wrapping_add(v))
}

/// Best-of-5 CPU write and read bandwidth over `x`, in GB/s.
fn time_cpu(name: &str, x: &mut [u32]) {
    let bytes = (x.len() * size_of::<u32>()) as f64;
    let mut best_write = f64::MAX;
    let mut best_read = f64::MAX;
    let mut sum = 0;

    for _ in 0..5 {
        let t0 = Instant::now();
        x.fill(1);
        let t1 = Instant::now();
        sum = black_box(cpu_read(black_box(&*x)));

        best_write = best_write.min((t1 - t0).as_secs_f64());
        best_read = best_read.min(t1.elapsed().as_secs_f64());
    }

    let verdict = if sum as usize == x.len() { "PASSED" } else { "FAILED" };
    let (write, read) = (bytes / best_write / 1e9, bytes / best_read / 1e9);
    println!("{name:<24} write {write:6.2} GB/s   read {read:6.2} GB/s   {verdict}");
}

fn spin_us(us: f64) {
    let start = Instant::now();
    while start.elapsed().as_secs_f64() * 1e6 < us {
        spin_loop();
    }
}

const CAPACITY: u32 = 64;

/// Keeps 1 value alone on a 64-byte cache line, as each register has its own address.
#[repr(align(64))]
struct Line<T>(T);

/// A ring of entries (opcode in the high 32 bits, argument in the low 32) and 2 counters.
struct Channel {
    ring: [AtomicU64; CAPACITY as usize],
    doorbell: Line<AtomicU32>, // PUT: entries written so far
    get: Line<AtomicU32>,      // GET: entries the device has finished
    stop: Line<AtomicBool>,
}

/// The device: wait for the doorbell, run every entry up to PUT, then report GET.
fn device(ch: &Channel, work_us: f64) -> u64 {
    let mut get = 0u32;
    let mut sum = 0u64;

    while !ch.stop.0.load(Ordering::Acquire) {
        let put = ch.doorbell.0.load(Ordering::Acquire);

        while get != put {
            let entry = ch.ring[(get % CAPACITY) as usize].load(Ordering::Relaxed);
            spin_us(work_us);
            sum += if entry >> 32 == 1 { entry & 0xffff_ffff } else { 0 };

            get += 1;
            ch.get.0.store(get, Ordering::Release);
        }
    }

    sum
}

/// The host: write the entry, then ring the doorbell. Waits only when the ring is full.
fn submit(ch: &Channel, depth: u32, put: &mut u32, arg: u32) {
    while put.wrapping_sub(ch.get.0.load(Ordering::Acquire)) >= depth {
        spin_loop();
    }

    let entry = (1u64 << 32) | arg as u64;
    ch.ring[(*put % CAPACITY) as usize].store(entry, Ordering::Relaxed);
    ch.doorbell.0.store(*put + 1, Ordering::Release);

    *put += 1;
}

/// 200 bursts: submit 32 entries of 2 us each, then do 64 us of the host's own work.
fn run_queue(depth: u32) {
    let ch = Channel {
        ring: std::array::from_fn(|_| AtomicU64::new(0)),
        doorbell: Line(AtomicU32::new(0)),
        get: Line(AtomicU32::new(0)),
        stop: Line(AtomicBool::new(false)),
    };

    let (sum, expected, ms) = thread::scope(|s| {
        let dev = s.spawn(|| device(&ch, 2.0));

        let mut put = 0u32;
        let mut expected = 0u64;
        let start = Instant::now();

        for _ in 0..200 {
            for _ in 0..32 {
                let arg = put + 1;
                expected += arg as u64;
                submit(&ch, depth, &mut put, arg);
            }
            spin_us(64.0);
        }

        while ch.get.0.load(Ordering::Acquire) != put {
            spin_loop();
        }

        let ms = start.elapsed().as_secs_f64() * 1e3;
        ch.stop.0.store(true, Ordering::Release);
        (dev.join().unwrap_or(0), expected, ms)
    });

    let verdict = if sum == expected { "PASSED" } else { "FAILED" };
    println!("depth {depth:2}  {ms:7.2} ms   sum {sum} of {expected}   {verdict}");
}

fn main() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let device = ctx.cu_device();

    let (major, minor) = ctx.compute_capability()?;
    println!("device      {}, sm_{major}{minor}", ctx.device_name()?);
    println!("SMs         {}", ctx.multiprocessor_count()?);
    println!("integrated  {}\n", attribute(device, INTEGRATED)?);

    let words = 16 << 20; // 16,777,216 words, 64 MiB
    let mut pageable = vec![0u32; words];
    let mut pinned = PinnedHostBuffer::<u32>::zeroed(&ctx, words)?;

    println!("CPU access to 64 MiB through each mapping:");
    time_cpu("Vec (pageable)", &mut pageable);
    time_cpu("PinnedHostBuffer", pinned.as_mut_slice());

    println!("\nWork queue: 200 bursts of 32 entries, 2 us each, 64 us of host work:");
    for depth in [1, 4, 32, 64] {
        run_queue(depth);
    }

    Ok(())
}
