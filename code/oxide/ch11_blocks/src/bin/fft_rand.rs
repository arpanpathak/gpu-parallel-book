//! Chapter 11 in Rust: random numbers with cuRAND, and the frequencies in a signal with
//! cuFFT, both called through their C interfaces.
//!
//!     cargo oxide run --bin fft_rand

use std::ffi::{c_int, c_void};
use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer};

type Error = Box<dyn std::error::Error>;

/// A cuRAND generator: a pointer to state that only cuRAND reads.
type Generator = *mut c_void;

const CURAND_RNG_PSEUDO_PHILOX4_32_10: c_int = 161;
const CUFFT_R2C: c_int = 0x2a;

#[link(name = "curand")]
unsafe extern "C" {
    fn curandCreateGenerator(generator: *mut Generator, kind: c_int) -> c_int;
    fn curandSetPseudoRandomGeneratorSeed(generator: Generator, seed: u64) -> c_int;
    fn curandSetStream(generator: Generator, stream: *mut c_void) -> c_int;
    fn curandGenerateNormal(
        generator: Generator,
        out: *mut f32,
        n: usize,
        mean: f32,
        sd: f32,
    ) -> c_int;
    fn curandDestroyGenerator(generator: Generator) -> c_int;
}

#[link(name = "cufft")]
unsafe extern "C" {
    fn cufftPlan1d(plan: *mut c_int, n: c_int, kind: c_int, batch: c_int) -> c_int;
    fn cufftSetStream(plan: c_int, stream: *mut c_void) -> c_int;
    fn cufftExecR2C(plan: c_int, input: *mut f32, output: *mut f32) -> c_int;
    fn cufftDestroy(plan: c_int) -> c_int;
}

/// cuRAND and cuFFT return 0 for success and an error number otherwise.
fn ok(result: c_int, call: &str) -> Result<(), Error> {
    if result == 0 {
        return Ok(());
    }

    Err(format!("{call} failed with error {result}").into())
}

/// The device address of a buffer, as the pointer type the C libraries take.
fn ptr(buffer: &DeviceBuffer<f32>) -> *mut f32 {
    buffer.cu_deviceptr() as *mut f32
}

/// 16 Mi numbers from the normal distribution; their mean should be near 0 and their
/// standard deviation near 1.
fn normal_numbers(stream: &Arc<CudaStream>) -> Result<bool, Error> {
    let n: usize = 16 << 20;
    let x = DeviceBuffer::<f32>::zeroed(stream, n)?;

    let mut generator: Generator = std::ptr::null_mut();
    let kind = CURAND_RNG_PSEUDO_PHILOX4_32_10;
    let out = ptr(&x);

    // SAFETY: `generator` is a valid place for cuRAND to store its generator; `out`
    // addresses n floats; the stream outlives the generator.
    let ms = unsafe {
        let result = curandCreateGenerator(&mut generator, kind);
        ok(result, "curandCreateGenerator")?;
        let result = curandSetPseudoRandomGeneratorSeed(generator, 2024);
        ok(result, "curandSetPseudoRandomGeneratorSeed")?;
        let result = curandSetStream(generator, stream.cu_stream().cast());
        ok(result, "curandSetStream")?;

        let result = curandGenerateNormal(generator, out, n, 0.0, 1.0); // warm-up
        ok(result, "curandGenerateNormal")?;
        stream.synchronize()?;

        let start = Instant::now();
        let result = curandGenerateNormal(generator, out, n, 0.0, 1.0);
        ok(result, "curandGenerateNormal")?;
        stream.synchronize()?;

        start.elapsed().as_secs_f64() * 1e3
    };

    // The 2 sums that the C++ program computes with Thrust, here on the host.
    let numbers = x.to_host_vec(stream)?;
    let sum: f64 = numbers.iter().map(|&v| v as f64).sum();
    let sum_of_squares: f64 = numbers.iter().map(|&v| v as f64 * v as f64).sum();

    let mean = sum / n as f64;
    let sd = (sum_of_squares / n as f64 - mean * mean).sqrt();
    let rate = n as f64 / ms / 1e3;

    println!("cuRAND, {n} normal numbers in {ms:.2} ms ({rate:.0} million per second)");
    println!("  mean {mean:.5}, standard deviation {sd:.5}");

    // SAFETY: the generator is live and is not used after this call.
    let result = unsafe { curandDestroyGenerator(generator) };
    ok(result, "curandDestroyGenerator")?;

    let mean_ok = mean.abs() < 1e-3;
    let sd_ok = (sd - 1.0).abs() < 1e-3;
    Ok(mean_ok && sd_ok)
}

/// 1 sample of sin(50 t) + 0.5 sin(120 t), at step t of n around the circle.
fn sample(t: usize, n: usize) -> f32 {
    let angle = 2.0 * std::f64::consts::PI * t as f64 / n as f64;
    let wave = (50.0 * angle).sin() + 0.5 * (120.0 * angle).sin();
    wave as f32
}

/// 1,024 samples of sin(50 t) + 0.5 sin(120 t), with t running once around the circle.
/// The transform should show 2 peaks, at frequencies 50 and 120, and nothing else.
fn spectrum(stream: &Arc<CudaStream>) -> Result<bool, Error> {
    let n = 1024;
    let bins = n / 2 + 1; // a real signal needs only half the spectrum
    let signal: Vec<f32> = (0..n).map(|t| sample(t, n)).collect();

    let x = DeviceBuffer::from_host(stream, &signal)?;
    let spectrum = DeviceBuffer::<f32>::zeroed(stream, 2 * bins)?; // (re, im) pairs

    let mut plan: c_int = 0;
    // SAFETY: `plan` is a valid place for the plan; `x` holds n floats and `spectrum`
    // holds `bins` complex numbers, the sizes of a real-to-complex transform of n.
    unsafe {
        let result = cufftPlan1d(&mut plan, n as c_int, CUFFT_R2C, 1);
        ok(result, "cufftPlan1d")?;
        let result = cufftSetStream(plan, stream.cu_stream().cast());
        ok(result, "cufftSetStream")?;
        let result = cufftExecR2C(plan, ptr(&x), ptr(&spectrum));
        ok(result, "cufftExecR2C")?;
    }
    let result = spectrum.to_host_vec(stream)?;

    println!("cuFFT, {n} samples of sin(50 t) + 0.5 sin(120 t)");
    let mut peaks = 0;
    let mut passed = true;

    for (f, pair) in result.chunks(2).enumerate() {
        let size = (pair[0] as f64).hypot(pair[1] as f64);
        if size <= 1.0 {
            continue;
        }

        println!("  frequency {f:3}: magnitude {size:.1}");
        peaks += 1;
        passed &= f == 50 || f == 120;
    }

    // SAFETY: the plan is live and is not used after this call.
    let result = unsafe { cufftDestroy(plan) };
    ok(result, "cufftDestroy")?;
    Ok(passed && peaks == 2)
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let mut passed = normal_numbers(&stream)?;
    passed &= spectrum(&stream)?;

    let verdict = if passed { "PASSED" } else { "FAILED" };
    println!("{verdict}");
    Ok(())
}
