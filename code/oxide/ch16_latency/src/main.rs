//! Chapter 16 in Rust: latencies in cycles, measured inside a kernel with clock64().
//!
//!     cargo oxide run

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_contract, thread};
use cuda_device::{debug, float};

const STEPS: u32 = 4096;

type Error = Box<dyn std::error::Error>;

#[cuda_module]
mod kernels {
    use super::*;

    /// 1 thread, 1 chain: each multiply-add needs the result of the one before.
    #[kernel]
    #[launch_contract(domain = 1, block = (1, 1, 1))]
    pub fn dependent_fma(
        start: f32,
        mut out: DisjointSlice<f32>,
        mut cycles: DisjointSlice<u64>,
    ) {
        let mut v = start;

        let t0 = debug::clock64();
        let mut k = 0;
        #[unroll(16)]
        while k < STEPS {
            v = float::fma_rn_f32(v, 0.999, 1.0);
            k += 1;
        }
        let t1 = debug::clock64();

        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = v;
        }
        if let Some(c) = cycles.get_mut(thread::index_1d()) {
            *c = t1 - t0;
        }
    }

    /// 1 thread, 4 chains: the multiply-adds of 1 step do not wait for each other.
    #[kernel]
    #[launch_contract(domain = 1, block = (1, 1, 1))]
    pub fn independent_fma(
        start: f32,
        mut out: DisjointSlice<f32>,
        mut cycles: DisjointSlice<u64>,
    ) {
        let (mut v0, mut v1, mut v2, mut v3) = (start, start, start, start);

        let t0 = debug::clock64();
        let mut k = 0;
        #[unroll(4)]
        while k < STEPS / 4 {
            v0 = float::fma_rn_f32(v0, 0.999, 1.0);
            v1 = float::fma_rn_f32(v1, 0.999, 1.0);
            v2 = float::fma_rn_f32(v2, 0.999, 1.0);
            v3 = float::fma_rn_f32(v3, 0.999, 1.0);
            k += 1;
        }
        let t1 = debug::clock64();

        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = v0 + v1 + v2 + v3;
        }
        if let Some(c) = cycles.get_mut(thread::index_1d()) {
            *c = t1 - t0;
        }
    }

    /// 1 thread follows a chain of indices; the first `warm` hops are not timed.
    #[kernel]
    #[launch_contract(domain = 1, block = (1, 1, 1))]
    pub fn chase(
        next: &[u32],
        warm: u32,
        hops: u32,
        mut out: DisjointSlice<u32>,
        mut cycles: DisjointSlice<u64>,
    ) {
        let mut j = 0;
        let mut k = 0;
        while k < warm {
            // SAFETY: the host builds every entry of `next` below next.len().
            j = unsafe { *next.get_unchecked(j as usize) };
            k += 1;
        }

        let t0 = debug::clock64();
        k = 0;
        while k < hops {
            // SAFETY: as above.
            j = unsafe { *next.get_unchecked(j as usize) };
            k += 1;
        }
        let t1 = debug::clock64();

        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = j;
        }
        if let Some(c) = cycles.get_mut(thread::index_1d()) {
            *c = t1 - t0;
        }
    }
}

/// Cycles per load along a chain of `count` indices, each pointing `stride` ahead,
/// and whether the chain ended where the host expects.
fn chase_cycles(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
    count: u32,
    stride: u32,
    warm: u32,
    hops: u32,
) -> Result<(f64, bool), Error> {
    let chain: Vec<u32> = (0..count).map(|i| (i + stride) % count).collect();
    let next = DeviceBuffer::from_host(stream, &chain)?;
    let mut out = DeviceBuffer::<u32>::zeroed(stream, 1)?;
    let mut cycles = DeviceBuffer::<u64>::zeroed(stream, 1)?;

    let one = module.prepare_chase(LaunchConfig1D::new(1, 1, 0))?;
    module.chase(stream, &one, &next, warm, hops, &mut out, &mut cycles)?;

    let cycles = cycles.to_host_vec(stream)?[0] as f64 / hops as f64;
    let end = out.to_host_vec(stream)?[0] as u64;
    let expected = (warm + hops) as u64 * stride as u64 % count as u64;
    Ok((cycles, end == expected))
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };

    let mut out = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut cycles = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
    let one = LaunchConfig1D::new(1, 1, 0);

    let dep = module.prepare_dependent_fma(one)?;
    module.dependent_fma(&stream, &dep, 0.0, &mut out, &mut cycles)?;
    let dependent = cycles.to_host_vec(&stream)?[0] as f64 / STEPS as f64;

    let ind = module.prepare_independent_fma(one)?;
    module.independent_fma(&stream, &ind, 0.0, &mut out, &mut cycles)?;
    let independent = cycles.to_host_vec(&stream)?[0] as f64 / STEPS as f64;

    println!("{STEPS} multiply-adds in 1 thread, cycles per multiply-add");
    println!("  1 dependent chain      {dependent:6.2}");
    println!("  4 independent chains   {independent:6.2}");

    // 128-byte steps (32 indices), so that every load touches a new cache line.
    // The DRAM chain jumps 1 MiB + 128 bytes per load and is not warmed first.
    let (l1, ok1) = chase_cycles(&stream, &module, 1024, 32, 1024, 4096)?;
    let (l2, ok2) = chase_cycles(&stream, &module, 256 << 10, 32, 8192, 8192)?;
    let (dram, ok3) = chase_cycles(&stream, &module, 64 << 20, (1 << 18) + 32, 0, 4096)?;

    println!("1 thread follows a chain of loads, cycles per load");
    println!("  4 KiB chain, from L1      {l1:6.0}");
    println!("  1 MiB chain, from L2      {l2:6.0}");
    println!("  256 MiB chain, from DRAM  {dram:6.0}");

    let wrong = [ok1, ok2, ok3].iter().filter(|ok| !**ok).count();
    if wrong == 0 {
        println!("PASSED: every chain ends where the host expects");
        Ok(())
    } else {
        Err(format!("FAILED: {wrong} of 3 chains end in the wrong place").into())
    }
}
