//! Chapter 22: histograms of 4-bit digits with atomics, a radix sort by 1-bit splits,
//! and a radix sort by 4-bit digits whose offsets come from a histogram and a scan.
//!
//!     cargo oxide run

use std::sync::Arc;
use std::time::{Duration, Instant};

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, BlockAtomicU32, DeviceAtomicU32};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, device, kernel, launch_contract};
use cuda_device::{thread, warp};

const TILE: usize = 1024; // keys per block in the scan and the sort
const N: usize = 16 << 20; // keys in the measured runs
const RADIX: usize = 16; // digit values of a 4-bit pass
const GRID: u32 = 1024; // blocks of the histogram kernels

type Error = Box<dyn std::error::Error>;
type Levels = Vec<(DeviceBuffer<u32>, DeviceBuffer<u32>)>;
type Bins = DeviceBuffer<DeviceAtomicU32>;
type Module = kernels::LoadedModule;

#[cuda_module]
mod kernels {
    use super::*;

    #[device]
    fn digit_of(key: u32, shift: u32) -> usize {
        ((key >> shift) as usize) & (RADIX - 1)
    }

    /// Bin `d` of a block's private bins in shared memory, viewed as an atomic.
    ///
    /// # Safety
    /// `local` must point to the block's bins and `d` must be below RADIX.
    #[device]
    unsafe fn shared_bin<'a>(local: *mut u32, d: usize) -> &'a BlockAtomicU32 {
        unsafe { BlockAtomicU32::from_ptr(local.add(d)) }
    }

    /// 1. Every count is an atomic addition on 1 of 16 bins in global memory.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn hist_global(keys: &[u32], shift: u32, bins: &[DeviceAtomicU32]) {
        let stride = (thread::gridDim_x() * 256) as usize;
        let mut index = thread::index_1d().get();
        while index < keys.len() {
            bins[digit_of(keys[index], shift)].fetch_add(1, AtomicOrdering::Relaxed);
            index += stride;
        }
    }

    /// 2. Each block counts into private bins in shared memory and merges them once.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn hist_shared(keys: &[u32], shift: u32, bins: &[DeviceAtomicU32]) {
        static mut LOCAL: SharedArray<u32, RADIX> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        if tid < RADIX {
            // SAFETY: threads 0 to 15 each zero their own bin, before the barrier.
            unsafe { LOCAL[tid] = 0 };
        }
        thread::sync_threads();

        // SAFETY: between the 2 barriers every access to LOCAL is atomic.
        let local = unsafe { (*&raw mut LOCAL).as_mut_ptr() };
        let stride = (thread::gridDim_x() * 256) as usize;
        let mut index = thread::index_1d().get();
        while index < keys.len() {
            let digit = digit_of(keys[index], shift);
            // SAFETY: d < RADIX, and LOCAL holds RADIX bins.
            unsafe { shared_bin(local, digit) }.fetch_add(1, AtomicOrdering::Relaxed);
            index += stride;
        }
        thread::sync_threads();

        if tid < RADIX {
            // SAFETY: after the barrier no thread writes LOCAL again.
            let count = unsafe { LOCAL[tid] };
            bins[tid].fetch_add(count, AtomicOrdering::Relaxed);
        }
    }

    /// 3. As 2, but the lanes of a warp with equal digits add their count with 1 atomic.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn hist_aggregated(keys: &[u32], shift: u32, bins: &[DeviceAtomicU32]) {
        static mut LOCAL: SharedArray<u32, RADIX> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        if tid < RADIX {
            // SAFETY: threads 0 to 15 each zero their own bin, before the barrier.
            unsafe { LOCAL[tid] = 0 };
        }
        thread::sync_threads();

        // SAFETY: between the 2 barriers every access to LOCAL is atomic.
        let local = unsafe { (*&raw mut LOCAL).as_mut_ptr() };
        let lane = warp::lane_id();
        let stride = (thread::gridDim_x() * 256) as usize;
        let mut base = thread::blockIdx_x() as usize * 256;
        while base < keys.len() {
            let index = base + tid;
            let inside = index < keys.len();
            let digit = if inside { digit_of(keys[index], shift) } else { RADIX };
            let peers = warp::match_any_sync(u32::MAX, digit as u32);
            let leader = peers.trailing_zeros();
            if lane == leader && digit < RADIX {
                // SAFETY: d < RADIX, and LOCAL holds RADIX bins.
                let bin = unsafe { shared_bin(local, digit) };
                bin.fetch_add(peers.count_ones(), AtomicOrdering::Relaxed);
            }
            base += stride;
        }
        thread::sync_threads();

        if tid < RADIX {
            // SAFETY: after the barrier no thread writes LOCAL again.
            let count = unsafe { LOCAL[tid] };
            bins[tid].fetch_add(count, AtomicOrdering::Relaxed);
        }
    }

    /// The inclusive scan of 1 value per lane across a warp.
    #[device]
    fn warp_scan(mut value: u32) -> u32 {
        let lane = warp::lane_id();
        let mut step = 0u32;
        #[unroll]
        while step < 5 {
            let offset = 1 << step; // 1, 2, 4, 8, 16
            let left = warp::shuffle_up_sync(u32::MAX, value, offset);
            if lane >= offset {
                value += left;
            }
            step += 1;
        }
        value
    }

    /// The scan of Chapter 21: 256 threads, 4 elements each, warp shuffles and 2 barriers.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn scan_warps(
        x: &[u32],
        mut y: DisjointSlice<u32>,
        mut sums: DisjointSlice<u32>,
    ) {
        static mut WARPS: SharedArray<u32, 8> = SharedArray::UNINIT;
        let first = thread::index_1d().get() * 4;
        let lane = warp::lane_id() as usize;
        let warp = thread::threadIdx_x() as usize / 32;

        let mut prefix = [0u32; 4];
        let mut own = 0;
        let mut element = 0;
        #[unroll]
        while element < 4 {
            if first + element < x.len() {
                own += x[first + element];
            }
            prefix[element] = own;
            element += 1;
        }

        let warp_prefix = warp_scan(own);
        if lane == 31 {
            // SAFETY: lane 31 of each warp writes its own slot, before the barrier.
            unsafe { WARPS[warp] = warp_prefix };
        }
        thread::sync_threads();

        if warp == 0 {
            // SAFETY: only warp 0 touches the slots between the 2 barriers.
            let warp_total = if lane < 8 { unsafe { WARPS[lane] } } else { 0 };
            let scanned_total = warp_scan(warp_total);
            if lane < 8 {
                unsafe { WARPS[lane] = scanned_total };
            }
        }
        thread::sync_threads();

        // SAFETY: after the second barrier the slots are only read.
        let earlier_warps = if warp > 0 { unsafe { WARPS[warp - 1] } } else { 0 };
        let before = warp_prefix - own + earlier_warps;
        let mut element = 0;
        #[unroll]
        while element < 4 {
            if first + element < y.len() {
                // SAFETY: thread t alone writes elements 4t to 4t + 3.
                unsafe { *y.get_unchecked_mut(first + element) = before + prefix[element] };
            }
            element += 1;
        }
        if thread::threadIdx_x() == 255 {
            let block = thread::blockIdx_x() as usize;
            // SAFETY: only the last thread of each block writes, at its own block number.
            unsafe { *sums.get_unchecked_mut(block) = before + prefix[3] };
        }
    }

    /// Adds to every element of tile b the total of tiles 0 to b - 1.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn add_offsets(mut y: DisjointSlice<u32>, scanned: &[u32]) {
        let block = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;

        if block > 0 {
            let offset = scanned[block - 1];
            let mut step = 0;
            #[unroll]
            while step < 4 {
                let index = block * TILE + step * 256 + tid;
                if index < y.len() {
                    // SAFETY: element g is this thread's alone and lies below y.len().
                    unsafe { *y.get_unchecked_mut(index) += offset };
                }
                step += 1;
            }
        }
    }

    /// The split on 1 bit: flags[i] = 1 if the bit of keys[i] is 0.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn zero_flags(keys: &[u32], bit: u32, mut flags: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let index = idx.get();

        if let Some(flag) = flags.get_mut(idx) {
            *flag = 1 - ((keys[index] >> bit) & 1);
        }
    }

    /// Zero-bit keys go to their exclusive scan, one-bit keys after all zero-bit keys.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn split_scatter(
        keys: &[u32],
        flags: &[u32],
        scanned: &[u32],
        mut out: DisjointSlice<u32>,
    ) {
        let index = thread::index_1d().get();

        if index < keys.len() {
            let zeros = scanned[keys.len() - 1] as usize;
            let upto = scanned[index] as usize; // zero-bit keys among 0 to i
            let at = if flags[index] == 1 { upto - 1 } else { zeros + index - upto };
            // SAFETY: zero-bit keys receive 0 to zeros - 1 and one-bit keys zeros to
            // n - 1, each position once: a permutation of 0 to n - 1.
            unsafe { *out.get_unchecked_mut(at) = keys[index] };
        }
    }

    /// The 4-bit pass, step 1: the digit histogram of every tile, stored digit by digit:
    /// counts[digit * tiles + tile] is the number of keys with that digit in that tile.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn digit_counts(keys: &[u32], shift: u32, mut counts: DisjointSlice<u32>) {
        static mut LOCAL: SharedArray<u32, RADIX> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let block = thread::blockIdx_x() as usize;
        if tid < RADIX {
            // SAFETY: threads 0 to 15 each zero their own bin, before the barrier.
            unsafe { LOCAL[tid] = 0 };
        }
        thread::sync_threads();

        // SAFETY: between the 2 barriers every access to LOCAL is atomic.
        let local = unsafe { (*&raw mut LOCAL).as_mut_ptr() };
        let mut step = 0;
        #[unroll]
        while step < 4 {
            let index = block * TILE + step * 256 + tid;
            if index < keys.len() {
                let digit = digit_of(keys[index], shift);
                // SAFETY: d < RADIX, and LOCAL holds RADIX bins.
                unsafe { shared_bin(local, digit) }.fetch_add(1, AtomicOrdering::Relaxed);
            }
            step += 1;
        }
        thread::sync_threads();

        if tid < RADIX {
            let tiles = thread::gridDim_x() as usize;
            // SAFETY: thread d of block b alone writes entry d * tiles + b, and the
            // table holds RADIX * tiles entries.
            unsafe { *counts.get_unchecked_mut(tid * tiles + block) = LOCAL[tid] };
        }
    }

    /// The 4-bit pass, step 3: every key goes to the start of its digit in its tile, plus
    /// the keys with the same digit before it in the tile. 4 steps of 256 keys, in order.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn scatter_digits(
        keys: &[u32],
        shift: u32,
        counts: &[u32],
        scanned: &[u32],
        mut out: DisjointSlice<u32>,
    ) {
        static mut START: SharedArray<u32, RADIX> = SharedArray::UNINIT;
        // The count of digit d in warp w, at slot w * RADIX + d.
        static mut WARP_COUNTS: SharedArray<u32, 128> = SharedArray::UNINIT;
        let thread_idx = thread::threadIdx_x() as usize;
        let block = thread::blockIdx_x() as usize;
        let lane = warp::lane_id();
        let warp = thread_idx / 32;
        let below = (1u32 << lane) - 1;

        if thread_idx < RADIX {
            let entry = thread_idx * thread::gridDim_x() as usize + block;
            // SAFETY: thread t alone writes START[t], before the first barrier.
            unsafe { START[thread_idx] = scanned[entry] - counts[entry] };
        }
        let mut step = 0;
        #[unroll]
        while step < 4 {
            if thread_idx < 8 * RADIX {
                // SAFETY: thread t alone writes slot t, between 2 barriers.
                unsafe { WARP_COUNTS[thread_idx] = 0 };
            }
            thread::sync_threads();

            let index = block * TILE + step * 256 + thread_idx;
            let key = if index < keys.len() { keys[index] } else { 0 };
            let digit = if index < keys.len() { digit_of(key, shift) } else { RADIX };
            let peers = warp::match_any_sync(u32::MAX, digit as u32);
            let rank = (peers & below).count_ones();
            if rank == 0 && digit < RADIX {
                // SAFETY: 1 lane per digit and warp writes slot warp * 16 + d.
                unsafe { WARP_COUNTS[warp * RADIX + digit] = peers.count_ones() };
            }
            thread::sync_threads();

            if digit < RADIX {
                let mut before = 0;
                let mut other = 0;
                while other < warp {
                    // SAFETY: between the 2 barriers the slots are only read.
                    before += unsafe { WARP_COUNTS[other * RADIX + digit] };
                    other += 1;
                }
                // SAFETY: START is only read between these barriers.
                let at = (unsafe { START[digit] } + before + rank) as usize;
                // SAFETY: the keys of each digit receive distinct consecutive positions
                // after the keys of the smaller digits and of the earlier tiles: the
                // positions of all keys form a permutation of 0 to n - 1.
                unsafe { *out.get_unchecked_mut(at) = key };
            }
            thread::sync_threads();

            if thread_idx < RADIX {
                let mut other = 0;
                #[unroll]
                while other < 8 {
                    // SAFETY: thread t alone updates START[t]; the slots are only read.
                    unsafe { START[thread_idx] += WARP_COUNTS[other * RADIX + thread_idx] };
                    other += 1;
                }
            }
            thread::sync_threads();
            step += 1;
        }
    }
}

/// Runs `launch` for 0.4 s so that the clocks rise, then averages 20 runs, in ms.
fn time_ms(
    stream: &Arc<CudaStream>,
    mut launch: impl FnMut() -> Result<(), Error>,
) -> Result<f64, Error> {
    let warm = Instant::now();
    while warm.elapsed() < Duration::from_millis(400) {
        launch()?;
        stream.synchronize()?;
    }

    let start = Instant::now();
    for _ in 0..20 {
        launch()?;
    }
    stream.synchronize()?;

    Ok(start.elapsed().as_secs_f64() * 1e3 / 20.0)
}

/// Numbers from a small linear congruential generator.
fn random(n: usize, seed: u32) -> Vec<u32> {
    let mut state = seed;

    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        })
        .collect()
}

/// Buffers for the tile totals and their scan, per level: n / 1,024, / 1,024 again, ... 1.
fn scan_levels(stream: &Arc<CudaStream>, mut n: usize) -> Result<Levels, Error> {
    let mut levels = Vec::new();
    while n > 1 {
        n = n.div_ceil(TILE);
        let totals = DeviceBuffer::<u32>::zeroed(stream, n)?;
        let scanned = DeviceBuffer::<u32>::zeroed(stream, n)?;
        levels.push((totals, scanned));
    }
    Ok(levels)
}

/// Inclusive scan: the tiles, then the tile totals (by the same function), then offsets.
fn scan(
    module: &Module,
    stream: &Arc<CudaStream>,
    x: &DeviceBuffer<u32>,
    y: &mut DeviceBuffer<u32>,
    levels: &mut [(DeviceBuffer<u32>, DeviceBuffer<u32>)],
) -> Result<(), Error> {
    let tiles = x.len().div_ceil(TILE) as u32;
    let config = LaunchConfig1D::new(tiles, 256, 0);
    let ((totals, scanned), rest) = levels.split_first_mut().ok_or("no level left")?;
    module.scan_warps(stream, &module.prepare_scan_warps(config)?, x, y, totals)?;

    if tiles > 1 {
        scan(module, stream, totals, scanned, rest)?;
        module.add_offsets(stream, &module.prepare_add_offsets(config)?, y, scanned)?;
    }
    Ok(())
}

/// The buffers that a radix sort of n keys needs besides the keys themselves.
struct SortBuffers {
    spare: DeviceBuffer<u32>,     // the other half of each pass
    flags: DeviceBuffer<u32>,     // 1 per key
    scanned: DeviceBuffer<u32>,   // 1 per key
    levels: Levels,               // the scan of the flags
    counts: DeviceBuffer<u32>,    // RADIX per tile
    positions: DeviceBuffer<u32>, // the scan of the counts
    count_levels: Levels,
}

impl SortBuffers {
    fn new(stream: &Arc<CudaStream>, n: usize) -> Result<Self, Error> {
        let table = RADIX * n.div_ceil(TILE);
        Ok(SortBuffers {
            spare: DeviceBuffer::<u32>::zeroed(stream, n)?,
            flags: DeviceBuffer::<u32>::zeroed(stream, n)?,
            scanned: DeviceBuffer::<u32>::zeroed(stream, n)?,
            levels: scan_levels(stream, n)?,
            counts: DeviceBuffer::<u32>::zeroed(stream, table)?,
            positions: DeviceBuffer::<u32>::zeroed(stream, table)?,
            count_levels: scan_levels(stream, table)?,
        })
    }
}

/// 1 stable split on `bit`, from `keys` into `buffers.spare`.
fn split(
    module: &Module,
    stream: &Arc<CudaStream>,
    keys: &DeviceBuffer<u32>,
    bit: u32,
    buffers: &mut SortBuffers,
) -> Result<(), Error> {
    let every = LaunchConfig1D::new(keys.len().div_ceil(256) as u32, 256, 0);
    let flag = module.prepare_zero_flags(every)?;
    module.zero_flags(stream, &flag, keys, bit, &mut buffers.flags)?;
    scan(module, stream, &buffers.flags, &mut buffers.scanned, &mut buffers.levels)?;

    let scatter = module.prepare_split_scatter(every)?;
    let (flags, scanned, out) = (&buffers.flags, &buffers.scanned, &mut buffers.spare);
    module.split_scatter(stream, &scatter, keys, flags, scanned, out)?;
    Ok(())
}

/// 1 pass on the 4-bit digit at `shift`, from `keys` into `buffers.spare`.
fn digit_pass(
    module: &Module,
    stream: &Arc<CudaStream>,
    keys: &DeviceBuffer<u32>,
    shift: u32,
    buffers: &mut SortBuffers,
) -> Result<(), Error> {
    let tiles = LaunchConfig1D::new(keys.len().div_ceil(TILE) as u32, 256, 0);
    let count = module.prepare_digit_counts(tiles)?;
    module.digit_counts(stream, &count, keys, shift, &mut buffers.counts)?;
    let levels = &mut buffers.count_levels;
    scan(module, stream, &buffers.counts, &mut buffers.positions, levels)?;
    let scatter = module.prepare_scatter_digits(tiles)?;
    let (counts, positions) = (&buffers.counts, &buffers.positions);
    let out = &mut buffers.spare;
    module.scatter_digits(stream, &scatter, keys, shift, counts, positions, out)?;
    Ok(())
}

/// 32 stable splits of 1 bit, bit 0 first. The result ends in `keys`.
fn sort_bits(
    module: &Module,
    stream: &Arc<CudaStream>,
    keys: &mut DeviceBuffer<u32>,
    buffers: &mut SortBuffers,
) -> Result<(), Error> {
    for bit in 0..32 {
        split(module, stream, keys, bit, buffers)?;
        std::mem::swap(keys, &mut buffers.spare);
    }
    Ok(())
}

/// 8 passes of 4 bits, digit 0 first. The result ends in `keys`.
fn sort_digits(
    module: &Module,
    stream: &Arc<CudaStream>,
    keys: &mut DeviceBuffer<u32>,
    buffers: &mut SortBuffers,
) -> Result<(), Error> {
    for shift in (0..32).step_by(4) {
        digit_pass(module, stream, keys, shift, buffers)?;
        std::mem::swap(keys, &mut buffers.spare);
    }
    Ok(())
}

fn print(name: &str, values: &[u32]) {
    let cells: Vec<String> = values.iter().map(|v| v.to_string()).collect();
    println!("  {name:13} {}", cells.join(" "));
}

/// The worked example: 8 keys of 3 bits, 3 splits, then 1 pass on a 4-bit digit.
fn small_example(module: &Module, stream: &Arc<CudaStream>) -> Result<bool, Error> {
    let host = [5u32, 2, 7, 2, 0, 6, 3, 5];
    let sorted = [0u32, 2, 2, 3, 5, 5, 6, 7];
    let mut buffers = SortBuffers::new(stream, 8)?;
    let mut keys = DeviceBuffer::from_host(stream, &host)?;
    println!("radix sort of 5 2 7 2 0 6 3 5");

    for bit in 0..3 {
        split(module, stream, &keys, bit, &mut buffers)?;
        std::mem::swap(&mut keys, &mut buffers.spare);
        print(&format!("split bit {bit}"), &keys.to_host_vec(stream)?);
    }
    let by_splits = keys.to_host_vec(stream)?;

    let keys = DeviceBuffer::from_host(stream, &host)?;
    let bins = DeviceBuffer::<u32>::zeroed(stream, RADIX)?.cast_elem::<DeviceAtomicU32>();
    let one = module.prepare_hist_shared(LaunchConfig1D::new(1, 256, 0))?;
    module.hist_shared(stream, &one, &keys, 0, &bins)?;
    let hist = bins.cast_elem::<u32>().to_host_vec(stream)?[..8].to_vec();
    print("histogram", &hist);

    digit_pass(module, stream, &keys, 0, &mut buffers)?;
    let inclusive = buffers.positions.to_host_vec(stream)?;
    let starts: Vec<u32> = (0..8).map(|d| inclusive[d] - hist[d]).collect();
    print("digit starts", &starts);
    let by_digits = buffers.spare.to_host_vec(stream)?;
    print("4-bit pass", &by_digits);

    Ok(by_splits == sorted && by_digits == sorted)
}

/// The 3 histograms of the low 4 bits of N keys, on random keys and on keys all equal.
fn histograms(module: &Module, stream: &Arc<CudaStream>) -> Result<bool, Error> {
    let grid = LaunchConfig1D::new(GRID, 256, 0);
    let global = module.prepare_hist_global(grid)?;
    let shared = module.prepare_hist_shared(grid)?;
    let aggregated = module.prepare_hist_aggregated(grid)?;
    let gb = 4.0 * N as f64 / 1e9;

    let mut ok = true;
    println!("histograms of the low 4 bits of {N} keys, average of 20 runs");
    print!("  keys        global ms   GB/s   shared ms   GB/s");
    println!("   warp-aggregated ms   GB/s");
    for (name, host) in [("random", random(N, 7)), ("all 7", vec![7u32; N])] {
        let keys = DeviceBuffer::from_host(stream, &host)?;
        let mut expected = vec![0u32; RADIX];
        for k in &host {
            expected[(k & 15) as usize] += 1;
        }

        let run = |kernel: usize, bins: &Bins| -> Result<(), Error> {
            match kernel {
                0 => module.hist_global(stream, &global, &keys, 0, bins)?,
                1 => module.hist_shared(stream, &shared, &keys, 0, bins)?,
                _ => module.hist_aggregated(stream, &aggregated, &keys, 0, bins)?,
            }
            Ok(())
        };

        let mut ms = [0.0; 3];
        for (kernel, time) in ms.iter_mut().enumerate() {
            let scratch = DeviceBuffer::<u32>::zeroed(stream, RADIX)?.cast_elem();
            *time = time_ms(stream, || run(kernel, &scratch))?;

            let bins = DeviceBuffer::<u32>::zeroed(stream, RADIX)?.cast_elem();
            run(kernel, &bins)?;
            ok &= bins.cast_elem::<u32>().to_host_vec(stream)? == expected;
        }

        let rate = |t: f64| gb / (t / 1e3);
        let (a, b, c) = (ms[0], ms[1], ms[2]);
        print!("  {name:10} {a:10.2} {:6.1} {b:11.2} {:6.1}", rate(a), rate(b));
        println!(" {c:20.2} {:6.1}", rate(c));
    }
    Ok(ok)
}

/// Both radix sorts of N random keys, best of 3 runs each, checked against sort_unstable.
fn sorts(module: &Module, stream: &Arc<CudaStream>) -> Result<bool, Error> {
    let host = random(N, 11);
    let mut expected = host.clone();
    expected.sort_unstable();

    let mut buffers = SortBuffers::new(stream, N)?;
    let mut ok = true;
    let mut best = [f64::MAX; 2];
    for (method, fastest) in best.iter_mut().enumerate() {
        for run in 0..4 {
            let mut keys = DeviceBuffer::from_host(stream, &host)?;
            stream.synchronize()?;
            let start = Instant::now();
            if method == 0 {
                sort_bits(module, stream, &mut keys, &mut buffers)?;
            } else {
                sort_digits(module, stream, &mut keys, &mut buffers)?;
            }
            stream.synchronize()?;
            let ms = start.elapsed().as_secs_f64() * 1e3;
            if run > 0 {
                *fastest = fastest.min(ms); // run 0 warms up
            }
            ok &= keys.to_host_vec(stream)? == expected;
        }
    }

    let rate = |ms: f64| N as f64 / 1e6 / (ms / 1e3);
    println!("radix sorts of {N} random 32-bit keys, best of 3 runs");
    println!("  32 passes of 1 bit  {:8.2} ms {:7.1} Mkeys/s", best[0], rate(best[0]));
    println!("  8 passes of 4 bits  {:8.2} ms {:7.1} Mkeys/s", best[1], rate(best[1]));

    // The 3 parts of 1 pass of 4 bits on the random keys, averaged like the histograms.
    let keys = DeviceBuffer::from_host(stream, &host)?;
    let tiles = LaunchConfig1D::new(N.div_ceil(TILE) as u32, 256, 0);
    let count = module.prepare_digit_counts(tiles)?;
    let scatter = module.prepare_scatter_digits(tiles)?;
    let count_ms = time_ms(stream, || {
        Ok(module.digit_counts(stream, &count, &keys, 0, &mut buffers.counts)?)
    })?;
    let scan_ms = time_ms(stream, || {
        let levels = &mut buffers.count_levels;
        scan(module, stream, &buffers.counts, &mut buffers.positions, levels)
    })?;
    let (counts, positions) = (&buffers.counts, &buffers.positions);
    let out = &mut buffers.spare;
    let scatter_ms = time_ms(stream, || {
        Ok(module.scatter_digits(stream, &scatter, &keys, 0, counts, positions, out)?)
    })?;
    print!("  1 pass of 4 bits: counts {count_ms:.2} ms, scan {scan_ms:.2} ms,");
    println!(" scatter {scatter_ms:.2} ms");
    Ok(ok)
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };

    let mut passed = small_example(&module, &stream)?;
    passed &= histograms(&module, &stream)?;
    passed &= sorts(&module, &stream)?;

    println!("{}", if passed { "PASSED" } else { "FAILED" });
    Ok(())
}
