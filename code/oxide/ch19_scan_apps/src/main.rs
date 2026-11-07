//! Chapter 19: stream compaction, a radix sort and histogram equalisation in Rust,
//! built from the scan and the histogram of Chapter 8.
//!
//!     cargo oxide run

use std::sync::Arc;
use std::time::{Duration, Instant};

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, BlockAtomicU32, DeviceAtomicU32};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, device, kernel, launch_contract};
use cuda_device::{thread, warp};

const BLOCK: u32 = 256;
const TILE: usize = 1024; // elements per block in the scan and in the vote compaction
const N: usize = 16 << 20; // elements in the compaction and the sort
const SIDE: usize = 8192; // the image is SIDE x SIDE pixels of 1 byte

type Error = Box<dyn std::error::Error>;
type Levels = Vec<(DeviceBuffer<u32>, DeviceBuffer<u32>)>;
type Module = kernels::LoadedModule;

#[cuda_module]
mod kernels {
    use super::*;

    /// The inclusive scan of 1 value per lane across a warp (Section 8.7.2).
    #[device]
    fn warp_scan(mut v: u32) -> u32 {
        let lane = warp::lane_id();
        for offset in [1, 2, 4, 8, 16] {
            let left = warp::shuffle_up_sync(u32::MAX, v, offset);
            if lane >= offset {
                v += left;
            }
        }
        v
    }

    /// An inclusive scan of 1,024 values per block, 4 per thread (Section 8.7.3).
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn scan_blocks(
        input: &[u32],
        mut out: DisjointSlice<u32>,
        mut sums: DisjointSlice<u32>,
    ) {
        static mut WARPS: SharedArray<u32, 8> = SharedArray::UNINIT;
        let first = thread::index_1d().get() * 4;
        let lane = warp::lane_id() as usize;
        let warp = thread::threadIdx_x() as usize / 32;

        let mut a = [0u32; 4];
        let mut own = 0;
        let mut k = 0;
        #[unroll]
        while k < 4 {
            if first + k < input.len() {
                own += input[first + k];
            }
            a[k] = own;
            k += 1;
        }

        let v = warp_scan(own);
        if lane == 31 {
            // SAFETY: lane 31 of each warp writes its own slot, before the barrier.
            unsafe { WARPS[warp] = v };
        }
        thread::sync_threads();

        if warp == 0 {
            // SAFETY: only warp 0 touches the slots between the 2 barriers.
            let w = warp_scan(if lane < 8 { unsafe { WARPS[lane] } } else { 0 });
            if lane < 8 {
                unsafe { WARPS[lane] = w };
            }
        }
        thread::sync_threads();

        // SAFETY: after the second barrier the slots are only read.
        let before = v - own + if warp > 0 { unsafe { WARPS[warp - 1] } } else { 0 };
        let mut k = 0;
        #[unroll]
        while k < 4 {
            if first + k < out.len() {
                // SAFETY: thread t alone writes elements 4t to 4t + 3.
                unsafe { *out.get_unchecked_mut(first + k) = before + a[k] };
            }
            k += 1;
        }
        if thread::threadIdx_x() == BLOCK - 1 {
            let block = thread::blockIdx_x() as usize;
            // SAFETY: only the last thread of each block writes, at its own block number.
            unsafe { *sums.get_unchecked_mut(block) = before + a[3] };
        }
    }

    /// Adds to every value of block b the total of blocks 0 to b - 1.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn add_offsets(mut out: DisjointSlice<u32>, scanned: &[u32]) {
        let block = thread::blockIdx_x() as usize;
        let first = thread::index_1d().get() * 4;

        if block > 0 {
            let mut k = 0;
            #[unroll]
            while k < 4 {
                if first + k < out.len() {
                    // SAFETY: thread t alone writes elements 4t to 4t + 3.
                    unsafe { *out.get_unchecked_mut(first + k) += scanned[block - 1] };
                }
                k += 1;
            }
        }
    }

    /// flags[i] = 1 if input[i] is kept, 0 if it is removed.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn keep_flags(input: &[f32], threshold: f32, mut flags: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();

        if let Some(flag) = flags.get_mut(idx) {
            *flag = (input[i] >= threshold) as u32;
        }
    }

    /// Writes kept element i at its exclusive scan: the inclusive scan minus 1.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn scatter_kept(
        input: &[f32],
        flags: &[u32],
        scanned: &[u32],
        mut out: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();

        if i < input.len() && flags[i] == 1 {
            let at = (scanned[i] - 1) as usize;
            // SAFETY: the kept elements receive the distinct positions 0 to count - 1,
            // and out holds input.len() >= count elements.
            unsafe { *out.get_unchecked_mut(at) = input[i] };
        }
    }

    /// The number of kept elements in each tile of 1,024, counted with warp votes.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn count_kept(input: &[f32], threshold: f32, mut counts: DisjointSlice<u32>) {
        static mut WARPS: SharedArray<u32, 8> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let base = thread::blockIdx_x() as usize * TILE;

        let mut count = 0;
        let mut k = 0;
        #[unroll]
        while k < 4 {
            let i = base + k * 256 + tid;
            let keep = i < input.len() && input[i] >= threshold;
            count += warp::ballot(keep).count_ones();
            k += 1;
        }

        if warp::lane_id() == 0 {
            // SAFETY: lane 0 of each warp writes its own slot, before the barrier.
            unsafe { WARPS[tid / 32] = count };
        }
        thread::sync_threads();

        if tid == 0 {
            let mut total = 0;
            let mut w = 0;
            #[unroll]
            while w < 8 {
                // SAFETY: after the barrier the slots are only read.
                total += unsafe { WARPS[w] };
                w += 1;
            }
            let block = thread::blockIdx_x() as usize;
            // SAFETY: only thread 0 of each block writes, at its own block number.
            unsafe { *counts.get_unchecked_mut(block) = total };
        }
    }

    /// Writes the kept elements of each tile in order, after those of the earlier tiles.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn compact_votes(
        input: &[f32],
        threshold: f32,
        scanned: &[u32],
        mut out: DisjointSlice<f32>,
    ) {
        static mut WARPS: SharedArray<u32, 8> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id();
        let warp = tid / 32;
        let block = thread::blockIdx_x() as usize;
        let below = (1u32 << lane) - 1; // the lanes to the left of this one

        let mut next = if block > 0 { scanned[block - 1] } else { 0 };
        let mut k = 0;
        #[unroll]
        while k < 4 {
            let i = block * TILE + k * 256 + tid;
            let keep = i < input.len() && input[i] >= threshold;
            let votes = warp::ballot(keep);
            if lane == 0 {
                // SAFETY: lane 0 of each warp writes its own slot, before the barrier.
                unsafe { WARPS[warp] = votes.count_ones() };
            }
            thread::sync_threads();

            let mut before = 0;
            let mut total = 0;
            let mut w = 0;
            #[unroll]
            while w < 8 {
                // SAFETY: between the 2 barriers the slots are only read.
                let c = unsafe { WARPS[w] };
                before += if w < warp { c } else { 0 };
                total += c;
                w += 1;
            }

            if keep {
                let rank = (votes & below).count_ones();
                let at = (next + before + rank) as usize;
                // SAFETY: the kept elements receive the distinct positions 0 to count - 1,
                // in input order, and out holds input.len() >= count elements.
                unsafe { *out.get_unchecked_mut(at) = input[i] };
            }
            next += total;
            thread::sync_threads();
            k += 1;
        }
    }

    /// flags[i] = 1 if bit `bit` of keys[i] is 0: those keys go to the front.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn zero_flags(keys: &[u32], bit: u32, mut flags: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();

        if let Some(flag) = flags.get_mut(idx) {
            *flag = 1 - ((keys[i] >> bit) & 1);
        }
    }

    /// A stable split: keys with a 0 bit first, then keys with a 1 bit, each in order.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn split_scatter(
        keys: &[u32],
        flags: &[u32],
        scanned: &[u32],
        mut out: DisjointSlice<u32>,
    ) {
        let i = thread::index_1d().get();

        if i < keys.len() {
            let zeros = scanned[keys.len() - 1] as usize;
            let upto = scanned[i] as usize; // zero-bit keys among 0 to i
            let at = if flags[i] == 1 { upto - 1 } else { zeros + i - upto };
            // SAFETY: zero-bit keys receive 0 to zeros - 1 and one-bit keys zeros to
            // n - 1, each position once: a permutation of 0 to n - 1.
            unsafe { *out.get_unchecked_mut(at) = keys[i] };
        }
    }

    /// A histogram of bytes in shared bins, reading 4 bytes per load (Section 8.8).
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn hist_bytes(words: &[u32], bins: &[DeviceAtomicU32]) {
        static mut LOCAL: SharedArray<u32, 256> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;

        // SAFETY: each thread zeroes only its own bin, before the barrier.
        unsafe { LOCAL[tid] = 0 };
        thread::sync_threads();

        let stride = (thread::gridDim_x() * BLOCK) as usize;
        let mut i = thread::index_1d().get();
        while i < words.len() {
            let mut k = 0;
            #[unroll]
            while k < 4 {
                let byte = (words[i] >> (8 * k)) & 255;
                // SAFETY: between the 2 barriers every access to LOCAL is atomic.
                let bin = unsafe {
                    let base = (*&raw mut LOCAL).as_mut_ptr();
                    BlockAtomicU32::from_ptr(base.add(byte as usize))
                };
                bin.fetch_add(1, AtomicOrdering::Relaxed);
                k += 1;
            }
            i += stride;
        }
        thread::sync_threads();

        // SAFETY: after the barrier no thread writes LOCAL again.
        let local = unsafe { LOCAL[tid] };
        bins[tid].fetch_add(local, AtomicOrdering::Relaxed);
    }

    /// The equalisation table, 1 block of 256 threads, 1 thread per grey level.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn equalise_lut(bins: &[u32], pixels: u32, mut lut: DisjointSlice<u32>) {
        static mut WARPS: SharedArray<u32, 8> = SharedArray::UNINIT;
        static mut FIRST: SharedArray<u32, 1> = SharedArray::UNINIT;
        let idx = thread::index_1d();
        let level = idx.get();
        let warp = level / 32;

        let count = bins[level];
        let mut cdf = warp_scan(count);
        if warp::lane_id() == 31 {
            // SAFETY: lane 31 of each warp writes its own slot, before the barrier.
            unsafe { WARPS[warp] = cdf };
        }
        thread::sync_threads();

        let mut w = 0;
        while w < warp {
            // SAFETY: after the barrier the slots are only read.
            cdf += unsafe { WARPS[w] };
            w += 1;
        }
        if count > 0 && cdf == count {
            // SAFETY: only the lowest occupied level has cdf == count > 0.
            unsafe { FIRST[0] = cdf };
        }
        thread::sync_threads();

        // SAFETY: after the barrier FIRST is only read.
        let low = unsafe { FIRST[0] } as u64;
        let range = (pixels as u64 - low).max(1);
        let above = (cdf as u64).saturating_sub(low);
        if let Some(entry) = lut.get_mut(idx) {
            *entry = ((255 * above + range / 2) / range) as u32;
        }
    }

    /// Replaces each of the 4 bytes of every word by its entry in the table.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn apply_lut(words: &[u32], lut: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();

        if let Some(word) = out.get_mut(idx) {
            let w = words[i];
            let mut result = 0;
            let mut k = 0;
            #[unroll]
            while k < 4 {
                let level = (w >> (8 * k)) & 255;
                result |= lut[level as usize] << (8 * k);
                k += 1;
            }
            *word = result;
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

/// Numbers from a small linear congruential generator, so the run needs no extra crate.
fn random(n: usize, seed: u32) -> Vec<u32> {
    let mut state = seed;

    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        })
        .collect()
}

fn blocks_for(n: usize) -> u32 {
    n.div_ceil(BLOCK as usize) as u32
}

/// Buffers for the block totals and their scan, per level: n / 1,024, / 1,024 again, ... 1.
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

/// The inclusive scan of Section 8.7.4: blocks, then block totals, then offsets.
fn scan(
    module: &Module,
    stream: &Arc<CudaStream>,
    input: &DeviceBuffer<u32>,
    out: &mut DeviceBuffer<u32>,
    levels: &mut [(DeviceBuffer<u32>, DeviceBuffer<u32>)],
) -> Result<(), Error> {
    let blocks = input.len().div_ceil(TILE) as u32;
    let config = LaunchConfig1D::new(blocks, BLOCK, 0);
    let ((totals, scanned), rest) = levels.split_first_mut().ok_or("no level left")?;
    module.scan_blocks(stream, &module.prepare_scan_blocks(config)?, input, out, totals)?;

    if blocks > 1 {
        scan(module, stream, totals, scanned, rest)?;
        module.add_offsets(stream, &module.prepare_add_offsets(config)?, out, scanned)?;
    }
    Ok(())
}

/// Flags, scan and scatter on 8 values, printed step by step.
fn small_examples(module: &Module, stream: &Arc<CudaStream>) -> Result<bool, Error> {
    let keys = [3.0f32, 8.0, 1.0, 9.0, 4.0, 7.0, 2.0, 6.0];
    let input = DeviceBuffer::from_host(stream, &keys)?;
    let mut flags = DeviceBuffer::<u32>::zeroed(stream, 8)?;
    let mut scanned = DeviceBuffer::<u32>::zeroed(stream, 8)?;
    let mut out = DeviceBuffer::<f32>::zeroed(stream, 8)?;
    let mut levels = scan_levels(stream, 8)?;
    let one = LaunchConfig1D::new(1, BLOCK, 0);

    module.keep_flags(stream, &module.prepare_keep_flags(one)?, &input, 5.0, &mut flags)?;
    scan(module, stream, &flags, &mut scanned, &mut levels)?;
    let scatter = module.prepare_scatter_kept(one)?;
    module.scatter_kept(stream, &scatter, &input, &flags, &scanned, &mut out)?;

    let inclusive = scanned.to_host_vec(stream)?;
    let flags_host = flags.to_host_vec(stream)?;
    let pairs = inclusive.iter().zip(&flags_host);
    let exclusive: Vec<u32> = pairs.map(|(s, f)| s - f).collect();
    let count = inclusive[7] as usize;
    let kept = out.to_host_vec(stream)?[..count].to_vec();
    println!("compaction of {keys:?}, keep x >= 5");
    println!("  flags      {flags_host:?}");
    println!("  inclusive  {inclusive:?}");
    println!("  exclusive  {exclusive:?}");
    println!("  output     {kept:?}");

    let keys = [5u32, 2, 7, 0, 3, 6, 1, 4];
    let input = DeviceBuffer::from_host(stream, &keys)?;
    let mut out = DeviceBuffer::<u32>::zeroed(stream, 8)?;
    module.zero_flags(stream, &module.prepare_zero_flags(one)?, &input, 0, &mut flags)?;
    scan(module, stream, &flags, &mut scanned, &mut levels)?;
    let split = module.prepare_split_scatter(one)?;
    module.split_scatter(stream, &split, &input, &flags, &scanned, &mut out)?;

    let split_host = out.to_host_vec(stream)?;
    println!("split of {keys:?} by bit 0");
    println!("  zero flags {:?}", flags.to_host_vec(stream)?);
    println!("  inclusive  {:?}", scanned.to_host_vec(stream)?);
    println!("  output     {split_host:?}");

    Ok(kept == [8.0, 9.0, 7.0, 6.0] && split_host == [2, 0, 6, 4, 5, 7, 3, 1])
}

/// Stream compaction of N floats at 3 thresholds, by 3 passes and by warp votes.
fn compaction(module: &Module, stream: &Arc<CudaStream>) -> Result<bool, Error> {
    let values: Vec<f32> = random(N, 1)
        .iter()
        .map(|&r| (r >> 8) as f32 / 16_777_216.0)
        .collect();
    let input = DeviceBuffer::from_host(stream, &values)?;
    let mut flags = DeviceBuffer::<u32>::zeroed(stream, N)?;
    let mut scanned = DeviceBuffer::<u32>::zeroed(stream, N)?;
    let mut levels = scan_levels(stream, N)?;

    let tiles = N.div_ceil(TILE);
    let mut counts = DeviceBuffer::<u32>::zeroed(stream, tiles)?;
    let mut tile_scan = DeviceBuffer::<u32>::zeroed(stream, tiles)?;
    let mut tile_levels = scan_levels(stream, tiles)?;

    let every = LaunchConfig1D::new(blocks_for(N), BLOCK, 0);
    let per_tile = LaunchConfig1D::new(tiles as u32, BLOCK, 0);
    let flag_cfg = module.prepare_keep_flags(every)?;
    let scatter_cfg = module.prepare_scatter_kept(every)?;
    let count_cfg = module.prepare_count_kept(per_tile)?;
    let votes_cfg = module.prepare_compact_votes(per_tile)?;

    println!("compaction of {N} floats in [0, 1), average of 20 runs");
    println!("  keep x >=   kept   3 passes ms   GB/s   warp votes ms   GB/s   Gelem/s");
    let mut correct = true;
    for threshold in [0.9f32, 0.5, 0.1] {
        let wanted = values.iter().copied().filter(|&v| v >= threshold);
        let expected: Vec<f32> = wanted.collect();
        let kept = expected.len();
        let bytes = (4 * N + 4 * kept) as f64;

        let mut out = DeviceBuffer::<f32>::zeroed(stream, N)?;
        let mut three = || -> Result<(), Error> {
            module.keep_flags(stream, &flag_cfg, &input, threshold, &mut flags)?;
            scan(module, stream, &flags, &mut scanned, &mut levels)?;
            module.scatter_kept(stream, &scatter_cfg, &input, &flags, &scanned, &mut out)?;
            Ok(())
        };
        let three_ms = time_ms(stream, &mut three)?;
        correct &= out.to_host_vec(stream)?[..kept] == expected[..];

        out = DeviceBuffer::<f32>::zeroed(stream, N)?;
        let mut votes = || -> Result<(), Error> {
            module.count_kept(stream, &count_cfg, &input, threshold, &mut counts)?;
            scan(module, stream, &counts, &mut tile_scan, &mut tile_levels)?;
            let scanned = &tile_scan;
            module.compact_votes(stream, &votes_cfg, &input, threshold, scanned, &mut out)?;
            Ok(())
        };
        let votes_ms = time_ms(stream, &mut votes)?;
        correct &= out.to_host_vec(stream)?[..kept] == expected[..];

        let three_gb = bytes / 1e9 / (three_ms / 1e3);
        let votes_gb = bytes / 1e9 / (votes_ms / 1e3);
        let gelem = N as f64 / 1e9 / (votes_ms / 1e3);
        print!("  {threshold:9.1} {kept:9} {three_ms:10.2} {three_gb:9.1}");
        println!(" {votes_ms:12.2} {votes_gb:9.1} {gelem:8.2}");
    }

    Ok(correct)
}

/// The buffers that a radix sort of n keys needs besides the keys themselves.
struct SortBuffers {
    spare: DeviceBuffer<u32>,
    flags: DeviceBuffer<u32>,
    scanned: DeviceBuffer<u32>,
    levels: Levels,
}

/// Sorts `keys` by 32 stable splits, bit 0 first. The result ends in `keys`.
fn radix_sort(
    module: &Module,
    stream: &Arc<CudaStream>,
    keys: &mut DeviceBuffer<u32>,
    b: &mut SortBuffers,
) -> Result<(), Error> {
    let every = LaunchConfig1D::new(blocks_for(keys.len()), BLOCK, 0);
    let flag_cfg = module.prepare_zero_flags(every)?;
    let split_cfg = module.prepare_split_scatter(every)?;

    for bit in 0..32 {
        module.zero_flags(stream, &flag_cfg, keys, bit, &mut b.flags)?;
        scan(module, stream, &b.flags, &mut b.scanned, &mut b.levels)?;
        module.split_scatter(stream, &split_cfg, keys, &b.flags, &b.scanned, &mut b.spare)?;
        std::mem::swap(keys, &mut b.spare);
    }
    Ok(())
}

/// A radix sort of N random 32-bit keys, against 1 CPU core.
fn sorting(module: &Module, stream: &Arc<CudaStream>) -> Result<bool, Error> {
    let host = random(N, 7);
    let mut keys = DeviceBuffer::from_host(stream, &host)?;
    let mut buffers = SortBuffers {
        spare: DeviceBuffer::<u32>::zeroed(stream, N)?,
        flags: DeviceBuffer::<u32>::zeroed(stream, N)?,
        scanned: DeviceBuffer::<u32>::zeroed(stream, N)?,
        levels: scan_levels(stream, N)?,
    };

    let warm = Instant::now();
    while warm.elapsed() < Duration::from_millis(400) {
        radix_sort(module, stream, &mut keys, &mut buffers)?;
        stream.synchronize()?;
    }

    let mut best = f64::MAX;
    let mut correct = true;
    let mut expected = host.clone();
    let start = Instant::now();
    expected.sort_unstable();
    let cpu_ms = start.elapsed().as_secs_f64() * 1e3;

    for _ in 0..3 {
        let mut keys = DeviceBuffer::from_host(stream, &host)?;
        stream.synchronize()?;
        let start = Instant::now();
        radix_sort(module, stream, &mut keys, &mut buffers)?;
        stream.synchronize()?;
        best = best.min(start.elapsed().as_secs_f64() * 1e3);
        correct &= keys.to_host_vec(stream)? == expected;
    }

    let gpu_rate = N as f64 / 1e6 / (best / 1e3);
    let cpu_rate = N as f64 / 1e6 / (cpu_ms / 1e3);
    println!("radix sort of {N} keys, 32 passes of 1 bit, best of 3 runs");
    println!("  GPU        {best:8.2} ms  {gpu_rate:7.1} Mkeys/s");
    println!("  CPU 1 core {cpu_ms:8.2} ms  {cpu_rate:7.1} Mkeys/s  (sort_unstable)");

    Ok(correct)
}

/// A low-contrast image: levels 96 to 158, a smooth pattern plus noise, 4 pixels per word.
fn image_words() -> Vec<u32> {
    let noise = random(SIDE * SIDE / 4, 11);

    (0..SIDE * SIDE / 4)
        .map(|w| {
            let mut word = 0u32;
            for k in 0..4 {
                let p = 4 * w + k;
                let (x, y) = (p % SIDE, p / SIDE);
                let smooth = ((x * 7 + y * 3) >> 6) % 32;
                let grain = (noise[w] >> (8 * k + 3)) & 31;
                word |= (96 + smooth as u32 + grain) << (8 * k);
            }
            word
        })
        .collect()
}

/// Shares of the pixels in the 8 groups of 32 grey levels, in percent.
fn shares(bins: &[u32]) -> Vec<f64> {
    let total: u64 = bins.iter().map(|&c| c as u64).sum();

    bins.chunks(32)
        .map(|group| group.iter().map(|&c| c as u64).sum::<u64>() as f64 * 100.0)
        .map(|count| count / total as f64)
        .collect()
}

fn histogram_of(words: &[u32]) -> Vec<u32> {
    let mut bins = vec![0u32; 256];
    for w in words {
        for k in 0..4 {
            bins[((w >> (8 * k)) & 255) as usize] += 1;
        }
    }
    bins
}

/// Histogram, table and mapping on a SIDE x SIDE image, checked against the CPU.
fn equalisation(module: &Module, stream: &Arc<CudaStream>) -> Result<bool, Error> {
    let host = image_words();
    let words = DeviceBuffer::from_host(stream, &host)?;
    let mut out = DeviceBuffer::<u32>::zeroed(stream, host.len())?;
    let mut lut = DeviceBuffer::<u32>::zeroed(stream, 256)?;
    let pixels = (SIDE * SIDE) as u32;

    let hist_cfg = module.prepare_hist_bytes(LaunchConfig1D::new(1024, BLOCK, 0))?;
    let lut_cfg = module.prepare_equalise_lut(LaunchConfig1D::new(1, BLOCK, 0))?;
    let every = LaunchConfig1D::new(blocks_for(host.len()), BLOCK, 0);
    let apply_cfg = module.prepare_apply_lut(every)?;

    let scratch = DeviceBuffer::<u32>::zeroed(stream, 256)?.cast_elem::<DeviceAtomicU32>();
    let hist = || Ok(module.hist_bytes(stream, &hist_cfg, &words, &scratch)?);
    let hist_ms = time_ms(stream, hist)?;

    let bins = DeviceBuffer::<u32>::zeroed(stream, 256)?.cast_elem::<DeviceAtomicU32>();
    module.hist_bytes(stream, &hist_cfg, &words, &bins)?;
    let bins = bins.cast_elem::<u32>();
    let table = || Ok(module.equalise_lut(stream, &lut_cfg, &bins, pixels, &mut lut)?);
    let lut_ms = time_ms(stream, table)?;
    let apply = || Ok(module.apply_lut(stream, &apply_cfg, &words, &lut, &mut out)?);
    let apply_ms = time_ms(stream, apply)?;

    let before = bins.to_host_vec(stream)?;
    let table = lut.to_host_vec(stream)?;
    let after_words = out.to_host_vec(stream)?;
    let after = histogram_of(&after_words);

    let expected_bins = histogram_of(&host);
    let low = expected_bins.iter().copied().find(|&b| b > 0).unwrap_or(0) as u64;
    let mut cdf = 0u64;
    let range = (pixels as u64 - low).max(1);
    let expected_lut: Vec<u32> = expected_bins
        .iter()
        .map(|&b| {
            cdf += b as u64;
            ((255 * cdf.saturating_sub(low) + range / 2) / range) as u32
        })
        .collect();
    let mapped = host.iter().zip(&after_words).all(|(&w, &o)| {
        (0..4).all(|k| {
            let level = (w >> (8 * k)) & 255;
            expected_lut[level as usize] == (o >> (8 * k)) & 255
        })
    });

    let mb = (SIDE * SIDE) as f64 / 1e6;
    println!("histogram equalisation of {SIDE} x {SIDE} pixels, average of 20 runs");
    let hist_gb = mb / 1e3 / (hist_ms / 1e3);
    println!("  histogram  {hist_ms:7.3} ms  {hist_gb:6.1} GB/s");
    println!("  table      {lut_ms:7.3} ms");
    let apply_gb = 2.0 * mb / 1e3 / (apply_ms / 1e3);
    println!("  mapping    {apply_ms:7.3} ms  {apply_gb:6.1} GB/s");
    println!("  levels    0-31  32-63  64-95 96-127 128-159 160-191 192-223 224-255");
    let row = |name: &str, s: &[f64]| {
        let cells: Vec<String> = s.iter().map(|v| format!("{v:6.1}")).collect();
        println!("  {name:7} {} %", cells.join(" "));
    };
    row("before", &shares(&before));
    row("after", &shares(&after));
    let (a, b, c) = (table[96], table[127], table[158]);
    println!("  table: level 96 -> {a}, 127 -> {b}, 158 -> {c}");

    Ok(before == expected_bins && table == expected_lut && mapped)
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };

    let mut passed = small_examples(&module, &stream)?;
    passed &= compaction(&module, &stream)?;
    passed &= sorting(&module, &stream)?;
    passed &= equalisation(&module, &stream)?;

    println!("{}", if passed { "PASSED" } else { "FAILED" });
    Ok(())
}
