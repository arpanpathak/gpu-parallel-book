//! Chapter 21: 3 inclusive scans of a tile (Hillis-Steele, Blelloch, warp shuffles),
//! the scan of a long array, and stream compaction by a scan and by warp votes.
//!
//!     cargo oxide run

use std::sync::Arc;
use std::time::{Duration, Instant};

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, device, kernel, launch_contract};
use cuda_device::{thread, warp};

const TILE: usize = 1024; // elements scanned by 1 block
const N: usize = 16 << 20; // elements in the measured runs

type Error = Box<dyn std::error::Error>;
type Levels = Vec<(DeviceBuffer<u32>, DeviceBuffer<u32>)>;
type Module = kernels::LoadedModule;

#[derive(Clone, Copy)]
enum Method {
    HillisSteele,
    Blelloch,
    Warps,
}

const METHODS: [(Method, &str); 3] = [
    (Method::HillisSteele, "Hillis-Steele"),
    (Method::Blelloch, "Blelloch"),
    (Method::Warps, "warp shuffles"),
];

#[cuda_module]
mod kernels {
    use super::*;

    /// Hillis-Steele: 1,024 threads, 1 element each, 2 buffers in shared memory.
    #[kernel]
    #[launch_contract(domain = 1, block = (1024, 1, 1))]
    pub fn scan_hillis_steele(
        x: &[u32],
        mut y: DisjointSlice<u32>,
        mut sums: DisjointSlice<u32>,
    ) {
        static mut BUF: SharedArray<u32, 2048> = SharedArray::UNINIT;
        let idx = thread::index_1d();
        let index = idx.get();
        let thread_idx = thread::threadIdx_x() as usize;

        // SAFETY: each thread writes only its own slot thread_idx, before the barrier.
        unsafe { BUF[thread_idx] = if index < x.len() { x[index] } else { 0 } };
        thread::sync_threads();

        let mut src = 0;
        let mut step = 0usize;
        #[unroll]
        while step < 10 {
            let distance = 1 << step; // 1, 2, 4, ..., 512
            let dst = TILE - src;
            // SAFETY: a step reads only buffer src and writes only this thread's slot of
            // buffer dst; the barrier ends the step before the next one reads buffer dst.
            unsafe {
                let mut value = BUF[src + thread_idx];
                if thread_idx >= distance {
                    value += BUF[src + thread_idx - distance];
                }
                BUF[dst + thread_idx] = value;
            }
            thread::sync_threads();
            src = dst;
            step += 1;
        }

        // SAFETY: after the last barrier the buffers are only read.
        let total = unsafe { BUF[src + thread_idx] };
        if let Some(out) = y.get_mut(idx) {
            *out = total;
        }
        if thread_idx == TILE - 1 {
            let block = thread::blockIdx_x() as usize;
            // SAFETY: only the last thread of each block writes, at its own block number.
            unsafe { *sums.get_unchecked_mut(block) = total };
        }
    }

    /// Blelloch: 512 threads, 2 elements each, 1 buffer; an up-sweep, then a down-sweep.
    #[kernel]
    #[launch_contract(domain = 1, block = (512, 1, 1))]
    pub fn scan_blelloch(
        x: &[u32],
        mut y: DisjointSlice<u32>,
        mut sums: DisjointSlice<u32>,
    ) {
        static mut CELLS: SharedArray<u32, 1024> = SharedArray::UNINIT;
        let thread_idx = thread::threadIdx_x() as usize;
        let block = thread::blockIdx_x() as usize;
        let index = block * TILE + 2 * thread_idx;

        let first = if index < x.len() { x[index] } else { 0 };
        let second = if index + 1 < x.len() { x[index + 1] } else { 0 };
        // SAFETY: each thread writes only its own 2 slots, before a barrier.
        unsafe {
            CELLS[2 * thread_idx] = first;
            CELLS[2 * thread_idx + 1] = second;
        }

        let mut step = 0usize;
        #[unroll]
        while step < 10 {
            let distance = 1 << step; // up-sweep: 1, 2, 4, ..., 512
            thread::sync_threads();
            if thread_idx < TILE / (2 * distance) {
                let right = (2 * thread_idx + 2) * distance - 1;
                // SAFETY: at each distance, the pairs (right - distance, right) of
                // different threads are disjoint, and the barriers separate the levels.
                unsafe { CELLS[right] += CELLS[right - distance] };
            }
            step += 1;
        }
        thread::sync_threads();
        if thread_idx == 0 {
            // SAFETY: thread 0 alone touches slot TILE - 1 between 2 barriers.
            unsafe {
                *sums.get_unchecked_mut(block) = CELLS[TILE - 1];
                CELLS[TILE - 1] = 0;
            }
        }

        let mut step = 0usize;
        #[unroll]
        while step < 10 {
            let distance = 512 >> step; // down-sweep: 512, 256, ..., 1
            thread::sync_threads();
            if thread_idx < TILE / (2 * distance) {
                let right = (2 * thread_idx + 2) * distance - 1;
                // SAFETY: as in the up-sweep, each pair belongs to 1 thread per level.
                unsafe {
                    let left = CELLS[right - distance];
                    CELLS[right - distance] = CELLS[right];
                    CELLS[right] += left;
                }
            }
            step += 1;
        }
        thread::sync_threads();

        // SAFETY: after the last barrier CELLS is only read; each thread alone writes its 2
        // elements of y, and both lie below y.len() when the test passes.
        unsafe {
            if index < y.len() {
                *y.get_unchecked_mut(index) = CELLS[2 * thread_idx] + first;
            }
            if index + 1 < y.len() {
                *y.get_unchecked_mut(index + 1) = CELLS[2 * thread_idx + 1] + second;
            }
        }
    }

    /// The inclusive scan of 1 value per lane across a warp (Section 8.7.2).
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

    /// The scan of Chapter 8: 256 threads, 4 elements each, warp shuffles and 2 barriers.
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
                // SAFETY: each thread alone writes its own 4 elements.
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

    /// flags[i] = 1 if x[i] is kept, 0 if it is removed.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn keep_flags(x: &[u32], threshold: u32, mut flags: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let index = idx.get();

        if let Some(flag) = flags.get_mut(idx) {
            *flag = (x[index] >= threshold) as u32;
        }
    }

    /// Writes each kept element at its exclusive scan: the inclusive scan minus its flag 1.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn scatter_kept(
        x: &[u32],
        flags: &[u32],
        scanned: &[u32],
        mut out: DisjointSlice<u32>,
    ) {
        let index = thread::index_1d().get();

        if index < x.len() && flags[index] == 1 {
            let at = (scanned[index] - 1) as usize;
            // SAFETY: the kept elements receive the distinct positions 0 to count - 1,
            // and out holds x.len() >= count elements.
            unsafe { *out.get_unchecked_mut(at) = x[index] };
        }
    }

    /// The number of kept elements in each tile of 1,024, counted with warp votes.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn count_kept(x: &[u32], threshold: u32, mut counts: DisjointSlice<u32>) {
        static mut WARPS: SharedArray<u32, 8> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let base = thread::blockIdx_x() as usize * TILE;

        let mut count = 0;
        let mut step = 0;
        #[unroll]
        while step < 4 {
            let index = base + step * 256 + tid;
            let keep = index < x.len() && x[index] >= threshold;
            count += warp::ballot(keep).count_ones();
            step += 1;
        }

        if warp::lane_id() == 0 {
            // SAFETY: lane 0 of each warp writes its own slot, before the barrier.
            unsafe { WARPS[tid / 32] = count };
        }
        thread::sync_threads();

        if tid == 0 {
            let mut total = 0;
            let mut other = 0;
            #[unroll]
            while other < 8 {
                // SAFETY: after the barrier the slots are only read.
                total += unsafe { WARPS[other] };
                other += 1;
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
        x: &[u32],
        threshold: u32,
        scanned: &[u32],
        mut out: DisjointSlice<u32>,
    ) {
        static mut WARPS: SharedArray<u32, 8> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id();
        let warp = tid / 32;
        let block = thread::blockIdx_x() as usize;
        let below = (1u32 << lane) - 1; // the lanes to the left of this one

        let mut next = if block > 0 { scanned[block - 1] } else { 0 };
        let mut step = 0;
        #[unroll]
        while step < 4 {
            let index = block * TILE + step * 256 + tid;
            let keep = index < x.len() && x[index] >= threshold;
            let votes = warp::ballot(keep);
            if lane == 0 {
                // SAFETY: lane 0 of each warp writes its own slot, before the barrier.
                unsafe { WARPS[warp] = votes.count_ones() };
            }
            thread::sync_threads();

            let mut before = 0;
            let mut total = 0;
            let mut other = 0;
            #[unroll]
            while other < 8 {
                // SAFETY: between the 2 barriers the slots are only read.
                let count = unsafe { WARPS[other] };
                before += if other < warp { count } else { 0 };
                total += count;
                other += 1;
            }

            if keep {
                let at = (next + before + (votes & below).count_ones()) as usize;
                // SAFETY: the kept elements receive the distinct positions 0 to count - 1,
                // in input order, and out holds x.len() >= count elements.
                unsafe { *out.get_unchecked_mut(at) = x[index] };
            }
            next += total;
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

/// Numbers below `limit` from a small linear congruential generator.
fn random_below(n: usize, limit: u32, seed: u32) -> Vec<u32> {
    let mut state = seed;

    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) % limit
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

/// Scans every tile of 1,024 elements on its own and writes the tile totals.
fn scan_tiles(
    module: &Module,
    stream: &Arc<CudaStream>,
    method: Method,
    x: &DeviceBuffer<u32>,
    y: &mut DeviceBuffer<u32>,
    sums: &mut DeviceBuffer<u32>,
) -> Result<(), Error> {
    let tiles = x.len().div_ceil(TILE) as u32;
    match method {
        Method::HillisSteele => {
            let grid = LaunchConfig1D::new(tiles, 1024, 0);
            let config = module.prepare_scan_hillis_steele(grid)?;
            module.scan_hillis_steele(stream, &config, x, y, sums)?;
        }
        Method::Blelloch => {
            let grid = LaunchConfig1D::new(tiles, 512, 0);
            let config = module.prepare_scan_blelloch(grid)?;
            module.scan_blelloch(stream, &config, x, y, sums)?;
        }
        Method::Warps => {
            let grid = LaunchConfig1D::new(tiles, 256, 0);
            let config = module.prepare_scan_warps(grid)?;
            module.scan_warps(stream, &config, x, y, sums)?;
        }
    }
    Ok(())
}

/// Inclusive scan: the tiles, then the tile totals (by the same function), then offsets.
fn scan(
    module: &Module,
    stream: &Arc<CudaStream>,
    method: Method,
    x: &DeviceBuffer<u32>,
    y: &mut DeviceBuffer<u32>,
    levels: &mut [(DeviceBuffer<u32>, DeviceBuffer<u32>)],
) -> Result<(), Error> {
    let tiles = x.len().div_ceil(TILE) as u32;
    let ((totals, scanned), rest) = levels.split_first_mut().ok_or("no level left")?;
    scan_tiles(module, stream, method, x, y, totals)?;

    if tiles > 1 {
        scan(module, stream, method, totals, scanned, rest)?;
        let config = module.prepare_add_offsets(LaunchConfig1D::new(tiles, 256, 0))?;
        module.add_offsets(stream, &config, y, scanned)?;
    }
    Ok(())
}

fn print(name: &str, values: &[u32]) {
    let cells: Vec<String> = values.iter().map(|v| v.to_string()).collect();
    println!("  {name:13} {}", cells.join(" "));
}

/// The worked example of the chapter: 8 values, 3 scans and 1 compaction.
fn small_example(module: &Module, stream: &Arc<CudaStream>) -> Result<bool, Error> {
    let values = [3u32, 1, 7, 0, 4, 1, 6, 3];
    let x = DeviceBuffer::from_host(stream, &values)?;
    let mut y = DeviceBuffer::<u32>::zeroed(stream, 8)?;
    let mut levels = scan_levels(stream, 8)?;

    let mut ok = true;
    println!("scans of 3 1 7 0 4 1 6 3");
    for (method, name) in METHODS {
        scan(module, stream, method, &x, &mut y, &mut levels)?;
        let result = y.to_host_vec(stream)?;
        print(name, &result);
        ok &= result == [3, 4, 11, 11, 15, 16, 22, 25];
    }

    let one = LaunchConfig1D::new(1, 256, 0);
    let mut flags = DeviceBuffer::<u32>::zeroed(stream, 8)?;
    let mut out = DeviceBuffer::<u32>::zeroed(stream, 8)?;
    module.keep_flags(stream, &module.prepare_keep_flags(one)?, &x, 3, &mut flags)?;
    scan(module, stream, Method::HillisSteele, &flags, &mut y, &mut levels)?;
    let scatter = module.prepare_scatter_kept(one)?;
    module.scatter_kept(stream, &scatter, &x, &flags, &y, &mut out)?;

    let f = flags.to_host_vec(stream)?;
    let s = y.to_host_vec(stream)?;
    let exclusive: Vec<u32> = s.iter().zip(&f).map(|(s, f)| s - f).collect();
    let kept = out.to_host_vec(stream)?[..s[7] as usize].to_vec();
    println!("compaction of 3 1 7 0 4 1 6 3, keep x >= 3");
    print("flags", &f);
    print("exclusive", &exclusive);
    print("output", &kept);

    Ok(ok && kept == [3, 7, 4, 6, 3])
}

/// The 3 scans of N values: tiles alone, and the whole array, checked against the CPU.
fn scans(module: &Module, stream: &Arc<CudaStream>) -> Result<bool, Error> {
    let host = random_below(N, 4, 1);
    let expected: Vec<u32> = host
        .iter()
        .scan(0u32, |run, &v| {
            *run += v;
            Some(*run)
        })
        .collect();
    let x = DeviceBuffer::from_host(stream, &host)?;
    let mut y = DeviceBuffer::<u32>::zeroed(stream, N)?;
    let mut levels = scan_levels(stream, N)?;
    let gb = 2.0 * N as f64 * 4.0 / 1e9;

    let mut ok = true;
    println!("inclusive scans of {N} values, average of 20 runs");
    println!("  method          tiles ms   GB/s   whole ms   GB/s");
    for (method, name) in METHODS {
        let (totals, _) = levels.first_mut().ok_or("no level")?;
        let one_level = || scan_tiles(module, stream, method, &x, &mut y, totals);
        let tiles = time_ms(stream, one_level)?;
        let all_levels = || scan(module, stream, method, &x, &mut y, &mut levels);
        let whole = time_ms(stream, all_levels)?;
        ok &= y.to_host_vec(stream)? == expected;

        let (tiles_gb, whole_gb) = (gb / (tiles / 1e3), gb / (whole / 1e3));
        println!("  {name:14} {tiles:9.3} {tiles_gb:6.1} {whole:10.3} {whole_gb:6.1}");
    }
    Ok(ok)
}

/// Compaction of N values in [0, 100) at 3 thresholds, by 3 passes and by warp votes.
fn compaction(module: &Module, stream: &Arc<CudaStream>) -> Result<bool, Error> {
    let host = random_below(N, 100, 2);
    let x = DeviceBuffer::from_host(stream, &host)?;
    let mut flags = DeviceBuffer::<u32>::zeroed(stream, N)?;
    let mut scanned = DeviceBuffer::<u32>::zeroed(stream, N)?;
    let mut levels = scan_levels(stream, N)?;

    let tiles = N.div_ceil(TILE);
    let mut counts = DeviceBuffer::<u32>::zeroed(stream, tiles)?;
    let mut tile_scan = DeviceBuffer::<u32>::zeroed(stream, tiles)?;
    let mut tile_levels = scan_levels(stream, tiles)?;

    let every = LaunchConfig1D::new(N.div_ceil(256) as u32, 256, 0);
    let per_tile = LaunchConfig1D::new(tiles as u32, 256, 0);
    let flag_cfg = module.prepare_keep_flags(every)?;
    let scatter_cfg = module.prepare_scatter_kept(every)?;
    let count_cfg = module.prepare_count_kept(per_tile)?;
    let votes_cfg = module.prepare_compact_votes(per_tile)?;

    let mut ok = true;
    println!("compaction of {N} values in [0, 100), average of 20 runs");
    println!("  keep x >=      kept   3 passes ms   GB/s   warp votes ms   GB/s");
    for threshold in [90u32, 50, 10] {
        let expected: Vec<u32> = host.iter().copied().filter(|&v| v >= threshold).collect();
        let kept = expected.len();
        let gb = (4 * N + 4 * kept) as f64 / 1e9;

        let mut out = DeviceBuffer::<u32>::zeroed(stream, N)?;
        let three = || -> Result<(), Error> {
            module.keep_flags(stream, &flag_cfg, &x, threshold, &mut flags)?;
            scan(module, stream, Method::Warps, &flags, &mut scanned, &mut levels)?;
            module.scatter_kept(stream, &scatter_cfg, &x, &flags, &scanned, &mut out)?;
            Ok(())
        };
        let three_ms = time_ms(stream, three)?;
        ok &= out.to_host_vec(stream)?[..kept] == expected[..];

        out = DeviceBuffer::<u32>::zeroed(stream, N)?;
        let votes = || -> Result<(), Error> {
            module.count_kept(stream, &count_cfg, &x, threshold, &mut counts)?;
            scan(module, stream, Method::Warps, &counts, &mut tile_scan, &mut tile_levels)?;
            module.compact_votes(stream, &votes_cfg, &x, threshold, &tile_scan, &mut out)?;
            Ok(())
        };
        let votes_ms = time_ms(stream, votes)?;
        ok &= out.to_host_vec(stream)?[..kept] == expected[..];

        let (three_gb, votes_gb) = (gb / (three_ms / 1e3), gb / (votes_ms / 1e3));
        print!("  {threshold:9} {kept:9} {three_ms:13.2} {three_gb:6.1}");
        println!(" {votes_ms:15.2} {votes_gb:6.1}");
    }
    Ok(ok)
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };

    let mut passed = small_example(&module, &stream)?;
    passed &= scans(&module, &stream)?;
    passed &= compaction(&module, &stream)?;

    println!("{}", if passed { "PASSED" } else { "FAILED" });
    Ok(())
}
