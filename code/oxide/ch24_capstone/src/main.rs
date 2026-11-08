//! Chapter 24: an image pipeline. Gray conversion, histogram equalization, a separable
//! Gaussian blur and Sobel edges, checked pixel by pixel against the CPU, then a batch of
//! images on 1, 2 and 3 streams.
//!
//!     cargo oxide run

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D, LaunchConfig2D};
use cuda_core::PinnedHostBuffer;
use cuda_device::atomic::{AtomicOrdering, BlockAtomicU32, DeviceAtomicU32};
use cuda_device::float::sqrt_rn_f32;
use cuda_device::thread::Index2D;
use cuda_device::{DisjointSlice, SharedArray, cuda_module, device, kernel, launch_contract};
use cuda_device::{thread, warp};

const W: usize = 1920;
const H: usize = 1080;
const PIXELS: usize = W * H; // 2,073,600
const BATCH: usize = 128;

const ROW_T: usize = 128; // outputs per block of the row pass
const COL_W: usize = 32; // columns per block of the column pass
const COL_H: usize = 8; // rows per block of the column pass

type Error = Box<dyn std::error::Error>;

#[cuda_module]
mod kernels {
    use super::*;

    /// Position `i - 2` clamped to [0, n): slot `i` of a window that starts 2 before.
    #[device]
    fn tap(i: usize, n: usize) -> usize {
        if i < 2 { 0 } else { (i - 2).min(n - 1) }
    }

    #[device]
    fn warp_scan(mut v: u32) -> u32 {
        let lane = warp::lane_id();
        let mut step = 0;
        #[unroll]
        while step < 5 {
            let offset = 1 << step;
            let left = warp::shuffle_up_sync(u32::MAX, v, offset);
            if lane >= offset {
                v += left;
            }
            step += 1;
        }
        v
    }

    /// 1 thread per pixel. BT.601 weights in 8-bit fixed point: 77 + 150 + 29 = 256.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn grayscale(rgb: &[u8], mut gray: DisjointSlice<u8>) {
        let idx = thread::index_1d();
        let p = 3 * idx.get();

        if let Some(g) = gray.get_mut(idx) {
            let (r, gr, b) = (rgb[p] as u32, rgb[p + 1] as u32, rgb[p + 2] as u32);
            *g = ((77 * r + 150 * gr + 29 * b + 128) >> 8) as u8;
        }
    }

    /// Counts gray levels in shared bins, then adds each block's bins to the global ones.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn histogram(gray: &[u8], bins: &[DeviceAtomicU32]) {
        static mut LOCAL: SharedArray<u32, 256> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;

        // SAFETY: each thread zeroes only its own bin, before the barrier.
        unsafe { LOCAL[tid] = 0 };
        thread::sync_threads();

        let stride = (thread::gridDim_x() * 256) as usize;
        let mut i = thread::index_1d().get();
        while i < gray.len() {
            // SAFETY: between the 2 barriers every access to LOCAL is atomic.
            let bin = unsafe {
                let base = (*&raw mut LOCAL).as_mut_ptr();
                BlockAtomicU32::from_ptr(base.add(gray[i] as usize))
            };
            bin.fetch_add(1, AtomicOrdering::Relaxed);
            i += stride;
        }
        thread::sync_threads();

        // SAFETY: after the barrier no thread writes LOCAL again.
        let local = unsafe { LOCAL[tid] };
        if local > 0 {
            bins[tid].fetch_add(local, AtomicOrdering::Relaxed);
        }
    }

    /// 1 block of 256 threads, thread v for gray level v: the cumulative count of level v,
    /// then its new level. Each bin is read and reset to 0 for the next image.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn equalise_table(bins: &[DeviceAtomicU32], mut table: DisjointSlice<u8>) {
        static mut WARPS: SharedArray<u32, 8> = SharedArray::UNINIT;
        static mut FIRST: SharedArray<u32, 1> = SharedArray::UNINIT;
        let v = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let warp = v / 32;

        let count = bins[v].swap(0, AtomicOrdering::Relaxed);
        let inclusive = warp_scan(count);
        if lane == 31 {
            // SAFETY: lane 31 of each warp writes its own slot, before the barrier.
            unsafe { WARPS[warp] = inclusive };
        }
        if v == 0 {
            // SAFETY: only thread 0 writes FIRST before the barrier.
            unsafe { FIRST[0] = u32::MAX };
        }
        thread::sync_threads();

        if warp == 0 {
            // SAFETY: only warp 0 touches WARPS between the 2 barriers.
            let w = warp_scan(if lane < 8 { unsafe { WARPS[lane] } } else { 0 });
            if lane < 8 {
                unsafe { WARPS[lane] = w };
            }
        }
        thread::sync_threads();

        // SAFETY: WARPS is only read from here on; FIRST is changed only atomically.
        let cdf = inclusive + if warp > 0 { unsafe { WARPS[warp - 1] } } else { 0 };
        let first = unsafe { BlockAtomicU32::from_ptr((*&raw mut FIRST).as_mut_ptr()) };
        if count > 0 {
            first.fetch_min(cdf, AtomicOrdering::Relaxed);
        }
        thread::sync_threads();

        let min = first.load(AtomicOrdering::Relaxed);
        let span = (unsafe { WARPS[7] } - min) as u64;
        let above = cdf.saturating_sub(min) as u64;
        let level = if span == 0 { v as u64 } else { (above * 255 + span / 2) / span };

        if let Some(t) = table.get_mut(thread::index_1d()) {
            *t = level as u8;
        }
    }

    /// 1 thread per pixel: the new level of each pixel, looked up in the table.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn apply_table(gray: &[u8], table: &[u8], mut out: DisjointSlice<u8>) {
        let idx = thread::index_1d();
        let i = idx.get();

        if let Some(o) = out.get_mut(idx) {
            *o = table[gray[i] as usize];
        }
    }

    /// Row pass, weights 1 4 6 4 1. Slot s of the tile holds pixel x0 + s - 2 of the row,
    /// clamped: 2 halo slots on the left, 128 own pixels, 2 halo slots on the right.
    #[kernel]
    #[launch_contract(domain = 2, coordinates = u32, block = (128, 1, 1))]
    pub fn blur_rows(src: &[u8], mut out: DisjointSlice<u16, Index2D<W>>) {
        static mut TILE: SharedArray<u32, { ROW_T + 4 }> = SharedArray::UNINIT;
        let tx = thread::threadIdx_x() as usize;
        let x0 = thread::blockIdx_x() as usize * ROW_T;
        let row = thread::blockIdx_y() as usize * W;
        let pixel = |s: usize| src[row + tap(x0 + s, W)] as u32;

        // SAFETY: each slot is written by exactly 1 thread, before the barrier.
        unsafe { TILE[tx + 2] = pixel(tx + 2) };
        if tx < 2 {
            unsafe { TILE[tx] = pixel(tx) };
            unsafe { TILE[ROW_T + 2 + tx] = pixel(ROW_T + 2 + tx) };
        }
        thread::sync_threads();

        // SAFETY: after the barrier the tile is only read.
        let t = |k: usize| unsafe { TILE[tx + k] };
        let sum = t(0) + 4 * t(1) + 6 * t(2) + 4 * t(3) + t(4);

        if let Some(idx) = thread::index_2d::<W>()
            && let Some(o) = out.get_mut(idx)
        {
            *o = sum as u16;
        }
    }

    /// Column pass on a tile of 32 columns and 8 + 4 rows. The 2 passes weigh by
    /// 16 x 16 = 256, so the result is the rounded sum shifted right by 8.
    #[kernel]
    #[launch_contract(domain = 2, coordinates = u32, block = (32, 8, 1))]
    pub fn blur_cols(src: &[u16], mut out: DisjointSlice<u8, Index2D<W>>) {
        static mut TILE: SharedArray<u32, { (COL_H + 4) * COL_W }> = SharedArray::UNINIT;
        let (tx, ty) = (thread::threadIdx_x() as usize, thread::threadIdx_y() as usize);
        let x = thread::blockIdx_x() as usize * COL_W + tx;
        let y0 = thread::blockIdx_y() as usize * COL_H;
        let value = |s: usize| src[tap(y0 + s, H) * W + x] as u32;

        // SAFETY: each slot is written by exactly 1 thread, before the barrier.
        unsafe { TILE[(ty + 2) * COL_W + tx] = value(ty + 2) };
        if ty < 2 {
            unsafe { TILE[ty * COL_W + tx] = value(ty) };
            unsafe { TILE[(COL_H + 2 + ty) * COL_W + tx] = value(COL_H + 2 + ty) };
        }
        thread::sync_threads();

        // SAFETY: after the barrier the tile is only read.
        let t = |k: usize| unsafe { TILE[(ty + k) * COL_W + tx] };
        let sum = t(0) + 4 * t(1) + 6 * t(2) + 4 * t(3) + t(4);

        if let Some(idx) = thread::index_2d::<W>()
            && let Some(o) = out.get_mut(idx)
        {
            *o = ((sum + 128) >> 8) as u8;
        }
    }

    /// Sobel gradient magnitude divided by 4, rounded and clamped to 255. The 8 neighbors
    /// come from global memory; neighboring threads share them through the L1 cache.
    #[kernel]
    #[launch_contract(domain = 2, coordinates = u32, block = (32, 8, 1))]
    pub fn sobel(src: &[u8], mut out: DisjointSlice<u8, Index2D<W>>) {
        let (x, y) = (thread::index_2d_col(), thread::index_2d_row());
        let (xl, xr) = (x.saturating_sub(1), (x + 1).min(W - 1));
        let (yu, yd) = (y.saturating_sub(1), (y + 1).min(H - 1));
        let p = |r: usize, c: usize| src[r * W + c] as i32;

        if let Some(idx) = thread::index_2d::<W>()
            && let Some(o) = out.get_mut(idx)
        {
            let right = p(yu, xr) + 2 * p(y, xr) + p(yd, xr);
            let left = p(yu, xl) + 2 * p(y, xl) + p(yd, xl);
            let below = p(yd, xl) + 2 * p(yd, x) + p(yd, xr);
            let above = p(yu, xl) + 2 * p(yu, x) + p(yu, xr);
            let (gx, gy) = (right - left, below - above);

            let magnitude = sqrt_rn_f32((gx * gx + gy * gy) as f32);
            *o = (magnitude * 0.25 + 0.5).min(255.0) as u8;
        }
    }
}

/// The device buffers of 1 image in flight.
struct Buffers {
    rgb: DeviceBuffer<u8>,
    gray: DeviceBuffer<u8>,
    bins: DeviceBuffer<DeviceAtomicU32>,
    table: DeviceBuffer<u8>,
    equal: DeviceBuffer<u8>,
    rows: DeviceBuffer<u16>,
    blurred: DeviceBuffer<u8>,
    edges: DeviceBuffer<u8>,
}

impl Buffers {
    fn new(stream: &Arc<CudaStream>) -> Result<Self, Error> {
        Ok(Buffers {
            rgb: DeviceBuffer::zeroed(stream, 3 * PIXELS)?,
            gray: DeviceBuffer::zeroed(stream, PIXELS)?,
            bins: DeviceBuffer::<u32>::zeroed(stream, 256)?.cast_elem(),
            table: DeviceBuffer::zeroed(stream, 256)?,
            equal: DeviceBuffer::zeroed(stream, PIXELS)?,
            rows: DeviceBuffer::zeroed(stream, PIXELS)?,
            blurred: DeviceBuffer::zeroed(stream, PIXELS)?,
            edges: DeviceBuffer::zeroed(stream, PIXELS)?,
        })
    }
}

/// Name of each stage, and the bytes it must read and write per pixel.
const STAGES: [(&str, usize); 7] = [
    ("grayscale", 4),
    ("histogram", 1),
    ("table", 0),
    ("apply table", 2),
    ("blur rows", 3),
    ("blur columns", 3),
    ("sobel", 2),
];

/// Launches stage `k` of the pipeline on `stream`.
fn stage(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    b: &mut Buffers,
    k: usize,
) -> Result<(), Error> {
    let line = LaunchConfig1D::new((PIXELS / 256) as u32, 256, 0);
    let rows = LaunchConfig2D::new(((W / ROW_T) as u32, H as u32), (ROW_T as u32, 1), 0);
    let grid = ((W / COL_W) as u32, (H / COL_H) as u32);
    let tiles = LaunchConfig2D::new(grid, (COL_W as u32, COL_H as u32), 0);

    match k {
        0 => {
            let launch = module.prepare_grayscale(line)?;
            module.grayscale(stream, &launch, &b.rgb, &mut b.gray)?;
        }
        1 => {
            let launch = module.prepare_histogram(LaunchConfig1D::new(64, 256, 0))?;
            module.histogram(stream, &launch, &b.gray, &b.bins)?;
        }
        2 => {
            let launch = module.prepare_equalise_table(LaunchConfig1D::new(1, 256, 0))?;
            module.equalise_table(stream, &launch, &b.bins, &mut b.table)?;
        }
        3 => {
            let launch = module.prepare_apply_table(line)?;
            module.apply_table(stream, &launch, &b.gray, &b.table, &mut b.equal)?;
        }
        4 => {
            let launch = module.prepare_blur_rows(rows)?;
            module.blur_rows(stream, &launch, &b.equal, &mut b.rows)?;
        }
        5 => {
            let launch = module.prepare_blur_cols(tiles)?;
            module.blur_cols(stream, &launch, &b.rows, &mut b.blurred)?;
        }
        _ => {
            let launch = module.prepare_sobel(tiles)?;
            module.sobel(stream, &launch, &b.blurred, &mut b.edges)?;
        }
    }
    Ok(())
}

/// All 7 stages of 1 image, in order, on `stream`.
fn process(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    b: &mut Buffers,
) -> Result<(), Error> {
    for k in 0..STAGES.len() {
        stage(module, stream, b, k)?;
    }
    Ok(())
}

/// Average time of 100 runs of `launch`, in milliseconds, after 1 warm-up run.
fn time_ms(
    stream: &Arc<CudaStream>,
    mut launch: impl FnMut() -> Result<(), Error>,
) -> Result<f64, Error> {
    launch()?;
    stream.synchronize()?;

    let start = Instant::now();
    for _ in 0..100 {
        launch()?;
    }
    stream.synchronize()?;

    Ok(start.elapsed().as_secs_f64() * 1e3 / 100.0)
}

/// Uploads, processes and downloads `count` images, image i on stream i % streams.len().
/// Returns images per second.
fn batch(
    module: &kernels::LoadedModule,
    streams: &[Arc<CudaStream>],
    lanes: &mut [Buffers],
    inputs: &[PinnedHostBuffer<u8>],
    outputs: &mut [PinnedHostBuffer<u8>],
    count: usize,
) -> Result<f64, Error> {
    let start = Instant::now();

    for i in 0..count {
        let k = i % streams.len();
        let (stream, b) = (&streams[k], &mut lanes[k]);

        // SAFETY: no host buffer is read, written or dropped until every stream is
        // synchronized below, after all copies have completed.
        unsafe { b.rgb.copy_from_pinned_host_async(stream, &inputs[i % inputs.len()])? };
        process(module, stream, b)?;
        unsafe { b.edges.copy_to_pinned_host_async(stream, &mut outputs[k])? };
    }

    for stream in streams {
        stream.synchronize()?;
    }
    Ok(count as f64 / start.elapsed().as_secs_f64())
}

/// The 4 images of the pipeline that are kept: gray, equalized, blurred and edges.
struct Images {
    gray: Vec<u8>,
    equal: Vec<u8>,
    blurred: Vec<u8>,
    edges: Vec<u8>,
}

/// Runs `f(y, row)` for every row of `out`, the rows split into 1 band per CPU core.
fn par_rows<T: Send>(out: &mut [T], f: impl Fn(usize, &mut [T]) + Sync) {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let band = H.div_ceil(cores);

    std::thread::scope(|s| {
        for (k, chunk) in out.chunks_mut(band * W).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (r, row) in chunk.chunks_mut(W).enumerate() {
                    f(k * band + r, row);
                }
            });
        }
    });
}

/// The CPU version of `tap`.
fn tap(i: usize, n: usize) -> usize {
    if i < 2 { 0 } else { (i - 2).min(n - 1) }
}

/// The same 7 stages on the CPU, with the same integer arithmetic, rows on every core.
fn cpu_pipeline(rgb: &[u8]) -> Images {
    let mut gray = vec![0u8; PIXELS];
    par_rows(&mut gray, |y, row| {
        for (x, g) in row.iter_mut().enumerate() {
            let p = 3 * (y * W + x);
            let (r, gr, b) = (rgb[p] as u32, rgb[p + 1] as u32, rgb[p + 2] as u32);
            *g = ((77 * r + 150 * gr + 29 * b + 128) >> 8) as u8;
        }
    });

    let mut bins = [0u32; 256];
    for &g in &gray {
        bins[g as usize] += 1;
    }

    let min = bins.iter().copied().find(|&c| c > 0).unwrap_or(0);
    let span = (PIXELS as u32 - min) as u64;
    let mut table = [0u8; 256];
    let mut cdf = 0u32;
    for v in 0..256 {
        cdf += bins[v];
        let above = cdf.saturating_sub(min) as u64;
        let level = if span == 0 { v as u64 } else { (above * 255 + span / 2) / span };
        table[v] = level as u8;
    }

    let mut equal = vec![0u8; PIXELS];
    par_rows(&mut equal, |y, row| {
        for (x, e) in row.iter_mut().enumerate() {
            *e = table[gray[y * W + x] as usize];
        }
    });

    let mut rows = vec![0u32; PIXELS];
    par_rows(&mut rows, |y, row| {
        let t = |x: usize| equal[y * W + tap(x, W)] as u32;
        for (x, r) in row.iter_mut().enumerate() {
            *r = t(x) + 4 * t(x + 1) + 6 * t(x + 2) + 4 * t(x + 3) + t(x + 4);
        }
    });

    let mut blurred = vec![0u8; PIXELS];
    par_rows(&mut blurred, |y, row| {
        for (x, b) in row.iter_mut().enumerate() {
            let t = |k: usize| rows[tap(y + k, H) * W + x];
            let sum = t(0) + 4 * t(1) + 6 * t(2) + 4 * t(3) + t(4);
            *b = ((sum + 128) >> 8) as u8;
        }
    });

    let mut edges = vec![0u8; PIXELS];
    par_rows(&mut edges, |y, row| {
        let p = |r: usize, c: usize| blurred[r * W + c] as i32;
        let (yu, yd) = (y.saturating_sub(1), (y + 1).min(H - 1));
        for (x, e) in row.iter_mut().enumerate() {
            let (xl, xr) = (x.saturating_sub(1), (x + 1).min(W - 1));
            let right = p(yu, xr) + 2 * p(y, xr) + p(yd, xr);
            let left = p(yu, xl) + 2 * p(y, xl) + p(yd, xl);
            let below = p(yd, xl) + 2 * p(yd, x) + p(yd, xr);
            let above = p(yu, xl) + 2 * p(yu, x) + p(yu, xr);
            let (gx, gy) = (right - left, below - above);

            let magnitude = ((gx * gx + gy * gy) as f32).sqrt();
            *e = (magnitude * 0.25 + 0.5).min(255.0) as u8;
        }
    });

    Images { gray, equal, blurred, edges }
}

/// A synthetic 1920 x 1080 scene with low contrast: sky, sun, hills, a house, grain.
fn scene(seed: u32) -> Vec<u8> {
    let mut state = seed;
    let mut rgb = vec![0u8; 3 * PIXELS];
    let shift = seed as f32 * 0.05;

    for y in 0..H {
        for x in 0..W {
            let (u, v) = (x as f32 / W as f32, y as f32 / H as f32);
            let hill = 0.62 + 0.07 * (u * 9.0 + shift * 20.0).sin();
            let sun = (x as f32 - (0.72 + shift) * W as f32).hypot(y as f32 - 260.0);
            let house = (0.18..0.34).contains(&u) && (0.48..0.80).contains(&v);
            let window = (0.22..0.27).contains(&u) && (0.56..0.64).contains(&v);
            let roof = v < 0.48 && v > 0.36 && (u - 0.26).abs() < (v - 0.36) * 0.75;

            let base = if window {
                [92, 96, 88]
            } else if house {
                [150, 118, 104]
            } else if roof {
                [128, 92, 84]
            } else if sun < 72.0 {
                [176, 168, 120]
            } else if v > hill {
                [84, 112, 78]
            } else {
                let sky = (v * 40.0) as i32;
                [110 + sky, 126 + sky, 150]
            };

            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let grain = (state >> 28) as i32 - 8;
            let p = 3 * (y * W + x);
            for c in 0..3 {
                rgb[p + c] = (base[c] + grain).clamp(0, 255) as u8;
            }
        }
    }
    rgb
}

/// Writes a binary PGM (gray) or PPM (RGB) file.
fn write_image(path: &str, data: &[u8], channels: usize) -> Result<(), Error> {
    let mut file = BufWriter::new(File::create(path)?);
    let magic = if channels == 3 { "P6" } else { "P5" };
    write!(file, "{magic}\n{W} {H}\n255\n")?;
    file.write_all(data)?;
    Ok(())
}

/// Prints how many pixels of 2 images differ, and the first one that does.
fn report(label: &str, gpu: &[u8], cpu: &[u8]) -> usize {
    let wrong = gpu.iter().zip(cpu).filter(|(a, b)| a != b).count();
    print!("  {label:<10} {wrong:8}");

    if let Some(i) = (0..PIXELS).find(|&i| gpu[i] != cpu[i]) {
        let (x, y) = (i % W, i / W);
        print!("   first at ({x}, {y}): GPU {}, CPU {}", gpu[i], cpu[i]);
    }
    println!();
    wrong
}

/// FNV-1a hash of an image, to compare the outputs of 2 programs.
fn checksum(data: &[u8]) -> u32 {
    let mut h = 2_166_136_261u32;
    for &byte in data {
        h = (h ^ byte as u32).wrapping_mul(16_777_619);
    }
    h
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let streams = [ctx.new_stream()?, ctx.new_stream()?, ctx.new_stream()?];
    let stream = &streams[0];
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };

    let name = ctx.device_name()?;
    let sms = ctx.multiprocessor_count()?;
    println!("GPU: {name}, {sms} SMs   image {W} x {H} = {PIXELS} pixels");

    let scenes: Vec<Vec<u8>> = (0..4).map(scene).collect();
    let reference: Vec<Images> = scenes.iter().map(|rgb| cpu_pipeline(rgb)).collect();

    // 1. Each stage of 1 image, timed alone.
    let mut b = Buffers::new(stream)?;
    b.rgb = DeviceBuffer::from_host(stream, &scenes[0])?;
    let warm = Instant::now();
    while warm.elapsed().as_secs_f64() < 0.4 {
        time_ms(stream, || process(&module, stream, &mut b))?; // raise the GPU clock
    }

    println!("stage              ms     GB/s");
    let mut total = 0.0;
    for (k, (label, bytes)) in STAGES.iter().enumerate() {
        let ms = time_ms(stream, || stage(&module, stream, &mut b, k))?;
        let gb = (bytes * PIXELS) as f64 / 1e9 / (ms / 1e3);
        println!("  {label:<13} {ms:7.3}  {gb:7.1}");
        total += ms;
    }
    let all = time_ms(stream, || process(&module, stream, &mut b))?;
    println!("  sum of stages {total:7.3}");
    println!("  all 7 in turn {all:7.3}");

    // 2. The GPU's images against the CPU's, pixel by pixel.
    let gpu = Images {
        gray: b.gray.to_host_vec(stream)?,
        equal: b.equal.to_host_vec(stream)?,
        blurred: b.blurred.to_host_vec(stream)?,
        edges: b.edges.to_host_vec(stream)?,
    };
    let cpu = &reference[0];
    println!("image      pixels that differ from the CPU");
    let mut wrong = report("gray", &gpu.gray, &cpu.gray);
    wrong += report("equalised", &gpu.equal, &cpu.equal);
    wrong += report("blurred", &gpu.blurred, &cpu.blurred);
    wrong += report("edges", &gpu.edges, &cpu.edges);
    println!("edge image checksum: {:08x}", checksum(&gpu.edges));

    std::fs::create_dir_all("target/images")?;
    write_image("target/images/input.ppm", &scenes[0], 3)?;
    write_image("target/images/gray.pgm", &gpu.gray, 1)?;
    write_image("target/images/equalised.pgm", &gpu.equal, 1)?;
    write_image("target/images/blurred.pgm", &gpu.blurred, 1)?;
    write_image("target/images/edges.pgm", &gpu.edges, 1)?;

    // 3. The CPU on every core, 4 images.
    let start = Instant::now();
    for rgb in &scenes {
        cpu_pipeline(rgb);
    }
    let cpu_rate = scenes.len() as f64 / start.elapsed().as_secs_f64();
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());

    // 4. The copies alone, then a batch: copy in, 7 stages, copy out, on 1 to 3 streams.
    let mut inputs = Vec::new();
    for rgb in &scenes {
        inputs.push(PinnedHostBuffer::from_slice(&ctx, rgb)?);
    }
    let mut outputs = Vec::new();
    for _ in 0..3 {
        outputs.push(PinnedHostBuffer::from_slice(&ctx, &vec![0u8; PIXELS])?);
    }
    let mut lanes = vec![b, Buffers::new(stream)?, Buffers::new(stream)?];
    stream.synchronize()?;

    let lane = &mut lanes[0];
    let copy_in = time_ms(stream, || {
        // SAFETY: time_ms synchronizes the stream before the host buffers are used again.
        unsafe { lane.rgb.copy_from_pinned_host_async(stream, &inputs[0])? };
        Ok(())
    })?;
    let copy_out = time_ms(stream, || {
        // SAFETY: as above.
        unsafe { lane.edges.copy_to_pinned_host_async(stream, &mut outputs[0])? };
        Ok(())
    })?;
    let rate_in = (3 * PIXELS) as f64 / 1e9 / (copy_in / 1e3);
    let rate_out = PIXELS as f64 / 1e9 / (copy_out / 1e3);
    println!("copy in  {copy_in:6.3} ms  {rate_in:5.1} GB/s");
    println!("copy out {copy_out:6.3} ms  {rate_out:5.1} GB/s");

    batch(&module, &streams, &mut lanes, &inputs, &mut outputs, BATCH)?; // warm up
    println!("batch of {BATCH} images: copy in, 7 stages, copy out");
    let mut correct = true;
    for n in 1..=3 {
        let (some, used, out) = (&streams[..n], &mut lanes[..n], &mut outputs[..n]);
        let rate = batch(&module, some, used, &inputs, out, BATCH)?;
        println!("  {n} stream(s)          {rate:6.1} images/s");

        for k in 0..n {
            let last = (BATCH - 1 - (BATCH - 1 - k) % n) % inputs.len();
            correct &= outputs[k].as_slice() == reference[last].edges.as_slice();
        }
    }
    println!("  CPU, {cores} threads       {cpu_rate:6.1} images/s");

    let verdict = if wrong == 0 && correct { "PASSED" } else { "FAILED" };
    println!("batch outputs match the CPU: {}", if correct { "yes" } else { "no" });
    println!("{verdict}");
    Ok(())
}
