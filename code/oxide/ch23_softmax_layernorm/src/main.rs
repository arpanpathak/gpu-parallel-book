//! Chapter 23: softmax and layer normalization of the rows of a matrix, in cuda-oxide.
//!
//!     cargo oxide run

use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, device, kernel, launch_contract};
use cuda_device::{thread, warp};

const BLOCK: u32 = 256;
const WARPS: usize = BLOCK as usize / 32;
const ROWS: usize = 4096;
const COLS: usize = 4096;
const PER_THREAD: usize = COLS / BLOCK as usize;
const EPS: f32 = 1e-5;

type Error = Box<dyn std::error::Error>;

#[cuda_module]
mod kernels {
    use super::*;

    #[device]
    fn warp_max(mut v: f32) -> f32 {
        let mut mask = 16;
        while mask > 0 {
            v = v.max(warp::shuffle_xor_f32(v, mask));
            mask /= 2;
        }
        v
    }

    #[device]
    fn warp_sum(mut v: f32) -> f32 {
        let mut mask = 16;
        while mask > 0 {
            v += warp::shuffle_xor_f32(v, mask);
            mask /= 2;
        }
        v
    }

    /// The maximum over the block, returned to every thread. `slots` holds 8 floats.
    #[device]
    fn block_max(v: f32, slots: *mut f32) -> f32 {
        let v = warp_max(v);
        let warp = thread::threadIdx_x() as usize / 32;
        if warp::lane_id() == 0 {
            // SAFETY: lane 0 of each warp writes its own slot, before the barrier.
            unsafe { *slots.add(warp) = v };
        }
        thread::sync_threads();

        // SAFETY: between the 2 barriers the slots are only read.
        let mut m = unsafe { *slots };
        let mut w = 1;
        while w < WARPS {
            m = m.max(unsafe { *slots.add(w) });
            w += 1;
        }
        thread::sync_threads();
        m
    }

    /// The sum over the block, returned to every thread. `slots` holds 8 floats.
    #[device]
    fn block_sum(v: f32, slots: *mut f32) -> f32 {
        let v = warp_sum(v);
        let warp = thread::threadIdx_x() as usize / 32;
        if warp::lane_id() == 0 {
            // SAFETY: lane 0 of each warp writes its own slot, before the barrier.
            unsafe { *slots.add(warp) = v };
        }
        thread::sync_threads();

        // SAFETY: between the 2 barriers the slots are only read.
        let mut s = 0.0;
        let mut w = 0;
        while w < WARPS {
            s += unsafe { *slots.add(w) };
            w += 1;
        }
        thread::sync_threads();
        s
    }

    /// Merges 2 softmax states (maximum m, sum of exp(x - m)); (-inf, 0) is empty.
    #[device]
    fn merge_softmax(m: f32, s: f32, m2: f32, s2: f32) -> (f32, f32) {
        let top = m.max(m2);
        if top == f32::NEG_INFINITY {
            return (top, 0.0);
        }
        (top, s * (m - top).exp() + s2 * (m2 - top).exp())
    }

    /// Merges 2 Welford states (count n, mean, sum of squared deviations q).
    #[device]
    fn merge_welford(a: (f32, f32, f32), b: (f32, f32, f32)) -> (f32, f32, f32) {
        let n = a.0 + b.0;
        if n == 0.0 {
            return a;
        }
        let delta = b.1 - a.1;
        let mean = a.1 + delta * b.0 / n;
        (n, mean, a.2 + b.2 + delta * delta * a.0 * b.0 / n)
    }

    /// Softmax without the maximum: exp(x) / sum of exp(x). Overflows for large x.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn softmax_naive(input: &[f32], mut out: DisjointSlice<f32>, cols: u32) {
        static mut SLOTS: SharedArray<f32, 8> = SharedArray::UNINIT;
        // SAFETY: the pointer is only used by the block helpers, between barriers.
        let slots = unsafe { (*&raw mut SLOTS).as_mut_ptr() };
        let base = thread::blockIdx_x() as usize * cols as usize;
        let tid = thread::threadIdx_x() as usize;

        let mut sum = 0.0;
        let mut j = tid;
        while j < cols as usize {
            sum += input[base + j].exp();
            j += BLOCK as usize;
        }
        let total = block_sum(sum, slots);

        let mut j = tid;
        while j < cols as usize {
            // SAFETY: block b writes only row b; thread t only columns t, t + 256, ...
            unsafe { *out.get_unchecked_mut(base + j) = input[base + j].exp() / total };
            j += BLOCK as usize;
        }
    }

    /// 3 passes over the row: the maximum, the sum of exp(x - max), the outputs.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn softmax_three_pass(input: &[f32], mut out: DisjointSlice<f32>, cols: u32) {
        static mut SLOTS: SharedArray<f32, 8> = SharedArray::UNINIT;
        // SAFETY: the pointer is only used by the block helpers, between barriers.
        let slots = unsafe { (*&raw mut SLOTS).as_mut_ptr() };
        let base = thread::blockIdx_x() as usize * cols as usize;
        let tid = thread::threadIdx_x() as usize;

        let mut m = f32::NEG_INFINITY;
        let mut j = tid;
        while j < cols as usize {
            m = m.max(input[base + j]);
            j += BLOCK as usize;
        }
        let top = block_max(m, slots);

        let mut sum = 0.0;
        let mut j = tid;
        while j < cols as usize {
            sum += (input[base + j] - top).exp();
            j += BLOCK as usize;
        }
        let total = block_sum(sum, slots);

        let mut j = tid;
        while j < cols as usize {
            let p = (input[base + j] - top).exp() / total;
            // SAFETY: block b writes only row b; thread t only columns t, t + 256, ...
            unsafe { *out.get_unchecked_mut(base + j) = p };
            j += BLOCK as usize;
        }
    }

    /// 2 passes: a running maximum and a rescaled running sum, then the outputs.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn softmax_online(input: &[f32], mut out: DisjointSlice<f32>, cols: u32) {
        static mut MAXES: SharedArray<f32, 8> = SharedArray::UNINIT;
        static mut SUMS: SharedArray<f32, 8> = SharedArray::UNINIT;
        let base = thread::blockIdx_x() as usize * cols as usize;
        let tid = thread::threadIdx_x() as usize;

        let mut m = f32::NEG_INFINITY;
        let mut s = 0.0;
        let mut j = tid;
        while j < cols as usize {
            let v = input[base + j];
            if v > m {
                s = s * (m - v).exp() + 1.0;
                m = v;
            } else {
                s += (v - m).exp();
            }
            j += BLOCK as usize;
        }

        let mut mask = 16;
        while mask > 0 {
            let m2 = warp::shuffle_xor_f32(m, mask);
            let s2 = warp::shuffle_xor_f32(s, mask);
            (m, s) = merge_softmax(m, s, m2, s2);
            mask /= 2;
        }
        let warp = tid / 32;
        if warp::lane_id() == 0 {
            // SAFETY: lane 0 of each warp writes its own slots, before the barrier.
            unsafe {
                MAXES[warp] = m;
                SUMS[warp] = s;
            }
        }
        thread::sync_threads();

        // SAFETY: after the barrier the slots are only read.
        let (mut top, mut total) = unsafe { (MAXES[0], SUMS[0]) };
        let mut w = 1;
        while w < WARPS {
            (top, total) = unsafe { merge_softmax(top, total, MAXES[w], SUMS[w]) };
            w += 1;
        }

        let mut j = tid;
        while j < cols as usize {
            let p = (input[base + j] - top).exp() / total;
            // SAFETY: block b writes only row b; thread t only columns t, t + 256, ...
            unsafe { *out.get_unchecked_mut(base + j) = p };
            j += BLOCK as usize;
        }
    }

    /// 1 read: each thread keeps its 16 values of the row in registers.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn softmax_fused(input: &[f32], mut out: DisjointSlice<f32>) {
        static mut SLOTS: SharedArray<f32, 8> = SharedArray::UNINIT;
        // SAFETY: the pointer is only used by the block helpers, between barriers.
        let slots = unsafe { (*&raw mut SLOTS).as_mut_ptr() };
        let base = thread::blockIdx_x() as usize * COLS;
        let tid = thread::threadIdx_x() as usize;

        let mut x = [0.0f32; PER_THREAD];
        let mut m = f32::NEG_INFINITY;
        let mut k = 0;
        #[unroll]
        while k < PER_THREAD {
            x[k] = input[base + k * BLOCK as usize + tid];
            m = m.max(x[k]);
            k += 1;
        }
        let top = block_max(m, slots);

        let mut sum = 0.0;
        let mut k = 0;
        #[unroll]
        while k < PER_THREAD {
            x[k] = (x[k] - top).exp();
            sum += x[k];
            k += 1;
        }
        let scale = 1.0 / block_sum(sum, slots);

        let mut k = 0;
        #[unroll]
        while k < PER_THREAD {
            let j = k * BLOCK as usize + tid;
            // SAFETY: block b writes only row b; thread t only columns t, t + 256, ...
            unsafe { *out.get_unchecked_mut(base + j) = x[k] * scale };
            k += 1;
        }
    }

    /// Layer norm with variance = mean of squares - square of mean, in 1 pass of sums.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn layernorm_naive(
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mut out: DisjointSlice<f32>,
        cols: u32,
    ) {
        static mut SLOTS: SharedArray<f32, 8> = SharedArray::UNINIT;
        // SAFETY: the pointer is only used by the block helpers, between barriers.
        let slots = unsafe { (*&raw mut SLOTS).as_mut_ptr() };
        let base = thread::blockIdx_x() as usize * cols as usize;
        let tid = thread::threadIdx_x() as usize;

        let mut sum = 0.0;
        let mut squares = 0.0;
        let mut j = tid;
        while j < cols as usize {
            let v = input[base + j];
            sum += v;
            squares += v * v;
            j += BLOCK as usize;
        }
        let mean = block_sum(sum, slots) / cols as f32;
        let var = block_sum(squares, slots) / cols as f32 - mean * mean;
        let rstd = 1.0 / (var + EPS).sqrt();

        let mut j = tid;
        while j < cols as usize {
            let y = (input[base + j] - mean) * rstd * gamma[j] + beta[j];
            // SAFETY: block b writes only row b; thread t only columns t, t + 256, ...
            unsafe { *out.get_unchecked_mut(base + j) = y };
            j += BLOCK as usize;
        }
    }

    /// Layer norm with Welford's running mean and squared deviations, merged per block.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn layernorm_welford(
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mut out: DisjointSlice<f32>,
        cols: u32,
    ) {
        static mut STATE: SharedArray<f32, 24> = SharedArray::UNINIT;
        let base = thread::blockIdx_x() as usize * cols as usize;
        let tid = thread::threadIdx_x() as usize;

        let mut st = (0.0f32, 0.0f32, 0.0f32);
        let mut j = tid;
        while j < cols as usize {
            let v = input[base + j];
            let n = st.0 + 1.0;
            let delta = v - st.1;
            let mean = st.1 + delta / n;
            st = (n, mean, st.2 + delta * (v - mean));
            j += BLOCK as usize;
        }

        let mut mask = 16;
        while mask > 0 {
            let other = (
                warp::shuffle_xor_f32(st.0, mask),
                warp::shuffle_xor_f32(st.1, mask),
                warp::shuffle_xor_f32(st.2, mask),
            );
            st = merge_welford(st, other);
            mask /= 2;
        }
        let warp = tid / 32;
        if warp::lane_id() == 0 {
            // SAFETY: lane 0 of each warp writes its own 3 slots, before the barrier.
            unsafe {
                STATE[3 * warp] = st.0;
                STATE[3 * warp + 1] = st.1;
                STATE[3 * warp + 2] = st.2;
            }
        }
        thread::sync_threads();

        // SAFETY: after the barrier the slots are only read.
        let mut all = unsafe { (STATE[0], STATE[1], STATE[2]) };
        let mut w = 1;
        while w < WARPS {
            let next = unsafe { (STATE[3 * w], STATE[3 * w + 1], STATE[3 * w + 2]) };
            all = merge_welford(all, next);
            w += 1;
        }
        let mean = all.1;
        let rstd = 1.0 / (all.2 / cols as f32 + EPS).sqrt();

        let mut j = tid;
        while j < cols as usize {
            let y = (input[base + j] - mean) * rstd * gamma[j] + beta[j];
            // SAFETY: block b writes only row b; thread t only columns t, t + 256, ...
            unsafe { *out.get_unchecked_mut(base + j) = y };
            j += BLOCK as usize;
        }
    }

    /// 1 read: the row stays in registers; the mean first, then the squared deviations.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn layernorm_fused(
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        static mut SLOTS: SharedArray<f32, 8> = SharedArray::UNINIT;
        // SAFETY: the pointer is only used by the block helpers, between barriers.
        let slots = unsafe { (*&raw mut SLOTS).as_mut_ptr() };
        let base = thread::blockIdx_x() as usize * COLS;
        let tid = thread::threadIdx_x() as usize;

        let mut x = [0.0f32; PER_THREAD];
        let mut sum = 0.0;
        let mut k = 0;
        #[unroll]
        while k < PER_THREAD {
            x[k] = input[base + k * BLOCK as usize + tid];
            sum += x[k];
            k += 1;
        }
        let mean = block_sum(sum, slots) / COLS as f32;

        let mut squares = 0.0;
        let mut k = 0;
        #[unroll]
        while k < PER_THREAD {
            x[k] -= mean;
            squares += x[k] * x[k];
            k += 1;
        }
        let rstd = 1.0 / (block_sum(squares, slots) / COLS as f32 + EPS).sqrt();

        let mut k = 0;
        #[unroll]
        while k < PER_THREAD {
            let j = k * BLOCK as usize + tid;
            // SAFETY: block b writes only row b; thread t only columns t, t + 256, ...
            unsafe { *out.get_unchecked_mut(base + j) = x[k] * rstd * gamma[j] + beta[j] };
            k += 1;
        }
    }
}

/// Time of 1 run of `launch`, in milliseconds: the median of 7 batches of 20 runs,
/// after 100 warm-up runs.
fn time_ms(
    stream: &Arc<CudaStream>,
    mut launch: impl FnMut() -> Result<(), Error>,
) -> Result<f64, Error> {
    for _ in 0..100 {
        launch()?;
    }
    stream.synchronize()?;

    let mut batches = Vec::new();
    for _ in 0..7 {
        let start = Instant::now();
        for _ in 0..20 {
            launch()?;
        }
        stream.synchronize()?;
        batches.push(start.elapsed().as_secs_f64() * 1e3 / 20.0);
    }

    batches.sort_by(f64::total_cmp);
    Ok(batches[3])
}

/// Uniform numbers in [low, high) from a small linear congruential generator.
fn uniform(n: usize, low: f32, high: f32, seed: u32) -> Vec<f32> {
    let mut state = seed;

    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            low + (high - low) * ((state >> 8) as f32 / 16_777_216.0)
        })
        .collect()
}

/// The softmax of each row, computed in f64.
fn softmax_reference(input: &[f32], cols: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(input.len());

    for row in input.chunks(cols) {
        let top = row.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v as f64));
        let total: f64 = row.iter().map(|&v| (v as f64 - top).exp()).sum();
        out.extend(row.iter().map(|&v| (v as f64 - top).exp() / total));
    }
    out
}

/// The layer norm of each row, computed in f64.
fn layernorm_reference(input: &[f32], gamma: &[f32], beta: &[f32]) -> Vec<f64> {
    let cols = gamma.len();
    let mut out = Vec::with_capacity(input.len());

    for row in input.chunks(cols) {
        let mean = row.iter().map(|&v| v as f64).sum::<f64>() / cols as f64;
        let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / cols as f64;
        let rstd = 1.0 / (var + EPS as f64).sqrt();

        for (j, &v) in row.iter().enumerate() {
            out.push((v as f64 - mean) * rstd * gamma[j] as f64 + beta[j] as f64);
        }
    }
    out
}

/// The largest difference between `gpu` and `exact`; NaN if any GPU value is NaN.
fn max_error(gpu: &[f32], exact: &[f64]) -> f64 {
    let mut worst = 0.0f64;

    for (&g, &e) in gpu.iter().zip(exact) {
        let diff = (g as f64 - e).abs();
        if diff.is_nan() {
            return f64::NAN;
        }
        worst = worst.max(diff);
    }
    worst
}

/// Prints a matrix of `cols` columns, 1 row per line.
fn print_matrix(name: &str, values: &[f32], cols: usize) {
    println!("  {name}");
    for row in values.chunks(cols) {
        let shown: Vec<String> = row.iter().map(|v| format!("{v:8.4}")).collect();
        println!("    {}", shown.join(" "));
    }
}

/// Prints 1 line of a results table: bytes per element, time, bandwidth and error.
fn report(name: &str, bytes: usize, ms: f64, error: f64) {
    let gb = (ROWS * COLS * bytes) as f64 / 1e9 / (ms / 1e3);
    println!("  {name:<11} {bytes:>10} {ms:7.3} {gb:7.1} {error:11.2e}");
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };
    let all_rows = LaunchConfig1D::new(ROWS as u32, BLOCK, 0);
    let header = "  kernel      bytes/elem      ms    GB/s   max error";

    // 1. The 2 x 4 example X, and X shifted by 100 and by 10,000.
    let x = [1.0f32, 2.0, 3.0, 4.0, 2.0, 4.0, 4.0, 5.0];
    let x100: Vec<f32> = x.iter().map(|v| v + 100.0).collect();
    let x10k: Vec<f32> = x.iter().map(|v| v + 10_000.0).collect();
    let ones = DeviceBuffer::from_host(&stream, &[1.0f32; 4])?;
    let zeros = DeviceBuffer::from_host(&stream, &[0.0f32; 4])?;
    let small = DeviceBuffer::from_host(&stream, &x)?;
    let shifted = DeviceBuffer::from_host(&stream, &x100)?;
    let far = DeviceBuffer::from_host(&stream, &x10k)?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, 8)?;

    let two_rows = LaunchConfig1D::new(2, BLOCK, 0);
    let naive_two = module.prepare_softmax_naive(two_rows)?;
    let three_two = module.prepare_softmax_three_pass(two_rows)?;
    let online_two = module.prepare_softmax_online(two_rows)?;
    let sq_two = module.prepare_layernorm_naive(two_rows)?;
    let welford_two = module.prepare_layernorm_welford(two_rows)?;

    println!("the 2 x 4 example X = [1 2 3 4; 2 4 4 5]");
    module.softmax_naive(&stream, &naive_two, &small, &mut out, 4)?;
    print_matrix("softmax without the maximum, X", &out.to_host_vec(&stream)?, 4);
    module.softmax_naive(&stream, &naive_two, &shifted, &mut out, 4)?;
    let naive_big = out.to_host_vec(&stream)?;
    print_matrix("softmax without the maximum, X + 100", &naive_big, 4);

    module.softmax_three_pass(&stream, &three_two, &small, &mut out, 4)?;
    let three_x = out.to_host_vec(&stream)?;
    print_matrix("softmax, 3 passes, X", &three_x, 4);
    module.softmax_three_pass(&stream, &three_two, &shifted, &mut out, 4)?;
    let three_big = out.to_host_vec(&stream)?;
    print_matrix("softmax, 3 passes, X + 100", &three_big, 4);
    module.softmax_online(&stream, &online_two, &small, &mut out, 4)?;
    let online_x = out.to_host_vec(&stream)?;
    print_matrix("softmax, online, X", &online_x, 4);

    module.layernorm_naive(&stream, &sq_two, &small, &ones, &zeros, &mut out, 4)?;
    let sq_x = out.to_host_vec(&stream)?;
    print_matrix("layer norm, sum of squares, X", &sq_x, 4);
    module.layernorm_naive(&stream, &sq_two, &far, &ones, &zeros, &mut out, 4)?;
    let sq_far = out.to_host_vec(&stream)?;
    print_matrix("layer norm, sum of squares, X + 10000", &sq_far, 4);
    module.layernorm_welford(&stream, &welford_two, &far, &ones, &zeros, &mut out, 4)?;
    let welford_far = out.to_host_vec(&stream)?;
    print_matrix("layer norm, Welford, X + 10000", &welford_far, 4);

    let p = softmax_reference(&x, 4);
    let y = layernorm_reference(&x, &[1.0; 4], &[0.0; 4]);
    let example_ok = max_error(&three_x, &p) < 1e-6
        && max_error(&three_big, &p) < 1e-6
        && max_error(&online_x, &p) < 1e-6
        && max_error(&sq_x, &y) < 1e-4
        && max_error(&welford_far, &y) < 1e-4
        && naive_big.iter().all(|v| v.is_nan())
        && sq_far.iter().all(|v| v.is_nan());

    // 2. Softmax of 4,096 rows of 4,096 logits in [-10, 10): 16 KiB per row.
    let n = ROWS * COLS;
    let cols = COLS as u32;
    let logits = uniform(n, -10.0, 10.0, 1);
    let input = DeviceBuffer::from_host(&stream, &logits)?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    let exact = softmax_reference(&logits, COLS);

    let three = module.prepare_softmax_three_pass(all_rows)?;
    let online = module.prepare_softmax_online(all_rows)?;
    let fused = module.prepare_softmax_fused(all_rows)?;

    println!("softmax of {ROWS} rows x {COLS} columns, median of 7 x 20 runs");
    println!("{header}");

    let ms = time_ms(&stream, || {
        Ok(module.softmax_three_pass(&stream, &three, &input, &mut out, cols)?)
    })?;
    let three_error = max_error(&out.to_host_vec(&stream)?, &exact);
    report("3 passes", 16, ms, three_error);

    let ms = time_ms(&stream, || {
        Ok(module.softmax_online(&stream, &online, &input, &mut out, cols)?)
    })?;
    let online_error = max_error(&out.to_host_vec(&stream)?, &exact);
    report("online", 12, ms, online_error);

    let ms = time_ms(&stream, || {
        Ok(module.softmax_fused(&stream, &fused, &input, &mut out)?)
    })?;
    let fused_error = max_error(&out.to_host_vec(&stream)?, &exact);
    report("fused", 8, ms, fused_error);

    // 3. The same logits as 256 rows of 65,536: 256 KiB per row.
    let long = 65_536;
    let long_rows = LaunchConfig1D::new((n / long) as u32, BLOCK, 0);
    let three_long = module.prepare_softmax_three_pass(long_rows)?;
    let online_long = module.prepare_softmax_online(long_rows)?;
    let wide = long as u32;
    let exact = softmax_reference(&logits, long);

    println!("softmax of {} rows x {long} columns, median of 7 x 20 runs", n / long);
    println!("{header}");

    let ms = time_ms(&stream, || {
        Ok(module.softmax_three_pass(&stream, &three_long, &input, &mut out, wide)?)
    })?;
    let three_long_error = max_error(&out.to_host_vec(&stream)?, &exact);
    report("3 passes", 16, ms, three_long_error);

    let ms = time_ms(&stream, || {
        Ok(module.softmax_online(&stream, &online_long, &input, &mut out, wide)?)
    })?;
    let online_long_error = max_error(&out.to_host_vec(&stream)?, &exact);
    report("online", 12, ms, online_long_error);

    // 4. Layer norm of 4,096 rows of values 1,000 + [-1, 1).
    let values = uniform(n, 999.0, 1001.0, 2);
    let gamma: Vec<f32> = (0..COLS).map(|j| 0.5 + 0.25 * (j % 3) as f32).collect();
    let beta: Vec<f32> = (0..COLS).map(|j| 0.1 * (j % 5) as f32 - 0.2).collect();
    let input = DeviceBuffer::from_host(&stream, &values)?;
    let g = DeviceBuffer::from_host(&stream, &gamma)?;
    let b = DeviceBuffer::from_host(&stream, &beta)?;
    let exact = layernorm_reference(&values, &gamma, &beta);

    let naive = module.prepare_layernorm_naive(all_rows)?;
    let welford = module.prepare_layernorm_welford(all_rows)?;
    let fused = module.prepare_layernorm_fused(all_rows)?;

    println!("layer norm of {ROWS} rows x {COLS} columns, values 1,000 + [-1, 1)");
    println!("{header}");

    let ms = time_ms(&stream, || {
        Ok(module.layernorm_naive(&stream, &naive, &input, &g, &b, &mut out, cols)?)
    })?;
    let naive_error = max_error(&out.to_host_vec(&stream)?, &exact);
    report("sum, sum x2", 12, ms, naive_error);

    let ms = time_ms(&stream, || {
        Ok(module.layernorm_welford(&stream, &welford, &input, &g, &b, &mut out, cols)?)
    })?;
    let welford_error = max_error(&out.to_host_vec(&stream)?, &exact);
    report("Welford", 12, ms, welford_error);

    let ms = time_ms(&stream, || {
        Ok(module.layernorm_fused(&stream, &fused, &input, &g, &b, &mut out)?)
    })?;
    let norm_error = max_error(&out.to_host_vec(&stream)?, &exact);
    report("fused", 8, ms, norm_error);

    // 5. Softmax within 1e-6, layer norm within 1e-3; the sum of squares must be off.
    let short = [three_error, online_error, fused_error];
    let long = [three_long_error, online_long_error];
    let softmax_ok = short.iter().chain(&long).all(|&e| e < 1e-6);
    let layernorm_ok = welford_error < 1e-3 && norm_error < 1e-3 && naive_error > 0.1;
    let ok = example_ok && softmax_ok && layernorm_ok;
    println!("{}", if ok { "PASSED" } else { "FAILED" });

    Ok(())
}
