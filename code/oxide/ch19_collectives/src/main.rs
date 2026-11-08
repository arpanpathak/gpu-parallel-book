//! Chapter 19 in Rust: ring and tree all-reduce, simulated on 1 GPU.
//! Each rank is a slice of 1 buffer in the memory of 1 GPU. No network is measured.
//!
//!     cargo oxide run

use cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT as TIMING;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, LaunchConfig1D};
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_contract, thread};

const BLOCK: u32 = 256;

type Error = Box<dyn std::error::Error>;

/// Milliseconds per all-reduce, and whether the result was exact.
type Timed = Result<(f32, bool), Error>;

#[cuda_module]
mod kernels {
    use super::*;

    /// `per` blocks for each transfer of the step; block b works on transfer b / per.
    /// `plan` holds (from, to) offset pairs; the step's pairs start at pair `first`.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn step(
        mut all: DisjointSlice<f32>,
        plan: &[u32],
        first: u32,
        len: u32,
        per: u32,
        add: u32,
    ) {
        let b = thread::blockIdx_x();
        let t = (first + b / per) as usize;
        let from = plan[2 * t];
        let to = plan[2 * t + 1];
        let stride = per * BLOCK;
        let mut i = (b % per) * BLOCK + thread::threadIdx_x();

        while i < len {
            // SAFETY: within 1 step no element is both read and written, and no 2
            // threads write the same element, so these accesses cannot race.
            unsafe {
                let v = *all.get_unchecked_mut((from + i) as usize);
                let d = all.get_unchecked_mut((to + i) as usize);
                *d = if add != 0 { *d + v } else { v };
            }
            i += stride;
        }
    }

    /// Rank r, element i starts as i % 16 + r, or as r + 1 in the demo.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn fill(mut all: DisjointSlice<f32>, n: u32, demo: u32) {
        let g = thread::index_1d();
        let r = g.get() as u32 / n;
        let i = g.get() as u32 % n;

        if let Some(x) = all.get_mut(g) {
            *x = if demo != 0 { (r + 1) as f32 } else { (i % 16 + r) as f32 };
        }
    }
}

/// 1 step of a collective: `count` transfers of `len` floats that run at the same time.
struct Step {
    first: usize,
    count: usize,
    len: usize,
    add: bool,
}

/// The (from, to) offsets of every transfer, and the steps that group them.
#[derive(Default)]
struct Plan {
    pairs: Vec<u32>,
    steps: Vec<Step>,
}

impl Plan {
    fn add_step(&mut self, sends: &[(usize, usize)], len: usize, add: bool) {
        let first = self.pairs.len() / 2;

        for &(from, to) in sends {
            self.pairs.push(from as u32);
            self.pairs.push(to as u32);
        }
        self.steps.push(Step { first, count: sends.len(), len, add });
    }

    /// Bytes sent by the rank that sends the most; rank r owns offsets [r n, r n + n).
    fn busiest_rank(&self, k: usize, n: usize) -> usize {
        let mut sent = vec![0; k];

        for st in &self.steps {
            for t in st.first..st.first + st.count {
                sent[self.pairs[2 * t] as usize / n] += st.len * 4;
            }
        }

        sent.into_iter().max().unwrap_or(0)
    }
}

/// Ring: reduce-scatter in k - 1 steps, then all-gather in k - 1 steps.
fn ring(k: usize, n: usize) -> Plan {
    let mut plan = Plan::default();
    let chunk = n / k;
    let send = |r: usize, c: usize| (r * n + c * chunk, (r + 1) % k * n + c * chunk);

    for s in 0..k - 1 {
        let sends: Vec<_> = (0..k).map(|r| send(r, (r + k - s) % k)).collect();
        plan.add_step(&sends, chunk, true);
    }

    for s in 0..k - 1 {
        let sends: Vec<_> = (0..k).map(|r| send(r, (r + 1 + k - s) % k)).collect();
        plan.add_step(&sends, chunk, false);
    }

    plan
}

/// Binomial tree, k a power of 2: sums climb to rank 0, then the total comes back down.
fn tree(k: usize, n: usize) -> Plan {
    let mut plan = Plan::default();
    let mut d = 1;

    while d < k {
        let sends: Vec<_> = (d..k).step_by(2 * d).map(|r| (r * n, (r - d) * n)).collect();
        plan.add_step(&sends, n, true);
        d *= 2;
    }

    d = k / 2;
    while d >= 1 {
        let sends: Vec<_> = (0..k).step_by(2 * d).map(|r| (r * n, (r + d) * n)).collect();
        plan.add_step(&sends, n, false);
        d /= 2;
    }

    plan
}

/// The ranks' buffer, and the stream and module that run the kernels on it.
struct Gpu<'a> {
    stream: &'a CudaStream,
    module: &'a kernels::LoadedModule,
    all: DeviceBuffer<f32>,
}

impl Gpu<'_> {
    fn fill(&mut self, k: usize, n: usize, demo: bool) -> Result<(), Error> {
        let blocks = (k * n).div_ceil(BLOCK as usize) as u32;
        let launch = self.module.prepare_fill(LaunchConfig1D::new(blocks, BLOCK, 0))?;
        let all = &mut self.all;
        Ok(self.module.fill(self.stream, &launch, all, n as u32, demo as u32)?)
    }

    fn run(&mut self, steps: &[Step], pairs: &DeviceBuffer<u32>) -> Result<(), Error> {
        for st in steps {
            let per = st.len.div_ceil(BLOCK as usize).min((1024 / st.count).max(1)) as u32;
            let config = LaunchConfig1D::new(st.count as u32 * per, BLOCK, 0);
            let launch = self.module.prepare_step(config)?;

            let (first, len, add) = (st.first as u32, st.len as u32, st.add as u32);
            let all = &mut self.all;
            self.module.step(self.stream, &launch, all, pairs, first, len, per, add)?;
        }

        Ok(())
    }

    fn print_ranks(&self, k: usize, n: usize, when: &str) -> Result<(), DriverError> {
        let h = self.all.to_host_vec(self.stream)?;

        println!("{when}");
        for r in 0..k {
            let rank = &h[r * n..r * n + n];
            let row: String = rank.iter().map(|v| format!("{v:3}")).collect();
            println!("  rank {r}  {row}");
        }

        Ok(())
    }

    /// True if every rank holds the exact sum of all ranks' starting values.
    fn correct(&self, k: usize, n: usize) -> Result<bool, DriverError> {
        let h = self.all.to_host_vec(self.stream)?;

        for i in 0..n {
            let sum: f32 = (0..k).map(|r| (i % 16 + r) as f32).sum();

            if (0..k).any(|r| h[r * n + i] != sum) {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Average time of 1 all-reduce in milliseconds, over `reps` runs from fresh data.
    fn time(&mut self, plan: &Plan, k: usize, n: usize, reps: usize) -> Timed {
        let pairs = DeviceBuffer::from_host(self.stream, &plan.pairs)?;
        let mut total = 0.0;

        for _ in 0..reps {
            self.fill(k, n, false)?;

            let start = self.stream.record_event(Some(TIMING))?;
            self.run(&plan.steps, &pairs)?;
            let stop = self.stream.record_event(Some(TIMING))?;

            stop.synchronize()?;
            total += start.elapsed_ms(&stop)?;
        }

        Ok((total / reps as f32, self.correct(k, n)?))
    }
}

/// 4 ranks of 8 floats, printed after every step of the ring.
fn demo(gpu: &mut Gpu) -> Result<(), Error> {
    let (k, n) = (4, 8);
    let plan = ring(k, n);
    let pairs = DeviceBuffer::from_host(gpu.stream, &plan.pairs)?;

    let title = "ring all-reduce, 4 ranks of 8 floats: rank r starts with r + 1";
    gpu.fill(k, n, true)?;
    gpu.print_ranks(k, n, title)?;

    for s in 0..plan.steps.len() {
        gpu.run(&plan.steps[s..s + 1], &pairs)?;

        let phase = if s < k - 1 { "reduce-scatter" } else { "all-gather" };
        gpu.print_ranks(k, n, &format!("after step {} ({phase})", s + 1))?;
    }

    Ok(())
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };

    let k = 8;
    let max_n = 4 << 20; // 4 Mi floats, 16 MiB per rank
    let all = DeviceBuffer::<f32>::zeroed(&stream, k * max_n)?;
    let mut gpu = Gpu { stream: &stream, module: &module, all };

    demo(&mut gpu)?;

    for _ in 0..40 {
        gpu.time(&ring(k, max_n), k, max_n, 1)?; // about 0.4 s: let the clocks rise
    }

    let message = max_n * 4;
    let (r, t) = (ring(k, max_n), tree(k, max_n));
    let ring_share = r.busiest_rank(k, max_n) as f64 / message as f64;
    let tree_share = t.busiest_rank(k, max_n) as f64 / message as f64;

    println!("\nall-reduce on {k} simulated ranks");
    println!("          steps   busiest rank sends");
    println!("  ring    {:4}     {ring_share:.2} x the message", r.steps.len());
    println!("  tree    {:4}     {tree_share:.2} x the message", t.steps.len());

    println!("\n  message     ring        tree");
    let mut all_ok = true;
    let mut n = 1024;

    while n <= max_n {
        let (ring_ms, ring_ok) = gpu.time(&ring(k, n), k, n, 50)?;
        let (tree_ms, tree_ok) = gpu.time(&tree(k, n), k, n, 50)?;
        all_ok = all_ok && ring_ok && tree_ok;

        let kib = (n * 4) as f64 / 1024.0;
        println!("  {kib:6.0} KiB  {ring_ms:7.3} ms  {tree_ms:7.3} ms");
        n *= 16;
    }

    let verdict = if all_ok { "PASSED" } else { "FAILED" };
    println!("every rank holds the exact sum: {verdict}");
    Ok(())
}
