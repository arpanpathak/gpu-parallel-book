//! Chapter 12 in Rust: a kernel compiled by NVRTC while the program runs, loaded and
//! launched through cuda-core, the crate that wraps the CUDA driver API.
//!
//!     cargo oxide run

use std::ffi::{CString, c_char, c_int, c_void};
use std::time::Instant;

use cuda_core::launch_kernel_on_stream;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_contract, thread};

const N: usize = 1 << 24; // 16,777,216 values of x
const DEGREE: usize = 32; // p(x) = c[0] + c[1] x + ... + c[32] x^32
const BLOCK: u32 = 256;

type Error = Box<dyn std::error::Error>;

#[cuda_module]
mod kernels {
    use super::*;

    /// The degree and the coefficients arrive at run time, in `c`.
    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn poly_generic(x: &[f32], c: &[f32], mut y: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();

        if let Some(out) = y.get_mut(idx) {
            let v = x[i];
            let mut k = c.len() - 1;
            let mut p = c[k];
            while k > 0 {
                k -= 1;
                p = p.mul_add(v, c[k]);
            }
            *out = p;
        }
    }
}

/// An NVRTC program handle, and the 7 NVRTC functions this program calls.
type Program = *mut c_void;

#[link(name = "nvrtc")]
unsafe extern "C" {
    fn nvrtcCreateProgram(
        prog: *mut Program,
        src: *const c_char,
        name: *const c_char,
        num_headers: c_int,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> c_int;
    fn nvrtcCompileProgram(
        prog: Program,
        num_options: c_int,
        options: *const *const c_char,
    ) -> c_int;
    fn nvrtcGetProgramLogSize(prog: Program, size: *mut usize) -> c_int;
    fn nvrtcGetProgramLog(prog: Program, log: *mut c_char) -> c_int;
    fn nvrtcGetCUBINSize(prog: Program, size: *mut usize) -> c_int;
    fn nvrtcGetCUBIN(prog: Program, cubin: *mut c_char) -> c_int;
    fn nvrtcDestroyProgram(prog: *mut Program) -> c_int;
}

/// The fixed first lines of the generated kernel.
const HEAD: &str = r#"extern "C" __global__ void poly(const float* x, float* y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    float v = x[i];
"#;

/// CUDA C++ source text for the polynomial, with its coefficients written in.
fn specialised_source(c: &[f32]) -> String {
    let mut src = String::from(HEAD);

    src += &format!("    float p = {:.9e}f;\n", c[DEGREE]);
    for k in (0..DEGREE).rev() {
        src += &format!("    p = p * v + {:.9e}f;\n", c[k]);
    }

    src += "    y[i] = p;\n}\n";
    src
}

/// Compile `source` with NVRTC into a cubin for the GPU `arch`, such as "sm_87".
fn compile(source: &str, arch: &str) -> Result<Vec<u8>, Error> {
    let src = CString::new(source)?;
    let name = CString::new("poly.cu")?;
    let option = CString::new(format!("--gpu-architecture={arch}"))?;
    let options = [option.as_ptr()];
    let none = std::ptr::null();

    let mut prog: Program = std::ptr::null_mut();
    // SAFETY: every pointer refers to a live, NUL-terminated string, to `prog`, or to a
    // buffer of the size NVRTC reported.
    unsafe {
        let (src, name) = (src.as_ptr(), name.as_ptr());
        ok(nvrtcCreateProgram(&mut prog, src, name, 0, none, none), "nvrtcCreateProgram")?;

        if nvrtcCompileProgram(prog, 1, options.as_ptr()) != 0 {
            let mut size = 0;
            ok(nvrtcGetProgramLogSize(prog, &mut size), "nvrtcGetProgramLogSize")?;
            let mut log = vec![0u8; size];
            ok(nvrtcGetProgramLog(prog, log.as_mut_ptr().cast()), "nvrtcGetProgramLog")?;
            nvrtcDestroyProgram(&mut prog);
            log.pop(); // the terminating NUL
            return Err(String::from_utf8_lossy(&log).into());
        }

        let mut size = 0;
        ok(nvrtcGetCUBINSize(prog, &mut size), "nvrtcGetCUBINSize")?;
        let mut cubin = vec![0u8; size];
        ok(nvrtcGetCUBIN(prog, cubin.as_mut_ptr().cast()), "nvrtcGetCUBIN")?;
        ok(nvrtcDestroyProgram(&mut prog), "nvrtcDestroyProgram")?;
        Ok(cubin)
    }
}

/// NVRTC returns 0 for success and an error number otherwise.
fn ok(result: c_int, call: &str) -> Result<(), Error> {
    if result == 0 {
        Ok(())
    } else {
        Err(format!("{call} failed with NVRTC error {result}").into())
    }
}

/// Average time of `runs` runs of `launch`, in milliseconds, after 1 warm-up run.
fn time_ms<F>(stream: &CudaStream, runs: u32, mut launch: F) -> Result<f64, Error>
where
    F: FnMut() -> Result<(), Error>,
{
    launch()?;
    stream.synchronize()?;

    let start = Instant::now();
    for _ in 0..runs {
        launch()?;
    }
    stream.synchronize()?;

    Ok(start.elapsed().as_secs_f64() * 1e3 / runs as f64)
}

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let (major, minor) = ctx.compute_capability()?;
    let arch = format!("sm_{major}{minor}");
    let name = ctx.device_name()?;
    println!("GPU: {name}, compute capability {major}.{minor}, target {arch}");

    let c_host: Vec<f32> = (0..=DEGREE).map(|k| 1.0 / (k + 1) as f32).collect();
    let x_host: Vec<f32> = (0..N).map(|i| 2.0 * i as f32 / N as f32 - 1.0).collect();

    let source = specialised_source(&c_host);
    let start = Instant::now();
    let cubin = compile(&source, &arch)?;
    let compile_ms = start.elapsed().as_secs_f64() * 1e3;
    println!("NVRTC, {} bytes of source", source.len());
    println!("  compile to cubin   {compile_ms:8.2} ms  {:6} bytes", cubin.len());

    let start = Instant::now();
    let special = ctx.load_module_from_image(&cubin)?;
    let poly = special.load_function("poly")?;
    let load_ms = start.elapsed().as_secs_f64() * 1e3;
    println!("  load cubin         {load_ms:8.2} ms");

    // SAFETY: the loaded code was compiled from the `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };

    let x = DeviceBuffer::from_host(&stream, &x_host)?;
    let c = DeviceBuffer::from_host(&stream, &c_host)?;
    let mut y = DeviceBuffer::<f32>::zeroed(&stream, N)?;

    let blocks = (N as u32).div_ceil(BLOCK);
    let config = module.prepare_poly_generic(LaunchConfig1D::new(blocks, BLOCK, 0))?;

    let mut x_ptr = x.cu_deviceptr();
    let mut y_ptr = y.cu_deviceptr();
    let mut n = N as c_int;
    let mut args: [*mut c_void; 3] = [
        (&raw mut x_ptr).cast(),
        (&raw mut y_ptr).cast(),
        (&raw mut n).cast(),
    ];

    let mut generic = || -> Result<(), Error> {
        Ok(module.poly_generic(&stream, &config, &x, &c, &mut y)?)
    };
    let mut specialised = || -> Result<(), Error> {
        let grid = (blocks, 1, 1);
        let block = (BLOCK, 1, 1);
        // SAFETY: `poly` takes (const float*, float*, int); args points to 2 device
        // pointers and an int, all alive, and the grid covers N exactly once.
        unsafe { launch_kernel_on_stream(&poly, grid, block, 0, &stream, &mut args)? };
        Ok(())
    };

    for _ in 0..20 {
        time_ms(&stream, 5, &mut generic)?; // warm up: clocks rise under sustained load
    }

    let flop = 2.0 * DEGREE as f64 * N as f64;
    println!("p(x) for {N} values of x");

    let ms = time_ms(&stream, 20, &mut generic)?;
    let y_generic = y.to_host_vec(&stream)?;
    println!("  generic, typed launch      {ms:7.3} ms  {:6.1} GFLOP/s", flop / ms / 1e6);

    let ms = time_ms(&stream, 20, &mut specialised)?;
    let y_special = y.to_host_vec(&stream)?;
    println!("  specialised, raw launch    {ms:7.3} ms  {:6.1} GFLOP/s", flop / ms / 1e6);

    let mut worst: f64 = 0.0;
    for i in (0..N).step_by(4099) {
        let mut exact = 0.0;
        for k in (0..=DEGREE).rev() {
            exact = exact * x_host[i] as f64 + c_host[k] as f64;
        }
        worst = worst.max((exact - y_special[i] as f64).abs());
    }

    let mut differ: f32 = 0.0;
    for i in 0..N {
        differ = differ.max((y_generic[i] - y_special[i]).abs());
    }
    println!("  max error {worst:.1e}, generic vs specialised {differ:.1e}");

    println!("a source with a mistake");
    let broken = source.replace("y[i] = p", "y[i] = q");
    if let Err(log) = compile(&broken, &arch) {
        print!("  compile failed, log:\n{log}");
    }

    if worst < 1e-4 && differ < 1e-5 {
        println!("PASSED");
        Ok(())
    } else {
        Err("FAILED".into())
    }
}
