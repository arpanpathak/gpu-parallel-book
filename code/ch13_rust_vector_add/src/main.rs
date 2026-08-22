// Chapter 13 - Rust host driving a CUDA kernel via cudarc.
// Verbatim from the book (13.4). Requires vector_add.ptx in the working
// directory (the file is committed next to Cargo.toml).

// main.rs - Rust host driving the vector_add kernel via cudarc.
// Requires: CUDA toolkit installed (for the driver and nvrtc), and the
// vector_add.ptx file in the working directory (or embedded; see 13.5).

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- Device handle -----------------------------------------------------
    // CudaDevice::new(0) opens the first GPU, initialising the driver and
    // creating the CUDA context. It returns Result: no GPU -> Err here,
    // reported as a typed error instead of a crash.
    let dev = CudaDevice::new(0)?;

    // --- Problem size ------------------------------------------------------
    const N: usize = 1 << 20;              // 1,048,576 elements
    let n32: i32 = N as i32;               // kernel expects a 32-bit int

    // --- Host data ----------------------------------------------------------
    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| 2.0 * i as f32).collect();

    // --- Device allocation ---------------------------------------------------
    // alloc_zeros allocates device memory and zero-initialises it. The
    // returned CudaSlice<f32> OWNS the allocation: dropping it frees it.
    // No cudaFree call anywhere in this program.
    let mut d_a: CudaSlice<f32> = dev.alloc_zeros::<f32>(N)?;
    let mut d_b: CudaSlice<f32> = dev.alloc_zeros::<f32>(N)?;
    let mut d_c: CudaSlice<f32> = dev.alloc_zeros::<f32>(N)?;

    // --- Host -> device copies ----------------------------------------------
    // htod_copy_into takes ownership of the host Vec (the copy is queued on
    // the device's stream, so no other code can mutate the buffer while it
    // is in flight - the ownership story again). We pass clones so the
    // originals remain usable for verification below. The '&mut' on the
    // destination is the ownership language: the copy mutates the device
    // buffer, and Rust requires exclusive access to do so.
    dev.htod_copy_into(a.clone(), &mut d_a)?;
    dev.htod_copy_into(b.clone(), &mut d_b)?;

    // --- Load the PTX and fetch the kernel handle ---------------------------
    // load_ptx loads the module and registers the named kernel. The PTX is
    // wrapped in a Ptx (Ptx::from_file for a path, Ptx::from_src for an
    // embedded string - see 13.5). Errors (missing file, missing symbol)
    // surface as Results.
    let module = "vector_add";
    dev.load_ptx(Ptx::from_file("vector_add.ptx"), module, &["vector_add"])?;
    let f = dev.get_func(module, "vector_add")
        .ok_or("kernel 'vector_add' not found in the loaded module")?;

    // --- Launch --------------------------------------------------------------
    // The launch is marked unsafe: the configuration (grid/block shape) and
    // the argument tuple must match the kernel's real signature and index
    // space. cudarc checks argument arity and types at the type level but
    // cannot verify the kernel's internal assumptions - hence SAFETY.
    //
    // SAFETY: the grid covers exactly N threads (LaunchConfig::for_num_elems
    // rounds up to whole warps), the kernel guards with `if (i < n)`, and the
    // argument tuple (a, b, &mut c, n) matches the extern "C" signature.
    unsafe {
        f.launch(LaunchConfig::for_num_elems(N as u32),
                 (&d_a, &d_b, &mut d_c, n32))
    }?;

    // --- Device -> host copy -------------------------------------------------
    // dtoh_sync_copy blocks until the stream's work completes and copies the
    // result back. The trailing ? propagates any device error encountered.
    let c: Vec<f32> = dev.dtoh_sync_copy(&d_c)?;

    // --- Verify ---------------------------------------------------------------
    let max_err = c.iter().zip(a.iter().zip(b.iter()))
        .map(|(c, (a, b))| (c - (a + b)).abs())
        .fold(0.0f32, f32::max);
    println!("max error = {max_err}");

    // d_a, d_b, d_c are dropped here; the allocations are freed by their
    // destructors. The device handle's context is cleaned up on drop.
    Ok(())
}
