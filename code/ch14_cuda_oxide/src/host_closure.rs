// Chapter 14 - CUDA-Oxide: the generic #[kernel] map example.
// Verbatim from the book (14.4). Build with: cargo oxide run host_closure

// Single source file: device AND host code together.
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

// ---------------------------------------------------------------------------
// Device side: a generic kernel that applies any function to each element.
// F can be a closure with captures - rustc monomorphises it to a concrete
// type at compile time, exactly like a C++ template instantiation.
// ---------------------------------------------------------------------------
#[cuda_module]
mod kernels {
    use super::*;

    // The #[kernel] attribute tells the backend to compile this function
    // to PTX. It is the Rust equivalent of __global__.
    #[kernel]
    pub fn map<T: Copy, F: Fn(T) -> T + Copy>(f: F, input: &[T],
                                              mut out: DisjointSlice<T>) {
        let idx = thread::index_1d();        // threadIdx/blockIdx, fused
        let i = idx.get();                    // the global linear index
        // DisjointSlice guarantees this thread's slot is exclusive:
        // two threads can never get_mut the same element.
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = f(input[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Host side: allocate, load the module, launch.
// ---------------------------------------------------------------------------
fn main() {
    let ctx = CudaContext::new(0).unwrap();       // open GPU 0
    let stream = ctx.default_stream();

    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let input  = DeviceBuffer::from_host(&stream, &data).unwrap();
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, 1024).unwrap();

    // Load the module: #[cuda_module] embeds the compiled PTX in the binary
    // and generates a typed module.map::<f32, _>(...) launch method.
    let module = kernels::load(&ctx).unwrap();

    // Launch with a closure. `factor` is captured and passed to the GPU
    // automatically (scalarised into a kernel parameter).
    let factor = 2.5f32;
    // SAFETY: this raw configuration is fully 1-D, matches index_1d(), and
    // launches one thread per output element. A launch contract can move
    // this proof into the generated safe API.
    unsafe {
        module.map::<f32, _>(
            &stream,
            LaunchConfig::for_num_elems(1024),
            move |x: f32| x * factor,
            &input,
            &mut output,
        )
    }
    .unwrap();

    let result = output.to_host_vec(&stream).unwrap();
    assert!((result[1] - 2.5).abs() < 1e-5);
}
