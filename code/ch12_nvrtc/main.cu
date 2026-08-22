// Chapter 12 - NVRTC runtime compilation + driver API launch, runnable.
// This is the book's compileToPtx + launchFromPtx (12.3-12.4) with every
// driver call checked, plus a small hybrid host program.
//
// Build: nvcc -arch=compute_60 main.cu -o nvrtc_example -lnvrtc -lcuda
//        (Jetson Orin: -arch=sm_87; A100: -arch=sm_80; H100: -arch=sm_90)
// Run:   ./nvrtc_example

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        const cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                         __FILE__, __LINE__, cudaGetErrorString(err));    \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

#define CHECK_DRV(call)                                                   \
    do {                                                                  \
        const CUresult res = (call);                                      \
        if (res != CUDA_SUCCESS) {                                        \
            const char* name = nullptr;                                   \
            cuGetErrorName(res, &name);                                   \
            std::fprintf(stderr, "CUDA driver error at %s:%d: %s\n",      \
                         __FILE__, __LINE__, name ? name : "unknown");    \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

// ---------------------------------------------------------------------------
// 12.3 Compile CUDA source to PTX at run time with NVRTC.
// ---------------------------------------------------------------------------
std::string compileToPtx(const char* source, const char* name)
{
    // 1. Create an NVRTC program from the source text.
    nvrtcProgram prog;
    nvrtcResult res = nvrtcCreateProgram(&prog, source, name, 0, nullptr,
                                         nullptr);
    if (res != NVRTC_SUCCESS) throw std::runtime_error("nvrtcCreateProgram");

    // 2. Compile. compute_60 PTX runs on any CUDA 12.x GPU (Pascal or newer)
    //    via the driver's JIT; use compute_87 on Jetson Orin, compute_80 on
    //    A100, compute_90 on H100 for native SASS.
    const char* options[] = {"-arch=compute_60", "-std=c++17"};
    res = nvrtcCompileProgram(prog, 2, options);

    // 3. On failure, fetch the compilation log and report it.
    if (res != NVRTC_SUCCESS)
    {
        size_t logSize = 0;
        nvrtcGetProgramLogSize(prog, &logSize);
        std::vector<char> log(logSize);
        nvrtcGetProgramLog(prog, log.data());
        nvrtcDestroyProgram(&prog);
        throw std::runtime_error(std::string("NVRTC compile failed:\n") +
                                 log.data());
    }

    // 4. Fetch the PTX text.
    size_t ptxSize = 0;
    nvrtcGetPTXSize(prog, &ptxSize);
    std::vector<char> ptx(ptxSize);
    nvrtcGetPTX(prog, ptx.data());
    nvrtcDestroyProgram(&prog);
    return std::string(ptx.data(), ptxSize);
}

// ---------------------------------------------------------------------------
// 12.4 Load PTX into the current driver context and launch a kernel.
// Uses the hybrid pattern from 12.6: cudaMalloc/cudaMemcpy for memory, the
// driver API for the JIT-compiled kernel.
// ---------------------------------------------------------------------------
void launchFromPtx(const std::string& ptx,
                   const float* d_a, const float* d_b, float* d_c, int n)
{
    // 1. Initialise the driver API (idempotent).
    CHECK_DRV(cuInit(0));

    // 2. Use the runtime's current context (the hybrid bridge, 12.6).
    CUcontext context = nullptr;
    CHECK_DRV(cuCtxGetCurrent(&context));
    if (context == nullptr)
    {
        CUdevice device;
        CHECK_DRV(cuDeviceGet(&device, 0));
        CHECK_DRV(cuDevicePrimaryCtxRetain(&context, device));
        CHECK_DRV(cuCtxSetCurrent(context));
    }

    // 3. Load the PTX into a MODULE.
    CUmodule module;
    CHECK_DRV(cuModuleLoadData(&module, ptx.c_str()));

    // 4. Get a handle to the kernel by NAME.
    CUfunction kernel;
    CHECK_DRV(cuModuleGetFunction(&kernel, module, "addVectors"));

    // 5. Package the kernel arguments: pointers TO the arguments.
    void* args[] = { &d_a, &d_b, &d_c, &n };

    // 6. Launch. The grid must cover all n elements: 256 threads per block,
    //    rounded up. (Hardcoding 1024 blocks would only cover 262,144
    //    elements and silently leave the rest untouched.)
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;
    CHECK_DRV(cuLaunchKernel(kernel,
                             blocks, 1, 1,        // grid
                             threads, 1, 1,       // block
                             0, nullptr,          // no dynamic shared, default stream
                             args, nullptr));     // arguments, no extra options

    CHECK_DRV(cuCtxSynchronize());

    // 7. Unload the module. The context belongs to the CUDA runtime (the
    //    hybrid pattern, 12.6): do NOT release/destroy it here.
    CHECK_DRV(cuModuleUnload(module));
}

int main()
{
    const int n = 1 << 20;

    const char* source =
        "extern \"C\" __global__ void addVectors(const float* a, "
        "const float* b, float* c, int n) {\n"
        "    const int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (i < n) c[i] = a[i] + b[i];\n"
        "}\n";

    std::string ptx;
    try
    {
        ptx = compileToPtx(source, "addVectors.cu");
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    std::printf("NVRTC: compiled %zu bytes of PTX\n", ptx.size());

    std::vector<float> h_a(n), h_b(n), h_c(n);
    for (int i = 0; i < n; ++i)
    {
        h_a[i] = 1.0f * i;
        h_b[i] = 2.0f * i;
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));

    launchFromPtx(ptx, d_a, d_b, d_c, n);

    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, n * sizeof(float),
                          cudaMemcpyDeviceToHost));

    double maxErr = 0.0;
    for (int i = 0; i < n; ++i)
        maxErr = std::max(maxErr, std::abs((double)h_c[i] - 3.0 * i));
    std::printf("NVRTC + driver: max error = %g\n", maxErr);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    if (maxErr > 0.0)
    {
        std::fprintf(stderr, "CHAPTER 12 TEST FAILED\n");
        return 1;
    }
    std::printf("Chapter 12 test PASSED (NVRTC + driver API)\n");
    return 0;
}
