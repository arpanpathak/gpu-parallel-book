// Chapter 12: a kernel written, compiled and launched while the program runs.
// Build: nvcc -arch=native nvrtc.cu -o nvrtc -lnvrtc -lcuda

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

const int N = 1 << 24;                        // 16,777,216 values of x
const int DEGREE = 32;                        // p(x) = c[0] + c[1] x + ... + c[32] x^32

void check(cudaError_t err)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA runtime error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

void check(CUresult err)
{
    if (err != CUDA_SUCCESS) {
        const char* name = "unknown";
        cuGetErrorName(err, &name);
        std::fprintf(stderr, "CUDA driver error: %s\n", name);
        std::exit(1);
    }
}

void check(nvrtcResult err)
{
    if (err != NVRTC_SUCCESS) {
        std::fprintf(stderr, "NVRTC error: %s\n", nvrtcGetErrorString(err));
        std::exit(1);
    }
}

double nowMs()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Average time of `runs` runs of `launch`, in milliseconds, after 1 warm-up run.
template <typename F>
double timeKernel(F launch, int runs)
{
    launch();
    check(cudaDeviceSynchronize());

    double t0 = nowMs();
    for (int r = 0; r < runs; ++r) {
        launch();
    }
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    return (nowMs() - t0) / runs;
}

// Compiled by nvcc with the program: the degree and the coefficients arrive at run time.
__global__ void polyGeneric(const float* x, float* y, int n, const float* c, int degree)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    float v = x[i];
    float p = c[degree];
    for (int k = degree - 1; k >= 0; --k) {
        p = p * v + c[k];
    }
    y[i] = p;
}

// CUDA C++ source text for the same polynomial, with its coefficients written in.
std::string specialisedSource(const std::vector<float>& c)
{
    std::string src = "extern \"C\" __global__ void poly(const float* x, float* y, int n)\n"
                      "{\n"
                      "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                      "    if (i >= n) {\n"
                      "        return;\n"
                      "    }\n"
                      "\n"
                      "    float v = x[i];\n";

    char line[64];
    std::snprintf(line, sizeof line, "    float p = %.9ef;\n", c[DEGREE]);
    src += line;
    for (int k = DEGREE - 1; k >= 0; --k) {
        std::snprintf(line, sizeof line, "    p = p * v + %.9ef;\n", c[k]);
        src += line;
    }

    src += "    y[i] = p;\n}\n";
    return src;
}

// Compile `source` with NVRTC: a cubin for 1 GPU ("sm_87"), or PTX for the driver.
std::vector<char> compile(const std::string& source, const std::string& arch, bool cubin)
{
    nvrtcProgram prog;
    check(nvrtcCreateProgram(&prog, source.c_str(), "poly.cu", 0, nullptr, nullptr));

    std::string target = cubin ? arch : "compute_" + arch.substr(3);
    std::string option = "--gpu-architecture=" + target;
    const char* options[] = {option.c_str()};
    nvrtcResult result = nvrtcCompileProgram(prog, 1, options);

    if (result != NVRTC_SUCCESS) {
        size_t logSize;
        check(nvrtcGetProgramLogSize(prog, &logSize));
        std::string log(logSize, '\0');
        check(nvrtcGetProgramLog(prog, log.data()));
        std::printf("  compile failed, log:\n%s", log.c_str());

        check(nvrtcDestroyProgram(&prog));
        return {};
    }

    size_t size;
    std::vector<char> image;
    if (cubin) {
        check(nvrtcGetCUBINSize(prog, &size));
        image.resize(size);
        check(nvrtcGetCUBIN(prog, image.data()));
    } else {
        check(nvrtcGetPTXSize(prog, &size));
        image.resize(size);
        check(nvrtcGetPTX(prog, image.data()));
    }

    check(nvrtcDestroyProgram(&prog));
    return image;
}

// Load a PTX or cubin image into the current context and find the kernel `name` in it.
CUfunction loadKernel(const std::vector<char>& image, const char* name, CUmodule& module)
{
    check(cuModuleLoadData(&module, image.data()));

    CUfunction kernel;
    check(cuModuleGetFunction(&kernel, module, name));
    return kernel;
}

int main()
{
    check(cuInit(0));
    CUdevice device;
    check(cuDeviceGet(&device, 0));
    CUcontext context;
    check(cuDevicePrimaryCtxRetain(&context, device));
    check(cuCtxSetCurrent(context));

    int major, minor;
    auto majorAttr = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
    auto minorAttr = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;
    check(cuDeviceGetAttribute(&major, majorAttr, device));
    check(cuDeviceGetAttribute(&minor, minorAttr, device));
    std::string arch = "sm_" + std::to_string(major * 10 + minor);

    char name[256];
    check(cuDeviceGetName(name, sizeof name, device));
    std::printf("GPU: %s, compute capability %d.%d, ", name, major, minor);
    std::printf("target %s\n", arch.c_str());

    std::vector<float> c(DEGREE + 1), x(N), yGeneric(N), ySpecial(N);
    for (int k = 0; k <= DEGREE; ++k) {
        c[k] = 1.0f / (k + 1);
    }
    for (int i = 0; i < N; ++i) {
        x[i] = 2.0f * i / N - 1.0f;
    }

    std::string source = specialisedSource(c);
    std::printf("NVRTC, a polynomial of degree %d, ", DEGREE);
    std::printf("%zu bytes of source\n", source.size());

    double t0 = nowMs();
    std::vector<char> ptx = compile(source, arch, false);
    double t1 = nowMs();
    std::vector<char> cubin = compile(source, arch, true);
    double t2 = nowMs();
    std::printf("  compile to PTX     %8.2f ms  %6zu bytes\n", t1 - t0, ptx.size());
    std::printf("  compile to cubin   %8.2f ms  %6zu bytes\n", t2 - t1, cubin.size());

    CUmodule ptxModule, cubinModule;
    t0 = nowMs();
    loadKernel(ptx, "poly", ptxModule);
    t1 = nowMs();
    CUfunction poly = loadKernel(cubin, "poly", cubinModule);
    t2 = nowMs();
    std::printf("  load PTX           %8.2f ms\n", t1 - t0);
    std::printf("  load cubin         %8.2f ms\n", t2 - t1);

    size_t bytes = size_t(N) * sizeof(float);
    size_t coeffBytes = c.size() * sizeof(float);
    float *X, *Y, *C;
    check(cudaMalloc(&X, bytes));
    check(cudaMalloc(&Y, bytes));
    check(cudaMalloc(&C, coeffBytes));
    check(cudaMemcpy(X, x.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(C, c.data(), coeffBytes, cudaMemcpyHostToDevice));

    int n = N;
    int degree = DEGREE;
    unsigned threads = 256;
    unsigned blocks = (N + threads - 1) / threads;
    void* args[] = {&X, &Y, &n};

    auto generic = [&] { polyGeneric<<<blocks, threads>>>(X, Y, n, C, degree); };
    auto special = [&] {
        check(cuLaunchKernel(poly, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
    };

    for (int k = 0; k < 20; ++k) {            // warm up: clocks rise under sustained load
        timeKernel(generic, 5);
    }

    double flop = 2.0 * DEGREE * N;
    std::printf("p(x) for %d values of x\n", N);

    double ms = timeKernel(generic, 20);
    check(cudaMemcpy(yGeneric.data(), Y, bytes, cudaMemcpyDeviceToHost));
    std::printf("  generic, nvcc, <<<...>>>        %7.3f ms", ms);
    std::printf("  %6.1f GFLOP/s\n", flop / ms / 1e6);

    ms = timeKernel(special, 20);
    check(cudaMemcpy(ySpecial.data(), Y, bytes, cudaMemcpyDeviceToHost));
    std::printf("  specialised, NVRTC, cuLaunch    %7.3f ms", ms);
    std::printf("  %6.1f GFLOP/s\n", flop / ms / 1e6);

    double worst = 0.0;
    for (int i = 0; i < N; i += 4099) {
        double exact = 0.0;
        for (int k = DEGREE; k >= 0; --k) {
            exact = exact * x[i] + c[k];
        }
        worst = std::fmax(worst, std::fabs(exact - ySpecial[i]));
    }

    double differ = 0.0;
    for (int i = 0; i < N; ++i) {
        differ = std::fmax(differ, std::fabs(yGeneric[i] - ySpecial[i]));
    }
    std::printf("  max error %.1e, generic vs specialised %.1e\n", worst, differ);

    std::printf("a source with a mistake\n");
    std::string broken = source;
    broken.replace(broken.find("y[i] = p"), 8, "y[i] = q");
    compile(broken, arch, true);

    check(cuModuleUnload(ptxModule));
    check(cuModuleUnload(cubinModule));
    check(cudaFree(X));
    check(cudaFree(Y));
    check(cudaFree(C));
    check(cuDevicePrimaryCtxRelease(device));

    std::printf(worst < 1e-4 && differ == 0.0 ? "PASSED\n" : "FAILED\n");
}
