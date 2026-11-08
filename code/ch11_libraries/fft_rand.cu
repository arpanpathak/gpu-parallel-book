// Chapter 11: random numbers with cuRAND, and the frequencies in a signal with cuFFT.
// Build: nvcc -arch=native -O2 fft_rand.cu -o fft_rand -lcurand -lcufft

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda/std/functional>
#include <cuda_runtime.h>
#include <cufft.h>
#include <curand.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

void check(cudaError_t err)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

void check(curandStatus_t status)
{
    if (status != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuRAND error %d\n", int(status));
        std::exit(1);
    }
}

void check(cufftResult status)
{
    if (status != CUFFT_SUCCESS) {
        std::fprintf(stderr, "cuFFT error %d\n", int(status));
        std::exit(1);
    }
}

struct Square {
    __host__ __device__ float operator()(float v) const { return v * v; }
};

// 16 Mi numbers from the normal distribution; their mean should be near 0 and their
// standard deviation near 1.
bool normalNumbers()
{
    const int n = 16 << 20;
    float* x;
    check(cudaMalloc(&x, n * sizeof(float)));

    curandGenerator_t gen;
    check(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    check(curandSetPseudoRandomGeneratorSeed(gen, 2024));

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    check(curandGenerateNormal(gen, x, n, 0.0f, 1.0f));          // warm-up
    check(cudaEventRecord(start));
    check(curandGenerateNormal(gen, x, n, 0.0f, 1.0f));
    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(stop));
    float ms;
    check(cudaEventElapsedTime(&ms, start, stop));

    Square square;
    cuda::std::plus<float> plus;
    float sum = thrust::reduce(thrust::device, x, x + n, 0.0f);
    float squares = thrust::transform_reduce(thrust::device, x, x + n, square, 0.0f, plus);
    double mean = sum / n;
    double sd = std::sqrt(squares / n - mean * mean);

    std::printf("cuRAND, %d normal numbers in %.2f ms", n, ms);
    std::printf(" (%.0f million per second)\n", n / ms / 1e3);
    std::printf("  mean %.5f, standard deviation %.5f\n", mean, sd);

    check(curandDestroyGenerator(gen));
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    check(cudaFree(x));
    return std::fabs(mean) < 1e-3 && std::fabs(sd - 1.0) < 1e-3;
}

// 1,024 samples of sin(50 t) + 0.5 sin(120 t), with t running once around the circle.
// The transform should show 2 peaks, at frequencies 50 and 120, and nothing else.
bool spectrum()
{
    const int n = 1024;
    const int bins = n / 2 + 1;               // a real signal needs only half the spectrum
    const double pi = 3.14159265358979323846;

    std::vector<float> signal(n);
    for (int t = 0; t < n; ++t) {
        double angle = 2.0 * pi * t / n;
        signal[t] = std::sin(50 * angle) + 0.5 * std::sin(120 * angle);
    }

    size_t spectrumBytes = bins * sizeof(cufftComplex);
    float* x;
    cufftComplex* spectrum;
    check(cudaMalloc(&x, n * sizeof(float)));
    check(cudaMalloc(&spectrum, spectrumBytes));
    check(cudaMemcpy(x, signal.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    cufftHandle plan;
    check(cufftPlan1d(&plan, n, CUFFT_R2C, 1));
    check(cufftExecR2C(plan, x, spectrum));

    std::vector<cufftComplex> result(bins);
    check(cudaMemcpy(result.data(), spectrum, spectrumBytes, cudaMemcpyDeviceToHost));

    std::printf("cuFFT, %d samples of sin(50 t) + 0.5 sin(120 t)\n", n);
    int peaks = 0;
    bool ok = true;
    for (int f = 0; f < bins; ++f) {
        double size = std::hypot(result[f].x, result[f].y);
        if (size > 1.0) {
            std::printf("  frequency %3d: magnitude %.1f\n", f, size);
            ++peaks;
            ok = ok && (f == 50 || f == 120);
        }
    }

    check(cufftDestroy(plan));
    check(cudaFree(x));
    check(cudaFree(spectrum));
    return ok && peaks == 2;
}

int main()
{
    bool ok = normalNumbers();
    ok &= spectrum();
    std::printf("%s\n", ok ? "PASSED" : "FAILED");
}
