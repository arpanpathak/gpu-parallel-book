// Chapter 11, 11.2 - the Thrust version of reduction + SAXPY, runnable.
//
// Build: nvcc --extended-lambda -arch=compute_60 thrust_example.cu -o thrust_example
//        (Jetson Orin: -arch=sm_87; A100: -arch=sm_80; H100: -arch=sm_90)
//        (--extended-lambda is required by the __device__ lambda annotation)
// Run:   ./thrust_example

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

int main()
{
    const int n = 1 << 20;

    // device_vector: a RAII device array (like Chapter 10's DeviceBuffer).
    thrust::device_vector<float> x(n), y(n);

    // thrust::sequence fills x with 0..n-1 (parallel, device-side).
    thrust::sequence(x.begin(), x.end());

    // thrust::transform applies a functor elementwise. thrust::device is the
    // execution policy and must come FIRST (policy, first, last, result, op).
    thrust::transform(thrust::device,
                      x.begin(), x.end(), y.begin(),
                      [] __device__ (float v) { return v * 2.0f + 1.0f; });

    // thrust::reduce folds the array (the Chapter 8 reduction, tuned).
    const float total = thrust::reduce(y.begin(), y.end(), 0.0f,
                                       thrust::plus<float>());

    // thrust::sort, thrust::exclusive_scan, etc. follow the same shape.
    thrust::sort(y.begin(), y.end());

    // For y[i] = 2*i + 1 the exact total is n^2. Float accumulation is not
    // exact at this magnitude, so use a relative tolerance.
    const double expected = static_cast<double>(n) * n;
    const double err = std::abs(static_cast<double>(total) - expected);
    std::printf("thrust: total = %.6g, expected = %.6g, rel err = %.3g\n",
                static_cast<double>(total), expected, err / expected);
    std::printf("thrust: sorted[0] = %g, sorted[n-1] = %g\n",
                static_cast<double>(y[0]), static_cast<double>(y[n - 1]));

    if (err > 1e-3 * expected)
    {
        std::fprintf(stderr, "CHAPTER 11 THRUST TEST FAILED\n");
        return 1;
    }
    std::printf("Chapter 11 Thrust test PASSED\n");
    return 0;
}
