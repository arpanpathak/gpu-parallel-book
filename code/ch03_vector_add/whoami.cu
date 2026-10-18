// Chapter 3: every thread of a small launch prints who it is.
// Build: nvcc -arch=native whoami.cu -o whoami

#include <cstdio>
#include <cuda_runtime.h>

__global__ void whoAmI()
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    printf("block %d, position %d -> index %d\n", blockIdx.x, threadIdx.x, index);
}

int main()
{
    whoAmI<<<2, 4>>>();           // 2 blocks of 4 threads: indexes 0 to 7
    cudaDeviceSynchronize();      // wait, so the printed lines reach the screen
    return 0;
}
