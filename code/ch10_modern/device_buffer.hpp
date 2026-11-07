// Chapter 10: a check without a macro, and a device allocation that frees itself.
#pragma once

#include <cstddef>
#include <source_location>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <cuda/std/span>
#include <cuda_runtime.h>

using Where = std::source_location;

// Throw if a CUDA call failed. The default argument records the caller's file and line.
inline void check(cudaError_t err, Where where = Where::current())
{
    if (err == cudaSuccess) {
        return;
    }

    std::string place = std::string(where.file_name()) + ":" + std::to_string(where.line());
    throw std::runtime_error(place + ": " + cudaGetErrorString(err));
}

// Bytes in use from the pool that cudaMallocAsync draws on, in MiB.
inline double poolMiB()
{
    cudaMemPool_t pool;
    check(cudaDeviceGetDefaultMemPool(&pool, 0));

    unsigned long long used = 0;
    check(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used));
    return used / 1048576.0;
}

// `count` values of T in device memory, owned by exactly 1 object.
template <typename T>
class DeviceBuffer {
    static_assert(std::is_trivially_copyable_v<T>, "device memory holds plain bytes");

public:
    explicit DeviceBuffer(std::size_t count) : count_(count)
    {
        check(cudaMallocAsync(&data_, bytes(), 0));
    }

    explicit DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size())
    {
        check(cudaMemcpy(data_, host.data(), bytes(), cudaMemcpyHostToDevice));
    }

    ~DeviceBuffer()
    {
        release();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0))
    {
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other) {
            release();
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    T* data() const { return data_; }
    std::size_t size() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

    cuda::std::span<const T> view() const { return {data_, count_}; }
    cuda::std::span<T> span() { return {data_, count_}; }

    std::vector<T> toHost() const
    {
        std::vector<T> host(count_);
        check(cudaMemcpy(host.data(), data_, bytes(), cudaMemcpyDeviceToHost));
        return host;
    }

private:
    void release()
    {
        if (data_ != nullptr) {               // a moved-from buffer owns nothing
            cudaFreeAsync(data_, 0);
        }
    }

    T* data_ = nullptr;
    std::size_t count_ = 0;
};
