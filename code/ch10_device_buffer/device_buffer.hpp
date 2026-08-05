// Chapter 10, 10.2 - RAII DeviceBuffer<T>, verbatim from the book.
// Include from any .cu or host .cpp file: #include "device_buffer.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <type_traits>

// Throw a std::runtime_error describing a failed CUDA call.
[[noreturn]] inline void throwCudaError(cudaError_t err, const char* what)
{
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(err));
}

// RAII wrapper for a device allocation of T.
template <typename T>
class DeviceBuffer
{
public:
    // --- Construction: allocate on the device -----------------------------
    // static_assert: only trivially-copyable types may live in device memory
    // without custom copy semantics. This converts a runtime confusion into
    // a compile-time error.
    static_assert(std::is_trivially_copyable_v<T>,
                  "DeviceBuffer<T> requires a trivially copyable T");

    explicit DeviceBuffer(std::size_t count) : count_(count)
    {
        const cudaError_t err = cudaMalloc((void**)&ptr_, count_ * sizeof(T));
        if (err != cudaSuccess) throwCudaError(err, "cudaMalloc");
    }

    // --- No copying (a device buffer is a unique resource) ----------------
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // --- Move semantics: transfer ownership, never copy the bytes ---------
    // After a move, the source is empty (nullptr). The destructor must
    // handle nullptr gracefully - hence the check in ~DeviceBuffer.
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_)
    {
        other.ptr_   = nullptr;    // source relinquishes the allocation
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other)
        {
            reset();               // release what we held
            ptr_   = other.ptr_;   // take ownership
            count_ = other.count_;
            other.ptr_   = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    // --- Destruction: release the device allocation -----------------------
    ~DeviceBuffer() { reset(); }

    // --- Accessors ---------------------------------------------------------
    T*       data()       noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return count_; }

    // Host <-> device transfer helpers (keep them explicit and checked).
    void copyToDevice(const T* hostSrc)
    {
        const cudaError_t err = cudaMemcpy(ptr_, hostSrc,
                                           count_ * sizeof(T),
                                           cudaMemcpyHostToDevice);
        if (err != cudaSuccess) throwCudaError(err, "cudaMemcpy H2D");
    }

    void copyToHost(T* hostDst) const
    {
        const cudaError_t err = cudaMemcpy(hostDst, ptr_,
                                           count_ * sizeof(T),
                                           cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throwCudaError(err, "cudaMemcpy D2H");
    }

private:
    void reset() noexcept
    {
        if (ptr_ != nullptr)
        {
            cudaFree(ptr_);        // best-effort: destructors must not throw
            ptr_   = nullptr;
            count_ = 0;
        }
    }

    T*         ptr_ = nullptr;   // device pointer
    std::size_t count_ = 0;      // number of elements
};
