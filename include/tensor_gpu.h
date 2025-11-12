#ifndef TENSOR_GPU_H
#define TENSOR_GPU_H

#ifdef USE_CUBLAS
#include <cublas_v2.h>
#endif
#include <stdexcept>

extern "C" {
void* cuda_malloc_wrapper(size_t bytes);
void cuda_free_wrapper(void* ptr);
void cuda_memcpy_h2d_wrapper(void* dst, const void* src, size_t bytes);
void cuda_memcpy_d2h_wrapper(void* dst, const void* src, size_t bytes);
void cuda_memcpy_d2d_wrapper(void* dst, const void* src, size_t bytes);
}

namespace tensor {

#ifdef USE_CUBLAS
/// @brief Singleton class for managing cuBLAS handle
class CublasHandle {
private:
    inline static cublasHandle_t handle_ = nullptr;
    inline static bool initialized_ = false;
    
public:
    /// @brief Get the global cuBLAS handle (creates it if needed)
    static cublasHandle_t get() {
        if (!initialized_) {
            cublasStatus_t status = cublasCreate(&handle_);
            if (status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("Failed to create cuBLAS handle");
            }
            // Use host pointers for alpha/beta parameters
            cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST);
            initialized_ = true;
        }
        return handle_;
    }
    
    /// @brief Clean up cuBLAS handle
    static void cleanup() {
        if (initialized_) {
            cublasDestroy(handle_);
            handle_ = nullptr;
            initialized_ = false;
        }
    }
};
#endif // USE_CUBLAS

// Forward declarations for GPU direct functions used in tensor_base.h
template<typename T>
void sum_gpu_direct(const T* d_a, T* d_result, size_t n);

template<typename T>
void min_gpu_direct(const T* d_a, T* d_result, size_t n);

template<typename T>
void max_gpu_direct(const T* d_a, T* d_result, size_t n);

template<typename T>
void abs_gpu_direct(const T* d_src, T* d_dst, size_t n);

} // namespace tensor

#endif // TENSOR_GPU_H
