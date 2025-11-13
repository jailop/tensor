#include <tensor_gpu.cuh>
#include <tensor_backend.h>

namespace tensor {

#ifndef USE_GPU
/**
 * Fallout implementation when GPU support is not compiled in.
 * Otherwise, the function defined in tensor_gpu.cuh is used.
 */
static inline bool is_gpu_available() {
    return false;
}
#endif

/**
 * Check if BLAS backend is available.
 */
static inline constexpr bool is_blas_available() {
#ifdef USE_BLAS
    return true;
#else
    return false;
#endif
}

Backend get_active_backend() {
#ifdef USE_GPU
    if (is_gpu_available()) {
        return Backend::GPU;
    }
#endif
#ifdef USE_BLAS
    return Backend::BLAS;
#endif
    return Backend::CPU;
}

}  // namespace tensor
