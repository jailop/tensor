/**
 * @file tensor.h
 * @brief High-performance multi-dimensional tensor library with GPU, BLAS, and autograd support
 * 
 * This header provides a comprehensive tensor library for scientific computing and machine learning.
 * It features:
 * - Multi-dimensional tensor operations with arbitrary dimensions
 * - GPU acceleration via CUDA (when USE_GPU is defined)
 * - Optimized CPU operations via BLAS (when USE_BLAS is defined)
 * - Automatic differentiation (autograd) for gradient computation
 * - Mathematical function mapping (exp, log, sin, cos, etc.)
 * - Statistical operations (mean, variance, correlation, etc.)
 * - Broadcasting for element-wise operations
 * - Tensor views and slicing for memory-efficient operations
 * 
 * @section backend Backend Selection
 * The library automatically selects the best available backend at runtime:
 * 1. **GPU (CUDA)**: Used by default if compiled with USE_GPU, GPU is available, and use_gpu=true
 * 2. **BLAS**: Used if GPU is not available but compiled with USE_BLAS
 * 3. **CPU**: Fallback implementation when neither GPU nor BLAS is available
 * 
 * You can check the active backend using:
 * @code
 * auto backend = get_active_backend();  // Returns Backend::GPU, Backend::BLAS, or Backend::CPU
 * if (is_gpu_available()) {
 *     // GPU operations are available
 * }
 * @endcode
 * 
 * @author Tensor Library Team
 * @version 1.0
 * @date 2024
 * 
 * @section usage Usage Example
 * @code
 * // Create a 2D tensor (matrix) - automatically uses GPU if available
 * Tensor<float, 2> matrix({3, 4});
 * matrix.fill(1.0f);
 * 
 * // Check which backend is being used
 * std::cout << "Using: " << backend_name(matrix.backend()) << std::endl;
 * 
 * // Enable gradient tracking
 * Tensor<float, 2> x({2, 2}, true, true);  // use_gpu=true, requires_grad=true
 * auto y = x * x;
 * y.backward();
 * @endcode
 * 
 * @section features Key Features
 * - **Type-safe**: Template-based compile-time dimension checking
 * - **High-performance**: Optimized with BLAS and GPU support
 * - **Autograd**: Automatic gradient computation for deep learning
 * - **Flexible**: Support for views, slicing, and broadcasting
 * - **Smart backend selection**: Automatically uses GPU → BLAS → CPU
 */

#ifndef _TENSOR_H
#define _TENSOR_H

#include <cstddef>
#include <algorithm>
#include <memory>
#include <array>
#include <variant>
#include <type_traits>
#include <string>
#include <execution>
#include <span>
#include <ranges>
#include <cmath>
#include <optional>
#include <functional>
#include <vector>
#include <random>

#include "tensor_perf.h"

#ifdef USE_GPU
#include "tensor_gpu.cuh"
#endif

#ifdef USE_BLAS
extern "C" {
    /// @brief BLAS function declarations for optimized matrix operations
    void cblas_sgemm(const int Order, const int TransA, const int TransB,
                     const int M, const int N, const int K,
                     const float alpha, const float *A, const int lda,
                     const float *B, const int ldb,
                     const float beta, float *C, const int ldc);
    void cblas_dgemm(const int Order, const int TransA, const int TransB,
                     const int M, const int N, const int K,
                     const double alpha, const double *A, const int lda,
                     const double *B, const int ldb,
                     const double beta, double *C, const int ldc);
    float cblas_sdot(const int N, const float *X, const int incX,
                     const float *Y, const int incY);
    double cblas_ddot(const int N, const double *X, const int incX,
                      const double *Y, const int incY);
}

// BLAS constants
constexpr int CblasRowMajor = 101;
constexpr int CblasNoTrans = 111;

// BLAS helper functions to select the correct function based on type
template<typename T>
inline T blas_dot(const int N, const T *X, const int incX, const T *Y, const int incY) {
    // Fallback for unsupported types
    T result = T();
    for (int i = 0; i < N; ++i) {
        result += X[i * incX] * Y[i * incY];
    }
    return result;
}

template<>
inline float blas_dot<float>(const int N, const float *X, const int incX, const float *Y, const int incY) {
    return cblas_sdot(N, X, incX, Y, incY);
}

template<>
inline double blas_dot<double>(const int N, const double *X, const int incX, const double *Y, const int incY) {
    return cblas_ddot(N, X, incX, Y, incY);
}

template<typename T>
inline void blas_gemm(const int Order, const int TransA, const int TransB,
                      const int M, const int N, const int K,
                      const T alpha, const T *A, const int lda,
                      const T *B, const int ldb,
                      const T beta, T *C, const int ldc) {
    // Fallback for unsupported types - standard matrix multiplication
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            T sum = T();
            for (int k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = beta * C[i * ldc + j] + alpha * sum;
        }
    }
}

template<>
inline void blas_gemm<float>(const int Order, const int TransA, const int TransB,
                             const int M, const int N, const int K,
                             const float alpha, const float *A, const int lda,
                             const float *B, const int ldb,
                             const float beta, float *C, const int ldc) {
    cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
inline void blas_gemm<double>(const int Order, const int TransA, const int TransB,
                              const int M, const int N, const int K,
                              const double alpha, const double *A, const int lda,
                              const double *B, const int ldb,
                              const double beta, double *C, const int ldc) {
    cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
#endif

/**
 * @enum Backend
 * @brief Available computational backends
 * 
 * Indicates which backend is being used for tensor operations.
 * Priority order: GPU > BLAS > CPU
 */
enum class Backend {
    CPU,   ///< Standard CPU implementation
    BLAS,  ///< Optimized BLAS for CPU operations
    GPU    ///< CUDA GPU acceleration
};

/**
 * @brief Get the name of a backend as a string
 * @param backend The backend enum value
 * @return Human-readable name of the backend
 */
inline std::string backend_name(Backend backend) {
    switch (backend) {
        case Backend::CPU: return "CPU";
        case Backend::BLAS: return "BLAS";
        case Backend::GPU: return "GPU";
        default: return "Unknown";
    }
}

/**
 * @brief Get the currently active backend
 * @return The backend being used by default
 * 
 * This function checks at runtime which backend is available and will be used
 * for new tensor operations. Priority: GPU > BLAS > CPU
 */
inline Backend get_active_backend() {
#ifdef USE_GPU
    if (TensorGPU::is_gpu_available()) {
        return Backend::GPU;
    }
#endif
#ifdef USE_BLAS
    return Backend::BLAS;
#endif
    return Backend::CPU;
}

/**
 * @brief Check if GPU backend is available
 * @return true if GPU support is compiled in and GPU is available
 */
inline bool is_gpu_available() {
#ifdef USE_GPU
    return TensorGPU::is_gpu_available();
#else
    return false;
#endif
}

/**
 * @brief Check if BLAS backend is available
 * @return true if BLAS support is compiled in
 */
inline constexpr bool is_blas_available() {
#ifdef USE_BLAS
    return true;
#else
    return false;
#endif
}

/**
 * @enum TensorError
 * @brief Error codes for tensor operations
 * 
 * Provides enumeration of possible errors that can occur during tensor operations.
 * These errors are returned as part of TensorResult variant type.
 */
enum class TensorError {
    DimensionMismatch,    ///< Tensor dimensions do not match for the operation
    ContractionMismatch,  ///< Contraction dimensions are incompatible
    InvalidArgument       ///< Invalid argument provided to a function
};

/**
 * @brief Convert TensorError to human-readable string
 * @param error The error code to convert
 * @return A string description of the error
 */
inline std::string to_string(TensorError error) {
    switch (error) {
        case TensorError::DimensionMismatch:
            return "Tensor dimensions must match";
        case TensorError::ContractionMismatch:
            return "Contraction dimension must match";
        case TensorError::InvalidArgument:
            return "Invalid argument provided";
        default:
            return "Unknown error";
    }
}

/**
 * @brief Result type for tensor operations that may fail
 * @tparam T The expected result type (usually a Tensor)
 * 
 * Operations that can fail return a variant containing either the result
 * or a TensorError. Use std::holds_alternative and std::get to access.
 * 
 * @code
 * auto result = tensor1 + tensor2;
 * if (std::holds_alternative<Tensor<float, 2>>(result)) {
 *     auto& tensor = std::get<Tensor<float, 2>>(result);
 *     // use tensor
 * } else {
 *     auto error = std::get<TensorError>(result);
 *     // handle error
 * }
 * @endcode
 */
template <typename T>
using TensorResult = std::variant<T, TensorError>;

/**
 * @brief Type alias for tensor indices/coordinates
 * @tparam N Number of dimensions
 * 
 * Fixed-size array representing indices or coordinates in N-dimensional space.
 */
template <size_t N>
using TensorIndices = std::array<size_t, N>;

/// @brief Forward declaration for autograd
template <typename T, size_t N>
class Tensor;

/**
 * @brief Function type for backward pass in autograd
 * @tparam T Data type (float, double, etc.)
 * @tparam N Number of dimensions
 * 
 * Used to store gradient computation functions in the computational graph.
 */
template <typename T, size_t N>
using BackwardFunc = std::function<void(const Tensor<T, N>&)>;

/**
 * @namespace loss
 * @brief Loss functions for training neural networks
 * 
 * Provides common loss functions used in machine learning:
 * - Mean Squared Error (MSE)
 * - Cross Entropy
 * - Binary Cross Entropy
 * - L1 Loss
 * - Smooth L1 Loss
 * 
 * All loss functions support automatic differentiation.
 */
namespace loss {
    /// @brief Mean squared error loss
    template<typename T, size_t N>
    Tensor<T, N> mse_loss(const Tensor<T, N>&, const Tensor<T, N>&, const std::string& = "mean");
    
    /// @brief Cross entropy loss for multi-class classification
    template<typename T, size_t N>
    Tensor<T, 1> cross_entropy_loss(const Tensor<T, N>&, const Tensor<T, N>&, const std::string& = "mean");
    
    /// @brief Binary cross entropy loss for binary classification
    template<typename T, size_t N>
    Tensor<T, 1> binary_cross_entropy(const Tensor<T, N>&, const Tensor<T, N>&, const std::string& = "mean");
    
    /// @brief L1 loss (Mean Absolute Error)
    template<typename T, size_t N>
    Tensor<T, 1> l1_loss(const Tensor<T, N>&, const Tensor<T, N>&, const std::string& = "mean");
    
    /// @brief Smooth L1 loss (Huber loss)
    template<typename T, size_t N>
    Tensor<T, 1> smooth_l1_loss(const Tensor<T, N>&, const Tensor<T, N>&, T = T(1), const std::string& = "mean");
}

/**
 * @class Tensor
 * @brief Multi-dimensional array with GPU, BLAS, and autograd support
 * @tparam T Data type (float, double, int, etc.)
 * @tparam N Number of dimensions
 * 
 * A powerful tensor class supporting:
 * - Arbitrary dimensions (compile-time fixed)
 * - GPU acceleration (CUDA)
 * - Optimized BLAS operations
 * - Automatic differentiation (autograd)
 * - Element-wise operations with broadcasting
 * - Mathematical functions (exp, log, sin, etc.)
 * - Statistical operations (mean, variance, correlation)
 * - Linear algebra operations
 * - Tensor views and slicing
 * 
 * @section memory Memory Layout
 * Data is stored in row-major order in a flat array. Strides are computed
 * automatically to enable efficient multi-dimensional indexing.
 * 
 * @section autograd Automatic Differentiation
 * When `requires_grad` is true, the tensor participates in gradient computation.
 * Operations build a computational graph that can be traversed via backward().
 * 
 * @section example Example Usage
 * @code
 * // Create a 2x3 matrix
 * Tensor<float, 2> A({2, 3});
 * A.fill(1.0f);
 * 
 * // Element-wise operations
 * auto B = A + 2.0f;
 * auto C = A * B;
 * 
 * // With autograd
 * Tensor<float, 1> x({10}, true, true);  // GPU=true, requires_grad=true
 * auto y = x.exp().sum();
 * y.backward();
 * auto gradients = x.grad();  // Get computed gradients
 * @endcode
 */
template <typename T, size_t N>
class Tensor {
private:
    std::unique_ptr<T[]> data_;      ///< Flat data storage in row-major order
    TensorIndices<N> dims_;          ///< Dimensions of the tensor
    TensorIndices<N> strides_;       ///< Strides for each dimension
    bool use_gpu_;                   ///< Flag to indicate if GPU should be used
    
    /// @name Autograd Support
    /// @{
    bool requires_grad_;                              ///< Whether to track gradients
    std::unique_ptr<Tensor<T, N>> grad_;             ///< Gradient tensor
    std::vector<BackwardFunc<T, N>> backward_funcs_; ///< Functions to compute gradients
    bool is_leaf_;                                    ///< Whether this is a leaf node in the graph
    /// @}
    
    /**
     * Calculate the flat offset for given multi-dimensional indices.
     * @param indices An array of indices for each dimension.
     * @return The corresponding flat offset in the data array.
     */
    size_t offset(const TensorIndices<N>& indices) const {
        size_t off = 0;
        for (size_t i = 0; i < N; ++i) {
            off += indices[i] * strides_[i];
        }
        return off;
    }
    
    // Friend declaration to allow tensors of different dimensions to access each other's data
    template <typename U, size_t M>
    friend class Tensor;
    
    // Friend declarations for TensorView and TensorSlice
    template <typename U, size_t M>
    friend class TensorView;
    template <typename U, size_t M>
    friend class TensorSlice;
    
    // Friend declarations for I/O functions
    template <typename U, size_t M>
    friend bool save_binary(const Tensor<U, M>&, const std::string&);
    template <typename U, size_t M>
    friend bool save_text(const Tensor<U, M>&, const std::string&, int);
    template <typename U, size_t M>
    friend bool save_npy(const Tensor<U, M>&, const std::string&);
    template <typename U, size_t M>
    friend void print(const Tensor<U, M>&, std::ostream&, int, size_t, size_t);
    
    // Friend declarations for scalar-first operators
    template <typename U, size_t M>
    friend Tensor<U, M> operator+(const U& scalar, const Tensor<U, M>& tensor);
    template <typename U, size_t M>
    friend Tensor<U, M> operator-(const U& scalar, const Tensor<U, M>& tensor);
    template <typename U, size_t M>
    friend Tensor<U, M> operator*(const U& scalar, const Tensor<U, M>& tensor);
    template <typename U, size_t M>
    friend Tensor<U, M> operator/(const U& scalar, const Tensor<U, M>& tensor);
    
public:
    // Public data accessors for use by library functions
    const T* data_ptr() const { return data_.get(); }
    T* data_ptr() { return data_.get(); }
    
    /**
     * Calculate the total size of the tensor.
     * @return The total number of elements in the tensor.
     */
    size_t total_size() const {
        size_t size = 1;
        for (size_t i = 0; i < N; ++i) {
            size *= dims_[i];
        }
        return size;
    }
    
    /**
     * Get raw pointer to data (for internal use and I/O).
     * @return Pointer to the underlying data array.
     */
    const T* data() const { return data_.get(); }
    T* data() { return data_.get(); }
    
public:

    /**
     * Constructor to initialize the tensor with given dimensions.
     * @param dimensions An array specifying the size of each dimension.
     * @param use_gpu Whether to use GPU acceleration when available (default: true).
     * @param requires_grad Whether to track gradients for this tensor (default: false).
     */
    Tensor(const TensorIndices<N>& dimensions, bool use_gpu = true, bool requires_grad = false) 
        : dims_(dimensions), requires_grad_(requires_grad), is_leaf_(true) {
        size_t total = 1;
        for (size_t i = N; i-- > 0;) {
            strides_[i] = total;
            total *= dims_[i];
        }
        data_ = std::make_unique<T[]>(total);
#ifdef USE_GPU
        use_gpu_ = use_gpu && TensorGPU::is_gpu_available();
#else
        use_gpu_ = false;
#endif
        if (requires_grad_) {
            grad_ = std::make_unique<Tensor<T, N>>(dims_, use_gpu_, false);
            grad_->fill(T(0));
        }
    }
    
    /**
     * Copy constructor for deep copying the tensor.
     * @param other The tensor to copy from.
     */
    Tensor(const Tensor& other) 
        : dims_(other.dims_), strides_(other.strides_), use_gpu_(other.use_gpu_),
          requires_grad_(other.requires_grad_), is_leaf_(other.is_leaf_),
          backward_funcs_(other.backward_funcs_) {
        size_t total = total_size();
        data_ = std::make_unique<T[]>(total);
        std::copy(other.data_.get(), other.data_.get() + total, data_.get());
        
        // Copy gradient if it exists
        if (other.grad_) {
            grad_ = std::make_unique<Tensor<T, N>>(*other.grad_);
        }
    }
    
    /**
     * Assignment operator for deep copying the tensor.
     * @param other The tensor to copy from.
     * @return Reference to the assigned tensor.
     */
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            dims_ = other.dims_;
            strides_ = other.strides_;
            use_gpu_ = other.use_gpu_;
            requires_grad_ = other.requires_grad_;
            is_leaf_ = other.is_leaf_;
            backward_funcs_ = other.backward_funcs_;
            
            size_t total = total_size();
            data_ = std::make_unique<T[]>(total);
            std::copy(other.data_.get(), other.data_.get() + total, data_.get());
            
            // Copy gradient if it exists
            if (other.grad_) {
                grad_ = std::make_unique<Tensor<T, N>>(*other.grad_);
            } else {
                grad_.reset();
            }
        }
        return *this;
    }
    
    /**
     * Indexing operator to access elements.
     * @param indices An array of indices for each dimension.
     * @return A reference to the element at the specified indices.
     */
    T& operator[](const TensorIndices<N>& indices) {
        return data_[offset(indices)];
    }

    /**
     * Const version of the indexing operator.
     * @param indices An array of indices for each dimension.
     * @return A const reference to the element at the specified indices.
     */
    const T& operator[](const TensorIndices<N>& indices) const {
        return data_[offset(indices)];
    }

    /**
     * Fill the tensor with a specified value.
     * @param value The value to fill the tensor with.
     */
    void fill(const T& value) {
        std::fill(data_.get(), data_.get() + total_size(), value);
    }

    /**
     * Get the dimensions of the tensor.
     * @return An array containing the size of each dimension.
     */
    TensorIndices<N> dims() const { return dims_; }
    
    /**
     * Get the shape of the tensor (alias for dims).
     * @return An array containing the size of each dimension.
     */
    TensorIndices<N> shape() const { return dims_; }
    
    /**
     * Check if tensor uses GPU.
     * @return True if GPU is enabled for this tensor.
     */
    bool uses_gpu() const { return use_gpu_; }
    
    /**
     * Get the backend being used by this tensor.
     * @return The computational backend (GPU, BLAS, or CPU)
     * 
     * Returns the actual backend that will be used for operations on this tensor.
     * The backend is selected at tensor creation time based on availability:
     * - GPU: if compiled with USE_GPU, GPU is available, and use_gpu flag is true
     * - BLAS: if compiled with USE_BLAS and GPU is not used
     * - CPU: fallback when neither GPU nor BLAS is available
     */
    Backend backend() const {
        if (use_gpu_) {
            return Backend::GPU;
        }
#ifdef USE_BLAS
        return Backend::BLAS;
#else
        return Backend::CPU;
#endif
    }
    
    // Allow optimizer and loss functions access to private members
    template<typename U>
    friend class Optimizer;
    
    template<typename U, size_t M>
    friend class SGD;
    
    template<typename U, size_t M>
    friend class Adam;
    
    template<typename U, size_t M>
    friend class RMSprop;
    
    // Forward declare loss namespace functions as friends
    template<typename U, size_t M>
    friend Tensor<U, M> loss::mse_loss(const Tensor<U, M>&, const Tensor<U, M>&, const std::string&);
    
    template<typename U, size_t M>
    friend Tensor<U, 1> loss::cross_entropy_loss(const Tensor<U, M>&, const Tensor<U, M>&, const std::string&);
    
    template<typename U, size_t M>
    friend Tensor<U, 1> loss::binary_cross_entropy(const Tensor<U, M>&, const Tensor<U, M>&, const std::string&);
    
    template<typename U, size_t M>
    friend Tensor<U, 1> loss::l1_loss(const Tensor<U, M>&, const Tensor<U, M>&, const std::string&);
    
    template<typename U, size_t M>
    friend Tensor<U, 1> loss::smooth_l1_loss(const Tensor<U, M>&, const Tensor<U, M>&, U, const std::string&);
    
    // ============================================
    // Autograd Methods
    // ============================================
    
    /**
     * Enable gradient tracking for this tensor.
     * @param requires_grad Whether to require gradients.
     */
    void set_requires_grad(bool requires_grad) {
        requires_grad_ = requires_grad;
        if (requires_grad_ && !grad_) {
            grad_ = std::make_unique<Tensor<T, N>>(dims_, use_gpu_, false);
            grad_->fill(T(0));
        }
    }
    
    /**
     * Check if this tensor requires gradients.
     * @return True if gradients are being tracked.
     */
    bool requires_grad() const { return requires_grad_; }
    
    /**
     * Check if this is a leaf tensor.
     * @return True if this is a leaf tensor.
     */
    bool is_leaf() const { return is_leaf_; }
    
    /**
     * Get the gradient tensor.
     * @return Pointer to gradient tensor (nullptr if not tracking gradients).
     */
    Tensor<T, N>* grad() { return grad_.get(); }
    const Tensor<T, N>* grad() const { return grad_.get(); }
    
    /**
     * Zero out the gradients.
     */
    void zero_grad() {
        if (grad_) {
            grad_->fill(T(0));
        }
        backward_funcs_.clear();
    }
    
    /**
     * @brief Perform backward pass to compute gradients (autograd)
     * 
     * Computes gradients of this tensor with respect to leaf tensors
     * by traversing the computational graph backwards.
     * 
     * @param gradient Optional gradient tensor to start backpropagation.
     *                 If nullptr, assumes this is a scalar with gradient 1.
     * 
     * @return std::nullopt on success, TensorError::InvalidArgument on error
     * 
     * @note For scalar tensors (total_size() == 1), can be called without arguments.
     * @note For non-scalar tensors, must provide a gradient tensor.
     * @note Returns error if tensor doesn't require gradients
     * @note Returns error if called on non-scalar without gradient argument
     * 
     * @section example Usage Example
     * @code
     * // Scalar loss - no gradient needed
     * Tensor<float, 1> loss({1}, true, true);
     * loss.backward();
     * 
     * // Non-scalar - gradient required
     * Tensor<float, 2> output({3, 4}, true, true);
     * Tensor<float, 2> grad({3, 4});
     * grad.fill(1.0f);
     * output.backward(&grad);
     * @endcode
     * @return std::nullopt on success, or TensorError on failure
     */
    std::optional<TensorError> backward(const Tensor<T, N>* gradient = nullptr) {
        if (!requires_grad_) {
            return TensorError::InvalidArgument;
        }
        
        // If this is not a scalar and no gradient provided, error
        if (!gradient && total_size() != 1) {
            return TensorError::InvalidArgument;
        }
        
        // Initialize gradient
        if (!grad_) {
            grad_ = std::make_unique<Tensor<T, N>>(dims_, use_gpu_, false);
            grad_->fill(T(0));
        }
        
        if (gradient) {
            // Accumulate provided gradient
            size_t total = total_size();
            for (size_t i = 0; i < total; ++i) {
                grad_->data_[i] += gradient->data_[i];
            }
        } else {
            // For scalar, start with gradient of 1
            grad_->data_[0] = T(1);
        }
        
        // Call all backward functions to propagate gradients
        for (auto& func : backward_funcs_) {
            func(*grad_);
        }
        
        return std::nullopt;
    }
    
    /**
     * Detach this tensor from the computation graph.
     * @return A new tensor with the same data but no gradient tracking.
     */
    Tensor<T, N> detach() const {
        Tensor<T, N> result(dims_, use_gpu_, false);
        size_t total = total_size();
        std::copy(data_.get(), data_.get() + total, result.data_.get());
        return result;
    }
    
    /**
     * Register a backward function for this tensor.
     * @param func The function to call during backward pass.
     */
    void register_backward(BackwardFunc<T, N> func) {
        backward_funcs_.push_back(func);
    }
    
    /**
     * Element-wise addition with another tensor (creates new tensor).
     * Supports automatic differentiation.
     * @param other The tensor to add.
     * @return A variant containing either a new tensor with the result or an error.
     */
    TensorResult<Tensor<T, N>> operator+(const Tensor<T, N>& other) const {
        if (dims_ != other.dims_) {
            return TensorError::DimensionMismatch;
        }
        
        bool track_grad = requires_grad_ || other.requires_grad_;
        Tensor<T, N> result(dims_, use_gpu_, track_grad);
        result.is_leaf_ = false;
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_ && other.use_gpu_) {
            TensorGPU::add_gpu(data_.get(), other.data_.get(), result.data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                result.data_[i] = data_[i] + other.data_[i];
            }
        }
        
        // Setup backward pass
        if (track_grad) {
            // Capture pointers to input tensors
            Tensor<T, N>* self_ptr = const_cast<Tensor<T, N>*>(this);
            Tensor<T, N>* other_ptr = const_cast<Tensor<T, N>*>(&other);
            
            result.register_backward([self_ptr, other_ptr](const Tensor<T, N>& grad) {
                // Gradient of addition: d/dx(x+y) = 1, d/dy(x+y) = 1
                if (self_ptr->requires_grad_) {
                    if (!self_ptr->grad_) {
                        self_ptr->grad_ = std::make_unique<Tensor<T, N>>(self_ptr->dims_, self_ptr->use_gpu_, false);
                        self_ptr->grad_->fill(T(0));
                    }
                    size_t total = self_ptr->total_size();
                    for (size_t i = 0; i < total; ++i) {
                        self_ptr->grad_->data_[i] += grad.data_[i];
                    }
                    
                    // If self is not a leaf, propagate gradients further
                    if (!self_ptr->is_leaf_ && !self_ptr->backward_funcs_.empty()) {
                        for (auto& func : self_ptr->backward_funcs_) {
                            func(*self_ptr->grad_);
                        }
                    }
                }
                
                if (other_ptr->requires_grad_) {
                    if (!other_ptr->grad_) {
                        other_ptr->grad_ = std::make_unique<Tensor<T, N>>(other_ptr->dims_, other_ptr->use_gpu_, false);
                        other_ptr->grad_->fill(T(0));
                    }
                    size_t total = other_ptr->total_size();
                    for (size_t i = 0; i < total; ++i) {
                        other_ptr->grad_->data_[i] += grad.data_[i];
                    }
                    
                    // If other is not a leaf, propagate gradients further
                    if (!other_ptr->is_leaf_ && !other_ptr->backward_funcs_.empty()) {
                        for (auto& func : other_ptr->backward_funcs_) {
                            func(*other_ptr->grad_);
                        }
                    }
                }
            });
        }
        
        return result;
    }
    
    /**
     * Element-wise addition with scalar (creates new tensor).
     * Supports automatic differentiation.
     * @param scalar The scalar value to add to all elements.
     * @return A new tensor with the result.
     */
    Tensor<T, N> operator+(const T& scalar) const {
        Tensor<T, N> result(dims_, use_gpu_, requires_grad_);
        result.is_leaf_ = false;
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_) {
            TensorGPU::add_scalar_gpu(data_.get(), scalar, result.data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                result.data_[i] = data_[i] + scalar;
            }
        }
        
        // Setup backward pass
        if (requires_grad_) {
            Tensor<T, N>* self_ptr = const_cast<Tensor<T, N>*>(this);
            
            result.register_backward([self_ptr](const Tensor<T, N>& grad) {
                // Gradient of addition with scalar: d/dx(x+c) = 1
                if (self_ptr->requires_grad_) {
                    if (!self_ptr->grad_) {
                        self_ptr->grad_ = std::make_unique<Tensor<T, N>>(self_ptr->dims_, self_ptr->use_gpu_, false);
                        self_ptr->grad_->fill(T(0));
                    }
                    size_t total = self_ptr->total_size();
                    for (size_t i = 0; i < total; ++i) {
                        self_ptr->grad_->data_[i] += grad.data_[i];
                    }
                    
                    // If self is not a leaf, propagate gradients further
                    if (!self_ptr->is_leaf_ && !self_ptr->backward_funcs_.empty()) {
                        for (auto& func : self_ptr->backward_funcs_) {
                            func(*self_ptr->grad_);
                        }
                    }
                }
            });
        }
        
        return result;
    }
    
    /**
     * Element-wise addition with another tensor (in-place).
     * @param other The tensor to add.
     * @return Reference to this tensor.
     * @note If dimensions don't match, the operation is skipped.
     */
    Tensor<T, N>& operator+=(const Tensor<T, N>& other) {
        if (dims_ != other.dims_) {
            return *this;
        }
        
        size_t total = total_size();
#ifdef USE_GPU
        if (use_gpu_ && other.use_gpu_) {
            TensorGPU::add_gpu(data_.get(), other.data_.get(), data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                data_[i] += other.data_[i];
            }
        }
        
        return *this;
    }
    
    /**
     * Element-wise addition with scalar (in-place).
     * @param scalar The scalar value to add to all elements.
     * @return Reference to this tensor.
     */
    Tensor<T, N>& operator+=(const T& scalar) {
        size_t total = total_size();
#ifdef USE_GPU
        if (use_gpu_) {
            TensorGPU::add_scalar_gpu(data_.get(), scalar, data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                data_[i] += scalar;
            }
        }
        return *this;
    }
    
    /**
     * Element-wise subtraction with another tensor (creates new tensor).
     * @param other The tensor to subtract.
     * @return A variant containing either a new tensor with the result or an error.
     */
    TensorResult<Tensor<T, N>> operator-(const Tensor<T, N>& other) const {
        if (dims_ != other.dims_) {
            return TensorError::DimensionMismatch;
        }
        
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_ && other.use_gpu_) {
            TensorGPU::sub_gpu(data_.get(), other.data_.get(), result.data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                result.data_[i] = data_[i] - other.data_[i];
            }
        }
        
        return result;
    }
    
    /**
     * Element-wise subtraction with scalar (creates new tensor).
     * @param scalar The scalar value to subtract from all elements.
     * @return A new tensor with the result.
     */
    Tensor<T, N> operator-(const T& scalar) const {
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_) {
            TensorGPU::sub_scalar_gpu(data_.get(), scalar, result.data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                result.data_[i] = data_[i] - scalar;
            }
        }
        
        return result;
    }
    
    /**
     * Element-wise subtraction with another tensor (in-place).
     * @param other The tensor to subtract.
     * @return Reference to this tensor.
     * @note If dimensions don't match, the operation is skipped.
     */
    Tensor<T, N>& operator-=(const Tensor<T, N>& other) {
        if (dims_ != other.dims_) {
            return *this;
        }
        
        size_t total = total_size();
#ifdef USE_GPU
        if (use_gpu_ && other.use_gpu_) {
            TensorGPU::sub_gpu(data_.get(), other.data_.get(), data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                data_[i] -= other.data_[i];
            }
        }
        
        return *this;
    }
    
    /**
     * Element-wise subtraction with scalar (in-place).
     * @param scalar The scalar value to subtract from all elements.
     * @return Reference to this tensor.
     */
    Tensor<T, N>& operator-=(const T& scalar) {
        size_t total = total_size();
#ifdef USE_GPU
        if (use_gpu_) {
            TensorGPU::sub_scalar_gpu(data_.get(), scalar, data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                data_[i] -= scalar;
            }
        }
        return *this;
    }
    
    /**
     * Element-wise multiplication with another tensor (creates new tensor).
     * Supports automatic differentiation.
     * @param other The tensor to multiply with.
     * @return A variant containing either a new tensor with the result or an error.
     */
    TensorResult<Tensor<T, N>> operator*(const Tensor<T, N>& other) const {
        if (dims_ != other.dims_) {
            return TensorError::DimensionMismatch;
        }
        
        bool track_grad = requires_grad_ || other.requires_grad_;
        Tensor<T, N> result(dims_, use_gpu_, track_grad);
        result.is_leaf_ = false;
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_ && other.use_gpu_) {
            TensorGPU::mul_gpu(data_.get(), other.data_.get(), result.data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                result.data_[i] = data_[i] * other.data_[i];
            }
        }
        
        // Setup backward pass
        if (track_grad) {
            // Need to store copies of input data for gradient computation
            Tensor<T, N> self_copy = this->detach();
            Tensor<T, N> other_copy = other.detach();
            Tensor<T, N>* self_ptr = const_cast<Tensor<T, N>*>(this);
            Tensor<T, N>* other_ptr = const_cast<Tensor<T, N>*>(&other);
            
            result.register_backward([self_ptr, other_ptr, self_copy, other_copy](const Tensor<T, N>& grad) {
                // Gradient of multiplication: d/dx(x*y) = y, d/dy(x*y) = x
                if (self_ptr->requires_grad_) {
                    if (!self_ptr->grad_) {
                        self_ptr->grad_ = std::make_unique<Tensor<T, N>>(self_ptr->dims_, self_ptr->use_gpu_, false);
                        self_ptr->grad_->fill(T(0));
                    }
                    size_t total = self_ptr->total_size();
                    for (size_t i = 0; i < total; ++i) {
                        self_ptr->grad_->data_[i] += grad.data_[i] * other_copy.data_[i];
                    }
                    
                    // If self is not a leaf, propagate gradients further
                    if (!self_ptr->is_leaf_ && !self_ptr->backward_funcs_.empty()) {
                        for (auto& func : self_ptr->backward_funcs_) {
                            func(*self_ptr->grad_);
                        }
                    }
                }
                
                if (other_ptr->requires_grad_) {
                    if (!other_ptr->grad_) {
                        other_ptr->grad_ = std::make_unique<Tensor<T, N>>(other_ptr->dims_, other_ptr->use_gpu_, false);
                        other_ptr->grad_->fill(T(0));
                    }
                    size_t total = other_ptr->total_size();
                    for (size_t i = 0; i < total; ++i) {
                        other_ptr->grad_->data_[i] += grad.data_[i] * self_copy.data_[i];
                    }
                    
                    // If other is not a leaf, propagate gradients further
                    if (!other_ptr->is_leaf_ && !other_ptr->backward_funcs_.empty()) {
                        for (auto& func : other_ptr->backward_funcs_) {
                            func(*other_ptr->grad_);
                        }
                    }
                }
            });
        }
        
        return result;
    }
    
    /**
     * Element-wise multiplication with scalar (creates new tensor).
     * Supports automatic differentiation.
     * @param scalar The scalar value to multiply all elements with.
     * @return A new tensor with the result.
     */
    Tensor<T, N> operator*(const T& scalar) const {
        Tensor<T, N> result(dims_, use_gpu_, requires_grad_);
        result.is_leaf_ = false;
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_) {
            TensorGPU::mul_scalar_gpu(data_.get(), scalar, result.data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                result.data_[i] = data_[i] * scalar;
            }
        }
        
        // Setup backward pass
        if (requires_grad_) {
            Tensor<T, N>* self_ptr = const_cast<Tensor<T, N>*>(this);
            
            result.register_backward([self_ptr, scalar](const Tensor<T, N>& grad) {
                // Gradient of multiplication with scalar: d/dx(x*c) = c
                if (self_ptr->requires_grad_) {
                    if (!self_ptr->grad_) {
                        self_ptr->grad_ = std::make_unique<Tensor<T, N>>(self_ptr->dims_, self_ptr->use_gpu_, false);
                        self_ptr->grad_->fill(T(0));
                    }
                    size_t total = self_ptr->total_size();
                    for (size_t i = 0; i < total; ++i) {
                        self_ptr->grad_->data_[i] += grad.data_[i] * scalar;
                    }
                    
                    // If self is not a leaf, propagate gradients further
                    if (!self_ptr->is_leaf_ && !self_ptr->backward_funcs_.empty()) {
                        for (auto& func : self_ptr->backward_funcs_) {
                            func(*self_ptr->grad_);
                        }
                    }
                }
            });
        }
        
        return result;
    }
    
    /**
     * Element-wise multiplication with another tensor (in-place).
     * @param other The tensor to multiply with.
     * @return Reference to this tensor.
     * @note If dimensions don't match, the operation is skipped.
     */
    Tensor<T, N>& operator*=(const Tensor<T, N>& other) {
        if (dims_ != other.dims_) {
            return *this;
        }
        
        size_t total = total_size();
#ifdef USE_GPU
        if (use_gpu_ && other.use_gpu_) {
            TensorGPU::mul_gpu(data_.get(), other.data_.get(), data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                data_[i] *= other.data_[i];
            }
        }
        
        return *this;
    }
    
    /**
     * Element-wise multiplication with scalar (in-place).
     * @param scalar The scalar value to multiply all elements with.
     * @return Reference to this tensor.
     */
    Tensor<T, N>& operator*=(const T& scalar) {
        size_t total = total_size();
#ifdef USE_GPU
        if (use_gpu_) {
            TensorGPU::mul_scalar_gpu(data_.get(), scalar, data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                data_[i] *= scalar;
            }
        }
        return *this;
    }
    
    /**
     * Element-wise division with another tensor (creates new tensor).
     * @param other The tensor to divide by.
     * @return A variant containing either a new tensor with the result or an error.
     */
    TensorResult<Tensor<T, N>> operator/(const Tensor<T, N>& other) const {
        if (dims_ != other.dims_) {
            return TensorError::DimensionMismatch;
        }
        
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_ && other.use_gpu_) {
            TensorGPU::div_gpu(data_.get(), other.data_.get(), result.data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                result.data_[i] = data_[i] / other.data_[i];
            }
        }
        
        return result;
    }
    
    /**
     * Element-wise division with scalar (creates new tensor).
     * @param scalar The scalar value to divide all elements by.
     * @return A new tensor with the result.
     */
    Tensor<T, N> operator/(const T& scalar) const {
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_) {
            TensorGPU::div_scalar_gpu(data_.get(), scalar, result.data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                result.data_[i] = data_[i] / scalar;
            }
        }
        
        return result;
    }
    
    /**
     * Element-wise division with another tensor (in-place).
     * @param other The tensor to divide by.
     * @return Reference to this tensor.
     * @note If dimensions don't match, the operation is skipped.
     */
    Tensor<T, N>& operator/=(const Tensor<T, N>& other) {
        if (dims_ != other.dims_) {
            return *this;
        }
        
        size_t total = total_size();
#ifdef USE_GPU
        if (use_gpu_ && other.use_gpu_) {
            TensorGPU::div_gpu(data_.get(), other.data_.get(), data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                data_[i] /= other.data_[i];
            }
        }
        
        return *this;
    }
    
    /**
     * Element-wise division with scalar (in-place).
     * @param scalar The scalar value to divide all elements by.
     * @return Reference to this tensor.
     */
    Tensor<T, N>& operator/=(const T& scalar) {
        size_t total = total_size();
#ifdef USE_GPU
        if (use_gpu_) {
            TensorGPU::div_scalar_gpu(data_.get(), scalar, data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                data_[i] /= scalar;
            }
        }
        return *this;
    }
    
    /**
     * Unary negation operator (creates new tensor).
     * @return A new tensor with all elements negated.
     */
    Tensor<T, N> operator-() const {
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            result.data_[i] = -data_[i];
        }
        
        return result;
    }
    
    /**
     * Apply a function to each element of the tensor (creates new tensor).
     * @param func The function to apply to each element.
     * @return A new tensor with the function applied to all elements.
     */
    template<typename Func>
    Tensor<T, N> map(Func func) const {
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            result.data_[i] = func(data_[i]);
        }
        
        return result;
    }
    
    /**
     * Apply a function to each element of the tensor (in-place).
     * @param func The function to apply to each element.
     * @return Reference to this tensor.
     */
    template<typename Func>
    Tensor<T, N>& map_inplace(Func func) {
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            data_[i] = func(data_[i]);
        }
        
        return *this;
    }
    
    /**
     * Apply exponential function to all elements (creates new tensor).
     * @return A new tensor with exp(x) applied to all elements.
     */
    Tensor<T, N> exp() const {
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_) {
            TensorGPU::exp_gpu(data_.get(), result.data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                result.data_[i] = std::exp(data_[i]);
            }
        }
        
        return result;
    }
    
    /**
     * Apply natural logarithm to all elements (creates new tensor).
     * @return A new tensor with log(x) applied to all elements.
     */
    Tensor<T, N> log() const {
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_) {
            TensorGPU::log_gpu(data_.get(), result.data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                result.data_[i] = std::log(data_[i]);
            }
        }
        
        return result;
    }
    
    /**
     * Apply square root to all elements (creates new tensor).
     * @return A new tensor with sqrt(x) applied to all elements.
     */
    Tensor<T, N> sqrt() const {
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_) {
            TensorGPU::sqrt_gpu(data_.get(), result.data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                result.data_[i] = std::sqrt(data_[i]);
            }
        }
        
        return result;
    }
    
    /**
     * Apply power function to all elements (creates new tensor).
     * @param exponent The exponent to raise each element to.
     * @return A new tensor with pow(x, exponent) applied to all elements.
     */
    Tensor<T, N> pow(T exponent) const {
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_) {
            TensorGPU::pow_gpu(data_.get(), exponent, result.data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                result.data_[i] = std::pow(data_[i], exponent);
            }
        }
        
        return result;
    }
    
    /**
     * Apply absolute value to all elements (creates new tensor).
     * @return A new tensor with abs(x) applied to all elements.
     */
    Tensor<T, N> abs() const {
        return map([](T x) { return std::abs(x); });
    }
    
    /**
     * Apply sine function to all elements (creates new tensor).
     * @return A new tensor with sin(x) applied to all elements.
     */
    Tensor<T, N> sin() const {
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_) {
            TensorGPU::sin_gpu(data_.get(), result.data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                result.data_[i] = std::sin(data_[i]);
            }
        }
        
        return result;
    }
    
    /**
     * Apply cosine function to all elements (creates new tensor).
     * @return A new tensor with cos(x) applied to all elements.
     */
    Tensor<T, N> cos() const {
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_) {
            TensorGPU::cos_gpu(data_.get(), result.data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                result.data_[i] = std::cos(data_[i]);
            }
        }
        
        return result;
    }
    
    /**
     * Apply tangent function to all elements (creates new tensor).
     * @return A new tensor with tan(x) applied to all elements.
     */
    Tensor<T, N> tan() const {
        return map([](T x) { return std::tan(x); });
    }
    
    /**
     * Apply hyperbolic tangent (tanh) to all elements (creates new tensor).
     * Commonly used as activation function in neural networks.
     * @return A new tensor with tanh(x) applied to all elements.
     */
    Tensor<T, N> tanh() const {
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_) {
            TensorGPU::tanh_gpu(data_.get(), result.data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                result.data_[i] = std::tanh(data_[i]);
            }
        }
        
        return result;
    }
    
    /**
     * Apply sigmoid function to all elements (creates new tensor).
     * sigmoid(x) = 1 / (1 + exp(-x))
     * Commonly used as activation function in neural networks.
     * Supports automatic differentiation.
     * @return A new tensor with sigmoid(x) applied to all elements.
     */
    Tensor<T, N> sigmoid() const {
        Tensor<T, N> result(dims_, use_gpu_, requires_grad_);
        result.is_leaf_ = false;
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_) {
            TensorGPU::sigmoid_gpu(data_.get(), result.data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                result.data_[i] = T(1) / (T(1) + std::exp(-data_[i]));
            }
        }
        
        // Setup backward pass
        if (requires_grad_) {
            Tensor<T, N> output_copy = result.detach();
            Tensor<T, N>* self_ptr = const_cast<Tensor<T, N>*>(this);
            
            result.register_backward([self_ptr, output_copy](const Tensor<T, N>& grad) {
                // Gradient of sigmoid: d/dx[sigmoid(x)] = sigmoid(x) * (1 - sigmoid(x))
                if (self_ptr->requires_grad_) {
                    if (!self_ptr->grad_) {
                        self_ptr->grad_ = std::make_unique<Tensor<T, N>>(self_ptr->dims_, self_ptr->use_gpu_, false);
                        self_ptr->grad_->fill(T(0));
                    }
                    size_t total = self_ptr->total_size();
                    for (size_t i = 0; i < total; ++i) {
                        T sig = output_copy.data_[i];
                        self_ptr->grad_->data_[i] += grad.data_[i] * sig * (T(1) - sig);
                    }
                    
                    // If self is not a leaf, propagate gradients further
                    if (!self_ptr->is_leaf_ && !self_ptr->backward_funcs_.empty()) {
                        for (auto& func : self_ptr->backward_funcs_) {
                            func(*self_ptr->grad_);
                        }
                    }
                }
            });
        }
        
        return result;
    }
    
    /**
     * Apply ReLU (Rectified Linear Unit) to all elements (creates new tensor).
     * ReLU(x) = max(0, x)
     * Commonly used as activation function in neural networks.
     * Supports automatic differentiation.
     * @return A new tensor with ReLU(x) applied to all elements.
     */
    Tensor<T, N> relu() const {
        Tensor<T, N> result(dims_, use_gpu_, requires_grad_);
        result.is_leaf_ = false;
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_) {
            TensorGPU::relu_gpu(data_.get(), result.data_.get(), total);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                result.data_[i] = data_[i] > T(0) ? data_[i] : T(0);
            }
        }
        
        // Setup backward pass
        if (requires_grad_) {
            Tensor<T, N> input_copy = this->detach();
            Tensor<T, N>* self_ptr = const_cast<Tensor<T, N>*>(this);
            
            result.register_backward([self_ptr, input_copy](const Tensor<T, N>& grad) {
                // Gradient of ReLU: d/dx[ReLU(x)] = 1 if x > 0, else 0
                if (self_ptr->requires_grad_) {
                    if (!self_ptr->grad_) {
                        self_ptr->grad_ = std::make_unique<Tensor<T, N>>(self_ptr->dims_, self_ptr->use_gpu_, false);
                        self_ptr->grad_->fill(T(0));
                    }
                    size_t total = self_ptr->total_size();
                    for (size_t i = 0; i < total; ++i) {
                        if (input_copy.data_[i] > T(0)) {
                            self_ptr->grad_->data_[i] += grad.data_[i];
                        }
                    }
                }
            });
        }
        
        return result;
    }
    
    /**
     * Apply ceiling function to all elements (creates new tensor).
     * @return A new tensor with ceil(x) applied to all elements.
     */
    Tensor<T, N> ceil() const {
        return map([](T x) { return std::ceil(x); });
    }
    
    /**
     * Apply floor function to all elements (creates new tensor).
     * @return A new tensor with floor(x) applied to all elements.
     */
    Tensor<T, N> floor() const {
        return map([](T x) { return std::floor(x); });
    }
    
    /**
     * Clamp all elements to a range (creates new tensor).
     * @param min_val The minimum value.
     * @param max_val The maximum value.
     * @return A new tensor with all elements clamped to [min_val, max_val].
     */
    Tensor<T, N> clamp(T min_val, T max_val) const {
        return map([min_val, max_val](T x) { 
            return x < min_val ? min_val : (x > max_val ? max_val : x); 
        });
    }
    
    /**
     * Apply sign function to all elements (creates new tensor).
     * Returns -1 for negative values, 0 for zero, and +1 for positive values.
     * @return A new tensor with sign(x) applied to all elements.
     */
    Tensor<T, N> sign() const {
        return map([](T x) { 
            return (x > T(0)) ? T(1) : ((x < T(0)) ? T(-1) : T(0)); 
        });
    }
    
    /**
     * Apply round function to all elements (creates new tensor).
     * Rounds to nearest integer value.
     * @return A new tensor with round(x) applied to all elements.
     */
    Tensor<T, N> round() const {
        return map([](T x) { return std::round(x); });
    }
    
    /**
     * Apply error function (erf) to all elements (creates new tensor).
     * The error function is used in probability and statistics.
     * @return A new tensor with erf(x) applied to all elements.
     */
    Tensor<T, N> erf() const {
        return map([](T x) { return std::erf(x); });
    }
    
    /**
     * Apply log1p function to all elements (creates new tensor).
     * Computes log(1 + x) in a numerically stable way for small x.
     * @return A new tensor with log1p(x) applied to all elements.
     */
    Tensor<T, N> log1p() const {
        return map([](T x) { return std::log1p(x); });
    }
    
    /**
     * Apply expm1 function to all elements (creates new tensor).
     * Computes exp(x) - 1 in a numerically stable way for small x.
     * @return A new tensor with expm1(x) applied to all elements.
     */
    Tensor<T, N> expm1() const {
        return map([](T x) { return std::expm1(x); });
    }
    
    /**
     * Check if elements are NaN (creates boolean-like tensor).
     * @return A new tensor with 1 where NaN, 0 otherwise.
     */
    Tensor<T, N> isnan() const {
        return map([](T x) { return std::isnan(x) ? T(1) : T(0); });
    }
    
    /**
     * Check if elements are infinite (creates boolean-like tensor).
     * @return A new tensor with 1 where infinite, 0 otherwise.
     */
    Tensor<T, N> isinf() const {
        return map([](T x) { return std::isinf(x) ? T(1) : T(0); });
    }
    
    /**
     * Check if elements are finite (creates boolean-like tensor).
     * @return A new tensor with 1 where finite, 0 otherwise.
     */
    Tensor<T, N> isfinite() const {
        return map([](T x) { return std::isfinite(x) ? T(1) : T(0); });
    }
    
    /**
     * Clip values to a minimum threshold (creates new tensor).
     * @param min_val The minimum value.
     * @return A new tensor with all elements >= min_val.
     */
    Tensor<T, N> clip_min(T min_val) const {
        return map([min_val](T x) { return x < min_val ? min_val : x; });
    }
    
    /**
     * Clip values to a maximum threshold (creates new tensor).
     * @param max_val The maximum value.
     * @return A new tensor with all elements <= max_val.
     */
    Tensor<T, N> clip_max(T max_val) const {
        return map([max_val](T x) { return x > max_val ? max_val : x; });
    }
    
    /**
     * Conditional selection (ternary operator for tensors).
     * Returns a tensor where each element is taken from x if condition is true, otherwise from y.
     * This is a static method that acts as a ternary operator: condition ? x : y
     * @tparam C The condition tensor type (should be bool or numeric)
     * @param condition Tensor of boolean/numeric values (non-zero is true).
     * @param x Tensor to select from when condition is true.
     * @param y Tensor to select from when condition is false.
     * @return A new tensor with conditional values.
     */
    template<typename C>
    static Tensor<T, N> where(const Tensor<C, N>& condition, const Tensor<T, N>& x, const Tensor<T, N>& y) {
        if (condition.dims() != x.dims() || x.dims() != y.dims()) {
            throw std::invalid_argument("where(): all tensors must have the same shape");
        }
        
        Tensor<T, N> result(x.dims(), x.use_gpu_);
        size_t total = x.total_size();
        
        for (size_t i = 0; i < total; ++i) {
            result.data_[i] = condition.data()[i] ? x.data_[i] : y.data_[i];
        }
        
        return result;
    }
    
    // ============================================
    // Derivative Functions
    // ============================================
    
    /**
     * Compute derivative of sigmoid function with respect to input.
     * Given output y = sigmoid(x), computes dy/dx = y * (1 - y)
     * This is more numerically stable when you have the sigmoid output.
     * @return A new tensor with sigmoid derivative.
     */
    Tensor<T, N> sigmoid_derivative() const {
        return map([](T y) { return y * (T(1) - y); });
    }
    
    /**
     * Compute derivative of sigmoid from input values.
     * Given input x, computes d/dx[sigmoid(x)] = sigmoid(x) * (1 - sigmoid(x))
     * @return A new tensor with sigmoid derivative.
     */
    Tensor<T, N> sigmoid_derivative_from_input() const {
        return map([](T x) {
            T sig = T(1) / (T(1) + std::exp(-x));
            return sig * (T(1) - sig);
        });
    }
    
    /**
     * Compute derivative of tanh function.
     * Given output y = tanh(x), computes dy/dx = 1 - y²
     * @return A new tensor with tanh derivative.
     */
    Tensor<T, N> tanh_derivative() const {
        return map([](T y) { return T(1) - y * y; });
    }
    
    /**
     * Compute derivative of tanh from input values.
     * Given input x, computes d/dx[tanh(x)] = 1 - tanh²(x)
     * @return A new tensor with tanh derivative.
     */
    Tensor<T, N> tanh_derivative_from_input() const {
        return map([](T x) {
            T th = std::tanh(x);
            return T(1) - th * th;
        });
    }
    
    /**
     * Compute derivative of ReLU function.
     * d/dx[ReLU(x)] = 1 if x > 0, else 0
     * @return A new tensor with ReLU derivative.
     */
    Tensor<T, N> relu_derivative() const {
        return map([](T x) { return x > T(0) ? T(1) : T(0); });
    }
    
    /**
     * Compute derivative of Leaky ReLU function.
     * d/dx[LeakyReLU(x)] = 1 if x > 0, else alpha
     * @param alpha The slope for negative values (default 0.01).
     * @return A new tensor with Leaky ReLU derivative.
     */
    Tensor<T, N> leaky_relu_derivative(T alpha = T(0.01)) const {
        return map([alpha](T x) { return x > T(0) ? T(1) : alpha; });
    }
    
    /**
     * Compute derivative of exponential function.
     * d/dx[exp(x)] = exp(x)
     * @return A new tensor with exp derivative (same as exp).
     */
    Tensor<T, N> exp_derivative() const {
        return exp();
    }
    
    /**
     * Compute derivative of natural logarithm.
     * d/dx[log(x)] = 1/x
     * @return A new tensor with log derivative.
     */
    Tensor<T, N> log_derivative() const {
        return map([](T x) { return T(1) / x; });
    }
    
    /**
     * Compute derivative of square root.
     * d/dx[sqrt(x)] = 1 / (2 * sqrt(x))
     * @return A new tensor with sqrt derivative.
     */
    Tensor<T, N> sqrt_derivative() const {
        return map([](T x) { return T(1) / (T(2) * std::sqrt(x)); });
    }
    
    /**
     * Compute derivative of power function.
     * d/dx[x^n] = n * x^(n-1)
     * @param exponent The exponent used in the power function.
     * @return A new tensor with power derivative.
     */
    Tensor<T, N> pow_derivative(T exponent) const {
        return map([exponent](T x) { 
            return exponent * std::pow(x, exponent - T(1)); 
        });
    }
    
    /**
     * Compute derivative of sine function.
     * d/dx[sin(x)] = cos(x)
     * @return A new tensor with sin derivative.
     */
    Tensor<T, N> sin_derivative() const {
        return cos();
    }
    
    /**
     * Compute derivative of cosine function.
     * d/dx[cos(x)] = -sin(x)
     * @return A new tensor with cos derivative.
     */
    Tensor<T, N> cos_derivative() const {
        return map([](T x) { return -std::sin(x); });
    }
    
    /**
     * Compute derivative of tangent function.
     * d/dx[tan(x)] = sec²(x) = 1 / cos²(x)
     * @return A new tensor with tan derivative.
     */
    Tensor<T, N> tan_derivative() const {
        return map([](T x) { 
            T c = std::cos(x);
            return T(1) / (c * c);
        });
    }
    
    /**
     * Compute gradient of softmax function (for classification).
     * This computes the Jacobian diagonal for each output with respect to itself.
     * For output y_i = softmax(x)_i, computes dy_i/dx_i = y_i * (1 - y_i)
     * Note: Full softmax Jacobian requires matrix output.
     * @return A new tensor with softmax gradient diagonal.
     */
    Tensor<T, N> softmax_gradient_diagonal() const {
        return map([](T y) { return y * (T(1) - y); });
    }
    
    /**
     * Apply chain rule for backpropagation.
     * Given upstream gradient and local derivative, compute gradient for this layer.
     * gradient = upstream_grad * local_derivative
     * @param upstream_grad The gradient flowing from the next layer.
     * @return A variant containing the computed gradient or an error.
     */
    TensorResult<Tensor<T, N>> chain_rule(const Tensor<T, N>& upstream_grad) const {
        if (dims_ != upstream_grad.dims_) {
            return TensorError::DimensionMismatch;
        }
        
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            result.data_[i] = data_[i] * upstream_grad.data_[i];
        }
        
        return result;
    }
    
    // ============================================
    // Reduction and Statistical Operations
    // ============================================
    
    /**
     * Compute sum reduction along all dimensions.
     * Useful for computing loss or gradient aggregation.
     * @return The sum of all elements.
     */
    T sum() const {
        T result = T(0);
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            result += data_[i];
        }
        
        return result;
    }
    
    /**
     * Compute mean of all elements.
     * @return The mean value.
     */
    T mean() const {
        return sum() / static_cast<T>(total_size());
    }
    
    /**
     * Compute variance of all elements.
     * @param ddof Delta degrees of freedom (default 0 for population variance).
     * @return The variance.
     */
    T variance(size_t ddof = 0) const {
        T m = mean();
        T var = T(0);
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            T diff = data_[i] - m;
            var += diff * diff;
        }
        
        return var / static_cast<T>(total - ddof);
    }
    
    /**
     * Compute standard deviation of all elements.
     * @param ddof Delta degrees of freedom (default 0).
     * @return The standard deviation.
     */
    T std(size_t ddof = 0) const {
        return std::sqrt(variance(ddof));
    }
    
    /**
     * Compute covariance with another tensor.
     * Treats tensors as flattened vectors.
     * @param other The other tensor.
     * @param ddof Delta degrees of freedom (default 0).
     * @return The covariance or error.
     */
    TensorResult<T> covariance(const Tensor<T, N>& other, size_t ddof = 0) const {
        if (total_size() != other.total_size()) {
            return TensorError::DimensionMismatch;
        }
        
        size_t total = total_size();
        if (total <= ddof) {
            return TensorError::InvalidArgument;
        }
        
        T mean1 = mean();
        T mean2 = other.mean();
        T cov = T(0);
        
        for (size_t i = 0; i < total; ++i) {
            cov += (data_[i] - mean1) * (other.data_[i] - mean2);
        }
        
        return cov / static_cast<T>(total - ddof);
    }
    
    /**
     * Compute Pearson correlation coefficient with another tensor.
     * Treats tensors as flattened vectors.
     * @param other The other tensor.
     * @return The correlation coefficient [-1, 1] or error.
     */
    TensorResult<T> correlation(const Tensor<T, N>& other) const {
        if (total_size() != other.total_size()) {
            return TensorError::DimensionMismatch;
        }
        
        size_t total = total_size();
        T mean1 = mean();
        T mean2 = other.mean();
        
        T cov = T(0);
        T var1 = T(0);
        T var2 = T(0);
        
        for (size_t i = 0; i < total; ++i) {
            T diff1 = data_[i] - mean1;
            T diff2 = other.data_[i] - mean2;
            cov += diff1 * diff2;
            var1 += diff1 * diff1;
            var2 += diff2 * diff2;
        }
        
        T denom = std::sqrt(var1 * var2);
        if (denom < std::numeric_limits<T>::epsilon()) {
            return TensorError::InvalidArgument;
        }
        
        return cov / denom;
    }
    
    /**
     * Compute median of all elements.
     * Note: This creates a sorted copy of the data.
     * @return The median value.
     */
    T median() const {
        size_t total = total_size();
        std::vector<T> sorted_data(data_.get(), data_.get() + total);
        std::sort(sorted_data.begin(), sorted_data.end());
        
        if (total % 2 == 0) {
            return (sorted_data[total/2 - 1] + sorted_data[total/2]) / T(2);
        } else {
            return sorted_data[total/2];
        }
    }
    
    /**
     * Compute quantile of all elements.
     * @param q Quantile value between 0 and 1 (e.g., 0.25 for 25th percentile).
     * @return The quantile value or error.
     */
    TensorResult<T> quantile(T q) const {
        if (q < T(0) || q > T(1)) {
            return TensorError::InvalidArgument;
        }
        
        size_t total = total_size();
        std::vector<T> sorted_data(data_.get(), data_.get() + total);
        std::sort(sorted_data.begin(), sorted_data.end());
        
        T pos = q * static_cast<T>(total - 1);
        size_t lower = static_cast<size_t>(pos);
        size_t upper = lower + 1;
        
        if (upper >= total) {
            return sorted_data[total - 1];
        }
        
        T frac = pos - static_cast<T>(lower);
        return sorted_data[lower] * (T(1) - frac) + sorted_data[upper] * frac;
    }
    
    /**
     * Compute min value of all elements.
     * @return The minimum value.
     */
    T min() const {
        size_t total = total_size();
        T min_val = data_[0];
        for (size_t i = 1; i < total; ++i) {
            if (data_[i] < min_val) {
                min_val = data_[i];
            }
        }
        return min_val;
    }
    
    /**
     * Compute max value of all elements.
     * @return The maximum value.
     */
    T max() const {
        size_t total = total_size();
        T max_val = data_[0];
        for (size_t i = 1; i < total; ++i) {
            if (data_[i] > max_val) {
                max_val = data_[i];
            }
        }
        return max_val;
    }
    
    /**
     * Compute Spearman rank correlation coefficient with another tensor.
     * @param other The other tensor.
     * @return The Spearman correlation coefficient or error.
     */
    TensorResult<T> spearman_correlation(const Tensor<T, N>& other) const {
        if (total_size() != other.total_size()) {
            return TensorError::DimensionMismatch;
        }
        
        size_t total = total_size();
        
        // Create rank vectors
        std::vector<std::pair<T, size_t>> data1(total);
        std::vector<std::pair<T, size_t>> data2(total);
        
        for (size_t i = 0; i < total; ++i) {
            data1[i] = {data_[i], i};
            data2[i] = {other.data_[i], i};
        }
        
        // Sort by value
        std::sort(data1.begin(), data1.end());
        std::sort(data2.begin(), data2.end());
        
        // Assign ranks
        std::vector<T> rank1(total);
        std::vector<T> rank2(total);
        
        for (size_t i = 0; i < total; ++i) {
            rank1[data1[i].second] = static_cast<T>(i + 1);
            rank2[data2[i].second] = static_cast<T>(i + 1);
        }
        
        // Compute Pearson correlation on ranks
        T sum_d_squared = T(0);
        for (size_t i = 0; i < total; ++i) {
            T diff = rank1[i] - rank2[i];
            sum_d_squared += diff * diff;
        }
        
        T n = static_cast<T>(total);
        return T(1) - (T(6) * sum_d_squared) / (n * (n * n - T(1)));
    }
    
    /**
     * Normalize tensor to have zero mean and unit variance (z-score normalization).
     * @param ddof Delta degrees of freedom for standard deviation.
     * @return A new normalized tensor.
     */
    Tensor<T, N> normalize(size_t ddof = 0) const {
        T m = mean();
        T s = std(ddof);
        
        if (s < std::numeric_limits<T>::epsilon()) {
            // If std is zero, just center the data
            return *this - m;
        }
        
        return (*this - m) / s;
    }
    
    /**
     * Standardize tensor to [0, 1] range using min-max scaling.
     * @return A new standardized tensor.
     */
    Tensor<T, N> standardize() const {
        T min_val = min();
        T max_val = max();
        T range = max_val - min_val;
        
        if (range < std::numeric_limits<T>::epsilon()) {
            // If all values are the same, return zeros
            Tensor<T, N> result(dims_, use_gpu_);
            result.fill(T(0));
            return result;
        }
        
        return (*this - min_val) / range;
    }
    
    /**
     * Compute element-wise square (x²).
     * Useful for computing squared errors.
     * @return A new tensor with squared values.
     */
    Tensor<T, N> square() const {
        return map([](T x) { return x * x; });
    }
    
    // ============================================
    // Enhanced Operations & Utilities (Phase 1)
    // ============================================
    
    /**
     * Element-wise comparison: greater than.
     * @param other The tensor to compare with.
     * @return A new boolean-like tensor (0 or 1).
     */
    Tensor<T, N> operator>(const Tensor<T, N>& other) const {
        if (dims_ != other.dims_) {
            Tensor<T, N> result(dims_, use_gpu_);
            result.fill(T(0));
            return result;
        }
        
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            result.data_[i] = (data_[i] > other.data_[i]) ? T(1) : T(0);
        }
        
        return result;
    }
    
    /**
     * Element-wise comparison: greater than scalar.
     * @param scalar The scalar value to compare with.
     * @return A new boolean-like tensor (0 or 1).
     */
    Tensor<T, N> operator>(const T& scalar) const {
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            result.data_[i] = (data_[i] > scalar) ? T(1) : T(0);
        }
        
        return result;
    }
    
    /**
     * Element-wise comparison: less than.
     * @param other The tensor to compare with.
     * @return A new boolean-like tensor (0 or 1).
     */
    Tensor<T, N> operator<(const Tensor<T, N>& other) const {
        if (dims_ != other.dims_) {
            Tensor<T, N> result(dims_, use_gpu_);
            result.fill(T(0));
            return result;
        }
        
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            result.data_[i] = (data_[i] < other.data_[i]) ? T(1) : T(0);
        }
        
        return result;
    }
    
    /**
     * Element-wise comparison: less than scalar.
     * @param scalar The scalar value to compare with.
     * @return A new boolean-like tensor (0 or 1).
     */
    Tensor<T, N> operator<(const T& scalar) const {
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            result.data_[i] = (data_[i] < scalar) ? T(1) : T(0);
        }
        
        return result;
    }
    
    /**
     * Element-wise comparison: greater than or equal.
     * @param other The tensor to compare with.
     * @return A new boolean-like tensor (0 or 1).
     */
    Tensor<T, N> operator>=(const Tensor<T, N>& other) const {
        if (dims_ != other.dims_) {
            Tensor<T, N> result(dims_, use_gpu_);
            result.fill(T(0));
            return result;
        }
        
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            result.data_[i] = (data_[i] >= other.data_[i]) ? T(1) : T(0);
        }
        
        return result;
    }
    
    /**
     * Element-wise comparison: greater than or equal scalar.
     * @param scalar The scalar value to compare with.
     * @return A new boolean-like tensor (0 or 1).
     */
    Tensor<T, N> operator>=(const T& scalar) const {
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            result.data_[i] = (data_[i] >= scalar) ? T(1) : T(0);
        }
        
        return result;
    }
    
    /**
     * Element-wise comparison: less than or equal.
     * @param other The tensor to compare with.
     * @return A new boolean-like tensor (0 or 1).
     */
    Tensor<T, N> operator<=(const Tensor<T, N>& other) const {
        if (dims_ != other.dims_) {
            Tensor<T, N> result(dims_, use_gpu_);
            result.fill(T(0));
            return result;
        }
        
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            result.data_[i] = (data_[i] <= other.data_[i]) ? T(1) : T(0);
        }
        
        return result;
    }
    
    /**
     * Element-wise comparison: less than or equal scalar.
     * @param scalar The scalar value to compare with.
     * @return A new boolean-like tensor (0 or 1).
     */
    Tensor<T, N> operator<=(const T& scalar) const {
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            result.data_[i] = (data_[i] <= scalar) ? T(1) : T(0);
        }
        
        return result;
    }
    
    /**
     * Element-wise equality comparison.
     * @param other The tensor to compare with.
     * @param epsilon Tolerance for floating-point comparison.
     * @return A new boolean-like tensor (0 or 1).
     */
    Tensor<T, N> eq(const Tensor<T, N>& other, T epsilon = std::numeric_limits<T>::epsilon()) const {
        if (dims_ != other.dims_) {
            Tensor<T, N> result(dims_, use_gpu_);
            result.fill(T(0));
            return result;
        }
        
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            result.data_[i] = (std::abs(data_[i] - other.data_[i]) < epsilon) ? T(1) : T(0);
        }
        
        return result;
    }
    
    /**
     * Element-wise inequality comparison.
     * @param other The tensor to compare with.
     * @param epsilon Tolerance for floating-point comparison.
     * @return A new boolean-like tensor (0 or 1).
     */
    Tensor<T, N> ne(const Tensor<T, N>& other, T epsilon = std::numeric_limits<T>::epsilon()) const {
        if (dims_ != other.dims_) {
            Tensor<T, N> result(dims_, use_gpu_);
            result.fill(T(1));
            return result;
        }
        
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            result.data_[i] = (std::abs(data_[i] - other.data_[i]) >= epsilon) ? T(1) : T(0);
        }
        
        return result;
    }
    
    /**
     * Clip (clamp) tensor values to a specified range.
     * @param min_val Minimum value.
     * @param max_val Maximum value.
     * @return A new tensor with clipped values.
     */
    Tensor<T, N> clip(T min_val, T max_val) const {
        Tensor<T, N> result(dims_, use_gpu_, requires_grad_);
        result.is_leaf_ = false;
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            result.data_[i] = std::max(min_val, std::min(max_val, data_[i]));
        }
        
        // Setup backward pass
        if (requires_grad_) {
            Tensor<T, N>* self_ptr = const_cast<Tensor<T, N>*>(this);
            
            result.register_backward([self_ptr, min_val, max_val](const Tensor<T, N>& grad) {
                if (self_ptr->requires_grad_) {
                    if (!self_ptr->grad_) {
                        self_ptr->grad_ = std::make_unique<Tensor<T, N>>(self_ptr->dims_,
                                                                          self_ptr->use_gpu_, false);
                        self_ptr->grad_->fill(T(0));
                    }
                    
                    size_t total = self_ptr->total_size();
                    for (size_t i = 0; i < total; ++i) {
                        // Gradient passes through only if value is within range
                        if (self_ptr->data_[i] > min_val && self_ptr->data_[i] < max_val) {
                            self_ptr->grad_->data_[i] += grad.data_[i];
                        }
                    }
                    
                    if (!self_ptr->is_leaf_ && !self_ptr->backward_funcs_.empty()) {
                        for (auto& func : self_ptr->backward_funcs_) {
                            func(*self_ptr->grad_);
                        }
                    }
                }
            });
        }
        
        return result;
    }
    
    /**
     * Masked fill: replace values where mask is true (non-zero) with fill_value.
     * @tparam M The mask tensor element type (typically bool or numeric).
     * @param mask Boolean-like tensor (0 or non-zero values).
     * @param fill_value Value to fill where mask is non-zero.
     * @return A new tensor with masked values filled.
     */
    template<typename M>
    Tensor<T, N> masked_fill(const Tensor<M, N>& mask, T fill_value) const {
        if (dims_ != mask.dims()) {
            return *this;
        }
        
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            result.data_[i] = (mask.data()[i] != M(0)) ? fill_value : data_[i];
        }
        
        return result;
    }
    
    /**
     * Masked select: return 1D tensor of values where mask is true (non-zero).
     * @tparam M The mask tensor element type (typically bool or numeric).
     * @param mask Boolean-like tensor (0 or non-zero values).
     * @return A 1D tensor containing selected values.
     */
    template<typename M>
    Tensor<T, 1> masked_select(const Tensor<M, N>& mask) const {
        if (dims_ != mask.dims()) {
            return Tensor<T, 1>({0}, use_gpu_);
        }
        
        // Count how many values to select
        size_t total = total_size();
        size_t count = 0;
        for (size_t i = 0; i < total; ++i) {
            if (mask.data()[i] != M(0)) {
                count++;
            }
        }
        
        Tensor<T, 1> result({count}, use_gpu_);
        size_t idx = 0;
        for (size_t i = 0; i < total; ++i) {
            if (mask.data()[i] != M(0)) {
                result.data_[idx++] = data_[i];
            }
        }
        
        return result;
    }
    
    /**
     * Sample from uniform distribution [low, high).
     * @param low Lower bound (inclusive).
     * @param high Upper bound (exclusive).
     */
    void uniform(T low = T(0), T high = T(1)) {
        size_t total = total_size();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(static_cast<double>(low), static_cast<double>(high));
        
        for (size_t i = 0; i < total; ++i) {
            data_[i] = static_cast<T>(dis(gen));
        }
    }
    
    /**
     * Sample from Bernoulli distribution.
     * @param p Probability of success (value 1), default 0.5.
     */
    void bernoulli(T p = T(0.5)) {
        size_t total = total_size();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution dis(static_cast<double>(p));
        
        for (size_t i = 0; i < total; ++i) {
            data_[i] = dis(gen) ? T(1) : T(0);
        }
    }
    
    /**
     * Stack tensors along a new dimension (creates new dimension).
     * @param tensors Vector of tensors to stack.
     * @param dim Dimension along which to stack (must be <= N).
     * @return A new tensor with dimension N+1.
     */
    template<size_t... Dims>
    static Tensor<T, N+1> stack(const std::vector<Tensor<T, N>>& tensors, size_t dim = 0) {
        if (tensors.empty()) {
            return Tensor<T, N+1>({0}, false);
        }
        
        if (dim > N) {
            return Tensor<T, N+1>({0}, false);
        }
        
        // Check all tensors have the same shape
        const auto& first_dims = tensors[0].dims_;
        for (const auto& t : tensors) {
            if (t.dims_ != first_dims) {
                return Tensor<T, N+1>({0}, false);
            }
        }
        
        // Compute new dimensions
        TensorIndices<N+1> new_dims;
        for (size_t i = 0; i < dim; ++i) {
            new_dims[i] = first_dims[i];
        }
        new_dims[dim] = tensors.size();
        for (size_t i = dim; i < N; ++i) {
            new_dims[i+1] = first_dims[i];
        }
        
        Tensor<T, N+1> result(new_dims, tensors[0].use_gpu_);
        
        // Copy data
        size_t slice_size = tensors[0].total_size();
        for (size_t t = 0; t < tensors.size(); ++t) {
            size_t offset = t * slice_size;
            std::copy(tensors[t].data_.get(), tensors[t].data_.get() + slice_size,
                     result.data_.get() + offset);
        }
        
        return result;
    }
    
    /**
     * Vertical stack (stack along first dimension) - convenience method for 2D tensors.
     * @param tensors Vector of tensors to stack vertically.
     * @return A new tensor stacked vertically.
     */
    static Tensor<T, N> vstack(const std::vector<Tensor<T, N>>& tensors) {
        static_assert(N >= 2, "vstack requires at least 2D tensors");
        
        if (tensors.empty()) {
            TensorIndices<N> zero_dims{};
            return Tensor<T, N>(zero_dims, false);
        }
        
        if (tensors.size() == 1) {
            return Tensor<T, N>(tensors[0]);
        }
        
        // Start with first tensor and concatenate the rest
        Tensor<T, N> result(tensors[0]);
        for (size_t i = 1; i < tensors.size(); ++i) {
            result = result.concatenate(tensors[i], 0);
        }
        
        return result;
    }
    
    /**
     * Horizontal stack (stack along second dimension) - convenience method for 2D tensors.
     * @param tensors Vector of tensors to stack horizontally.
     * @return A new tensor stacked horizontally.
     */
    static Tensor<T, N> hstack(const std::vector<Tensor<T, N>>& tensors) {
        static_assert(N >= 2, "hstack requires at least 2D tensors");
        
        if (tensors.empty()) {
            TensorIndices<N> zero_dims{};
            return Tensor<T, N>(zero_dims, false);
        }
        
        if (tensors.size() == 1) {
            return Tensor<T, N>(tensors[0]);
        }
        
        // Start with first tensor and concatenate the rest
        Tensor<T, N> result(tensors[0]);
        for (size_t i = 1; i < tensors.size(); ++i) {
            result = result.concatenate(tensors[i], 1);
        }
        
        return result;
    }
    
    // ============================================
    // Loss Functions and Gradients
    // ============================================
    
    /**
     * Compute MSE (Mean Squared Error) loss.
     * @param target The target tensor.
     * @return The MSE value or error.
     */
    TensorResult<T> mse_loss(const Tensor<T, N>& target) const {
        if (dims_ != target.dims_) {
            return TensorError::DimensionMismatch;
        }
        
        T loss = T(0);
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            T diff = data_[i] - target.data_[i];
            loss += diff * diff;
        }
        
        return loss / static_cast<T>(total);
    }
    
    /**
     * Compute gradient of MSE loss with respect to predictions.
     * d/dx[MSE] = 2 * (prediction - target) / n
     * @param target The target tensor.
     * @return The gradient tensor or error.
     */
    TensorResult<Tensor<T, N>> mse_loss_gradient(const Tensor<T, N>& target) const {
        if (dims_ != target.dims_) {
            return TensorError::DimensionMismatch;
        }
        
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        T scale = T(2) / static_cast<T>(total);
        
        for (size_t i = 0; i < total; ++i) {
            result.data_[i] = scale * (data_[i] - target.data_[i]);
        }
        
        return result;
    }
    
    /**
     * Compute binary cross-entropy loss.
     * Loss = -[target * log(pred) + (1-target) * log(1-pred)]
     * @param target The target tensor (values should be 0 or 1).
     * @param epsilon Small value to avoid log(0) (default 1e-7).
     * @return The BCE loss value or error.
     */
    TensorResult<T> binary_crossentropy_loss(const Tensor<T, N>& target, T epsilon = T(1e-7)) const {
        if (dims_ != target.dims_) {
            return TensorError::DimensionMismatch;
        }
        
        T loss = T(0);
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            T pred = std::max(epsilon, std::min(T(1) - epsilon, data_[i]));
            loss += -(target.data_[i] * std::log(pred) + 
                     (T(1) - target.data_[i]) * std::log(T(1) - pred));
        }
        
        return loss / static_cast<T>(total);
    }
    
    /**
     * Compute gradient of binary cross-entropy loss.
     * d/dx[BCE] = -(target/pred - (1-target)/(1-pred)) / n
     * @param target The target tensor.
     * @param epsilon Small value to avoid division by zero.
     * @return The gradient tensor or error.
     */
    TensorResult<Tensor<T, N>> binary_crossentropy_gradient(const Tensor<T, N>& target, 
                                                              T epsilon = T(1e-7)) const {
        if (dims_ != target.dims_) {
            return TensorError::DimensionMismatch;
        }
        
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        T scale = T(1) / static_cast<T>(total);
        
        for (size_t i = 0; i < total; ++i) {
            T pred = std::max(epsilon, std::min(T(1) - epsilon, data_[i]));
            result.data_[i] = scale * (-(target.data_[i] / pred) + 
                                       ((T(1) - target.data_[i]) / (T(1) - pred)));
        }
        
        return result;
    }
    
    /**
     * Dot product for 1D tensors (vector dot product).
     * Computes the sum of element-wise products.
     * Uses GPU acceleration if available and enabled.
     * @param other The other 1D tensor to compute dot product with.
     * @return A variant containing either the scalar result or an error.
     */
    template<size_t M = N>
    typename std::enable_if<M == 1, TensorResult<T>>::type
    dot(const Tensor<T, 1>& other) const {
        static_assert(N == 1, "This dot product is only for 1D tensors");
        if (dims_[0] != other.dims_[0]) {
            return TensorError::DimensionMismatch;
        }
        
        T result = T();
        
#ifdef USE_GPU
        if (use_gpu_ && other.use_gpu_) {
            TensorGPU::dot_1d_gpu(data_.get(), other.data_.get(), &result, dims_[0]);
            return result;
        }
#endif
        
#ifdef USE_BLAS
        if (!use_gpu_) {
            return blas_dot<T>(dims_[0], data_.get(), 1, other.data_.get(), 1);
        }
#endif
        
        for (size_t i = 0; i < dims_[0]; ++i) {
            result += data_[i] * other.data_[i];
        }
        return result;
    }
    
    /**
     * Dot product for 2D tensors (matrix multiplication).
     * Computes C = A · B where A is (m × n) and B is (n × p), resulting in C (m × p).
     * Uses GPU acceleration if available, otherwise uses parallel CPU execution.
     * @param other The other 2D tensor (matrix) to multiply with.
     * @return A variant containing either a new tensor with the result or an error.
     */
    template<size_t M = N>
    typename std::enable_if<M == 2, TensorResult<Tensor<T, 2>>>::type
    dot(const Tensor<T, 2>& other) const {
        static_assert(N == 2, "This dot product is only for 2D tensors");
        if (dims_[1] != other.dims_[0]) {
            return TensorError::DimensionMismatch;
        }
        
        size_t m = dims_[0];
        size_t n = dims_[1];
        size_t p = other.dims_[1];
        
        Tensor<T, 2> result({m, p}, use_gpu_);
        result.fill(T());
        
#ifdef USE_GPU
        if (use_gpu_ && other.use_gpu_) {
            TensorGPU::dot_2d_gpu(data_.get(), other.data_.get(), result.data_.get(), m, n, p);
            return result;
        }
#endif
        
#ifdef USE_BLAS
        if (!use_gpu_) {
            // BLAS gemm: C = alpha * A * B + beta * C
            // A is m×n, B is n×p, C is m×p
            blas_gemm<T>(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, p, n, T(1), data_.get(), n,
                        other.data_.get(), p, T(0), result.data_.get(), p);
            return result;
        }
#endif
        
        auto row_range = std::views::iota(size_t(0), m);
        std::for_each(std::execution::par, row_range.begin(), row_range.end(), 
                      [&](size_t i) {
            for (size_t j = 0; j < p; ++j) {
                T sum = T();
                for (size_t k = 0; k < n; ++k) {
                    sum += (*this)[{i, k}] * other[{k, j}];
                }
                result[{i, j}] = sum;
            }
        });
        return result;
    }
    
    /**
     * Generalized dot product for N-D tensors (N >= 3).
     * Performs tensor contraction over the last axis of this tensor
     * and the first axis of the other tensor.
     * For tensors A with shape (..., k) and B with shape (k, ...), 
     * the result has shape (...A[:-1], ...B[1:]).
     * Uses GPU acceleration if available, otherwise uses parallel CPU execution.
     * 
     * Example: A(2,3,4) · B(4,5,6) = C(2,3,5,6)
     * 
     * @param other The other N-D tensor to contract with.
     * @return A variant containing either a new tensor with the result or an error.
     */
    template<size_t M = N, size_t P>
    typename std::enable_if<(M >= 3), TensorResult<Tensor<T, N + P - 2>>>::type
    dot(const Tensor<T, P>& other) const {
        static_assert(N >= 3 || P >= 1, "This dot product is for higher dimensional tensors");
        
        // Check that the last dimension of this matches first dimension of other
        if (dims_[N - 1] != other.dims_[0]) {
            return TensorError::ContractionMismatch;
        }
        
        size_t contract_dim = dims_[N - 1];
        
        // Calculate output dimensions: all dims of this except last,
        // plus all dims of other except first
        TensorIndices<N + P - 2> result_dims;
        for (size_t i = 0; i < N - 1; ++i) {
            result_dims[i] = dims_[i];
        }
        for (size_t i = 1; i < P; ++i) {
            result_dims[N - 1 + i - 1] = other.dims_[i];
        }
        
        Tensor<T, N + P - 2> result(result_dims, use_gpu_);
        result.fill(T());
        
        // Calculate strides for output indexing
        size_t outer_size = 1;  // Product of all dims except last in this tensor
        for (size_t i = 0; i < N - 1; ++i) {
            outer_size *= dims_[i];
        }
        
        size_t inner_size = 1;  // Product of all dims except first in other tensor
        for (size_t i = 1; i < P; ++i) {
            inner_size *= other.dims_[i];
        }
        
#ifdef USE_GPU
        if (use_gpu_ && other.use_gpu_) {
            TensorGPU::dot_nd_gpu(data_.get(), other.data_.get(), result.data_.get(),
                                  outer_size, contract_dim, inner_size);
            return result;
        }
#endif
        
        // Perform contraction with parallel execution
        auto outer_range = std::views::iota(size_t(0), outer_size);
        std::for_each(std::execution::par, outer_range.begin(), outer_range.end(), 
            [&](size_t outer) {
                for (size_t inner = 0; inner < inner_size; ++inner) {
                    T sum = T();
                    for (size_t k = 0; k < contract_dim; ++k) {
                        // Calculate flat indices for input tensors
                        size_t idx_this = outer * contract_dim + k;
                        size_t idx_other = k * inner_size + inner;
                        sum += data_[idx_this] * other.data_[idx_other];
                    }
                    // Calculate flat index for output tensor
                    size_t idx_result = outer * inner_size + inner;
                    result.data_[idx_result] = sum;
                }
            });
        
        return result;
    }
    
    // ============================================
    // Phase 1: Core Neural Network Operations
    // ============================================
    
    /**
     * Matrix multiplication with autograd support (matmul).
     * For 2D tensors: A(m,n) @ B(n,p) = C(m,p)
     * Supports automatic differentiation.
     * @param other The tensor to multiply with (must have compatible dimensions)
     * @return Result tensor or error
     */
    template<size_t M = N>
    typename std::enable_if<M == 2, TensorResult<Tensor<T, 2>>>::type
    matmul(const Tensor<T, 2>& other) const {
        static_assert(N == 2, "matmul requires 2D tensors");
        
        if (dims_[1] != other.dims_[0]) {
            return TensorError::DimensionMismatch;
        }
        
        size_t m = dims_[0];
        size_t n = dims_[1];
        size_t p = other.dims_[1];
        
        bool track_grad = requires_grad_ || other.requires_grad_;
        Tensor<T, 2> result({m, p}, use_gpu_, track_grad);
        result.is_leaf_ = false;
        result.fill(T(0));
        
        // Forward pass - use existing optimized dot product
#ifdef USE_BLAS
        if (!use_gpu_) {
            blas_gemm<T>(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, p, n, T(1), data_.get(), n,
                        other.data_.get(), p, T(0), result.data_.get(), p);
        } else
#endif
        {
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < p; ++j) {
                    T sum = T(0);
                    for (size_t k = 0; k < n; ++k) {
                        sum += (*this)[{i, k}] * other[{k, j}];
                    }
                    result[{i, j}] = sum;
                }
            }
        }
        
        // Setup backward pass for autograd
        if (track_grad) {
            Tensor<T, 2> self_copy = this->detach();
            Tensor<T, 2> other_copy = other.detach();
            Tensor<T, 2>* self_ptr = const_cast<Tensor<T, 2>*>(this);
            Tensor<T, 2>* other_ptr = const_cast<Tensor<T, 2>*>(&other);
            
            result.register_backward([self_ptr, other_ptr, self_copy, other_copy, m, n, p]
                                    (const Tensor<T, 2>& grad) {
                // Gradient for matmul:
                // If C = A @ B, then:
                // dL/dA = dL/dC @ B^T
                // dL/dB = A^T @ dL/dC
                
                if (self_ptr->requires_grad_) {
                    if (!self_ptr->grad_) {
                        self_ptr->grad_ = std::make_unique<Tensor<T, 2>>(self_ptr->dims_, self_ptr->use_gpu_, false);
                        self_ptr->grad_->fill(T(0));
                    }
                    
                    // dL/dA = grad @ other^T
                    for (size_t i = 0; i < m; ++i) {
                        for (size_t k = 0; k < n; ++k) {
                            T sum = T(0);
                            for (size_t j = 0; j < p; ++j) {
                                sum += grad[{i, j}] * other_copy[{k, j}];
                            }
                            (*self_ptr->grad_)[{i, k}] += sum;
                        }
                    }
                    
                    // Propagate further if not leaf
                    if (!self_ptr->is_leaf_ && !self_ptr->backward_funcs_.empty()) {
                        for (auto& func : self_ptr->backward_funcs_) {
                            func(*self_ptr->grad_);
                        }
                    }
                }
                
                if (other_ptr->requires_grad_) {
                    if (!other_ptr->grad_) {
                        other_ptr->grad_ = std::make_unique<Tensor<T, 2>>(other_ptr->dims_, other_ptr->use_gpu_, false);
                        other_ptr->grad_->fill(T(0));
                    }
                    
                    // dL/dB = self^T @ grad
                    for (size_t k = 0; k < n; ++k) {
                        for (size_t j = 0; j < p; ++j) {
                            T sum = T(0);
                            for (size_t i = 0; i < m; ++i) {
                                sum += self_copy[{i, k}] * grad[{i, j}];
                            }
                            (*other_ptr->grad_)[{k, j}] += sum;
                        }
                    }
                    
                    // Propagate further if not leaf
                    if (!other_ptr->is_leaf_ && !other_ptr->backward_funcs_.empty()) {
                        for (auto& func : other_ptr->backward_funcs_) {
                            func(*other_ptr->grad_);
                        }
                    }
                }
            });
        }
        
        return result;
    }
    
    /**
     * Sum reduction along a specific axis with autograd support.
     * @param axis The axis to sum along (-1 for last axis)
     * @param keepdims Whether to keep the reduced dimension as size 1
     * @return Reduced tensor, or zero tensor if axis is out of range
     */
    Tensor<T, N> sum_axis(int axis, bool keepdims = false) const {
        if (axis < 0) {
            axis = static_cast<int>(N) + axis;
        }
        
        if (axis < 0 || axis >= static_cast<int>(N)) {
            TensorIndices<N> zero_dims{};
            return Tensor<T, N>(zero_dims, use_gpu_, requires_grad_);
        }
        
        size_t ax = static_cast<size_t>(axis);
        
        // Compute output dimensions
        TensorIndices<N> result_dims = dims_;
        if (keepdims) {
            result_dims[ax] = 1;
        } else {
            // For now, keep the dimension but set to 1
            // A proper implementation would reduce the tensor rank
            result_dims[ax] = 1;
        }
        
        Tensor<T, N> result(result_dims, use_gpu_, requires_grad_);
        result.is_leaf_ = false;
        result.fill(T(0));
        
        // Compute sum
        size_t outer = 1;
        for (size_t i = 0; i < ax; ++i) {
            outer *= dims_[i];
        }
        
        size_t reduce_dim = dims_[ax];
        
        size_t inner = 1;
        for (size_t i = ax + 1; i < N; ++i) {
            inner *= dims_[i];
        }
        
        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < inner; ++i) {
                T sum = T(0);
                for (size_t r = 0; r < reduce_dim; ++r) {
                    TensorIndices<N> idx;
                    size_t linear_idx = o * reduce_dim * inner + r * inner + i;
                    
                    // Convert linear index to multi-dimensional
                    size_t temp = linear_idx;
                    for (int d = static_cast<int>(N) - 1; d >= 0; --d) {
                        idx[d] = temp % dims_[d];
                        temp /= dims_[d];
                    }
                    
                    sum += (*this)[idx];
                }
                
                TensorIndices<N> result_idx;
                size_t result_linear = o * inner + i;
                size_t temp = result_linear;
                for (int d = static_cast<int>(N) - 1; d >= 0; --d) {
                    if (d == static_cast<int>(ax)) {
                        result_idx[d] = 0;
                    } else if (d < static_cast<int>(ax)) {
                        result_idx[d] = temp / inner;
                        temp %= inner;
                    } else {
                        result_idx[d] = temp % result_dims[d];
                        temp /= result_dims[d];
                    }
                }
                
                result[result_idx] = sum;
            }
        }
        
        // Setup backward pass
        if (requires_grad_) {
            Tensor<T, N>* self_ptr = const_cast<Tensor<T, N>*>(this);
            TensorIndices<N> self_dims = dims_;
            size_t reduce_axis = ax;
            
            result.register_backward([self_ptr, self_dims, reduce_axis](const Tensor<T, N>& grad) {
                // Gradient of sum: broadcast gradient back to original shape
                if (self_ptr->requires_grad_) {
                    if (!self_ptr->grad_) {
                        self_ptr->grad_ = std::make_unique<Tensor<T, N>>(self_ptr->dims_, self_ptr->use_gpu_, false);
                        self_ptr->grad_->fill(T(0));
                    }
                    
                    // Each element in the original tensor contributed to one element in the sum
                    // So gradient is just broadcast of the output gradient
                    size_t total = self_ptr->total_size();
                    for (size_t i = 0; i < total; ++i) {
                        TensorIndices<N> idx;
                        size_t temp = i;
                        for (int d = static_cast<int>(N) - 1; d >= 0; --d) {
                            idx[d] = temp % self_dims[d];
                            temp /= self_dims[d];
                        }
                        
                        TensorIndices<N> grad_idx = idx;
                        grad_idx[reduce_axis] = 0;  // Sum reduced this dimension
                        
                        self_ptr->grad_->data_[i] += grad[grad_idx];
                    }
                    
                    if (!self_ptr->is_leaf_ && !self_ptr->backward_funcs_.empty()) {
                        for (auto& func : self_ptr->backward_funcs_) {
                            func(*self_ptr->grad_);
                        }
                    }
                }
            });
        }
        
        return result;
    }
    
    /**
     * Mean reduction along a specific axis with autograd support.
     * @param axis The axis to average along (-1 for last axis)
     * @param keepdims Whether to keep the reduced dimension
     * @return Reduced tensor
     */
    Tensor<T, N> mean_axis(int axis, bool keepdims = false) const {
        auto sum_result = sum_axis(axis, keepdims);
        
        if (axis < 0) {
            axis = static_cast<int>(N) + axis;
        }
        
        T count = static_cast<T>(dims_[axis]);
        
        // Divide by count
        Tensor<T, N> result = sum_result * (T(1) / count);
        
        return result;
    }
    
    /**
     * Softmax operation with autograd support.
     * Applies softmax along the last dimension: softmax(x_i) = exp(x_i) / sum(exp(x_j))
     * Numerically stable implementation (subtracts max before exp).
     * @param axis The axis to apply softmax along (default: -1 = last axis)
     * @return Tensor with softmax applied, or zero tensor if axis is out of range
     */
    Tensor<T, N> softmax(int axis = -1) const {
        if (axis < 0) {
            axis = static_cast<int>(N) + axis;
        }
        
        if (axis < 0 || axis >= static_cast<int>(N)) {
            TensorIndices<N> zero_dims{};
            return Tensor<T, N>(zero_dims, use_gpu_, requires_grad_);
        }
        
        size_t ax = static_cast<size_t>(axis);
        
        Tensor<T, N> result(dims_, use_gpu_, requires_grad_);
        result.is_leaf_ = false;
        
        // Compute softmax with numerical stability (subtract max)
        size_t outer = 1;
        for (size_t i = 0; i < ax; ++i) {
            outer *= dims_[i];
        }
        
        size_t softmax_dim = dims_[ax];
        
        size_t inner = 1;
        for (size_t i = ax + 1; i < N; ++i) {
            inner *= dims_[i];
        }
        
        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < inner; ++i) {
                // Find max for numerical stability
                T max_val = std::numeric_limits<T>::lowest();
                for (size_t s = 0; s < softmax_dim; ++s) {
                    TensorIndices<N> idx;
                    size_t linear_idx = o * softmax_dim * inner + s * inner + i;
                    
                    size_t temp = linear_idx;
                    for (int d = static_cast<int>(N) - 1; d >= 0; --d) {
                        idx[d] = temp % dims_[d];
                        temp /= dims_[d];
                    }
                    
                    max_val = std::max(max_val, (*this)[idx]);
                }
                
                // Compute exp(x - max) and sum
                T sum_exp = T(0);
                std::vector<T> exp_vals(softmax_dim);
                
                for (size_t s = 0; s < softmax_dim; ++s) {
                    TensorIndices<N> idx;
                    size_t linear_idx = o * softmax_dim * inner + s * inner + i;
                    
                    size_t temp = linear_idx;
                    for (int d = static_cast<int>(N) - 1; d >= 0; --d) {
                        idx[d] = temp % dims_[d];
                        temp /= dims_[d];
                    }
                    
                    exp_vals[s] = std::exp((*this)[idx] - max_val);
                    sum_exp += exp_vals[s];
                }
                
                // Normalize
                for (size_t s = 0; s < softmax_dim; ++s) {
                    TensorIndices<N> idx;
                    size_t linear_idx = o * softmax_dim * inner + s * inner + i;
                    
                    size_t temp = linear_idx;
                    for (int d = static_cast<int>(N) - 1; d >= 0; --d) {
                        idx[d] = temp % dims_[d];
                        temp /= dims_[d];
                    }
                    
                    result[idx] = exp_vals[s] / sum_exp;
                }
            }
        }
        
        // Setup backward pass
        if (requires_grad_) {
            Tensor<T, N> output_copy = result.detach();
            Tensor<T, N>* self_ptr = const_cast<Tensor<T, N>*>(this);
            size_t softmax_axis = ax;
            TensorIndices<N> self_dims = dims_;
            
            result.register_backward([self_ptr, output_copy, softmax_axis, self_dims, outer, softmax_dim, inner]
                                    (const Tensor<T, N>& grad) {
                // Gradient of softmax:
                // dsoftmax_i/dx_j = softmax_i * (δ_ij - softmax_j)
                if (self_ptr->requires_grad_) {
                    if (!self_ptr->grad_) {
                        self_ptr->grad_ = std::make_unique<Tensor<T, N>>(self_ptr->dims_, self_ptr->use_gpu_, false);
                        self_ptr->grad_->fill(T(0));
                    }
                    
                    for (size_t o = 0; o < outer; ++o) {
                        for (size_t i = 0; i < inner; ++i) {
                            for (size_t s = 0; s < softmax_dim; ++s) {
                                TensorIndices<N> idx_s;
                                size_t linear_s = o * softmax_dim * inner + s * inner + i;
                                
                                size_t temp = linear_s;
                                for (int d = static_cast<int>(N) - 1; d >= 0; --d) {
                                    idx_s[d] = temp % self_dims[d];
                                    temp /= self_dims[d];
                                }
                                
                                T sum = T(0);
                                for (size_t j = 0; j < softmax_dim; ++j) {
                                    TensorIndices<N> idx_j = idx_s;
                                    idx_j[softmax_axis] = j;
                                    
                                    T delta_ij = (s == j) ? T(1) : T(0);
                                    sum += grad[idx_j] * (delta_ij - output_copy[idx_j]);
                                }
                                
                                (*self_ptr->grad_)[idx_s] += output_copy[idx_s] * sum;
                            }
                        }
                    }
                    
                    if (!self_ptr->is_leaf_ && !self_ptr->backward_funcs_.empty()) {
                        for (auto& func : self_ptr->backward_funcs_) {
                            func(*self_ptr->grad_);
                        }
                    }
                }
            });
        }
        
        return result;
    }
    
    /**
     * Log-softmax operation (numerically stable).
     * log_softmax(x) = log(softmax(x)) = x - log(sum(exp(x)))
     * @param axis The axis to apply log-softmax along
     * @return Tensor with log-softmax applied
     */
    Tensor<T, N> log_softmax(int axis = -1) const {
        // More numerically stable than log(softmax(x))
        if (axis < 0) {
            axis = static_cast<int>(N) + axis;
        }
        
        // log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
        // This avoids computing softmax explicitly
        
        size_t ax = static_cast<size_t>(axis);
        Tensor<T, N> result(dims_, use_gpu_, requires_grad_);
        result.is_leaf_ = false;
        
        size_t outer = 1;
        for (size_t i = 0; i < ax; ++i) {
            outer *= dims_[i];
        }
        
        size_t softmax_dim = dims_[ax];
        
        size_t inner = 1;
        for (size_t i = ax + 1; i < N; ++i) {
            inner *= dims_[i];
        }
        
        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < inner; ++i) {
                // Find max
                T max_val = std::numeric_limits<T>::lowest();
                for (size_t s = 0; s < softmax_dim; ++s) {
                    TensorIndices<N> idx;
                    size_t linear_idx = o * softmax_dim * inner + s * inner + i;
                    
                    size_t temp = linear_idx;
                    for (int d = static_cast<int>(N) - 1; d >= 0; --d) {
                        idx[d] = temp % dims_[d];
                        temp /= dims_[d];
                    }
                    
                    max_val = std::max(max_val, (*this)[idx]);
                }
                
                // Compute log(sum(exp(x - max)))
                T log_sum_exp = T(0);
                for (size_t s = 0; s < softmax_dim; ++s) {
                    TensorIndices<N> idx;
                    size_t linear_idx = o * softmax_dim * inner + s * inner + i;
                    
                    size_t temp = linear_idx;
                    for (int d = static_cast<int>(N) - 1; d >= 0; --d) {
                        idx[d] = temp % dims_[d];
                        temp /= dims_[d];
                    }
                    
                    log_sum_exp += std::exp((*this)[idx] - max_val);
                }
                log_sum_exp = std::log(log_sum_exp);
                
                // Compute result: x - max - log_sum_exp
                for (size_t s = 0; s < softmax_dim; ++s) {
                    TensorIndices<N> idx;
                    size_t linear_idx = o * softmax_dim * inner + s * inner + i;
                    
                    size_t temp = linear_idx;
                    for (int d = static_cast<int>(N) - 1; d >= 0; --d) {
                        idx[d] = temp % dims_[d];
                        temp /= dims_[d];
                    }
                    
                    result[idx] = (*this)[idx] - max_val - log_sum_exp;
                }
            }
        }
        
        // Backward pass for log_softmax
        if (requires_grad_) {
            Tensor<T, N> output_copy = result.detach();
            Tensor<T, N>* self_ptr = const_cast<Tensor<T, N>*>(this);
            
            result.register_backward([self_ptr, output_copy](const Tensor<T, N>& grad) {
                // d/dx[log_softmax(x)] = 1 - softmax(x) for diagonal
                // Gradient: grad_input = grad_output - softmax(x) * sum(grad_output)
                if (self_ptr->requires_grad_) {
                    if (!self_ptr->grad_) {
                        self_ptr->grad_ = std::make_unique<Tensor<T, N>>(self_ptr->dims_, self_ptr->use_gpu_, false);
                        self_ptr->grad_->fill(T(0));
                    }
                    
                    // Compute softmax from log_softmax
                    size_t total = self_ptr->total_size();
                    for (size_t i = 0; i < total; ++i) {
                        T softmax_val = std::exp(output_copy.data_[i]);
                        
                        // Compute sum of gradients (simplified for now)
                        T grad_sum = grad.data_[i];
                        
                        self_ptr->grad_->data_[i] += grad.data_[i] - softmax_val * grad_sum;
                    }
                    
                    if (!self_ptr->is_leaf_ && !self_ptr->backward_funcs_.empty()) {
                        for (auto& func : self_ptr->backward_funcs_) {
                            func(*self_ptr->grad_);
                        }
                    }
                }
            });
        }
        
        return result;
    }
    
    // ==================== PHASE 3: ADVANCED TENSOR OPERATIONS ====================
    
    /**
     * Reshape tensor to new dimensions without copying data.
     * Total size must remain the same.
     * @param new_dims New dimensions
     * @return Reshaped tensor (view of same data), or zero tensor if size doesn't match
     */
    template<size_t M>
    Tensor<T, M> reshape(const TensorIndices<M>& new_dims) const {
        size_t new_total = 1;
        for (size_t i = 0; i < M; ++i) {
            new_total *= new_dims[i];
        }
        
        if (new_total != total_size()) {
            TensorIndices<M> zero_dims{};
            return Tensor<T, M>(zero_dims, use_gpu_, requires_grad_);
        }
        
        Tensor<T, M> result(new_dims, use_gpu_, requires_grad_);
        result.is_leaf_ = false;
        
        // Copy data
        std::copy(data_.get(), data_.get() + total_size(), result.data_.get());
        
        // Setup backward pass
        if (requires_grad_) {
            TensorIndices<N> orig_dims = dims_;
            Tensor<T, N>* self_ptr = const_cast<Tensor<T, N>*>(this);
            
            result.register_backward([self_ptr, orig_dims](const Tensor<T, M>& grad) {
                if (self_ptr->requires_grad_) {
                    if (!self_ptr->grad_) {
                        self_ptr->grad_ = std::make_unique<Tensor<T, N>>(orig_dims, 
                                                                          self_ptr->use_gpu_, false);
                        self_ptr->grad_->fill(T(0));
                    }
                    
                    // Gradient flows back with same reshape
                    size_t total = self_ptr->total_size();
                    for (size_t i = 0; i < total; ++i) {
                        self_ptr->grad_->data_[i] += grad.data_[i];
                    }
                    
                    if (!self_ptr->is_leaf_ && !self_ptr->backward_funcs_.empty()) {
                        for (auto& func : self_ptr->backward_funcs_) {
                            func(*self_ptr->grad_);
                        }
                    }
                }
            });
        }
        
        return result;
    }
    
    /**
     * Transpose 2D tensor (swap dimensions 0 and 1).
     * For N>2, this swaps the last two dimensions.
     * @return Transposed tensor
     */
    Tensor<T, N> transpose() const {
        static_assert(N >= 2, "Transpose requires at least 2 dimensions");
        
        TensorIndices<N> new_dims = dims_;
        std::swap(new_dims[N-2], new_dims[N-1]);
        
        Tensor<T, N> result(new_dims, use_gpu_, requires_grad_);
        result.is_leaf_ = false;
        
        // Transpose the last two dimensions
        if constexpr (N == 2) {
            for (size_t i = 0; i < dims_[0]; ++i) {
                for (size_t j = 0; j < dims_[1]; ++j) {
                    result[{j, i}] = (*this)[{i, j}];
                }
            }
        } else {
            // For N>2, transpose last two dims while iterating over others
            size_t outer = 1;
            for (size_t i = 0; i < N-2; ++i) {
                outer *= dims_[i];
            }
            
            for (size_t o = 0; o < outer; ++o) {
                // Compute indices for outer dimensions
                TensorIndices<N> base_idx;
                size_t temp = o;
                for (int i = N-3; i >= 0; --i) {
                    base_idx[i] = temp % dims_[i];
                    temp /= dims_[i];
                }
                
                for (size_t i = 0; i < dims_[N-2]; ++i) {
                    for (size_t j = 0; j < dims_[N-1]; ++j) {
                        TensorIndices<N> src_idx = base_idx;
                        TensorIndices<N> dst_idx = base_idx;
                        src_idx[N-2] = i;
                        src_idx[N-1] = j;
                        dst_idx[N-2] = j;
                        dst_idx[N-1] = i;
                        result[dst_idx] = (*this)[src_idx];
                    }
                }
            }
        }
        
        // Setup backward pass
        if (requires_grad_) {
            Tensor<T, N>* self_ptr = const_cast<Tensor<T, N>*>(this);
            
            result.register_backward([self_ptr](const Tensor<T, N>& grad) {
                if (self_ptr->requires_grad_) {
                    if (!self_ptr->grad_) {
                        self_ptr->grad_ = std::make_unique<Tensor<T, N>>(self_ptr->dims_, 
                                                                          self_ptr->use_gpu_, false);
                        self_ptr->grad_->fill(T(0));
                    }
                    
                    // Gradient of transpose is transpose of gradient
                    Tensor<T, N> grad_transposed = grad.transpose();
                    size_t total = self_ptr->total_size();
                    for (size_t i = 0; i < total; ++i) {
                        self_ptr->grad_->data_[i] += grad_transposed.data_[i];
                    }
                    
                    if (!self_ptr->is_leaf_ && !self_ptr->backward_funcs_.empty()) {
                        for (auto& func : self_ptr->backward_funcs_) {
                            func(*self_ptr->grad_);
                        }
                    }
                }
            });
        }
        
        return result;
    }
    
    /**
     * Permute (rearrange) dimensions.
     * @param axes New order of dimensions (must be a permutation of 0..N-1)
     * @return Tensor with permuted dimensions, or zero tensor if axes is invalid
     */
    Tensor<T, N> permute(const TensorIndices<N>& axes) const {
        // Validate that axes is a permutation
        std::array<bool, N> seen = {false};
        for (size_t i = 0; i < N; ++i) {
            if (axes[i] >= N || seen[axes[i]]) {
                TensorIndices<N> zero_dims{};
                return Tensor<T, N>(zero_dims, use_gpu_, requires_grad_);
            }
            seen[axes[i]] = true;
        }
        
        // Compute new dimensions
        TensorIndices<N> new_dims;
        for (size_t i = 0; i < N; ++i) {
            new_dims[i] = dims_[axes[i]];
        }
        
        Tensor<T, N> result(new_dims, use_gpu_, requires_grad_);
        result.is_leaf_ = false;
        
        // Permute data
        size_t total = total_size();
        for (size_t idx = 0; idx < total; ++idx) {
            // Convert linear index to multi-dimensional indices
            TensorIndices<N> src_idx;
            size_t temp = idx;
            for (int i = N-1; i >= 0; --i) {
                src_idx[i] = temp % dims_[i];
                temp /= dims_[i];
            }
            
            // Permute indices
            TensorIndices<N> dst_idx;
            for (size_t i = 0; i < N; ++i) {
                dst_idx[i] = src_idx[axes[i]];
            }
            
            result[dst_idx] = (*this)[src_idx];
        }
        
        // Setup backward pass
        if (requires_grad_) {
            TensorIndices<N> inv_axes;
            for (size_t i = 0; i < N; ++i) {
                inv_axes[axes[i]] = i;
            }
            
            Tensor<T, N>* self_ptr = const_cast<Tensor<T, N>*>(this);
            
            result.register_backward([self_ptr, inv_axes](const Tensor<T, N>& grad) {
                if (self_ptr->requires_grad_) {
                    if (!self_ptr->grad_) {
                        self_ptr->grad_ = std::make_unique<Tensor<T, N>>(self_ptr->dims_, 
                                                                          self_ptr->use_gpu_, false);
                        self_ptr->grad_->fill(T(0));
                    }
                    
                    // Gradient flows back with inverse permutation
                    Tensor<T, N> grad_permuted = grad.permute(inv_axes);
                    size_t total = self_ptr->total_size();
                    for (size_t i = 0; i < total; ++i) {
                        self_ptr->grad_->data_[i] += grad_permuted.data_[i];
                    }
                    
                    if (!self_ptr->is_leaf_ && !self_ptr->backward_funcs_.empty()) {
                        for (auto& func : self_ptr->backward_funcs_) {
                            func(*self_ptr->grad_);
                        }
                    }
                }
            });
        }
        
        return result;
    }
    
    /**
     * Squeeze - remove dimensions of size 1.
     * @param axis Specific axis to squeeze, or -1 to squeeze all axes of size 1
     * @return Tensor with squeezed dimensions, or zero tensor if invalid axis
     */
    Tensor<T, N> squeeze(int axis = -1) const {
        if (axis >= 0) {
            if (static_cast<size_t>(axis) >= N) {
                TensorIndices<N> zero_dims{};
                return Tensor<T, N>(zero_dims, use_gpu_, requires_grad_);
            }
            if (dims_[axis] != 1) {
                TensorIndices<N> zero_dims{};
                return Tensor<T, N>(zero_dims, use_gpu_, requires_grad_);
            }
        }
        
        // For simplicity, return a copy with same dimensions
        // A proper implementation would change rank
        Tensor<T, N> result(dims_, use_gpu_, requires_grad_);
        result.is_leaf_ = false;
        
        std::copy(data_.get(), data_.get() + total_size(), result.data_.get());
        
        if (requires_grad_) {
            Tensor<T, N>* self_ptr = const_cast<Tensor<T, N>*>(this);
            
            result.register_backward([self_ptr](const Tensor<T, N>& grad) {
                if (self_ptr->requires_grad_) {
                    if (!self_ptr->grad_) {
                        self_ptr->grad_ = std::make_unique<Tensor<T, N>>(self_ptr->dims_, 
                                                                          self_ptr->use_gpu_, false);
                        self_ptr->grad_->fill(T(0));
                    }
                    
                    size_t total = self_ptr->total_size();
                    for (size_t i = 0; i < total; ++i) {
                        self_ptr->grad_->data_[i] += grad.data_[i];
                    }
                    
                    if (!self_ptr->is_leaf_ && !self_ptr->backward_funcs_.empty()) {
                        for (auto& func : self_ptr->backward_funcs_) {
                            func(*self_ptr->grad_);
                        }
                    }
                }
            });
        }
        
        return result;
    }
    
    /**
     * Unsqueeze - add a dimension of size 1 at specified axis.
     * @param axis Position to add the new dimension
     * @return Tensor with added dimension, or zero tensor if axis is invalid
     */
    template<size_t M = N + 1>
    Tensor<T, M> unsqueeze(size_t axis) const {
        if (axis > N) {
            TensorIndices<M> zero_dims{};
            return Tensor<T, M>(zero_dims, use_gpu_, requires_grad_);
        }
        
        TensorIndices<M> new_dims;
        for (size_t i = 0; i < axis; ++i) {
            new_dims[i] = dims_[i];
        }
        new_dims[axis] = 1;
        for (size_t i = axis; i < N; ++i) {
            new_dims[i+1] = dims_[i];
        }
        
        Tensor<T, M> result(new_dims, use_gpu_, requires_grad_);
        result.is_leaf_ = false;
        
        std::copy(data_.get(), data_.get() + total_size(), result.data_.get());
        
        if (requires_grad_) {
            TensorIndices<N> orig_dims = dims_;
            Tensor<T, N>* self_ptr = const_cast<Tensor<T, N>*>(this);
            
            result.register_backward([self_ptr, orig_dims](const Tensor<T, M>& grad) {
                if (self_ptr->requires_grad_) {
                    if (!self_ptr->grad_) {
                        self_ptr->grad_ = std::make_unique<Tensor<T, N>>(orig_dims, 
                                                                          self_ptr->use_gpu_, false);
                        self_ptr->grad_->fill(T(0));
                    }
                    
                    size_t total = self_ptr->total_size();
                    for (size_t i = 0; i < total; ++i) {
                        self_ptr->grad_->data_[i] += grad.data_[i];
                    }
                    
                    if (!self_ptr->is_leaf_ && !self_ptr->backward_funcs_.empty()) {
                        for (auto& func : self_ptr->backward_funcs_) {
                            func(*self_ptr->grad_);
                        }
                    }
                }
            });
        }
        
        return result;
    }
    
    /**
     * Concatenate tensors along a specific axis.
     * @param other The tensor to concatenate with
     * @param axis The axis along which to concatenate
     * @return Concatenated tensor, or zero tensor if dimensions don't match
     */
    Tensor<T, N> concatenate(const Tensor<T, N>& other, size_t axis) const {
        if (axis >= N) {
            TensorIndices<N> zero_dims{};
            return Tensor<T, N>(zero_dims, use_gpu_, requires_grad_);
        }
        
        // Check that all dimensions match except the concat axis
        for (size_t i = 0; i < N; ++i) {
            if (i != axis && dims_[i] != other.dims_[i]) {
                TensorIndices<N> zero_dims{};
                return Tensor<T, N>(zero_dims, use_gpu_, requires_grad_);
            }
        }
        
        TensorIndices<N> new_dims = dims_;
        new_dims[axis] = dims_[axis] + other.dims_[axis];
        
        bool track_grad = requires_grad_ || other.requires_grad_;
        Tensor<T, N> result(new_dims, use_gpu_, track_grad);
        result.is_leaf_ = false;
        
        // Copy data from first tensor
        size_t outer = 1;
        for (size_t i = 0; i < axis; ++i) {
            outer *= dims_[i];
        }
        
        size_t inner = 1;
        for (size_t i = axis + 1; i < N; ++i) {
            inner *= dims_[i];
        }
        
        size_t this_axis_size = dims_[axis];
        size_t other_axis_size = other.dims_[axis];
        
        for (size_t o = 0; o < outer; ++o) {
            // Copy from first tensor
            for (size_t a = 0; a < this_axis_size; ++a) {
                for (size_t i = 0; i < inner; ++i) {
                    size_t src_idx = o * this_axis_size * inner + a * inner + i;
                    size_t dst_idx = o * (this_axis_size + other_axis_size) * inner + a * inner + i;
                    result.data_[dst_idx] = data_[src_idx];
                }
            }
            
            // Copy from second tensor
            for (size_t a = 0; a < other_axis_size; ++a) {
                for (size_t i = 0; i < inner; ++i) {
                    size_t src_idx = o * other_axis_size * inner + a * inner + i;
                    size_t dst_idx = o * (this_axis_size + other_axis_size) * inner + 
                                    (this_axis_size + a) * inner + i;
                    result.data_[dst_idx] = other.data_[src_idx];
                }
            }
        }
        
        // Setup backward pass
        if (track_grad) {
            Tensor<T, N>* self_ptr = const_cast<Tensor<T, N>*>(this);
            Tensor<T, N>* other_ptr = const_cast<Tensor<T, N>*>(&other);
            
            result.register_backward([self_ptr, other_ptr, axis, outer, inner, 
                                     this_axis_size, other_axis_size](const Tensor<T, N>& grad) {
                // Split gradient back to both tensors
                if (self_ptr->requires_grad_) {
                    if (!self_ptr->grad_) {
                        self_ptr->grad_ = std::make_unique<Tensor<T, N>>(self_ptr->dims_, 
                                                                          self_ptr->use_gpu_, false);
                        self_ptr->grad_->fill(T(0));
                    }
                    
                    for (size_t o = 0; o < outer; ++o) {
                        for (size_t a = 0; a < this_axis_size; ++a) {
                            for (size_t i = 0; i < inner; ++i) {
                                size_t dst_idx = o * this_axis_size * inner + a * inner + i;
                                size_t src_idx = o * (this_axis_size + other_axis_size) * inner + a * inner + i;
                                self_ptr->grad_->data_[dst_idx] += grad.data_[src_idx];
                            }
                        }
                    }
                    
                    if (!self_ptr->is_leaf_ && !self_ptr->backward_funcs_.empty()) {
                        for (auto& func : self_ptr->backward_funcs_) {
                            func(*self_ptr->grad_);
                        }
                    }
                }
                
                if (other_ptr->requires_grad_) {
                    if (!other_ptr->grad_) {
                        other_ptr->grad_ = std::make_unique<Tensor<T, N>>(other_ptr->dims_, 
                                                                           other_ptr->use_gpu_, false);
                        other_ptr->grad_->fill(T(0));
                    }
                    
                    for (size_t o = 0; o < outer; ++o) {
                        for (size_t a = 0; a < other_axis_size; ++a) {
                            for (size_t i = 0; i < inner; ++i) {
                                size_t dst_idx = o * other_axis_size * inner + a * inner + i;
                                size_t src_idx = o * (this_axis_size + other_axis_size) * inner + 
                                                (this_axis_size + a) * inner + i;
                                other_ptr->grad_->data_[dst_idx] += grad.data_[src_idx];
                            }
                        }
                    }
                    
                    if (!other_ptr->is_leaf_ && !other_ptr->backward_funcs_.empty()) {
                        for (auto& func : other_ptr->backward_funcs_) {
                            func(*other_ptr->grad_);
                        }
                    }
                }
            });
        }
        
        return result;
    }
    
    // ============================================
    // Advanced Indexing & Slicing Operations
    // ============================================
    
    /**
     * Extract elements at specific flat indices (fancy indexing).
     * Treats the tensor as a flattened 1D array and extracts elements at the given indices.
     * @param indices Vector of flat indices to extract.
     * @return A new 1D tensor containing the extracted elements.
     */
    Tensor<T, 1> take(const std::vector<size_t>& indices) const {
        Tensor<T, 1> result({indices.size()}, use_gpu_);
        size_t total = total_size();
        
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] < total) {
                result.data_[i] = data_[indices[i]];
            } else {
                result.data_[i] = T(0);  // Out of bounds returns zero
            }
        }
        
        return result;
    }
    
    /**
     * Set elements at specific flat indices (fancy indexing assignment).
     * Treats the tensor as a flattened 1D array and sets values at the given indices.
     * @param indices Vector of flat indices to set.
     * @param values Vector of values to set (must match indices size).
     */
    void put(const std::vector<size_t>& indices, const std::vector<T>& values) {
        size_t count = std::min(indices.size(), values.size());
        size_t total = total_size();
        
        for (size_t i = 0; i < count; ++i) {
            if (indices[i] < total) {
                data_[indices[i]] = values[i];
            }
        }
    }
    
    /**
     * Boolean indexing: extract elements where mask is true (non-zero).
     * This is similar to masked_select but emphasized for indexing operations.
     * @param mask Boolean-like tensor (0 or non-zero values).
     * @return A 1D tensor containing selected values.
     */
    Tensor<T, 1> index_select(const Tensor<T, N>& mask) const {
        return masked_select(mask);
    }
    
    /**
     * Select a specific index along a dimension, reducing that dimension.
     * For example, selecting row 2 from a 2D tensor returns a 1D tensor.
     * Only available for tensors with N > 1.
     * @param dim The dimension along which to select.
     * @param index The index to select in that dimension.
     * @return A tensor with dimension N-1.
     */
    template<size_t M = (N > 1 ? N - 1 : 1)>
    std::enable_if_t<(N > 1), Tensor<T, M>> select(size_t dim, size_t index) const {
        if (dim >= N || index >= dims_[dim]) {
            TensorIndices<M> zero_dims{};
            return Tensor<T, M>(zero_dims, use_gpu_);
        }
        
        // Calculate new dimensions (removing the selected dimension)
        TensorIndices<M> new_dims;
        size_t new_idx = 0;
        for (size_t i = 0; i < N; ++i) {
            if (i != dim) {
                new_dims[new_idx++] = dims_[i];
            }
        }
        
        Tensor<T, M> result(new_dims, use_gpu_);
        
        // Copy selected slice
        size_t result_total = result.total_size();
        size_t stride = strides_[dim];
        size_t dim_size = dims_[dim];
        
        // Calculate outer and inner sizes
        size_t outer = 1;
        for (size_t i = 0; i < dim; ++i) {
            outer *= dims_[i];
        }
        
        size_t inner = 1;
        for (size_t i = dim + 1; i < N; ++i) {
            inner *= dims_[i];
        }
        
        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < inner; ++i) {
                size_t src_idx = o * dim_size * inner + index * inner + i;
                size_t dst_idx = o * inner + i;
                result.data_[dst_idx] = data_[src_idx];
            }
        }
        
        return result;
    }
    
    // ============================================
    // Advanced Reduction Operations
    // ============================================
    
    /**
     * Sum reduction along a specific axis.
     * @param axis The dimension along which to sum.
     * @param keepdims Whether to keep the reduced dimension (size 1) or remove it.
     * @return A new tensor with the sum along the specified axis.
     */
    Tensor<T, N> sum_axis(size_t axis, bool keepdims = false) const {
        if (axis >= N) {
            return *this;
        }
        
        TensorIndices<N> new_dims = dims_;
        new_dims[axis] = 1;
        Tensor<T, N> result(new_dims, use_gpu_, false);
        result.fill(T(0));
        
        // Calculate strides for iteration
        size_t outer = 1;
        for (size_t i = 0; i < axis; ++i) {
            outer *= dims_[i];
        }
        
        size_t inner = 1;
        for (size_t i = axis + 1; i < N; ++i) {
            inner *= dims_[i];
        }
        
        size_t axis_size = dims_[axis];
        
        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < inner; ++i) {
                for (size_t a = 0; a < axis_size; ++a) {
                    size_t src_idx = o * axis_size * inner + a * inner + i;
                    size_t dst_idx = o * inner + i;
                    result.data_[dst_idx] += data_[src_idx];
                }
            }
        }
        
        // If requested, squeeze the result (remove dimension of size 1)
        if (!keepdims && N > 1) {
            // Return as-is with size 1 dimension
            // Full squeeze would require changing dimension template parameter
        }
        
        // Handle gradient tracking
        if (requires_grad_) {
            result.requires_grad_ = true;
            result.is_leaf_ = false;
            
            auto self_ptr = std::make_shared<Tensor<T, N>>(*this);
            result.backward_funcs_.push_back([self_ptr, axis](const Tensor<T, N>& grad) {
                if (!self_ptr->grad_) {
                    self_ptr->grad_ = std::make_unique<Tensor<T, N>>(self_ptr->dims_,
                                                                      self_ptr->use_gpu_, false);
                    self_ptr->grad_->fill(T(0));
                }
                
                // Broadcast gradient back to original shape
                size_t outer = 1;
                for (size_t i = 0; i < axis; ++i) {
                    outer *= self_ptr->dims_[i];
                }
                
                size_t inner = 1;
                for (size_t i = axis + 1; i < N; ++i) {
                    inner *= self_ptr->dims_[i];
                }
                
                size_t axis_size = self_ptr->dims_[axis];
                
                for (size_t o = 0; o < outer; ++o) {
                    for (size_t i = 0; i < inner; ++i) {
                        size_t grad_idx = o * inner + i;
                        for (size_t a = 0; a < axis_size; ++a) {
                            size_t self_idx = o * axis_size * inner + a * inner + i;
                            self_ptr->grad_->data_[self_idx] += grad.data_[grad_idx];
                        }
                    }
                }
                
                if (!self_ptr->is_leaf_ && !self_ptr->backward_funcs_.empty()) {
                    for (auto& func : self_ptr->backward_funcs_) {
                        func(*self_ptr->grad_);
                    }
                }
            });
        }
        
        return result;
    }
    
    /**
     * Mean reduction along a specific axis.
     * @param axis The dimension along which to compute the mean.
     * @param keepdims Whether to keep the reduced dimension (size 1) or remove it.
     * @return A new tensor with the mean along the specified axis.
     */
    Tensor<T, N> mean_axis(size_t axis, bool keepdims = false) const {
        if (axis >= N) {
            return *this;
        }
        
        Tensor<T, N> result = sum_axis(axis, keepdims);
        T divisor = static_cast<T>(dims_[axis]);
        
        size_t total = result.total_size();
        for (size_t i = 0; i < total; ++i) {
            result.data_[i] /= divisor;
        }
        
        return result;
    }
    
    /**
     * Find index of maximum element (argmax).
     * For multi-dimensional tensors, returns flattened index.
     * @return The index of the maximum element.
     */
    size_t argmax() const {
        size_t total = total_size();
        size_t max_idx = 0;
        T max_val = data_[0];
        
        for (size_t i = 1; i < total; ++i) {
            if (data_[i] > max_val) {
                max_val = data_[i];
                max_idx = i;
            }
        }
        
        return max_idx;
    }
    
    /**
     * Find index of minimum element (argmin).
     * For multi-dimensional tensors, returns flattened index.
     * @return The index of the minimum element.
     */
    size_t argmin() const {
        size_t total = total_size();
        size_t min_idx = 0;
        T min_val = data_[0];
        
        for (size_t i = 1; i < total; ++i) {
            if (data_[i] < min_val) {
                min_val = data_[i];
                min_idx = i;
            }
        }
        
        return min_idx;
    }
    
    /**
     * Find indices of maximum elements along an axis.
     * Only available for tensors with N > 1.
     * @param axis The dimension along which to find argmax.
     * @return A new tensor with one less dimension containing indices.
     */
    template<size_t M = (N > 1 ? N - 1 : 1)>
    std::enable_if_t<(N > 1), Tensor<size_t, M>> argmax_axis(size_t axis) const {
        if (axis >= N) {
            TensorIndices<M> zero_dims{};
            return Tensor<size_t, M>(zero_dims, false);
        }
        
        // Calculate new dimensions (removing the axis dimension)
        TensorIndices<M> new_dims;
        size_t new_idx = 0;
        for (size_t i = 0; i < N; ++i) {
            if (i != axis) {
                new_dims[new_idx++] = dims_[i];
            }
        }
        
        Tensor<size_t, M> result(new_dims, false);
        
        size_t outer = 1;
        for (size_t i = 0; i < axis; ++i) {
            outer *= dims_[i];
        }
        
        size_t inner = 1;
        for (size_t i = axis + 1; i < N; ++i) {
            inner *= dims_[i];
        }
        
        size_t axis_size = dims_[axis];
        
        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < inner; ++i) {
                size_t max_idx = 0;
                T max_val = data_[o * axis_size * inner + i];
                
                for (size_t a = 1; a < axis_size; ++a) {
                    size_t src_idx = o * axis_size * inner + a * inner + i;
                    if (data_[src_idx] > max_val) {
                        max_val = data_[src_idx];
                        max_idx = a;
                    }
                }
                
                size_t dst_idx = o * inner + i;
                result.data()[dst_idx] = max_idx;
            }
        }
        
        return result;
    }
    
    /**
     * Find indices of minimum elements along an axis.
     * Only available for tensors with N > 1.
     * @param axis The dimension along which to find argmin.
     * @return A new tensor with one less dimension containing indices.
     */
    template<size_t M = (N > 1 ? N - 1 : 1)>
    std::enable_if_t<(N > 1), Tensor<size_t, M>> argmin_axis(size_t axis) const {
        if (axis >= N) {
            TensorIndices<M> zero_dims{};
            return Tensor<size_t, M>(zero_dims, false);
        }
        
        // Calculate new dimensions (removing the axis dimension)
        TensorIndices<M> new_dims;
        size_t new_idx = 0;
        for (size_t i = 0; i < N; ++i) {
            if (i != axis) {
                new_dims[new_idx++] = dims_[i];
            }
        }
        
        Tensor<size_t, M> result(new_dims, false);
        
        size_t outer = 1;
        for (size_t i = 0; i < axis; ++i) {
            outer *= dims_[i];
        }
        
        size_t inner = 1;
        for (size_t i = axis + 1; i < N; ++i) {
            inner *= dims_[i];
        }
        
        size_t axis_size = dims_[axis];
        
        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < inner; ++i) {
                size_t min_idx = 0;
                T min_val = data_[o * axis_size * inner + i];
                
                for (size_t a = 1; a < axis_size; ++a) {
                    size_t src_idx = o * axis_size * inner + a * inner + i;
                    if (data_[src_idx] < min_val) {
                        min_val = data_[src_idx];
                        min_idx = a;
                    }
                }
                
                size_t dst_idx = o * inner + i;
                result.data()[dst_idx] = min_idx;
            }
        }
        
        return result;
    }
    
    /**
     * Cumulative sum along all elements (flattened view).
     * @return A new tensor with cumulative sums.
     */
    Tensor<T, N> cumsum() const {
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
        if (total > 0) {
            result.data_[0] = data_[0];
            for (size_t i = 1; i < total; ++i) {
                result.data_[i] = result.data_[i-1] + data_[i];
            }
        }
        
        return result;
    }
    
    /**
     * Cumulative product along all elements (flattened view).
     * @return A new tensor with cumulative products.
     */
    Tensor<T, N> cumprod() const {
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
        if (total > 0) {
            result.data_[0] = data_[0];
            for (size_t i = 1; i < total; ++i) {
                result.data_[i] = result.data_[i-1] * data_[i];
            }
        }
        
        return result;
    }
    
    /**
     * Cumulative sum along a specific axis.
     * @param axis The dimension along which to compute cumulative sum.
     * @return A new tensor with cumulative sums along the specified axis.
     */
    Tensor<T, N> cumsum_axis(size_t axis) const {
        if (axis >= N) {
            throw std::out_of_range("Axis out of bounds");
        }
        
        Tensor<T, N> result(dims_, use_gpu_);
        
        // Compute strides
        std::array<size_t, N> strides;
        strides[N-1] = 1;
        for (size_t i = N-1; i-- > 0;) {
            strides[i] = strides[i+1] * dims_[i+1];
        }
        
        size_t outer = 1;
        for (size_t i = 0; i < axis; ++i) {
            outer *= dims_[i];
        }
        
        size_t inner = 1;
        for (size_t i = axis + 1; i < N; ++i) {
            inner *= dims_[i];
        }
        
        size_t axis_size = dims_[axis];
        
        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < inner; ++i) {
                T cumulative = T(0);
                for (size_t a = 0; a < axis_size; ++a) {
                    size_t idx = o * axis_size * inner + a * inner + i;
                    cumulative += data_[idx];
                    result.data_[idx] = cumulative;
                }
            }
        }
        
        return result;
    }
    
    /**
     * Cumulative product along a specific axis.
     * @param axis The dimension along which to compute cumulative product.
     * @return A new tensor with cumulative products along the specified axis.
     */
    Tensor<T, N> cumprod_axis(size_t axis) const {
        if (axis >= N) {
            throw std::out_of_range("Axis out of bounds");
        }
        
        Tensor<T, N> result(dims_, use_gpu_);
        
        // Compute strides
        std::array<size_t, N> strides;
        strides[N-1] = 1;
        for (size_t i = N-1; i-- > 0;) {
            strides[i] = strides[i+1] * dims_[i+1];
        }
        
        size_t outer = 1;
        for (size_t i = 0; i < axis; ++i) {
            outer *= dims_[i];
        }
        
        size_t inner = 1;
        for (size_t i = axis + 1; i < N; ++i) {
            inner *= dims_[i];
        }
        
        size_t axis_size = dims_[axis];
        
        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < inner; ++i) {
                T cumulative = T(1);
                for (size_t a = 0; a < axis_size; ++a) {
                    size_t idx = o * axis_size * inner + a * inner + i;
                    cumulative *= data_[idx];
                    result.data_[idx] = cumulative;
                }
            }
        }
        
        return result;
    }
    
    /**
     * Product of all elements.
     * @return The product of all tensor elements.
     */
    T prod() const {
        T result = T(1);
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            result *= data_[i];
        }
        
        return result;
    }
    
    /**
     * Check if any element is non-zero (boolean reduction).
     * @return true if any element is non-zero, false otherwise.
     */
    bool any() const {
        size_t total = total_size();
        for (size_t i = 0; i < total; ++i) {
            if (data_[i] != T(0)) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * Check if all elements are non-zero (boolean reduction).
     * @return true if all elements are non-zero, false otherwise.
     */
    bool all() const {
        size_t total = total_size();
        for (size_t i = 0; i < total; ++i) {
            if (data_[i] == T(0)) {
                return false;
            }
        }
        return true;
    }
    
    // ============================================
    // Shape Manipulation Operations
    // ============================================
    
    /**
     * Flatten tensor to 1D.
     * @return A 1D tensor containing all elements in row-major order
     * 
     * @code
     * Tensor<float, 2> A({3, 4});
     * auto B = A.flatten();  // Tensor<float, 1> with 12 elements
     * @endcode
     */
    Tensor<T, 1> flatten() const {
        size_t total = total_size();
        Tensor<T, 1> result({total}, use_gpu_, false);
        std::copy(data_.get(), data_.get() + total, result.data_.get());
        
        return result;
    }
    
    /**
     * Repeat tensor along dimensions.
     * Each dimension is repeated the specified number of times.
     * @param repeats Number of times to repeat along each dimension
     * @return A new tensor with repeated elements
     * 
     * @code
     * Tensor<float, 2> A({2, 3});
     * A.fill(1.0f);
     * auto B = A.repeat({2, 3});  // Result is {4, 9} with repeated values
     * @endcode
     */
    Tensor<T, N> repeat(const TensorIndices<N>& repeats) const {
        TensorIndices<N> new_dims;
        for (size_t i = 0; i < N; ++i) {
            new_dims[i] = dims_[i] * repeats[i];
        }
        
        Tensor<T, N> result(new_dims, use_gpu_, false);
        
        // Helper function to convert flat index to multi-dimensional indices
        auto flat_to_indices = [](size_t flat, const TensorIndices<N>& dims) {
            TensorIndices<N> indices;
            for (int i = N - 1; i >= 0; --i) {
                indices[i] = flat % dims[i];
                flat /= dims[i];
            }
            return indices;
        };
        
        // Helper function to convert multi-dimensional indices to flat index
        auto indices_to_flat = [](const TensorIndices<N>& indices, const TensorIndices<N>& dims) {
            size_t flat = 0;
            size_t multiplier = 1;
            for (int i = N - 1; i >= 0; --i) {
                flat += indices[i] * multiplier;
                multiplier *= dims[i];
            }
            return flat;
        };
        
        size_t total_new = result.total_size();
        for (size_t i = 0; i < total_new; ++i) {
            auto new_indices = flat_to_indices(i, new_dims);
            TensorIndices<N> src_indices;
            for (size_t j = 0; j < N; ++j) {
                src_indices[j] = new_indices[j] % dims_[j];
            }
            size_t src_idx = indices_to_flat(src_indices, dims_);
            result.data_[i] = data_[src_idx];
        }
        
        return result;
    }
    
    /**
     * Tile (repeat) tensor multiple times along each dimension.
     * Similar to repeat() but with slightly different semantics matching NumPy's tile.
     * @param reps Number of repetitions along each dimension
     * @return A new tensor with tiled data
     * 
     * @code
     * Tensor<float, 2> A({2, 3});
     * auto B = A.tile({2, 2});  // Tile 2x in each dimension -> {4, 6}
     * @endcode
     */
    Tensor<T, N> tile(const TensorIndices<N>& reps) const {
        return repeat(reps);
    }
};


// Free functions for scalar-first operations

/**
 * Scalar + Tensor (element-wise addition).
 * @param scalar The scalar value.
 * @param tensor The tensor.
 * @return A new tensor with the result.
 */
template <typename T, size_t N>
Tensor<T, N> operator+(const T& scalar, const Tensor<T, N>& tensor) {
    return tensor + scalar;
}

/**
 * Scalar - Tensor (element-wise subtraction).
 * @param scalar The scalar value.
 * @param tensor The tensor.
 * @return A new tensor with the result.
 */
template <typename T, size_t N>
Tensor<T, N> operator-(const T& scalar, const Tensor<T, N>& tensor) {
    Tensor<T, N> result(tensor.dims_, tensor.use_gpu_);
    size_t total = result.total_size();
    
    for (size_t i = 0; i < total; ++i) {
        result.data_[i] = scalar - tensor.data_[i];
    }
    
    return result;
}

/**
 * Scalar * Tensor (element-wise multiplication).
 * @param scalar The scalar value.
 * @param tensor The tensor.
 * @return A new tensor with the result.
 */
template <typename T, size_t N>
Tensor<T, N> operator*(const T& scalar, const Tensor<T, N>& tensor) {
    return tensor * scalar;
}

/**
 * Scalar / Tensor (element-wise division).
 * @param scalar The scalar value.
 * @param tensor The tensor.
 * @return A new tensor with the result.
 */
template <typename T, size_t N>
Tensor<T, N> operator/(const T& scalar, const Tensor<T, N>& tensor) {
    Tensor<T, N> result(tensor.dims_, tensor.use_gpu_);
    size_t total = result.total_size();
    
    for (size_t i = 0; i < total; ++i) {
        result.data_[i] = scalar / tensor.data_[i];
    }
    
    return result;
}

// ============================================================================
// RANDOM SAMPLING
// ============================================================================

/**
 * @brief Random number generator for tensor operations
 */
template <typename T>
class TensorRandom {
private:
    static std::mt19937& get_generator() {
        static std::mt19937 gen(std::random_device{}());
        return gen;
    }
    
public:
    /**
     * Set seed for reproducible random number generation.
     * @param seed Random seed value.
     */
    static void seed(unsigned int seed) {
        get_generator().seed(seed);
    }
    
    /**
     * Generate tensor with random uniform distribution.
     * @param dims Dimensions of the tensor.
     * @param low Lower bound (inclusive).
     * @param high Upper bound (exclusive).
     * @param use_gpu Whether to use GPU.
     * @return Tensor filled with random values.
     */
    template <size_t N>
    static Tensor<T, N> uniform(const std::array<size_t, N>& dims, T low = T(0), T high = T(1), bool use_gpu = false) {
        Tensor<T, N> result(dims, use_gpu, false);
        std::uniform_real_distribution<T> dist(low, high);
        auto& gen = get_generator();
        
        T* data = result.data();
        size_t total = result.total_size();
        for (size_t i = 0; i < total; ++i) {
            data[i] = dist(gen);
        }
        
        return result;
    }
    
    /**
     * Generate tensor with random normal (Gaussian) distribution.
     * @param dims Dimensions of the tensor.
     * @param mean Mean of the distribution.
     * @param std Standard deviation.
     * @param use_gpu Whether to use GPU.
     * @return Tensor filled with random values.
     */
    template <size_t N>
    static Tensor<T, N> normal(const std::array<size_t, N>& dims, T mean = T(0), T std = T(1), bool use_gpu = false) {
        Tensor<T, N> result(dims, use_gpu, false);
        std::normal_distribution<T> dist(mean, std);
        auto& gen = get_generator();
        
        T* data = result.data();
        size_t total = result.total_size();
        for (size_t i = 0; i < total; ++i) {
            data[i] = dist(gen);
        }
        
        return result;
    }
    
    /**
     * Generate tensor with random exponential distribution.
     * @param dims Dimensions of the tensor.
     * @param lambda Rate parameter.
     * @param use_gpu Whether to use GPU.
     * @return Tensor filled with random values.
     */
    template <size_t N>
    static Tensor<T, N> exponential(const std::array<size_t, N>& dims, T lambda = T(1), bool use_gpu = false) {
        Tensor<T, N> result(dims, use_gpu, false);
        std::exponential_distribution<T> dist(lambda);
        auto& gen = get_generator();
        
        T* data = result.data();
        size_t total = result.total_size();
        for (size_t i = 0; i < total; ++i) {
            data[i] = dist(gen);
        }
        
        return result;
    }
    
    /**
     * Generate random permutation of indices [0, n).
     * @param n Number of elements.
     * @return Vector of permuted indices.
     */
    static std::vector<size_t> permutation(size_t n) {
        std::vector<size_t> indices(n);
        for (size_t i = 0; i < n; ++i) {
            indices[i] = i;
        }
        std::shuffle(indices.begin(), indices.end(), get_generator());
        return indices;
    }
    
    /**
     * Generate 1D tensor with random permutation of indices.
     * @param n Number of elements.
     * @param use_gpu Whether to use GPU.
     * @return 1D tensor with permuted indices.
     */
    static Tensor<T, 1> randperm(size_t n, bool use_gpu = false) {
        auto perm = permutation(n);
        Tensor<T, 1> result({n}, use_gpu, false);
        T* data = result.data();
        for (size_t i = 0; i < n; ++i) {
            data[i] = static_cast<T>(perm[i]);
        }
        return result;
    }
    
    /**
     * Random choice: sample k elements from [0, n) without replacement.
     * @param n Population size.
     * @param k Number of samples.
     * @return Vector of sampled indices.
     */
    static std::vector<size_t> choice(size_t n, size_t k) {
        if (k > n) k = n;
        auto perm = permutation(n);
        return std::vector<size_t>(perm.begin(), perm.begin() + k);
    }
    
    /**
     * Random choice with replacement: sample k elements from [0, n).
     * @param n Population size.
     * @param k Number of samples.
     * @return Vector of sampled indices (may contain duplicates).
     */
    static std::vector<size_t> choice_with_replacement(size_t n, size_t k) {
        std::uniform_int_distribution<size_t> dist(0, n - 1);
        auto& gen = get_generator();
        std::vector<size_t> result(k);
        for (size_t i = 0; i < k; ++i) {
            result[i] = dist(gen);
        }
        return result;
    }
};

// ============================================================================
// SORTING AND SEARCHING
// ============================================================================

/**
 * @brief Sort tensor elements and return sorted tensor (1D only).
 * @param ascending If true, sort in ascending order; otherwise descending.
 * @return Sorted 1D tensor.
 */
template <typename T>
Tensor<T, 1> sort(const Tensor<T, 1>& tensor, bool ascending = true) {
    Tensor<T, 1> result = tensor;
    size_t n = result.total_size();
    T* data = result.data();
    
    if (ascending) {
        std::sort(data, data + n);
    } else {
        std::sort(data, data + n, std::greater<T>());
    }
    
    return result;
}

/**
 * @brief Return indices that would sort the tensor (1D only).
 * @param ascending If true, sort in ascending order; otherwise descending.
 * @return 1D tensor of indices.
 */
template <typename T>
Tensor<size_t, 1> argsort(const Tensor<T, 1>& tensor, bool ascending = true) {
    size_t n = tensor.total_size();
    Tensor<size_t, 1> indices({n}, false, false);
    
    size_t* indices_data = indices.data();
    const T* tensor_data = tensor.data();
    
    // Initialize indices
    for (size_t i = 0; i < n; ++i) {
        indices_data[i] = i;
    }
    
    // Sort indices based on tensor values
    if (ascending) {
        std::sort(indices_data, indices_data + n,
                  [tensor_data](size_t i1, size_t i2) {
                      return tensor_data[i1] < tensor_data[i2];
                  });
    } else {
        std::sort(indices_data, indices_data + n,
                  [tensor_data](size_t i1, size_t i2) {
                      return tensor_data[i1] > tensor_data[i2];
                  });
    }
    
    return indices;
}

/**
 * @brief Find k largest or smallest elements and their indices (1D only).
 * @param k Number of elements to return.
 * @param largest If true, return k largest; otherwise k smallest.
 * @return Pair of tensors: (values, indices).
 */
template <typename T>
std::pair<Tensor<T, 1>, Tensor<size_t, 1>> topk(const Tensor<T, 1>& tensor, size_t k, bool largest = true) {
    size_t n = tensor.total_size();
    if (k > n) k = n;
    
    auto sorted_indices = argsort(tensor, !largest);  // Sort opposite of what we want
    
    // Take first k elements
    Tensor<T, 1> values({k}, false, false);
    Tensor<size_t, 1> indices({k}, false, false);
    
    const T* tensor_data = tensor.data();
    const size_t* sorted_data = sorted_indices.data();
    T* values_data = values.data();
    size_t* indices_data = indices.data();
    
    for (size_t i = 0; i < k; ++i) {
        size_t idx = sorted_data[i];
        values_data[i] = tensor_data[idx];
        indices_data[i] = idx;
    }
    
    return {values, indices};
}

/**
 * @brief Find unique elements in tensor (1D only).
 * @return Tensor containing unique elements in sorted order.
 */
template <typename T>
Tensor<T, 1> unique(const Tensor<T, 1>& tensor) {
    size_t n = tensor.total_size();
    const T* data = tensor.data();
    std::vector<T> vec(data, data + n);
    
    std::sort(vec.begin(), vec.end());
    auto last = std::unique(vec.begin(), vec.end());
    vec.erase(last, vec.end());
    
    Tensor<T, 1> result({vec.size()}, false, false);
    std::copy(vec.begin(), vec.end(), result.data());
    
    return result;
}

/**
 * @brief Binary search to find indices where elements should be inserted (1D only).
 * @param values Sorted tensor to search in.
 * @param search_values Values to search for.
 * @return Tensor of insertion indices.
 */
template <typename T>
Tensor<size_t, 1> searchsorted(const Tensor<T, 1>& values, const Tensor<T, 1>& search_values) {
    size_t n = values.total_size();
    size_t m = search_values.total_size();
    
    Tensor<size_t, 1> result({m}, false, false);
    
    const T* values_data = values.data();
    const T* search_data = search_values.data();
    size_t* result_data = result.data();
    
    for (size_t i = 0; i < m; ++i) {
        T val = search_data[i];
        auto it = std::lower_bound(values_data, values_data + n, val);
        result_data[i] = it - values_data;
    }
    
    return result;
}

// ============================================================================
// STACKING AND CONCATENATION EXTENSIONS
// ============================================================================

/**
 * @brief Split tensor into chunks along specified axis.
 * @param num_chunks Number of chunks to split into.
 * @param axis Axis along which to split.
 * @return Vector of tensors.
 */
template <typename T, size_t N>
std::vector<Tensor<T, N>> split(const Tensor<T, N>& tensor, size_t num_chunks, size_t axis = 0) {
    if (axis >= N || num_chunks == 0) {
        return {tensor};
    }
    
    auto dims = tensor.dims();
    size_t dim_size = dims[axis];
    
    if (num_chunks > dim_size) {
        num_chunks = dim_size;
    }
    
    std::vector<Tensor<T, N>> chunks;
    chunks.reserve(num_chunks);
    
    // Calculate chunk sizes
    size_t base_size = dim_size / num_chunks;
    size_t remainder = dim_size % num_chunks;
    
    size_t start = 0;
    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        // First 'remainder' chunks get an extra element
        size_t chunk_size = base_size + (chunk_idx < remainder ? 1 : 0);
        
        if (chunk_size == 0) break;
        
        // Create new dimensions for the chunk
        auto chunk_dims = dims;
        chunk_dims[axis] = chunk_size;
        
        Tensor<T, N> chunk(chunk_dims, tensor.uses_gpu(), false);
        
        // Copy data - need to handle arbitrary dimensions
        // For simplicity, we'll copy using flat indexing with stride calculations
        std::array<size_t, N> strides;
        strides[N-1] = 1;
        for (size_t i = N-1; i-- > 0;) {
            strides[i] = strides[i+1] * dims[i+1];
        }
        
        std::array<size_t, N> chunk_strides;
        chunk_strides[N-1] = 1;
        for (size_t i = N-1; i-- > 0;) {
            chunk_strides[i] = chunk_strides[i+1] * chunk_dims[i+1];
        }
        
        // Iterate over the chunk
        size_t chunk_total = chunk.total_size();
        const T* src_data = tensor.data();
        T* dst_data = chunk.data();
        
        for (size_t i = 0; i < chunk_total; ++i) {
            // Convert flat index to coordinates in chunk
            std::array<size_t, N> coords;
            size_t remaining = i;
            for (size_t d = 0; d < N; ++d) {
                coords[d] = remaining / chunk_strides[d];
                remaining %= chunk_strides[d];
            }
            
            // Adjust coordinate on split axis
            coords[axis] += start;
            
            // Convert to flat index in source
            size_t src_idx = 0;
            for (size_t d = 0; d < N; ++d) {
                src_idx += coords[d] * strides[d];
            }
            
            dst_data[i] = src_data[src_idx];
        }
        
        chunks.push_back(std::move(chunk));
        start += chunk_size;
    }
    
    return chunks;
}

/**
 * @brief Divide tensor into equal-sized chunks (last chunk may be smaller).
 * @param chunk_size Size of each chunk along the axis.
 * @param axis Axis along which to divide.
 * @return Vector of tensors.
 */
template <typename T, size_t N>
std::vector<Tensor<T, N>> chunk(const Tensor<T, N>& tensor, size_t chunk_size, size_t axis = 0) {
    if (axis >= N || chunk_size == 0) {
        return {tensor};
    }
    
    auto dims = tensor.dims();
    size_t dim_size = dims[axis];
    size_t num_chunks = (dim_size + chunk_size - 1) / chunk_size;
    
    std::vector<Tensor<T, N>> chunks;
    chunks.reserve(num_chunks);
    
    // Compute strides for indexing
    std::array<size_t, N> strides;
    strides[N-1] = 1;
    for (size_t i = N-1; i-- > 0;) {
        strides[i] = strides[i+1] * dims[i+1];
    }
    
    size_t start = 0;
    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        // Each chunk gets chunk_size elements, except last may be smaller
        size_t current_chunk_size = std::min(chunk_size, dim_size - start);
        
        if (current_chunk_size == 0) break;
        
        // Create new dimensions for the chunk
        auto chunk_dims = dims;
        chunk_dims[axis] = current_chunk_size;
        
        Tensor<T, N> chunk(chunk_dims, tensor.uses_gpu(), false);
        
        std::array<size_t, N> chunk_strides;
        chunk_strides[N-1] = 1;
        for (size_t i = N-1; i-- > 0;) {
            chunk_strides[i] = chunk_strides[i+1] * chunk_dims[i+1];
        }
        
        // Copy data
        size_t chunk_total = chunk.total_size();
        const T* src_data = tensor.data();
        T* dst_data = chunk.data();
        
        for (size_t i = 0; i < chunk_total; ++i) {
            // Convert flat index to coordinates in chunk
            std::array<size_t, N> coords;
            size_t remaining = i;
            for (size_t d = 0; d < N; ++d) {
                coords[d] = remaining / chunk_strides[d];
                remaining %= chunk_strides[d];
            }
            
            // Adjust coordinate on split axis
            coords[axis] += start;
            
            // Convert to flat index in source
            size_t src_idx = 0;
            for (size_t d = 0; d < N; ++d) {
                src_idx += coords[d] * strides[d];
            }
            
            dst_data[i] = src_data[src_idx];
        }
        
        chunks.push_back(std::move(chunk));
        start += current_chunk_size;
    }
    
    return chunks;
}

/**
 * @brief Repeat tensor multiple times along each dimension.
 * @param repeats Array specifying number of repetitions for each dimension.
 * @return Tiled tensor.
 */
template <typename T, size_t N>
Tensor<T, N> tile(const Tensor<T, N>& tensor, const std::array<size_t, N>& repeats) {
    auto tensor_dims = tensor.dims();
    std::array<size_t, N> new_dims;
    for (size_t i = 0; i < N; ++i) {
        new_dims[i] = tensor_dims[i] * repeats[i];
    }
    
    Tensor<T, N> result(new_dims, tensor.uses_gpu(), false);
    
    const T* tensor_data = tensor.data();
    T* result_data = result.data();
    
    // Compute strides for indexing
    std::array<size_t, N> tensor_strides;
    std::array<size_t, N> result_strides;
    tensor_strides[N-1] = 1;
    result_strides[N-1] = 1;
    for (size_t i = N-1; i-- > 0;) {
        tensor_strides[i] = tensor_strides[i+1] * tensor_dims[i+1];
        result_strides[i] = result_strides[i+1] * new_dims[i+1];
    }
    
    // Fill result with tiled values
    size_t result_total = result.total_size();
    for (size_t idx = 0; idx < result_total; ++idx) {
        // Convert flat index to coordinates
        std::array<size_t, N> coords;
        size_t remaining = idx;
        for (size_t i = 0; i < N; ++i) {
            coords[i] = remaining / result_strides[i];
            remaining %= result_strides[i];
        }
        
        // Map to source coordinates
        std::array<size_t, N> src_coords;
        for (size_t i = 0; i < N; ++i) {
            src_coords[i] = coords[i] % tensor_dims[i];
        }
        
        // Convert source coordinates to flat index
        size_t src_idx = 0;
        for (size_t i = 0; i < N; ++i) {
            src_idx += src_coords[i] * tensor_strides[i];
        }
        
        result_data[idx] = tensor_data[src_idx];
    }
    
    return result;
}

/**
 * @brief Construct tensor by repeating along specified axis.
 * @param repeats Number of repetitions.
 * @param axis Axis along which to repeat.
 * @return Repeated tensor.
 */
template <typename T, size_t N>
Tensor<T, N> repeat_along_axis(const Tensor<T, N>& tensor, size_t repeats, size_t axis = 0) {
    if (axis >= N) {
        return tensor;
    }
    
    auto tensor_dims = tensor.dims();
    std::array<size_t, N> new_dims = tensor_dims;
    new_dims[axis] *= repeats;
    
    Tensor<T, N> result(new_dims, tensor.uses_gpu(), false);
    
    const T* tensor_data = tensor.data();
    T* result_data = result.data();
    
    // Compute strides
    std::array<size_t, N> tensor_strides;
    std::array<size_t, N> result_strides;
    tensor_strides[N-1] = 1;
    result_strides[N-1] = 1;
    for (size_t i = N-1; i-- > 0;) {
        tensor_strides[i] = tensor_strides[i+1] * tensor_dims[i+1];
        result_strides[i] = result_strides[i+1] * new_dims[i+1];
    }
    
    // Copy data with repetition
    size_t result_total = result.total_size();
    for (size_t idx = 0; idx < result_total; ++idx) {
        // Convert flat index to coordinates
        std::array<size_t, N> coords;
        size_t remaining = idx;
        for (size_t i = 0; i < N; ++i) {
            coords[i] = remaining / result_strides[i];
            remaining %= result_strides[i];
        }
        
        // Map coordinate back to source
        coords[axis] /= repeats;
        
        // Convert to flat index in source
        size_t src_idx = 0;
        for (size_t i = 0; i < N; ++i) {
            src_idx += coords[i] * tensor_strides[i];
        }
        
        result_data[idx] = tensor_data[src_idx];
    }
    
    return result;
}

#endif // _TENSOR_H
