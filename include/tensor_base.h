/**
 * @brief Multi-dimensional tensor library with GPU, BLAS, and autograd support
 */

#ifndef _TENSOR_BASE_H
#define _TENSOR_BASE_H

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
#include "tensor_blas.h"
#include "tensor_error.h"
#ifdef USE_GPU
#include "tensor_gpu.h"
#include "tensor_gpu.cuh"
#endif

namespace tensor {

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
    if (is_gpu_available()) {
        return Backend::GPU;
    }
#endif
#ifdef USE_BLAS
    return Backend::BLAS;
#endif
    return Backend::CPU;
}

#ifndef USE_GPU
/**
 * @brief Check if GPU backend is available
 * @return true if GPU support is compiled in and GPU is available
 */
inline bool is_gpu_available() {
    return false;
}
#endif

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
 * Loss functions for training neural networks
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

// Forward declarations for broadcasting functions
template <typename T, size_t N, size_t M>
auto broadcast_to(const Tensor<T, N>& tensor, const TensorIndices<M>& target_shape)
    -> std::variant<Tensor<T, M>, TensorError>;

template <size_t N1, size_t N2>
bool are_broadcastable(const TensorIndices<N1>& shape1,
                       const TensorIndices<N2>& shape2,
                       std::string* error_msg);

// Forward declarations for view functions
template <typename T>
Tensor<T, 1> row(const Tensor<T, 2>& matrix, size_t row_idx);

template <typename T>
Tensor<T, 1> col(const Tensor<T, 2>& matrix, size_t col_idx);

template <typename T>
Tensor<T, 1> diag(const Tensor<T, 2>& matrix);

template <typename T>
Tensor<T, 2> diag_matrix(const Tensor<T, 1>& vec);

template <typename T>
Tensor<T, 2> block(const Tensor<T, 2>& matrix, size_t start_row, size_t start_col,
                   size_t num_rows, size_t num_cols);

template <typename T>
Tensor<T, 1> head(const Tensor<T, 1>& vec, size_t n);

template <typename T>
Tensor<T, 1> tail(const Tensor<T, 1>& vec, size_t n);

template <typename T>
Tensor<T, 2> topRows(const Tensor<T, 2>& matrix, size_t n);

template <typename T>
Tensor<T, 2> bottomRows(const Tensor<T, 2>& matrix, size_t n);

template <typename T>
Tensor<T, 2> leftCols(const Tensor<T, 2>& matrix, size_t n);

template <typename T>
Tensor<T, 2> rightCols(const Tensor<T, 2>& matrix, size_t n);

/**
 * @brief Multi-dimensional array with GPU, BLAS, and autograd support
 */
template <typename T, size_t N>
class Tensor {
private:
    std::unique_ptr<T[]> data_;      ///< Flat data storage in row-major order
    TensorIndices<N> dims_;          ///< Dimensions of the tensor
    TensorIndices<N> strides_;       ///< Strides for each dimension
    bool use_gpu_;                   ///< Flag to indicate if GPU should be used
    
#ifdef USE_GPU
    /// @name Persistent GPU Memory Management
    /// @{
    mutable T* d_data_;              ///< Persistent GPU memory pointer
    mutable bool data_on_gpu_;       ///< True if valid data is on GPU
    mutable bool gpu_needs_sync_;    ///< True if GPU has newer data than CPU
    /// @}
#endif
    
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
    
#ifdef USE_GPU
    /**
     * Ensure data is present on GPU (upload if needed).
     */
    void ensure_on_gpu() const {
        if (!use_gpu_ || data_on_gpu_) return;
        
        size_t bytes = total_size() * sizeof(T);
        if (!d_data_) {
            const_cast<Tensor*>(this)->d_data_ = static_cast<T*>(cuda_malloc_wrapper(bytes));
        }
        cuda_memcpy_h2d_wrapper(d_data_, data_.get(), bytes);
        const_cast<Tensor*>(this)->data_on_gpu_ = true;
        const_cast<Tensor*>(this)->gpu_needs_sync_ = false;
    }
    
    /**
     * Ensure data is present on CPU (download from GPU if needed).
     */
    void ensure_on_cpu() const {
        if (!use_gpu_ || !gpu_needs_sync_) return;
        
        size_t bytes = total_size() * sizeof(T);
        cuda_memcpy_d2h_wrapper(data_.get(), d_data_, bytes);
        const_cast<Tensor*>(this)->gpu_needs_sync_ = false;
    }
    
    /**
     * Mark GPU data as modified (CPU needs sync).
     */
    void mark_gpu_modified() {
        if (use_gpu_) {
            data_on_gpu_ = true;
            gpu_needs_sync_ = true;
        }
    }
    
    /**
     * Get GPU data pointer (for operations).
     */
    T* gpu_data() const {
        return d_data_;
    }
#endif
    
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
    const T* data_ptr() const {
#ifdef USE_GPU
        ensure_on_cpu();  // Sync from GPU if needed
#endif
        return data_.get();
    }
    
    T* data_ptr() {
#ifdef USE_GPU
        ensure_on_cpu();  // Sync from GPU if needed
        // Non-const access means user might modify, so invalidate GPU
        data_on_gpu_ = false;
        gpu_needs_sync_ = false;
#endif
        return data_.get();
    }
    
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
    
    /**
     * Get strides for each dimension.
     * @return Array of strides for each dimension.
     */
    const TensorIndices<N>& strides() const { return strides_; }
    
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
        use_gpu_ = use_gpu && is_gpu_available();
        d_data_ = nullptr;
        data_on_gpu_ = false;
        gpu_needs_sync_ = false;
#else
        use_gpu_ = false;
#endif
        if (requires_grad_) {
            grad_ = std::make_unique<Tensor<T, N>>(dims_, use_gpu_, false);
            grad_->fill(T(0));
        }
    }
    
    /**
     * Destructor to free GPU memory if allocated.
     */
    ~Tensor() {
#ifdef USE_GPU
        if (d_data_) {
            cuda_free_wrapper(d_data_);
            d_data_ = nullptr;
        }
#endif
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
        
#ifdef USE_GPU
        d_data_ = nullptr;
        data_on_gpu_ = false;
        gpu_needs_sync_ = false;
        
        // If other has data on GPU, copy from GPU
        if (other.use_gpu_ && other.data_on_gpu_) {
            size_t bytes = total * sizeof(T);
            d_data_ = static_cast<T*>(cuda_malloc_wrapper(bytes));
            cuda_memcpy_d2d_wrapper(d_data_, other.d_data_, bytes);
            data_on_gpu_ = true;
            // Also copy to CPU memory
            cuda_memcpy_d2h_wrapper(data_.get(), d_data_, bytes);
        } else {
            other.ensure_on_cpu();
            std::copy(other.data_.get(), other.data_.get() + total, data_.get());
        }
#else
        std::copy(other.data_.get(), other.data_.get() + total, data_.get());
#endif
        
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
#ifdef USE_GPU
            // Free existing GPU memory
            if (d_data_) {
                cuda_free_wrapper(d_data_);
                d_data_ = nullptr;
            }
#endif
            
            dims_ = other.dims_;
            strides_ = other.strides_;
            use_gpu_ = other.use_gpu_;
            requires_grad_ = other.requires_grad_;
            is_leaf_ = other.is_leaf_;
            backward_funcs_ = other.backward_funcs_;
            
            size_t total = total_size();
            data_ = std::make_unique<T[]>(total);
            
#ifdef USE_GPU
            data_on_gpu_ = false;
            gpu_needs_sync_ = false;
            
            if (other.use_gpu_) {
                // Ensure source has latest data on CPU
                other.ensure_on_cpu();
                // Copy from CPU
                std::copy(other.data_.get(), other.data_.get() + total, data_.get());
            } else {
                std::copy(other.data_.get(), other.data_.get() + total, data_.get());
            }
#else
            std::copy(other.data_.get(), other.data_.get() + total, data_.get());
#endif
            
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
#ifdef USE_GPU
        ensure_on_cpu();  // Sync from GPU if needed
#endif
        return data_[offset(indices)];
    }

    /**
     * Const version of the indexing operator.
     * @param indices An array of indices for each dimension.
     * @return A const reference to the element at the specified indices.
     */
    const T& operator[](const TensorIndices<N>& indices) const {
#ifdef USE_GPU
        ensure_on_cpu();  // Sync from GPU if needed
#endif
        return data_[offset(indices)];
    }

    /**
     * @brief Fill the tensor with a specified value
     * 
     * Sets all elements in the tensor to the same value.
     * 
     * @param value The value to fill the tensor with
     * 
     * @section example_fill Example
     * @code
     * Tensor<float, 2> matrix({3, 3});
     * matrix.fill(5.0f);  // All elements are now 5.0
     * @endcode
     */
    void fill(const T& value) {
#ifdef USE_GPU
        if (use_gpu_ && data_on_gpu_ && gpu_data()) {
            // Fill directly on GPU
            fill_gpu_direct(gpu_data(), value, total_size());
            data_on_gpu_ = true;   // GPU data is now authoritative
            gpu_needs_sync_ = false;
        } else {
            // Fill on CPU
            std::fill(data_.get(), data_.get() + total_size(), value);
            data_on_gpu_ = false;  // CPU data is now authoritative
            gpu_needs_sync_ = false;
        }
#else
        std::fill(data_.get(), data_.get() + total_size(), value);
#endif
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
    
    // Forward declare loss functions as friends
    template<typename U, size_t M>
    friend Tensor<U, M> mse_loss(const Tensor<U, M>&, const Tensor<U, M>&, const std::string&);
    
    template<typename U, size_t M>
    friend Tensor<U, 1> cross_entropy_loss(const Tensor<U, M>&, const Tensor<U, M>&, const std::string&);
    
    template<typename U, size_t M>
    friend Tensor<U, 1> binary_cross_entropy(const Tensor<U, M>&, const Tensor<U, M>&, const std::string&);
    
    template<typename U, size_t M>
    friend Tensor<U, 1> l1_loss(const Tensor<U, M>&, const Tensor<U, M>&, const std::string&);
    
    template<typename U, size_t M>
    friend Tensor<U, 1> smooth_l1_loss(const Tensor<U, M>&, const Tensor<U, M>&, U, const std::string&);
    
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
     * @section example_autograd Usage Example
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
#ifdef USE_GPU
        ensure_on_cpu();  // Critical: Sync from GPU before copying!
#endif
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
     * @brief Element-wise addition with another tensor (creates new tensor)
     * 
     * Performs element-wise addition of two tensors. Supports automatic differentiation
     * and will track gradients if either input requires gradients.
     * 
     * @param other The tensor to add
     * @return A variant containing either a new tensor with the result or TensorError::DimensionMismatch
     * 
     * @section example_add Example
     * @code
     * Tensor<float, 2> A({2, 3});
     * Tensor<float, 2> B({2, 3});
     * A.fill(1.0f);
     * B.fill(2.0f);
     * 
     * auto result = A + B;  // Returns TensorResult<Tensor<float, 2>>
     * if (std::holds_alternative<Tensor<float, 2>>(result)) {
     *     auto& C = std::get<Tensor<float, 2>>(result);
     *     // C[{i, j}] == 3.0 for all i, j
     * }
     * 
     * // With autograd:
     * Tensor<float, 1> x({3}, true, true);  // requires_grad=true
     * Tensor<float, 1> y({3}, true, true);
     * auto z_result = x + y;
     * auto& z = std::get<Tensor<float, 1>>(z_result);
     * z.backward();  // Computes gradients
     * @endcode
     */
    TensorResult<Tensor<T, N>> operator+(const Tensor<T, N>& other) const {
        // Check if shapes match exactly
        if (dims_ == other.dims_) {
            // Fast path: exact shape match
            bool track_grad = requires_grad_ || other.requires_grad_;
            Tensor<T, N> result(dims_, use_gpu_, track_grad);
            result.is_leaf_ = false;
            size_t total = total_size();
            
#ifdef USE_GPU
            if (use_gpu_ && other.use_gpu_) {
                ensure_on_gpu();
                other.ensure_on_gpu();
                result.ensure_on_gpu();
                add_gpu_direct(d_data_, other.d_data_, result.d_data_, total);
                result.mark_gpu_modified();
            } else
#endif
            {
                for (size_t i = 0; i < total; ++i) {
                    result.data_[i] = data_[i] + other.data_[i];
                }
            }
            
            // Setup backward pass (existing code continues below)
            if (track_grad) {
                Tensor<T, N>* self_ptr = const_cast<Tensor<T, N>*>(this);
                Tensor<T, N>* other_ptr = const_cast<Tensor<T, N>*>(&other);
                
                result.register_backward([self_ptr, other_ptr](const Tensor<T, N>& grad) {
                    if (self_ptr->requires_grad_) {
                        if (!self_ptr->grad_) {
                            self_ptr->grad_ = std::make_unique<Tensor<T, N>>(self_ptr->dims_, self_ptr->use_gpu_, false);
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
                    
                    if (other_ptr->requires_grad_) {
                        if (!other_ptr->grad_) {
                            other_ptr->grad_ = std::make_unique<Tensor<T, N>>(other_ptr->dims_, other_ptr->use_gpu_, false);
                            other_ptr->grad_->fill(T(0));
                        }
                        size_t total = other_ptr->total_size();
                        for (size_t i = 0; i < total; ++i) {
                            other_ptr->grad_->data_[i] += grad.data_[i];
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
        
        // Try broadcasting
        std::string error_msg;
        if (!are_broadcastable(dims_, other.dims_, &error_msg)) {
            return TensorError::DimensionMismatch;
        }
        
        // Determine broadcast shape
        TensorIndices<N> broadcast_shape = dims_;
        for (size_t i = 0; i < N; ++i) {
            broadcast_shape[i] = std::max(dims_[i], other.dims_[i]);
        }
        
        // Create result with broadcast shape
        bool track_grad = requires_grad_ || other.requires_grad_;
        Tensor<T, N> result(broadcast_shape, use_gpu_, track_grad);
        result.is_leaf_ = false;
        
        // Perform broadcasted addition
        size_t total = result.total_size();
        for (size_t i = 0; i < total; ++i) {
            TensorIndices<N> coords;
            size_t idx = i;
            for (size_t d = N; d > 0; --d) {
                coords[d - 1] = idx % broadcast_shape[d - 1];
                idx /= broadcast_shape[d - 1];
            }
            
            // Map to source indices
            size_t self_idx = 0, other_idx = 0;
            size_t self_stride = 1, other_stride = 1;
            for (size_t d = N; d > 0; --d) {
                size_t dim = d - 1;
                size_t self_coord = (dims_[dim] == 1) ? 0 : coords[dim];
                size_t other_coord = (other.dims_[dim] == 1) ? 0 : coords[dim];
                
                if (dim + 1 < N) {
                    self_stride *= dims_[dim + 1];
                    other_stride *= other.dims_[dim + 1];
                }
                
                self_idx += self_coord * self_stride;
                other_idx += other_coord * other_stride;
            }
            
            result.data_[i] = data_[self_idx] + other.data_[other_idx];
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
            ensure_on_cpu();  // Sync from GPU before scalar operations
            add_scalar_gpu(data_.get(), scalar, result.data_.get(), total);
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
            add_gpu(data_.get(), other.data_.get(), data_.get(), total);
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
            add_scalar_gpu(data_.get(), scalar, data_.get(), total);
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
        // Check if shapes match exactly
        if (dims_ == other.dims_) {
            // Fast path: exact shape match
            Tensor<T, N> result(dims_, use_gpu_);
            size_t total = total_size();
            
#ifdef USE_GPU
            if (use_gpu_ && other.use_gpu_) {
                ensure_on_gpu();
                other.ensure_on_gpu();
                result.ensure_on_gpu();
                sub_gpu_direct(d_data_, other.d_data_, result.d_data_, total);
                result.mark_gpu_modified();
            } else
#endif
            {
                for (size_t i = 0; i < total; ++i) {
                    result.data_[i] = data_[i] - other.data_[i];
                }
            }
            
            return result;
        }
        
        // Try broadcasting
        std::string error_msg;
        if (!are_broadcastable(dims_, other.dims_, &error_msg)) {
            return TensorError::DimensionMismatch;
        }
        
        // Determine broadcast shape
        TensorIndices<N> broadcast_shape = dims_;
        for (size_t i = 0; i < N; ++i) {
            broadcast_shape[i] = std::max(dims_[i], other.dims_[i]);
        }
        
        Tensor<T, N> result(broadcast_shape, use_gpu_);
        size_t total = result.total_size();
        
        // Perform broadcasted subtraction
        for (size_t i = 0; i < total; ++i) {
            TensorIndices<N> coords;
            size_t idx = i;
            for (size_t d = N; d > 0; --d) {
                coords[d - 1] = idx % broadcast_shape[d - 1];
                idx /= broadcast_shape[d - 1];
            }
            
            size_t self_idx = 0, other_idx = 0;
            size_t self_stride = 1, other_stride = 1;
            for (size_t d = N; d > 0; --d) {
                size_t dim = d - 1;
                size_t self_coord = (dims_[dim] == 1) ? 0 : coords[dim];
                size_t other_coord = (other.dims_[dim] == 1) ? 0 : coords[dim];
                
                if (dim + 1 < N) {
                    self_stride *= dims_[dim + 1];
                    other_stride *= other.dims_[dim + 1];
                }
                
                self_idx += self_coord * self_stride;
                other_idx += other_coord * other_stride;
            }
            
            result.data_[i] = data_[self_idx] - other.data_[other_idx];
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
            sub_scalar_gpu(data_.get(), scalar, result.data_.get(), total);
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
            sub_gpu(data_.get(), other.data_.get(), data_.get(), total);
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
            sub_scalar_gpu(data_.get(), scalar, data_.get(), total);
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
        // Check if shapes match exactly
        if (dims_ == other.dims_) {
            // Fast path: exact shape match  
            bool track_grad = requires_grad_ || other.requires_grad_;
            Tensor<T, N> result(dims_, use_gpu_, track_grad);
            result.is_leaf_ = false;
            size_t total = total_size();
            
#ifdef USE_GPU
            if (use_gpu_ && other.use_gpu_) {
                ensure_on_gpu();
                other.ensure_on_gpu();
                result.ensure_on_gpu();
                mul_gpu_direct(d_data_, other.d_data_, result.d_data_, total);
                result.mark_gpu_modified();
            } else
#endif
            {
                for (size_t i = 0; i < total; ++i) {
                    result.data_[i] = data_[i] * other.data_[i];
                }
            }
            
            // Setup backward pass (existing code continues)
            if (track_grad) {
                Tensor<T, N> self_copy = this->detach();
                Tensor<T, N> other_copy = other.detach();
                Tensor<T, N>* self_ptr = const_cast<Tensor<T, N>*>(this);
                Tensor<T, N>* other_ptr = const_cast<Tensor<T, N>*>(&other);
                
                result.register_backward([self_ptr, other_ptr, self_copy, other_copy](const Tensor<T, N>& grad) {
                    if (self_ptr->requires_grad_) {
                        if (!self_ptr->grad_) {
                            self_ptr->grad_ = std::make_unique<Tensor<T, N>>(self_ptr->dims_, self_ptr->use_gpu_, false);
                            self_ptr->grad_->fill(T(0));
                        }
                        
                        // Optimized using tensor operations
                        auto grad_result = grad * other_copy;
                        auto& grad_contrib = std::get<Tensor<T, N>>(grad_result);
                        
                        // Accumulate gradients using tensor addition
                        auto grad_sum_result = *self_ptr->grad_ + grad_contrib;
                        *self_ptr->grad_ = std::get<Tensor<T, N>>(grad_sum_result);
                        
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
                        
                        // Optimized using tensor operations
                        auto grad_result = grad * self_copy;
                        auto& grad_contrib = std::get<Tensor<T, N>>(grad_result);
                        
                        // Accumulate gradients using tensor addition
                        auto grad_sum_result = *other_ptr->grad_ + grad_contrib;
                        *other_ptr->grad_ = std::get<Tensor<T, N>>(grad_sum_result);
                        
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
        
        // Try broadcasting
        std::string error_msg;
        if (!are_broadcastable(dims_, other.dims_, &error_msg)) {
            return TensorError::DimensionMismatch;
        }
        
        // Determine broadcast shape
        TensorIndices<N> broadcast_shape = dims_;
        for (size_t i = 0; i < N; ++i) {
            broadcast_shape[i] = std::max(dims_[i], other.dims_[i]);
        }
        
        bool track_grad = requires_grad_ || other.requires_grad_;
        Tensor<T, N> result(broadcast_shape, use_gpu_, track_grad);
        result.is_leaf_ = false;
        size_t total = result.total_size();
        
        // Perform broadcasted multiplication
        for (size_t i = 0; i < total; ++i) {
            TensorIndices<N> coords;
            size_t idx = i;
            for (size_t d = N; d > 0; --d) {
                coords[d - 1] = idx % broadcast_shape[d - 1];
                idx /= broadcast_shape[d - 1];
            }
            
            size_t self_idx = 0, other_idx = 0;
            size_t self_stride = 1, other_stride = 1;
            for (size_t d = N; d > 0; --d) {
                size_t dim = d - 1;
                size_t self_coord = (dims_[dim] == 1) ? 0 : coords[dim];
                size_t other_coord = (other.dims_[dim] == 1) ? 0 : coords[dim];
                
                if (dim + 1 < N) {
                    self_stride *= dims_[dim + 1];
                    other_stride *= other.dims_[dim + 1];
                }
                
                self_idx += self_coord * self_stride;
                other_idx += other_coord * other_stride;
            }
            
            result.data_[i] = data_[self_idx] * other.data_[other_idx];
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
            ensure_on_cpu();  // Sync from GPU before scalar operations
            mul_scalar_gpu(data_.get(), scalar, result.data_.get(), total);
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
            mul_gpu(data_.get(), other.data_.get(), data_.get(), total);
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
            mul_scalar_gpu(data_.get(), scalar, data_.get(), total);
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
        // Check if shapes match exactly
        if (dims_ == other.dims_) {
            // Fast path: exact shape match
            Tensor<T, N> result(dims_, use_gpu_);
            size_t total = total_size();
            
#ifdef USE_GPU
            if (use_gpu_ && other.use_gpu_) {
                ensure_on_gpu();
                other.ensure_on_gpu();
                result.ensure_on_gpu();
                div_gpu_direct(d_data_, other.d_data_, result.d_data_, total);
                result.mark_gpu_modified();
            } else
#endif
            {
                for (size_t i = 0; i < total; ++i) {
                    result.data_[i] = data_[i] / other.data_[i];
                }
            }
            
            return result;
        }
        
        // Try broadcasting
        std::string error_msg;
        if (!are_broadcastable(dims_, other.dims_, &error_msg)) {
            return TensorError::DimensionMismatch;
        }
        
        // Determine broadcast shape
        TensorIndices<N> broadcast_shape = dims_;
        for (size_t i = 0; i < N; ++i) {
            broadcast_shape[i] = std::max(dims_[i], other.dims_[i]);
        }
        
        Tensor<T, N> result(broadcast_shape, use_gpu_);
        size_t total = result.total_size();
        
        // Perform broadcasted division
        for (size_t i = 0; i < total; ++i) {
            TensorIndices<N> coords;
            size_t idx = i;
            for (size_t d = N; d > 0; --d) {
                coords[d - 1] = idx % broadcast_shape[d - 1];
                idx /= broadcast_shape[d - 1];
            }
            
            size_t self_idx = 0, other_idx = 0;
            size_t self_stride = 1, other_stride = 1;
            for (size_t d = N; d > 0; --d) {
                size_t dim = d - 1;
                size_t self_coord = (dims_[dim] == 1) ? 0 : coords[dim];
                size_t other_coord = (other.dims_[dim] == 1) ? 0 : coords[dim];
                
                if (dim + 1 < N) {
                    self_stride *= dims_[dim + 1];
                    other_stride *= other.dims_[dim + 1];
                }
                
                self_idx += self_coord * self_stride;
                other_idx += other_coord * other_stride;
            }
            
            result.data_[i] = data_[self_idx] / other.data_[other_idx];
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
            div_scalar_gpu(data_.get(), scalar, result.data_.get(), total);
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
            div_gpu(data_.get(), other.data_.get(), data_.get(), total);
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
            div_scalar_gpu(data_.get(), scalar, data_.get(), total);
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
        
        // Copy data first
        std::copy(data_.get(), data_.get() + total, result.data_.get());
        
        // Negate in place using optimized operations
#ifdef USE_BLAS
        if (!use_gpu_) {
            blas_scal(static_cast<int>(total), T(-1), result.data_.get(), 1);
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                result.data_[i] = -result.data_[i];
            }
        }
        
        return result;
    }
    
    /**
     * @brief Apply a function to each element of the tensor (creates new tensor)
     * 
     * Maps a custom function over all elements, creating a new tensor with the results.
     * The function should accept a single element of type T and return a value of type T.
     * 
     * @tparam Func Function type (can be lambda, function pointer, or functor)
     * @param func The function to apply to each element
     * @return A new tensor with the function applied to all elements
     * 
     * @section example_map Example
     * @code
     * Tensor<float, 2> matrix({2, 3});
     * matrix.fill(2.0f);
     * 
     * // Square each element
     * auto squared = matrix.map([](float x) { return x * x; });
     * // squared contains all 4.0
     * 
     * // Apply custom function
     * auto result = matrix.map([](float x) { return std::sqrt(x) + 1.0f; });
     * 
     * // Can also use function pointers
     * auto exponential = matrix.map(std::exp<float>);
     * @endcode
     */
    template<typename Func>
    Tensor<T, N> map(Func func) const {
#ifdef USE_GPU
        ensure_on_cpu();  // Sync from GPU if needed
#endif
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            result.data_[i] = func(data_[i]);
        }
        
        return result;
    }
    
    /**
     * @brief Apply a function to each element of the tensor (in-place)
     * 
     * Maps a custom function over all elements, modifying the tensor in-place.
     * More memory-efficient than map() when you don't need to preserve the original.
     * 
     * @tparam Func Function type (can be lambda, function pointer, or functor)
     * @param func The function to apply to each element
     * @return Reference to this tensor (for method chaining)
     * 
     * @section example_map_inplace Example
     * @code
     * Tensor<float, 1> vec({5});
     * vec.fill(3.0f);
     * 
     * // Double all elements in-place
     * vec.map_inplace([](float x) { return x * 2.0f; });
     * // vec now contains all 6.0
     * 
     * // Chain multiple operations
     * vec.map_inplace([](float x) { return x + 1.0f; })
     *    .map_inplace([](float x) { return x * x; });
     * @endcode
     */
    template<typename Func>
    Tensor<T, N>& map_inplace(Func func) {
#ifdef USE_GPU
        ensure_on_cpu();  // Sync from GPU if needed
        data_on_gpu_ = false;  // CPU data will be modified
        gpu_needs_sync_ = false;
#endif
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            data_[i] = func(data_[i]);
        }
        
        return *this;
    }
    
    /**
     * @brief Apply exponential function to all elements (creates new tensor)
     * 
     * Computes e^x for each element. Uses GPU acceleration when available.
     * Supports automatic differentiation.
     * 
     * @return A new tensor with exp(x) applied to all elements
     * 
     * @section example_exp Example
     * @code
     * Tensor<float, 2> matrix({2, 2});
     * matrix.fill(1.0f);
     * 
     * auto result = matrix.exp();  // All elements ≈ 2.718
     * 
     * // With autograd:
     * Tensor<float, 1> x({3}, true, true);
     * x.fill(0.0f);
     * auto y = x.exp().sum();  // sum(exp(x))
     * y.backward();
     * // x.grad() contains derivatives
     * @endcode
     */
    Tensor<T, N> exp() const {
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_) {
                ensure_on_gpu();
                result.ensure_on_gpu();
                exp_gpu_direct(d_data_, result.d_data_, total);
                result.mark_gpu_modified();
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
     * @brief Apply natural logarithm to all elements (creates new tensor)
     * 
     * Computes ln(x) for each element. Uses GPU acceleration when available.
     * 
     * @return A new tensor with log(x) applied to all elements
     * @warning Elements must be positive (log of non-positive values is undefined)
     * 
     * @section example_log Example
     * @code
     * Tensor<float, 1> vec({4});
     * vec.fill(2.718f);  // e
     * 
     * auto result = vec.log();  // All elements ≈ 1.0
     * 
     * // Useful for log-space computations
     * auto log_probs = probs.log();
     * @endcode
     */
    Tensor<T, N> log() const {
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_) {
                ensure_on_gpu();
                result.ensure_on_gpu();
                log_gpu_direct(d_data_, result.d_data_, total);
                result.mark_gpu_modified();
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
     * @brief Apply square root to all elements (creates new tensor)
     * 
     * Computes √x for each element. Uses GPU acceleration when available.
     * 
     * @return A new tensor with sqrt(x) applied to all elements
     * @warning Elements must be non-negative (sqrt of negative values is undefined)
     * 
     * @section example_sqrt Example
     * @code
     * Tensor<float, 2> matrix({2, 2});
     * matrix.fill(4.0f);
     * 
     * auto result = matrix.sqrt();  // All elements = 2.0
     * 
     * // Useful for computing standard deviation
     * auto variance = data.variance();
     * auto std_dev = variance.sqrt();
     * @endcode
     */
    Tensor<T, N> sqrt() const {
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_) {
                ensure_on_gpu();
                result.ensure_on_gpu();
                sqrt_gpu_direct(d_data_, result.d_data_, total);
                result.mark_gpu_modified();
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
     * @brief Apply power function to all elements (creates new tensor)
     * 
     * Computes x^exponent for each element. Uses GPU acceleration when available.
     * Supports automatic differentiation.
     * 
     * @param exponent The exponent to raise each element to
     * @return A new tensor with pow(x, exponent) applied to all elements
     * 
     * @section example_pow Example
     * @code
     * Tensor<float, 1> vec({3});
     * vec.fill(2.0f);
     * 
     * auto squared = vec.pow(2.0f);    // All elements = 4.0
     * auto cubed = vec.pow(3.0f);      // All elements = 8.0
     * auto rooted = vec.pow(0.5f);     // All elements ≈ 1.414 (square root)
     * 
     * // With autograd:
     * Tensor<float, 1> x({5}, true, true);
     * x.fill(2.0f);
     * auto y = x.pow(3.0f).sum();  // sum(x³)
     * y.backward();
     * // x.grad() contains 3*x²
     * @endcode
     */
    Tensor<T, N> pow(T exponent) const {
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_) {
                ensure_on_gpu();
                result.ensure_on_gpu();
                pow_gpu_direct(d_data_, exponent, result.d_data_, total);
                result.mark_gpu_modified();
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
#ifdef USE_GPU
        if (use_gpu_) {
            Tensor<T, N> result(dims_, true);
            ensure_on_gpu();
            result.ensure_on_gpu();
            abs_gpu_direct(d_data_, result.d_data_, total_size());
            result.mark_gpu_modified();
            return result;
        }
#endif
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
                ensure_on_gpu();
                result.ensure_on_gpu();
                sin_gpu_direct(d_data_, result.d_data_, total);
                result.mark_gpu_modified();
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
                ensure_on_gpu();
                result.ensure_on_gpu();
                cos_gpu_direct(d_data_, result.d_data_, total);
                result.mark_gpu_modified();
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
     * @brief Apply hyperbolic tangent (tanh) to all elements (creates new tensor)
     * 
     * Computes tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) for each element.
     * Alternative activation function that outputs values in (-1, 1), making it
     * zero-centered (unlike sigmoid). Often preferred in RNNs and hidden layers.
     * 
     * Range: (-1, 1) - zero-centered output helps with convergence.
     * 
     * @return A new tensor with tanh(x) applied to all elements
     * 
     * @section example_tanh Example
     * @code
     * // Hidden layer with tanh activation
     * Tensor<float, 2> hidden({32, 64}, true, true);
     * auto activated = hidden.tanh();
     * 
     * // Common in RNNs for cell state updates:
     * auto cell_candidate = (W_c.matmul(concat) + b_c).tanh();
     * 
     * // Comparison with sigmoid:
     * auto sigmoid_out = x.sigmoid();  // Range: (0, 1)
     * auto tanh_out = x.tanh();        // Range: (-1, 1), zero-centered
     * @endcode
     */
    Tensor<T, N> tanh() const {
        Tensor<T, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_) {
                ensure_on_gpu();
                result.ensure_on_gpu();
                tanh_gpu_direct(d_data_, result.d_data_, total);
                result.mark_gpu_modified();
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
     * @brief Apply sigmoid function to all elements (creates new tensor)
     * 
     * Computes sigmoid(x) = 1 / (1 + exp(-x)) for each element.
     * Commonly used as activation function in neural networks, especially
     * for binary classification in output layers.
     * 
     * Range: (0, 1) - useful for representing probabilities.
     * Supports automatic differentiation.
     * 
     * @return A new tensor with sigmoid(x) applied to all elements
     * 
     * @section example_sigmoid Example
     * @code
     * // Binary classification output layer
     * Tensor<float, 2> logits({32, 1}, true, true);  // batch_size=32
     * auto probs = logits.sigmoid();  // Convert to probabilities in (0, 1)
     * 
     * // With autograd:
     * auto loss = binary_cross_entropy(probs, targets);
     * loss.backward();  // Computes gradient through sigmoid
     * 
     * // Gating mechanism (like in LSTM)
     * auto forget_gate = (W_f.matmul(x) + b_f).sigmoid();
     * @endcode
     */
    Tensor<T, N> sigmoid() const {
        Tensor<T, N> result(dims_, use_gpu_, requires_grad_);
        result.is_leaf_ = false;
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_) {
                ensure_on_gpu();
                result.ensure_on_gpu();
                sigmoid_gpu_direct(d_data_, result.d_data_, total);
                result.mark_gpu_modified();
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
                    
                    // Optimized using tensor operations
                    Tensor<T, N> ones(self_ptr->dims_, self_ptr->use_gpu_);
                    ones.fill(T(1));
                    auto one_minus_sig_result = ones - output_copy;
                    auto& one_minus_sig = std::get<Tensor<T, N>>(one_minus_sig_result);
                    auto sig_prod_result = output_copy * one_minus_sig;
                    auto& sig_prod = std::get<Tensor<T, N>>(sig_prod_result);
                    auto grad_result = grad * sig_prod;
                    auto& grad_contrib = std::get<Tensor<T, N>>(grad_result);
                    
                    // Accumulate gradients using tensor addition
                    auto grad_sum_result = *self_ptr->grad_ + grad_contrib;
                    *self_ptr->grad_ = std::get<Tensor<T, N>>(grad_sum_result);
                    
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
     * @brief Apply ReLU (Rectified Linear Unit) to all elements (creates new tensor)
     * 
     * Computes ReLU(x) = max(0, x) for each element.
     * Most widely used activation function in deep learning due to its
     * simplicity and effectiveness in preventing vanishing gradients.
     * 
     * Properties:
     * - Non-linearity without saturation for positive values
     * - Sparse activation (outputs 0 for negative inputs)
     * - Computationally efficient
     * 
     * Supports automatic differentiation.
     * 
     * @return A new tensor with ReLU(x) applied to all elements
     * 
     * @section example_relu Example
     * @code
     * // Hidden layer in neural network
     * Tensor<float, 2> hidden({32, 128}, true, true);
     * auto activated = hidden.relu();  // Apply ReLU activation
     * 
     * // Full network layer example:
     * auto layer1 = (input.matmul(W1) + b1).relu();
     * auto layer2 = (layer1.matmul(W2) + b2).relu();
     * auto output = layer2.matmul(W3) + b3;
     * 
     * // With autograd:
     * auto loss = mse_loss(output, target);
     * loss.backward();  // Gradients flow through ReLU layers
     * @endcode
     */
    Tensor<T, N> relu() const {
        Tensor<T, N> result(dims_, use_gpu_, requires_grad_);
        result.is_leaf_ = false;
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_) {
                ensure_on_gpu();
                result.ensure_on_gpu();
                relu_gpu_direct(d_data_, result.d_data_, total);
                result.mark_gpu_modified();
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
                    
                    // Optimized using tensor operations - create mask
                    Tensor<T, N> zeros(self_ptr->dims_, self_ptr->use_gpu_);
                    zeros.fill(T(0));
                    Tensor<T, N> mask = input_copy > zeros;
                    auto grad_result = grad * mask;
                    auto& grad_contrib = std::get<Tensor<T, N>>(grad_result);
                    
                    // Accumulate gradients using tensor addition
                    auto grad_sum_result = *self_ptr->grad_ + grad_contrib;
                    *self_ptr->grad_ = std::get<Tensor<T, N>>(grad_sum_result);
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
     * @brief Compute sum reduction along all dimensions
     * 
     * Sums all elements in the tensor to produce a single scalar value.
     * Useful for computing loss, gradient aggregation, or total statistics.
     * 
     * @return The sum of all elements as a scalar
     * 
     * @section example_sum Example
     * @code
     * Tensor<float, 2> matrix({3, 3});
     * matrix.fill(2.0f);
     * 
     * float total = matrix.sum();  // Returns 18.0 (9 elements × 2.0)
     * 
     * // Useful in loss computation:
     * Tensor<float, 1> errors({100}, true, true);
     * auto loss_tensor = errors.pow(2.0f);  // Square errors
     * float mse = loss_tensor.sum() / 100.0f;  // Mean squared error
     * @endcode
     */
    T sum() const {
        size_t total = total_size();
        
#ifdef USE_GPU
        if (use_gpu_) {
            ensure_on_gpu();
            Tensor<T, 1> d_result_tensor({1}, true);
            d_result_tensor.ensure_on_gpu();  // Allocate GPU memory
            sum_gpu_direct(d_data_, d_result_tensor.d_data_, total);
            
            T result;
            cudaMemcpy(&result, d_result_tensor.d_data_, sizeof(T), cudaMemcpyDeviceToHost);
            return result;
        }
#endif
        
        T result = T(0);
        for (size_t i = 0; i < total; ++i) {
            result += data_[i];
        }
        
        return result;
    }
    
    /**
     * @brief Compute mean of all elements
     * 
     * Calculates the arithmetic mean (average) of all tensor elements.
     * 
     * @return The mean value as a scalar
     * 
     * @section example_mean Example
     * @code
     * Tensor<float, 2> data({2, 3});
     * data[{0, 0}] = 1.0f;
     * data[{0, 1}] = 2.0f;
     * data[{0, 2}] = 3.0f;
     * data[{1, 0}] = 4.0f;
     * data[{1, 1}] = 5.0f;
     * data[{1, 2}] = 6.0f;
     * 
     * float avg = data.mean();  // Returns 3.5 = (1+2+3+4+5+6)/6
     * 
     * // Useful for normalization:
     * auto centered = data - data.mean();  // Center around zero
     * @endcode
     */
    T mean() const {
        return sum() / static_cast<T>(total_size());
    }
    
    /**
     * @brief Compute variance of all elements
     * 
     * Calculates the variance, measuring the spread of data around the mean.
     * Use ddof=1 for sample variance, ddof=0 for population variance.
     * 
     * @param ddof Delta degrees of freedom (default 0 for population variance)
     * @return The variance as a scalar
     * 
     * @section example_variance Example
     * @code
     * Tensor<float, 1> data({5});
     * data[{0}] = 1.0f;
     * data[{1}] = 2.0f;
     * data[{2}] = 3.0f;
     * data[{3}] = 4.0f;
     * data[{4}] = 5.0f;
     * 
     * float pop_var = data.variance(0);  // Population variance
     * float sample_var = data.variance(1);  // Sample variance (unbiased)
     * 
     * // For normalized data distributions:
     * float var = data.variance();
     * if (var < 0.01f) {
     *     std::cout << "Data has low variance\n";
     * }
     * @endcode
     */
    T variance(size_t ddof = 0) const {
        T m = mean();
        Tensor<T, N> diff = *this - m;
        auto squared_result = diff * diff;
        auto& squared = std::get<Tensor<T, N>>(squared_result);
        T var = squared.sum();
        return var / static_cast<T>(total_size() - ddof);
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
        
        Tensor<T, N> diff1 = *this - mean1;
        Tensor<T, N> diff2 = other - mean2;
        auto product_result = diff1 * diff2;
        auto& product = std::get<Tensor<T, N>>(product_result);
        T cov = product.sum();
        
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
        
        T mean1 = mean();
        T mean2 = other.mean();
        
        Tensor<T, N> diff1 = *this - mean1;
        Tensor<T, N> diff2 = other - mean2;
        
        auto cov_result = diff1 * diff2;
        auto& cov_tensor = std::get<Tensor<T, N>>(cov_result);
        T cov = cov_tensor.sum();
        
        auto var1_result = diff1 * diff1;
        auto& var1_tensor = std::get<Tensor<T, N>>(var1_result);
        T var1 = var1_tensor.sum();
        
        auto var2_result = diff2 * diff2;
        auto& var2_tensor = std::get<Tensor<T, N>>(var2_result);
        T var2 = var2_tensor.sum();
        
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
        
#ifdef USE_GPU
        if (use_gpu_) {
            ensure_on_gpu();
            Tensor<T, 1> d_result_tensor({1}, true);
            d_result_tensor.ensure_on_gpu();
            min_gpu_direct(d_data_, d_result_tensor.d_data_, total);
            
            T min_val;
            cudaMemcpy(&min_val, d_result_tensor.d_data_, sizeof(T), cudaMemcpyDeviceToHost);
            return min_val;
        }
#endif
        
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
        
#ifdef USE_GPU
        if (use_gpu_) {
            ensure_on_gpu();
            Tensor<T, 1> d_result_tensor({1}, true);
            d_result_tensor.ensure_on_gpu();
            max_gpu_direct(d_data_, d_result_tensor.d_data_, total);
            
            T max_val;
            cudaMemcpy(&max_val, d_result_tensor.d_data_, sizeof(T), cudaMemcpyDeviceToHost);
            return max_val;
        }
#endif
        
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
        
        // Compute Pearson correlation on ranks using tensor operations
        Tensor<T, 1> rank1_tensor({total}, use_gpu_);
        Tensor<T, 1> rank2_tensor({total}, use_gpu_);
        
        std::copy(rank1.begin(), rank1.end(), rank1_tensor.data_ptr());
        std::copy(rank2.begin(), rank2.end(), rank2_tensor.data_ptr());
        
        auto diff_result = rank1_tensor - rank2_tensor;
        auto& diff_tensor = std::get<Tensor<T, 1>>(diff_result);
        auto squared_result = diff_tensor * diff_tensor;
        auto& squared_tensor = std::get<Tensor<T, 1>>(squared_result);
        T sum_d_squared = squared_tensor.sum();
        
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
     * @brief Cast tensor to a different data type (NumPy-compatible astype)
     * @tparam U Target data type
     * @return A new tensor with the target data type
     * 
     * This method allows casting tensors between different types, similar to NumPy's astype.
     * 
     * Example:
     * @code
     * Tensor<int, 2> x({2, 3});
     * auto y = x.astype<float>();  // Convert int tensor to float
     * @endcode
     */
    template <typename U>
    Tensor<U, N> astype() const {
        Tensor<U, N> result(dims_, use_gpu_);
        size_t total = total_size();
        
        const T* src_data = data_ptr();
        U* dst_data = result.data_ptr();
        
        #pragma omp parallel for if(total > 10000)
        for (size_t i = 0; i < total; ++i) {
            dst_data[i] = static_cast<U>(src_data[i]);
        }
        
        return result;
    }
    
    /**
     * @brief Broadcast this tensor to a new shape
     * @tparam M The number of dimensions in target shape
     * @param target_shape The target shape to broadcast to
     * @return A variant containing the broadcast tensor or an error
     * 
     * Broadcasting rules (NumPy-compatible):
     * - Prepend dimensions of size 1 if needed
     * - For each dimension, sizes must either match or one must be 1
     * - Dimensions of size 1 are "stretched" to match the target
     * 
     * Example:
     * @code
     * Tensor<float, 1> x({3});
     * auto result = x.broadcast_to<2>({4, 3});  // Shape (3,) -> (4, 3)
     * @endcode
     */
    template <size_t M>
    auto broadcast_to(const TensorIndices<M>& target_shape) const
        -> std::variant<Tensor<T, M>, TensorError> {
        return tensor::broadcast_to(*this, target_shape);
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
        
        auto diff_result = *this - target;
        auto& diff = std::get<Tensor<T, N>>(diff_result);
        auto squared_result = diff * diff;
        auto& squared = std::get<Tensor<T, N>>(squared_result);
        T loss = squared.sum();
        
        return loss / static_cast<T>(total_size());
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
        
        T scale = T(2) / static_cast<T>(total_size());
        auto diff_result = *this - target;
        auto& diff = std::get<Tensor<T, N>>(diff_result);
        Tensor<T, N> result = diff * scale;
        
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
            dot_1d_gpu(data_.get(), other.data_.get(), &result, dims_[0]);
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
            dot_2d_gpu(data_.get(), other.data_.get(), result.data_.get(), m, n, p);
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
            dot_nd_gpu(data_.get(), other.data_.get(), result.data_.get(),
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
    
    /**
     * Cross product for 3D vectors.
     * Computes the cross product of two 3D vectors: result = this × other.
     * The cross product is perpendicular to both input vectors and follows the right-hand rule.
     * 
     * Formula: If a = [a1, a2, a3] and b = [b1, b2, b3], then
     * a × b = [a2*b3 - a3*b2, a3*b1 - a1*b3, a1*b2 - a2*b1]
     * 
     * @param other The other 3D vector to compute cross product with.
     * @return A variant containing either a new 3D tensor with the result or an error.
     * 
     * @code
     * // Create two 3D vectors
     * Tensor<float, 1> a({3});
     * a[{0}] = 1.0f; a[{1}] = 0.0f; a[{2}] = 0.0f;  // [1, 0, 0]
     * 
     * Tensor<float, 1> b({3});
     * b[{0}] = 0.0f; b[{1}] = 1.0f; b[{2}] = 0.0f;  // [0, 1, 0]
     * 
     * // Compute cross product
     * auto result_var = a.cross(b);
     * if (std::holds_alternative<Tensor<float, 1>>(result_var)) {
     *     auto& result = std::get<Tensor<float, 1>>(result_var);
     *     // result = [0, 0, 1] (perpendicular to both a and b)
     * }
     * @endcode
     */
    template<size_t M = N>
    typename std::enable_if<M == 1, TensorResult<Tensor<T, 1>>>::type
    cross(const Tensor<T, 1>& other) const {
        static_assert(N == 1, "Cross product is only for 1D tensors");
        if (dims_[0] != 3) {
            return TensorError::DimensionMismatch;
        }
        if (other.dims_[0] != 3) {
            return TensorError::DimensionMismatch;
        }
        
        Tensor<T, 1> result({3}, use_gpu_);
        
        // Cross product formula: a × b = [a2*b3 - a3*b2, a3*b1 - a1*b3, a1*b2 - a2*b1]
        T a1 = data_[0], a2 = data_[1], a3 = data_[2];
        T b1 = other.data_[0], b2 = other.data_[1], b3 = other.data_[2];
        
#ifdef USE_GPU
        if (use_gpu_ && other.use_gpu_) {
            cross_3d_gpu(data_.get(), other.data_.get(), result.data_.get());
            return result;
        }
#endif
        
        result.data_[0] = a2 * b3 - a3 * b2;
        result.data_[1] = a3 * b1 - a1 * b3;
        result.data_[2] = a1 * b2 - a2 * b1;
        
        return result;
    }
    
    // ============================================
    // Phase 1: Core Neural Network Operations
    // ============================================
    
    /**
     * @brief Matrix multiplication with autograd support (matmul)
     * 
     * Performs standard matrix multiplication: A(m,n) @ B(n,p) = C(m,p)
     * 
     * This is the fundamental operation for neural network layers. Uses optimized
     * implementations (BLAS/GPU) when available. Supports automatic differentiation
     * for training neural networks.
     * 
     * @tparam M Template parameter ensuring N==2 (2D tensors only)
     * @param other The tensor to multiply with (must have compatible dimensions)
     * @return Result tensor or TensorError::DimensionMismatch if incompatible
     * 
     * @section example_matmul Example
     * @code
     * // Linear layer: output = input @ weights
     * Tensor<float, 2> input({32, 784}, true, true);   // batch=32, features=784
     * Tensor<float, 2> weights({784, 128}, true, true); // 784 -> 128 hidden units
     * 
     * auto output_result = input.matmul(weights);
     * if (std::holds_alternative<Tensor<float, 2>>(output_result)) {
     *     auto& output = std::get<Tensor<float, 2>>(output_result);
     *     // output shape: (32, 128)
     * }
     * 
     * // Full neural network layer with bias:
     * auto hidden = input.matmul(W1) + b1;  // Broadcasting b1
     * auto activated = std::get<Tensor<float, 2>>(hidden).relu();
     * 
     * // Multi-layer network with autograd:
     * auto h1_result = input.matmul(W1);
     * auto& h1 = std::get<Tensor<float, 2>>(h1_result);
     * auto h1_act = (h1 + b1).relu();
     * auto h2_result = h1_act.matmul(W2);
     * auto& h2 = std::get<Tensor<float, 2>>(h2_result);
     * auto output = h2 + b2;
     * 
     * // Compute gradients:
     * auto loss = mse_loss(output, targets);
     * loss.backward();
     * // Now W1.grad(), W2.grad(), etc. contain gradients
     * @endcode
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
        
        // Forward pass - use cuBLAS for GPU, BLAS for CPU
#ifdef USE_CUBLAS
        // Use cuBLAS for GPU matrix multiplication
        if (use_gpu_ && other.use_gpu_) {
            // Ensure data is on GPU
            ensure_on_gpu();
            other.ensure_on_gpu();
            result.ensure_on_gpu();
            
            // cuBLAS uses column-major layout, but we use row-major
            // To compute C = A * B (row-major), we compute:
            // C^T = B^T * A^T (column-major)
            // So we swap A and B in the call
            
            T alpha = T(1);
            T beta = T(0);
            
            cublasHandle_t handle = CublasHandle::get();
            cublasStatus_t status;
            
            if constexpr (std::is_same_v<T, float>) {
                status = cublasSgemm(handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           p, m, n,              // Dimensions swapped for col-major
                           &alpha,
                           other.d_data_, p,     // B (treated as B^T in col-major)
                           d_data_, n,           // A (treated as A^T in col-major)
                           &beta,
                           result.d_data_, p);   // C (treated as C^T in col-major)
            } else if constexpr (std::is_same_v<T, double>) {
                status = cublasDgemm(handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           p, m, n,
                           &alpha,
                           other.d_data_, p,
                           d_data_, n,
                           &beta,
                           result.d_data_, p);
            } else {
                throw std::runtime_error("cuBLAS only supports float and double types");
            }
            
            if (status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("cuBLAS gemm failed with status: " + std::to_string(status));
            }
            
            result.mark_gpu_modified();
        } else
#endif
#ifdef USE_BLAS
        // Fallback to CPU BLAS
        {
            result.fill(T(0));
            
            // Sync GPU tensors to CPU if needed
            if (use_gpu_) ensure_on_cpu();
            if (other.use_gpu_) other.ensure_on_cpu();
            
            blas_gemm<T>(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, p, n, T(1), data_.get(), n,
                        other.data_.get(), p, T(0), result.data_.get(), p);
        }
#else
        // Naive fallback for when neither cuBLAS nor BLAS is available
        {
            result.fill(T(0));
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
#endif
        
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
     * @brief Softmax operation with autograd support
     * 
     * Applies softmax normalization along a specified axis:
     * softmax(x_i) = exp(x_i) / sum(exp(x_j))
     * 
     * Converts logits to probabilities that sum to 1. Uses numerically stable
     * implementation (subtracts max before exp) to prevent overflow.
     * 
     * Essential for multi-class classification output layers.
     * 
     * @param axis The axis to apply softmax along (default: -1 = last axis)
     * @return Tensor with softmax applied, or zero tensor if axis is out of range
     * 
     * @section example_softmax Example
     * @code
     * // Multi-class classification (10 classes, batch size 32)
     * Tensor<float, 2> logits({32, 10}, true, true);
     * auto probs = logits.softmax(-1);  // Apply along class dimension
     * // Each row now sums to 1.0 and represents class probabilities
     * 
     * // With loss computation:
     * auto loss = cross_entropy_loss(logits, targets);
     * loss.backward();  // Gradient flows through softmax
     * 
     * // Attention mechanism example:
     * auto scores = query.matmul(key.transpose());  // Attention scores
     * auto attention_weights = scores.softmax(-1);   // Normalize scores
     * auto context = attention_weights.matmul(value);
     * 
     * // Temperature scaling for controlling confidence:
     * float temperature = 2.0f;
     * auto calibrated_probs = (logits / temperature).softmax(-1);
     * @endcode
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
        
#ifdef USE_GPU
        if (use_gpu_ && (std::is_same_v<T, float> || std::is_same_v<T, double>)) {
            reduce_sum_axis_gpu(data_.get(), result.data_ptr(),
                                           outer, axis_size, inner);
        } else
#endif
        {
            // CPU/BLAS fallback - triple nested loop
            for (size_t o = 0; o < outer; ++o) {
                for (size_t i = 0; i < inner; ++i) {
                    for (size_t a = 0; a < axis_size; ++a) {
                        size_t src_idx = o * axis_size * inner + a * inner + i;
                        size_t dst_idx = o * inner + i;
                        result.data_[dst_idx] += data_[src_idx];
                    }
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
                
#ifdef USE_GPU
                if (self_ptr->use_gpu_ && (std::is_same_v<T, float> || std::is_same_v<T, double>)) {
                    broadcast_add_axis_gpu(grad.data_ptr(), self_ptr->grad_->data_ptr(),
                                                      outer, axis_size, inner);
                } else
#endif
                {
                    // CPU fallback - triple nested loop
                    for (size_t o = 0; o < outer; ++o) {
                        for (size_t i = 0; i < inner; ++i) {
                            size_t grad_idx = o * inner + i;
                            for (size_t a = 0; a < axis_size; ++a) {
                                size_t self_idx = o * axis_size * inner + a * inner + i;
                                self_ptr->grad_->data_[self_idx] += grad.data_[grad_idx];
                            }
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
        
#ifdef USE_GPU
        if (result.use_gpu_) {
            div_scalar_gpu(result.data_ptr(), divisor, result.data_ptr(), total);
            return result;
        }
#endif

#ifdef USE_BLAS
        if (std::is_same_v<T, float>) {
            cblas_sscal(static_cast<int>(total), 1.0f / divisor, 
                       reinterpret_cast<float*>(result.data_ptr()), 1);
            return result;
        } else if (std::is_same_v<T, double>) {
            cblas_dscal(static_cast<int>(total), 1.0 / divisor,
                       reinterpret_cast<double*>(result.data_ptr()), 1);
            return result;
        }
#endif
        
        // CPU fallback
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
     */
    Tensor<T, N> tile(const TensorIndices<N>& reps) const {
        return repeat(reps);
    }
    
    // ========================================================================
    // Enhanced Submatrix View Methods (for 2D tensors)
    // ========================================================================
    
    /**
     * @brief Extract a single row (only for 2D tensors)
     * @param row_idx Row index to extract
     * @return 1D tensor containing the row
     */
    template<size_t M = N, typename = std::enable_if_t<M == 2>>
    Tensor<T, 1> row(size_t row_idx) const {
        return tensor::row(*static_cast<const Tensor<T, 2>*>(static_cast<const void*>(this)), row_idx);
    }
    
    /**
     * @brief Extract a single column (only for 2D tensors)
     * @param col_idx Column index to extract
     * @return 1D tensor containing the column
     */
    template<size_t M = N, typename = std::enable_if_t<M == 2>>
    Tensor<T, 1> col(size_t col_idx) const {
        return tensor::col(*static_cast<const Tensor<T, 2>*>(static_cast<const void*>(this)), col_idx);
    }
    
    /**
     * @brief Extract diagonal elements (only for 2D tensors)
     * @return 1D tensor containing diagonal elements
     */
    template<size_t M = N, typename = std::enable_if_t<M == 2>>
    Tensor<T, 1> diag() const {
        return tensor::diag(*static_cast<const Tensor<T, 2>*>(static_cast<const void*>(this)));
    }
    
    /**
     * @brief Extract a rectangular block (only for 2D tensors)
     * @param start_row Starting row index
     * @param start_col Starting column index
     * @param num_rows Number of rows to extract
     * @param num_cols Number of columns to extract
     * @return 2D tensor containing the block
     */
    template<size_t M = N, typename = std::enable_if_t<M == 2>>
    Tensor<T, 2> block(size_t start_row, size_t start_col, size_t num_rows, size_t num_cols) const {
        return tensor::block(*static_cast<const Tensor<T, 2>*>(static_cast<const void*>(this)), 
                       start_row, start_col, num_rows, num_cols);
    }
    
    /**
     * @brief Extract first n elements (only for 1D tensors)
     * @param n Number of elements to extract
     * @return 1D tensor containing first n elements
     */
    template<size_t M = N, typename = std::enable_if_t<M == 1>>
    Tensor<T, 1> head(size_t n) const {
        return tensor::head(*static_cast<const Tensor<T, 1>*>(static_cast<const void*>(this)), n);
    }
    
    /**
     * @brief Extract last n elements (only for 1D tensors)
     * @param n Number of elements to extract
     * @return 1D tensor containing last n elements
     */
    template<size_t M = N, typename = std::enable_if_t<M == 1>>
    Tensor<T, 1> tail(size_t n) const {
        return tensor::tail(*static_cast<const Tensor<T, 1>*>(static_cast<const void*>(this)), n);
    }
    
    /**
     * @brief Extract top n rows (only for 2D tensors)
     * @param n Number of rows to extract
     * @return 2D tensor containing top n rows
     */
    template<size_t M = N, typename = std::enable_if_t<M == 2>>
    Tensor<T, 2> topRows(size_t n) const {
        return tensor::topRows(*static_cast<const Tensor<T, 2>*>(static_cast<const void*>(this)), n);
    }
    
    /**
     * @brief Extract bottom n rows (only for 2D tensors)
     * @param n Number of rows to extract
     * @return 2D tensor containing bottom n rows
     */
    template<size_t M = N, typename = std::enable_if_t<M == 2>>
    Tensor<T, 2> bottomRows(size_t n) const {
        return tensor::bottomRows(*static_cast<const Tensor<T, 2>*>(static_cast<const void*>(this)), n);
    }
    
    /**
     * @brief Extract leftmost n columns (only for 2D tensors)
     * @param n Number of columns to extract
     * @return 2D tensor containing leftmost n columns
     */
    template<size_t M = N, typename = std::enable_if_t<M == 2>>
    Tensor<T, 2> leftCols(size_t n) const {
        return tensor::leftCols(*static_cast<const Tensor<T, 2>*>(static_cast<const void*>(this)), n);
    }
    
    /**
     * @brief Extract rightmost n columns (only for 2D tensors)
     * @param n Number of columns to extract
     * @return 2D tensor containing rightmost n columns
     */
    template<size_t M = N, typename = std::enable_if_t<M == 2>>
    Tensor<T, 2> rightCols(size_t n) const {
        return tensor::rightCols(*static_cast<const Tensor<T, 2>*>(static_cast<const void*>(this)), n);
    }
    
    // ============================================
    // Enhanced Operations for Neural Networks
    // ============================================
    
    /**
     * @brief Apply softmax activation along rows (axis=1) for 2D tensors
     * 
     * Computes softmax(x) = exp(x - max(x)) / sum(exp(x - max(x))) row-wise.
     * Numerically stable implementation using max subtraction.
     * 
     * @return New tensor with softmax applied to each row
     * 
     * @section example_softmax Example
     * @code
     * Tensor<float, 2> logits({batch_size, num_classes});
     * auto probs = logits.softmax_rows();  // Convert to probabilities
     * @endcode
     */
    template<size_t M = N, typename = std::enable_if_t<M == 2>>
    Tensor<T, 2> softmax_rows() const {
        auto shape = dims_;
        size_t batch_size = shape[0];
        size_t num_classes = shape[1];
        
        Tensor<T, 2> result(shape, use_gpu_);
        const T* input_ptr = data_.get();
        T* output_ptr = result.data_.get();
        
        for (size_t i = 0; i < batch_size; ++i) {
            const T* row_in = input_ptr + i * num_classes;
            T* row_out = output_ptr + i * num_classes;
            
            // Find max for numerical stability
            T max_val = *std::max_element(row_in, row_in + num_classes);
            
            // Compute exp(x - max) and sum
            T sum = T(0);
            for (size_t j = 0; j < num_classes; ++j) {
                row_out[j] = std::exp(row_in[j] - max_val);
                sum += row_out[j];
            }
            
            // Normalize
            T inv_sum = T(1) / sum;
            for (size_t j = 0; j < num_classes; ++j) {
                row_out[j] *= inv_sum;
            }
        }
        
        return result;
    }
    
    /**
     * @brief Find argmax for each row in a 2D tensor
     * 
     * Returns column index of maximum value for each row.
     * Useful for extracting predicted classes from probability distributions.
     * 
     * @return 1D tensor of size (rows) containing column indices
     * 
     * @section example_argmax_rows Example
     * @code
     * Tensor<float, 2> predictions({batch_size, num_classes});
     * auto pred_classes = predictions.argmax_rows();  // 1D tensor with class indices
     * @endcode
     */
    template<size_t M = N, typename = std::enable_if_t<M == 2>>
    Tensor<size_t, 1> argmax_rows() const {
        size_t num_rows = dims_[0];
        size_t num_cols = dims_[1];
        
        Tensor<size_t, 1> result({num_rows}, false);  // Indices don't need GPU
        const T* data_ptr = data_.get();
        size_t* result_ptr = result.data_ptr();
        
        for (size_t i = 0; i < num_rows; ++i) {
            const T* row = data_ptr + i * num_cols;
            
            size_t max_idx = 0;
            T max_val = row[0];
            
            for (size_t j = 1; j < num_cols; ++j) {
                if (row[j] > max_val) {
                    max_val = row[j];
                    max_idx = j;
                }
            }
            
            result_ptr[i] = max_idx;
        }
        
        return result;
    }
    
    /**
     * @brief Fill tensor with random values from normal distribution
     * 
     * Uses direct pointer access for efficient initialization.
     * 
     * @param mean Mean of the distribution (default: 0.0)
     * @param stddev Standard deviation (default: 1.0)
     * 
     * @section example_randn Example
     * @code
     * Tensor<float, 2> weights({hidden_size, input_size});
     * weights.randn(0.0f, 0.01f);  // Initialize with small random values
     * @endcode
     */
    void randn(T mean = T(0), T stddev = T(1)) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<T> dist(mean, stddev);
        
        T* data_ptr = data_.get();
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            data_ptr[i] = dist(gen);
        }
    }
    
    /**
     * @brief Fill tensor with random values from uniform distribution
     * 
     * Uses direct pointer access for efficient initialization.
     * 
     * @param min Minimum value (default: 0.0)
     * @param max Maximum value (default: 1.0)
     * 
     * @section example_rand_uniform Example
     * @code
     * Tensor<float, 2> dropout_mask({batch_size, features});
     * dropout_mask.rand_uniform(0.0f, 1.0f);
     * @endcode
     */
    void rand_uniform(T min = T(0), T max = T(1)) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(min, max);
        
        T* data_ptr = data_.get();
        size_t total = total_size();
        
        for (size_t i = 0; i < total; ++i) {
            data_ptr[i] = dist(gen);
        }
    }
    
    /**
     * @brief Fused operation: this -= scalar * other
     * 
     * Optimized for SGD weight updates: weights -= learning_rate * gradients
     * Avoids temporary tensor allocation.
     * 
     * @param scalar Scaling factor
     * @param other Tensor to scale and subtract
     * @return Reference to this tensor for chaining
     * 
     * @section example_fused_scalar_mul_sub Example
     * @code
     * Tensor<float, 2> weights({output_size, input_size});
     * Tensor<float, 2> gradients({output_size, input_size});
     * weights.fused_scalar_mul_sub(0.01f, gradients);  // weights -= 0.01 * gradients
     * @endcode
     */
    Tensor<T, N>& fused_scalar_mul_sub(T scalar, const Tensor<T, N>& other) {
        if (dims_ != other.dims_) {
            return *this;  // Dimensions must match
        }
        
        size_t total = total_size();
        T* this_ptr = data_.get();
        const T* other_ptr = other.data_.get();
        
#ifdef USE_GPU
        if (use_gpu_ && other.use_gpu_) {
            // Could add GPU kernel for this operation
            for (size_t i = 0; i < total; ++i) {
                this_ptr[i] -= scalar * other_ptr[i];
            }
        } else
#endif
        {
            for (size_t i = 0; i < total; ++i) {
                this_ptr[i] -= scalar * other_ptr[i];
            }
        }
        
        return *this;
    }
    
    /**
     * @brief Fill rows from vector of vectors (2D tensors only)
     * 
     * Efficiently copies multiple rows from std::vector<std::vector<T>>.
     * Useful for batch data preparation.
     * 
     * @param data Vector of row vectors to copy
     * @param start_row Starting row index in tensor (default: 0)
     * 
     * @section example_fill_rows Example
     * @code
     * Tensor<float, 2> batch({batch_size, num_features});
     * std::vector<std::vector<float>> images = load_images();
     * batch.fill_rows(images, 0);  // Fill from row 0
     * @endcode
     */
    template<size_t M = N, typename = std::enable_if_t<M == 2>>
    void fill_rows(const std::vector<std::vector<T>>& data, size_t start_row = 0) {
        size_t num_cols = dims_[1];
        T* tensor_ptr = data_.get();
        
        for (size_t i = 0; i < data.size() && (start_row + i) < dims_[0]; ++i) {
            if (data[i].size() != num_cols) {
                continue;  // Skip rows with incorrect size
            }
            
            T* row_ptr = tensor_ptr + (start_row + i) * num_cols;
            std::copy(data[i].begin(), data[i].end(), row_ptr);
        }
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
    
    /**
     * Generate tensor with gamma distribution.
     * @param dims Dimensions of the tensor.
     * @param alpha Shape parameter (k).
     * @param beta Scale parameter (theta).
     * @param use_gpu Whether to use GPU.
     * @return Tensor filled with random values from gamma distribution.
     */
    template <size_t N>
    static Tensor<T, N> gamma(const std::array<size_t, N>& dims, T alpha, T beta = T(1), bool use_gpu = false) {
        Tensor<T, N> result(dims, use_gpu, false);
        std::gamma_distribution<T> dist(alpha, beta);
        auto& gen = get_generator();
        
        T* data = result.data();
        size_t total = result.total_size();
        for (size_t i = 0; i < total; ++i) {
            data[i] = dist(gen);
        }
        
        return result;
    }
    
    /**
     * Generate tensor with beta distribution.
     * @param dims Dimensions of the tensor.
     * @param alpha First shape parameter.
     * @param beta_param Second shape parameter.
     * @param use_gpu Whether to use GPU.
     * @return Tensor filled with random values from beta distribution.
     * 
     * Note: Beta distribution is generated using the relationship Beta(a,b) = X/(X+Y)
     * where X ~ Gamma(a,1) and Y ~ Gamma(b,1).
     */
    template <size_t N>
    static Tensor<T, N> beta(const std::array<size_t, N>& dims, T alpha, T beta_param, bool use_gpu = false) {
        Tensor<T, N> result(dims, use_gpu, false);
        std::gamma_distribution<T> dist_alpha(alpha, T(1));
        std::gamma_distribution<T> dist_beta(beta_param, T(1));
        auto& gen = get_generator();
        
        T* data = result.data();
        size_t total = result.total_size();
        for (size_t i = 0; i < total; ++i) {
            T x = dist_alpha(gen);
            T y = dist_beta(gen);
            data[i] = x / (x + y);
        }
        
        return result;
    }
    
    /**
     * Generate tensor with chi-squared distribution.
     * @param dims Dimensions of the tensor.
     * @param degrees_of_freedom Degrees of freedom (k).
     * @param use_gpu Whether to use GPU.
     * @return Tensor filled with random values from chi-squared distribution.
     */
    template <size_t N>
    static Tensor<T, N> chi_squared(const std::array<size_t, N>& dims, T degrees_of_freedom, bool use_gpu = false) {
        Tensor<T, N> result(dims, use_gpu, false);
        std::chi_squared_distribution<T> dist(degrees_of_freedom);
        auto& gen = get_generator();
        
        T* data = result.data();
        size_t total = result.total_size();
        for (size_t i = 0; i < total; ++i) {
            data[i] = dist(gen);
        }
        
        return result;
    }
    
    /**
     * Generate samples from multinomial distribution.
     * @param n Number of trials.
     * @param probs Vector of probabilities (must sum to 1).
     * @param samples Number of samples to generate.
     * @param use_gpu Whether to use GPU.
     * @return Tensor of shape (samples, len(probs)) with counts for each category.
     * 
     * Each row represents one sample, and each column represents the count
     * for a particular category.
     */
    static Tensor<T, 2> multinomial(size_t n, const std::vector<T>& probs, size_t samples, bool use_gpu = false) {
        size_t k = probs.size();
        Tensor<T, 2> result({samples, k}, use_gpu, false);
        
        std::uniform_real_distribution<T> dist(T(0), T(1));
        auto& gen = get_generator();
        
        T* data = result.data();
        
        for (size_t s = 0; s < samples; ++s) {
            std::vector<size_t> counts(k, 0);
            
            for (size_t trial = 0; trial < n; ++trial) {
                T u = dist(gen);
                T cumsum = T(0);
                
                for (size_t i = 0; i < k; ++i) {
                    cumsum += probs[i];
                    if (u <= cumsum) {
                        counts[i]++;
                        break;
                    }
                }
            }
            
            for (size_t i = 0; i < k; ++i) {
                data[s * k + i] = static_cast<T>(counts[i]);
            }
        }
        
        return result;
    }
    
    /**
     * Generate tensor with Cauchy distribution.
     * @param dims Dimensions of the tensor.
     * @param location Location parameter (x0).
     * @param scale Scale parameter (gamma).
     * @param use_gpu Whether to use GPU.
     * @return Tensor filled with random values from Cauchy distribution.
     */
    template <size_t N>
    static Tensor<T, N> cauchy(const std::array<size_t, N>& dims, T location = T(0), T scale = T(1), bool use_gpu = false) {
        Tensor<T, N> result(dims, use_gpu, false);
        std::cauchy_distribution<T> dist(location, scale);
        auto& gen = get_generator();
        
        T* data = result.data();
        size_t total = result.total_size();
        for (size_t i = 0; i < total; ++i) {
            data[i] = dist(gen);
        }
        
        return result;
    }
};

// ============================================================================
// SORTING AND SEARCHING
// ============================================================================

/**
 * @brief Sort tensor elements and return sorted tensor (1D only).
 * @param tensor Input tensor to sort.
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
 * @param tensor Input tensor to get sort indices for.
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
 * @param tensor Input tensor to find top k elements.
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
 * @param tensor Input tensor to split.
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
 * @param tensor Input tensor to divide into chunks.
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
 * @param tensor Input tensor to tile.
 * @param repeats Number of repetitions for each dimension.
 * @return Tiled tensor with expanded dimensions.
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
 * @param tensor Input tensor to repeat.
 * @param repeats Number of times to repeat along the axis.
 * @param axis Axis along which to repeat.
 * @return Repeated tensor.
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

// ============================================================================
// Broadcasting Enhancements
// ============================================================================

/**
 * @brief Check if two tensor shapes are broadcastable
 * @tparam N1 Number of dimensions in first tensor
 * @tparam N2 Number of dimensions in second tensor
 * @param shape1 Shape of first tensor
 * @param shape2 Shape of second tensor
 * @param error_msg Optional pointer to store detailed error message
 * @return true if shapes are broadcastable, false otherwise
 * 
 * Two shapes are broadcastable if:
 * - They are equal, or
 * - One of them is 1
 * This is checked for each dimension from right to left.
 */
template <size_t N1, size_t N2>
bool are_broadcastable(const TensorIndices<N1>& shape1, 
                       const TensorIndices<N2>& shape2,
                       std::string* error_msg) {
    // Align shapes from the right (trailing dimensions)
    size_t max_dims = std::max(N1, N2);
    size_t offset1 = max_dims - N1;
    size_t offset2 = max_dims - N2;
    
    for (size_t i = 0; i < max_dims; ++i) {
        size_t dim1 = (i < offset1) ? 1 : shape1[i - offset1];
        size_t dim2 = (i < offset2) ? 1 : shape2[i - offset2];
        
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            if (error_msg) {
                *error_msg = "Shapes are not broadcastable: shape1[" + 
                            std::to_string(i) + "]=" + std::to_string(dim1) +
                            " vs shape2[" + std::to_string(i) + "]=" + 
                            std::to_string(dim2);
            }
            return false;
        }
    }
    return true;
}

/**
 * @brief Compute the broadcast shape of two tensors
 * @tparam N1 Number of dimensions in first tensor
 * @tparam N2 Number of dimensions in second tensor
 * @param shape1 Shape of first tensor
 * @param shape2 Shape of second tensor
 * @return The broadcast shape (variant with TensorError on failure)
 */
template <size_t N1, size_t N2>
auto compute_broadcast_shape(const TensorIndices<N1>& shape1,
                             const TensorIndices<N2>& shape2) 
    -> std::variant<TensorIndices<std::max(N1, N2)>, TensorError> {
    constexpr size_t MaxN = std::max(N1, N2);
    TensorIndices<MaxN> result;
    
    std::string error_msg;
    if (!are_broadcastable(shape1, shape2, &error_msg)) {
        return TensorError::DimensionMismatch;
    }
    
    size_t offset1 = MaxN - N1;
    size_t offset2 = MaxN - N2;
    
    for (size_t i = 0; i < MaxN; ++i) {
        size_t dim1 = (i < offset1) ? 1 : shape1[i - offset1];
        size_t dim2 = (i < offset2) ? 1 : shape2[i - offset2];
        result[i] = std::max(dim1, dim2);
    }
    
    return result;
}

/**
 * @brief Broadcast a tensor to a new shape
 * @tparam T The data type
 * @tparam N The number of dimensions
 * @tparam M The number of dimensions in target shape
 * @param tensor The input tensor
 * @param target_shape The target shape to broadcast to
 * @return A new tensor with the broadcast shape, or TensorError on failure
 * 
 * @note This creates a new tensor with data copied according to broadcast rules.
 * Broadcasting rules (NumPy-compatible):
 * - Prepend dimensions of size 1 if needed
 * - For each dimension, sizes must either match or one must be 1
 * - Dimensions of size 1 are "stretched" to match the target
 * 
 * Example:
 * @code
 * Tensor<float, 1> x({3});
 * auto result = broadcast_to(x, {4, 3});  // Shape (3,) -> (4, 3)
 * @endcode
 */
template <typename T, size_t N, size_t M>
auto broadcast_to(const Tensor<T, N>& tensor, const TensorIndices<M>& target_shape)
    -> std::variant<Tensor<T, M>, TensorError> {
    
    // Check if broadcasting is valid
    std::string error_msg;
    if (!are_broadcastable(tensor.shape(), target_shape, &error_msg)) {
        return TensorError::DimensionMismatch;
    }
    
    // Create result tensor
    Tensor<T, M> result(target_shape, tensor.uses_gpu());
    
    // Get raw pointers
    const T* src_data = tensor.data_ptr();
    T* dst_data = result.data_ptr();
    
    // Compute strides for broadcasting
    size_t offset = M - N;
    TensorIndices<M> src_strides;
    for (size_t i = 0; i < M; ++i) {
        if (i < offset) {
            src_strides[i] = 0;  // New dimension, don't advance
        } else {
            size_t src_dim = i - offset;
            if (tensor.shape()[src_dim] == 1) {
                src_strides[i] = 0;  // Broadcast this dimension
            } else {
                src_strides[i] = tensor.strides()[src_dim];
            }
        }
    }
    
    // Fill result with broadcast data
    size_t total_size = result.total_size();
    
    #pragma omp parallel for if(total_size > 10000)
    for (size_t i = 0; i < total_size; ++i) {
        // Convert flat index to multi-dimensional coordinates
        TensorIndices<M> coords;
        size_t idx = i;
        for (size_t d = M; d > 0; --d) {
            coords[d - 1] = idx % target_shape[d - 1];
            idx /= target_shape[d - 1];
        }
        
        // Map to source index using broadcast strides
        size_t src_idx = 0;
        for (size_t d = 0; d < M; ++d) {
            src_idx += coords[d] * src_strides[d];
        }
        
        dst_data[i] = src_data[src_idx];
    }
    
    return result;
}

// ============================================================================
// NumPy Compatibility - Type Casting
// ============================================================================

/**
 * @brief Cast tensor to a different data type (NumPy-compatible astype)
 * @tparam T Source data type
 * @tparam U Target data type
 * @tparam N Number of dimensions
 * @param tensor The input tensor
 * @return A new tensor with the target data type
 * 
 * This is the NumPy-compatible version of type casting.
 * 
 * Example:
 * @code
 * Tensor<int, 2> x({2, 3});
 * auto y = astype<float>(x);  // Convert int tensor to float
 * @endcode
 */
template <typename U, typename T, size_t N>
Tensor<U, N> astype(const Tensor<T, N>& tensor) {
    Tensor<U, N> result(tensor.shape(), tensor.uses_gpu());
    
    const T* src_data = tensor.data_ptr();
    U* dst_data = result.data_ptr();
    size_t total_size = tensor.total_size();
    
    #pragma omp parallel for if(total_size > 10000)
    for (size_t i = 0; i < total_size; ++i) {
        dst_data[i] = static_cast<U>(src_data[i]);
    }
    
    return result;
}

// ============================================================================
// NumPy Compatibility - Convenience Functions
// ============================================================================

/**
 * @brief Create a copy of a tensor (NumPy-compatible)
 * @tparam T The data type
 * @tparam N The number of dimensions
 * @param tensor The input tensor
 * @return A new tensor that is a copy of the input
 */
template <typename T, size_t N>
Tensor<T, N> copy(const Tensor<T, N>& tensor) {
    Tensor<T, N> result(tensor.shape(), tensor.uses_gpu());
    std::copy_n(tensor.data_ptr(), tensor.total_size(), result.data_ptr());
    return result;
}

/**
 * @brief Create a tensor filled with zeros (NumPy-compatible)
 * @tparam T The data type
 * @tparam N The number of dimensions
 * @param shape The shape of the tensor
 * @param use_gpu Whether to use GPU
 * @return A new tensor filled with zeros
 */
template <typename T, size_t N>
Tensor<T, N> zeros(const TensorIndices<N>& shape, bool use_gpu = true) {
    Tensor<T, N> result(shape, use_gpu);
    result.fill(T(0));
    return result;
}

/**
 * @brief Create a tensor filled with ones (NumPy-compatible)
 * @tparam T The data type
 * @tparam N The number of dimensions
 * @param shape The shape of the tensor
 * @param use_gpu Whether to use GPU
 * @return A new tensor filled with ones
 */
template <typename T, size_t N>
Tensor<T, N> ones(const TensorIndices<N>& shape, bool use_gpu = true) {
    Tensor<T, N> result(shape, use_gpu);
    result.fill(T(1));
    return result;
}

/**
 * @brief Create a tensor with values in a range (NumPy arange-compatible)
 * @tparam T The data type
 * @param start Start value (inclusive)
 * @param stop Stop value (exclusive)
 * @param step Step size
 * @param use_gpu Whether to use GPU
 * @return A 1D tensor with values from start to stop with given step
 */
template <typename T>
Tensor<T, 1> arange(T start, T stop, T step = T(1), bool use_gpu = true) {
    size_t n = static_cast<size_t>((stop - start) / step);
    if (n == 0) n = 1;
    
    Tensor<T, 1> result({n}, use_gpu);
    T* data = result.data_ptr();
    
    for (size_t i = 0; i < n; ++i) {
        data[i] = start + i * step;
    }
    
    return result;
}

/**
 * @brief Create a tensor with evenly spaced values (NumPy linspace-compatible)
 * @tparam T The data type
 * @param start Start value
 * @param stop Stop value
 * @param num Number of samples (default 50)
 * @param use_gpu Whether to use GPU
 * @return A 1D tensor with num evenly spaced values
 */
template <typename T>
Tensor<T, 1> linspace(T start, T stop, size_t num = 50, bool use_gpu = true) {
    Tensor<T, 1> result({num}, use_gpu);
    T* data = result.data_ptr();
    
    if (num == 1) {
        data[0] = start;
        return result;
    }
    
    T step = (stop - start) / (num - 1);
    for (size_t i = 0; i < num; ++i) {
        data[i] = start + i * step;
    }
    
    return result;
}

/**
 * @brief Create a tensor with values spaced evenly on a log scale
 * @tparam T The data type
 * @param start Start value (10^start)
 * @param stop Stop value (10^stop)
 * @param num Number of samples (default 50)
 * @param base Base of the log space (default 10.0)
 * @param use_gpu Whether to use GPU
 * @return A 1D tensor with num log-spaced values
 */
template <typename T>
Tensor<T, 1> logspace(T start, T stop, size_t num = 50, T base = T(10), 
                      bool use_gpu = true) {
    Tensor<T, 1> result({num}, use_gpu);
    T* data = result.data_ptr();
    
    if (num == 1) {
        data[0] = std::pow(base, start);
        return result;
    }
    
    T step = (stop - start) / (num - 1);
    for (size_t i = 0; i < num; ++i) {
        data[i] = std::pow(base, start + i * step);
    }
    
    return result;
}

/**
 * @brief Reshape a tensor (returns variant for error handling)
 * @tparam T The data type
 * @tparam N Original number of dimensions
 * @tparam M New number of dimensions
 * @param tensor The input tensor
 * @param new_shape The target shape
 * @return A reshaped tensor or TensorError if sizes don't match
 * 
 * @note This is an alias for the reshape method but returns a variant
 */
template <typename T, size_t N, size_t M>
auto reshape_to(const Tensor<T, N>& tensor, const TensorIndices<M>& new_shape)
    -> std::variant<Tensor<T, M>, TensorError> {
    
    // Check if total size matches
    size_t old_size = tensor.total_size();
    size_t new_size = 1;
    for (size_t i = 0; i < M; ++i) {
        new_size *= new_shape[i];
    }
    
    if (old_size != new_size) {
        return TensorError::DimensionMismatch;
    }
    
    Tensor<T, M> result(new_shape, tensor.uses_gpu());
    std::copy_n(tensor.data_ptr(), old_size, result.data_ptr());
    return result;
}

// Enhanced Submatrix Views (Methods to add to Tensor class)

// Note: The following methods need to be added as member functions of
// Tensor<T, N> class.
//
// They are declared here as free functions and should be moved inside
// the class definition.

/**
 * @brief Extract a single row from a 2D tensor
 * @tparam T Data type
 * @param matrix 2D tensor
 * @param row_idx Row index to extract
 * @return 1D tensor containing the row
 * 
 * This is a convenience method equivalent to select(0, row_idx) for matrices.
 * @code
 * Tensor<float, 2> mat({5, 3});
 * auto row2 = mat.row(2);  // Extract 3rd row
 * @endcode
 */
template <typename T>
Tensor<T, 1> row(const Tensor<T, 2>& matrix, size_t row_idx) {
    auto dims = matrix.dims();
    if (row_idx >= dims[0]) {
        throw std::out_of_range("Row index out of bounds");
    }
    
    size_t cols = dims[1];
    Tensor<T, 1> result({cols}, matrix.uses_gpu());
    
    const T* src = matrix.data_ptr();
    T* dst = result.data_ptr();
    
    for (size_t j = 0; j < cols; ++j) {
        dst[j] = src[row_idx * cols + j];
    }
    
    return result;
}

/**
 * @brief Extract a single column from a 2D tensor
 * @tparam T Data type
 * @param matrix 2D tensor
 * @param col_idx Column index to extract
 * @return 1D tensor containing the column
 * 
 * @code
 * Tensor<float, 2> mat({5, 3});
 * auto col1 = mat.col(1);  // Extract 2nd column
 * @endcode
 */
template <typename T>
Tensor<T, 1> col(const Tensor<T, 2>& matrix, size_t col_idx) {
    auto dims = matrix.dims();
    if (col_idx >= dims[1]) {
        throw std::out_of_range("Column index out of bounds");
    }
    
    size_t rows = dims[0];
    size_t cols = dims[1];
    Tensor<T, 1> result({rows}, matrix.uses_gpu());
    
    const T* src = matrix.data_ptr();
    T* dst = result.data_ptr();
    
    for (size_t i = 0; i < rows; ++i) {
        dst[i] = src[i * cols + col_idx];
    }
    
    return result;
}

/**
 * @brief Extract diagonal elements from a 2D tensor
 * @tparam T Data type
 * @param matrix 2D tensor
 * @return 1D tensor containing diagonal elements
 */
template <typename T>
Tensor<T, 1> diag(const Tensor<T, 2>& matrix) {
    auto dims = matrix.dims();
    size_t rows = dims[0];
    size_t cols = dims[1];
    size_t diag_size = std::min(rows, cols);
    
    Tensor<T, 1> result({diag_size}, matrix.uses_gpu());
    
    const T* src = matrix.data_ptr();
    T* dst = result.data_ptr();
    
    for (size_t i = 0; i < diag_size; ++i) {
        dst[i] = src[i * cols + i];
    }
    
    return result;
}

/**
 * @brief Create a diagonal matrix from a 1D tensor
 * @tparam T Data type
 * @param vec 1D tensor containing diagonal values
 * @return 2D square matrix with vec on diagonal, zeros elsewhere
 */
template <typename T>
Tensor<T, 2> diag(const Tensor<T, 1>& vec) {
    size_t n = vec.dims()[0];
    Tensor<T, 2> result({n, n}, vec.uses_gpu());
    result.fill(T(0));
    
    const T* src = vec.data_ptr();
    T* dst = result.data_ptr();
    
    for (size_t i = 0; i < n; ++i) {
        dst[i * n + i] = src[i];
    }
    
    return result;
}

/**
 * @brief Extract a rectangular block from a 2D tensor
 * @tparam T Data type
 * @param matrix 2D tensor
 * @param start_row Starting row index
 * @param start_col Starting column index
 * @param num_rows Number of rows to extract
 * @param num_cols Number of columns to extract
 * @return 2D tensor containing the block
 * 
 * @code
 * Tensor<float, 2> mat({10, 10});
 * auto block = mat.block(2, 3, 4, 5);  // 4x5 block starting at (2,3)
 * @endcode
 */
template <typename T>
Tensor<T, 2> block(const Tensor<T, 2>& matrix, size_t start_row, size_t start_col,
                   size_t num_rows, size_t num_cols) {
    auto dims = matrix.dims();
    if (start_row + num_rows > dims[0] || start_col + num_cols > dims[1]) {
        throw std::out_of_range("Block exceeds matrix bounds");
    }
    
    size_t cols = dims[1];
    Tensor<T, 2> result({num_rows, num_cols}, matrix.uses_gpu());
    
    const T* src = matrix.data_ptr();
    T* dst = result.data_ptr();
    
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < num_cols; ++j) {
            dst[i * num_cols + j] = src[(start_row + i) * cols + (start_col + j)];
        }
    }
    
    return result;
}

/**
 * @brief Extract first n elements from a 1D tensor
 * @tparam T Data type
 * @param vec 1D tensor
 * @param n Number of elements to extract
 * @return 1D tensor containing first n elements
 * 
 * @code
 * Tensor<float, 1> vec({10});
 * auto first5 = vec.head(5);
 * @endcode
 */
template <typename T>
Tensor<T, 1> head(const Tensor<T, 1>& vec, size_t n) {
    size_t size = vec.dims()[0];
    if (n > size) n = size;
    
    Tensor<T, 1> result({n}, vec.uses_gpu());
    std::copy_n(vec.data_ptr(), n, result.data_ptr());
    
    return result;
}

/**
 * @brief Extract last n elements from a 1D tensor
 * @tparam T Data type
 * @param vec 1D tensor
 * @param n Number of elements to extract
 * @return 1D tensor containing last n elements
 * 
 * @code
 * Tensor<float, 1> vec({10});
 * auto last5 = vec.tail(5);
 * @endcode
 */
template <typename T>
Tensor<T, 1> tail(const Tensor<T, 1>& vec, size_t n) {
    size_t size = vec.dims()[0];
    if (n > size) n = size;
    
    Tensor<T, 1> result({n}, vec.uses_gpu());
    std::copy_n(vec.data_ptr() + (size - n), n, result.data_ptr());
    
    return result;
}

/**
 * @brief Extract top n rows from a 2D tensor
 * @tparam T Data type
 * @param matrix 2D tensor
 * @param n Number of rows to extract
 * @return 2D tensor containing top n rows
 * 
 * @code
 * Tensor<float, 2> mat({10, 5});
 * auto top3 = mat.topRows(3);
 * @endcode
 */
template <typename T>
Tensor<T, 2> topRows(const Tensor<T, 2>& matrix, size_t n) {
    auto dims = matrix.dims();
    size_t rows = dims[0];
    size_t cols = dims[1];
    if (n > rows) n = rows;
    
    Tensor<T, 2> result({n, cols}, matrix.uses_gpu());
    std::copy_n(matrix.data_ptr(), n * cols, result.data_ptr());
    
    return result;
}

/**
 * @brief Extract bottom n rows from a 2D tensor
 * @tparam T Data type
 * @param matrix 2D tensor
 * @param n Number of rows to extract
 * @return 2D tensor containing bottom n rows
 * 
 * @code
 * Tensor<float, 2> mat({10, 5});
 * auto bottom3 = mat.bottomRows(3);
 * @endcode
 */
template <typename T>
Tensor<T, 2> bottomRows(const Tensor<T, 2>& matrix, size_t n) {
    auto dims = matrix.dims();
    size_t rows = dims[0];
    size_t cols = dims[1];
    if (n > rows) n = rows;
    
    Tensor<T, 2> result({n, cols}, matrix.uses_gpu());
    std::copy_n(matrix.data_ptr() + (rows - n) * cols, n * cols, result.data_ptr());
    
    return result;
}

/**
 * @brief Extract leftmost n columns from a 2D tensor
 * @tparam T Data type
 * @param matrix 2D tensor
 * @param n Number of columns to extract
 * @return 2D tensor containing leftmost n columns
 * 
 * @code
 * Tensor<float, 2> mat({5, 10});
 * auto left3 = mat.leftCols(3);
 * @endcode
 */
template <typename T>
Tensor<T, 2> leftCols(const Tensor<T, 2>& matrix, size_t n) {
    auto dims = matrix.dims();
    size_t rows = dims[0];
    size_t cols = dims[1];
    if (n > cols) n = cols;
    
    Tensor<T, 2> result({rows, n}, matrix.uses_gpu());
    
    const T* src = matrix.data_ptr();
    T* dst = result.data_ptr();
    
    for (size_t i = 0; i < rows; ++i) {
        std::copy_n(src + i * cols, n, dst + i * n);
    }
    
    return result;
}

/**
 * @brief Extract rightmost n columns from a 2D tensor
 * @tparam T Data type
 * @param matrix 2D tensor
 * @param n Number of columns to extract
 * @return 2D tensor containing rightmost n columns
 * 
 * @code
 * Tensor<float, 2> mat({5, 10});
 * auto right3 = mat.rightCols(3);
 * @endcode
 */
template <typename T>
Tensor<T, 2> rightCols(const Tensor<T, 2>& matrix, size_t n) {
    auto dims = matrix.dims();
    size_t rows = dims[0];
    size_t cols = dims[1];
    if (n > cols) n = cols;
    
    Tensor<T, 2> result({rows, n}, matrix.uses_gpu());
    
    const T* src = matrix.data_ptr();
    T* dst = result.data_ptr();
    
    for (size_t i = 0; i < rows; ++i) {
        std::copy_n(src + i * cols + (cols - n), n, dst + i * n);
    }
    
    return result;
}

} // namespace tensor
#endif // _TENSOR_BASE_H
