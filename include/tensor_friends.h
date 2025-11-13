#ifndef TENSOR_FRIENDS_H
#define TENSOR_FRIENDS_H

namespace tensor {

/// Forward declaration for autograd
template <typename T, size_t N>
class Tensor;

/**
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

} // namespace tensor

#endif // TENSOR_FRIENDS_H
