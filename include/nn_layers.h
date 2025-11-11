/**
 * @file nn_layers.h
 * @brief Common neural network layer implementations
 * 
 * This header provides implementations of common neural network layers
 * including linear (dense), convolutional, pooling, normalization, and
 * dropout layers.
 */

#ifndef NN_LAYERS_H
#define NN_LAYERS_H

#include "tensor.h"
#include "tensor_types.h"
#include <random>
#include <cmath>

// Only include CUDA headers if actually compiling with CUDA
#if defined(USE_GPU) && defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace tensor {

// Forward declarations
template<typename T>
class Layer;

template<typename T>
inline Tensor<T, 2> softmax_jacobian(const Tensor<T, 1>& softmax_output);

template<typename T>
inline std::vector<Tensor<T, 2>> softmax_jacobian_batch(const Tensor<T, 2>& softmax_output);

/**
 * @brief Base class for all neural network layers
 */
template<typename T>
class Layer {
public:
    virtual ~Layer() = default;
    
    /**
     * @brief Forward pass through the layer
     * @param input Input tensor
     * @return Output tensor
     */
    virtual Tensor<T, 2> forward(const Tensor<T, 2>& input) = 0;
    
    /**
     * @brief Backward pass (gradient computation)
     * @param grad_output Gradient from next layer
     * @return Gradient to pass to previous layer
     */
    virtual Tensor<T, 2> backward(const Tensor<T, 2>& grad_output) = 0;
    
    /**
     * @brief Get trainable parameters
     * @return Vector of parameter tensors
     */
    virtual std::vector<Tensor<T, 2>*> parameters() { return {}; }
    
    /**
     * @brief Set training mode
     * @param mode True for training, false for inference
     */
    virtual void train(bool mode = true) { training_ = mode; }
    
    /**
     * @brief Check if layer is in training mode
     */
    bool is_training() const { return training_; }
    
protected:
    bool training_ = true;
};

/**
 * @brief Linear (Dense/Fully Connected) layer
 * 
 * Performs: output = input @ weights^T + bias
 * 
 * Uses GPU if available and compiled with USE_GPU, otherwise falls back to BLAS or CPU.
 * Backend selection follows: GPU → BLAS → CPU
 */
template<typename T>
class Linear : public Layer<T> {
public:
    /**
     * @brief Construct a linear layer
     * @param in_features Number of input features
     * @param out_features Number of output features
     * @param use_bias Whether to use bias term
     * 
     * Note: GPU acceleration is automatically enabled if compiled with USE_GPU
     *       and GPU hardware is available. Backend selection: GPU → BLAS → CPU
     */
    Linear(size_t in_features, size_t out_features, bool use_bias = true)
        : in_features_(in_features), out_features_(out_features), use_bias_(use_bias),
#ifdef USE_GPU
          weights_({out_features, in_features}, true),
          bias_({1, out_features}, true),
          input_({1, in_features}, true),
          grad_weights_({out_features, in_features}, true),
          grad_bias_({1, out_features}, true) {
#else
          weights_({out_features, in_features}, false),
          bias_({1, out_features}, false),
          input_({1, in_features}, false),
          grad_weights_({out_features, in_features}, false),
          grad_bias_({1, out_features}, false) {
#endif
        
        // Initialize weights with Xavier/Glorot initialization using optimized pointer access
        T stddev = std::sqrt(2.0 / (in_features + out_features));
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<T> dist(0.0, stddev);
        
        // Use direct pointer access for efficient initialization
        weights_.randn(T(0), stddev);
        
        if (use_bias_) {
            bias_.fill(0);
        }
    }
    
    Tensor<T, 2> forward(const Tensor<T, 2>& input) override {
        input_ = input;
        
        // output = input @ weights^T
        auto weights_t = weights_.transpose();
        auto result_var = input.matmul(weights_t);
        auto output = std::get<Tensor<T, 2>>(result_var);
        
        // Add bias if present using broadcasting
        if (use_bias_) {
            auto output_with_bias_var = output + bias_;
            output = std::get<Tensor<T, 2>>(output_with_bias_var);
        }
        
        return output;
    }
    
    Tensor<T, 2> backward(const Tensor<T, 2>& grad_output) override {
        // Gradient w.r.t. weights: grad_output^T @ input
        auto grad_output_t = grad_output.transpose();
        auto grad_weights_var = grad_output_t.matmul(input_);
        grad_weights_ = std::get<Tensor<T, 2>>(grad_weights_var);
        
        // Gradient w.r.t. bias: sum over batch dimension using tensor operations
        if (use_bias_) {
            grad_bias_ = grad_output.sum_axis(0, true);
        }
        
        // Gradient w.r.t. input: grad_output @ weights
        auto grad_input_var = grad_output.matmul(weights_);
        return std::get<Tensor<T, 2>>(grad_input_var);
    }
    
    std::vector<Tensor<T, 2>*> parameters() override {
        std::vector<Tensor<T, 2>*> params;
        params.push_back(&weights_);
        if (use_bias_) {
            params.push_back(&bias_);
        }
        return params;
    }
    
    Tensor<T, 2>& weights() { return weights_; }
    Tensor<T, 2>& bias() { return bias_; }
    Tensor<T, 2>& grad_weights() { return grad_weights_; }
    Tensor<T, 2>& grad_bias() { return grad_bias_; }
    
private:
    size_t in_features_;
    size_t out_features_;
    bool use_bias_;
    
    Tensor<T, 2> weights_;
    Tensor<T, 2> bias_;
    Tensor<T, 2> input_;
    Tensor<T, 2> grad_weights_;
    Tensor<T, 2> grad_bias_;
};

/**
 * @brief ReLU activation layer
 * 
 * Automatically detects and uses GPU from input tensor.
 */
template<typename T>
class ReLU : public Layer<T> {
public:
    ReLU() : input_({1, 1}) {}  // Will auto-detect GPU when used
    
    Tensor<T, 2> forward(const Tensor<T, 2>& input) override {
        input_ = input;
        return input.relu();
    }
    
    Tensor<T, 2> backward(const Tensor<T, 2>& grad_output) override {
        // ReLU gradient: grad * (input > 0)
        // Optimized using tensor comparison operator
        Tensor<T, 2> zeros(input_.shape(), input_.uses_gpu());
        zeros.fill(T(0));
        auto mask = input_ > zeros;  // Returns 0 or 1 tensor
        auto result = grad_output * mask;
        return std::get<Tensor<T, 2>>(result);
    }
    
private:
    Tensor<T, 2> input_;
};

/**
 * @brief Sigmoid activation layer
 */
template<typename T>
class Sigmoid : public Layer<T> {
public:
    Sigmoid() : output_({1, 1}) {}
    
    Tensor<T, 2> forward(const Tensor<T, 2>& input) override {
        output_ = input.sigmoid();
        return output_;
    }
    
    Tensor<T, 2> backward(const Tensor<T, 2>& grad_output) override {
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        // Optimized using tensor operations
        Tensor<T, 2> ones(output_.shape(), output_.uses_gpu());
        ones.fill(T(1));
        auto one_minus_output_var = ones - output_;
        auto& one_minus_output = std::get<Tensor<T, 2>>(one_minus_output_var);
        auto temp = output_ * one_minus_output;
        auto result = grad_output * std::get<Tensor<T, 2>>(temp);
        return std::get<Tensor<T, 2>>(result);
    }
    
private:
    Tensor<T, 2> output_;
};

/**
 * @brief Tanh activation layer
 */
template<typename T>
class Tanh : public Layer<T> {
public:
    Tanh() : output_({1, 1}) {}
    
    Tensor<T, 2> forward(const Tensor<T, 2>& input) override {
        output_ = input.tanh();
        return output_;
    }
    
    Tensor<T, 2> backward(const Tensor<T, 2>& grad_output) override {
        // tanh'(x) = 1 - tanh^2(x)
        // Optimized using tensor operations
        auto tanh_squared_var = output_ * output_;
        auto& tanh_squared = std::get<Tensor<T, 2>>(tanh_squared_var);
        Tensor<T, 2> ones(output_.shape(), output_.uses_gpu());
        ones.fill(T(1));
        auto derivative_var = ones - tanh_squared;
        auto& derivative = std::get<Tensor<T, 2>>(derivative_var);
        auto result = grad_output * derivative;
        return std::get<Tensor<T, 2>>(result);
    }
    
private:
    Tensor<T, 2> output_;
};

/**
 * @brief Dropout layer for regularization
 */
template<typename T>
class Dropout : public Layer<T> {
public:
    /**
     * @brief Construct dropout layer
     * @param p Dropout probability (0 to 1)
     */
    explicit Dropout(T p = 0.5) : p_(p), mask_({1, 1}, false) {
        if (p < 0 || p > 1) {
            throw std::invalid_argument("Dropout probability must be between 0 and 1");
        }
    }
    
    Tensor<T, 2> forward(const Tensor<T, 2>& input) override {
        if (!this->training_) {
            return input;  // No dropout during inference
        }
        
        auto shape = input.shape();
        mask_ = Tensor<T, 2>(shape, input.uses_gpu());
        
        // Generate random mask
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution dist(1.0 - p_);
        
        T scale = 1.0 / (1.0 - p_);
        T* mask_ptr = mask_.data_ptr();
        size_t total_size = shape[0] * shape[1];
        
        for (size_t i = 0; i < total_size; ++i) {
            mask_ptr[i] = dist(gen) ? scale : 0;
        }
        
        // Apply mask using tensor multiplication
        auto result = input * mask_;
        return std::get<Tensor<T, 2>>(result);
    }
    
    Tensor<T, 2> backward(const Tensor<T, 2>& grad_output) override {
        if (!this->training_) {
            return grad_output;
        }
        
        // Gradient passes through mask
        auto result = grad_output * mask_;
        return std::get<Tensor<T, 2>>(result);
    }
    
private:
    T p_;  // Dropout probability
    Tensor<T, 2> mask_;
};

/**
 * @brief Batch Normalization layer
 * 
 * Automatically uses GPU if available.
 */
template<typename T>
class BatchNorm1d : public Layer<T> {
public:
    /**
     * @brief Construct batch normalization layer
     * @param num_features Number of features to normalize
     * @param eps Small constant for numerical stability
     * @param momentum Momentum for running statistics
     * 
     * Note: GPU acceleration is automatically enabled if available.
     */
    BatchNorm1d(size_t num_features, T eps = 1e-5, T momentum = 0.1)
        : num_features_(num_features), eps_(eps), momentum_(momentum),
          gamma_({1, num_features}), beta_({1, num_features}),
          running_mean_({1, num_features}), running_var_({1, num_features}),
          batch_mean_({1, num_features}), batch_var_({1, num_features}),
          input_({1, num_features}), input_normalized_({1, num_features}),
          grad_gamma_({1, num_features}), grad_beta_({1, num_features}) {
        
        gamma_.fill(1);
        beta_.fill(0);
        running_mean_.fill(0);
        running_var_.fill(1);
    }
    
    Tensor<T, 2> forward(const Tensor<T, 2>& input) override {
        auto shape = input.shape();
        size_t batch_size = shape[0];
        
        if (this->training_) {
            // Compute batch statistics using tensor operations
            batch_mean_ = input.mean_axis(0, true);
            
            // Compute variance: E[(X - mean)^2] using broadcasting
            auto centered_var = input - batch_mean_;
            auto centered = std::get<Tensor<T, 2>>(centered_var);
            auto squared_var = centered * centered;
            auto squared = std::get<Tensor<T, 2>>(squared_var);
            batch_var_ = squared.mean_axis(0, true);
            
            // Update running statistics
            auto running_mean_scaled = running_mean_ * (T(1) - momentum_);
            auto batch_mean_scaled = batch_mean_ * momentum_;
            auto new_running_mean_var = running_mean_scaled + batch_mean_scaled;
            running_mean_ = std::get<Tensor<T, 2>>(new_running_mean_var);
            
            auto running_var_scaled = running_var_ * (T(1) - momentum_);
            auto batch_var_scaled = batch_var_ * momentum_;
            auto new_running_var_var = running_var_scaled + batch_var_scaled;
            running_var_ = std::get<Tensor<T, 2>>(new_running_var_var);
        }
        
        // Normalize and scale using broadcasting
        const auto& mean = this->training_ ? batch_mean_ : running_mean_;
        const auto& var = this->training_ ? batch_var_ : running_var_;
        
        auto std_dev = var.map([this](T v) { return std::sqrt(v + eps_); });
        auto centered_var = input - mean;
        auto centered = std::get<Tensor<T, 2>>(centered_var);
        auto normalized_var = centered / std_dev;
        input_normalized_ = std::get<Tensor<T, 2>>(normalized_var);
        auto scaled_var = input_normalized_ * gamma_;
        auto scaled = std::get<Tensor<T, 2>>(scaled_var);
        auto output_var = scaled + beta_;
        auto output = std::get<Tensor<T, 2>>(output_var);
        
        input_ = input;
        return output;
    }
    
    Tensor<T, 2> backward(const Tensor<T, 2>& grad_output) override {
        // Gradients w.r.t. gamma and beta using tensor operations
        auto temp_var = grad_output * input_normalized_;
        auto temp = std::get<Tensor<T, 2>>(temp_var);
        grad_gamma_ = temp.sum_axis(0, true);
        grad_beta_ = grad_output.sum_axis(0, true);
        
        // Gradient w.r.t. input (simplified) using broadcasting
        auto inv_std = batch_var_.map([this](T v) { return T(1) / std::sqrt(v + eps_); });
        auto temp1_var = grad_output * gamma_;
        auto temp1 = std::get<Tensor<T, 2>>(temp1_var);
        auto grad_input_var = temp1 * inv_std;
        return std::get<Tensor<T, 2>>(grad_input_var);
    }
    
    std::vector<Tensor<T, 2>*> parameters() override {
        return {&gamma_, &beta_};
    }
    
private:
    size_t num_features_;
    T eps_;
    T momentum_;
    
    Tensor<T, 2> gamma_;  // Scale parameter
    Tensor<T, 2> beta_;   // Shift parameter
    Tensor<T, 2> running_mean_;
    Tensor<T, 2> running_var_;
    Tensor<T, 2> batch_mean_;
    Tensor<T, 2> batch_var_;
    Tensor<T, 2> input_;
    Tensor<T, 2> input_normalized_;
    Tensor<T, 2> grad_gamma_;
    Tensor<T, 2> grad_beta_;
};

/**
 * @brief Softmax activation layer
 * 
 * Applies softmax activation: converts logits to probabilities
 */
template<typename T>
class Softmax : public Layer<T> {
public:
    Softmax() : output_({1, 1}) {}
    
    Tensor<T, 2> forward(const Tensor<T, 2>& input) override {
        // Use optimized softmax_rows() from tensor.h
        output_ = input.softmax_rows();
        return output_;
    }
    
    Tensor<T, 2> backward(const Tensor<T, 2>& grad_output) override {
        auto shape = output_.shape();
        size_t batch_size = shape[0];
        size_t num_classes = shape[1];
        
#ifdef USE_GPU
        auto grad_input = Tensor<T, 2>(shape, output_.uses_gpu());
#else
        auto grad_input = Tensor<T, 2>(shape, output_.uses_gpu());
#endif
        
        // Compute Jacobian matrices for all samples in batch
        auto jacobians = softmax_jacobian_batch(output_);
        
        // For each sample: grad_input[i] = grad_output[i] @ Jacobian[i]
        for (size_t i = 0; i < batch_size; ++i) {
            // Extract row i from grad_output as (1 x num_classes)
            Tensor<T, 2> grad_row({1, num_classes}, output_.uses_gpu());
            const T* grad_data = grad_output.data_ptr() + i * num_classes;
            std::copy_n(grad_data, num_classes, grad_row.data_ptr());
            
            // Multiply: (1 x num_classes) @ (num_classes x num_classes) = (1 x num_classes)
            auto result_var = grad_row.matmul(jacobians[i]);
            auto result = std::get<Tensor<T, 2>>(result_var);
            
            // Copy result back to grad_input row i
            T* grad_input_row = grad_input.data_ptr() + i * num_classes;
            std::copy_n(result.data_ptr(), num_classes, grad_input_row);
        }
        
        return grad_input;
    }
    
private:
    Tensor<T, 2> output_;
};

// Type aliases for convenience
using Linearf = Linear<float>;
using Lineard = Linear<double>;
using ReLUf = ReLU<float>;
using ReLUd = ReLU<double>;
using Sigmoidf = Sigmoid<float>;
using Sigmoidd = Sigmoid<double>;
using Tanhf = Tanh<float>;
using Tanhd = Tanh<double>;
using Dropoutf = Dropout<float>;
using Dropoutd = Dropout<double>;
using BatchNorm1df = BatchNorm1d<float>;
using BatchNorm1dd = BatchNorm1d<double>;
using Softmaxf = Softmax<float>;
using Softmaxd = Softmax<double>;

// ============================================
// Utility Functions for Classification Tasks
// ============================================

/**
 * @brief Convert label to one-hot encoded vector
 * @param label The class label (integer)
 * @param onehot Output tensor to fill with one-hot encoding
 * @param batch_idx Index in batch dimension to set
 * @param num_classes Number of classes (columns in onehot tensor)
 * 
 * Optimized implementation using direct memory operations.
 * 
 * @section example_onehot Example
 * @code
 * Tensor<float, 2> targets({batch_size, 10});
 * label_to_onehot(3, targets, 0, 10);  // Sets row 0 with [0,0,0,1,0,0,0,0,0,0]
 * @endcode
 */
template<typename T>
inline void label_to_onehot(uint8_t label, Tensor<T, 2>& onehot, size_t batch_idx, size_t num_classes) {
    // Zero the entire row first, then set the appropriate index
    T* row_ptr = onehot.data_ptr() + batch_idx * num_classes;
    std::fill_n(row_ptr, num_classes, T(0));
    row_ptr[label] = T(1);
}

/**
 * @brief Compute cross-entropy loss between predictions and targets
 * @param predictions Predicted probabilities (batch_size x num_classes)
 * @param targets Target one-hot vectors (batch_size x num_classes)
 * @param epsilon Small value for numerical stability (default: 1e-7)
 * @return Average cross-entropy loss over the batch
 * 
 * Optimized implementation using vectorized tensor operations.
 * Computes: -sum(targets * log(predictions + epsilon)) / batch_size
 * 
 * @section example_ce_loss Example
 * @code
 * auto predictions = softmax.forward(logits);
 * float loss = cross_entropy_loss(predictions, targets);
 * @endcode
 */
template<typename T>
inline T cross_entropy_loss(const Tensor<T, 2>& predictions, const Tensor<T, 2>& targets, T epsilon = T(1e-7)) {
    // Compute using optimized tensor operations:
    // loss = -sum(targets * log(predictions + epsilon)) / batch_size
    
    // Add epsilon for numerical stability: predictions + epsilon
    auto pred_stable = predictions.map([epsilon](T x) { return x + epsilon; });
    
    // Compute log: log(predictions + epsilon)
    auto log_pred = pred_stable.log();
    
    // Element-wise multiplication: targets * log(predictions + epsilon)
    auto product_var = targets * log_pred;
    auto product = std::get<Tensor<T, 2>>(product_var);
    
    // Sum all elements and negate
    T total_loss = -product.sum();
    
    // Divide by batch size
    auto shape = predictions.shape();
    return total_loss / static_cast<T>(shape[0]);
}

/**
 * @brief Compute classification accuracy
 * @param predictions Predicted probabilities (batch_size x num_classes)
 * @param labels True labels (vector of class indices)
 * @param offset Starting index in labels vector for this batch
 * @return Accuracy as fraction of correct predictions (0.0 to 1.0)
 * 
 * Optimized implementation using argmax_rows() tensor operation.
 */
template<typename T>
inline T compute_accuracy(const Tensor<T, 2>& predictions, const std::vector<uint8_t>& labels, size_t offset = 0) {
    auto shape = predictions.shape();
    size_t batch_size = shape[0];
    
    // Use tensor operation to get predicted classes for all rows at once
    auto pred_classes = predictions.argmax_rows();
    
    // Count correct predictions
    size_t correct = 0;
    const size_t* pred_data = pred_classes.data_ptr();
    
    for (size_t i = 0; i < batch_size; ++i) {
        if (pred_data[i] == labels[offset + i]) {
            ++correct;
        }
    }
    
    return static_cast<T>(correct) / static_cast<T>(batch_size);
}

/**
 * @brief Update weights of a linear layer using SGD
 * @param layer The linear layer to update
 * @param lr Learning rate
 * 
 * Performs SGD weight update using optimized tensor operations:
 * weights -= lr * grad_weights
 * bias -= lr * grad_bias
 * 
 * Uses vectorized operations for efficient updates across all backends (GPU/BLAS/CPU).
 * 
 * @section example_update_layer Example
 * @code
 * Linear<float> fc1(784, 128);
 * // ... forward and backward passes ...
 * update_linear_layer(fc1, 0.01f);  // Update with learning rate 0.01
 * @endcode
 */
template<typename T>
inline void update_linear_layer(Linear<T>& layer, T lr) {
    // Get weights and their gradients
    auto& weights = layer.weights();
    auto& bias = layer.bias();
    auto& grad_w = layer.grad_weights();
    auto& grad_b = layer.grad_bias();
    
    // Update weights: w -= lr * grad_w using tensor operations
    auto weight_update_var = weights - (grad_w * lr);
    auto weight_update = std::get<Tensor<T, 2>>(weight_update_var);
    weights = weight_update;
    
    // Update bias: b -= lr * grad_b using tensor operations
    auto bias_update_var = bias - (grad_b * lr);
    auto bias_update = std::get<Tensor<T, 2>>(bias_update_var);
    bias = bias_update;
}

/**
 * @brief Compute the Jacobian matrix of softmax for a single row
 * @param softmax_output Softmax output vector (1D or single row from 2D tensor)
 * @return Jacobian matrix (n x n) where n is the length of the softmax output
 * 
 * For softmax function σ(x), the Jacobian is:
 * J_ij = σ_i * (δ_ij - σ_j)
 * where δ_ij is the Kronecker delta (1 if i==j, 0 otherwise).
 * 
 * This can be computed using tensor operations as:
 * J = diag(σ) - σ ⊗ σ^T
 * 
 * Uses optimized tensor operations for efficient computation across all backends.
 */
template<typename T>
inline Tensor<T, 2> softmax_jacobian(const Tensor<T, 1>& softmax_output) {
    auto dims = softmax_output.dims();
    size_t n = dims[0];
    bool use_gpu = softmax_output.uses_gpu();
    
    // For jacobian computation, we use CPU since it involves small matrices
    // (typically 10-1000 classes) and complex diagonal operations
    // The main GPU benefit is in the forward/backward matmul operations
    
    // Get CPU data
    const T* s_data = softmax_output.data_ptr();  // This syncs to CPU if needed
    
    // Compute outer product on CPU for simplicity
    Tensor<T, 2> outer_product({n, n}, false);  // CPU tensor
    T* outer_data = outer_product.data_ptr();
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            outer_data[i * n + j] = s_data[i] * s_data[j];
        }
    }
    
    // Create diagonal matrix on CPU
    Tensor<T, 2> diag_s({n, n}, false);  // CPU tensor
    diag_s.fill(T(0));
    T* diag_data = diag_s.data_ptr();
    
    for (size_t i = 0; i < n; ++i) {
        diag_data[i * n + i] = s_data[i];
    }
    
    // Compute J = diag(s) - outer_product using tensor operation
    auto result_var = diag_s - outer_product;
    auto jacobian = std::get<Tensor<T, 2>>(result_var);
    
    // If input was on GPU, optionally move result to GPU
    // For now, keep on CPU since jacobian is used immediately for gradient computation
    // which may involve element-wise operations easier on CPU
    
    return jacobian;
}

/**
 * @brief Compute softmax Jacobian for batched input (batch_size x num_classes)
 * @param softmax_output Softmax output tensor (batch_size x num_classes)
 * @return Vector of Jacobian matrices, one for each sample in the batch
 * 
 * For each row i in the batch, computes the Jacobian J^(i) where:
 * J^(i)_jk = σ^(i)_j * (δ_jk - σ^(i)_k)
 * 
 * Uses optimized tensor operations for efficient computation.
 */
template<typename T>
inline std::vector<Tensor<T, 2>> softmax_jacobian_batch(const Tensor<T, 2>& softmax_output) {
    auto shape = softmax_output.shape();
    size_t batch_size = shape[0];
    size_t num_classes = shape[1];
    
    std::vector<Tensor<T, 2>> jacobians;
    jacobians.reserve(batch_size);
    
    // Get CPU data for processing
    const T* data = softmax_output.data_ptr();
    
    // Process each sample - extract row and compute Jacobian
    for (size_t i = 0; i < batch_size; ++i) {
        // Extract row i into a 1D tensor (on CPU)
        Tensor<T, 1> row({num_classes}, false);
        std::copy_n(data + i * num_classes, num_classes, row.data_ptr());
        
        // Compute Jacobian for this row
        jacobians.push_back(softmax_jacobian(row));
    }
    
    return jacobians;
}

} // namespace tensor

#endif // NN_LAYERS_H
