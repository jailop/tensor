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

namespace tensor4d {
namespace nn {

// Forward declarations
template<typename T>
class Layer;

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
 * Automatically uses GPU if available, otherwise falls back to BLAS or CPU.
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
     * Note: GPU acceleration is automatically enabled if available.
     *       The backend selection is: GPU → BLAS → CPU
     */
    Linear(size_t in_features, size_t out_features, bool use_bias = true)
        : in_features_(in_features), out_features_(out_features), use_bias_(use_bias),
          weights_({out_features, in_features}),  // use_gpu defaults to true
          bias_({1, out_features}),
          input_({1, in_features}),
          grad_weights_({out_features, in_features}),
          grad_bias_({1, out_features}) {
        
        // Initialize weights with Xavier/Glorot initialization
        T stddev = std::sqrt(2.0 / (in_features + out_features));
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<T> dist(0.0, stddev);
        
        for (size_t i = 0; i < out_features; ++i) {
            for (size_t j = 0; j < in_features; ++j) {
                weights_[{i, j}] = dist(gen);
            }
        }
        
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
        auto mask = input_.map([](T val) { return val > 0 ? T(1) : T(0); });
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
        auto one_minus_output = output_.map([](T val) { return T(1) - val; });
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
        auto tanh_squared_var = output_ * output_;
        auto tanh_squared = std::get<Tensor<T, 2>>(tanh_squared_var);
        auto derivative = tanh_squared.map([](T val) { return T(1) - val; });
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
 */
template<typename T>
class Softmax : public Layer<T> {
public:
    Softmax() : output_({1, 1}) {}
    
    Tensor<T, 2> forward(const Tensor<T, 2>& input) override {
        auto shape = input.shape();
        output_ = Tensor<T, 2>(shape, input.uses_gpu());
        
        // Compute softmax for each sample in batch
        // Note: Row-wise operations require per-row processing for now
        const T* input_ptr = input.data_ptr();
        T* output_ptr = output_.data_ptr();
        
        for (size_t i = 0; i < shape[0]; ++i) {
            const T* row_input = input_ptr + i * shape[1];
            T* row_output = output_ptr + i * shape[1];
            
            // Find max for numerical stability
            T max_val = *std::max_element(row_input, row_input + shape[1]);
            
            // Compute exp and sum
            T sum = 0;
            for (size_t j = 0; j < shape[1]; ++j) {
                row_output[j] = std::exp(row_input[j] - max_val);
                sum += row_output[j];
            }
            
            // Normalize
            T inv_sum = T(1) / sum;
            for (size_t j = 0; j < shape[1]; ++j) {
                row_output[j] *= inv_sum;
            }
        }
        
        return output_;
    }
    
    Tensor<T, 2> backward(const Tensor<T, 2>& grad_output) override {
        auto shape = output_.shape();
        auto grad_input = Tensor<T, 2>(shape, output_.uses_gpu());
        
        // Jacobian of softmax is: S_i * (δ_ij - S_j)
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                T sum = 0;
                for (size_t k = 0; k < shape[1]; ++k) {
                    T delta = (j == k) ? 1.0 : 0.0;
                    sum += grad_output[{i, k}] * output_[{i, j}] * (delta - output_[{i, k}]);
                }
                grad_input[{i, j}] = sum;
            }
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

} // namespace nn
} // namespace tensor4d

#endif // NN_LAYERS_H
