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
 */
template<typename T>
class Linear : public Layer<T> {
public:
    /**
     * @brief Construct a linear layer
     * @param in_features Number of input features
     * @param out_features Number of output features
     * @param use_bias Whether to use bias term
     */
    Linear(size_t in_features, size_t out_features, bool use_bias = true)
        : in_features_(in_features), out_features_(out_features), use_bias_(use_bias),
          weights_({out_features, in_features}, false), 
          bias_({1, out_features}, false),
          input_({1, in_features}, false),
          grad_weights_({out_features, in_features}, false),
          grad_bias_({1, out_features}, false) {
        
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
        
        // Add bias if present (manually broadcast)
        if (use_bias_) {
            auto shape = output.shape();
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    output[{i, j}] += bias_[{0, j}];
                }
            }
        }
        
        return output;
    }
    
    Tensor<T, 2> backward(const Tensor<T, 2>& grad_output) override {
        // Gradient w.r.t. weights: grad_output^T @ input
        auto grad_output_t = grad_output.transpose();
        auto grad_weights_var = grad_output_t.matmul(input_);
        grad_weights_ = std::get<Tensor<T, 2>>(grad_weights_var);
        
        // Gradient w.r.t. bias: sum over batch dimension
        if (use_bias_) {
            auto batch_size = grad_output.shape()[0];
            grad_bias_ = Tensor<T, 2>({1, out_features_});
            grad_bias_.fill(0);
            
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t i = 0; i < out_features_; ++i) {
                    grad_bias_[{0, i}] += grad_output[{b, i}];
                }
            }
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
 */
template<typename T>
class ReLU : public Layer<T> {
public:
    ReLU() : input_({1, 1}, false) {}
    
    Tensor<T, 2> forward(const Tensor<T, 2>& input) override {
        input_ = input;
        return input.relu();
    }
    
    Tensor<T, 2> backward(const Tensor<T, 2>& grad_output) override {
        // ReLU gradient: grad * (input > 0)
        auto shape = input_.shape();
        Tensor<T, 2> grad_input(shape);
        
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                grad_input[{i, j}] = input_[{i, j}] > 0 ? grad_output[{i, j}] : 0;
            }
        }
        
        return grad_input;
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
    Sigmoid() : output_({1, 1}, false) {}
    
    Tensor<T, 2> forward(const Tensor<T, 2>& input) override {
        output_ = input.sigmoid();
        return output_;
    }
    
    Tensor<T, 2> backward(const Tensor<T, 2>& grad_output) override {
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        auto shape = output_.shape();
        Tensor<T, 2> grad_input(shape);
        
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                T sig = output_[{i, j}];
                grad_input[{i, j}] = grad_output[{i, j}] * sig * (1 - sig);
            }
        }
        
        return grad_input;
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
    Tanh() : output_({1, 1}, false) {}
    
    Tensor<T, 2> forward(const Tensor<T, 2>& input) override {
        output_ = input.tanh();
        return output_;
    }
    
    Tensor<T, 2> backward(const Tensor<T, 2>& grad_output) override {
        // tanh'(x) = 1 - tanh^2(x)
        auto shape = output_.shape();
        Tensor<T, 2> grad_input(shape);
        
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                T t = output_[{i, j}];
                grad_input[{i, j}] = grad_output[{i, j}] * (1 - t * t);
            }
        }
        
        return grad_input;
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
        mask_ = Tensor<T, 2>(shape);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution dist(1 - p_);
        
        T scale = 1.0 / (1.0 - p_);  // Inverted dropout
        
        auto output = input;
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                bool keep = dist(gen);
                mask_[{i, j}] = keep ? scale : 0;
                output[{i, j}] *= mask_[{i, j}];
            }
        }
        
        return output;
    }
    
    Tensor<T, 2> backward(const Tensor<T, 2>& grad_output) override {
        if (!this->training_) {
            return grad_output;
        }
        
        auto shape = grad_output.shape();
        auto grad_input = grad_output;
        
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                grad_input[{i, j}] *= mask_[{i, j}];
            }
        }
        
        return grad_input;
    }
    
private:
    T p_;  // Dropout probability
    Tensor<T, 2> mask_;
};

/**
 * @brief Batch Normalization layer
 */
template<typename T>
class BatchNorm1d : public Layer<T> {
public:
    /**
     * @brief Construct batch normalization layer
     * @param num_features Number of features to normalize
     * @param eps Small constant for numerical stability
     * @param momentum Momentum for running statistics
     */
    BatchNorm1d(size_t num_features, T eps = 1e-5, T momentum = 0.1)
        : num_features_(num_features), eps_(eps), momentum_(momentum),
          gamma_({1, num_features}, false), beta_({1, num_features}, false),
          running_mean_({1, num_features}, false), running_var_({1, num_features}, false),
          batch_mean_({1, num_features}, false), batch_var_({1, num_features}, false),
          input_({1, num_features}, false), input_normalized_({1, num_features}, false),
          grad_gamma_({1, num_features}, false), grad_beta_({1, num_features}, false) {
        
        gamma_ = Tensor<T, 2>({1, num_features});
        beta_ = Tensor<T, 2>({1, num_features});
        gamma_.fill(1);
        beta_.fill(0);
        
        running_mean_ = Tensor<T, 2>({1, num_features});
        running_var_ = Tensor<T, 2>({1, num_features});
        running_mean_.fill(0);
        running_var_.fill(1);
    }
    
    Tensor<T, 2> forward(const Tensor<T, 2>& input) override {
        auto shape = input.shape();
        size_t batch_size = shape[0];
        
        if (this->training_) {
            // Compute batch statistics
            batch_mean_ = Tensor<T, 2>({1, num_features_});
            batch_mean_.fill(0);
            
            for (size_t i = 0; i < batch_size; ++i) {
                for (size_t j = 0; j < num_features_; ++j) {
                    batch_mean_[{0, j}] += input[{i, j}];
                }
            }
            for (size_t j = 0; j < num_features_; ++j) {
                batch_mean_[{0, j}] /= batch_size;
            }
            
            // Compute variance
            batch_var_ = Tensor<T, 2>({1, num_features_});
            batch_var_.fill(0);
            
            for (size_t i = 0; i < batch_size; ++i) {
                for (size_t j = 0; j < num_features_; ++j) {
                    T diff = input[{i, j}] - batch_mean_[{0, j}];
                    batch_var_[{0, j}] += diff * diff;
                }
            }
            for (size_t j = 0; j < num_features_; ++j) {
                batch_var_[{0, j}] /= batch_size;
            }
            
            // Update running statistics
            for (size_t j = 0; j < num_features_; ++j) {
                running_mean_[{0, j}] = (1 - momentum_) * running_mean_[{0, j}] + 
                                        momentum_ * batch_mean_[{0, j}];
                running_var_[{0, j}] = (1 - momentum_) * running_var_[{0, j}] + 
                                       momentum_ * batch_var_[{0, j}];
            }
        }
        
        // Normalize
        input_normalized_ = Tensor<T, 2>(shape);
        auto output = Tensor<T, 2>(shape);
        
        const auto& mean = this->training_ ? batch_mean_ : running_mean_;
        const auto& var = this->training_ ? batch_var_ : running_var_;
        
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < num_features_; ++j) {
                input_normalized_[{i, j}] = (input[{i, j}] - mean[{0, j}]) / 
                                            std::sqrt(var[{0, j}] + eps_);
                output[{i, j}] = gamma_[{0, j}] * input_normalized_[{i, j}] + beta_[{0, j}];
            }
        }
        
        input_ = input;
        return output;
    }
    
    Tensor<T, 2> backward(const Tensor<T, 2>& grad_output) override {
        auto shape = grad_output.shape();
        size_t batch_size = shape[0];
        
        // Gradients w.r.t. gamma and beta
        grad_gamma_ = Tensor<T, 2>({1, num_features_});
        grad_beta_ = Tensor<T, 2>({1, num_features_});
        grad_gamma_.fill(0);
        grad_beta_.fill(0);
        
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < num_features_; ++j) {
                grad_gamma_[{0, j}] += grad_output[{i, j}] * input_normalized_[{i, j}];
                grad_beta_[{0, j}] += grad_output[{i, j}];
            }
        }
        
        // Gradient w.r.t. input (simplified)
        auto grad_input = Tensor<T, 2>(shape);
        T inv_std = 1.0 / std::sqrt(batch_var_[{0, 0}] + eps_);
        
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < num_features_; ++j) {
                grad_input[{i, j}] = grad_output[{i, j}] * gamma_[{0, j}] * inv_std;
            }
        }
        
        return grad_input;
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
    Softmax() : output_({1, 1}, false) {}
    
    Tensor<T, 2> forward(const Tensor<T, 2>& input) override {
        auto shape = input.shape();
        output_ = Tensor<T, 2>(shape);
        
        // Compute softmax for each sample in batch
        for (size_t i = 0; i < shape[0]; ++i) {
            // Find max for numerical stability
            T max_val = input[{i, 0}];
            for (size_t j = 1; j < shape[1]; ++j) {
                if (input[{i, j}] > max_val) {
                    max_val = input[{i, j}];
                }
            }
            
            // Compute exp and sum
            T sum = 0;
            for (size_t j = 0; j < shape[1]; ++j) {
                output_[{i, j}] = std::exp(input[{i, j}] - max_val);
                sum += output_[{i, j}];
            }
            
            // Normalize
            for (size_t j = 0; j < shape[1]; ++j) {
                output_[{i, j}] /= sum;
            }
        }
        
        return output_;
    }
    
    Tensor<T, 2> backward(const Tensor<T, 2>& grad_output) override {
        auto shape = output_.shape();
        auto grad_input = Tensor<T, 2>(shape);
        
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
