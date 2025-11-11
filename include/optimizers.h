/**
 * Provides common optimizers for gradient-based optimization:
 * - SGD (Stochastic Gradient Descent) with momentum
 * - Adam (Adaptive Moment Estimation)
 * - RMSprop (Root Mean Square Propagation)
 */

#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "tensor.h"
#include <vector>
#include <cmath>

namespace tensor {

/**
 * @class Optimizer
 * @brief Base class for optimization algorithms
 * @tparam T Data type (float, double)
 * 
 * Provides interface for all optimizers. Concrete implementations
 * include SGD, Adam, and RMSprop.
 */
template<typename T>
class Optimizer {
protected:
    T learning_rate_;  ///< Learning rate for parameter updates
    
public:
    /**
     * @brief Construct optimizer with learning rate
     * @param learning_rate Step size for updates
     */
    explicit Optimizer(T learning_rate) : learning_rate_(learning_rate) {}
    
    /// @brief Virtual destructor for polymorphism
    virtual ~Optimizer() = default;
    
    /**
     * @brief Perform one optimization step
     * 
     * Updates all registered parameters based on their computed gradients.
     * Must be called after backward() has been called on the loss.
     */
    virtual void step() = 0;
    
    /**
     * @brief Zero out all gradients
     * 
     * Should be called before each backward pass to avoid gradient accumulation
     * (unless accumulation is desired).
     */
    virtual void zero_grad() = 0;
    
    /**
     * @brief Get current learning rate
     * @return Current learning rate value
     */
    T get_lr() const { return learning_rate_; }
    
    /**
     * @brief Set learning rate
     * @param lr New learning rate value
     * 
     * Useful for learning rate scheduling during training.
     */
    void set_lr(T lr) { learning_rate_ = lr; }
};

/**
 * @brief SGD (Stochastic Gradient Descent) optimizer with optional momentum
 * @tparam T Data type (float, double)
 * @tparam N Number of tensor dimensions
 * 
 * Implements the classic SGD algorithm with optional Nesterov momentum
 * and weight decay (L2 regularization).
 * 
 * Update rule without momentum:
 * ```
 * param = param - lr * grad
 * ```
 * 
 * Update rule with momentum:
 * ```
 * v_t = momentum * v_{t-1} + grad
 * param = param - lr * v_t
 * ```
 * 
 * With weight decay:
 * ```
 * grad = grad + weight_decay * param
 * ```
 * 
 * @section example_sgd Example
 * @code
 * Tensor<float, 2> W({100, 50}, true, true);
 * std::vector<Tensor<float, 2>*> params = {&W};
 * 
 * // SGD with momentum and weight decay
 * SGD<float, 2> optimizer(params, 0.01f, 0.9f, 0.0001f);
 * 
 * // Training step
 * auto loss = compute_loss(W);
 * optimizer.zero_grad();
 * loss.backward();
 * optimizer.step();
 * @endcode
 */
template<typename T, size_t N>
class SGD : public Optimizer<T> {
private:
    std::vector<Tensor<T, N>*> parameters_;
    std::vector<Tensor<T, N>> velocities_;  // For momentum
    T momentum_;
    T weight_decay_;
    bool use_momentum_;
    
public:
    /**
     * Constructor for SGD optimizer.
     * @param parameters List of tensors to optimize
     * @param learning_rate Learning rate (step size)
     * @param momentum Momentum factor (default: 0.0, no momentum)
     * @param weight_decay L2 regularization coefficient (default: 0.0)
     */
    SGD(std::vector<Tensor<T, N>*> parameters, T learning_rate, 
        T momentum = T(0), T weight_decay = T(0))
        : Optimizer<T>(learning_rate), 
          parameters_(parameters),
          momentum_(momentum),
          weight_decay_(weight_decay),
          use_momentum_(momentum > T(0)) {
        
        // Initialize velocity buffers for momentum
        if (use_momentum_) {
            for (auto* param : parameters_) {
                velocities_.emplace_back(param->dims(), param->uses_gpu(), false);
                velocities_.back().fill(T(0));
            }
        }
    }
    
    void step() override {
        for (size_t i = 0; i < parameters_.size(); ++i) {
            auto* param = parameters_[i];
            
            if (!param->requires_grad() || !param->grad()) {
                continue;
            }
            
            auto* grad = param->grad();
            
            // Apply weight decay if specified
            if (weight_decay_ > T(0)) {
                size_t total = param->total_size();
                for (size_t j = 0; j < total; ++j) {
                    grad->data()[j] += weight_decay_ * param->data()[j];
                }
            }
            
            if (use_momentum_) {
                // Update velocity: v = momentum * v + lr * grad
                auto& velocity = velocities_[i];
                size_t total = param->total_size();
                
                for (size_t j = 0; j < total; ++j) {
                    velocity.data()[j] = momentum_ * velocity.data()[j] + 
                                        this->learning_rate_ * grad->data()[j];
                    param->data()[j] -= velocity.data()[j];
                }
            } else {
                // Standard SGD: param = param - lr * grad
                size_t total = param->total_size();
                for (size_t j = 0; j < total; ++j) {
                    param->data()[j] -= this->learning_rate_ * grad->data()[j];
                }
            }
        }
    }
    
    void zero_grad() override {
        for (auto* param : parameters_) {
            if (param->grad()) {
                param->zero_grad();
            }
        }
    }
};

/**
 * Adam (Adaptive Moment Estimation) optimizer.
 * 
 * Combines ideas from RMSprop and momentum.
 * Maintains running averages of both gradients and their squares.
 * 
 * Update rule:
 *   m_t = beta1 * m_{t-1} + (1 - beta1) * grad
 *   v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
 *   m_hat = m_t / (1 - beta1^t)
 *   v_hat = v_t / (1 - beta2^t)
 *   param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
 */
template<typename T, size_t N>
class Adam : public Optimizer<T> {
private:
    std::vector<Tensor<T, N>*> parameters_;
    std::vector<Tensor<T, N>> first_moments_;   // m_t (gradient moving average)
    std::vector<Tensor<T, N>> second_moments_;  // v_t (squared gradient moving average)
    T beta1_;
    T beta2_;
    T epsilon_;
    T weight_decay_;
    size_t step_count_;
    
public:
    /**
     * Constructor for Adam optimizer.
     * @param parameters List of tensors to optimize
     * @param learning_rate Learning rate (default: 0.001)
     * @param beta1 Exponential decay rate for first moment (default: 0.9)
     * @param beta2 Exponential decay rate for second moment (default: 0.999)
     * @param epsilon Small constant for numerical stability (default: 1e-8)
     * @param weight_decay L2 regularization coefficient (default: 0.0)
     */
    Adam(std::vector<Tensor<T, N>*> parameters, 
         T learning_rate = T(0.001),
         T beta1 = T(0.9),
         T beta2 = T(0.999),
         T epsilon = T(1e-8),
         T weight_decay = T(0))
        : Optimizer<T>(learning_rate),
          parameters_(parameters),
          beta1_(beta1),
          beta2_(beta2),
          epsilon_(epsilon),
          weight_decay_(weight_decay),
          step_count_(0) {
        
        // Initialize moment buffers
        for (auto* param : parameters_) {
            first_moments_.emplace_back(param->dims(), param->uses_gpu(), false);
            first_moments_.back().fill(T(0));
            
            second_moments_.emplace_back(param->dims(), param->uses_gpu(), false);
            second_moments_.back().fill(T(0));
        }
    }
    
    void step() override {
        step_count_++;
        
        // Compute bias correction terms
        T bias_correction1 = T(1) - std::pow(beta1_, static_cast<T>(step_count_));
        T bias_correction2 = T(1) - std::pow(beta2_, static_cast<T>(step_count_));
        
        for (size_t i = 0; i < parameters_.size(); ++i) {
            auto* param = parameters_[i];
            
            if (!param->requires_grad() || !param->grad()) {
                continue;
            }
            
            auto* grad = param->grad();
            auto& m = first_moments_[i];
            auto& v = second_moments_[i];
            
            size_t total = param->total_size();
            
            for (size_t j = 0; j < total; ++j) {
                T grad_val = grad->data()[j];
                
                // Apply weight decay
                if (weight_decay_ > T(0)) {
                    grad_val += weight_decay_ * param->data()[j];
                }
                
                // Update biased first moment estimate
                m.data()[j] = beta1_ * m.data()[j] + (T(1) - beta1_) * grad_val;
                
                // Update biased second moment estimate
                v.data()[j] = beta2_ * v.data()[j] + (T(1) - beta2_) * grad_val * grad_val;
                
                // Compute bias-corrected moment estimates
                T m_hat = m.data()[j] / bias_correction1;
                T v_hat = v.data()[j] / bias_correction2;
                
                // Update parameters
                param->data()[j] -= this->learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
            }
        }
    }
    
    void zero_grad() override {
        for (auto* param : parameters_) {
            if (param->grad()) {
                param->zero_grad();
            }
        }
    }
    
    /**
     * Reset optimizer state (useful for training restarts).
     */
    void reset() {
        step_count_ = 0;
        for (auto& m : first_moments_) {
            m.fill(T(0));
        }
        for (auto& v : second_moments_) {
            v.fill(T(0));
        }
    }
};

/**
 * RMSprop optimizer.
 * 
 * Maintains moving average of squared gradients and divides gradient by
 * the square root of this average. Good for non-stationary objectives.
 * 
 * Update rule:
 *   v_t = alpha * v_{t-1} + (1 - alpha) * grad^2
 *   param = param - lr * grad / (sqrt(v_t) + epsilon)
 */
template<typename T, size_t N>
class RMSprop : public Optimizer<T> {
private:
    std::vector<Tensor<T, N>*> parameters_;
    std::vector<Tensor<T, N>> square_avg_;  // Running average of squared gradients
    T alpha_;       // Smoothing constant
    T epsilon_;     // Small constant for numerical stability
    T weight_decay_;
    T momentum_;
    std::vector<Tensor<T, N>> momentum_buffer_;
    bool use_momentum_;
    
public:
    /**
     * Constructor for RMSprop optimizer.
     * @param parameters List of tensors to optimize
     * @param learning_rate Learning rate (default: 0.01)
     * @param alpha Smoothing constant (default: 0.99)
     * @param epsilon Small constant for numerical stability (default: 1e-8)
     * @param weight_decay L2 regularization coefficient (default: 0.0)
     * @param momentum Momentum factor (default: 0.0)
     */
    RMSprop(std::vector<Tensor<T, N>*> parameters,
            T learning_rate = T(0.01),
            T alpha = T(0.99),
            T epsilon = T(1e-8),
            T weight_decay = T(0),
            T momentum = T(0))
        : Optimizer<T>(learning_rate),
          parameters_(parameters),
          alpha_(alpha),
          epsilon_(epsilon),
          weight_decay_(weight_decay),
          momentum_(momentum),
          use_momentum_(momentum > T(0)) {
        
        // Initialize buffers
        for (auto* param : parameters_) {
            square_avg_.emplace_back(param->dims(), param->uses_gpu(), false);
            square_avg_.back().fill(T(0));
            
            if (use_momentum_) {
                momentum_buffer_.emplace_back(param->dims(), param->uses_gpu(), false);
                momentum_buffer_.back().fill(T(0));
            }
        }
    }
    
    void step() override {
        for (size_t i = 0; i < parameters_.size(); ++i) {
            auto* param = parameters_[i];
            
            if (!param->requires_grad() || !param->grad()) {
                continue;
            }
            
            auto* grad = param->grad();
            auto& v = square_avg_[i];
            
            size_t total = param->total_size();
            
            for (size_t j = 0; j < total; ++j) {
                T grad_val = grad->data()[j];
                
                // Apply weight decay
                if (weight_decay_ > T(0)) {
                    grad_val += weight_decay_ * param->data()[j];
                }
                
                // Update moving average of squared gradient
                v.data()[j] = alpha_ * v.data()[j] + (T(1) - alpha_) * grad_val * grad_val;
                
                if (use_momentum_) {
                    auto& buf = momentum_buffer_[i];
                    buf.data()[j] = momentum_ * buf.data()[j] + 
                                   this->learning_rate_ * grad_val / (std::sqrt(v.data()[j]) + epsilon_);
                    param->data()[j] -= buf.data()[j];
                } else {
                    // Update parameters
                    param->data()[j] -= this->learning_rate_ * grad_val / (std::sqrt(v.data()[j]) + epsilon_);
                }
            }
        }
    }
    
    void zero_grad() override {
        for (auto* param : parameters_) {
            if (param->grad()) {
                param->zero_grad();
            }
        }
    }
};

/**
 * Learning rate scheduler - exponential decay.
 */
template<typename T>
class ExponentialLR {
private:
    Optimizer<T>* optimizer_;
    T gamma_;
    T initial_lr_;
    
public:
    /**
     * Constructor for exponential learning rate scheduler.
     * @param optimizer Optimizer to schedule
     * @param gamma Multiplicative factor of learning rate decay
     */
    ExponentialLR(Optimizer<T>* optimizer, T gamma)
        : optimizer_(optimizer), gamma_(gamma), initial_lr_(optimizer->get_lr()) {}
    
    /**
     * Perform one step of learning rate decay.
     */
    void step() {
        T new_lr = optimizer_->get_lr() * gamma_;
        optimizer_->set_lr(new_lr);
    }
    
    /**
     * Reset learning rate to initial value.
     */
    void reset() {
        optimizer_->set_lr(initial_lr_);
    }
};

} // namespace tensor

#endif // OPTIMIZERS_H
