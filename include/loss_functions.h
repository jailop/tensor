#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include "tensor.h"
#include <cmath>

/**
 * @file loss_functions.h
 * @brief Loss functions for training neural networks
 * 
 * This header provides implementations of common loss functions
 * used in machine learning:
 * - MSE (Mean Squared Error) for regression
 * - Cross Entropy Loss for multi-class classification
 * - Binary Cross Entropy for binary classification
 * - L1 Loss (MAE) for robust regression
 * - Smooth L1 Loss (Huber loss)
 * 
 * All loss functions:
 * - Support automatic differentiation
 * - Provide reduction options (mean, sum, none)
 * - Work with GPU and BLAS acceleration
 * 
 * @author Tensor Library Team
 * @version 1.0
 * @date 2024
 * 
 * @section usage_loss Usage Example
 * @code
 * Tensor<float, 2> predictions({10, 5}, true, true);
 * Tensor<float, 2> targets({10, 5});
 * 
 * // Compute loss with autograd
 * auto loss = loss::mse_loss(predictions, targets);
 * loss.backward();  // Compute gradients
 * @endcode
 */

/**
 * @namespace loss
 * @brief Loss functions for neural network training
 * 
 * Provides differentiable loss functions with support for:
 * - Different reduction strategies (mean, sum, none)
 * - Automatic gradient computation
 * - Efficient GPU and BLAS implementations
 */
namespace loss {

/**
 * @brief Mean Squared Error (MSE) Loss
 * 
 * Computes: L = (1/n) * sum((predictions - targets)^2)
 * 
 * Commonly used for regression tasks. The gradient is smooth and proportional
 * to the error magnitude, making it well-suited for problems where large
 * errors should be heavily penalized.
 * 
 * @tparam T Data type (float, double)
 * @tparam N Number of dimensions
 * @param predictions Predicted values from the model
 * @param targets True target values
 * @param reduction Reduction strategy: "mean" (default), "sum", or "none"
 * @return Loss tensor with autograd support, or zero tensor if shapes don't match
 * 
 * @section example_mse Example
 * @code
 * // Regression example
 * Tensor<float, 2> predictions({32, 10}, true, true);  // batch_size=32, output_dim=10
 * Tensor<float, 2> targets({32, 10});
 * 
 * // Compute MSE loss
 * auto loss = loss::mse_loss(predictions, targets, "mean");
 * 
 * // Backward pass
 * loss.backward();
 * 
 * // Gradients are now in predictions.grad()
 * // Gradient: d/dx[(x-y)^2] = 2(x-y)/n
 * 
 * // Different reduction modes:
 * auto loss_sum = loss::mse_loss(pred, target, "sum");     // No averaging
 * auto loss_none = loss::mse_loss(pred, target, "none");   // Element-wise
 * @endcode
 */
template<typename T, size_t N>
Tensor<T, N> mse_loss(const Tensor<T, N>& predictions, const Tensor<T, N>& targets,
                      const std::string& reduction) {
    if (predictions.dims() != targets.dims()) {
        return Tensor<T, N>({1}, predictions.uses_gpu(), false);
    }
    
    bool track_grad = predictions.requires_grad();
    size_t total = predictions.total_size();
    
    // Compute mean squared error
    T loss_value = T(0);
    for (size_t i = 0; i < total; ++i) {
        T d = predictions.data()[i] - targets.data()[i];
        loss_value += d * d;
    }
    
    if (reduction == "mean") {
        loss_value /= static_cast<T>(total);
    }
    
    // Create result tensor filled with the loss value
    Tensor<T, N> result(predictions.dims(), predictions.uses_gpu(), track_grad);
    result.fill(loss_value);
    result.is_leaf_ = false;
    
    // Setup backward pass
    if (track_grad) {
        Tensor<T, N> pred_copy = predictions.detach();
        Tensor<T, N> targ_copy = targets.detach();
        Tensor<T, N>* pred_ptr = const_cast<Tensor<T, N>*>(&predictions);
        T scale = (reduction == "mean") ? T(2) / static_cast<T>(total) : T(2);
        
        result.register_backward([pred_ptr, pred_copy, targ_copy, scale, total]
                                 (const Tensor<T, N>& grad) {
            // Gradient of MSE: d/dx[(x-y)^2] = 2(x-y)
            if (pred_ptr->requires_grad()) {
                if (!pred_ptr->grad_) {
                    pred_ptr->grad_ = std::make_unique<Tensor<T, N>>(pred_ptr->dims_, 
                                                                      pred_ptr->use_gpu_, false);
                    pred_ptr->grad_->fill(T(0));
                }
                
                for (size_t i = 0; i < total; ++i) {
                    T diff = pred_copy.data()[i] - targ_copy.data()[i];
                    pred_ptr->grad_->data()[i] += grad.data()[i] * scale * diff;
                }
                
                if (!pred_ptr->is_leaf_ && !pred_ptr->backward_funcs_.empty()) {
                    for (auto& func : pred_ptr->backward_funcs_) {
                        func(*pred_ptr->grad_);
                    }
                }
            }
        });
    }
    
    return result;
}

/**
 * Cross Entropy Loss with logits.
 * L = -sum(targets * log(softmax(predictions)))
 * 
 * Numerically stable implementation that combines softmax and log.
 * Commonly used for multi-class classification.
 * 
 * @param logits Raw prediction scores (before softmax)
 * @param targets Target class indices or one-hot encoded targets
 * @param reduction "mean" or "sum" (default: "mean")
 * @return Scalar loss tensor with autograd support
 */
template<typename T, size_t N>
Tensor<T, 1> cross_entropy_loss(const Tensor<T, N>& logits, 
                                 const Tensor<T, N>& targets,
                                 const std::string& reduction) {
    // For now, implement for 2D tensors (batch_size, num_classes)
    static_assert(N == 2, "Cross entropy currently supports 2D tensors");
    
    size_t batch_size = logits.dims()[0];
    size_t num_classes = logits.dims()[1];
    
    bool track_grad = logits.requires_grad();
    
    // Compute log_softmax for numerical stability
    auto log_probs = logits.log_softmax(-1);
    
    // Compute negative log likelihood
    TensorIndices<1> loss_shape = {1};
    Tensor<T, 1> loss(loss_shape, logits.uses_gpu(), track_grad);
    loss.is_leaf_ = false;
    
    T total_loss = T(0);
    
    // Assuming targets are class indices (not one-hot)
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < num_classes; ++j) {
            // If targets are one-hot, multiply by target[i,j]
            // If targets are indices, only use the correct class
            total_loss += -targets[{i, j}] * log_probs[{i, j}];
        }
    }
    
    if (reduction == "mean") {
        loss[{0}] = total_loss / static_cast<T>(batch_size);
    } else {
        loss[{0}] = total_loss;
    }
    
    // Backward pass is handled by log_softmax
    
    return loss;
}

/**
 * Binary Cross Entropy Loss.
 * L = -sum(targets * log(predictions) + (1-targets) * log(1-predictions))
 * 
 * Used for binary classification tasks.
 * 
 * @param predictions Predicted probabilities (after sigmoid)
 * @param targets True binary labels (0 or 1)
 * @param reduction "mean", "sum", or "none" (default: "mean")
 * @return Scalar loss tensor with autograd support
 */
template<typename T, size_t N>
Tensor<T, 1> binary_cross_entropy(const Tensor<T, N>& predictions,
                                   const Tensor<T, N>& targets,
                                   const std::string& reduction) {
    if (predictions.dims() != targets.dims()) {
        return Tensor<T, 1>({1}, predictions.uses_gpu(), false);
    }
    
    size_t total = predictions.total_size();
    bool track_grad = predictions.requires_grad();
    
    T total_loss = T(0);
    T epsilon = T(1e-7);  // For numerical stability
    
    for (size_t i = 0; i < total; ++i) {
        T pred = std::max(std::min(predictions.data()[i], T(1) - epsilon), epsilon);
        T target = targets.data()[i];
        
        total_loss += -(target * std::log(pred) + (T(1) - target) * std::log(T(1) - pred));
    }
    
    TensorIndices<1> loss_shape = {1};
    Tensor<T, 1> loss(loss_shape, predictions.uses_gpu(), track_grad);
    loss.is_leaf_ = false;
    
    if (reduction == "mean") {
        loss[{0}] = total_loss / static_cast<T>(total);
    } else if (reduction == "sum") {
        loss[{0}] = total_loss;
    }
    
    // Setup backward pass
    if (track_grad) {
        Tensor<T, N> pred_copy = predictions.detach();
        Tensor<T, N> targ_copy = targets.detach();
        Tensor<T, N>* pred_ptr = const_cast<Tensor<T, N>*>(&predictions);
        T scale = (reduction == "mean") ? T(1) / static_cast<T>(total) : T(1);
        
        loss.register_backward([pred_ptr, pred_copy, targ_copy, scale, total, epsilon]
                              (const Tensor<T, 1>& grad) {
            // Gradient of BCE: -(y/x - (1-y)/(1-x))
            if (pred_ptr->requires_grad()) {
                if (!pred_ptr->grad_) {
                    pred_ptr->grad_ = std::make_unique<Tensor<T, N>>(pred_ptr->dims_, 
                                                                      pred_ptr->use_gpu_, false);
                    pred_ptr->grad_->fill(T(0));
                }
                
                for (size_t i = 0; i < total; ++i) {
                    T pred = std::max(std::min(pred_copy.data()[i], T(1) - epsilon), epsilon);
                    T target = targ_copy.data()[i];
                    
                    T grad_val = grad[{0}] * scale * (-(target / pred) + (T(1) - target) / (T(1) - pred));
                    pred_ptr->grad_->data()[i] += grad_val;
                }
                
                if (!pred_ptr->is_leaf_ && !pred_ptr->backward_funcs_.empty()) {
                    for (auto& func : pred_ptr->backward_funcs_) {
                        func(*pred_ptr->grad_);
                    }
                }
            }
        });
    }
    
    return loss;
}

/**
 * L1 Loss (Mean Absolute Error).
 * L = (1/n) * sum(|predictions - targets|)
 * 
 * More robust to outliers than MSE.
 * 
 * @param predictions Predicted values
 * @param targets True target values
 * @param reduction "mean", "sum", or "none" (default: "mean")
 * @return Scalar loss tensor with autograd support
 */
template<typename T, size_t N>
Tensor<T, 1> l1_loss(const Tensor<T, N>& predictions, const Tensor<T, N>& targets,
                     const std::string& reduction) {
    if (predictions.dims() != targets.dims()) {
        return Tensor<T, 1>({1}, predictions.uses_gpu(), false);
    }
    
    size_t total = predictions.total_size();
    bool track_grad = predictions.requires_grad();
    
    T total_loss = T(0);
    
    for (size_t i = 0; i < total; ++i) {
        total_loss += std::abs(predictions.data()[i] - targets.data()[i]);
    }
    
    TensorIndices<1> loss_shape = {1};
    Tensor<T, 1> loss(loss_shape, predictions.uses_gpu(), track_grad);
    loss.is_leaf_ = false;
    
    if (reduction == "mean") {
        loss[{0}] = total_loss / static_cast<T>(total);
    } else if (reduction == "sum") {
        loss[{0}] = total_loss;
    }
    
    // Setup backward pass
    if (track_grad) {
        Tensor<T, N> pred_copy = predictions.detach();
        Tensor<T, N> targ_copy = targets.detach();
        Tensor<T, N>* pred_ptr = const_cast<Tensor<T, N>*>(&predictions);
        T scale = (reduction == "mean") ? T(1) / static_cast<T>(total) : T(1);
        
        loss.register_backward([pred_ptr, pred_copy, targ_copy, scale, total]
                              (const Tensor<T, 1>& grad) {
            // Gradient of L1: sign(x - y)
            if (pred_ptr->requires_grad()) {
                if (!pred_ptr->grad_) {
                    pred_ptr->grad_ = std::make_unique<Tensor<T, N>>(pred_ptr->dims_, 
                                                                      pred_ptr->use_gpu_, false);
                    pred_ptr->grad_->fill(T(0));
                }
                
                for (size_t i = 0; i < total; ++i) {
                    T diff = pred_copy.data()[i] - targ_copy.data()[i];
                    T sign = (diff > T(0)) ? T(1) : ((diff < T(0)) ? T(-1) : T(0));
                    
                    pred_ptr->grad_->data()[i] += grad[{0}] * scale * sign;
                }
                
                if (!pred_ptr->is_leaf_ && !pred_ptr->backward_funcs_.empty()) {
                    for (auto& func : pred_ptr->backward_funcs_) {
                        func(*pred_ptr->grad_);
                    }
                }
            }
        });
    }
    
    return loss;
}

/**
 * Smooth L1 Loss (Huber Loss).
 * Combines MSE and L1 loss - less sensitive to outliers than MSE.
 * 
 * L = 0.5 * (x - y)^2  if |x - y| < beta
 * L = beta * (|x - y| - 0.5 * beta)  otherwise
 * 
 * @param predictions Predicted values
 * @param targets True target values
 * @param beta Threshold for switching between L1 and L2 (default: 1.0)
 * @param reduction "mean" or "sum" (default: "mean")
 * @return Scalar loss tensor with autograd support
 */
template<typename T, size_t N>
Tensor<T, 1> smooth_l1_loss(const Tensor<T, N>& predictions, 
                            const Tensor<T, N>& targets,
                            T beta,
                            const std::string& reduction) {
    if (predictions.dims() != targets.dims()) {
        return Tensor<T, 1>({1}, predictions.uses_gpu(), false);
    }
    
    size_t total = predictions.total_size();
    bool track_grad = predictions.requires_grad();
    
    T total_loss = T(0);
    
    for (size_t i = 0; i < total; ++i) {
        T diff = std::abs(predictions.data()[i] - targets.data()[i]);
        
        if (diff < beta) {
            total_loss += T(0.5) * diff * diff / beta;
        } else {
            total_loss += diff - T(0.5) * beta;
        }
    }
    
    TensorIndices<1> loss_shape = {1};
    Tensor<T, 1> loss(loss_shape, predictions.uses_gpu(), track_grad);
    loss.is_leaf_ = false;
    
    if (reduction == "mean") {
        loss[{0}] = total_loss / static_cast<T>(total);
    } else {
        loss[{0}] = total_loss;
    }
    
    // Setup backward pass
    if (track_grad) {
        Tensor<T, N> pred_copy = predictions.detach();
        Tensor<T, N> targ_copy = targets.detach();
        Tensor<T, N>* pred_ptr = const_cast<Tensor<T, N>*>(&predictions);
        T scale = (reduction == "mean") ? T(1) / static_cast<T>(total) : T(1);
        
        loss.register_backward([pred_ptr, pred_copy, targ_copy, scale, total, beta]
                              (const Tensor<T, 1>& grad) {
            // Gradient of smooth L1
            if (pred_ptr->requires_grad()) {
                if (!pred_ptr->grad_) {
                    pred_ptr->grad_ = std::make_unique<Tensor<T, N>>(pred_ptr->dims_, 
                                                                      pred_ptr->use_gpu_, false);
                    pred_ptr->grad_->fill(T(0));
                }
                
                for (size_t i = 0; i < total; ++i) {
                    T diff = pred_copy.data()[i] - targ_copy.data()[i];
                    T abs_diff = std::abs(diff);
                    T grad_val;
                    
                    if (abs_diff < beta) {
                        grad_val = diff / beta;
                    } else {
                        grad_val = (diff > T(0)) ? T(1) : T(-1);
                    }
                    
                    pred_ptr->grad_->data()[i] += grad[{0}] * scale * grad_val;
                }
                
                if (!pred_ptr->is_leaf_ && !pred_ptr->backward_funcs_.empty()) {
                    for (auto& func : pred_ptr->backward_funcs_) {
                        func(*pred_ptr->grad_);
                    }
                }
            }
        });
    }
    
    return loss;
}

} // namespace loss

#endif // LOSS_FUNCTIONS_H
