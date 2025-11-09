# Machine Learning Features

This section covers the machine learning components of the library, including neural network layers, loss functions, and optimizers.

## Neural Network Layers

The library provides a comprehensive set of neural network layers through `nn_layers.h`. All layers inherit from a base `Layer<T>` class and support forward/backward passes.

### Layer Base Class

```cpp
#include "nn_layers.h"
using namespace tensor4d::nn;

// All layers inherit from Layer<T>
template<typename T>
class Layer {
public:
    virtual Tensor<T, 2> forward(const Tensor<T, 2>& input) = 0;
    virtual Tensor<T, 2> backward(const Tensor<T, 2>& grad_output) = 0;
    virtual std::vector<Tensor<T, 2>*> parameters() { return {}; }
    virtual void train(bool mode = true) { training_ = mode; }
    bool is_training() const { return training_; }
};
```

### Linear (Fully Connected) Layer

```cpp
// Linear layer: output = input @ weights^T + bias
Linearf linear(784, 128, true);  // in_features=784, out_features=128, use_bias=true

// Type aliases for convenience
using Linearf = Linear<float>;
using Lineard = Linear<double>;

// Forward pass
Matrixf input({64, 784});  // Batch of 64 samples
auto output = linear.forward(input);  // Shape: {64, 128}

// Get parameters for optimizer
auto params = linear.parameters();  // Returns {&weights, &bias}
```

**Xavier/Glorot Initialization**: Weights are initialized with `N(0, sqrt(2/(in+out)))` for better training convergence.

### Activation Layers

#### ReLU (Rectified Linear Unit)

```cpp
ReLUf relu;
auto output = relu.forward(input);  // max(0, input)
```

#### Sigmoid

```cpp
Sigmoidf sigmoid;
auto output = sigmoid.forward(input);  // 1 / (1 + exp(-input))
```

#### Tanh (Hyperbolic Tangent)

```cpp
Tanhf tanh;
auto output = tanh.forward(input);  // (exp(x) - exp(-x)) / (exp(x) + exp(-x))
```

#### Softmax

```cpp
Softmaxf softmax(1);  // dim=1 (apply along columns)
auto probs = softmax.forward(logits);  // Outputs sum to 1.0 along dim
```

### Regularization Layers

#### Dropout

```cpp
Dropoutf dropout(0.5f);  // Drop probability = 0.5

// Training mode: randomly zero elements and scale remaining
dropout.train(true);
auto train_output = dropout.forward(input);

// Inference mode: pass through unchanged
dropout.train(false);
auto eval_output = dropout.forward(input);
```

**Inverted Dropout**: During training, remaining values are scaled by `1/(1-p)` to maintain expected values.

#### Batch Normalization (1D)

```cpp
BatchNorm1df bn(128);  // num_features=128

// Training: normalize using batch statistics
bn.train(true);
auto normalized = bn.forward(input);  // Mean≈0, Std≈1 per feature

// Inference: use running statistics
bn.train(false);
auto test_output = bn.forward(test_input);

// Learnable parameters: gamma (scale) and beta (shift)
auto params = bn.parameters();  // Returns {&gamma, &beta}
```

**Running Statistics**: During training, maintains exponential moving average of mean and variance (momentum=0.1).

### Building Neural Networks

```cpp
#include "nn_layers.h"
using namespace tensor4d::nn;

// Define network architecture
Linearf fc1(784, 256, true);   // Input layer
ReLUf relu1;
Dropoutf dropout1(0.2f);
Linearf fc2(256, 128, true);   // Hidden layer
ReLUf relu2;
Dropoutf dropout2(0.2f);
Linearf fc3(128, 10, true);    // Output layer
Softmaxf softmax;

// Forward pass
auto h1 = fc1.forward(input);
auto h1_act = relu1.forward(h1);
auto h1_drop = dropout1.forward(h1_act);

auto h2 = fc2.forward(h1_drop);
auto h2_act = relu2.forward(h2);
auto h2_drop = dropout2.forward(h2_act);

auto logits = fc3.forward(h2_drop);
auto probs = softmax.forward(logits);

// Collect all parameters for optimizer
std::vector<Tensor<float, 2>*> all_params;
auto p1 = fc1.parameters();
auto p2 = fc2.parameters();
auto p3 = fc3.parameters();
all_params.insert(all_params.end(), p1.begin(), p1.end());
all_params.insert(all_params.end(), p2.begin(), p2.end());
all_params.insert(all_params.end(), p3.begin(), p3.end());
```

## Loss Functions

```cpp
#include "loss_functions.h"

// Mean Squared Error
MSELoss mse_loss;
auto pred = Matrix<float>::randn({64, 10});
auto target = Matrix<float>::randn({64, 10});
auto loss_var = mse_loss.forward(pred, target);
auto loss = std::get<Tensor<float, 1>>(loss_var);

// Cross Entropy Loss
CrossEntropyLoss ce_loss;
auto logits = Matrix<float>::randn({64, 10});  // Raw scores
auto labels = Vector<int>::from_array({...}, {64});  // Class indices
auto ce_var = ce_loss.forward(logits, labels);

// Binary Cross Entropy
BinaryCrossEntropyLoss bce_loss;
auto probs = Matrix<float>::sigmoid(...);  // Must be in [0, 1]
auto binary_targets = Matrix<float>::from_array({...}, {...});
auto bce_var = bce_loss.forward(probs, binary_targets);

// L1 Loss (MAE)
L1Loss l1_loss;
auto mae_var = l1_loss.forward(pred, target);

// Smooth L1 Loss (Huber)
SmoothL1Loss smooth_l1(1.0f);  // beta=1.0
auto huber_var = smooth_l1.forward(pred, target);
```

## Optimizers

```cpp
#include "optimizers.h"

// Stochastic Gradient Descent
auto params = {&weights, &bias};  // Tensors requiring gradients
SGD optimizer(params, 0.01f);  // learning_rate=0.01

// Training loop
for (int epoch = 0; epoch < num_epochs; epoch++) {
    // Forward pass
    auto pred = model.forward(input);
    auto loss = loss_fn.forward(pred, target);
    
    // Backward pass
    loss.backward();
    
    // Update parameters
    optimizer.step();
    
    // Zero gradients
    optimizer.zero_grad();
}

// Adam Optimizer
Adam adam_opt(params, 0.001f, 0.9f, 0.999f, 1e-8f);

// AdamW (Adam with weight decay)
AdamW adamw_opt(params, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);

// RMSprop
RMSprop rmsprop_opt(params, 0.01f, 0.9f, 1e-8f);
```

## Complete MNIST Training Example

Here's a complete example training a neural network on the MNIST digit classification dataset:

```cpp
#include "nn_layers.h"
#include "tensor_types.h"
#include "loss_functions.h"
#include "optimizers.h"
#include <iostream>

using namespace tensor4d;
using namespace tensor4d::nn;

int main() {
    // Network architecture: 784 -> 256 -> 128 -> 10
    Linearf fc1(784, 256);
    ReLUf relu1;
    Dropoutf dropout1(0.2f);
    
    Linearf fc2(256, 128);
    ReLUf relu2;
    Dropoutf dropout2(0.2f);
    
    Linearf fc3(128, 10);
    Softmaxf softmax;
    
    // Collect all trainable parameters
    std::vector<Tensor<float, 2>*> params;
    auto p1 = fc1.parameters();
    auto p2 = fc2.parameters();
    auto p3 = fc3.parameters();
    params.insert(params.end(), p1.begin(), p1.end());
    params.insert(params.end(), p2.begin(), p2.end());
    params.insert(params.end(), p3.begin(), p3.end());
    
    // Setup optimizer and loss
    Adam optimizer(params, 0.001f);
    CrossEntropyLoss criterion;
    
    // Training loop
    constexpr int num_epochs = 10;
    constexpr int batch_size = 128;
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Set layers to training mode
        dropout1.train(true);
        dropout2.train(true);
        
        float epoch_loss = 0.0f;
        int num_batches = 0;
        
        // Iterate through mini-batches
        for (auto [X_batch, y_batch] : train_loader) {
            // Forward pass
            auto h1 = fc1.forward(X_batch);
            auto h1_act = relu1.forward(h1);
            auto h1_drop = dropout1.forward(h1_act);
            
            auto h2 = fc2.forward(h1_drop);
            auto h2_act = relu2.forward(h2);
            auto h2_drop = dropout2.forward(h2_act);
            
            auto logits = fc3.forward(h2_drop);
            
            // Compute loss
            auto loss = criterion.forward(logits, y_batch);
            epoch_loss += loss[{0}];
            
            // Backward pass
            loss.backward();
            
            // Update parameters
            optimizer.step();
            optimizer.zero_grad();
            
            num_batches++;
        }
        
        // Evaluation on validation set
        dropout1.train(false);
        dropout2.train(false);
        
        float val_acc = 0.0f;
        int total_samples = 0;
        
        for (auto [X_val, y_val] : val_loader) {
            auto h1 = fc1.forward(X_val);
            auto h1_act = relu1.forward(h1);
            
            auto h2 = fc2.forward(h1_act);
            auto h2_act = relu2.forward(h2);
            
            auto logits = fc3.forward(h2_act);
            auto probs = softmax.forward(logits);
            
            // Compute accuracy
            for (size_t i = 0; i < probs.shape()[0]; ++i) {
                int pred_class = 0;
                float max_prob = probs[{i, 0}];
                for (size_t j = 1; j < 10; ++j) {
                    if (probs[{i, j}] > max_prob) {
                        max_prob = probs[{i, j}];
                        pred_class = j;
                    }
                }
                if (pred_class == y_val[{i}]) {
                    val_acc += 1.0f;
                }
                total_samples++;
            }
        }
        
        val_acc /= total_samples;
        std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs
                  << " - Loss: " << (epoch_loss / num_batches)
                  << " - Val Acc: " << (val_acc * 100.0f) << "%" << std::endl;
    }
    
    return 0;
}
```

**Key Points:**
- Use `train(true/false)` to toggle training/inference mode for Dropout and BatchNorm
- Collect all layer parameters before creating the optimizer
- Call `zero_grad()` after each `step()` to clear accumulated gradients
- Evaluate on validation set without dropout for accurate metrics

## Layer Type Aliases

For convenience, the library provides type aliases for common data types:

```cpp
// Float variants
using Linearf = Linear<float>;
using ReLUf = ReLU<float>;
using Sigmoidf = Sigmoid<float>;
using Tanhf = Tanh<float>;
using Dropoutf = Dropout<float>;
using BatchNorm1df = BatchNorm1d<float>;
using Softmaxf = Softmax<float>;

// Double variants
using Lineard = Linear<double>;
using ReLUd = ReLU<double>;
// ... and so on
```

## Best Practices

1. **Parameter Initialization**: Linear layers use Xavier initialization by default, which works well for most cases
2. **Learning Rate**: Start with 0.001 for Adam, 0.01 for SGD, and adjust based on training curves
3. **Dropout**: Use 0.2-0.5 for hidden layers; disable during inference
4. **Batch Normalization**: Place after linear layers but before activation functions
5. **Gradient Accumulation**: Always call `zero_grad()` before or after `step()` to prevent gradient accumulation
6. **Training Mode**: Remember to toggle `train(true/false)` when switching between training and evaluation

## Performance Tips

- **Batch Size**: Larger batches (64-256) improve GPU/BLAS utilization
- **Vectorization**: Process multiple samples simultaneously using batch dimension
- **Mixed Precision**: Use `float` (FP32) for most cases; consider `half` (FP16) for GPU training
- **Memory**: Gradients are only stored for tensors with `requires_grad=true`

## Complete Training Example

```cpp
// Define model parameters
auto W1 = Matrix<float>::randn({784, 128}, true);  // Input layer
auto b1 = Vector<float>::zeros({128}, true);
auto W2 = Matrix<float>::randn({128, 10}, true);   // Output layer
auto b2 = Vector<float>::zeros({10}, true);

// Setup
auto params = {&W1, &b1, &W2, &b2};
Adam optimizer(params, 0.001f);
CrossEntropyLoss loss_fn;

// Training loop
for (int epoch = 0; epoch < 10; epoch++) {
    for (auto [X_batch, y_batch] : dataloader) {
        // Forward pass
        auto h1_var = matmul(X_batch, W1) + b1;
        auto h1 = std::get<Matrix<float>>(h1_var);
        auto h1_act = h1.relu();
        
        auto h2_var = matmul(h1_act, W2) + b2;
        auto logits = std::get<Matrix<float>>(h2_var);
        
        // Compute loss
        auto loss = loss_fn.forward(logits, y_batch);
        
        // Backward pass
        loss.backward();
        
        // Update weights
        optimizer.step();
        optimizer.zero_grad();
    }
}
```

---

**Previous**: [← Autograd](06-autograd.md) | **Next**: [Advanced Indexing →](08-advanced-indexing.md)
