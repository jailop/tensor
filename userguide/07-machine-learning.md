# Machine Learning Features

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
