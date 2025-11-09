# Automatic Differentiation (Autograd)

## Basic Concepts

The library provides automatic differentiation for building and training neural networks.

```cpp
// Create tensor that tracks gradients
auto x = Tensor<float, 1>::from_array({2.0f}, {1}, true);  // requires_grad=true

// Forward pass
auto y_var = x * x;  // y = x²
auto y = std::get<Tensor<float, 1>>(y_var);

// Backward pass: compute gradients
y.backward();

// Access gradient
if (x.grad()) {
    std::cout << "dy/dx = " << (*x.grad())[{0}] << std::endl;  // 4.0
}
```

## Computational Graph

```cpp
auto x = Tensor<float, 1>::from_array({3.0f}, {1}, true);
auto y = Tensor<float, 1>::from_array({4.0f}, {1}, true);

// Build computation: z = x² + x*y + y²
auto x2_var = x * x;
auto xy_var = x * y;
auto y2_var = y * y;
auto x2 = std::get<Tensor<float, 1>>(x2_var);
auto xy = std::get<Tensor<float, 1>>(xy_var);
auto y2 = std::get<Tensor<float, 1>>(y2_var);

auto sum1_var = x2 + xy;
auto sum1 = std::get<Tensor<float, 1>>(sum1_var);
auto z_var = sum1 + y2;
auto z = std::get<Tensor<float, 1>>(z_var);

// Compute gradients
z.backward();

// dz/dx = 2x + y = 10
// dz/dy = x + 2y = 11
std::cout << "dz/dx = " << (*x.grad())[{0}] << std::endl;
std::cout << "dz/dy = " << (*y.grad())[{0}] << std::endl;
```

## Gradient Management

```cpp
// Zero gradients before new backward pass
x.zero_grad();
y.zero_grad();

// Detach from graph (stop gradient tracking)
auto x_detached = x.detach();  // Shares data, no gradients

// Check if tensor is leaf (created by user, not by operation)
bool is_leaf = x.is_leaf();
```

## Training Example

```cpp
// Simple linear regression: y = w*x + b
auto w = Tensor<float, 1>::from_array({0.5f}, {1}, true);
auto b = Tensor<float, 1>::from_array({0.0f}, {1}, true);

for (int epoch = 0; epoch < 100; epoch++) {
    // Forward pass
    auto pred_var = w * x + b;
    auto pred = std::get<Tensor<float, 1>>(pred_var);
    
    // Loss: MSE
    auto diff_var = pred - y_true;
    auto diff = std::get<Tensor<float, 1>>(diff_var);
    auto loss_var = diff * diff;
    auto loss = std::get<Tensor<float, 1>>(loss_var);
    
    // Backward pass
    loss.backward();
    
    // Update weights
    float lr = 0.01f;
    w[{0}] -= lr * (*w.grad())[{0}];
    b[{0}] -= lr * (*b.grad())[{0}];
    
    // Zero gradients
    w.zero_grad();
    b.zero_grad();
}
```

---

**Previous**: [← Linear Algebra](05-linear-algebra.md) | **Next**: [Machine Learning →](07-machine-learning.md)
