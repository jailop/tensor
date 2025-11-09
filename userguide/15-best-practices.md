# Best Practices

## Error Handling

```cpp
// Always check for errors
auto result_var = matmul(A, B);
if (std::holds_alternative<TensorError>(result_var)) {
    auto error = std::get<TensorError>(result_var);
    std::cerr << "Error: " << error.message << std::endl;
    return;
}
auto result = std::get<Matrix<float>>(result_var);
```

## Memory Management

```cpp
// Use in-place operations when possible
A += B;  // More efficient than A = A + B

// Reuse tensors
Tensor<float, 2> temp({100, 100});
for (int i = 0; i < iterations; i++) {
    temp.fill(0.0f);  // Reuse instead of recreating
    // ... operations on temp ...
}

// Enable memory pooling for frequent allocations
TensorMemoryPool::enable(true);
```

## Performance Tips

```cpp
// 1. Batch operations
auto batch_result = matmul(batch_A, batch_B);  // Better than loop

// 2. Use appropriate precision
Matrix<float> weights;  // float is usually sufficient for ML

// 3. Minimize data transfers (GPU)
// Keep data on GPU as long as possible
auto gpu_result = gpu_matmul(gpu_A, gpu_B);  // Don't transfer back until needed

// 4. Use broadcasting instead of loops
auto broadcasted = matrix + vector;  // Better than manual loop
```

## Code Organization

```cpp
// Define model as class
class SimpleNN {
    Matrix<float> W1, W2;
    Vector<float> b1, b2;
    
public:
    SimpleNN(int input_dim, int hidden_dim, int output_dim)
        : W1(Matrix<float>::randn({input_dim, hidden_dim}, true))
        , b1(Vector<float>::zeros({hidden_dim}, true))
        , W2(Matrix<float>::randn({hidden_dim, output_dim}, true))
        , b2(Vector<float>::zeros({output_dim}, true)) {}
    
    Matrix<float> forward(const Matrix<float>& x) {
        auto h1 = matmul(x, W1) + b1;
        auto h1_act = h1.relu();
        auto out = matmul(h1_act, W2) + b2;
        return out;
    }
    
    std::vector<Tensor*> parameters() {
        return {&W1, &b1, &W2, &b2};
    }
};
```

## Common Pitfalls

```cpp
// ❌ DON'T: Forget to check errors
auto result = matmul(A, B);  // Might be TensorError!
result.print();  // Crash if error

// ✅ DO: Always check
if (std::holds_alternative<Matrix<float>>(result_var)) {
    auto result = std::get<Matrix<float>>(result_var);
    result.print();
}

// ❌ DON'T: Reshape incompatible sizes
auto bad = tensor.reshape({5, 7});  // If tensor has 24 elements

// ✅ DO: Use -1 for auto-inference
auto good = tensor.reshape({4, -1});  // Auto-computes second dim

// ❌ DON'T: Forget to zero gradients
for (int i = 0; i < epochs; i++) {
    loss.backward();
    optimizer.step();
    // Missing: optimizer.zero_grad()
}

// ✅ DO: Zero gradients each iteration
for (int i = 0; i < epochs; i++) {
    loss.backward();
    optimizer.step();
    optimizer.zero_grad();
}
```

---

**Previous**: [← Stacking and Concatenation](14-stacking-concatenation.md) | **Next**: [API Reference →](16-api-reference.md)
