# Performance Optimization

## Backend Selection

The library automatically selects the best backend:
1. **GPU** (if available) - CUDA acceleration
2. **BLAS** (if available) - Optimized CPU operations
3. **CPU** - Fallback implementation

Configure at compile time:
```bash
cmake -DUSE_GPU=ON -DUSE_BLAS=ON ..
```

## Memory Pooling

```cpp
#include "tensor_perf.h"

// Enable memory pooling
TensorMemoryPool::enable(true);
TensorMemoryPool::set_block_size(1024 * 1024);  // 1MB blocks

// Operations now reuse memory
for (int i = 0; i < 1000; i++) {
    auto temp = Matrix<float>::randn({100, 100});
    // Memory is pooled, not allocated/freed each time
}
```

## Multi-threading

```cpp
// Parallel operations (automatically used when beneficial)
auto large_mat = Matrix<float>::randn({10000, 10000});
auto result = large_mat + 1.0f;  // Uses multi-threading internally
```

## Mixed Precision

```cpp
// FP16 for memory savings
Tensor<half_t, 2> fp16_tensor({1000, 1000});

// Convert between precisions
auto fp32_var = fp16_tensor.astype<float>();
```

## Lazy Evaluation

```cpp
// Enable lazy evaluation for operation fusion
auto A = Matrix<float>::randn({1000, 1000});
auto B = Matrix<float>::randn({1000, 1000});
auto C = Matrix<float>::randn({1000, 1000});

// These operations can be fused
auto result_var = (A + B) * C - 1.0f;  // Single kernel launch on GPU
```

---

**Previous**: [← I/O Operations](09-io-operations.md) | **Next**: [Normalization and Views →](11-normalization-views.md)
