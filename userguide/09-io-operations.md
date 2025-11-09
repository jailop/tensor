# I/O Operations

## Saving and Loading Tensors

### Binary Format (Fast, Compact)

```cpp
#include "tensor_io.h"

auto data = Matrix<float>::randn({100, 50});

// Save in binary format (fastest, most compact)
data.save("data.bin");

// Load from binary
auto loaded_var = Matrix<float>::load("data.bin");
if (std::holds_alternative<Matrix<float>>(loaded_var)) {
    auto loaded = std::get<Matrix<float>>(loaded_var);
    loaded.print();
}
```

### Text Format (Human-Readable)

```cpp
// Save in text format (human-readable, larger file size)
data.save_txt("data.txt");

// Load from text
auto loaded_var = Matrix<float>::load_txt("data.txt");
auto loaded = std::get<Matrix<float>>(loaded_var);
```

### NumPy Format (.npy)

```cpp
// Save in NumPy .npy format (cross-language compatibility)
data.save_npy("data.npy");

// Load from .npy
auto loaded_var = Matrix<float>::load_npy("data.npy");
auto loaded = std::get<Matrix<float>>(loaded_var);
```

## NumPy Interoperability

### C++ → Python

```cpp
// Save tensor from C++
auto tensor = Matrix<float>::randn({50, 100});
tensor.save_npy("tensor.npy");
```

```python
# Load in Python/NumPy
import numpy as np
data = np.load("tensor.npy")
print(data.shape)  # (50, 100)
print(data.dtype)  # float32

# Process with NumPy
mean = data.mean()
std = data.std()
```

### Python → C++

```python
# Create and save in Python
import numpy as np
arr = np.random.randn(100, 50).astype(np.float32)
np.save("from_python.npy", arr)
```

```cpp
// Load in C++
auto tensor_var = Matrix<float>::load_npy("from_python.npy");
auto tensor = std::get<Matrix<float>>(tensor_var);
// Process with C++ library
auto result = tensor.mean();
```

## Format Comparison

| Format | Speed | Size | Human-Readable | Cross-Platform | NumPy Compatible |
|--------|-------|------|----------------|----------------|------------------|
| Binary | Fastest | Smallest | ❌ | ⚠️ (endianness) | ❌ |
| Text | Slowest | Largest | ✅ | ✅ | ❌ |
| NPY | Fast | Small | ❌ | ✅ | ✅ |

**Recommendation**: Use `.npy` format for most cases - it's fast, compact, and works with NumPy.

## Printing

```cpp
auto mat = Matrix<float>::randn({5, 5});

// Basic print to stdout
mat.print();

// Convert to string
std::string str = mat.to_string();
std::cout << str << std::endl;

// Format output
std::cout << "Matrix shape: " << mat.shape()[0] << "x" << mat.shape()[1] << std::endl;
mat.print();
```

Output example:
```
Matrix shape: 5x5
[[ 0.5377, -1.2141,  0.6715, -0.8230,  0.3426],
 [ 1.8339, -0.0301, -1.2075,  0.7924,  3.5784],
 [-2.2588,  0.4889,  0.7172, -0.3100,  2.7694],
 [ 0.8622,  1.0347, -0.1061,  0.4889, -1.3499],
 [ 0.3188, -0.7549,  1.5326,  1.0347,  3.0349]]
```

## Batch I/O Operations

### Saving Multiple Tensors

```cpp
// Save model parameters
auto W1 = Matrix<float>::randn({784, 256});
auto b1 = Vector<float>::randn({256});
auto W2 = Matrix<float>::randn({256, 10});
auto b2 = Vector<float>::randn({10});

W1.save_npy("model_W1.npy");
b1.save_npy("model_b1.npy");
W2.save_npy("model_W2.npy");
b2.save_npy("model_b2.npy");
```

### Loading Multiple Tensors

```cpp
// Load model parameters
auto W1_var = Matrix<float>::load_npy("model_W1.npy");
auto b1_var = Vector<float>::load_npy("model_b1.npy");
auto W2_var = Matrix<float>::load_npy("model_W2.npy");
auto b2_var = Vector<float>::load_npy("model_b2.npy");

auto W1 = std::get<Matrix<float>>(W1_var);
auto b1 = std::get<Vector<float>>(b1_var);
auto W2 = std::get<Matrix<float>>(W2_var);
auto b2 = std::get<Vector<float>>(b2_var);
```

## Error Handling

```cpp
auto result_var = Matrix<float>::load("nonexistent.npy");

if (std::holds_alternative<TensorError>(result_var)) {
    auto error = std::get<TensorError>(result_var);
    std::cerr << "Failed to load: " << error.message << std::endl;
    return;
}

auto data = std::get<Matrix<float>>(result_var);
// Use data...
```

---

**Previous**: [← Advanced Indexing](08-advanced-indexing.md) | **Next**: [Performance Optimization →](10-performance-optimization.md)
