# I/O Operations

## Saving and Loading Tensors

```cpp
#include "tensor_io.h"

auto data = Matrix<float>::randn({100, 50});

// Save in binary format (fastest)
save_tensor(data, "data.bin");

// Load from binary
auto loaded_var = load_tensor<float, 2>("data.bin");
auto loaded = std::get<Matrix<float>>(loaded_var);

// Save in text format (human-readable)
save_tensor(data, "data.txt", SaveFormat::Text);

// Save in NumPy format (.npy)
save_tensor(data, "data.npy", SaveFormat::NPY);
```

## NumPy Interoperability

```cpp
// Save for use in Python/NumPy
auto tensor = Matrix<float>::randn({50, 100});
save_tensor(tensor, "tensor.npy", SaveFormat::NPY);
```

Python side:
```python
import numpy as np
data = np.load("tensor.npy")
print(data.shape)  # (50, 100)
```

## Printing

```cpp
auto mat = Matrix<float>::randn({5, 5});

// Basic print
mat.print();

// Convert to string
std::string str = mat.to_string();
std::cout << str << std::endl;
```

---

**Previous**: [← Advanced Indexing](08-advanced-indexing.md) | **Next**: [Performance Optimization →](10-performance-optimization.md)
