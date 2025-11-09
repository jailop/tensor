# Getting Started

## Installation

### Prerequisites

- **C++ Compiler**: C++17 or later (GCC 7+, Clang 5+, MSVC 2017+)
- **CMake**: Version 3.10 or later
- **Optional Dependencies**:
  - CUDA Toolkit (for GPU support)
  - BLAS library (OpenBLAS, Intel MKL, or Apple Accelerate)
  - Doxygen (for documentation generation)
  - Google Test (for running tests)

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/tensor-library.git
cd tensor-library

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Optional: Enable GPU support
cmake -DUSE_GPU=ON ..

# Optional: Enable BLAS support
cmake -DUSE_BLAS=ON ..

# Build the library
make -j$(nproc)

# Run tests to verify installation
make test
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_GPU` | OFF | Enable CUDA GPU acceleration |
| `USE_BLAS` | OFF | Enable BLAS optimization |
| `BUILD_SHARED_LIBS` | ON | Build shared library (.so) |
| `BUILD_TESTS` | ON | Build test suite |
| `BUILD_DOCS` | ON | Generate Doxygen documentation |

### Linking Against the Library

#### Using CMake

```cmake
# In your CMakeLists.txt
find_package(Tensor REQUIRED)

add_executable(myapp main.cc)
target_link_libraries(myapp Tensor::tensor)
```

#### Manual Linking

```bash
# Compile
g++ -std=c++17 -I/path/to/tensor/include myapp.cc -o myapp \
    -L/path/to/tensor/build -ltensor

# Run (you may need to set LD_LIBRARY_PATH)
export LD_LIBRARY_PATH=/path/to/tensor/build:$LD_LIBRARY_PATH
./myapp
```

## Basic Concepts

### Tensor

A **Tensor** is a multi-dimensional array with:
- **Type**: Data type (float, double, int, etc.)
- **Rank**: Number of dimensions (1D vector, 2D matrix, 3D cube, etc.)
- **Shape**: Size of each dimension

```cpp
// Tensor<float, 2> means:
// - Type: float
// - Rank: 2 (a matrix)
// - Shape: specified at runtime, e.g., {3, 4} for 3 rows, 4 columns
```

### Type Aliases

For convenience, the library provides type aliases:

```cpp
// Matrix: 2D tensor
Matrix<float> m = Matrix<float>::zeros({3, 4});  // 3x4 matrix

// Vector: 1D tensor
Vector<float> v = Vector<float>::ones({5});      // 5-element vector

// General tensors
Tensor<float, 3> cube({2, 3, 4});                // 2x3x4 tensor
```

### Error Handling

The library uses `std::variant` for error handling instead of exceptions:

```cpp
#include "tensor.h"
#include <variant>

// Operations return TensorResult<T> = variant<T, TensorError>
TensorResult<Tensor<float, 2>> result = matmul(A, B);

// Check for errors
if (std::holds_alternative<TensorError>(result)) {
    auto error = std::get<TensorError>(result);
    std::cerr << "Error: " << error.message << std::endl;
    return;
}

// Extract successful result
Tensor<float, 2> C = std::get<Tensor<float, 2>>(result);
```

### Backend Selection

The library automatically selects the best available backend:

1. **GPU** (if `USE_GPU` enabled and GPU available) - Highest performance
2. **BLAS** (if `USE_BLAS` enabled) - Optimized CPU
3. **CPU** (fallback) - Pure C++ implementation

No code changes needed - the library handles this automatically!

## Hello World Example

### Example 1: Basic Tensor Operations

```cpp
#include "tensor.h"
#include <iostream>

int main() {
    // Create a 3x3 matrix filled with ones
    auto A = Matrix<float>::ones({3, 3});
    
    // Create another matrix with a specific value
    auto B = Matrix<float>::full({3, 3}, 2.0f);
    
    // Add them together
    auto C_var = A + B;
    
    // Check for errors
    if (std::holds_alternative<TensorError>(C_var)) {
        std::cerr << "Error!" << std::endl;
        return 1;
    }
    
    // Extract result
    auto C = std::get<Matrix<float>>(C_var);
    
    // Print the result
    std::cout << "A + B = " << std::endl;
    C.print();
    
    // Access elements
    std::cout << "C[0,0] = " << C[{0, 0}] << std::endl;
    std::cout << "C[1,1] = " << C[{1, 1}] << std::endl;
    
    return 0;
}
```

Output:
```
A + B = 
[[3, 3, 3],
 [3, 3, 3],
 [3, 3, 3]]
C[0,0] = 3
C[1,1] = 3
```

### Example 2: Matrix Multiplication

```cpp
#include "tensor.h"
#include "linalg.h"
#include <iostream>

int main() {
    // Create two matrices
    auto A = Matrix<float>::randn({3, 4});  // 3x4 matrix, random normal
    auto B = Matrix<float>::randn({4, 2});  // 4x2 matrix
    
    // Matrix multiplication
    auto C_var = matmul(A, B);
    
    if (std::holds_alternative<Matrix<float>>(C_var)) {
        auto C = std::get<Matrix<float>>(C_var);
        
        std::cout << "Matrix A (3x4):" << std::endl;
        A.print();
        
        std::cout << "\nMatrix B (4x2):" << std::endl;
        B.print();
        
        std::cout << "\nResult C = A @ B (3x2):" << std::endl;
        C.print();
    }
    
    return 0;
}
```

### Example 3: Automatic Differentiation

```cpp
#include "tensor.h"
#include <iostream>

int main() {
    // Create a tensor that requires gradients
    auto x = Tensor<float, 1>::from_array({2.0f}, {1}, true);  // requires_grad=true
    
    // Compute y = x^2
    auto y_var = x * x;
    auto y = std::get<Tensor<float, 1>>(y_var);
    
    // Backward pass: compute dy/dx = 2x
    y.backward();
    
    // Get gradient
    if (x.grad()) {
        std::cout << "x = " << x[{0}] << std::endl;
        std::cout << "y = x^2 = " << y[{0}] << std::endl;
        std::cout << "dy/dx = 2x = " << (*x.grad())[{0}] << std::endl;
    }
    
    return 0;
}
```

Output:
```
x = 2
y = x^2 = 4
dy/dx = 2x = 4
```

## Compiling Your First Program

```bash
# If you've installed the library:
g++ -std=c++17 hello_world.cc -o hello_world -ltensor

# If using from source tree:
g++ -std=c++17 -I../include hello_world.cc -o hello_world \
    -L../build -ltensor -Wl,-rpath,../build

./hello_world
```

## Next Steps

Now that you have the library installed and running:

1. **Learn core operations**: [Core Tensor Operations →](02-core-tensor-operations.md)
2. **Explore examples**: Check the `tests/` directory for comprehensive examples
3. **Read API docs**: Generated Doxygen documentation in `docs/html/`
4. **Performance tuning**: [Performance Optimization →](10-performance-optimization.md)

## Troubleshooting

### Common Issues

**1. Link errors about CUDA or BLAS**

If you get undefined references to CUDA or BLAS functions, make sure:
- You've enabled the corresponding CMake option (`USE_GPU` or `USE_BLAS`)
- The libraries are installed and findable by CMake
- You're linking against the same configuration you built

**2. "Cannot find tensor.h"**

Make sure you're including the header search path:
```bash
-I/path/to/tensor/include
```

**3. Runtime errors about missing .so file**

Set `LD_LIBRARY_PATH`:
```bash
export LD_LIBRARY_PATH=/path/to/tensor/build:$LD_LIBRARY_PATH
```

Or use `-Wl,-rpath` during compilation.

## Getting Help

- **Documentation**: Check the [API Reference](16-api-reference.md)
- **Examples**: Browse test files in `tests/`
- **Issues**: Report bugs on GitHub
- **Community**: [Your community links]

---

**Previous**: [← Index](00-index.md) | **Next**: [Core Tensor Operations →](02-core-tensor-operations.md)
