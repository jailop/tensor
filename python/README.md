# Tensor4D Python Bindings

Python bindings for the Tensor4D C++ tensor library, providing high-performance tensor operations with a NumPy-like interface.

## Features

- **NumPy Interoperability**: Seamless conversion between Tensor4D and NumPy arrays
- **High Performance**: GPU acceleration (CUDA) and BLAS support when available
- **Automatic Differentiation**: Built-in autograd for gradient computation
- **Type Safety**: Strongly-typed tensors (float32, float64)
- **Rich API**: Comprehensive mathematical, statistical, and linear algebra operations

## Installation

### Build from Source

```bash
cd python
./build.sh
```

Requirements:
- Python 3.6+
- pybind11
- NumPy (optional, for interoperability features)
- C++17 compatible compiler
- CMake 3.10+

## Quick Start

### Basic Operations

```python
import tensor4d as t4d

# Create tensors
v = t4d.Vectorf([1.0, 2.0, 3.0, 4.0])
m = t4d.Matrixf([[1.0, 2.0], [3.0, 4.0]])

# Element-wise operations
result = v * 2.0
squared = v * v

# Math functions
exp_v = v.exp()
log_v = v.log()
sigmoid_v = v.sigmoid()

# Statistical operations
mean = v.mean()
std = v.std()
total = v.sum()
```

### NumPy Integration

```python
import numpy as np
import tensor4d as t4d

# Create from NumPy array
np_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
tensor = t4d.Matrixf.from_numpy(np_array)

# Or use constructor directly
tensor2 = t4d.Matrixf(np_array)

# Perform operations
result = tensor + tensor2
product = tensor.matmul(tensor2)

# Convert back to NumPy
np_result = result.numpy()

# Use with NumPy ecosystem
import matplotlib.pyplot as plt
plt.imshow(np_result)
plt.show()
```

### Linear Algebra

```python
# Matrix operations
A = t4d.Matrixf.from_numpy(np.random.randn(10, 10).astype(np.float32))
B = t4d.Matrixf.from_numpy(np.random.randn(10, 5).astype(np.float32))

# Matrix multiplication
C = A.matmul(B)

# Transpose
At = A.transpose()

# Decompositions (wraps LAPACK/cuBLAS)
U, S, Vt = t4d.svd(A)
Q, R = t4d.qr(A)

# Solve linear systems
b = t4d.Vectorf.from_numpy(np.random.randn(10).astype(np.float32))
x = t4d.solve_lu(A, b)

# Matrix properties
det = t4d.determinant(A)
inv = t4d.inverse(A)
rank = A.rank()
```

### Automatic Differentiation

```python
# Enable gradient tracking
x = t4d.Vectorf([2.0, 3.0, 4.0])
x.set_requires_grad(True)

# Forward pass
y = x * x  # y = x²
z = y.sum()  # z = sum(x²)

# Backward pass
z.backward()

# Get gradients
if x.grad():
    grad_np = x.grad().numpy()
    print(f"Gradients: {grad_np}")
```

## API Reference

### Tensor Types

| Type | Description | NumPy Equivalent |
|------|-------------|------------------|
| `Vectorf` | 1D float32 tensor | `np.ndarray` with `dtype=np.float32` |
| `Vectord` | 1D float64 tensor | `np.ndarray` with `dtype=np.float64` |
| `Matrixf` | 2D float32 tensor | 2D `np.ndarray` with `dtype=np.float32` |
| `Matrixd` | 2D float64 tensor | 2D `np.ndarray` with `dtype=np.float64` |
| `Tensor3f` | 3D float32 tensor | 3D `np.ndarray` with `dtype=np.float32` |
| `Tensor3d` | 3D float64 tensor | 3D `np.ndarray` with `dtype=np.float64` |
| `Tensor4f` | 4D float32 tensor | 4D `np.ndarray` with `dtype=np.float32` |
| `Tensor4d` | 4D float64 tensor | 4D `np.ndarray` with `dtype=np.float64` |

### NumPy Interoperability

```python
# Create from NumPy
tensor = TensorType.from_numpy(np_array)
tensor = TensorType(np_array)  # Auto-detects NumPy arrays

# Convert to NumPy
np_array = tensor.numpy()

# Convert to Python list
py_list = tensor.tolist()
```

### Mathematical Operations

```python
# Element-wise operations
result = a + b    # Addition
result = a - b    # Subtraction
result = a * b    # Multiplication
result = a / b    # Division

# In-place operations
a += b
a -= b
a *= b
a /= b

# Math functions
tensor.exp()      # Exponential
tensor.log()      # Natural logarithm
tensor.sqrt()     # Square root
tensor.pow(n)     # Power
tensor.abs()      # Absolute value
tensor.sin()      # Sine
tensor.cos()      # Cosine
tensor.tan()      # Tangent
tensor.clip(min, max)  # Clip values

# Activation functions
tensor.sigmoid()  # Sigmoid activation
tensor.relu()     # ReLU activation
tensor.tanh()     # Tanh activation
```

### Statistical Operations

```python
tensor.sum()      # Sum of all elements
tensor.mean()     # Mean
tensor.variance() # Variance
tensor.std()      # Standard deviation
tensor.min()      # Minimum value
tensor.max()      # Maximum value
tensor.median()   # Median
tensor.prod()     # Product of all elements
```

## Examples

See the `examples/` directory for comprehensive examples:

- `example_basic.py` - Basic tensor operations
- `example_numpy_interop.py` - NumPy interoperability
- `example_linalg.py` - Linear algebra operations
- `example_autograd.py` - Automatic differentiation
- `example_training.py` - Training a simple neural network

## Testing

Run the test suite:

```bash
python3 test_numpy_interop.py
python3 test_bindings.py
```

## Performance

The library automatically uses:
1. **GPU (CUDA)** - If available and data is on GPU
2. **BLAS (OpenBLAS/MKL)** - If available for CPU operations
3. **CPU fallback** - Optimized CPU implementations

Performance tips:
- Use float32 for better GPU performance
- Keep data on GPU when doing multiple operations
- Use in-place operations (`+=`, `-=`, etc.) to reduce memory allocations
- Batch operations when possible

## Integration with Python Ecosystem

The NumPy interoperability enables seamless integration with:

- **NumPy**: Universal array interface
- **Pandas**: Data manipulation (via NumPy)
- **Matplotlib**: Visualization
- **Scikit-learn**: Machine learning preprocessing/evaluation
- **SciPy**: Scientific computing
- **OpenCV**: Computer vision (via NumPy arrays)

## License

See the main project LICENSE file.

## Documentation

Full documentation is available in the `userguide/` directory, particularly:
- `17-python-integration.md` - Comprehensive Python integration guide

## Support

For issues, questions, or contributions, please refer to the main project repository.
