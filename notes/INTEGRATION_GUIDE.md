# Python Bindings Integration Guide

## Overview

This directory contains Python bindings for the Tensor4D C++ library using pybind11. The bindings provide a NumPy-like interface with automatic differentiation, GPU acceleration, and seamless interoperability with NumPy arrays.

## Directory Structure

```
python/
├── tensor_wrapper.cc       # Main pybind11 bindings
├── setup.py               # Main setup script (CMake-based)
├── setup_simple.py        # Alternative simple setup
├── CMakeLists.txt         # CMake configuration
├── build.sh               # Convenience build script
├── README.md              # User documentation
├── __init__.py            # Python package initialization
├── test_bindings.py       # Unit tests
└── examples/
    ├── example_basic.py
    ├── example_autograd.py
    ├── example_linalg.py
    ├── example_numpy_interop.py
    └── example_training.py
```

## Installation Methods

### Method 1: Using build.sh (Recommended)

```bash
cd python
./build.sh
```

This will:
1. Check for dependencies (pybind11, NumPy)
2. Install missing dependencies if needed
3. Build the extension module
4. Run basic tests

### Method 2: Using pip (Development Mode)

```bash
cd python
pip install -e .
```

This installs in "editable" mode, allowing you to modify the C++ code and rebuild without reinstalling.

### Method 3: Manual Build

```bash
cd python
python3 setup.py build_ext --inplace
```

### Method 4: System-Wide Installation

```bash
cd python
python3 setup.py install
```

## Dependencies

### Required

- **Python 3.7+**
- **pybind11 >= 2.6.0**: `pip install pybind11`
- **NumPy >= 1.19.0**: `pip install numpy`
- **C++17 compiler**: GCC 7+, Clang 5+, or MSVC 2017+

### Optional (for acceleration)

- **CUDA Toolkit**: For GPU support
- **BLAS/LAPACK**: For CPU acceleration (OpenBLAS, MKL, etc.)

## Building with Optional Features

### With BLAS Support

The build system automatically detects BLAS/LAPACK if installed:

```bash
# Ubuntu/Debian
sudo apt-get install libblas-dev liblapack-dev

# Then build normally
cd python
./build.sh
```

### With CUDA Support

Ensure CUDA is installed and visible to CMake:

```bash
export CUDACXX=/usr/local/cuda/bin/nvcc
cd python
./build.sh
```

## Testing

### Run Unit Tests

```bash
cd python
python3 test_bindings.py
```

### Run Examples

```bash
cd python
python3 example_basic.py
python3 example_autograd.py
python3 example_linalg.py
python3 example_numpy_interop.py
python3 example_training.py
```

## Usage in Python

### Basic Import

```python
import tensor4d as t4d
import numpy as np

# Create a matrix
m = t4d.Matrixf([3, 3])
m.fill(1.0)

# Convert to NumPy
np_array = m.numpy()
```

### Full Example

```python
import tensor4d as t4d
import numpy as np

# Create tensors from NumPy
data = np.random.randn(10, 5).astype(np.float32)
weights = np.random.randn(5, 3).astype(np.float32) * 0.1

# Convert to Tensor4D
X = t4d.Matrixf(data)
W = t4d.Matrixf(weights)
W.set_requires_grad(True)

# Forward pass
output = X.matmul(W)
output_act = output.relu()
loss = output_act.sum()

# Backward pass
loss.backward()

# Get gradients
if W.grad():
    grad_np = W.grad().numpy()
    print("Gradient shape:", grad_np.shape)
```

## API Design Philosophy

The Python API closely mirrors the C++ API with these enhancements:

1. **NumPy Interoperability**: Automatic conversion to/from NumPy arrays
2. **Buffer Protocol**: Zero-copy access via Python buffer protocol
3. **Pythonic Interface**: Using properties, `__repr__`, operator overloading
4. **Error Handling**: C++ exceptions converted to Python exceptions
5. **Type Safety**: Separate classes for different tensor types

## Wrapped Components

### Tensor Types

- `Vectorf`, `Vectord` (1D)
- `Matrixf`, `Matrixd` (2D)
- `Tensor3f`, `Tensor3d` (3D)
- `Tensor4f`, `Tensor4d` (4D)

### Operations

- Arithmetic: `+`, `-`, `*`, `/`, `+=`, `-=`, `*=`, `/=`
- Math functions: `exp`, `log`, `sqrt`, `pow`, `sin`, `cos`, `tan`, `abs`
- Activation: `sigmoid`, `tanh_`, `relu`, `leaky_relu`
- Statistical: `sum`, `mean`, `variance`, `std`, `min`, `max`, `median`
- Reduction: `all`, `any`, `argmin`, `argmax`, `cumsum`, `cumprod`
- Linear algebra: `matmul`, `transpose`, `inverse`, `det`, `trace`, `diagonal`
- Vector ops: `dot`, `norm`

### Autograd

- `set_requires_grad(bool)`
- `backward()`
- `grad()`
- `zero_grad()`
- `detach()`

### Loss Functions

- `mse_loss(pred, target)`
- `cross_entropy_loss(pred, target)`
- `binary_cross_entropy_loss(pred, target)`

### Optimizers

- `SGD(lr, momentum, weight_decay, nesterov)`
- `Adam(lr, beta1, beta2, epsilon)`

### I/O

- `save_tensor(tensor, filename, format)`
- `load_tensor_f2(filename)`
- Formats: `BINARY`, `TEXT`, `NPY`

## Performance Considerations

1. **Data Type Matching**: Always use `np.float32` or `np.float64` to match C++ types
2. **Batch Operations**: Larger tensors benefit more from GPU/BLAS acceleration
3. **Memory Layout**: Tensors use row-major (C-style) ordering like NumPy
4. **Gradient Accumulation**: Gradients accumulate; call `zero_grad()` between iterations

## Troubleshooting

### ImportError: dynamic module does not define module export function

The module wasn't properly compiled. Try:
```bash
cd python
python3 setup.py clean --all
./build.sh
```

### Symbol not found errors

Missing library dependencies. Check that BLAS/CUDA libraries are in your library path:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### NumPy version mismatch

Ensure NumPy is compiled with the same Python version:
```bash
pip3 install --upgrade numpy
```

### Permission errors during installation

Use user installation:
```bash
pip3 install --user -e .
```

## Integration with Existing Projects

### As a Python Package

```python
# your_project.py
import tensor4d as t4d
import numpy as np

def train_model(data, labels):
    # Your training code using tensor4d
    pass
```

### With PyTorch/TensorFlow

Tensor4D can complement PyTorch/TensorFlow by providing:
- High-performance custom operations
- Lightweight inference without heavy frameworks
- Educational purposes for understanding autodiff

```python
import numpy as np
import torch
import tensor4d as t4d

# Convert PyTorch → Tensor4D
torch_tensor = torch.randn(5, 5)
np_array = torch_tensor.numpy()
t4d_tensor = t4d.Matrixf(np_array.astype(np.float32))

# Operate with Tensor4D
result = t4d_tensor.exp()

# Convert back to PyTorch
result_torch = torch.from_numpy(result.numpy())
```

## Advanced Topics

### Custom Type Bindings

To add bindings for new tensor types, edit `tensor_wrapper.cc`:

```cpp
// Add integer matrix support
bind_tensor<int, 2>(m, "Matrixi");
```

### Extending the API

Add new methods in `tensor_wrapper.cc`:

```cpp
.def("my_custom_op", &TensorType::my_custom_op, 
     py::arg("param"), "Documentation")
```

### Performance Profiling

Use Python profilers with tensor4d:

```python
import cProfile
import tensor4d as t4d

def benchmark():
    m = t4d.Matrixf([1000, 1000])
    m.fill(1.0)
    result = m.exp().sum()
    return result

cProfile.run('benchmark()')
```

## Maintenance

### Updating Bindings

When C++ API changes:
1. Update `tensor_wrapper.cc`
2. Update examples and tests
3. Rebuild: `./build.sh`
4. Run tests: `python3 test_bindings.py`

### Version Management

Version is defined in:
- `setup.py`: Python package version
- `tensor_wrapper.cc`: `m.attr("__version__")`
- `__init__.py`: Module-level `__version__`

Keep these synchronized.

## Contributing

When adding new bindings:
1. Add C++ binding in `tensor_wrapper.cc`
2. Add Python example in `example_*.py`
3. Add unit test in `test_bindings.py`
4. Update `README.md` with API documentation
5. Test on multiple platforms if possible

## License

Same as the main Tensor4D library.

## Support

For issues specific to Python bindings:
- Check examples in `python/` directory
- Run `python3 test_bindings.py` to verify installation
- Ensure all dependencies are properly installed
- Check that C++ library builds correctly standalone first

For general library issues, refer to the main project documentation.
