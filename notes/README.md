# Tensor4D Python Bindings

Python bindings for the Tensor4D C++ library using pybind11.

## Features

- **Pure Python Interface**: No NumPy dependency required - works with native Python lists/tuples
- **Automatic Differentiation**: Built-in autograd for gradient computation
- **GPU Acceleration**: Automatic GPU support when CUDA is available
- **BLAS Optimization**: CPU acceleration via BLAS/LAPACK
- **Matrix Operations**: Full linear algebra support (matmul, inverse, det, etc.)
- **Loss Functions**: MSE, Cross Entropy, Binary Cross Entropy
- **Optimizers**: SGD, Adam, AdamW, RMSprop

## Installation

### Prerequisites

```bash
# Install pybind11
pip install pybind11

# Optional: Install development tools
sudo apt-get install python3-dev
```

### Build from source

```bash
cd python
python setup.py install
```

Or using pip in development mode:

```bash
cd python
pip install -e .
```

## Quick Start

### Basic Operations

```python
import tensor4d as t4d

# Create a matrix
m = t4d.Matrixf([3, 3])
m.fill(2.0)

# Create from Python list
data = [[1, 2], [3, 4]]
tensor = t4d.Matrixf(data)

# Arithmetic operations
result = tensor + tensor
result = tensor * 2.0

# Mathematical functions
exp_tensor = tensor.exp()
log_tensor = tensor.log()

# Convert to Python list
list_result = result.tolist()
```

### Autograd

```python
import tensor4d as t4d

# Create tensor with gradient tracking
x = t4d.Matrixf([[1.0, 2.0]])
x.set_requires_grad(True)

# Forward pass
y = x * x
loss = y.sum()

# Backward pass
loss.backward()

# Access gradients
grad = x.grad()
print(grad.tolist())
```

### Linear Algebra

```python
import tensor4d as t4d

# Matrix multiplication
A = t4d.Matrixf([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Identity matrix
B = t4d.Matrixf([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
C = A.matmul(B)

# Matrix operations
A_inv = A.inverse()
det_A = A.det()
trace_A = A.trace()
A_T = A.transpose()

# Vector operations
v1 = t4d.Vectorf([1, 2, 3])
v2 = t4d.Vectorf([4, 5, 6])
dot_product = v1.dot(v2)
norm = v1.norm()
```

### Training Example

```python
import tensor4d as t4d
import random

# Create parameters
W_data = [[random.gauss(0, 0.1) for _ in range(5)] for _ in range(10)]
W = t4d.Matrixf(W_data)
W.set_requires_grad(True)

# Create optimizer (pass parameters at construction)
optimizer = t4d.Adam([W], learning_rate=0.001)

# Training loop
for epoch in range(100):
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    X_data = [[random.gauss(0, 1) for _ in range(10)] for _ in range(10)]
    X = t4d.Matrixf(X_data)
    output = X.matmul(W)
    target = t4d.Matrixf([[1.0] * 5 for _ in range(10)])
    
    # Compute loss
    diff = output - target
    loss = diff * diff
    loss_val = loss.mean()
    
    # Update parameters (optimizer tracks them)
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss_val:.4f}")
```

### Optimizers

The library provides several optimization algorithms:

#### SGD (Stochastic Gradient Descent)

```python
# Create SGD optimizer with momentum
optimizer = t4d.SGD(
    [W, b],              # List of parameters to optimize
    learning_rate=0.01,  # Learning rate
    momentum=0.9,        # Momentum factor (default: 0.0)
    weight_decay=0.0001  # L2 regularization (default: 0.0)
)
```

#### Adam (Adaptive Moment Estimation)

```python
# Create Adam optimizer
optimizer = t4d.Adam(
    [W, b],                  # List of parameters
    learning_rate=0.001,     # Learning rate (default: 0.001)
    beta1=0.9,               # First moment decay (default: 0.9)
    beta2=0.999,             # Second moment decay (default: 0.999)
    epsilon=1e-8,            # Numerical stability (default: 1e-8)
    weight_decay=0.0         # L2 regularization (default: 0.0)
)

# Reset optimizer state (useful for multi-stage training)
optimizer.reset()
```

#### RMSprop

```python
# Create RMSprop optimizer
optimizer = t4d.RMSprop(
    [W, b],              # List of parameters
    learning_rate=0.01,  # Learning rate
    alpha=0.99,          # Smoothing constant (default: 0.99)
    epsilon=1e-8,        # Numerical stability (default: 1e-8)
    weight_decay=0.0,    # L2 regularization (default: 0.0)
    momentum=0.0         # Momentum factor (default: 0.0)
)
```

#### Learning Rate Scheduling

```python
# Create optimizer
optimizer = t4d.SGD([W], learning_rate=1.0)

# Create exponential decay scheduler
scheduler = t4d.ExponentialLR(optimizer, gamma=0.9)

# Training loop
for epoch in range(100):
    # ... training code ...
    optimizer.step()
    
    # Decay learning rate
    scheduler.step()
    
    print(f"LR: {optimizer.get_lr():.6f}")

# Reset to initial learning rate
scheduler.reset()
```

#### Optimizer Usage Pattern

```python
# 1. Create parameters with gradient tracking
W = t4d.Matrixf(np.random.randn(10, 5).astype(np.float32))
W.set_requires_grad(True)

# 2. Create optimizer with parameters
optimizer = t4d.Adam([W], learning_rate=0.001)

# 3. Training loop
for epoch in range(num_epochs):
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass (compute loss)
    loss = compute_loss(W)
    
    # Backward pass (compute gradients)
    loss.backward()
    
    # Update parameters
    optimizer.step()
    
    # Adjust learning rate (optional)
    if use_scheduler:
        scheduler.step()
```


## API Reference

### Tensor Types

- `Vectorf` / `Vectord`: 1D tensors (float/double)
- `Matrixf` / `Matrixd`: 2D tensors (float/double)
- `Tensor3f` / `Tensor3d`: 3D tensors (float/double)
- `Tensor4f` / `Tensor4d`: 4D tensors (float/double)

### Common Methods

#### Creation and Initialization
- `Tensor(shape)`: Create tensor with given shape
- `Tensor(list_or_tuple)`: Create from Python list/tuple
- `fill(value)`: Fill with scalar value
- `zeros()`: Fill with zeros
- `ones()`: Fill with ones

#### Element Access
- `tensor[index]`: Get/set element
- `at(index)`: Access element (alternative)

#### Arithmetic Operations
- `+`, `-`, `*`, `/`: Element-wise operations
- `+=`, `-=`, `*=`, `/=`: In-place operations

#### Mathematical Functions
- `exp()`, `log()`, `sqrt()`, `pow(n)`
- `sin()`, `cos()`, `tan()`
- `abs()`, `clip(min, max)`

#### Activation Functions
- `sigmoid()`, `tanh_()`, `relu()`, `leaky_relu(alpha)`

#### Statistical Operations
- `sum()`, `mean()`, `variance()`, `std()`
- `min()`, `max()`, `median()`, `prod()`
- `all()`, `any()`
- `argmin()`, `argmax()`
- `cumsum()`, `cumprod()`

#### Autograd
- `set_requires_grad(bool)`: Enable gradient tracking
- `backward()`: Compute gradients
- `grad()`: Get gradient tensor
- `zero_grad()`: Zero out gradients
- `detach()`: Detach from computation graph

#### Matrix Operations (2D tensors)
- `matmul(other)`: Matrix multiplication
- `transpose()`: Matrix transpose
- `inverse()`: Matrix inverse
- `det()`: Determinant
- `trace()`: Trace
- `diagonal()`: Get diagonal elements

#### Vector Operations (1D tensors)
- `dot(other)`: Dot product
- `norm()`: Euclidean norm

#### Conversion
- `tolist()`: Convert to Python list
- `shape`: Get tensor shape (property)
- `size`: Get total number of elements (property)

### Loss Functions

- `mse_loss(pred, target)`: Mean Squared Error
- `cross_entropy_loss(pred, target)`: Cross Entropy
- `binary_cross_entropy_loss(pred, target)`: Binary Cross Entropy

### Optimizers

#### SGD
```python
optimizer = t4d.SGD(
    learning_rate=0.01,
    momentum=0.0,
    weight_decay=0.0,
    nesterov=False
)
```

#### Adam
```python
optimizer = t4d.Adam(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8
)
```

### I/O Operations

```python
# Save tensor
t4d.save_tensor(tensor, "file.npy", t4d.TensorIOFormat.NPY)

# Load tensor
tensor = t4d.load_tensor_f2("file.npy")

# Formats: BINARY, TEXT, NPY
```

## Examples

See the `examples/` directory for complete examples:

- `example_basic.py`: Basic tensor operations
- `example_autograd.py`: Automatic differentiation
- `example_linalg.py`: Linear algebra operations
- `example_training.py`: Training with optimizers

## Performance Tips

1. **Use Python lists with correct type**: Python's `float` type works well for tensor4d
2. **Enable GPU if available**: The library automatically uses GPU when CUDA is available
3. **Batch operations**: Work with larger tensors to benefit from vectorization
4. **Reuse tensors**: Minimize allocations by reusing tensor objects
5. **Use in-place operations**: `+=`, `-=`, etc. when possible

## Troubleshooting

### ImportError: No module named 'tensor4d'

Make sure you've installed the module:
```bash
cd python
pip install -e .
```

### Compilation errors

Ensure you have:
- C++17 compatible compiler
- pybind11 installed
- Python development headers

### Runtime errors with GPU

Check if CUDA is properly installed:
```bash
nvidia-smi
nvcc --version
```

## License

Same as the main Tensor4D library.

## Contributing

Contributions are welcome! Please submit pull requests to the main repository.
