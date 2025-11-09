# Python Integration

The tensor library provides comprehensive Python bindings through pybind11, offering a NumPy-like interface for all major features.

## Installation

### Building the Python Module

```bash
cd python
./build.sh
```

This will:
1. Build the C++ library with Python bindings
2. Create a `tensor` Python module
3. Install it in development mode

### Requirements

- Python 3.6+
- pybind11
- C++17 compatible compiler
- CMake 3.10+

## Basic Usage

### Creating Tensors

```python
import tensor

# Create tensors from lists (no NumPy dependency)
t1 = tensor.TensorFloat1D([1.0, 2.0, 3.0, 4.0])
t2 = tensor.TensorFloat2D([[1.0, 2.0], [3.0, 4.0]])

# Create from shape
t_zeros = tensor.zeros_float_2d([3, 4])
t_ones = tensor.ones_float_2d([2, 5])
t_rand = tensor.randn_float_2d([10, 10])

# Identity matrix
eye = tensor.eye_float([5, 5])
```

### Accessing Data

```python
# Convert to Python list
data = t1.to_list()
print(f"Data: {data}")

# Get shape
shape = t1.shape()
print(f"Shape: {shape}")

# Element access
value = t1.item([0])  # First element
t2_elem = t2.item([1, 1])  # Element at (1,1)
```

## NumPy Interoperability

The library provides seamless conversion between Tensor4D tensors and NumPy arrays, enabling integration with the broader Python scientific computing ecosystem.

### Converting Between NumPy and Tensor4D

```python
import tensor4d as t4d
import numpy as np

# Create NumPy array
np_data = np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]], dtype=np.float32)

# Method 1: Using from_numpy() class method
matrix1 = t4d.Matrixf.from_numpy(np_data)

# Method 2: Pass NumPy array to constructor (auto-detects)
matrix2 = t4d.Matrixf(np_data)

# Convert tensor back to NumPy
np_result = matrix1.numpy()
print(f"NumPy result:\n{np_result}")
```

### Working with NumPy Data

```python
# Load data with NumPy
raw_data = np.random.randn(100, 20).astype(np.float32)

# Preprocess in NumPy
normalized = (raw_data - raw_data.mean(axis=0)) / raw_data.std(axis=0)

# Convert to Tensor4D for model operations
tensor = t4d.Matrixf.from_numpy(normalized)

# Perform operations
scaled = tensor * 2.0
activated = scaled.sigmoid()

# Convert back to NumPy for analysis/saving
result = activated.numpy()
np.save('result.npy', result)
```

### NumPy Compatibility

All instantiated tensor types support NumPy conversion:

```python
# 1D tensors (Vectors)
v_float = t4d.Vectorf([1.0, 2.0, 3.0])
v_double = t4d.Vectord.from_numpy(np.array([1.0, 2.0]))

# 2D tensors (Matrices)
m_float = t4d.Matrixf([[1.0, 2.0], [3.0, 4.0]])
m_double = t4d.Matrixd.from_numpy(np.eye(3))

# 3D tensors
t3_float = t4d.Tensor3f.from_numpy(np.random.randn(2, 3, 4).astype(np.float32))
t3_double = t4d.Tensor3d.from_numpy(np.zeros((5, 5, 5)))

# 4D tensors
t4_float = t4d.Tensor4f.from_numpy(np.ones((2, 3, 4, 5), dtype=np.float32))
t4_double = t4d.Tensor4d.from_numpy(np.random.randn(2, 2, 2, 2))
```

### Operations Preserve NumPy Compatibility

```python
# Create tensors from NumPy
np_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
np_b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

a = t4d.Matrixf.from_numpy(np_a)
b = t4d.Matrixf.from_numpy(np_b)

# Perform operations in Tensor4D
c = a + b
d = a.matmul(b)
e = a.exp()

# Convert back to NumPy
np_c = c.numpy()
np_d = d.numpy()
np_e = e.numpy()

# Verify results match NumPy
assert np.allclose(np_c, np_a + np_b)
assert np.allclose(np_d, np_a @ np_b)
assert np.allclose(np_e, np.exp(np_a))
```

### Mathematical Functions with NumPy

```python
# All math functions preserve NumPy compatibility
np_input = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
vec = t4d.Vectorf.from_numpy(np_input)

# Apply functions
exp_result = vec.exp().numpy()
log_result = vec.log().numpy()
sqrt_result = vec.sqrt().numpy()
sin_result = vec.sin().numpy()

# Results match NumPy
assert np.allclose(exp_result, np.exp(np_input))
assert np.allclose(log_result, np.log(np_input))
assert np.allclose(sqrt_result, np.sqrt(np_input))
assert np.allclose(sin_result, np.sin(np_input))
```

### Integration Example

```python
import numpy as np
import tensor4d as t4d

# Load data with standard Python/NumPy tools
train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')

# Preprocessing with NumPy/SciPy
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)

# Convert to Tensor4D for training
X = t4d.Matrixf.from_numpy(train_data.astype(np.float32))
y = t4d.Vectorf.from_numpy(train_labels.astype(np.float32))

# Train model with Tensor4D
model = train_model(X, y)

# Convert predictions back to NumPy for evaluation
predictions = model.predict(X).numpy()

# Use NumPy/SciPy for metrics
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(train_labels, predictions > 0.5)
print(f"Accuracy: {accuracy:.4f}")
```

## Mathematical Operations

### Element-wise Operations

```python
import tensor

a = tensor.TensorFloat1D([1.0, 2.0, 3.0, 4.0])
b = tensor.TensorFloat1D([2.0, 3.0, 4.0, 5.0])

# Arithmetic (modifies left operand)
a.add_(b)      # a += b
a.sub_(b)      # a -= b
a.mul_(b)      # a *= b (element-wise)
a.div_(b)      # a /= b (element-wise)

# Non-modifying (returns new tensor)
c = a.add(b)   # c = a + b
d = a.sub(b)   # d = a - b
e = a.mul(b)   # e = a * b
f = a.div(b)   # f = a / b

# Math functions
exp_a = a.exp()
log_a = a.log()
sqrt_a = a.sqrt()
abs_a = a.abs()

# Trigonometric
sin_a = a.sin()
cos_a = a.cos()
tan_a = a.tan()

# Activation functions
sigmoid_a = a.sigmoid()
tanh_a = a.tanh()
relu_a = a.relu()
```

### Reductions

```python
# Statistical operations
total = a.sum()
average = a.mean()
variance = a.variance()
std_dev = a.std()
minimum = a.min()
maximum = a.max()

# Along specific axis
mat = tensor.randn_float_2d([100, 20])
col_means = mat.mean(0)  # Mean of each column
row_means = mat.mean(1)  # Mean of each row
```

## Linear Algebra

### Matrix Operations

```python
import tensor

# Matrix multiplication
A = tensor.randn_float_2d([100, 50])
B = tensor.randn_float_2d([50, 30])
C = tensor.matmul(A, B)  # 100x30

# Vector dot product
v1 = tensor.randn_float_1d([100])
v2 = tensor.randn_float_1d([100])
dot_prod = tensor.dot(v1, v2)

# Cross product (3D vectors)
a = tensor.TensorFloat1D([1.0, 0.0, 0.0])
b = tensor.TensorFloat1D([0.0, 1.0, 0.0])
c = tensor.cross(a, b)  # [0, 0, 1]

# Transpose
At = A.transpose()
```

### Decompositions

```python
# SVD
A = tensor.randn_float_2d([10, 5])
U, S, Vt = tensor.svd(A)

# QR
Q, R = tensor.qr(A)

# Cholesky (for SPD matrices)
A_spd = tensor.eye_float([5, 5])
A_spd = A_spd.add(tensor.ones_float_2d([5, 5]))
L = tensor.cholesky(A_spd)

# Eigenvalues
A_square = tensor.randn_float_2d([10, 10])
eigenvals = tensor.eigenvalues(A_square)
eigenvecs = tensor.eigenvectors(A_square)
```

### Solvers

```python
# Solve linear system Ax = b
A = tensor.randn_float_2d([10, 10])
b = tensor.randn_float_1d([10])

x_lu = tensor.solve_lu(A, b)
x_qr = tensor.solve_qr(A, b)
x_chol = tensor.solve_cholesky(A, b)  # For SPD matrices

# Least squares
A_rect = tensor.randn_float_2d([100, 10])
b_rect = tensor.randn_float_1d([100])
x_ls = tensor.lstsq(A_rect, b_rect)

# Matrix inverse
A_inv = tensor.inverse(A)

# Determinant
det = tensor.determinant(A)
```

## Automatic Differentiation

### Basic Autograd

```python
# Enable gradient tracking
x = tensor.TensorFloat1D([2.0, 3.0])
x.set_requires_grad(True)

# Forward pass
y = x.mul(x)  # y = x²
z = y.sum()   # z = sum(x²)

# Backward pass
z.backward()

# Access gradient
grad_x = x.grad()
print(f"Gradient: {grad_x.to_list()}")  # [4.0, 6.0]

# Clear gradients
x.zero_grad()
```

### Training Example

```python
import tensor

# Create data
X = tensor.randn_float_2d([64, 10])  # 64 samples, 10 features
y = tensor.randn_float_2d([64, 1])   # 64 labels

# Model parameters
W = tensor.randn_float_2d([10, 1])
b = tensor.zeros_float_2d([1, 1])
W.set_requires_grad(True)
b.set_requires_grad(True)

# Loss function
loss_fn = tensor.MSELoss()

# Optimizer
optimizer = tensor.Adam([W, b], lr=0.001)

# Training loop
for epoch in range(100):
    # Forward pass
    y_pred = tensor.matmul(X, W)
    y_pred = y_pred.add(b)
    
    # Compute loss
    loss = loss_fn.forward(y_pred, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Update parameters
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item([0])}")
```

## Loss Functions

```python
# Mean Squared Error
mse = tensor.MSELoss()
loss_mse = mse.forward(predictions, targets)

# Cross Entropy
ce = tensor.CrossEntropyLoss()
loss_ce = ce.forward(logits, targets)

# Binary Cross Entropy
bce = tensor.BinaryCrossEntropyLoss()
loss_bce = bce.forward(predictions, targets)

# L1 Loss (MAE)
l1 = tensor.L1Loss()
loss_l1 = l1.forward(predictions, targets)

# Smooth L1 Loss
smooth_l1 = tensor.SmoothL1Loss(beta=1.0)
loss_smooth = smooth_l1.forward(predictions, targets)
```

## Optimizers

```python
# Create parameters
params = [W, b]

# SGD with momentum
sgd = tensor.SGD(params, lr=0.01, momentum=0.9)

# Adam
adam = tensor.Adam(params, lr=0.001, beta1=0.9, beta2=0.999)

# AdamW (Adam with weight decay)
adamw = tensor.AdamW(params, lr=0.001, weight_decay=0.01)

# RMSprop
rmsprop = tensor.RMSprop(params, lr=0.001, alpha=0.99)

# Training step
optimizer.zero_grad()  # Clear gradients
loss.backward()        # Compute gradients
optimizer.step()       # Update parameters
```

## I/O Operations

### Saving and Loading

```python
# Save tensor (binary format)
t = tensor.randn_float_2d([100, 50])
t.save("tensor.bin")

# Load tensor
t_loaded = tensor.TensorFloat2D.load("tensor.bin")

# Save in NumPy format (.npy)
t.save_npy("tensor.npy")

# Load from .npy
t_from_npy = tensor.TensorFloat2D.load_npy("tensor.npy")

# Save as text
t.save_txt("tensor.txt")

# Load from text
t_from_txt = tensor.TensorFloat2D.load_txt("tensor.txt")
```

## Shape Manipulation

```python
# Reshape
t = tensor.randn_float_1d([24])
t_reshaped = t.reshape([4, 6])

# Flatten
t_flat = t_reshaped.flatten()

# Squeeze (remove dimensions of size 1)
t_squeezed = t.squeeze()

# Unsqueeze (add dimension of size 1)
t_unsqueezed = t.unsqueeze(0)

# Permute (transpose with custom order)
t_3d = tensor.randn_float_3d([2, 3, 4])
t_permuted = t_3d.permute([2, 0, 1])  # (4, 2, 3)
```

## Advanced Indexing

```python
# Fancy indexing
t = tensor.arange_float(0.0, 100.0, 1.0)
indices = tensor.TensorInt1D([5, 10, 15, 20])
selected = tensor.take(t, indices)

# Boolean masking
mask = tensor.TensorBool1D([True, False, True, False])
masked = tensor.masked_select(t, mask)

# Conditional operations
condition = t.gt(50.0)  # t > 50
result = tensor.where(condition, t, tensor.zeros_float_1d([100]))
```

## Random Operations

```python
# Set seed for reproducibility
tensor.seed(42)

# Distributions
uniform = tensor.rand_float_2d([10, 10])  # Uniform [0, 1)
normal = tensor.randn_float_2d([10, 10])  # Normal(0, 1)

# Random permutation
perm = tensor.randperm(100)

# Random choice
choices = tensor.choice_float(t, 10, replace=False)
```

## Normalization

```python
# L1 normalization
normalized_l1 = tensor.normalize_l1(t, axis=1)

# L2 normalization
normalized_l2 = tensor.normalize_l2(t, axis=1)

# Z-score normalization
normalized_z = tensor.normalize_zscore(t, axis=1, eps=1e-8)

# Min-max normalization
normalized_minmax = tensor.normalize_minmax(t, axis=1, min_val=0.0, max_val=1.0)
```

## Statistical Functions

```python
# Correlations
mat = tensor.randn_float_2d([100, 20])

# Pearson correlation
pearson = tensor.pearson_correlation(mat.col(0), mat.col(1))

# Covariance matrix
cov = tensor.covariance_matrix(mat)

# Quantiles
q25 = tensor.quantile(mat, 0.25, axis=0)
q50 = tensor.quantile(mat, 0.50, axis=0)  # Median
q75 = tensor.quantile(mat, 0.75, axis=0)
```

## Best Practices

### Memory Management

```python
# In-place operations save memory
a.add_(b)  # Modifies a in-place
# vs
c = a.add(b)  # Creates new tensor

# Detach from computation graph when not needed
x_detached = x.detach()

# Clear gradients regularly
optimizer.zero_grad()
```

### Error Handling

```python
# The library uses variants internally
# Python bindings handle errors through exceptions

try:
    result = tensor.inverse(singular_matrix)
except RuntimeError as e:
    print(f"Matrix inversion failed: {e}")
```

### Performance Tips

1. **Use in-place operations** when possible to reduce memory allocation
2. **Batch operations** instead of loops for better performance
3. **Enable GPU/BLAS** at compile time for acceleration
4. **Pre-allocate tensors** when sizes are known
5. **Use appropriate data types** (float32 vs float64)

## Complete Training Example

```python
import tensor

# Synthetic dataset
X_train = tensor.randn_float_2d([1000, 20])
y_train = tensor.randn_float_2d([1000, 1])

# Model
class LinearModel:
    def __init__(self, in_features, out_features):
        scale = (2.0 / in_features) ** 0.5
        self.W = tensor.randn_float_2d([in_features, out_features])
        self.W = self.W.mul(tensor.full_float_2d([in_features, out_features], scale))
        self.b = tensor.zeros_float_2d([1, out_features])
        self.W.set_requires_grad(True)
        self.b.set_requires_grad(True)
    
    def forward(self, x):
        out = tensor.matmul(x, self.W)
        return out.add(self.b)
    
    def parameters(self):
        return [self.W, self.b]

# Initialize
model = LinearModel(20, 1)
optimizer = tensor.Adam(model.parameters(), lr=0.001)
loss_fn = tensor.MSELoss()

# Training
batch_size = 32
epochs = 100

for epoch in range(epochs):
    epoch_loss = 0.0
    
    # Mini-batch training
    for i in range(0, 1000, batch_size):
        # Get batch (simplified - in practice, use proper batching)
        X_batch = X_train  # Use actual batching logic
        y_batch = y_train
        
        # Forward
        y_pred = model.forward(X_batch)
        loss = loss_fn.forward(y_pred, y_batch)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item([0])
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
```

## Interoperability Notes

The Python bindings are designed to work **without NumPy dependency**:

- All tensor creation uses Python lists
- Data extraction returns Python lists
- No implicit NumPy conversions

This makes the library lightweight and suitable for environments where NumPy is not available or desired.

For NumPy interoperability, use the `.npy` file format:

```python
# Save from Python
t = tensor.randn_float_2d([100, 50])
t.save_npy("data.npy")

# Load in NumPy (separate script)
import numpy as np
arr = np.load("data.npy")
```

---

**Previous**: [← API Reference](16-api-reference.md) | **Next**: [C Interface →](18-c-interface.md) | **Up**: [Index ↑](00-index.md)
