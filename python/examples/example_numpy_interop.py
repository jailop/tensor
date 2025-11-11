#!/usr/bin/env python3
"""
Example: NumPy Interoperability with Tensor4D

This example demonstrates seamless conversion between Tensor4D tensors
and NumPy arrays, enabling integration with the broader Python scientific
computing ecosystem.
"""

import numpy as np
import tensor4d as t4d

print("=" * 70)
print("NumPy Interoperability Example")
print("=" * 70 + "\n")

# ============================================================================
# 1. Creating Tensors from NumPy Arrays
# ============================================================================
print("1. Creating Tensors from NumPy Arrays")
print("-" * 70)

# Create a NumPy array
np_data = np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]], dtype=np.float32)
print(f"Original NumPy array:\n{np_data}\n")

# Method 1: Using from_numpy() class method
matrix1 = t4d.Matrixf.from_numpy(np_data)
print(f"Tensor from from_numpy():")
print(f"  Shape: {matrix1.shape}")
print(f"  Data: {matrix1.tolist()}\n")

# Method 2: Passing NumPy array to constructor
matrix2 = t4d.Matrixf(np_data)
print(f"Tensor from constructor:")
print(f"  Shape: {matrix2.shape}")
print(f"  Data: {matrix2.tolist()}\n")

# ============================================================================
# 2. Converting Tensors to NumPy Arrays
# ============================================================================
print("2. Converting Tensors to NumPy Arrays")
print("-" * 70)

# Create a tensor
tensor_data = [[7.0, 8.0], [9.0, 10.0]]
tensor = t4d.Matrixf(tensor_data)
print(f"Original tensor:\n{tensor.tolist()}\n")

# Convert to NumPy
np_result = tensor.numpy()
print(f"Converted to NumPy:\n{np_result}")
print(f"NumPy dtype: {np_result.dtype}\n")

# ============================================================================
# 3. Interoperability with NumPy Operations
# ============================================================================
print("3. Interoperability with NumPy Operations")
print("-" * 70)

# Create data in NumPy
np_a = np.random.randn(3, 3).astype(np.float32)
np_b = np.random.randn(3, 3).astype(np.float32)

print(f"NumPy matrix A:\n{np_a}\n")
print(f"NumPy matrix B:\n{np_b}\n")

# Convert to Tensor4D for operations
a = t4d.Matrixf.from_numpy(np_a)
b = t4d.Matrixf.from_numpy(np_b)

# Perform operations in Tensor4D
c = a + b
d = a * b  # Element-wise
e = a.matmul(b)  # Matrix multiplication

# Convert back to NumPy for comparison
np_c = c.numpy()
np_d = d.numpy()
np_e = e.numpy()

print(f"Tensor4D addition result:\n{np_c}\n")
print(f"NumPy verification (A + B):\n{np_a + np_b}\n")
print(f"Match: {np.allclose(np_c, np_a + np_b)}\n")

print(f"Tensor4D element-wise multiplication:\n{np_d}\n")
print(f"NumPy verification (A * B):\n{np_a * np_b}\n")
print(f"Match: {np.allclose(np_d, np_a * np_b)}\n")

print(f"Tensor4D matrix multiplication:\n{np_e}\n")
print(f"NumPy verification (A @ B):\n{np_a @ np_b}\n")
print(f"Match: {np.allclose(np_e, np_a @ np_b)}\n")

# ============================================================================
# 4. Mathematical Functions
# ============================================================================
print("4. Mathematical Functions")
print("-" * 70)

# Create test data
np_input = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
vec = t4d.Vectorf.from_numpy(np_input)

print(f"Input: {np_input}\n")

# Apply functions and compare
functions = [
    ('exp', lambda x: x.exp(), np.exp),
    ('log', lambda x: x.log(), np.log),
    ('sqrt', lambda x: x.sqrt(), np.sqrt),
    ('sin', lambda x: x.sin(), np.sin),
    ('cos', lambda x: x.cos(), np.cos),
]

for name, t4d_func, np_func in functions:
    t4d_result = t4d_func(vec).numpy()
    np_result = np_func(np_input)
    match = np.allclose(t4d_result, np_result, rtol=1e-5)
    print(f"{name:8s}: Tensor4D={t4d_result}, NumPy={np_result}, Match={match}")

print()

# ============================================================================
# 5. Working with Higher-Dimensional Tensors
# ============================================================================
print("5. Working with Higher-Dimensional Tensors")
print("-" * 70)

# Create a 3D tensor in NumPy (e.g., a batch of images)
np_3d = np.random.randn(2, 3, 4).astype(np.float64)
print(f"NumPy 3D tensor shape: {np_3d.shape}")

# Convert to Tensor4D
tensor_3d = t4d.Tensor3d.from_numpy(np_3d)
print(f"Tensor3d shape: {tensor_3d.shape}")

# Apply operations
scaled = tensor_3d * 2.0
normalized = tensor_3d - tensor_3d.mean()

# Convert back
np_scaled = scaled.numpy()
np_normalized = normalized.numpy()

print(f"Scaled shape: {np_scaled.shape}")
print(f"Normalized shape: {np_normalized.shape}")
print(f"Normalized mean: {np_normalized.mean():.6f} (should be ~0)\n")

# ============================================================================
# 6. Autograd with NumPy Data
# ============================================================================
print("6. Autograd with NumPy Data")
print("-" * 70)

# Create training data in NumPy
np_x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
np_y = np.array([2.0, 4.0, 6.0], dtype=np.float32)  # y = 2*x

# Convert to tensors
x = t4d.Vectorf.from_numpy(np_x)
y = t4d.Vectorf.from_numpy(np_y)

# Create a simple weight
w = t4d.Vectorf([1.5, 1.5, 1.5])
w.set_requires_grad(True)

# Forward pass: prediction = w * x
pred = w * x

# Loss: mean squared error
diff = pred - y
loss_vec = diff * diff

print(f"Prediction: {pred.numpy()}")
print(f"Target: {y.numpy()}")
print(f"Loss vector: {loss_vec.numpy()}")
print(f"Mean loss: {loss_vec.mean():.4f}\n")

# ============================================================================
# 7. Integration Example: Using NumPy for Data Preprocessing
# ============================================================================
print("7. Integration Example: Data Preprocessing with NumPy")
print("-" * 70)

# Simulate loading data with NumPy
raw_data = np.random.randn(10, 5).astype(np.float32)
print(f"Raw data shape: {raw_data.shape}")
print(f"Raw data mean: {raw_data.mean():.4f}, std: {raw_data.std():.4f}")

# Preprocessing in NumPy
normalized_np = (raw_data - raw_data.mean(axis=0)) / (raw_data.std(axis=0) + 1e-8)
print(f"Normalized in NumPy - mean: {normalized_np.mean():.6f}, std: {normalized_np.std():.4f}")

# Convert to Tensor4D for model operations
data_tensor = t4d.Matrixf.from_numpy(normalized_np)
print(f"Tensor shape: {data_tensor.shape}")

# Perform operations in Tensor4D
scaled = data_tensor * 2.0
result = scaled.sigmoid()

# Convert back for saving/analysis
final_np = result.numpy()
print(f"Final result shape: {final_np.shape}")
print(f"Final result range: [{final_np.min():.4f}, {final_np.max():.4f}]\n")

# ============================================================================
# Summary
# ============================================================================
print("=" * 70)
print("Summary:")
print("  ✓ Seamless conversion between Tensor4D and NumPy")
print("  ✓ from_numpy() class method for explicit conversion")
print("  ✓ Constructor accepts NumPy arrays directly")
print("  ✓ numpy() method converts tensors to NumPy arrays")
print("  ✓ All data types and dimensions supported")
print("  ✓ Enables integration with NumPy ecosystem")
print("=" * 70)
