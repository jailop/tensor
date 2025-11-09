# Shape Manipulation

Shape manipulation operations allow you to change how tensor data is organized without modifying the underlying values.

## Reshape

Change the shape of a tensor while preserving the total number of elements.

```cpp
#include "tensor.h"

// Create a 1D vector with 12 elements
auto v = Vector<float>::arange(0.0f, 12.0f, 1.0f);  // [0, 1, 2, ..., 11]

// Reshape to 3x4 matrix
auto mat_var = v.reshape({3, 4});
auto mat = std::get<Matrix<float>>(mat_var);
mat.print();
// Output:
// [[0, 1, 2, 3],
//  [4, 5, 6, 7],
//  [8, 9, 10, 11]]

// Reshape to 2x6 matrix
auto mat2_var = v.reshape({2, 6});

// Use -1 to infer one dimension
auto mat3_var = v.reshape({-1, 3});  // Automatically becomes 4x3
auto mat4_var = v.reshape({4, -1});  // Automatically becomes 4x3
```

### Important Notes

- The total number of elements must remain the same
- Reshaping returns a view when possible (no data copy)
- Use `-1` for one dimension to let the library infer it automatically

## Flatten

Convert a multi-dimensional tensor into a 1D vector.

```cpp
auto mat = Matrix<float>::from_array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
mat.print();
// [[1, 2],
//  [3, 4]]

// Flatten to 1D
auto flat_var = mat.flatten();
auto flat = std::get<Vector<float>>(flat_var);
flat.print();
// [1, 2, 3, 4]

// Flatten is equivalent to reshape({-1})
auto flat2_var = mat.reshape({-1});
```

## Squeeze and Unsqueeze

### Squeeze: Remove Dimensions of Size 1

```cpp
// Create a tensor with unnecessary dimensions
Tensor<float, 3> t3({1, 5, 1});
t3.fill(1.0f);

// Remove all dimensions of size 1
auto squeezed_var = t3.squeeze();
auto squeezed = std::get<Vector<float>>(squeezed_var);
// Result shape: (5,)

// Squeeze specific dimension
Tensor<float, 3> t4({1, 3, 4});
auto squeezed_0_var = t4.squeeze(0);  // Remove first dimension
auto squeezed_0 = std::get<Matrix<float>>(squeezed_0_var);
// Result shape: (3, 4)
```

### Unsqueeze: Add a Dimension of Size 1

```cpp
auto vec = Vector<float>::arange(0.0f, 5.0f, 1.0f);  // Shape: (5,)

// Add dimension at position 0 → row vector
auto row_var = vec.unsqueeze(0);
auto row = std::get<Matrix<float>>(row_var);
// Result shape: (1, 5)

// Add dimension at position 1 → column vector
auto col_var = vec.unsqueeze(1);
auto col = std::get<Matrix<float>>(col_var);
// Result shape: (5, 1)
```

**Use Case**: Preparing tensors for broadcasting

```cpp
auto vec = Vector<float>::ones({3});
auto mat = Matrix<float>::ones({3, 4});

// To add vec to each column of mat:
auto col_vec_var = vec.unsqueeze(1);  // Shape: (3, 1)
auto result_var = mat + std::get<Matrix<float>>(col_vec_var);  // Broadcasting

// To add vec to each row of mat:
auto row_vec_var = vec.unsqueeze(0);  // Shape: (1, 3)
// Then need to broadcast along axis 1...
```

## Transpose

Swap dimensions of a tensor.

### 2D Transpose (Matrix)

```cpp
auto mat = Matrix<float>::from_array({1.0f, 2.0f, 3.0f, 
                                       4.0f, 5.0f, 6.0f}, {2, 3});
mat.print();
// [[1, 2, 3],
//  [4, 5, 6]]

auto transposed_var = mat.transpose();
auto transposed = std::get<Matrix<float>>(transposed_var);
transposed.print();
// [[1, 4],
//  [2, 5],
//  [3, 6]]
```

### Multi-dimensional Transpose

```cpp
// For higher-rank tensors, specify which dimensions to swap
Tensor<float, 3> cube({2, 3, 4});
// Original shape: (2, 3, 4)

auto swapped_var = cube.transpose(0, 2);
// New shape: (4, 3, 2) - swapped first and last dimensions
```

## Permute

Reorder all dimensions according to a specified permutation.

```cpp
// Create a 3D tensor representing: batch x channels x length
Tensor<float, 3> t({32, 3, 100});  // 32 samples, 3 channels, 100 time steps

// Permute to: channels x batch x length
auto permuted_var = t.permute({1, 0, 2});
auto permuted = std::get<Tensor<float, 3>>(permuted_var);
// New shape: (3, 32, 100)

// Permute to: length x batch x channels (common for RNNs)
auto permuted2_var = t.permute({2, 0, 1});
// New shape: (100, 32, 3)
```

**Common Use Cases**:
- Converting between NCHW and NHWC image formats
- Preparing data for different neural network architectures
- Transposing batched matrices

## Repeat

Repeat a tensor along its dimensions.

```cpp
auto vec = Vector<float>::from_array({1.0f, 2.0f, 3.0f}, {3});

// Repeat each element 3 times
auto repeated_var = vec.repeat({3});
auto repeated = std::get<Vector<float>>(repeated_var);
repeated.print();
// [1, 1, 1, 2, 2, 2, 3, 3, 3]

// For 2D tensors
auto mat = Matrix<float>::from_array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
// [[1, 2],
//  [3, 4]]

auto repeated_mat_var = mat.repeat({2, 3});  // Repeat 2x along axis 0, 3x along axis 1
auto repeated_mat = std::get<Matrix<float>>(repeated_mat_var);
repeated_mat.print();
// [[1, 2, 1, 2, 1, 2],
//  [3, 4, 3, 4, 3, 4],
//  [1, 2, 1, 2, 1, 2],
//  [3, 4, 3, 4, 3, 4]]
```

### Repeat Along Axis

```cpp
auto mat = Matrix<float>::from_array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});

// Repeat along axis 0 (rows)
auto rep_rows_var = mat.repeat_along_axis(3, 0);
auto rep_rows = std::get<Matrix<float>>(rep_rows_var);
// Shape: (6, 2) - each row repeated 3 times

// Repeat along axis 1 (columns)
auto rep_cols_var = mat.repeat_along_axis(2, 1);
auto rep_cols = std::get<Matrix<float>>(rep_cols_var);
// Shape: (2, 4) - each column repeated 2 times
```

## Tile

Repeat the entire tensor multiple times along each dimension.

```cpp
auto mat = Matrix<float>::from_array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
// [[1, 2],
//  [3, 4]]

// Tile 2 times vertically, 3 times horizontally
auto tiled_var = mat.tile({2, 3});
auto tiled = std::get<Matrix<float>>(tiled_var);
tiled.print();
// [[1, 2, 1, 2, 1, 2],
//  [3, 4, 3, 4, 3, 4],
//  [1, 2, 1, 2, 1, 2],
//  [3, 4, 3, 4, 3, 4]]
```

**Difference between `repeat()` and `tile()`**:
- `repeat()`: Repeats individual elements
- `tile()`: Repeats the entire tensor as a block

## Expand and Broadcast To

Explicitly broadcast a tensor to a new shape.

```cpp
auto vec = Vector<float>::from_array({1.0f, 2.0f, 3.0f}, {3});

// Broadcast to matrix shape
auto broadcasted_var = vec.broadcast_to({4, 3});
auto broadcasted = std::get<Matrix<float>>(broadcasted_var);
broadcasted.print();
// [[1, 2, 3],
//  [1, 2, 3],
//  [1, 2, 3],
//  [1, 2, 3]]

// Works with unsqueeze for more control
auto col_vec_var = vec.unsqueeze(1);  // Shape: (3, 1)
auto col_vec = std::get<Matrix<float>>(col_vec_var);
auto broadcast_cols_var = col_vec.broadcast_to({3, 5});
auto broadcast_cols = std::get<Matrix<float>>(broadcast_cols_var);
// Each element of vec becomes a row
```

## Practical Examples

### Example 1: Batch Matrix Multiplication Preparation

```cpp
// Single matrix
auto W = Matrix<float>::randn({10, 5});  // Weight matrix

// Batch of input vectors
auto X = Matrix<float>::randn({32, 10});  // 32 samples, 10 features each

// Matrix multiplication: X @ W.T
auto result_var = matmul(X, W.transpose());
// Result shape: (32, 5) - 32 samples, 5 output features
```

### Example 2: Image Transformations

```cpp
// Batch of images: NCHW format (batch, channels, height, width)
Tensor<float, 4> images({32, 3, 224, 224});

// Convert to NHWC format for some operations
auto nhwc_var = images.permute({0, 2, 3, 1});
auto nhwc = std::get<Tensor<float, 4>>(nhwc_var);
// New shape: (32, 224, 224, 3)

// Flatten each image for a fully connected layer
auto batch_flat_var = images.reshape({32, -1});  // Infer second dimension
auto batch_flat = std::get<Matrix<float>>(batch_flat_var);
// Shape: (32, 3*224*224) = (32, 150528)
```

### Example 3: Creating Positional Embeddings

```cpp
// Position indices
auto positions = Vector<float>::arange(0.0f, 100.0f, 1.0f);  // [0, 1, ..., 99]

// Feature dimension indices
auto dims = Vector<float>::arange(0.0f, 512.0f, 1.0f);  // [0, 1, ..., 511]

// Create 2D grid for positional encoding calculation
auto pos_2d_var = positions.unsqueeze(1);  // Shape: (100, 1)
auto dim_2d_var = dims.unsqueeze(0);        // Shape: (1, 512)

auto pos_2d = std::get<Matrix<float>>(pos_2d_var);
auto dim_2d = std::get<Matrix<float>>(dim_2d_var);

auto pos_expanded_var = pos_2d.broadcast_to({100, 512});
auto dim_expanded_var = dim_2d.broadcast_to({100, 512});
// Now compute positional encodings using broadcasting
```

### Example 4: Attention Mechanism Reshaping

```cpp
// Multi-head attention: split embedding into multiple heads
auto embeddings = Matrix<float>::randn({32, 512});  // batch_size=32, d_model=512
int num_heads = 8;
int d_head = 512 / num_heads;  // 64

// Reshape to (batch, num_heads, seq_len, d_head)
// This is tricky with 2D input, so typically:
// First expand sequence dimension
auto seq_len = 10;  // Assume sequence length of 10
Tensor<float, 3> seq_embeddings({32, seq_len, 512});

// Reshape for multi-head attention
auto reshaped_var = seq_embeddings.reshape({32, seq_len, num_heads, d_head});
auto reshaped = std::get<Tensor<float, 4>>(reshaped_var);

// Permute to (batch, num_heads, seq_len, d_head)
auto permuted_var = reshaped.permute({0, 2, 1, 3});
auto permuted = std::get<Tensor<float, 4>>(permuted_var);
// Shape: (32, 8, 10, 64)
```

### Example 5: Data Augmentation

```cpp
// Original image batch
auto images = Tensor<float, 4>::randn({16, 3, 64, 64});

// Horizontal flip (simple version - reverse width dimension)
// This is conceptual - actual implementation would need slicing
auto flipped = images;  // Copy
// Manual flip logic here...

// Tile image to create augmented dataset
auto tiled_var = images.tile({2, 1, 1, 1});  // Double the batch
auto tiled = std::get<Tensor<float, 4>>(tiled_var);
// New shape: (32, 3, 64, 64)
```

## Common Pitfalls

### 1. Shape Compatibility

```cpp
auto v = Vector<float>::ones({12});

// This works
auto mat_var = v.reshape({3, 4});  // 12 = 3 * 4 ✓

// This fails
auto bad_var = v.reshape({3, 5});  // 12 ≠ 3 * 5 ✗
// Returns TensorError
```

### 2. Transpose vs. Reshape

```cpp
auto mat = Matrix<float>::from_array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
// [[1, 2],
//  [3, 4]]

// Transpose: swaps dimensions, reorders data
auto transposed = mat.transpose();
// [[1, 3],
//  [2, 4]]

// Reshape: changes shape, keeps data order
auto reshaped = mat.reshape({4, 1});
// [[1],
//  [2],
//  [3],
//  [4]]
```

### 3. Memory Considerations

- `reshape()`, `transpose()`, and `permute()` typically return views (no copy)
- `repeat()` and `tile()` create new tensors (copy data)
- Modifying a view affects the original tensor

## Best Practices

1. **Use `-1` for automatic dimension inference** in reshape
2. **Prefer transpose over permute** for 2D tensors (clearer intent)
3. **Check shapes** before operations to catch errors early
4. **Use unsqueeze/squeeze** to adjust dimensions for broadcasting
5. **Document reshaping logic** in complex transformations (especially for attention, convolutions)

---

**Previous**: [← Core Tensor Operations](02-core-tensor-operations.md) | **Next**: [Mathematical Operations →](04-mathematical-operations.md)
