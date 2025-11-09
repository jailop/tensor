# Core Tensor Operations

## Creating Tensors

### Factory Functions

The library provides several convenient factory functions for creating tensors:

```cpp
#include "tensor.h"

// Zeros: all elements set to 0
auto zeros_mat = Matrix<float>::zeros({3, 4});        // 3x4 matrix
auto zeros_vec = Vector<double>::zeros({10});         // 10-element vector

// Ones: all elements set to 1
auto ones_mat = Matrix<float>::ones({2, 3});

// Full: all elements set to a specific value
auto filled = Matrix<float>::full({4, 4}, 3.14f);

// Identity matrix
auto identity = Matrix<float>::eye({5, 5});           // 5x5 identity

// Random initialization
auto uniform_mat = Matrix<float>::uniform({3, 3}, 0.0f, 1.0f);  // Uniform [0, 1)
auto normal_mat = Matrix<float>::randn({3, 3});                  // Normal(0, 1)

// From existing data
std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
auto from_vec = Matrix<float>::from_array(data, {2, 2});

// Sequences
auto range = Vector<float>::arange(0.0f, 10.0f, 1.0f);    // [0, 1, 2, ..., 9]
auto linsp = Vector<float>::linspace(0.0f, 1.0f, 11);     // 11 evenly spaced values
auto logsp = Vector<float>::logspace(0.0f, 3.0f, 4);      // [10^0, 10^1, 10^2, 10^3]
```

### Creating Higher-Rank Tensors

```cpp
// 3D tensor (e.g., for RGB images: height x width x channels)
Tensor<float, 3> image = Tensor<float, 3>::zeros({256, 256, 3});

// 4D tensor (e.g., for batches of images: batch x channels x height x width)
Tensor<float, 4> batch = Tensor<float, 4>::zeros({32, 3, 224, 224});
```

## Indexing and Element Access

### Basic Indexing

```cpp
auto A = Matrix<float>::ones({3, 4});

// Access individual elements
float val = A[{0, 0}];           // First element
A[{1, 2}] = 5.0f;                // Set element at row 1, col 2

// For vectors
auto v = Vector<float>::arange(0.0f, 5.0f, 1.0f);
float first = v[{0}];            // 0.0
float last = v[{4}];             // 4.0
```

### Advanced Indexing

```cpp
// Take elements at specific indices
auto indices = Vector<int>::from_array({0, 2, 4}, {3});
auto selected_var = v.take(indices);
auto selected = std::get<Vector<float>>(selected_var);

// Put values at specific indices
auto values = Vector<float>::from_array({10.0f, 20.0f, 30.0f}, {3});
v.put(indices, values);

// Boolean masking
auto mask = Vector<bool>::from_array({true, false, true, false, true}, {5});
auto masked_var = v.masked_select(mask);
auto masked = std::get<Vector<float>>(masked_var);

// Conditional selection
auto condition = v > 2.0f;  // Returns boolean tensor
auto result_var = where(condition, v, Vector<float>::zeros({5}));
```

## Basic Arithmetic Operations

### Element-wise Operations

```cpp
auto A = Matrix<float>::ones({2, 3});
auto B = Matrix<float>::full({2, 3}, 2.0f);

// Addition
auto C_var = A + B;                    // Element-wise addition
auto C = std::get<Matrix<float>>(C_var);

// Subtraction
auto D_var = B - A;
auto D = std::get<Matrix<float>>(D_var);

// Multiplication
auto E_var = A * B;                    // Element-wise multiplication (Hadamard product)
auto E = std::get<Matrix<float>>(E_var);

// Division
auto F_var = B / A;
auto F = std::get<Matrix<float>>(F_var);

// Power
auto G_var = A.pow(2.0f);              // Element-wise square
auto G = std::get<Matrix<float>>(G_var);
```

### In-place Operations

```cpp
auto A = Matrix<float>::ones({2, 3});
auto B = Matrix<float>::full({2, 3}, 2.0f);

// In-place operations (modify left operand)
A += B;        // A = A + B
A -= B;        // A = A - B
A *= B;        // A = A * B (element-wise)
A /= B;        // A = A / B
```

### Scalar Operations

```cpp
auto A = Matrix<float>::ones({2, 3});

// Scalar operations
auto B_var = A + 5.0f;      // Add 5 to all elements
auto C_var = A * 3.0f;      // Multiply all elements by 3
auto D_var = A / 2.0f;      // Divide all elements by 2
auto E_var = 10.0f - A;     // Scalar - tensor
```

## Broadcasting

Broadcasting allows operations between tensors of different shapes by automatically expanding dimensions.

### Broadcasting Rules

1. If tensors have different ranks, prepend dimensions of size 1 to the smaller-rank tensor
2. Two dimensions are compatible if they are equal or one of them is 1
3. The result shape is the element-wise maximum of input shapes

### Broadcasting Examples

```cpp
// Example 1: Vector + Matrix (column broadcasting)
auto vec = Vector<float>::from_array({1.0f, 2.0f, 3.0f}, {3});
auto mat = Matrix<float>::ones({3, 4});

// vec shape: (3,) → broadcast to (3, 4)
// mat shape: (3, 4)
auto result_var = mat + vec;  // Adds vec to each column of mat

// Example 2: Row vector + Matrix
auto row = Matrix<float>::from_array({1.0f, 2.0f, 3.0f, 4.0f}, {1, 4});
auto mat2 = Matrix<float>::ones({3, 4});

// row shape: (1, 4) → broadcast to (3, 4)
// mat2 shape: (3, 4)
auto result2_var = mat2 + row;  // Adds row to each row of mat2

// Example 3: Explicit broadcasting
auto target_shape = std::vector<size_t>{3, 4};
auto broadcasted_var = vec.broadcast_to(target_shape);
```

### Common Broadcasting Patterns

```cpp
// Pattern 1: Normalize rows (subtract mean per row)
auto mat = Matrix<float>::randn({10, 5});
auto row_means = mat.mean(1);  // Shape: (10, 1)
auto centered_var = mat - row_means;  // Broadcasting: (10, 5) - (10, 1)

// Pattern 2: Normalize columns (standardize features)
auto col_means = mat.mean(0);  // Shape: (1, 5)
auto col_stds = mat.std(0);    // Shape: (1, 5)
auto standardized_var = (mat - col_means) / col_stds;

// Pattern 3: Batch operations
Tensor<float, 3> batch({32, 10, 10});  // 32 images of 10x10
Vector<float> bias({10});               // Per-channel bias
// Broadcasting: (32, 10, 10) + (10,) broadcasts to (32, 10, 10)
auto biased_var = batch + bias;
```

## Shape Information

```cpp
auto tensor = Matrix<float>::ones({3, 4});

// Get shape information
size_t rank = tensor.rank();              // 2
auto shape = tensor.shape();              // {3, 4}
size_t rows = tensor.shape(0);            // 3
size_t cols = tensor.shape(1);            // 4
size_t total = tensor.size();             // 12 (total elements)

// Check dimensions
bool is_vector = (tensor.rank() == 1);
bool is_matrix = (tensor.rank() == 2);
bool is_square = (tensor.shape(0) == tensor.shape(1));
```

## Copying and Cloning

```cpp
auto A = Matrix<float>::randn({3, 3});

// Shallow copy (shares data)
auto B = A;  // B and A point to same data

// Deep copy (independent data)
auto C = A.copy();  // C has its own copy of data

// Detach from computational graph (for autograd)
auto D = A.detach();  // D shares data but no gradient tracking
```

## Comparison Operations

```cpp
auto A = Matrix<float>::from_array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
auto B = Matrix<float>::full({2, 2}, 2.5f);

// Element-wise comparisons (return boolean tensors)
auto gt = A > B;          // Greater than
auto lt = A < B;          // Less than
auto ge = A >= B;         // Greater than or equal
auto le = A <= B;         // Less than or equal
auto eq = A == B;         // Equal
auto ne = A != B;         // Not equal

// Reductions on boolean tensors
bool all_positive = (A > 0.0f).all();  // All elements > 0?
bool any_negative = (A < 0.0f).any();  // Any elements < 0?
```

## Clipping and Clamping

```cpp
auto A = Matrix<float>::randn({3, 3});

// Clip values to range [min, max]
auto clipped_var = A.clip(-1.0f, 1.0f);    // All values in [-1, 1]
auto clamped_var = A.clamp(0.0f, 10.0f);   // Same as clip

// Conditional filling based on mask
auto mask = A > 0.0f;
auto filled_var = A.masked_fill(mask, 999.0f);  // Set positive values to 999
```

## Type Conversion

```cpp
auto float_mat = Matrix<float>::randn({3, 3});

// Convert to different type
auto double_mat_var = float_mat.astype<double>();
auto double_mat = std::get<Matrix<double>>(double_mat_var);

auto int_mat_var = float_mat.astype<int>();
auto int_mat = std::get<Matrix<int>>(int_mat_var);
```

## Printing and Visualization

```cpp
auto mat = Matrix<float>::randn({4, 5});

// Basic printing
mat.print();

// Pretty printing (for large tensors, shows truncated view)
std::cout << mat.to_string() << std::endl;

// Print specific properties
std::cout << "Shape: ";
for (size_t i = 0; i < mat.rank(); i++) {
    std::cout << mat.shape(i) << " ";
}
std::cout << std::endl;
std::cout << "Total elements: " << mat.size() << std::endl;
```

## Practical Examples

### Example 1: Creating a Batch of Images

```cpp
// Create a batch of 32 RGB images of size 224x224
Tensor<float, 4> images = Tensor<float, 4>::randn({32, 3, 224, 224});

// Normalize images: (image - mean) / std
float mean = 0.5f, std = 0.25f;
auto normalized_var = (images - mean) / std;
auto normalized = std::get<Tensor<float, 4>>(normalized_var);

// Clip to valid range [0, 1]
auto clipped_var = normalized.clip(0.0f, 1.0f);
```

### Example 2: Feature Scaling

```cpp
// Dataset with 1000 samples, 20 features
auto X = Matrix<float>::randn({1000, 20});

// Standardization: (X - mean) / std
auto mean = X.mean(0);       // Mean of each feature (shape: 1x20)
auto std = X.std(0);         // Std of each feature (shape: 1x20)
auto X_scaled_var = (X - mean) / std;  // Broadcasting happens automatically
auto X_scaled = std::get<Matrix<float>>(X_scaled_var);

// Alternative: Min-Max scaling to [0, 1]
auto X_min = X.min(0);
auto X_max = X.max(0);
auto X_minmax_var = (X - X_min) / (X_max - X_min);
```

### Example 3: Creating One-Hot Encodings

```cpp
// Class labels (0, 1, 2, 3, 4 for 5 classes)
auto labels = Vector<int>::from_array({0, 2, 1, 4, 3, 0, 1}, {7});

// Create one-hot encoding
int num_classes = 5;
auto one_hot = Matrix<float>::zeros({7, num_classes});

for (int i = 0; i < 7; i++) {
    int label = labels[{i}];
    one_hot[{i, label}] = 1.0f;
}

one_hot.print();
// Output:
// [[1, 0, 0, 0, 0],
//  [0, 0, 1, 0, 0],
//  [0, 1, 0, 0, 0],
//  [0, 0, 0, 0, 1],
//  [0, 0, 0, 1, 0],
//  [1, 0, 0, 0, 0],
//  [0, 1, 0, 0, 0]]
```

## Best Practices

1. **Use type aliases**: Prefer `Matrix<float>` and `Vector<float>` over `Tensor<float, 2>` and `Tensor<float, 1>` for clarity
2. **Check for errors**: Always check `std::holds_alternative<TensorError>()` for operations that can fail
3. **Leverage broadcasting**: Use broadcasting instead of explicit loops when possible
4. **Choose appropriate types**: Use `float` for most ML tasks, `double` for numerical precision
5. **Memory efficiency**: Use in-place operations (`+=`, `-=`, etc.) when you don't need to preserve the original tensor

---

**Previous**: [← Getting Started](01-getting-started.md) | **Next**: [Shape Manipulation →](03-shape-manipulation.md)
