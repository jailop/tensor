# Normalization and Views

## Normalization Functions

```cpp
#include "normalization.h"

auto data = Matrix<float>::randn({100, 20});

// L1 Normalization (sum of absolute values = 1)
auto l1_norm_var = normalize_l1(data, 1);  // Normalize each row

// L2 Normalization (Euclidean norm = 1)
auto l2_norm_var = normalize_l2(data, 1);  // Normalize each row

// Z-Score Normalization (mean=0, std=1)
auto z_norm_var = normalize_zscore(data, 1, 1e-8f);

// Min-Max Normalization (scale to [0, 1])
auto minmax_var = normalize_minmax(data, 1, 0.0f, 1.0f);
```

## Submatrix Views

```cpp
#include "tensor_views.h"

auto mat = Matrix<float>::randn({10, 10});

// Extract single row or column
auto row_5 = mat.row(5);     // 6th row as 1D tensor
auto col_3 = mat.col(3);     // 4th column as 1D tensor

// Diagonal
auto diag = mat.diag();      // Main diagonal as 1D tensor

// Create diagonal matrix from vector
auto vec = Vector<float>::from_array({1.0f, 2.0f, 3.0f}, {3});
auto diag_mat_var = diag_matrix(vec);
// Result: [[1, 0, 0],
//          [0, 2, 0],
//          [0, 0, 3]]

// Block extraction
auto block = mat.block(2, 3, 4, 5);  // 4x5 block starting at (2,3)

// Edge extractions
auto top_3 = mat.topRows(3);       // First 3 rows
auto bottom_3 = mat.bottomRows(3); // Last 3 rows
auto left_4 = mat.leftCols(4);     // Leftmost 4 columns
auto right_4 = mat.rightCols(4);   // Rightmost 4 columns

// For vectors
auto v = Vector<float>::arange(0.0f, 100.0f, 1.0f);
auto head_10 = v.head(10);  // First 10 elements
auto tail_10 = v.tail(10);  // Last 10 elements
```

---

**Previous**: [← Performance Optimization](10-performance-optimization.md) | **Next**: [Random Sampling →](12-random-sampling.md)
