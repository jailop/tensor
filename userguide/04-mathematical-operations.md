# Mathematical Operations

## Element-wise Mathematical Functions

### Basic Math Functions

```cpp
#include "tensor.h"

auto x = Matrix<float>::from_array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});

// Absolute value
auto abs_x = x.abs();

// Exponential
auto exp_x = x.exp();

// Natural logarithm
auto log_x = x.log();

// Square root
auto sqrt_x = x.sqrt();

// Power
auto squared = x.pow(2.0f);
auto cubed = x.pow(3.0f);

// Ceiling and floor
auto ceil_x = x.ceil();
auto floor_x = x.floor();
```

### Trigonometric Functions

```cpp
auto angles = Vector<float>::linspace(0.0f, 3.14159f, 10);

// Basic trig
auto sin_vals = angles.sin();
auto cos_vals = angles.cos();
auto tan_vals = angles.tan();

// Inverse trig (if implemented)
// auto asin_vals = sin_vals.asin();
// auto acos_vals = cos_vals.acos();
// auto atan_vals = tan_vals.atan();
```

### Activation Functions

```cpp
auto logits = Matrix<float>::randn({10, 5});

// Sigmoid: 1 / (1 + exp(-x))
auto sigmoid_out = logits.sigmoid();

// Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
auto tanh_out = logits.tanh();

// ReLU: max(0, x)
auto relu_out = logits.relu();

// Leaky ReLU: max(alpha * x, x)
auto leaky_relu_out = logits.leaky_relu(0.01f);

// Softmax (along last dimension)
auto softmax_out = logits.softmax();
```

## Reduction Operations

### Basic Reductions

```cpp
auto mat = Matrix<float>::from_array({1.0f, 2.0f, 3.0f, 
                                       4.0f, 5.0f, 6.0f}, {2, 3});

// Sum of all elements
float total = mat.sum();  // 21.0

// Product of all elements
float product = mat.prod();  // 720.0

// Mean
float average = mat.mean();  // 3.5

// Min and Max
float minimum = mat.min();  // 1.0
float maximum = mat.max();  // 6.0
```

### Axis-wise Reductions

```cpp
auto mat = Matrix<float>::randn({10, 5});

// Sum along axis 0 (collapse rows → result shape: 1x5)
auto col_sums = mat.sum(0);

// Sum along axis 1 (collapse columns → result shape: 10x1)
auto row_sums = mat.sum(1);

// Mean per column
auto col_means = mat.mean(0);

// Mean per row
auto row_means = mat.mean(1);

// Min/Max per axis
auto col_mins = mat.min(0);
auto col_maxs = mat.max(0);
```

### Statistical Functions

```cpp
auto data = Vector<float>::randn({1000});

// Variance
float var = data.variance();

// Standard deviation
float std = data.std();

// Median
float med = data.median();

// Quantiles
float q25 = data.quantile(0.25f);  // 25th percentile
float q50 = data.quantile(0.50f);  // median
float q75 = data.quantile(0.75f);  // 75th percentile
```

### Correlation and Covariance

```cpp
auto x = Vector<float>::randn({100});
auto y = Vector<float>::randn({100});

// Pearson correlation
auto corr_var = pearson_correlation(x, y);
float corr = std::get<float>(corr_var);

// Spearman correlation (rank-based)
auto spearman_var = spearman_correlation(x, y);
float spearman = std::get<float>(spearman_var);

// Covariance
auto cov_var = covariance(x, y);
float cov = std::get<float>(cov_var);

// Covariance matrix for multi-dimensional data
auto X = Matrix<float>::randn({100, 5});  // 100 samples, 5 features
auto cov_matrix_var = covariance_matrix(X);
auto cov_matrix = std::get<Matrix<float>>(cov_matrix_var);
// Result: 5x5 covariance matrix
```

## Cumulative Operations

```cpp
auto v = Vector<float>::from_array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {5});

// Cumulative sum
auto cumsum = v.cumsum();  // [1, 3, 6, 10, 15]

// Cumulative product
auto cumprod = v.cumprod();  // [1, 2, 6, 24, 120]

// Along specific axis for matrices
auto mat = Matrix<float>::from_array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
auto cumsum_axis0 = mat.cumsum_axis(0);  // Cumsum down columns
auto cumsum_axis1 = mat.cumsum_axis(1);  // Cumsum across rows
```

## Argument Reductions

Find indices of minimum/maximum values.

```cpp
auto v = Vector<float>::from_array({3.0f, 1.0f, 4.0f, 1.0f, 5.0f}, {5});

// Index of minimum value
size_t argmin_idx = v.argmin();  // 1

// Index of maximum value
size_t argmax_idx = v.argmax();  // 4

// For matrices, along specific axis
auto mat = Matrix<float>::randn({10, 5});
auto argmin_cols = mat.argmin_axis(0);  // Shape: (1, 5)
auto argmin_rows = mat.argmin_axis(1);  // Shape: (10, 1)
```

## Boolean Reductions

```cpp
auto mat = Matrix<float>::from_array({-1.0f, 2.0f, -3.0f, 4.0f}, {2, 2});

// Check if all elements satisfy condition
bool all_positive = (mat > 0.0f).all();  // false

// Check if any element satisfies condition
bool any_positive = (mat > 0.0f).any();  // true
bool any_negative = (mat < 0.0f).any();  // true

// Count number of true elements
auto positive_mask = mat > 0.0f;
int num_positive = positive_mask.sum();  // Treating bool as int
```

## Practical Examples

### Example 1: Data Normalization

```cpp
auto data = Matrix<float>::randn({1000, 20});

// Z-score normalization
auto mean = data.mean(0);
auto std = data.std(0);
auto normalized = (data - mean) / (std + 1e-8f);  // Add epsilon for numerical stability

// Min-max normalization to [0, 1]
auto min_vals = data.min(0);
auto max_vals = data.max(0);
auto minmax_normalized = (data - min_vals) / (max_vals - min_vals + 1e-8f);
```

### Example 2: Computing Class Probabilities

```cpp
// Raw logits from neural network
auto logits = Matrix<float>::randn({64, 10});  // batch=64, num_classes=10

// Apply softmax to get probabilities
auto probs = logits.softmax();  // Each row sums to 1.0

// Get predicted class (argmax)
auto predictions = probs.argmax_axis(1);  // Shape: (64, 1)
```

### Example 3: Finding Outliers

```cpp
auto data = Vector<float>::randn({1000});

// Calculate z-scores
float mean = data.mean();
float std = data.std();
auto z_scores = (data - mean) / std;

// Find outliers (|z| > 3)
auto is_outlier = z_scores.abs() > 3.0f;
int num_outliers = is_outlier.sum();

// Get outlier values
auto outliers = data.masked_select(is_outlier);
```

### Example 4: Moving Average

```cpp
auto time_series = Vector<float>::randn({100});
int window = 5;

// Simple moving average using cumsum
auto cs = time_series.cumsum();
Vector<float> moving_avg({time_series.size() - window + 1});

for (size_t i = 0; i < moving_avg.size(); i++) {
    float sum = (i == 0) ? cs[{window - 1}] : cs[{i + window - 1}] - cs[{i - 1}];
    moving_avg[{i}] = sum / window;
}
```

### Example 5: Correlation Matrix

```cpp
// Dataset with multiple features
auto X = Matrix<float>::randn({100, 10});  // 100 samples, 10 features

// Compute pairwise correlations
Matrix<float> corr_matrix({10, 10});

for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
        auto col_i = X.col(i);
        auto col_j = X.col(j);
        auto corr_var = pearson_correlation(col_i, col_j);
        corr_matrix[{i, j}] = std::get<float>(corr_var);
    }
}
```

## Best Practices

1. **Add epsilon for numerical stability** when dividing (e.g., `std + 1e-8`)
2. **Use axis parameter** for efficient computation on specific dimensions
3. **Leverage boolean masks** for conditional operations
4. **Check for NaN/Inf** after operations like log, sqrt, division
5. **Use cumulative operations** instead of loops when possible

---

**Previous**: [← Shape Manipulation](03-shape-manipulation.md) | **Next**: [Linear Algebra →](05-linear-algebra.md)
