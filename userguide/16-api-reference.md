# API Reference

## Quick Reference

### Creation Functions
- `zeros()`, `ones()`, `full()`, `eye()`
- `randn()`, `uniform()`, `normal()`
- `arange()`, `linspace()`, `logspace()`
- `from_array()`

### Shape Operations
- `reshape()`, `flatten()`, `squeeze()`, `unsqueeze()`
- `transpose()`, `permute()`
- `repeat()`, `tile()`, `broadcast_to()`

### Arithmetic
- `+`, `-`, `*`, `/` (element-wise and broadcasting)
- `matmul()`, `dot()`, `cross()`

### Mathematical Functions
- `abs()`, `exp()`, `log()`, `sqrt()`, `pow()`
- `sin()`, `cos()`, `tan()`
- `sigmoid()`, `tanh()`, `relu()`, `leaky_relu()`, `softmax()`
- `ceil()`, `floor()`, `clip()`

### Reductions
- `sum()`, `mean()`, `variance()`, `std()`
- `min()`, `max()`, `median()`, `quantile()`
- `cumsum()`, `cumprod()`
- `argmin()`, `argmax()`
- `all()`, `any()`, `prod()`

### Linear Algebra
- `matmul()`, `dot()`, `cross()`
- `transpose()`, `inverse()`, `determinant()`
- `svd()`, `qr()`, `cholesky()`
- `eigenvalues()`, `eigenvectors()`

### Indexing
- `take()`, `put()`
- `masked_select()`, `masked_fill()`
- `where()`, `clip()`
- `select()`

### Views
- `row()`, `col()`, `diag()`, `block()`
- `head()`, `tail()`
- `topRows()`, `bottomRows()`, `leftCols()`, `rightCols()`

### I/O
- `save_tensor()`, `load_tensor()`
- `print()`, `to_string()`

### Normalization
- `normalize_l1()`, `normalize_l2()`
- `normalize_zscore()`, `normalize_minmax()`

### Sorting/Searching
- `sort()`, `argsort()`, `topk()`
- `unique()`, `searchsorted()`

### Stacking
- `concatenate()`, `stack()`
- `vstack()`, `hstack()`
- `split()`, `chunk()`

### Autograd
- `backward()`, `zero_grad()`, `detach()`
- `requires_grad()`, `is_leaf()`, `grad()`

### Loss Functions
- `MSELoss`, `L1Loss`, `SmoothL1Loss`
- `CrossEntropyLoss`, `BinaryCrossEntropyLoss`

### Optimizers
- `SGD`, `Adam`, `AdamW`, `RMSprop`
- `step()`, `zero_grad()`

## Type Aliases

```cpp
template <typename T> using Matrix = Tensor<T, 2>;
template <typename T> using Vector = Tensor<T, 1>;
template <typename T> using TensorResult = std::variant<T, TensorError>;
```

## Backend Configuration

Compile-time flags:
- `USE_GPU`: Enable CUDA acceleration
- `USE_BLAS`: Enable BLAS optimization
- `USE_LAPACK`: Enable advanced linear algebra

Runtime checks:
```cpp
#ifdef USE_GPU
    // GPU code
#elif defined(USE_BLAS)
    // BLAS code
#else
    // CPU fallback
#endif
```

---

**Previous**: [← Best Practices](15-best-practices.md) | **Up**: [Index ↑](00-index.md)
