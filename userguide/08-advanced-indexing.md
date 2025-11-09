# Advanced Indexing and Slicing

## Fancy Indexing

```cpp
auto data = Vector<float>::arange(0.0f, 10.0f, 1.0f);

// Take elements at specific indices
auto indices = Vector<int>::from_array({0, 2, 5, 7}, {4});
auto selected_var = data.take(indices);
auto selected = std::get<Vector<float>>(selected_var);
// Result: [0, 2, 5, 7]

// Put values at specific indices
auto values = Vector<float>::from_array({100.0f, 200.0f}, {2});
auto idx = Vector<int>::from_array({1, 8}, {2});
data.put(idx, values);  // Modifies data in-place
```

## Boolean Indexing

```cpp
auto data = Vector<float>::randn({100});

// Create boolean mask
auto mask = data > 0.0f;

// Select elements where mask is true
auto positive_var = data.masked_select(mask);
auto positive = std::get<Vector<float>>(positive_var);

// Fill elements where mask is true
auto filled_var = data.masked_fill(mask, 999.0f);
```

## Conditional Operations

```cpp
auto x = Matrix<float>::randn({10, 10});

// Where: select elements based on condition
auto condition = x > 0.0f;
auto y = Matrix<float>::zeros({10, 10});
auto result_var = where(condition, x, y);  // x if condition, else y

// Clip values to range
auto clipped_var = x.clip(-1.0f, 1.0f);  // Clamp to [-1, 1]
```

---

**Previous**: [← Machine Learning](07-machine-learning.md) | **Next**: [I/O Operations →](09-io-operations.md)
