# Sorting and Searching

```cpp
auto data = Vector<float>::from_array({3.0f, 1.0f, 4.0f, 1.0f, 5.0f}, {5});

// Sort (returns sorted tensor)
auto sorted_var = data.sort(true);  // ascending
auto sorted = std::get<Vector<float>>(sorted_var);  // [1, 1, 3, 4, 5]

// Argsort (returns indices that would sort the array)
auto indices_var = data.argsort(true);
auto indices = std::get<Vector<int>>(indices_var);  // [1, 3, 0, 2, 4]

// Top-k largest/smallest elements
auto [values, indices] = data.topk(3, true);  // 3 largest
// values: [5, 4, 3], indices: [4, 2, 0]

// Unique elements
auto unique_var = data.unique();
auto unique = std::get<Vector<float>>(unique_var);  // [1, 3, 4, 5]

// Binary search (searchsorted)
auto sorted_data = Vector<float>::from_array({1.0f, 3.0f, 5.0f, 7.0f}, {4});
auto insert_idx_var = sorted_data.searchsorted(4.0f);  // Index where 4.0 would be inserted
```

---

**Previous**: [← Random Sampling](12-random-sampling.md) | **Next**: [Stacking and Concatenation →](14-stacking-concatenation.md)
