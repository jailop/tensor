# Stacking and Concatenation

```cpp
auto A = Matrix<float>::ones({2, 3});
auto B = Matrix<float>::full({2, 3}, 2.0f);

// Concatenate along axis 0 (vertical stack)
auto cat0_var = concatenate({A, B}, 0);
auto cat0 = std::get<Matrix<float>>(cat0_var);  // Shape: (4, 3)

// Concatenate along axis 1 (horizontal stack)
auto cat1_var = concatenate({A, B}, 1);
auto cat1 = std::get<Matrix<float>>(cat1_var);  // Shape: (2, 6)

// Stack (creates new dimension)
auto stacked_var = stack({A, B}, 0);
auto stacked = std::get<Tensor<float, 3>>(stacked_var);  // Shape: (2, 2, 3)

// Convenience functions
auto vstack_var = vstack({A, B});  // Vertical stack
auto hstack_var = hstack({A, B});  // Horizontal stack

// Split
auto mat = Matrix<float>::randn({10, 5});
auto splits_var = mat.split(2, 0);  // Split into 2 chunks along axis 0
auto splits = std::get<std::vector<Matrix<float>>>(splits_var);
// splits[0]: (5, 5), splits[1]: (5, 5)

// Chunk
auto chunks_var = mat.chunk(3, 0);  // Divide into 3 equal chunks
```

---

**Previous**: [← Sorting and Searching](13-sorting-searching.md) | **Next**: [Best Practices →](15-best-practices.md)
