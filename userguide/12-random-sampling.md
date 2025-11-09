# Random Sampling

```cpp
auto rng = TensorRandom(42);  // Seed for reproducibility

// Uniform distribution [0, 1)
auto uniform_var = rng.uniform({100}, 0.0f, 1.0f);

// Normal distribution (mean=0, std=1)
auto normal_var = rng.normal({100}, 0.0f, 1.0f);

// Exponential distribution
auto exp_var = rng.exponential({100}, 1.0f);

// Random permutation
auto perm_var = rng.randperm(100);  // Random shuffle of [0..99]

// Random choice (sampling without replacement)
auto samples_var = rng.choice(100, 10);  // Sample 10 from 100

// Random choice with replacement
auto with_repl_var = rng.choice_with_replacement(100, 20);
```

---

**Previous**: [← Normalization and Views](11-normalization-views.md) | **Next**: [Sorting and Searching →](13-sorting-searching.md)
