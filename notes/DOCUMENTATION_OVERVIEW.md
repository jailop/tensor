# Documentation Overview

This document provides an overview of all documentation available for the Tensor Library.

## Quick Navigation

| Document Type | Location | Description |
|--------------|----------|-------------|
| **User Guide** | `./userguide/` | Comprehensive 18-section guide covering all features |
| **C Interface** | `./c_interop.md` | Technical specification for C API design |
| **API Docs** | `./docs/html/` | Doxygen-generated API documentation (after build) |
| **Features** | `./features.md` | Feature tracking and implementation status |
| **Examples** | `./tests/` | Working code examples (411+ test cases) |
| **Python** | `./python/` | Python bindings and examples |

## Documentation Files

### Main User Guide (`./userguide/`)

A complete, structured guide with 18 sections:

#### Core Documentation (1-9)
- **01** - Getting Started: Installation, setup, first examples
- **02** - Core Tensor Operations: Creation, indexing, arithmetic, broadcasting
- **03** - Shape Manipulation: Reshape, transpose, permute, squeeze
- **04** - Mathematical Operations: Element-wise functions, reductions, statistics
- **05** - Linear Algebra: Matrix operations, decompositions (SVD, QR, etc.)
- **06** - Automatic Differentiation: Gradient tracking, backward propagation
- **07** - Machine Learning: Loss functions, optimizers (SGD, Adam, etc.)
- **08** - Advanced Indexing: Fancy indexing, masking, conditional operations
- **09** - I/O Operations: Save/load, NumPy format, CSV

#### Advanced Topics (10-15)
- **10** - Performance Optimization: GPU, BLAS, threading, memory pooling
- **11** - Normalization & Views: Data preprocessing, submatrix views
- **12** - Random Sampling: Distributions, permutations, sampling
- **13** - Sorting & Searching: Sort, argsort, topk, unique, binary search
- **14** - Stacking & Concatenation: Combining and splitting tensors
- **15** - Best Practices: Guidelines, patterns, common pitfalls

#### Integration & Reference (16-18)
- **16** - API Reference: Quick lookup for all functions
- **17** - Python Integration: Python bindings with NumPy interop
- **18** - C Interface: Using the library from C code

### Technical Specifications

#### `c_interop.md`
Detailed specification for creating a C interface:
- Opaque handle design patterns
- Error handling strategies
- Memory management rules
- API design patterns for tensors, matrices, vectors
- Complete function signature examples
- Build configuration
- Usage examples in C

#### `features.md`
Feature tracking document with implementation status:
- ✅ Completed features (comprehensive list)
- 🚧 In-progress features
- 📋 Planned enhancements
- Priority classifications
- Library comparisons (Armadillo, Eigen)

### Generated Documentation

#### Doxygen API Documentation
Generate with:
```bash
cd build
make doc
```

Output location: `./docs/html/index.html`

Contents:
- Class documentation for all tensor types
- Function signatures with parameter descriptions
- Code examples in comments
- Inheritance diagrams
- File dependency graphs

## Documentation by Use Case

### For Beginners
1. Start: `userguide/01-getting-started.md`
2. Learn basics: `userguide/02-core-tensor-operations.md`
3. Practice: Examples in `tests/tensor_test.cc`

### For Python Users
1. Setup: `userguide/17-python-integration.md`
2. Examples: `python/examples/`
3. NumPy: See section on `.from_numpy()` and `.numpy()`

### For C Users
1. Overview: `c_interop.md`
2. Usage: `userguide/18-c-interface.md`
3. Examples: Complete C programs in user guide

### For ML Practitioners
1. Autograd: `userguide/06-autograd.md`
2. Training: `userguide/07-machine-learning.md`
3. Optimization: `userguide/10-performance-optimization.md`

### For Scientists/Engineers
1. Linear Algebra: `userguide/05-linear-algebra.md`
2. Statistics: `userguide/04-mathematical-operations.md`
3. I/O: `userguide/09-io-operations.md`

### For Library Developers
1. Source: `include/*.h` (well-commented headers)
2. Tests: `tests/*.cc` (comprehensive test coverage)
3. Doxygen: Generated API docs

## Example Code Locations

### C++ Examples
- **Basic operations**: `tests/tensor_test.cc`
- **Linear algebra**: `tests/tensor_linalg_test.cc`
- **Autograd**: `tests/tensor_test.cc` (autograd section)
- **I/O**: `tests/tensor_io_test.cc`
- **Broadcasting**: `tests/tensor_broadcasting_test.cc`
- **Indexing**: `tests/tensor_indexing_test.cc`
- **Random**: `tests/tensor_random_test.cc`
- **Sorting**: `tests/tensor_sorting_test.cc`
- **Stacking**: `tests/tensor_stacking_test.cc`
- **Shape**: `tests/tensor_shape_test.cc`

### Python Examples
- **Basic usage**: `python/examples/basic_operations.py`
- **Linear regression**: `python/examples/linear_regression.py`
- **Neural network**: `python/examples/simple_nn.py`
- **NumPy interop**: `python/examples/numpy_interop.py`

### C Examples
- Complete examples in `userguide/18-c-interface.md`
- Linear regression training example
- Matrix operations example

## Building Documentation

### User Guide
User guide is pre-written markdown:
```bash
cd userguide
# Open in browser or markdown viewer
```

### API Documentation
Generate Doxygen docs:
```bash
mkdir -p build
cd build
cmake ..
make doc
# Open docs/html/index.html
```

### Python Documentation
Python docstrings are included in bindings:
```python
import tensor4d
help(tensor4d.Matrixf)
```

## Documentation Standards

### Code Comments
- All public APIs have Doxygen comments
- Examples included in comments where helpful
- Parameter descriptions for all functions
- Return value documentation

### User Guide
- Step-by-step explanations
- Complete code examples
- Expected output shown
- Common pitfalls highlighted
- Cross-references between sections

### Test Files
- Tests serve as executable examples
- Clear test names describe functionality
- Comprehensive coverage (411+ tests)

## Maintenance

### Updating Documentation

When adding features:
1. Update relevant user guide section
2. Add Doxygen comments to code
3. Add test cases to demonstrate usage
4. Update `features.md` status
5. Rebuild Doxygen documentation

### Documentation Checklist

- [ ] User guide section updated
- [ ] Doxygen comments added/updated
- [ ] Test cases added
- [ ] `features.md` updated
- [ ] Python bindings documented (if applicable)
- [ ] C interface documented (if applicable)
- [ ] Examples in user guide tested
- [ ] API reference updated

## Getting Help

### Finding Information

1. **Quick lookup**: `userguide/16-api-reference.md`
2. **Detailed guide**: Relevant section in `userguide/`
3. **API details**: `docs/html/index.html` (Doxygen)
4. **Working examples**: `tests/` directory
5. **Implementation status**: `features.md`

### Common Questions

**Q: How do I use feature X?**
A: Check the user guide section for that category, or search API reference.

**Q: Is feature Y implemented?**
A: Check `features.md` for implementation status.

**Q: How do I build from Python?**
A: See `userguide/17-python-integration.md`.

**Q: Can I use this from C?**
A: Yes! See `c_interop.md` and `userguide/18-c-interface.md`.

**Q: Where are the examples?**
A: Test files in `tests/` serve as comprehensive examples.

## Document Versions

- **User Guide**: Version 1.0 (Complete)
- **C Interface**: Version 1.0 (Specification)
- **API Documentation**: Auto-generated from code
- **Features**: Living document (continuously updated)

## Contributing to Documentation

When contributing:
1. Follow existing structure and style
2. Include code examples that work
3. Add cross-references to related sections
4. Keep examples concise but complete
5. Update table of contents if adding sections

---

**Last Updated**: 2025-11-09  
**Library Version**: 1.4.2  
**Documentation Status**: Complete
