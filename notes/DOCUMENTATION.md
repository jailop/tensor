# Tensor Library Documentation

## Overview

This project includes comprehensive API documentation generated with **Doxygen**. The documentation covers all classes, functions, namespaces, and types in the tensor library.

## Features Documented

### Core Components
- **Tensor Class**: Multi-dimensional arrays with GPU, BLAS, and autograd support
- **Linear Algebra**: Vector and matrix operations (norm, dot, matmul, inverse, etc.)
- **Tensor Operations**: Broadcasting, reduction, softmax, and other neural network operations
- **Autograd**: Automatic differentiation for gradient computation
- **Loss Functions**: MSE, cross-entropy, binary cross-entropy, L1, smooth L1
- **Optimizers**: SGD, Adam, RMSprop with momentum and weight decay

### Advanced Features
- **GPU Acceleration**: CUDA support for tensor operations
- **BLAS Integration**: Optimized CPU operations using BLAS libraries
- **Broadcasting**: NumPy-style broadcasting for element-wise operations
- **Views and Slices**: Memory-efficient tensor sub-regions
- **Statistical Operations**: Mean, variance, standard deviation, correlation
- **Mathematical Functions**: exp, log, sin, cos, tanh, sigmoid, relu, etc.

## Building Documentation

### Prerequisites
- **Doxygen** (version 1.9+)
- **Graphviz** (optional, for call graphs and diagrams)

On Ubuntu/Debian:
```bash
sudo apt-get install doxygen graphviz
```

On macOS:
```bash
brew install doxygen graphviz
```

### Generate Documentation

From the project root:

```bash
# Create build directory and run CMake
mkdir -p build && cd build
cmake ..

# Generate documentation
make doc

# Or generate and open in browser (Linux/macOS)
make doc_open
```

The documentation will be generated in the `docs/html/` directory.

### View Documentation

Open `docs/html/index.html` in your web browser:

```bash
# Linux
xdg-open docs/html/index.html

# macOS
open docs/html/index.html

# Windows
start docs/html/index.html
```

## Documentation Structure

### Main Sections

1. **Main Page** (`README.md`): Project overview and quick start
2. **Classes**: All tensor, optimizer, and utility classes
3. **Namespaces**: 
   - `loss`: Loss functions
   - `linalg`: Linear algebra operations
   - `tensor_ops`: Core tensor operations
4. **Files**: Header file documentation with detailed function descriptions
5. **Examples**: Code examples showing usage patterns

### Key Documentation Pages

- **Tensor Class**: Core multi-dimensional array implementation
  - Constructors and memory management
  - Element access and indexing
  - Arithmetic operations
  - Mathematical functions
  - Autograd methods
  - Statistical operations

- **Linear Algebra** (`linalg.h`):
  - Vector operations (norm, dot, cross)
  - Matrix operations (matmul, transpose, inverse, determinant)
  - Decompositions (LU, Cholesky, QR, SVD)
  - Eigenvalue computation
  - Linear system solving

- **Tensor Operations** (`tensor_ops.h`):
  - Broadcasting rules and implementation
  - Reduction operations (sum, mean, max, min)
  - Softmax and log-softmax
  - Matrix multiplication

- **Loss Functions** (`loss_functions.h`):
  - MSE Loss (regression)
  - Cross Entropy Loss (multi-class classification)
  - Binary Cross Entropy (binary classification)
  - L1 Loss (robust regression)
  - Smooth L1 Loss (Huber loss)

- **Optimizers** (`optimizers.h`):
  - SGD with momentum
  - Adam optimizer
  - RMSprop optimizer
  - Learning rate scheduling

## Navigation Tips

### Search Functionality
Use the search box in the top-right corner to quickly find:
- Classes: `Tensor`, `Optimizer`, `SGD`, `Adam`
- Functions: `matmul`, `softmax`, `norm`, `inverse`
- Namespaces: `loss`, `linalg`, `tensor_ops`

### Class Documentation
Each class page includes:
- **Brief description** at the top
- **Detailed description** with usage examples
- **Member functions** with parameters and return values
- **Call graphs** showing function relationships (if Graphviz is enabled)
- **Code examples** demonstrating usage

### Function Documentation
Each function includes:
- **Brief description**
- **Template parameters** (for template functions)
- **Parameters** with types and descriptions
- **Return value** type and description
- **Exceptions** that may be thrown
- **Usage examples**
- **See also** links to related functions

## Code Examples in Documentation

The documentation includes extensive code examples:

### Basic Tensor Operations
```cpp
// Create and manipulate tensors
Tensor<float, 2> matrix({3, 4});
matrix.fill(1.0f);

// Element-wise operations
auto result = matrix + 2.0f;
auto product = matrix * matrix;
```

### Autograd Example
```cpp
// Enable gradient tracking
Tensor<float, 2> x({2, 2}, true, true);
auto y = x * x + x;
y.sum().backward();
auto gradients = x.grad();
```

### Linear Algebra
```cpp
Matrix<float> A({3, 3});
Vector<float> v({3});

// Matrix operations
auto inv_A = linalg::inverse(A);
auto det = linalg::determinant(A);
auto norm_v = linalg::norm(v);
```

### Neural Network Training
```cpp
// Setup
Matrix<float> weights({784, 10}, true, true);
SGD<float> optimizer({&weights}, 0.01, 0.9);

// Training loop
for (auto& batch : training_data) {
    auto output = matmul(batch, weights);
    auto loss = loss::cross_entropy_loss(output, labels);
    
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
}
```

## Customizing Documentation

### Modify Doxyfile

Edit `Doxyfile` in the project root to customize:
- Project name and version
- Output directory
- Which files to include
- Extraction options
- Graph generation
- Output format (HTML, LaTeX, XML)

Key configuration options:

```
PROJECT_NAME           = "Tensor Library"
PROJECT_BRIEF          = "High-performance C++ tensor library"
OUTPUT_DIRECTORY       = docs
INPUT                  = include/ README.md
EXTRACT_ALL            = YES
GENERATE_LATEX         = NO
HAVE_DOT               = YES
CALL_GRAPH             = YES
```

### Add More Documentation

To add documentation to your code:

1. **File-level documentation**:
```cpp
/**
 * @file myfile.h
 * @brief Brief description
 * @author Your Name
 */
```

2. **Class documentation**:
```cpp
/**
 * @class MyClass
 * @brief Brief description
 * @details Detailed description with examples
 */
```

3. **Function documentation**:
```cpp
/**
 * @brief Brief description
 * @param param1 Description of param1
 * @param param2 Description of param2
 * @return Description of return value
 * @throws ExceptionType Description of when thrown
 */
```

4. **Code examples**:
```cpp
/**
 * @code
 * MyClass obj;
 * obj.doSomething();
 * @endcode
 */
```

## Troubleshooting

### Doxygen Not Found
If CMake cannot find Doxygen:
```bash
# Check if installed
which doxygen

# Install if missing
sudo apt-get install doxygen  # Ubuntu/Debian
brew install doxygen          # macOS
```

### Missing Graphs
If call graphs are not generated:
```bash
# Check if Graphviz is installed
which dot

# Install Graphviz
sudo apt-get install graphviz  # Ubuntu/Debian
brew install graphviz          # macOS
```

### Incomplete Documentation
If some items are not documented:
- Check that `EXTRACT_ALL = YES` in Doxyfile
- Ensure files are in the `INPUT` path
- Add `/** */` style comments to undocumented items

## Contributing

When adding new features:
1. Document all public APIs with Doxygen comments
2. Include usage examples in `@code` blocks
3. Document parameters, return values, and exceptions
4. Regenerate documentation and verify it looks correct
5. Keep README.md in sync with major changes

## Resources

- **Doxygen Manual**: https://www.doxygen.nl/manual/
- **Doxygen Comments**: https://www.doxygen.nl/manual/docblocks.html
- **Markdown in Doxygen**: https://www.doxygen.nl/manual/markdown.html
- **Graphviz**: https://graphviz.org/

## License

The documentation is part of the Tensor Library project and follows the same license as the source code.
