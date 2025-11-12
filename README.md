# Tensor Library

**Disclaimer**: This is a personal project to learn about numerical
computing and optimization techniques. It is only intended to be used
for educative purposes and not at all for production use.

**Warning**: The library is under permanent development and the API may
change without notice. The API is still too ugly. Backward compatibility
is not guaranteed at all. As an example of the thinking/discussions
behind some design choices, see the notes in [this example](https://github.com/jailop/tensor/blob/main/examples/example01.cc).

## Overview

A multi-dimensional tensor library for scientific computing and machine
learning, written in C++ with an in-progress C/Python API. It features
automatic GPU acceleration, optimized CPU operations, and support for
building basic neural networks.

- [User Guide](https://github.com/jailop/tensor/blob/main/userguide/00-index.md)
- [API Documentation](https://jailop.github.io/tensor/html)

## Acknowledgments

This project was created to explore:

- High-performance numerical computing
- GPU programming with CUDA
- BLAS and LAPACK integration
- Template metaprogramming in C++
- Neural network implementation from scratch
- Library design and API ergonomics
- Multi-language bindings (C, Python)

I want to express my admiration and gratitude for the
people who have created or are maintaining numerical libraries that
power much of today's scientific and business computing. Everyday, I get
surprised by the genius, dedication, and effort that has been put those
projects, and by the quality and performance they achieve.

### Key Features

The most of the features listed below are wrappers of well-known
libraries (BLAS, LAPACK, CUDA) or adaptations of standard patterns found
in other libraries (NumPy, PyTorch, Eigen, Armadillo)

* Core Tensor Operations
* Performance & Backends
* Mathematical Operations
* Linear Algebra (wrapping third-party libraries)
* Neural Networks & Deep Learning
* Data I/O & Serialization
* Multi-Language Support
* Backend Selection

