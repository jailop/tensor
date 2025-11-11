// A simple test program for tensor4d library
// This program creates two 3x3 matrices, adds them together, and prints
// the result.
//
// To compile this program, use the following command:
//
// g++ -o test test.cc -std=c++20 -ltensor4d -ltbb
//
// TODO: how to compile with gpu support?
//
// SUGGESTION: GPU support is enabled at library build time, not compile time.
// If the library was built with CUDA support (USE_GPU flag), GPU acceleration
// is automatically available. To verify GPU support:
//
//   1. Build library with: cmake -DUSE_GPU=ON ..  
//   2. Check at runtime:
//   if (is_gpu_available()) { /* GPU is ready */ } 
//   3. No special
//   compiler flags needed - tensors use GPU by default when available
//
// The library automatically selects: GPU > BLAS > CPU based on
// availability.

#include <tensor4d/tensor.h>
#include <tensor4d/tensor_types.h>
#include <tensor4d/tensor_io.h>
#include <iostream>

using namespace tensor4d;

int main() {
    // TODO: There should a more idiomatic way to do this to create and
    // fill matrices.
    //
    // SUGGESTION: The library provides NumPy-style factory functions:
    //
    //   - ones<float, 2>({3, 3})    // Matrix filled with 1.0
    //   - zeros<float, 2>({3, 3})   // Matrix filled with 0.0
    //   - arange<float>(0, 10, 1)   // 1D tensor: [0, 1, 2, ..., 9]
    //   - linspace<float>(0, 1, 5)  // 1D tensor: [0.0, 0.25, 0.5, 0.75, 1.0]
    //
    // For custom values, constructor + fill() is the idiomatic way:
    //
    //   Matrixf B({3, 3}); B.fill(2.0f);
    //
    // Alternatively, use scalar multiplication:
    //
    //   auto B = ones<float, 2>({3, 3}) * 2.0f;
    auto A = ones<float, 2>({3, 3});
    auto B = Matrixf({3, 3});
    B.fill(2.0f);
    // Add them together
    auto C_var = A + B;
    // Check for errors
    // TODO: A macro or a helper can be added to make this easier to
    // read. More in the C++ idiomatic way.
    //
    // SUGGESTION: Consider adding helpers like:
    //
    //   1. Macro: TENSOR_TRY(result, expression)
    //      Example: TENSOR_TRY(C, A + B) { /* use C */ } else { /* error */ }
    //   2. Helper function: auto C = unwrap_or_throw(A + B);
    //   3. Monadic style: result.and_then([](auto& t) { /* use t */ });
    //   4. Pattern: if (auto* tensor = std::get_if<Matrixf>(&C_var)) { /* use tensor */ }
    // Current pattern is explicit and safe, but verbose for simple cases.
    if (std::holds_alternative<TensorError>(C_var)) {
        std::cerr << "Error!" << std::endl;
        return 1;
    }
    // Extract result
    auto C = std::get<Matrixf>(C_var);
    // Print the result
    std::cout << "A + B = " << std::endl;
    // TODO: the printing interface is ugly
    //
    // SUGGESTION: Consider adding more user-friendly interfaces:
    //
    //   1. Overload operator<<: std::cout << C << std::endl;
    //   2. Method wrapper: C.print() or C.print(precision=4)
    //   3. Named parameter idiom: print(C).precision(4).max_items(10)
    //   4. Default arguments: print(C) // uses sensible defaults
    //
    // Current signature: print(tensor, stream, precision, max_items, line_width)
    // The explicit parameters give full control but are verbose for simple cases.
    print(C, std::cout, 4, 6, 75);
    // Access elements
    std::cout << "C[0,0] = " << C[{0, 0}] << std::endl;
    std::cout << "C[1,1] = " << C[{1, 1}] << std::endl;
    return 0;
}
