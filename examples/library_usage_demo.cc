/**
 * @file library_usage_demo.cc
 * @brief Demonstration of using the tensor4d library (static/shared)
 * 
 * This example shows how to use the pre-instantiated Vector and Matrix
 * types from the tensor4d library.
 * 
 * Compile with header-only:
 *   g++ -std=c++20 library_usage_demo.cc -Iinclude -lpthread -o demo_header_only
 * 
 * Compile with static library:
 *   g++ -std=c++20 library_usage_demo.cc -Iinclude -Lbuild -ltensor4d -lpthread -o demo_static
 * 
 * Compile with shared library:
 *   g++ -std=c++20 library_usage_demo.cc -Iinclude -Lbuild -ltensor4d -lpthread -o demo_shared
 *   export LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH
 */

#include <iostream>
#include "tensor_types.h"
#include "linalg.h"

using namespace tensor4d;

int main() {
    std::cout << "=== Tensor4D Library Usage Demo ===" << std::endl;
    std::cout << std::endl;

    // ========================================
    // Vector Examples
    // ========================================
    std::cout << "--- Vector Examples ---" << std::endl;
    
    // Create vectors using type aliases
    Vectorf v1({5}, false);  // CPU-only for demo
    Vectorf v2({5}, false);
    
    // Fill with values
    for (size_t i = 0; i < 5; ++i) {
        v1[{i}] = static_cast<float>(i);
        v2[{i}] = 1.0f;
    }
    
    std::cout << "v1 = ";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << v1[{i}] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "v2 = ";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << v2[{i}] << " ";
    }
    std::cout << std::endl;
    
    // Vector operations
    auto v_add_result = v1 + v2;
    if (std::holds_alternative<Vectorf>(v_add_result)) {
        auto v_add = std::get<Vectorf>(v_add_result);
        std::cout << "v1 + v2 = ";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << v_add[{i}] << " ";
        }
        std::cout << std::endl;
    }
    
    auto v_mul_result = v1 * v2;
    if (std::holds_alternative<Vectorf>(v_mul_result)) {
        auto v_mul = std::get<Vectorf>(v_mul_result);
        std::cout << "v1 * v2 = ";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << v_mul[{i}] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << std::endl;

    // ========================================
    // Matrix Examples
    // ========================================
    std::cout << "--- Matrix Examples ---" << std::endl;
    
    // Create matrices
    Matrixf m1({3, 3}, false);  // CPU-only
    Matrixf m2({3, 3}, false);  // CPU-only
    
    // Fill matrix m1
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            m1[{i, j}] = static_cast<float>(i * 3 + j + 1);
        }
    }
    
    // Fill m2 as identity matrix
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            m2[{i, j}] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    std::cout << "m1 (3x3):" << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            std::cout << m1[{i, j}] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nm2 (identity):" << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            std::cout << m2[{i, j}] << " ";
        }
        std::cout << std::endl;
    }
    
    // Matrix addition
    auto m_add_result = m1 + m2;
    if (std::holds_alternative<Matrixf>(m_add_result)) {
        auto m_add = std::get<Matrixf>(m_add_result);
        std::cout << "\nm1 + m2:" << std::endl;
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                std::cout << m_add[{i, j}] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    // Matrix multiplication (element-wise)
    auto m_mul_result = m1 * m2;
    if (std::holds_alternative<Matrixf>(m_mul_result)) {
        auto m_mul = std::get<Matrixf>(m_mul_result);
        std::cout << "\nm1 * m2 (element-wise):" << std::endl;
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                std::cout << m_mul[{i, j}] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    // Matrix-matrix multiplication using linalg
    auto m_matmul = linalg::matmul(m1, m2);
    std::cout << "\nmatmul(m1, m2):" << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            std::cout << m_matmul[{i, j}] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "=== Demo completed successfully ===" << std::endl;
    
    return 0;
}
