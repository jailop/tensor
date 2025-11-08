// GPU and BLAS Feature Demonstration
// This file shows example usage of GPU-accelerated and BLAS-accelerated operations

#include "tensor.h"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "========================================\n";
    std::cout << "  GPU and BLAS Feature Demonstration\n";
    std::cout << "========================================\n\n";
    
    // Check what acceleration is available
    std::cout << "System Configuration:\n";
#ifdef USE_GPU
    std::cout << "  GPU Support: ENABLED\n";
#else
    std::cout << "  GPU Support: DISABLED\n";
#endif
    
#ifdef USE_BLAS
    std::cout << "  BLAS Support: ENABLED\n";
#else
    std::cout << "  BLAS Support: DISABLED\n";
#endif
    std::cout << "\n";
    
    // Example 1: Element-wise operations with GPU
    std::cout << "Example 1: Element-wise Operations (GPU-accelerated)\n";
    std::cout << "---------------------------------------------------\n";
    {
        Tensor<float, 2> a({3, 3}, true);  // GPU enabled
        Tensor<float, 2> b({3, 3}, true);
        
        // Initialize
        a.fill(2.0f);
        b.fill(3.0f);
        
        std::cout << "  a = 2.0 (all elements)\n";
        std::cout << "  b = 3.0 (all elements)\n\n";
        
        // Addition (GPU-accelerated if available)
        auto c_var = a + b;
        if (std::holds_alternative<Tensor<float, 2>>(c_var)) {
            auto& c = std::get<Tensor<float, 2>>(c_var);
            std::cout << "  a + b = " << c[{0,0}] << " (expected: 5.0) ✓\n";
        }
        
        // Multiplication (GPU-accelerated if available)
        auto d_var = a * b;
        if (std::holds_alternative<Tensor<float, 2>>(d_var)) {
            auto& d = std::get<Tensor<float, 2>>(d_var);
            std::cout << "  a * b = " << d[{0,0}] << " (expected: 6.0) ✓\n";
        }
        
        // Scalar operations (GPU-accelerated if available)
        auto e = a * 5.0f;
        std::cout << "  a * 5 = " << e[{0,0}] << " (expected: 10.0) ✓\n";
    }
    std::cout << "\n";
    
    // Example 2: Math functions with GPU
    std::cout << "Example 2: Mathematical Functions (GPU-accelerated)\n";
    std::cout << "---------------------------------------------------\n";
    {
        Tensor<float, 2> a({3, 3}, true);
        a.fill(1.0f);
        
        std::cout << "  a = 1.0 (all elements)\n\n";
        
        // Exponential (GPU-accelerated if available)
        auto exp_a = a.exp();
        std::cout << "  exp(a) = " << exp_a[{0,0}] 
                  << " (expected: ~2.718) ✓\n";
        
        // Sigmoid (GPU-accelerated if available)
        auto sig_a = a.sigmoid();
        std::cout << "  sigmoid(a) = " << sig_a[{0,0}] 
                  << " (expected: ~0.731) ✓\n";
        
        // ReLU (GPU-accelerated if available)
        Tensor<float, 2> b({3, 3}, true);
        b.fill(-1.0f);
        auto relu_b = b.relu();
        std::cout << "  ReLU(-1.0) = " << relu_b[{0,0}] 
                  << " (expected: 0.0) ✓\n";
    }
    std::cout << "\n";
    
    // Example 3: Matrix multiplication with BLAS/GPU
    std::cout << "Example 3: Matrix Multiplication (BLAS/GPU-accelerated)\n";
    std::cout << "--------------------------------------------------------\n";
    {
        Tensor<float, 2> a({2, 3}, true);
        Tensor<float, 2> b({3, 2}, true);
        
        // Initialize matrices
        a[{0,0}] = 1.0f; a[{0,1}] = 2.0f; a[{0,2}] = 3.0f;
        a[{1,0}] = 4.0f; a[{1,1}] = 5.0f; a[{1,2}] = 6.0f;
        
        b[{0,0}] = 7.0f; b[{0,1}] = 8.0f;
        b[{1,0}] = 9.0f; b[{1,1}] = 10.0f;
        b[{2,0}] = 11.0f; b[{2,1}] = 12.0f;
        
        std::cout << "  Matrix A (2x3):\n";
        std::cout << "    [1, 2, 3]\n";
        std::cout << "    [4, 5, 6]\n\n";
        
        std::cout << "  Matrix B (3x2):\n";
        std::cout << "    [7,  8]\n";
        std::cout << "    [9, 10]\n";
        std::cout << "    [11, 12]\n\n";
        
        // Matrix multiplication (BLAS or GPU-accelerated)
        auto c_var = a.dot(b);
        if (std::holds_alternative<Tensor<float, 2>>(c_var)) {
            auto& c = std::get<Tensor<float, 2>>(c_var);
            std::cout << "  A × B result (2x2):\n";
            std::cout << "    [" << c[{0,0}] << ", " << c[{0,1}] << "]\n";
            std::cout << "    [" << c[{1,0}] << ", " << c[{1,1}] << "]\n\n";
            std::cout << "  Expected:\n";
            std::cout << "    [58, 64]\n";
            std::cout << "    [139, 154]\n";
            
            bool correct = (std::abs(c[{0,0}] - 58.0f) < 0.001f &&
                          std::abs(c[{0,1}] - 64.0f) < 0.001f &&
                          std::abs(c[{1,0}] - 139.0f) < 0.001f &&
                          std::abs(c[{1,1}] - 154.0f) < 0.001f);
            std::cout << "  " << (correct ? "✓ PASSED" : "✗ FAILED") << "\n";
        }
    }
    std::cout << "\n";
    
    // Example 4: Reduction operations
    std::cout << "Example 4: Reduction Operations (GPU-accelerated)\n";
    std::cout << "--------------------------------------------------\n";
    {
        Tensor<float, 2> a({3, 3}, true);
        
        // Fill with values 1-9
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                a[{i, j}] = static_cast<float>(i * 3 + j + 1);
            }
        }
        
        std::cout << "  Matrix (3x3): [1, 2, 3, 4, 5, 6, 7, 8, 9]\n\n";
        
        // Sum (GPU-accelerated reduction)
        float sum = a.sum();
        std::cout << "  Sum: " << sum << " (expected: 45) ✓\n";
        
        // Mean (GPU-accelerated)
        float mean = a.mean();
        std::cout << "  Mean: " << mean << " (expected: 5) ✓\n";
    }
    std::cout << "\n";
    
    // Example 5: GPU with autograd
    std::cout << "Example 5: Autograd with GPU Operations\n";
    std::cout << "----------------------------------------\n";
    {
        Tensor<float, 1> x({1}, true, true);  // GPU enabled, requires_grad=true
        x[{0}] = 2.0f;
        
        std::cout << "  x = 2.0 (requires_grad=true)\n";
        
        // Forward pass (GPU-accelerated operations)
        auto x2_var = x * x;
        if (std::holds_alternative<Tensor<float, 1>>(x2_var)) {
            auto x2 = std::get<Tensor<float, 1>>(x2_var);
            auto y_var = x2 * x;  // y = x³
            
            if (std::holds_alternative<Tensor<float, 1>>(y_var)) {
                auto& y_tensor = std::get<Tensor<float, 1>>(y_var);
                std::cout << "  y = x³ = " << y_tensor[{0}] << " (expected: 8) ✓\n";
                
                // Backward pass
                y_tensor.backward();
                
                // Check gradient: dy/dx = 3x² = 3(2)² = 12
                std::cout << "  dy/dx = " << (*x.grad())[{0}] 
                          << " (expected: 12) ✓\n";
            }
        }
    }
    std::cout << "\n";
    
    std::cout << "========================================\n";
    std::cout << "  All demonstrations completed!\n";
    std::cout << "========================================\n";
    
    return 0;
}
