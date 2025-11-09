/**
 * @file backend_demo.cc
 * @brief Demonstrates automatic backend selection (GPU → BLAS → CPU)
 * 
 * This program shows how the tensor library automatically selects the best
 * available computational backend at runtime.
 */

#include "tensor.h"
#include <iostream>

int main() {
    std::cout << "=== Tensor Library Backend Demo ===\n\n";
    
    // Check what backends are available at compile time
    std::cout << "Compile-time backend support:\n";
#ifdef USE_GPU
    std::cout << "  GPU (CUDA): Compiled in\n";
#else
    std::cout << "  GPU (CUDA): Not compiled\n";
#endif
    
#ifdef USE_BLAS
    std::cout << "  BLAS: Compiled in\n";
#else
    std::cout << "  BLAS: Not compiled\n";
#endif
    std::cout << "  CPU: Always available (fallback)\n\n";
    
    // Check what backends are actually available at runtime
    std::cout << "Runtime backend availability:\n";
    std::cout << "  GPU available: " << (is_gpu_available() ? "Yes" : "No") << "\n";
    std::cout << "  BLAS available: " << (is_blas_available() ? "Yes" : "No") << "\n";
    std::cout << "  CPU available: Yes (always)\n\n";
    
    // Get the active backend
    Backend active = get_active_backend();
    std::cout << "Active backend (default for new tensors): " 
              << backend_name(active) << "\n\n";
    
    // Create tensors and show which backend they use
    std::cout << "Creating tensors:\n";
    
    Tensor<float, 2> t1({3, 3});  // Default: use_gpu=true (will use GPU if available)
    std::cout << "  Tensor<float, 2> t1 (default settings): " 
              << backend_name(t1.backend()) << "\n";
    
    Tensor<float, 2> t2({3, 3}, true);  // Explicitly request GPU
    std::cout << "  Tensor<float, 2> t2 (use_gpu=true): " 
              << backend_name(t2.backend()) << "\n";
    
    Tensor<float, 2> t3({3, 3}, false);  // Explicitly request CPU/BLAS
    std::cout << "  Tensor<float, 2> t3 (use_gpu=false): " 
              << backend_name(t3.backend()) << "\n";
    
    // Perform a simple operation
    std::cout << "\nPerforming matrix operation (t1 + t2):\n";
    t1.fill(1.0f);
    t2.fill(2.0f);
    
    auto result_var = t1 + t2;
    if (std::holds_alternative<Tensor<float, 2>>(result_var)) {
        auto& result = std::get<Tensor<float, 2>>(result_var);
        std::cout << "  Result backend: " << backend_name(result.backend()) << "\n";
        std::cout << "  Result[0,0]: " << result[{0, 0}] << " (expected: 3.0)\n";
    }
    
    std::cout << "\n=== Backend Selection Summary ===\n";
    std::cout << "The library uses the following priority:\n";
    std::cout << "  1. GPU (CUDA) - if available and use_gpu=true (default)\n";
    std::cout << "  2. BLAS - if GPU not available and BLAS compiled in\n";
    std::cout << "  3. CPU - fallback when neither GPU nor BLAS available\n";
    std::cout << "\nYou can override by setting use_gpu=false in constructor.\n";
    
    return 0;
}
