/**
 * @file cross_product_demo.cc
 * @brief Demonstrates the cross product operation for 3D vectors
 */

#include "include/tensor.h"
#include <iostream>
#include <iomanip>

using namespace tensor;

int main() {
    std::cout << "=== Cross Product Demo ===" << std::endl << std::endl;

    // Example 1: Basic orthogonal vectors
    std::cout << "Example 1: Standard basis vectors" << std::endl;
    Tensor<float, 1> i({3});
    i[{0}] = 1.0f; i[{1}] = 0.0f; i[{2}] = 0.0f;  // i = [1, 0, 0]
    
    Tensor<float, 1> j({3});
    j[{0}] = 0.0f; j[{1}] = 1.0f; j[{2}] = 0.0f;  // j = [0, 1, 0]
    
    auto k_var = i.cross(j);  // k = i × j should be [0, 0, 1]
    
    if (std::holds_alternative<Tensor<float, 1>>(k_var)) {
        auto& k = std::get<Tensor<float, 1>>(k_var);
        std::cout << "i × j = [" << k[{0}] << ", " << k[{1}] << ", " << k[{2}] << "]" << std::endl;
        std::cout << "Expected: [0, 0, 1]" << std::endl << std::endl;
    }

    // Example 2: General vectors
    std::cout << "Example 2: General vectors" << std::endl;
    Tensor<float, 1> a({3});
    a[{0}] = 2.0f; a[{1}] = 3.0f; a[{2}] = 4.0f;
    
    Tensor<float, 1> b({3});
    b[{0}] = 5.0f; b[{1}] = 6.0f; b[{2}] = 7.0f;
    
    auto c_var = a.cross(b);
    
    if (std::holds_alternative<Tensor<float, 1>>(c_var)) {
        auto& c = std::get<Tensor<float, 1>>(c_var);
        std::cout << "a = [" << a[{0}] << ", " << a[{1}] << ", " << a[{2}] << "]" << std::endl;
        std::cout << "b = [" << b[{0}] << ", " << b[{1}] << ", " << b[{2}] << "]" << std::endl;
        std::cout << "a × b = [" << c[{0}] << ", " << c[{1}] << ", " << c[{2}] << "]" << std::endl;
        
        // Verify perpendicularity with dot product
        auto dot_a = a.dot(c);
        auto dot_b = b.dot(c);
        
        if (std::holds_alternative<float>(dot_a) && std::holds_alternative<float>(dot_b)) {
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "a · (a × b) = " << std::get<float>(dot_a) << " (should be ~0)" << std::endl;
            std::cout << "b · (a × b) = " << std::get<float>(dot_b) << " (should be ~0)" << std::endl;
        }
        std::cout << std::endl;
    }

    // Example 3: Physics application - torque
    std::cout << "Example 3: Physics - Computing torque" << std::endl;
    std::cout << "τ = r × F (torque = position × force)" << std::endl;
    
    Tensor<float, 1> r({3});  // Position vector (in meters)
    r[{0}] = 0.5f; r[{1}] = 0.3f; r[{2}] = 0.0f;
    
    Tensor<float, 1> F({3});  // Force vector (in Newtons)
    F[{0}] = 0.0f; F[{1}] = 0.0f; F[{2}] = -10.0f;
    
    auto torque_var = r.cross(F);
    
    if (std::holds_alternative<Tensor<float, 1>>(torque_var)) {
        auto& torque = std::get<Tensor<float, 1>>(torque_var);
        std::cout << "Position r = [" << r[{0}] << ", " << r[{1}] << ", " << r[{2}] << "] m" << std::endl;
        std::cout << "Force F = [" << F[{0}] << ", " << F[{1}] << ", " << F[{2}] << "] N" << std::endl;
        std::cout << "Torque τ = [" << torque[{0}] << ", " << torque[{1}] << ", " 
                  << torque[{2}] << "] N·m" << std::endl;
    }

    std::cout << "\n=== Backend Information ===" << std::endl;
    std::cout << "Active backend: " << backend_name(get_active_backend()) << std::endl;
    
    return 0;
}
