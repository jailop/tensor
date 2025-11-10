/**
 * @file broadcasting_test.cc
 * @brief Test broadcasting operations in tensor library
 */

#include "tensor.h"
#include "tensor_types.h"
#include <iostream>
#include <iomanip>

using namespace tensor4d;

int main() {
    std::cout << "=== Tensor Broadcasting Test ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    
    // Test 1: Add vector to matrix (row broadcasting)
    std::cout << "\n1. Broadcasting vector to matrix rows:" << std::endl;
    {
        Matrixf matrix({3, 4});
        Matrixf bias({1, 4});  // Row vector
        
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                matrix[{i, j}] = static_cast<float>(i * 4 + j);
            }
        }
        
        for (size_t j = 0; j < 4; ++j) {
            bias[{0, j}] = 10.0f + j;
        }
        
        auto result_var = matrix + bias;
        if (std::holds_alternative<Matrixf>(result_var)) {
            auto result = std::get<Matrixf>(result_var);
            std::cout << "   Matrix (3x4) + Bias (1x4):" << std::endl;
            std::cout << "   Result[0,0] = " << result[{0, 0}] << " (expected 10.0)" << std::endl;
            std::cout << "   Result[2,3] = " << result[{2, 3}] << " (expected 24.0)" << std::endl;
        }
    }
    
    // Test 2: Element-wise multiplication with broadcasting
    std::cout << "\n2. Broadcasting multiplication:" << std::endl;
    {
        Matrixf matrix({2, 3});
        Matrixf scale({1, 3});
        
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                matrix[{i, j}] = 1.0f;
            }
        }
        
        for (size_t j = 0; j < 3; ++j) {
            scale[{0, j}] = static_cast<float>(j + 1);
        }
        
        auto result_var = matrix * scale;
        if (std::holds_alternative<Matrixf>(result_var)) {
            auto result = std::get<Matrixf>(result_var);
            std::cout << "   Matrix (2x3) * Scale (1x3):" << std::endl;
            std::cout << "   Result: [" << result[{0, 0}] << ", " << result[{0, 1}] 
                      << ", " << result[{0, 2}] << "]" << std::endl;
            std::cout << "   Expected: [1.0, 2.0, 3.0]" << std::endl;
        }
    }
    
    // Test 3: Subtraction with broadcasting
    std::cout << "\n3. Broadcasting subtraction (mean centering):" << std::endl;
    {
        Matrixf data({2, 3});
        Matrixf mean({1, 3});
        
        data[{0, 0}] = 5.0f; data[{0, 1}] = 10.0f; data[{0, 2}] = 15.0f;
        data[{1, 0}] = 15.0f; data[{1, 1}] = 20.0f; data[{1, 2}] = 25.0f;
        
        mean[{0, 0}] = 10.0f; mean[{0, 1}] = 15.0f; mean[{0, 2}] = 20.0f;
        
        auto result_var = data - mean;
        if (std::holds_alternative<Matrixf>(result_var)) {
            auto result = std::get<Matrixf>(result_var);
            std::cout << "   Data (2x3) - Mean (1x3):" << std::endl;
            std::cout << "   Row 0: [" << result[{0, 0}] << ", " << result[{0, 1}] 
                      << ", " << result[{0, 2}] << "]" << std::endl;
            std::cout << "   Row 1: [" << result[{1, 0}] << ", " << result[{1, 1}] 
                      << ", " << result[{1, 2}] << "]" << std::endl;
            std::cout << "   Expected Row 0: [-5.0, -5.0, -5.0]" << std::endl;
            std::cout << "   Expected Row 1: [5.0, 5.0, 5.0]" << std::endl;
        }
    }
    
    // Test 4: Division with broadcasting (normalization)
    std::cout << "\n4. Broadcasting division (normalization):" << std::endl;
    {
        Matrixf data({2, 2});
        Matrixf std_dev({1, 2});
        
        data[{0, 0}] = 10.0f; data[{0, 1}] = 20.0f;
        data[{1, 0}] = 30.0f; data[{1, 1}] = 40.0f;
        
        std_dev[{0, 0}] = 2.0f; std_dev[{0, 1}] = 4.0f;
        
        auto result_var = data / std_dev;
        if (std::holds_alternative<Matrixf>(result_var)) {
            auto result = std::get<Matrixf>(result_var);
            std::cout << "   Data (2x2) / StdDev (1x2):" << std::endl;
            std::cout << "   Result[0,0] = " << result[{0, 0}] << " (expected 5.0)" << std::endl;
            std::cout << "   Result[1,1] = " << result[{1, 1}] << " (expected 10.0)" << std::endl;
        }
    }
    
    std::cout << "\n=== All broadcasting tests completed ===" << std::endl;
    
    return 0;
}
