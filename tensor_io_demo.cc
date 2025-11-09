/**
 * @file tensor_io_demo.cc
 * @brief Demonstration of tensor I/O operations.
 */

#include <iostream>
#include "tensor.h"
#include "tensor_io.h"

int main() {
    std::cout << "=== Tensor I/O Operations Demo ===\n\n";
    
    // Create a 2D tensor
    std::cout << "1. Creating a 3x4 tensor...\n";
    Tensor<float, 2> tensor({3, 4});
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            tensor[{i, j}] = static_cast<float>(i * 4 + j + 1);
        }
    }
    
    // Print the tensor
    std::cout << "\nOriginal tensor:\n";
    print(tensor, std::cout, 2);
    
    // Save to binary format
    std::cout << "\n2. Saving to binary format (tensor.tnsr)...\n";
    if (save_binary(tensor, "tensor.tnsr")) {
        std::cout << "Successfully saved to binary format\n";
    }
    
    // Save to text format
    std::cout << "\n3. Saving to text format (tensor.txt)...\n";
    if (save_text(tensor, "tensor.txt", 2)) {
        std::cout << "Successfully saved to text format\n";
    }
    
    // Save to NumPy format
    std::cout << "\n4. Saving to NumPy format (tensor.npy)...\n";
    if (save_npy(tensor, "tensor.npy")) {
        std::cout << "Successfully saved to NumPy format\n";
        std::cout << "You can load this file in Python with: numpy.load('tensor.npy')\n";
    }
    
    // Load from binary format
    std::cout << "\n5. Loading from binary format...\n";
    auto result = load_binary<float, 2>("tensor.tnsr");
    if (std::holds_alternative<Tensor<float, 2>>(result)) {
        auto loaded = std::get<Tensor<float, 2>>(result);
        std::cout << "Successfully loaded tensor:\n";
        print(loaded, std::cout, 2);
        
        // Verify data
        bool match = true;
        for (size_t i = 0; i < 3 && match; ++i) {
            for (size_t j = 0; j < 4 && match; ++j) {
                if (loaded[{i, j}] != tensor[{i, j}]) {
                    match = false;
                }
            }
        }
        std::cout << "\nData verification: " << (match ? "PASSED" : "FAILED") << "\n";
    } else {
        std::cout << "Failed to load tensor\n";
    }
    
    // Demonstrate auto-detection
    std::cout << "\n6. Auto-detecting format and loading...\n";
    auto result2 = load<float, 2>("tensor.tnsr");
    if (std::holds_alternative<Tensor<float, 2>>(result2)) {
        std::cout << "Successfully auto-detected binary format and loaded\n";
    }
    
    // Demonstrate to_string
    std::cout << "\n7. Converting tensor to string:\n";
    Tensor<float, 1> vec({5});
    for (size_t i = 0; i < 5; ++i) {
        vec[{i}] = static_cast<float>(i * 10);
    }
    std::string vec_str = to_string(vec, 1);
    std::cout << vec_str;
    
    // Demonstrate truncated printing for large tensors
    std::cout << "\n8. Printing large tensor (10x10) with truncation:\n";
    Tensor<float, 2> large({10, 10});
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            large[{i, j}] = static_cast<float>(i * 10 + j);
        }
    }
    print(large, std::cout, 1, 6);  // Print with truncation at 6 items
    
    // Clean up
    std::remove("tensor.tnsr");
    std::remove("tensor.txt");
    std::remove("tensor.npy");
    
    std::cout << "\n=== Demo Complete ===\n";
    return 0;
}
