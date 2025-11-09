#include "tensor.h"
#include "tensor_types.h"
#include <iostream>

using namespace tensor4d;

int main() {
    std::cout << "=== GPU Detection Test ===" << std::endl;
    
#ifdef USE_GPU
    std::cout << "USE_GPU is defined" << std::endl;
    if (is_gpu_available()) {
        std::cout << "✓ GPU is available!" << std::endl;
    } else {
        std::cout << "✗ GPU is NOT available" << std::endl;
    }
#else
    std::cout << "USE_GPU is NOT defined" << std::endl;
#endif
    
    // Create a tensor with default constructor (should auto-select GPU)
    Matrixf tensor({100, 100});
    std::cout << "\nCreated tensor with default constructor" << std::endl;
    std::cout << "Tensor uses GPU: " << (tensor.uses_gpu() ? "YES" : "NO") << std::endl;
    std::cout << "Backend: " << backend_name(tensor.backend()) << std::endl;
    
    // Create another tensor explicitly asking for GPU
    Matrixf tensor2({100, 100}, true);
    std::cout << "\nCreated tensor with use_gpu=true" << std::endl;
    std::cout << "Tensor uses GPU: " << (tensor2.uses_gpu() ? "YES" : "NO") << std::endl;
    std::cout << "Backend: " << backend_name(tensor2.backend()) << std::endl;
    
    return 0;
}
