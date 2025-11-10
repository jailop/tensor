#include "nn_layers.h"
#include "tensor_types.h"
#include <iostream>

using namespace tensor4d;
using namespace tensor4d::nn;

int main() {
    std::cout << "=== Simple Auto-GPU Demo ===" << std::endl;
    
    // Check what backend will be used
#ifdef USE_GPU
    if (is_gpu_available()) {
        std::cout << "✓ GPU will be used automatically\n" << std::endl;
    } else {
        std::cout << "Using CPU/BLAS (no GPU available)\n" << std::endl;
    }
#else
    std::cout << "Using CPU/BLAS (compiled without GPU)\n" << std::endl;
#endif
    
    // Create a simple neural network - GPU is automatic!
    std::cout << "Creating neural network layers..." << std::endl;
    Linearf fc1(784, 128, true);  // No device parameter needed!
    ReLUf relu;
    Linearf fc2(128, 10, true);
    Softmaxf softmax;
    
    std::cout << "✓ Layers created" << std::endl;
    
    // Create input tensor - GPU is automatic!
    std::cout << "\nCreating input tensor..." << std::endl;
    Matrixf input({32, 784});  // Batch of 32, 784 features
    input.fill(0.5f);
    
    std::cout << "✓ Tensor created" << std::endl;
    std::cout << "  Backend: " << backend_name(input.backend()) << std::endl;
    std::cout << "  Uses GPU: " << (input.uses_gpu() ? "YES" : "NO") << std::endl;
    
    // Forward pass
    std::cout << "\nRunning forward pass..." << std::endl;
    auto h1 = fc1.forward(input);
    auto a1 = relu.forward(h1);
    auto h2 = fc2.forward(a1);
    auto output = softmax.forward(h2);
    
    std::cout << "✓ Forward pass complete" << std::endl;
    std::cout << "  Output shape: " << output.shape()[0] << "x" << output.shape()[1] << std::endl;
    std::cout << "  Output backend: " << backend_name(output.backend()) << std::endl;
    
    std::cout << "\n=== All operations used optimal backend automatically! ===" << std::endl;
    
    return 0;
}
