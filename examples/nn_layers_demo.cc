/**
 * @file nn_layers_demo.cc
 * @brief Demo of neural network layers in C++
 */

#include "nn_layers.h"
#include "tensor_types.h"
#include <iostream>
#include <iomanip>

using namespace tensor;

int main() {
    std::cout << "=== Neural Network Layers Demo ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    
    // Example 1: Linear Layer
    std::cout << "\n1. Linear Layer (2 -> 3):" << std::endl;
    {
        Linearf linear(2, 3, true);  // 2 inputs, 3 outputs, with bias
        
        // Create input: batch of 4 samples, 2 features each
        Matrixf input({4, 2});
        input.fill(1.0f);
        
        auto output = linear.forward(input);
        std::cout << "   Input shape: " << input.shape()[0] << "x" << input.shape()[1] << std::endl;
        std::cout << "   Output shape: " << output.shape()[0] << "x" << output.shape()[1] << std::endl;
        std::cout << "   Output[0,0]: " << output[{0, 0}] << std::endl;
        
        auto params = linear.parameters();
        std::cout << "   Number of parameter tensors: " << params.size() << std::endl;
    }
    
    // Example 2: ReLU Activation
    std::cout << "\n2. ReLU Activation:" << std::endl;
    {
        ReLUf relu;
        
        Matrixf input({2, 3});
        input[{0, 0}] = -1.0f;
        input[{0, 1}] = 2.0f;
        input[{0, 2}] = -0.5f;
        input[{1, 0}] = 3.0f;
        input[{1, 1}] = -2.0f;
        input[{1, 2}] = 1.0f;
        
        auto output = relu.forward(input);
        
        std::cout << "   Input:  [" << input[{0, 0}] << ", " << input[{0, 1}] << ", " << input[{0, 2}] << "]" << std::endl;
        std::cout << "   Output: [" << output[{0, 0}] << ", " << output[{0, 1}] << ", " << output[{0, 2}] << "]" << std::endl;
        std::cout << "   (Negative values zeroed out)" << std::endl;
    }
    
    // Example 3: Sigmoid Activation
    std::cout << "\n3. Sigmoid Activation:" << std::endl;
    {
        Sigmoidf sigmoid;
        
        Matrixf input({1, 4});
        input[{0, 0}] = -2.0f;
        input[{0, 1}] = -1.0f;
        input[{0, 2}] = 1.0f;
        input[{0, 3}] = 2.0f;
        
        auto output = sigmoid.forward(input);
        
        std::cout << "   Input:  [";
        for (size_t i = 0; i < 4; ++i) {
            std::cout << (i > 0 ? ", " : "") << input[{0, i}];
        }
        std::cout << "]" << std::endl;
        
        std::cout << "   Output: [";
        for (size_t i = 0; i < 4; ++i) {
            std::cout << (i > 0 ? ", " : "") << output[{0, i}];
        }
        std::cout << "]" << std::endl;
    }
    
    // Example 4: Softmax
    std::cout << "\n4. Softmax Activation:" << std::endl;
    {
        Softmaxf softmax;
        
        Matrixf input({1, 3});
        input[{0, 0}] = 1.0f;
        input[{0, 1}] = 2.0f;
        input[{0, 2}] = 3.0f;
        
        auto output = softmax.forward(input);
        
        float sum = 0;
        std::cout << "   Input:  [" << input[{0, 0}] << ", " << input[{0, 1}] << ", " << input[{0, 2}] << "]" << std::endl;
        std::cout << "   Output: [";
        for (size_t i = 0; i < 3; ++i) {
            std::cout << (i > 0 ? ", " : "") << output[{0, i}];
            sum += output[{0, i}];
        }
        std::cout << "]" << std::endl;
        std::cout << "   Sum of probabilities: " << sum << " (should be 1.0)" << std::endl;
    }
    
    // Example 5: Dropout
    std::cout << "\n5. Dropout Layer (p=0.5):" << std::endl;
    {
        Dropoutf dropout(0.5f);
        
        Matrixf input({2, 4});
        input.fill(1.0f);
        
        dropout.train(true);  // Training mode
        auto output_train = dropout.forward(input);
        
        dropout.train(false);  // Inference mode
        auto output_eval = dropout.forward(input);
        
        std::cout << "   Training mode - some values dropped:" << std::endl;
        std::cout << "   [" << output_train[{0, 0}] << ", " << output_train[{0, 1}] 
                  << ", " << output_train[{0, 2}] << ", " << output_train[{0, 3}] << "]" << std::endl;
        
        std::cout << "   Eval mode - no dropout:" << std::endl;
        std::cout << "   [" << output_eval[{0, 0}] << ", " << output_eval[{0, 1}] 
                  << ", " << output_eval[{0, 2}] << ", " << output_eval[{0, 3}] << "]" << std::endl;
    }
    
    // Example 6: Batch Normalization
    std::cout << "\n6. Batch Normalization:" << std::endl;
    {
        BatchNorm1df bn(3);  // 3 features
        
        Matrixf input({2, 3});
        input[{0, 0}] = 1.0f; input[{0, 1}] = 2.0f; input[{0, 2}] = 3.0f;
        input[{1, 0}] = 4.0f; input[{1, 1}] = 5.0f; input[{1, 2}] = 6.0f;
        
        auto output = bn.forward(input);
        
        std::cout << "   Input:" << std::endl;
        for (size_t i = 0; i < 2; ++i) {
            std::cout << "   [" << input[{i, 0}] << ", " << input[{i, 1}] << ", " << input[{i, 2}] << "]" << std::endl;
        }
        
        std::cout << "   Normalized output:" << std::endl;
        for (size_t i = 0; i < 2; ++i) {
            std::cout << "   [" << output[{i, 0}] << ", " << output[{i, 1}] << ", " << output[{i, 2}] << "]" << std::endl;
        }
    }
    
    // Example 7: Simple Neural Network (2 layers)
    std::cout << "\n7. Simple 2-Layer Network:" << std::endl;
    {
        // Network: Input(4) -> Linear(4, 8) -> ReLU -> Linear(8, 2) -> Softmax
        Linearf fc1(4, 8, true);
        ReLUf relu;
        Linearf fc2(8, 2, true);
        Softmaxf softmax;
        
        // Input: batch of 3 samples
        Matrixf input({3, 4});
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                input[{i, j}] = static_cast<float>(i + j) * 0.1f;
            }
        }
        
        // Forward pass
        auto h1 = fc1.forward(input);
        auto h2 = relu.forward(h1);
        auto h3 = fc2.forward(h2);
        auto output = softmax.forward(h3);
        
        std::cout << "   Network architecture: 4 -> 8 -> 2" << std::endl;
        std::cout << "   Input shape: " << input.shape()[0] << "x" << input.shape()[1] << std::endl;
        std::cout << "   Output shape: " << output.shape()[0] << "x" << output.shape()[1] << std::endl;
        std::cout << "   Output probabilities:" << std::endl;
        for (size_t i = 0; i < 3; ++i) {
            std::cout << "   Sample " << i << ": [" << output[{i, 0}] << ", " << output[{i, 1}] << "]" << std::endl;
        }
    }
    
    std::cout << "\n=== Demo completed successfully ===" << std::endl;
    
    return 0;
}
