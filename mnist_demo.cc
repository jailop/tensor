/**
 * @file mnist_demo.cc
 * @brief MNIST handwritten digit classification using nn_layers.h
 * 
 * This demo implements a simple feedforward neural network for MNIST
 * digit classification using the layers defined in nn_layers.h.
 * 
 * Dataset download instructions:
 * 1. Download MNIST dataset from: http://yann.lecun.com/exdb/mnist/
 *    Required files:
 *    - train-images-idx3-ubyte.gz (Training images)
 *    - train-labels-idx1-ubyte.gz (Training labels)
 *    - t10k-images-idx3-ubyte.gz  (Test images)
 *    - t10k-labels-idx1-ubyte.gz  (Test labels)
 * 
 * 2. Extract the .gz files:
 *    gunzip *.gz
 * 
 * 3. Place the extracted files in a 'data/mnist/' directory relative to
 *    the executable, or specify the path as a command line argument.
 * 
 * Alternative: Use the provided download script (see below)
 */

#include "nn_layers.h"
#include "tensor_types.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

using namespace tensor4d;
using namespace tensor4d::nn;

// MNIST dataset dimensions
constexpr size_t IMAGE_SIZE = 28;
constexpr size_t IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE;  // 784
constexpr size_t NUM_CLASSES = 10;

/**
 * @brief Print system and backend information
 */
void print_system_info() {
    std::cout << "\n=== System Information ===" << std::endl;
    
#ifdef USE_GPU
    std::cout << "GPU Support: Enabled" << std::endl;
    if (is_gpu_available()) {
        std::cout << "GPU Status: Available and will be used" << std::endl;
    } else {
        std::cout << "GPU Status: Compiled with GPU support but no GPU detected" << std::endl;
    }
#else
    std::cout << "GPU Support: Disabled (not compiled with USE_GPU)" << std::endl;
#endif

#ifdef USE_BLAS
    std::cout << "BLAS Support: Enabled" << std::endl;
#else
    std::cout << "BLAS Support: Disabled" << std::endl;
#endif
    
    std::cout << "Active Backend: " << backend_name(get_active_backend()) << std::endl;
}

/**
 * @brief Read 32-bit big-endian integer from file
 */
int32_t read_int32(std::ifstream& file) {
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

/**
 * @brief Load MNIST images from IDX file format
 */
bool load_mnist_images(const std::string& filename, std::vector<std::vector<float>>& images) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }
    
    int32_t magic = read_int32(file);
    if (magic != 2051) {
        std::cerr << "Error: Invalid MNIST image file (magic number mismatch)" << std::endl;
        return false;
    }
    
    int32_t num_images = read_int32(file);
    int32_t rows = read_int32(file);
    int32_t cols = read_int32(file);
    
    std::cout << "Loading " << num_images << " images (" << rows << "x" << cols << ")..." << std::endl;
    
    images.resize(num_images);
    for (int32_t i = 0; i < num_images; ++i) {
        images[i].resize(IMAGE_PIXELS);
        unsigned char pixel;
        for (size_t j = 0; j < IMAGE_PIXELS; ++j) {
            file.read(reinterpret_cast<char*>(&pixel), 1);
            // Normalize to [0, 1]
            images[i][j] = pixel / 255.0f;
        }
    }
    
    file.close();
    return true;
}

/**
 * @brief Load MNIST labels from IDX file format
 */
bool load_mnist_labels(const std::string& filename, std::vector<uint8_t>& labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }
    
    int32_t magic = read_int32(file);
    if (magic != 2049) {
        std::cerr << "Error: Invalid MNIST label file (magic number mismatch)" << std::endl;
        return false;
    }
    
    int32_t num_labels = read_int32(file);
    std::cout << "Loading " << num_labels << " labels..." << std::endl;
    
    labels.resize(num_labels);
    file.read(reinterpret_cast<char*>(labels.data()), num_labels);
    
    file.close();
    return true;
}

/**
 * @brief Simple SGD optimizer - update weights using optimized fused operation
 */
void sgd_update(std::vector<Tensor<float, 2>*>& params, 
                std::vector<Tensor<float, 2>>& grads,
                float learning_rate) {
    for (size_t i = 0; i < params.size(); ++i) {
        // Use fused operation: params -= learning_rate * grads
        // Avoids temporary tensor allocation
        params[i]->fused_scalar_mul_sub(learning_rate, grads[i]);
    }
}

/**
 * @brief Improved Neural Network for MNIST
 * 
 * Automatically uses GPU if available, otherwise falls back to BLAS or CPU.
 */
class MNISTNet {
public:
    MNISTNet() 
        : fc1_(IMAGE_PIXELS, 512, true),
          fc2_(512, 256, true),
          fc3_(256, 128, true),
          fc4_(128, NUM_CLASSES, true),
          relu1_(),
          relu2_(),
          relu3_(),
          softmax_() {
    }
    
    Tensor<float, 2> forward(const Tensor<float, 2>& input) {
        auto h1 = fc1_.forward(input);
        auto a1 = relu1_.forward(h1);
        
        auto h2 = fc2_.forward(a1);
        auto a2 = relu2_.forward(h2);
        
        auto h3 = fc3_.forward(a2);
        auto a3 = relu3_.forward(h3);
        
        auto h4 = fc4_.forward(a3);
        auto output = softmax_.forward(h4);
        return output;
    }
    
    void backward(const Tensor<float, 2>& grad_output) {
        // Backward through network - computes gradients for all layers
        auto grad = softmax_.backward(grad_output);
        grad = fc4_.backward(grad);
        grad = relu3_.backward(grad);
        grad = fc3_.backward(grad);
        grad = relu2_.backward(grad);
        grad = fc2_.backward(grad);
        grad = relu1_.backward(grad);
        grad = fc1_.backward(grad);
    }
    
    void update_weights(float lr) {
        // SGD update: param -= lr * grad
        update_linear_layer(fc1_, lr);
        update_linear_layer(fc2_, lr);
        update_linear_layer(fc3_, lr);
        update_linear_layer(fc4_, lr);
    }
    
    void train(bool mode = true) {
        fc1_.train(mode);
        fc2_.train(mode);
        fc3_.train(mode);
        fc4_.train(mode);
        relu1_.train(mode);
        relu2_.train(mode);
        relu3_.train(mode);
    }
    
private:
    Linear<float> fc1_;
    Linear<float> fc2_;
    Linear<float> fc3_;
    Linear<float> fc4_;
    ReLU<float> relu1_;
    ReLU<float> relu2_;
    ReLU<float> relu3_;
    Softmax<float> softmax_;
    
    void update_linear_layer(Linear<float>& layer, float lr) {
        // Get weights and their gradients
        auto& weights = layer.weights();
        auto& bias = layer.bias();
        auto& grad_w = layer.grad_weights();
        auto& grad_b = layer.grad_bias();
        
        auto w_shape = weights.shape();
        auto b_shape = bias.shape();
        
        // Update weights: w -= lr * grad_w
        for (size_t i = 0; i < w_shape[0]; ++i) {
            for (size_t j = 0; j < w_shape[1]; ++j) {
                weights[{i, j}] -= lr * grad_w[{i, j}];
            }
        }
        
        // Update bias: b -= lr * grad_b
        for (size_t i = 0; i < b_shape[1]; ++i) {
            bias[{0, i}] -= lr * grad_b[{0, i}];
        }
    }
};

/**
 * @brief Print download instructions
 */
void print_download_instructions() {
    std::cout << "\n=== MNIST Dataset Download Instructions ===\n" << std::endl;
    std::cout << "Option 1: Manual Download" << std::endl;
    std::cout << "--------------------------" << std::endl;
    std::cout << "1. Visit: http://yann.lecun.com/exdb/mnist/" << std::endl;
    std::cout << "2. Download these files:" << std::endl;
    std::cout << "   - train-images-idx3-ubyte.gz" << std::endl;
    std::cout << "   - train-labels-idx1-ubyte.gz" << std::endl;
    std::cout << "   - t10k-images-idx3-ubyte.gz" << std::endl;
    std::cout << "   - t10k-labels-idx1-ubyte.gz" << std::endl;
    std::cout << "3. Extract: gunzip *.gz" << std::endl;
    std::cout << "4. Place in: data/mnist/" << std::endl;
    
    std::cout << "\nOption 2: Using wget/curl" << std::endl;
    std::cout << "-------------------------" << std::endl;
    std::cout << "mkdir -p data/mnist && cd data/mnist" << std::endl;
    std::cout << "wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz" << std::endl;
    std::cout << "wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz" << std::endl;
    std::cout << "wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz" << std::endl;
    std::cout << "wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz" << std::endl;
    std::cout << "gunzip *.gz" << std::endl;
    std::cout << "cd ../.." << std::endl;
    
    std::cout << "\nOption 3: One-line script" << std::endl;
    std::cout << "------------------------" << std::endl;
    std::cout << "mkdir -p data/mnist && cd data/mnist && \\" << std::endl;
    std::cout << "for file in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do \\" << std::endl;
    std::cout << "  wget http://yann.lecun.com/exdb/mnist/${file}.gz && gunzip ${file}.gz; \\" << std::endl;
    std::cout << "done && cd ../.." << std::endl;
    std::cout << "\n==========================================\n" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== MNIST Digit Classification Demo ===" << std::endl;
    std::cout << "Using nn_layers.h neural network implementation\n" << std::endl;
    
    print_system_info();
    
    // Default data path
    std::string data_path = "data/mnist/";
    if (argc > 1) {
        data_path = argv[1];
        if (data_path.back() != '/') {
            data_path += '/';
        }
    }
    
    std::cout << "\n=== Loading Dataset ===" << std::endl;
    std::cout << "Data path: " << data_path << std::endl;
    
    // Load training data
    std::vector<std::vector<float>> train_images;
    std::vector<uint8_t> train_labels;
    
    std::cout << "\n--- Loading Training Data ---" << std::endl;
    if (!load_mnist_images(data_path + "train-images-idx3-ubyte", train_images)) {
        print_download_instructions();
        return 1;
    }
    if (!load_mnist_labels(data_path + "train-labels-idx1-ubyte", train_labels)) {
        print_download_instructions();
        return 1;
    }
    
    // Load test data
    std::vector<std::vector<float>> test_images;
    std::vector<uint8_t> test_labels;
    
    std::cout << "\n--- Loading Test Data ---" << std::endl;
    if (!load_mnist_images(data_path + "t10k-images-idx3-ubyte", test_images)) {
        print_download_instructions();
        return 1;
    }
    if (!load_mnist_labels(data_path + "t10k-labels-idx1-ubyte", test_labels)) {
        print_download_instructions();
        return 1;
    }
    
    std::cout << "\nDataset loaded successfully!" << std::endl;
    std::cout << "Training samples: " << train_images.size() << std::endl;
    std::cout << "Test samples: " << test_images.size() << std::endl;
    
    // Display backend information
#ifdef USE_GPU
    if (is_gpu_available()) {
        std::cout << "\n*** GPU acceleration available! ***" << std::endl;
        std::cout << "Backend: GPU → BLAS → CPU (automatic selection)" << std::endl;
    } else {
        std::cout << "\nGPU not available, using BLAS or CPU" << std::endl;
    }
#else
    std::cout << "\nCompiled without GPU support, using BLAS or CPU" << std::endl;
#endif
    
    // Network architecture
    std::cout << "\n--- Network Architecture ---" << std::endl;
    std::cout << "Input: " << IMAGE_PIXELS << " (28x28 flattened)" << std::endl;
    std::cout << "Hidden Layer 1: 512 neurons + ReLU" << std::endl;
    std::cout << "Hidden Layer 2: 256 neurons + ReLU" << std::endl;
    std::cout << "Hidden Layer 3: 128 neurons + ReLU" << std::endl;
    std::cout << "Output: " << NUM_CLASSES << " classes (Softmax)" << std::endl;
    std::cout << "Total parameters: ~561K" << std::endl;
    
    // Create network (automatically uses GPU if available)
    MNISTNet net;
    
    // Training hyperparameters
    const size_t batch_size = 64;
    const size_t num_epochs = 10;
    const float learning_rate = 0.005f;
    const size_t num_batches = train_images.size() / batch_size;
    
    std::cout << "\n--- Training Configuration ---" << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Number of epochs: " << num_epochs << std::endl;
    std::cout << "Learning rate: " << learning_rate << std::endl;
    std::cout << "Batches per epoch: " << num_batches << std::endl;
    
    // Training loop
    std::cout << "\n=== Training Started ===" << std::endl;
    net.train(true);
    
#ifdef USE_GPU
    // Create a test tensor to verify which backend is being used
    Tensor<float, 2> test_tensor({1, 1});
    std::cout << "Tensor backend in use: " << backend_name(test_tensor.backend()) << std::endl;
#endif
    
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        float epoch_accuracy = 0.0f;
        
        for (size_t batch = 0; batch < num_batches; ++batch) {
            size_t start_idx = batch * batch_size;
            
#ifdef USE_GPU
            // Explicitly enable GPU (default is true, but being explicit per tensor_perf.cc schema)
            Tensor<float, 2> batch_input({batch_size, IMAGE_PIXELS});
            Tensor<float, 2> batch_targets({batch_size, NUM_CLASSES});
#else
            Tensor<float, 2> batch_input({batch_size, IMAGE_PIXELS});
            Tensor<float, 2> batch_targets({batch_size, NUM_CLASSES});
#endif
            
            // Optimized batch preparation using row-wise copy
            for (size_t i = 0; i < batch_size; ++i) {
                size_t idx = start_idx + i;
                // Use std::copy for efficient row copy
                float* batch_row = batch_input.data_ptr() + i * IMAGE_PIXELS;
                std::copy(train_images[idx].begin(), train_images[idx].end(), batch_row);
                label_to_onehot(train_labels[idx], batch_targets, i, NUM_CLASSES);
            }
            
            // Forward pass
            auto predictions = net.forward(batch_input);
            
            // Compute loss
            float loss = cross_entropy_loss(predictions, batch_targets);
            float acc = compute_accuracy(predictions, train_labels, start_idx);
            
            epoch_loss += loss;
            epoch_accuracy += acc;
            
            // Backward pass: compute gradient of loss w.r.t. output using optimized tensor operations
            // For cross-entropy + softmax: grad = (predictions - targets) / batch_size
            auto grad_diff_var = predictions - batch_targets;
            auto grad_diff = std::get<Tensor<float, 2>>(grad_diff_var);
            auto grad_output = grad_diff / static_cast<float>(batch_size);
            
            // Backpropagate gradients through network
            net.backward(grad_output);
            
            // Update weights with SGD
            net.update_weights(learning_rate);
            
            // Print progress every 100 batches
            if ((batch + 1) % 100 == 0) {
                std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs 
                         << ", Batch " << (batch + 1) << "/" << num_batches
                         << ", Loss: " << std::fixed << std::setprecision(4) << loss
                         << ", Acc: " << std::setprecision(2) << (acc * 100) << "%"
                         << std::endl;
            }
        }
        
        epoch_loss /= num_batches;
        epoch_accuracy /= num_batches;
        
        std::cout << "\n>>> Epoch " << (epoch + 1) << " Summary:" << std::endl;
        std::cout << "    Average Loss: " << std::fixed << std::setprecision(4) << epoch_loss << std::endl;
        std::cout << "    Average Accuracy: " << std::setprecision(2) << (epoch_accuracy * 100) << "%\n" << std::endl;
    }
    
    // Evaluation on test set
    std::cout << "\n=== Evaluating on Test Set ===" << std::endl;
    net.train(false);
    
    const size_t test_batch_size = 100;
    const size_t num_test_batches = test_images.size() / test_batch_size;
    float test_accuracy = 0.0f;
    
    for (size_t batch = 0; batch < num_test_batches; ++batch) {
        size_t start_idx = batch * test_batch_size;
        
#ifdef USE_GPU
        Tensor<float, 2> batch_input({test_batch_size, IMAGE_PIXELS});
#else
        Tensor<float, 2> batch_input({test_batch_size, IMAGE_PIXELS});
#endif
        
        // Optimized test batch preparation using row-wise copy
        for (size_t i = 0; i < test_batch_size; ++i) {
            size_t idx = start_idx + i;
            float* batch_row = batch_input.data_ptr() + i * IMAGE_PIXELS;
            std::copy(test_images[idx].begin(), test_images[idx].end(), batch_row);
        }
        
        auto predictions = net.forward(batch_input);
        float acc = compute_accuracy(predictions, test_labels, start_idx);
        test_accuracy += acc;
    }
    
    test_accuracy /= num_test_batches;
    
    std::cout << "Test Accuracy: " << std::fixed << std::setprecision(2) 
              << (test_accuracy * 100) << "%" << std::endl;
    
    // Display a few predictions
    std::cout << "\n=== Sample Predictions ===" << std::endl;
#ifdef USE_GPU
    Tensor<float, 2> sample_input({1, IMAGE_PIXELS});
#else
    Tensor<float, 2> sample_input({1, IMAGE_PIXELS});
#endif
    
    for (size_t i = 0; i < 5; ++i) {
        // Optimized input preparation using std::copy
        float* input_row = sample_input.data_ptr();
        std::copy(test_images[i].begin(), test_images[i].end(), input_row);
        
        auto pred = net.forward(sample_input);
        
        // Use tensor's argmax for efficient prediction - argmax returns flattened index
        size_t argmax_idx = pred.argmax();
        size_t pred_class = argmax_idx % NUM_CLASSES;
        float max_val = pred[{0, pred_class}];
        
        std::cout << "Image " << i << ": True label = " << static_cast<int>(test_labels[i])
                  << ", Predicted = " << pred_class
                  << ", Confidence = " << std::setprecision(2) << (max_val * 100) << "%"
                  << std::endl;
    }
    
    std::cout << "\n=== Demo completed successfully ===" << std::endl;
    
    return 0;
}
