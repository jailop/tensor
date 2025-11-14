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

using namespace tensor;

constexpr size_t IMAGE_SIZE = 28;
constexpr size_t IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE;  // 784
constexpr size_t NUM_CLASSES = 10;

/**
 * @brief Read 32-bit big-endian integer from file
 */
int32_t read_int32(std::ifstream& file) {
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

/**
 * @brief Load MNIST images directly into a Tensor (matrix)
 * @param filename Path to MNIST images file
 * @param images_tensor Output tensor (num_images x IMAGE_PIXELS)
 * @param use_gpu Whether to allocate on GPU
 * @return True on success, false on failure
 */
bool load_mnist_images(const std::string& filename, Tensor<float, 2>& images_tensor, bool use_gpu = false) {
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
    
    std::cout << "Loading " << num_images << " images (" << rows << "x" << cols << ")";
    if (use_gpu) {
        std::cout << " to GPU..." << std::endl;
    } else {
        std::cout << "..." << std::endl;
    }
    
    // Create tensor to hold all images: (num_images x 784)
    images_tensor = Tensor<float, 2>({static_cast<size_t>(num_images), IMAGE_PIXELS}, use_gpu);
    
    // Load images directly into tensor using bulk operations
    float* data_ptr = images_tensor.data();
    unsigned char pixel;
    for (int32_t i = 0; i < num_images; ++i) {
        for (size_t j = 0; j < IMAGE_PIXELS; ++j) {
            file.read(reinterpret_cast<char*>(&pixel), 1);
            // Normalize to [0, 1] and store directly using pointer
            data_ptr[i * IMAGE_PIXELS + j] = pixel / 255.0f;
        }
    }
    
    // Mark data as modified and sync to GPU if needed
    if (use_gpu) {
        images_tensor.mark_cpu_modified();
        images_tensor.sync_to_gpu();
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
 * @brief Improved Neural Network for MNIST
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
        auto x = input;
        x = linear_relu_forward(fc1_, relu1_, x);
        x = linear_relu_forward(fc2_, relu2_, x);
        x = linear_relu_forward(fc3_, relu3_, x);
        x = fc4_.forward(x);
        return softmax_.forward(x);
    }
    
    void backward(const Tensor<float, 2>& grad_output) {
        auto grad = softmax_.backward(grad_output);
        grad = fc4_.backward(grad);
        grad = relu_linear_backward(relu3_, fc3_, grad);
        grad = relu_linear_backward(relu2_, fc2_, grad);
        grad = relu_linear_backward(relu1_, fc1_, grad);
    }
    
    void update_weights(float lr) {
        for (auto* layer : {&fc1_, &fc2_, &fc3_, &fc4_}) {
            update_linear_layer(*layer, lr);
        }
    }
    
    void train(bool mode = true) {
        for (auto* layer : {&fc1_, &fc2_, &fc3_, &fc4_}) {
            layer->train(mode);
        }
        for (auto* relu : {&relu1_, &relu2_, &relu3_}) {
            relu->train(mode);
        }
    }
    
private:
    Tensor<float, 2> linear_relu_forward(Linear<float>& fc, ReLU<float>& relu, 
                                         const Tensor<float, 2>& input) {
        return relu.forward(fc.forward(input));
    }
    
    Tensor<float, 2> relu_linear_backward(ReLU<float>& relu, Linear<float>& fc, 
                                          const Tensor<float, 2>& grad) {
        return fc.backward(relu.backward(grad));
    }
    
    Linear<float> fc1_;
    Linear<float> fc2_;
    Linear<float> fc3_;
    Linear<float> fc4_;
    ReLU<float> relu1_;
    ReLU<float> relu2_;
    ReLU<float> relu3_;
    Softmax<float> softmax_;
};

int main(int argc, char* argv[]) {
    std::cout << "=== MNIST Digit Classification Demo ===" << std::endl;
    std::string data_path = "data/mnist/";
    if (argc > 1) {
        data_path = argv[1];
        if (data_path.back() != '/') {
            data_path += '/';
        }
    }
    
    bool use_gpu = get_active_backend() == Backend::GPU;
    
    std::cout << "Active backend: "
              << toString(get_active_backend())
              << std::endl;
    
    std::cout << "\n=== Loading Dataset ===" << std::endl;
    Tensor<float, 2> train_images({1, 1}); // Placeholder, will be reassigned
    std::vector<uint8_t> train_labels;
    if (!load_mnist_images(data_path + "train-images-idx3-ubyte", train_images, use_gpu)) {
        return 1;
    }
    if (!load_mnist_labels(data_path + "train-labels-idx1-ubyte", train_labels)) {
        return 1;
    }
    Tensor<float, 2> test_images({1, 1}); // Placeholder, will be reassigned
    std::vector<uint8_t> test_labels;
    
    if (!load_mnist_images(data_path + "t10k-images-idx3-ubyte", test_images, use_gpu)) {
        return 1;
    }
    if (!load_mnist_labels(data_path + "t10k-labels-idx1-ubyte", test_labels)) {
        return 1;
    }
    std::cout << "\nDataset loaded successfully!" << std::endl;
    std::cout << "Training samples: " << train_images.dims()[0] << std::endl;
    std::cout << "Test samples: " << test_images.dims()[0] << std::endl;
    
    MNISTNet net;
    
    // Training hyperparameters
    const size_t batch_size = 32;
    const size_t max_epochs = 100;
    const float learning_rate = 0.005f;
    const float target_accuracy = 0.95f;
    const size_t patience = 5;
    const size_t num_batches = train_images.dims()[0] / batch_size;
    
    std::cout << "\n=== Training Started ===" << std::endl;
    net.train(true);
    float best_accuracy = 0.0f;
    size_t epochs_without_improvement = 0;
    size_t epoch;
    float epoch_loss = 0.0f;
    float epoch_accuracy = 0.0f;
    
    std::cout << "\n=== Training Started ===" << std::endl;
    
    // Allocate batch tensors on same device as training data
    Tensor<float, 2> batch_input({batch_size, IMAGE_PIXELS}, use_gpu);
    Tensor<float, 2> batch_targets({batch_size, NUM_CLASSES}, use_gpu);
    Tensor<float, 2> predictions({batch_size, NUM_CLASSES}, use_gpu);
    Tensor<float, 2> grad_output({batch_size, NUM_CLASSES}, use_gpu);
    
    for (epoch = 0; epoch < max_epochs; ++epoch) {
        epoch_loss = 0.0f;
        epoch_accuracy = 0.0f;
        
        for (size_t batch = 0; batch < num_batches; ++batch) {
            size_t start_idx = batch * batch_size;
            
            // Copy batch data using bulk memory copy for efficiency
            const float* src_data = train_images.data();
            float* dst_data = batch_input.data();
            for (size_t i = 0; i < batch_size; ++i) {
                std::memcpy(dst_data + i * IMAGE_PIXELS,
                           src_data + (start_idx + i) * IMAGE_PIXELS,
                           IMAGE_PIXELS * sizeof(float));
            }
            if (use_gpu) {
                batch_input.mark_cpu_modified();
                batch_input.sync_to_gpu();
            }
            
            // Fill one-hot targets (reusing tensor)
            batch_targets.fill(0.0f);
            for (size_t i = 0; i < batch_size; ++i) {
                size_t idx = start_idx + i;
                label_to_onehot(train_labels[idx], batch_targets, i, NUM_CLASSES);
            }
            // Sync targets to GPU after batch fill
            if (use_gpu) {
                batch_targets.mark_cpu_modified();
                batch_targets.sync_to_gpu();
            }
            
            // Forward pass - reuse predictions tensor
            predictions = net.forward(batch_input);
            float loss = cross_entropy_loss(predictions, batch_targets);
            float acc = compute_accuracy(predictions, train_labels, start_idx);
            epoch_loss += loss;
            epoch_accuracy += acc;

            // Compute gradient - reuse grad_output tensor
            auto grad_diff_var = predictions - batch_targets;
            auto grad_diff = std::get<Tensor<float, 2>>(grad_diff_var);
            grad_output = grad_diff / static_cast<float>(batch_size);
            
            net.backward(grad_output);
            net.update_weights(learning_rate);
            
            if ((batch + 1) % 100 == 0) {
                std::cout << "Epoch " << (epoch + 1) << "/" << max_epochs 
                         << ", Batch " << (batch + 1) << "/" << num_batches
                         << ", Loss: " << std::fixed << std::setprecision(4) << loss
                         << ", Acc: " << std::setprecision(2) << (acc * 100) << "%"
                         << std::endl;
            }
        }
        
        epoch_loss /= num_batches;
        epoch_accuracy /= num_batches;
        
        std::cout << "\n>>> Epoch " << (epoch + 1) << " Summary:" << std::endl;
        std::cout << "    Average Loss: "
                  << std::fixed << std::setprecision(4)
                  << epoch_loss
                  << std::endl;
        std::cout << "    Average Accuracy: "
                  << std::setprecision(2)
                  << (epoch_accuracy * 100) << "%";
        
        if (epoch_accuracy > best_accuracy) {
            float improvement = epoch_accuracy - best_accuracy;
            best_accuracy = epoch_accuracy;
            epochs_without_improvement = 0;
            std::cout << " ✓ New best! (+" << std::setprecision(2) << (improvement * 100) << "%)" << std::endl;
        } else {
            epochs_without_improvement++;
            std::cout << " (no improvement for " << epochs_without_improvement 
                     << " epoch" << (epochs_without_improvement > 1 ? "s" : "") << ")" << std::endl;
        }
        std::cout << std::endl;
        
        if (epoch_accuracy >= target_accuracy) {
            std::cout << "🎉 Target accuracy " << std::setprecision(2) << (target_accuracy * 100) 
                     << "% reached! Stopping training." << std::endl;
            break;
        }
        
        if (epochs_without_improvement >= patience) {
            std::cout << "⏹️  No improvement for " << patience << " epochs. Early stopping." << std::endl;
            break;
        }
    }
    
    std::cout << "\n=== Training Complete ===" << std::endl;
    std::cout << "Final training accuracy: " << std::setprecision(2) << (epoch_accuracy * 100) << "%" << std::endl;
    std::cout << "Total epochs: " << (epoch + 1) << "\n" << std::endl;
    
    std::cout << "=== Evaluating on Test Set ===" << std::endl;
    net.train(false);
    
    const size_t test_batch_size = 100;
    const size_t num_test_batches = test_images.dims()[0] / test_batch_size;
    float test_accuracy = 0.0f;
    
    for (size_t batch = 0; batch < num_test_batches; ++batch) {
        size_t start_idx = batch * test_batch_size;
        Tensor<float, 2> batch_input({test_batch_size, IMAGE_PIXELS}, use_gpu);
        
        // Copy data using bulk memory copy
        const float* src_data = test_images.data();
        float* dst_data = batch_input.data();
        for (size_t i = 0; i < test_batch_size; ++i) {
            std::memcpy(dst_data + i * IMAGE_PIXELS,
                       src_data + (start_idx + i) * IMAGE_PIXELS,
                       IMAGE_PIXELS * sizeof(float));
        }
        if (use_gpu) {
            batch_input.mark_cpu_modified();
            batch_input.sync_to_gpu();
        }
        
        auto predictions = net.forward(batch_input);
        float acc = compute_accuracy(predictions, test_labels, start_idx);
        test_accuracy += acc;
    }
    
    test_accuracy /= num_test_batches;
    
    std::cout << "Test Accuracy: " << std::fixed << std::setprecision(2) 
              << (test_accuracy * 100) << "%" << std::endl;
    
    std::cout << "\n=== Sample Predictions ===" << std::endl;
    Tensor<float, 2> sample_input({1, IMAGE_PIXELS}, use_gpu);
    for (size_t i = 0; i < 5; ++i) {
        // Copy single image using bulk memory copy
        const float* src_data = test_images.data();
        float* dst_data = sample_input.data();
        std::memcpy(dst_data, src_data + i * IMAGE_PIXELS, IMAGE_PIXELS * sizeof(float));
        if (use_gpu) {
            sample_input.mark_cpu_modified();
            sample_input.sync_to_gpu();
        }
        
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
