/**
 * @file mnist_demo_c.c
 * @brief MNIST handwritten digit classification using C API (tensor_c.h)
 * 
 * This demo implements a simple feedforward neural network for MNIST
 * digit classification using the C bindings to tensor4d library.
 * 
 * Dataset download instructions:
 * 1. Download MNIST dataset from: http://yann.lecun.com/exdb/mnist/
 * 2. Extract the .gz files: gunzip *.gz
 * 3. Place files in 'data/mnist/' directory
 */

#include "tensor_c.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* MNIST dataset dimensions */
#define IMAGE_SIZE 28
#define IMAGE_PIXELS (IMAGE_SIZE * IMAGE_SIZE)  /* 784 */
#define NUM_CLASSES 10

/* Training hyperparameters */
/* NOTE: GPU operations involve CPU->GPU->CPU data transfers for each operation.
 * Use larger batch sizes (128-256) to amortize transfer overhead and improve GPU utilization.
 * Smaller batch sizes result in more frequent transfers, causing low GPU utilization. */
#define BATCH_SIZE 256
#define NUM_EPOCHS 10
#define LEARNING_RATE 0.005f

/**
 * @brief Read 32-bit big-endian integer from file
 */
int32_t read_int32(FILE* file) {
    unsigned char bytes[4];
    fread(bytes, 1, 4, file);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

/**
 * @brief Load MNIST images directly into a Tensor (matrix)
 * @param filename Path to MNIST images file
 * @param images_tensor Output tensor handle (num_images x IMAGE_PIXELS)
 * @param num_images Output number of images loaded
 * @return 1 on success, 0 on failure
 */
int load_mnist_images(const char* filename, MatrixFloatHandle* images_tensor, size_t* num_images) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return 0;
    }
    
    int32_t magic = read_int32(file);
    if (magic != 2051) {
        fprintf(stderr, "Error: Invalid MNIST image file (magic number mismatch)\n");
        fclose(file);
        return 0;
    }
    
    int32_t count = read_int32(file);
    int32_t rows = read_int32(file);
    int32_t cols = read_int32(file);
    
    printf("Loading %d images (%dx%d)...\n", count, rows, cols);
    
    *num_images = count;
    
    /* Create tensor to hold all images: (num_images x 784) */
    matrix_float_zeros(count, IMAGE_PIXELS, images_tensor);
    
    /* Load images directly into tensor */
    unsigned char pixel;
    for (int32_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < IMAGE_PIXELS; ++j) {
            fread(&pixel, 1, 1, file);
            /* Normalize to [0, 1] and store directly in tensor */
            float normalized = pixel / 255.0f;
            matrix_float_set(*images_tensor, i, j, normalized);
        }
    }
    
    fclose(file);
    return 1;
}

/**
 * @brief Load MNIST labels from IDX file format
 */
int load_mnist_labels(const char* filename, uint8_t** labels, size_t* num_labels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return 0;
    }
    
    int32_t magic = read_int32(file);
    if (magic != 2049) {
        fprintf(stderr, "Error: Invalid MNIST label file (magic number mismatch)\n");
        fclose(file);
        return 0;
    }
    
    int32_t count = read_int32(file);
    printf("Loading %d labels...\n", count);
    
    *num_labels = count;
    *labels = (uint8_t*)malloc(count);
    fread(*labels, 1, count, file);
    
    fclose(file);
    return 1;
}

/**
 * @brief Convert label to one-hot encoded vector
 */
void label_to_onehot(uint8_t label, MatrixFloatHandle onehot, size_t batch_idx) {
    for (size_t i = 0; i < NUM_CLASSES; ++i) {
        float value = (i == label) ? 1.0f : 0.0f;
        matrix_float_set(onehot, batch_idx, i, value);
    }
}

/**
 * @brief Compute cross-entropy loss using optimized tensor operations
 */
float cross_entropy_loss(MatrixFloatHandle predictions, MatrixFloatHandle targets) {
    float loss;
    matrix_float_cross_entropy_loss(predictions, targets, &loss);
    return loss;
}

/**
 * @brief Compute accuracy using optimized tensor operations
 */
float compute_accuracy(MatrixFloatHandle predictions, const uint8_t* labels, size_t start_idx) {
    size_t rows, cols;
    matrix_float_shape(predictions, &rows, &cols);
    
    float accuracy;
    matrix_float_compute_accuracy(predictions, &labels[start_idx], rows, &accuracy);
    return accuracy;
}

/**
 * @brief Compute softmax using optimized tensor operations
 * Note: This creates a new matrix. For in-place operation, use layer_softmax_forward.
 */
void compute_softmax(MatrixFloatHandle matrix) {
    /* For backward compatibility, we compute in-place using the C API's optimized softmax.
     * This is less efficient than using layer_softmax_forward_float but maintains the same interface. */
    MatrixFloatHandle result;
    matrix_float_softmax(matrix, &result);
    
    /* Copy result back to original matrix */
    size_t rows, cols;
    matrix_float_shape(result, &rows, &cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            float val;
            matrix_float_get(result, i, j, &val);
            matrix_float_set(matrix, i, j, val);
        }
    }
    matrix_float_destroy(result);
}

/**
 * @brief Print download instructions
 */
void print_download_instructions() {
    printf("\n=== MNIST Dataset Not Found ===\n");
    printf("Please download the MNIST dataset:\n\n");
    printf("1. Download files from http://yann.lecun.com/exdb/mnist/\n");
    printf("   - train-images-idx3-ubyte.gz\n");
    printf("   - train-labels-idx1-ubyte.gz\n");
    printf("   - t10k-images-idx3-ubyte.gz\n");
    printf("   - t10k-labels-idx1-ubyte.gz\n\n");
    printf("2. Extract: gunzip *.gz\n\n");
    printf("3. Place files in data/mnist/ directory\n\n");
    printf("Or run: ./download_mnist.sh\n");
}

int main(int argc, char* argv[]) {
    printf("=== MNIST Digit Classification Demo (C API) ===\n");
    printf("Using tensor_c.h C interface\n\n");
    
    /* Determine data path */
    const char* data_path = (argc > 1) ? argv[1] : "data/mnist/";
    printf("Data path: %s\n", data_path);
    
    /* Build full file paths */
    char train_images_path[256], train_labels_path[256];
    char test_images_path[256], test_labels_path[256];
    snprintf(train_images_path, sizeof(train_images_path), "%strain-images-idx3-ubyte", data_path);
    snprintf(train_labels_path, sizeof(train_labels_path), "%strain-labels-idx1-ubyte", data_path);
    snprintf(test_images_path, sizeof(test_images_path), "%st10k-images-idx3-ubyte", data_path);
    snprintf(test_labels_path, sizeof(test_labels_path), "%st10k-labels-idx1-ubyte", data_path);
    
    /* Load training data directly into tensors */
    MatrixFloatHandle train_images_tensor = NULL;
    uint8_t* train_labels = NULL;
    size_t num_train_images, num_train_labels;
    
    printf("\n--- Loading Training Data ---\n");
    if (!load_mnist_images(train_images_path, &train_images_tensor, &num_train_images)) {
        print_download_instructions();
        return 1;
    }
    if (!load_mnist_labels(train_labels_path, &train_labels, &num_train_labels)) {
        print_download_instructions();
        matrix_float_destroy(train_images_tensor);
        return 1;
    }
    
    /* Load test data directly into tensors */
    MatrixFloatHandle test_images_tensor = NULL;
    uint8_t* test_labels = NULL;
    size_t num_test_images, num_test_labels;
    
    printf("\n--- Loading Test Data ---\n");
    if (!load_mnist_images(test_images_path, &test_images_tensor, &num_test_images)) {
        print_download_instructions();
        matrix_float_destroy(train_images_tensor);
        free(train_labels);
        return 1;
    }
    if (!load_mnist_labels(test_labels_path, &test_labels, &num_test_labels)) {
        print_download_instructions();
        matrix_float_destroy(train_images_tensor);
        free(train_labels);
        matrix_float_destroy(test_images_tensor);
        return 1;
    }
    
    printf("\nDataset loaded successfully!\n");
    printf("Training samples: %zu\n", num_train_images);
    printf("Test samples: %zu\n", num_test_images);
    
    /* Display backend information */
    if (tensor_c_is_gpu_available()) {
        printf("\n*** GPU acceleration available! ***\n");
        printf("Backend: GPU → BLAS → CPU (automatic selection)\n");
    } else {
        printf("\nUsing CPU/BLAS (no GPU available)\n");
    }
    
    /* Create neural network layers */
    printf("\n--- Network Architecture ---\n");
    printf("Input: %d (28x28 flattened)\n", IMAGE_PIXELS);
    printf("Hidden Layer 1: 512 neurons + ReLU\n");
    printf("Hidden Layer 2: 256 neurons + ReLU\n");
    printf("Hidden Layer 3: 128 neurons + ReLU\n");
    printf("Output: %d classes (Softmax)\n", NUM_CLASSES);
    printf("Total parameters: ~561K\n");
    
    LayerHandle fc1, fc2, fc3, fc4;
    LayerHandle relu1, relu2, relu3;
    LayerHandle softmax;
    
    /* Create layers (automatically use GPU if available) */
    if (layer_linear_create_float(IMAGE_PIXELS, 512, true, &fc1) != TENSOR_SUCCESS) {
        fprintf(stderr, "Error creating fc1 layer\n");
        goto cleanup;
    }
    if (layer_linear_create_float(512, 256, true, &fc2) != TENSOR_SUCCESS) {
        fprintf(stderr, "Error creating fc2 layer\n");
        goto cleanup;
    }
    if (layer_linear_create_float(256, 128, true, &fc3) != TENSOR_SUCCESS) {
        fprintf(stderr, "Error creating fc3 layer\n");
        goto cleanup;
    }
    if (layer_linear_create_float(128, NUM_CLASSES, true, &fc4) != TENSOR_SUCCESS) {
        fprintf(stderr, "Error creating fc4 layer\n");
        goto cleanup;
    }
    
    if (layer_relu_create_float(&relu1) != TENSOR_SUCCESS ||
        layer_relu_create_float(&relu2) != TENSOR_SUCCESS ||
        layer_relu_create_float(&relu3) != TENSOR_SUCCESS) {
        fprintf(stderr, "Error creating ReLU layers\n");
        goto cleanup;
    }
    
    if (layer_softmax_create_float(&softmax) != TENSOR_SUCCESS) {
        fprintf(stderr, "Error creating softmax layer\n");
        goto cleanup;
    }
    
    printf("\n--- Training Configuration ---\n");
    printf("Batch size: %d\n", BATCH_SIZE);
    printf("Number of epochs: %d\n", NUM_EPOCHS);
    printf("Learning rate: %.4f\n", LEARNING_RATE);
    printf("Batches per epoch: %zu\n", num_train_images / BATCH_SIZE);
    printf("Images per epoch: %zu (%.2f%% of dataset)\n", 
           (num_train_images / BATCH_SIZE) * BATCH_SIZE,
           ((num_train_images / BATCH_SIZE) * BATCH_SIZE) * 100.0f / num_train_images);
    if (num_train_images % BATCH_SIZE != 0) {
        printf("Note: %zu images dropped per epoch (incomplete last batch)\n", 
               num_train_images % BATCH_SIZE);
    }
    
    /* Training loop */
    printf("\n=== Training Started ===\n");
    size_t num_batches = num_train_images / BATCH_SIZE;
    
    for (size_t epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        float epoch_loss = 0.0f;
        float epoch_accuracy = 0.0f;
        
        for (size_t batch = 0; batch < num_batches; ++batch) {
            size_t start_idx = batch * BATCH_SIZE;
            
            /* Prepare batch */
            /* Extract batch using submatrix (efficient tensor slicing) */
            MatrixFloatHandle batch_input, batch_targets;
            matrix_float_submatrix(train_images_tensor, start_idx, start_idx + BATCH_SIZE, 
                                  0, IMAGE_PIXELS, &batch_input);
            matrix_float_zeros(BATCH_SIZE, NUM_CLASSES, &batch_targets);
            
            /* Fill one-hot targets */
            for (size_t i = 0; i < BATCH_SIZE; ++i) {
                size_t idx = start_idx + i;
                label_to_onehot(train_labels[idx], batch_targets, i);
            }
            
            /* Forward pass */
            MatrixFloatHandle h1, a1, h2, a2, h3, a3, h4, predictions;
            layer_linear_forward_float(fc1, batch_input, &h1);
            layer_relu_forward_float(relu1, h1, &a1);
            layer_linear_forward_float(fc2, a1, &h2);
            layer_relu_forward_float(relu2, h2, &a2);
            layer_linear_forward_float(fc3, a2, &h3);
            layer_relu_forward_float(relu3, h3, &a3);
            layer_linear_forward_float(fc4, a3, &h4);
            layer_softmax_forward_float(softmax, h4, &predictions);
            
            /* Compute loss and accuracy */
            float loss = cross_entropy_loss(predictions, batch_targets);
            float acc = compute_accuracy(predictions, train_labels, start_idx);
            
            epoch_loss += loss;
            epoch_accuracy += acc;
            
            /* Backward pass - compute gradients */
            /* Gradient of loss w.r.t. predictions (for cross-entropy + softmax) */
            MatrixFloatHandle grad_predictions;
            matrix_float_copy(predictions, &grad_predictions);
            
            /* For cross-entropy loss with softmax: grad = (predictions - targets) / batch_size */
            size_t rows, cols;
            matrix_float_shape(grad_predictions, &rows, &cols);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    float pred, target;
                    matrix_float_get(grad_predictions, i, j, &pred);
                    matrix_float_get(batch_targets, i, j, &target);
                    /* Use actual batch size (rows), not BATCH_SIZE constant */
                    matrix_float_set(grad_predictions, i, j, (pred - target) / (float)rows);
                }
            }
            
            /* Backpropagate through network */
            /* Pass gradient through softmax backward */
            MatrixFloatHandle grad_h4, grad_a3, grad_h3, grad_a2, grad_h2, grad_a1, grad_h1, grad_input;
            
            layer_softmax_backward_float(softmax, grad_predictions, &grad_h4);
            layer_linear_backward_float(fc4, grad_h4, &grad_a3);
            layer_relu_backward_float(relu3, grad_a3, &grad_h3);
            layer_linear_backward_float(fc3, grad_h3, &grad_a2);
            layer_relu_backward_float(relu2, grad_a2, &grad_h2);
            layer_linear_backward_float(fc2, grad_h2, &grad_a1);
            layer_relu_backward_float(relu1, grad_a1, &grad_h1);
            layer_linear_backward_float(fc1, grad_h1, &grad_input);
            
            /* Update weights using gradient descent */
            layer_linear_update_weights_float(fc1, LEARNING_RATE);
            layer_linear_update_weights_float(fc2, LEARNING_RATE);
            layer_linear_update_weights_float(fc3, LEARNING_RATE);
            layer_linear_update_weights_float(fc4, LEARNING_RATE);
            
            /* Debug: Check first batch to verify learning */
            if (epoch == 0 && batch == 0) {
                printf("\n=== DEBUG: First Batch Analysis ===\n");
                printf("Loss: %.6f, Accuracy: %.2f%%\n", loss, acc * 100);
                
                /* Check gradient magnitude */
                MatrixFloatHandle grad_w;
                layer_linear_get_grad_weights_float(fc4, &grad_w);
                size_t g_rows, g_cols;
                matrix_float_shape(grad_w, &g_rows, &g_cols);
                float grad_sum = 0.0f, grad_max = 0.0f;
                for (size_t i = 0; i < g_rows; ++i) {
                    for (size_t j = 0; j < g_cols; ++j) {
                        float g;
                        matrix_float_get(grad_w, i, j, &g);
                        grad_sum += fabsf(g);
                        if (fabsf(g) > grad_max) grad_max = fabsf(g);
                    }
                }
                printf("FC4 gradient: sum=%.6f, max=%.6f, avg=%.8f\n", 
                       grad_sum, grad_max, grad_sum / (g_rows * g_cols));
                
                /* Check predictions stats */
                float pred_sum = 0.0f;
                for (size_t i = 0; i < rows; ++i) {
                    for (size_t j = 0; j < cols; ++j) {
                        float p;
                        matrix_float_get(predictions, i, j, &p);
                        pred_sum += p;
                    }
                }
                printf("Predictions: avg=%.6f (should be ~0.1 for 10 classes)\n", 
                       pred_sum / (rows * cols));
                printf("=====================================\n\n");
            }
            
            /* Clean up temporary tensors */
            matrix_float_destroy(batch_input);
            matrix_float_destroy(batch_targets);
            matrix_float_destroy(h1);
            matrix_float_destroy(a1);
            matrix_float_destroy(h2);
            matrix_float_destroy(a2);
            matrix_float_destroy(h3);
            matrix_float_destroy(a3);
            matrix_float_destroy(h4);
            matrix_float_destroy(predictions);
            matrix_float_destroy(grad_predictions);
            matrix_float_destroy(grad_h4);
            matrix_float_destroy(grad_a3);
            matrix_float_destroy(grad_h3);
            matrix_float_destroy(grad_a2);
            matrix_float_destroy(grad_h2);
            matrix_float_destroy(grad_a1);
            matrix_float_destroy(grad_h1);
            matrix_float_destroy(grad_input);
            
            /* Print progress every 100 batches */
            if ((batch + 1) % 100 == 0) {
                printf("Epoch %zu/%d, Batch %zu/%zu, Loss: %.4f, Acc: %.2f%%\n",
                       epoch + 1, NUM_EPOCHS, batch + 1, num_batches,
                       loss, acc * 100);
            }
        }
        
        epoch_loss /= num_batches;
        epoch_accuracy /= num_batches;
        
        printf("\n>>> Epoch %zu Summary:\n", epoch + 1);
        printf("    Average Loss: %.4f\n", epoch_loss);
        printf("    Average Accuracy: %.2f%%\n\n", epoch_accuracy * 100);
    }
    
    /* Evaluation on test set */
    printf("\n=== Evaluating on Test Set ===\n");
    size_t test_batch_size = 100;
    size_t num_test_batches = num_test_images / test_batch_size;
    float test_accuracy = 0.0f;
    
    for (size_t batch = 0; batch < num_test_batches; ++batch) {
        size_t start_idx = batch * test_batch_size;
        
        /* Extract batch using submatrix */
        MatrixFloatHandle batch_input;
        matrix_float_submatrix(test_images_tensor, start_idx, start_idx + test_batch_size,
                              0, IMAGE_PIXELS, &batch_input);
        
        /* Forward pass */
        MatrixFloatHandle h1, a1, h2, a2, h3, a3, h4, predictions;
        layer_linear_forward_float(fc1, batch_input, &h1);
        layer_relu_forward_float(relu1, h1, &a1);
        layer_linear_forward_float(fc2, a1, &h2);
        layer_relu_forward_float(relu2, h2, &a2);
        layer_linear_forward_float(fc3, a2, &h3);
        layer_relu_forward_float(relu3, h3, &a3);
        layer_linear_forward_float(fc4, a3, &h4);
        layer_softmax_forward_float(softmax, h4, &predictions);
        
        float acc = compute_accuracy(predictions, test_labels, start_idx);
        test_accuracy += acc;
        
        /* Clean up */
        matrix_float_destroy(batch_input);
        matrix_float_destroy(h1);
        matrix_float_destroy(a1);
        matrix_float_destroy(h2);
        matrix_float_destroy(a2);
        matrix_float_destroy(h3);
        matrix_float_destroy(a3);
        matrix_float_destroy(h4);
        matrix_float_destroy(predictions);
    }
    
    test_accuracy /= num_test_batches;
    printf("Test Accuracy: %.2f%%\n", test_accuracy * 100);
    
    /* Display a few predictions */
    printf("\n=== Sample Predictions ===\n");
    for (size_t i = 0; i < 5; ++i) {
        /* Extract single image using submatrix */
        MatrixFloatHandle sample_input;
        matrix_float_submatrix(test_images_tensor, i, i + 1, 0, IMAGE_PIXELS, &sample_input);
        
        MatrixFloatHandle h1, a1, h2, a2, h3, a3, h4, pred;
        layer_linear_forward_float(fc1, sample_input, &h1);
        layer_relu_forward_float(relu1, h1, &a1);
        layer_linear_forward_float(fc2, a1, &h2);
        layer_relu_forward_float(relu2, h2, &a2);
        layer_linear_forward_float(fc3, a2, &h3);
        layer_relu_forward_float(relu3, h3, &a3);
        layer_linear_forward_float(fc4, a3, &h4);
        layer_softmax_forward_float(softmax, h4, &pred);
        
        size_t pred_class = 0;
        float max_val;
        matrix_float_get(pred, 0, 0, &max_val);
        for (size_t j = 1; j < NUM_CLASSES; ++j) {
            float val;
            matrix_float_get(pred, 0, j, &val);
            if (val > max_val) {
                max_val = val;
                pred_class = j;
            }
        }
        
        printf("Image %zu: True label = %d, Predicted = %zu, Confidence = %.2f%%\n",
               i, test_labels[i], pred_class, max_val * 100);
        
        matrix_float_destroy(sample_input);
        matrix_float_destroy(h1);
        matrix_float_destroy(a1);
        matrix_float_destroy(h2);
        matrix_float_destroy(a2);
        matrix_float_destroy(h3);
        matrix_float_destroy(a3);
        matrix_float_destroy(h4);
        matrix_float_destroy(pred);
    }
    
    printf("\n=== Training and Evaluation Completed Successfully ===\n");
    printf("\nThis demo demonstrates:\n");
    printf("- Full neural network training with backpropagation\n");
    printf("- Automatic gradient computation\n");
    printf("- Weight updates using gradient descent\n");
    printf("- GPU acceleration (when available)\n");
    printf("- Forward and backward passes through multiple layers\n");
    
cleanup:
    /* Clean up layers */
    layer_linear_destroy(fc1);
    layer_linear_destroy(fc2);
    layer_linear_destroy(fc3);
    layer_linear_destroy(fc4);
    layer_relu_destroy(relu1);
    layer_relu_destroy(relu2);
    layer_relu_destroy(relu3);
    layer_softmax_destroy(softmax);
    
    /* Clean up data */
    matrix_float_destroy(train_images_tensor);
    free(train_labels);
    matrix_float_destroy(test_images_tensor);
    free(test_labels);
    
    return 0;
}
