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

#define IMAGE_SIZE 28
#define IMAGE_PIXELS (IMAGE_SIZE * IMAGE_SIZE)  /* 784 */
#define NUM_CLASSES 10

/* Training hyperparameters */
#define BATCH_SIZE 32
#define MAX_EPOCHS 100
#define LEARNING_RATE 0.005f
#define TARGET_ACCURACY 0.95f
#define PATIENCE 5

/**
 * @brief Neural Network structure for MNIST classification
 * 
 * Architecture: 784 -> 512 -> 256 -> 128 -> 10
 * Layers: Linear + ReLU (x3) + Linear + Softmax
 */
#define NUM_LINEAR_LAYERS 4
#define NUM_RELU_LAYERS 3

typedef struct {
    /* Layer handles as arrays */
    LayerHandle fc[NUM_LINEAR_LAYERS];      /* Linear layers */
    LayerHandle relu[NUM_RELU_LAYERS];      /* ReLU activations */
    LayerHandle softmax;
    
    /* Cache for intermediate activations (forward pass) */
    MatrixFloatHandle h[NUM_LINEAR_LAYERS];    /* Linear outputs */
    MatrixFloatHandle a[NUM_RELU_LAYERS];      /* ReLU outputs */
    MatrixFloatHandle predictions;             /* Softmax output */
    
    /* Cache for gradients (backward pass) */
    MatrixFloatHandle grad_h[NUM_LINEAR_LAYERS];
    MatrixFloatHandle grad_a[NUM_RELU_LAYERS];
    MatrixFloatHandle grad_input;
} MNISTNetwork;

/**
 * @brief Clean up all network resources
 */
void mnist_network_destroy(MNISTNetwork* net) {
    if (!net) return;
    
    /* Destroy layers using loops */
    for (int i = 0; i < NUM_LINEAR_LAYERS; i++) {
        if (net->fc[i]) layer_linear_destroy(net->fc[i]);
    }
    for (int i = 0; i < NUM_RELU_LAYERS; i++) {
        if (net->relu[i]) layer_relu_destroy(net->relu[i]);
    }
    if (net->softmax) layer_softmax_destroy(net->softmax);
    
    /* Destroy cached activations */
    for (int i = 0; i < NUM_LINEAR_LAYERS; i++) {
        if (net->h[i]) matrix_float_destroy(net->h[i]);
    }
    for (int i = 0; i < NUM_RELU_LAYERS; i++) {
        if (net->a[i]) matrix_float_destroy(net->a[i]);
    }
    if (net->predictions) matrix_float_destroy(net->predictions);
    
    /* Destroy cached gradients */
    for (int i = 0; i < NUM_LINEAR_LAYERS; i++) {
        if (net->grad_h[i]) matrix_float_destroy(net->grad_h[i]);
    }
    for (int i = 0; i < NUM_RELU_LAYERS; i++) {
        if (net->grad_a[i]) matrix_float_destroy(net->grad_a[i]);
    }
    if (net->grad_input) matrix_float_destroy(net->grad_input);
    
    /* Clear structure */
    memset(net, 0, sizeof(MNISTNetwork));
}



/**
 * @brief Create and initialize the MNIST network
 */
int mnist_network_create(MNISTNetwork* net) {
    if (!net) return 0;
    memset(net, 0, sizeof(MNISTNetwork));
    
    /* Define layer sizes */
    const size_t layer_sizes[] = {IMAGE_PIXELS, 512, 256, 128, NUM_CLASSES};
    
    /* Create linear layers using loop */
    for (int i = 0; i < NUM_LINEAR_LAYERS; i++) {
        if (layer_linear_create_float(layer_sizes[i], layer_sizes[i+1], true, &net->fc[i]) != TENSOR_SUCCESS) {
            fprintf(stderr, "Error creating fc%d layer\n", i+1);
            mnist_network_destroy(net);
            return 0;
        }
    }
    
    /* Create ReLU activation layers using loop */
    for (int i = 0; i < NUM_RELU_LAYERS; i++) {
        if (layer_relu_create_float(&net->relu[i]) != TENSOR_SUCCESS) {
            fprintf(stderr, "Error creating relu%d layer\n", i+1);
            mnist_network_destroy(net);
            return 0;
        }
    }
    
    if (layer_softmax_create_float(&net->softmax) != TENSOR_SUCCESS) {
        fprintf(stderr, "Error creating softmax layer\n");
        mnist_network_destroy(net);
        return 0;
    }
    
    return 1;
}

/**
 * @brief Forward pass through the network
 */
int mnist_network_forward(MNISTNetwork* net, MatrixFloatHandle input) {
    if (!net || !input) return 0;
    
    /* Clean up previous activations if they exist */
    for (int i = 0; i < NUM_LINEAR_LAYERS; i++) {
        if (net->h[i]) matrix_float_destroy(net->h[i]);
    }
    for (int i = 0; i < NUM_RELU_LAYERS; i++) {
        if (net->a[i]) matrix_float_destroy(net->a[i]);
    }
    if (net->predictions) matrix_float_destroy(net->predictions);
    
    /* Forward pass through first 3 layers: linear+relu blocks using loop */
    MatrixFloatHandle current_input = input;
    for (int i = 0; i < NUM_RELU_LAYERS; i++) {
        layer_linear_forward_float(net->fc[i], current_input, &net->h[i]);
        layer_relu_forward_float(net->relu[i], net->h[i], &net->a[i]);
        current_input = net->a[i];
    }
    
    /* Final linear + softmax */
    layer_linear_forward_float(net->fc[3], net->a[2], &net->h[3]);
    layer_softmax_forward_float(net->softmax, net->h[3], &net->predictions);
    
    return 1;
}

/**
 * @brief Backward pass through the network
 */
int mnist_network_backward(MNISTNetwork* net, MatrixFloatHandle grad_output) {
    if (!net || !grad_output) return 0;
    
    /* Clean up previous gradients if they exist */
    for (int i = 0; i < NUM_LINEAR_LAYERS; i++) {
        if (net->grad_h[i]) matrix_float_destroy(net->grad_h[i]);
    }
    for (int i = 0; i < NUM_RELU_LAYERS; i++) {
        if (net->grad_a[i]) matrix_float_destroy(net->grad_a[i]);
    }
    if (net->grad_input) matrix_float_destroy(net->grad_input);
    
    /* Backward pass through network: softmax + final linear first */
    layer_softmax_backward_float(net->softmax, grad_output, &net->grad_h[3]);
    layer_linear_backward_float(net->fc[3], net->grad_h[3], &net->grad_a[2]);
    
    /* Three relu+linear backward blocks using loop */
    for (int i = NUM_RELU_LAYERS - 1; i >= 0; i--) {
        layer_relu_backward_float(net->relu[i], net->grad_a[i], &net->grad_h[i]);
        if (i > 0) {
            layer_linear_backward_float(net->fc[i], net->grad_h[i], &net->grad_a[i-1]);
        } else {
            layer_linear_backward_float(net->fc[i], net->grad_h[i], &net->grad_input);
        }
    }
    
    return 1;
}

/**
 * @brief Update network weights using gradient descent
 */
void mnist_network_update_weights(MNISTNetwork* net, float learning_rate) {
    if (!net) return;
    
    for (int i = 0; i < NUM_LINEAR_LAYERS; i++) {
        layer_linear_update_weights_float(net->fc[i], learning_rate);
    }
}



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
    
    /* Get direct pointer to tensor data for efficient loading */
    const float* data_ptr;
    matrix_float_data(*images_tensor, &data_ptr);
    float* data = (float*)data_ptr;  /* Cast away const for writing */
    
    /* Load images directly into tensor memory */
    unsigned char pixel;
    for (int32_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < IMAGE_PIXELS; ++j) {
            fread(&pixel, 1, 1, file);
            /* Normalize to [0, 1] and store directly */
            data[i * IMAGE_PIXELS + j] = pixel / 255.0f;
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
    /* Direct pointer access for efficiency */
    const float* data_ptr;
    matrix_float_data(onehot, &data_ptr);
    float* data = (float*)data_ptr;
    
    size_t rows, cols;
    matrix_float_shape(onehot, &rows, &cols);
    
    size_t offset = batch_idx * cols;
    for (size_t i = 0; i < cols; ++i) {
        data[offset + i] = (i == label) ? 1.0f : 0.0f;
    }
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
        return 1;
    }
    if (!load_mnist_labels(train_labels_path, &train_labels, &num_train_labels)) {
        matrix_float_destroy(train_images_tensor);
        return 1;
    }
    
    /* Load test data directly into tensors */
    MatrixFloatHandle test_images_tensor = NULL;
    uint8_t* test_labels = NULL;
    size_t num_test_images, num_test_labels;
    
    printf("\n--- Loading Test Data ---\n");
    if (!load_mnist_images(test_images_path, &test_images_tensor, &num_test_images)) {
        matrix_float_destroy(train_images_tensor);
        free(train_labels);
        return 1;
    }
    if (!load_mnist_labels(test_labels_path, &test_labels, &num_test_labels)) {
        matrix_float_destroy(train_images_tensor);
        free(train_labels);
        matrix_float_destroy(test_images_tensor);
        return 1;
    }
    
    printf("\nDataset loaded successfully!\n");
    printf("Training samples: %zu\n", num_train_images);
    printf("Test samples: %zu\n", num_test_images);
    
    /* Create neural network */
    MNISTNetwork net;
    if (!mnist_network_create(&net)) {
        fprintf(stderr, "Failed to create network\n");
        goto cleanup_data;
    }
    
    printf("\n=== Training Started ===\n");
    size_t num_batches = num_train_images / BATCH_SIZE;
    float best_accuracy = 0.0f;
    size_t epochs_without_improvement = 0;
    size_t epoch;
    float epoch_loss = 0.0f;
    float epoch_accuracy = 0.0f;
    
    for (epoch = 0; epoch < MAX_EPOCHS; ++epoch) {
        epoch_loss = 0.0f;
        epoch_accuracy = 0.0f;
        
        for (size_t batch = 0; batch < num_batches; ++batch) {
            size_t start_idx = batch * BATCH_SIZE;
            
            /* Prepare batch */
            MatrixFloatHandle batch_input, batch_targets;
            matrix_float_submatrix(train_images_tensor, start_idx, start_idx + BATCH_SIZE, 
                                  0, IMAGE_PIXELS, &batch_input);
            matrix_float_zeros(BATCH_SIZE, NUM_CLASSES, &batch_targets);
            
            /* Fill one-hot targets */
            for (size_t i = 0; i < BATCH_SIZE; ++i) {
                size_t idx = start_idx + i;
                label_to_onehot(train_labels[idx], batch_targets, i);
            }
            
            mnist_network_forward(&net, batch_input);
            float loss;
            matrix_float_cross_entropy_loss(net.predictions, batch_targets, &loss);
            float acc;
            matrix_float_compute_accuracy(net.predictions, &train_labels[start_idx], BATCH_SIZE, &acc);
            epoch_loss += loss;
            epoch_accuracy += acc;
            MatrixFloatHandle grad_output;
            matrix_float_subtract(net.predictions, batch_targets, &grad_output);
            size_t rows, cols;
            matrix_float_shape(grad_output, &rows, &cols);
            const float* diff_ptr;
            matrix_float_data(grad_output, &diff_ptr);
            float* diff_data = (float*)diff_ptr;
            float scale = 1.0f / (float)rows;
            size_t total = rows * cols;
            for (size_t i = 0; i < total; ++i) {
                diff_data[i] *= scale;
            }
            mnist_network_backward(&net, grad_output);
            mnist_network_update_weights(&net, LEARNING_RATE);
            if (epoch == 0 && batch == 0) {
                printf("\n=== DEBUG: First Batch Analysis ===\n");
                printf("Loss: %.6f, Accuracy: %.2f%%\n", loss, acc * 100);
                
                /* Check gradient magnitude using direct pointer access */
                MatrixFloatHandle grad_w;
                layer_linear_get_grad_weights_float(net.fc[3], &grad_w);
                size_t g_rows, g_cols;
                matrix_float_shape(grad_w, &g_rows, &g_cols);
                
                const float* grad_ptr;
                matrix_float_data(grad_w, &grad_ptr);
                
                float grad_sum = 0.0f, grad_max = 0.0f;
                size_t grad_total = g_rows * g_cols;
                for (size_t i = 0; i < grad_total; ++i) {
                    float g = fabsf(grad_ptr[i]);
                    grad_sum += g;
                    if (g > grad_max) grad_max = g;
                }
                printf("FC4 gradient: sum=%.6f, max=%.6f, avg=%.8f\n", 
                       grad_sum, grad_max, grad_sum / grad_total);
                
                /* Check predictions stats using direct pointer access */
                const float* pred_ptr;
                matrix_float_data(net.predictions, &pred_ptr);
                
                float pred_sum = 0.0f;
                size_t pred_total = rows * cols;
                for (size_t i = 0; i < pred_total; ++i) {
                    pred_sum += pred_ptr[i];
                }
                printf("Predictions: avg=%.6f (should be ~0.1 for 10 classes)\n", 
                       pred_sum / pred_total);
                printf("=====================================\n\n");
            }
            
            /* Clean up temporary tensors */
            matrix_float_destroy(batch_input);
            matrix_float_destroy(batch_targets);
            matrix_float_destroy(grad_output);
            
            /* Print progress every 100 batches */
            if ((batch + 1) % 100 == 0) {
                printf("Epoch %zu/%d, Batch %zu/%zu, Loss: %.4f, Acc: %.2f%%\n",
                       epoch + 1, MAX_EPOCHS, batch + 1, num_batches,
                       loss, acc * 100);
            }
        }
        
        epoch_loss /= num_batches;
        epoch_accuracy /= num_batches;
        
        printf("\n>>> Epoch %zu Summary:\n", epoch + 1);
        printf("    Average Loss: %.4f\n", epoch_loss);
        printf("    Average Accuracy: %.2f%%", epoch_accuracy * 100);
        
        /* Check for improvement */
        if (epoch_accuracy > best_accuracy) {
            float improvement = epoch_accuracy - best_accuracy;
            best_accuracy = epoch_accuracy;
            epochs_without_improvement = 0;
            printf(" ✓ New best! (+%.2f%%)\n\n", improvement * 100);
        } else {
            epochs_without_improvement++;
            printf(" (no improvement for %zu epoch%s)\n\n", 
                   epochs_without_improvement, 
                   epochs_without_improvement > 1 ? "s" : "");
        }
        
        /* Check early stopping conditions */
        if (epoch_accuracy >= TARGET_ACCURACY) {
            printf("🎉 Target accuracy %.2f%% reached! Stopping training.\n", 
                   TARGET_ACCURACY * 100);
            break;
        }
        
        if (epochs_without_improvement >= PATIENCE) {
            printf("⏹️  No improvement for %d epochs. Early stopping.\n", PATIENCE);
            break;
        }
    }
    
    printf("\n=== Training Complete ===\n");
    printf("Final training accuracy: %.2f%%\n", epoch_accuracy * 100);
    printf("Total epochs: %zu\n\n", epoch + 1);
    
    /* Evaluation on test set */
    printf("=== Evaluating on Test Set ===\n");
    size_t test_batch_size = 100;
    size_t num_test_batches = num_test_images / test_batch_size;
    float test_accuracy = 0.0f;
    
    for (size_t batch = 0; batch < num_test_batches; ++batch) {
        size_t start_idx = batch * test_batch_size;
        
        /* Extract batch using submatrix */
        MatrixFloatHandle batch_input;
        matrix_float_submatrix(test_images_tensor, start_idx, start_idx + test_batch_size,
                              0, IMAGE_PIXELS, &batch_input);
        
        /* Forward pass through network */
        mnist_network_forward(&net, batch_input);
        
        float acc;
        matrix_float_compute_accuracy(net.predictions, &test_labels[start_idx], test_batch_size, &acc);
        test_accuracy += acc;
        
        /* Clean up */
        matrix_float_destroy(batch_input);
    }
    
    test_accuracy /= num_test_batches;
    printf("Test Accuracy: %.2f%%\n", test_accuracy * 100);
    
    /* Display a few predictions */
    printf("\n=== Sample Predictions ===\n");
    for (size_t i = 0; i < 5; ++i) {
        /* Extract single image using submatrix */
        MatrixFloatHandle sample_input;
        matrix_float_submatrix(test_images_tensor, i, i + 1, 0, IMAGE_PIXELS, &sample_input);
        
        /* Forward pass */
        mnist_network_forward(&net, sample_input);
        
        /* Use direct API for argmax */
        size_t pred_class;
        matrix_float_argmax_rows(net.predictions, &pred_class, 1);
        
        float max_val;
        matrix_float_get(net.predictions, 0, pred_class, &max_val);
        
        printf("Image %zu: True label = %d, Predicted = %zu, Confidence = %.2f%%\n",
               i, test_labels[i], pred_class, max_val * 100);
        
        matrix_float_destroy(sample_input);
    }
    
    mnist_network_destroy(&net);
    
cleanup_data:
    /* Clean up data */
    matrix_float_destroy(train_images_tensor);
    free(train_labels);
    matrix_float_destroy(test_images_tensor);
    free(test_labels);
    
    return 0;
}
