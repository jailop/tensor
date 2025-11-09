/**
 * @file mnist_demo_c_simple.c
 * @brief Simple C API demo showing layer creation and forward pass
 */

#include "tensor_c.h"
#include <stdio.h>
#include <stdlib.h>

#define IMAGE_PIXELS 784
#define NUM_CLASSES 10
#define BATCH_SIZE 32

int main() {
    printf("=== MNIST C API Demo (Simple Version) ===\n\n");
    
    /* Check GPU availability */
    if (tensor_c_is_gpu_available()) {
        printf("✓ GPU is available - layers will use GPU automatically\n\n");
    } else {
        printf("Using CPU/BLAS backend\n\n");
    }
    
    /* Create neural network layers (GPU auto-selected) */
    printf("Creating neural network layers...\n");
    LayerHandle fc1, fc2, fc3, fc4;
    LayerHandle relu1, relu2, relu3;
    LayerHandle softmax;
    
    TensorErrorCode err;
    
    err = layer_linear_create_float(IMAGE_PIXELS, 512, true, &fc1);
    if (err != TENSOR_SUCCESS) {
        fprintf(stderr, "Error creating fc1: %s\n", tensor_c_last_error());
        return 1;
    }
    printf("✓ FC1 layer created (784 -> 512)\n");
    
    err = layer_linear_create_float(512, 256, true, &fc2);
    if (err != TENSOR_SUCCESS) {
        fprintf(stderr, "Error creating fc2: %s\n", tensor_c_last_error());
        return 1;
    }
    printf("✓ FC2 layer created (512 -> 256)\n");
    
    err = layer_linear_create_float(256, 128, true, &fc3);
    if (err != TENSOR_SUCCESS) {
        fprintf(stderr, "Error creating fc3: %s\n", tensor_c_last_error());
        return 1;
    }
    printf("✓ FC3 layer created (256 -> 128)\n");
    
    err = layer_linear_create_float(128, NUM_CLASSES, true, &fc4);
    if (err != TENSOR_SUCCESS) {
        fprintf(stderr, "Error creating fc4: %s\n", tensor_c_last_error());
        return 1;
    }
    printf("✓ FC4 layer created (128 -> 10)\n");
    
    layer_relu_create_float(&relu1);
    layer_relu_create_float(&relu2);
    layer_relu_create_float(&relu3);
    printf("✓ ReLU layers created\n");
    
    layer_softmax_create_float(&softmax);
    printf("✓ Softmax layer created\n");
    
    /* Create input tensor */
    printf("\nCreating input tensor (%d samples, %d features)...\n", BATCH_SIZE, IMAGE_PIXELS);
    MatrixFloatHandle input;
    err = matrix_float_zeros(BATCH_SIZE, IMAGE_PIXELS, &input);
    if (err != TENSOR_SUCCESS) {
        fprintf(stderr, "Error creating input: %s\n", tensor_c_last_error());
        return 1;
    }
    
    /* Fill with some dummy data */
    for (size_t i = 0; i < BATCH_SIZE; ++i) {
        for (size_t j = 0; j < IMAGE_PIXELS; ++j) {
            matrix_float_set(input, i, j, 0.5f);
        }
    }
    printf("✓ Input tensor created and filled\n");
    
    /* Check backend being used */
    TensorBackend backend;
    matrix_float_get_backend(input, &backend);
    printf("  Backend: %s\n", tensor_c_backend_name(backend));
    
    /* Forward pass */
    printf("\nPerforming forward pass...\n");
    MatrixFloatHandle h1, a1, h2, a2, h3, a3, h4, output;
    
    layer_linear_forward_float(fc1, input, &h1);
    printf("  → FC1 forward complete\n");
    
    layer_relu_forward_float(relu1, h1, &a1);
    printf("  → ReLU1 forward complete\n");
    
    layer_linear_forward_float(fc2, a1, &h2);
    printf("  → FC2 forward complete\n");
    
    layer_relu_forward_float(relu2, h2, &a2);
    printf("  → ReLU2 forward complete\n");
    
    layer_linear_forward_float(fc3, a2, &h3);
    printf("  → FC3 forward complete\n");
    
    layer_relu_forward_float(relu3, h3, &a3);
    printf("  → ReLU3 forward complete\n");
    
    layer_linear_forward_float(fc4, a3, &h4);
    printf("  → FC4 forward complete\n");
    
    layer_softmax_forward_float(softmax, h4, &output);
    printf("  → Softmax forward complete\n");
    
    /* Check output shape and values */
    size_t out_rows, out_cols;
    matrix_float_shape(output, &out_rows, &out_cols);
    printf("\n✓ Output shape: %zux%zu\n", out_rows, out_cols);
    
    /* Display first sample's predictions */
    printf("\nFirst sample predictions:\n");
    for (size_t i = 0; i < NUM_CLASSES; ++i) {
        float prob;
        matrix_float_get(output, 0, i, &prob);
        printf("  Class %zu: %.4f\n", i, prob);
    }
    
    /* Verify probabilities sum to 1 */
    float sum = 0.0f;
    for (size_t i = 0; i < NUM_CLASSES; ++i) {
        float prob;
        matrix_float_get(output, 0, i, &prob);
        sum += prob;
    }
    printf("\nSum of probabilities: %.6f (should be ~1.0)\n", sum);
    
    /* Clean up */
    printf("\nCleaning up...\n");
    matrix_float_destroy(input);
    matrix_float_destroy(h1);
    matrix_float_destroy(a1);
    matrix_float_destroy(h2);
    matrix_float_destroy(a2);
    matrix_float_destroy(h3);
    matrix_float_destroy(a3);
    matrix_float_destroy(h4);
    matrix_float_destroy(output);
    
    layer_linear_destroy(fc1);
    layer_linear_destroy(fc2);
    layer_linear_destroy(fc3);
    layer_linear_destroy(fc4);
    layer_relu_destroy(relu1);
    layer_relu_destroy(relu2);
    layer_relu_destroy(relu3);
    layer_softmax_destroy(softmax);
    
    printf("\n=== Demo completed successfully! ===\n");
    printf("\nAll operations used automatic GPU/BLAS/CPU backend selection.\n");
    printf("No manual device management was needed!\n");
    
    return 0;
}
