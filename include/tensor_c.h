#ifndef TENSOR_C_H
#define TENSOR_C_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdbool.h>

/* Opaque handle types */
typedef void* TensorFloatHandle;
typedef void* TensorDoubleHandle;
typedef void* MatrixFloatHandle;
typedef void* MatrixDoubleHandle;
typedef void* VectorFloatHandle;
typedef void* VectorDoubleHandle;
typedef void* OptimizerHandle;

/* Error codes */
typedef enum {
    TENSOR_SUCCESS = 0,
    TENSOR_ERROR_ALLOCATION = 1,
    TENSOR_ERROR_SHAPE = 2,
    TENSOR_ERROR_INDEX = 3,
    TENSOR_ERROR_COMPUTATION = 4,
    TENSOR_ERROR_NULL_POINTER = 5,
    TENSOR_ERROR_INVALID_OPERATION = 6,
    TENSOR_ERROR_FILE_IO = 7
} TensorErrorCode;

/* Device types */
typedef enum {
    TENSOR_DEVICE_CPU = 0,
    TENSOR_DEVICE_GPU = 1
} TensorDevice;

/* ===== Vector Operations ===== */

/* Creation and destruction */
TensorErrorCode vector_float_create(size_t size, const float* data, VectorFloatHandle* out_handle);
TensorErrorCode vector_float_zeros(size_t size, VectorFloatHandle* out_handle);
TensorErrorCode vector_float_ones(size_t size, VectorFloatHandle* out_handle);
TensorErrorCode vector_float_destroy(VectorFloatHandle handle);

TensorErrorCode vector_double_create(size_t size, const double* data, VectorDoubleHandle* out_handle);
TensorErrorCode vector_double_zeros(size_t size, VectorDoubleHandle* out_handle);
TensorErrorCode vector_double_ones(size_t size, VectorDoubleHandle* out_handle);
TensorErrorCode vector_double_destroy(VectorDoubleHandle handle);

/* Element access */
TensorErrorCode vector_float_get(VectorFloatHandle handle, size_t index, float* out_value);
TensorErrorCode vector_float_set(VectorFloatHandle handle, size_t index, float value);
TensorErrorCode vector_double_get(VectorDoubleHandle handle, size_t index, double* out_value);
TensorErrorCode vector_double_set(VectorDoubleHandle handle, size_t index, double value);

/* Size */
TensorErrorCode vector_float_size(VectorFloatHandle handle, size_t* out_size);
TensorErrorCode vector_double_size(VectorDoubleHandle handle, size_t* out_size);

/* Arithmetic operations */
TensorErrorCode vector_float_add(VectorFloatHandle lhs, VectorFloatHandle rhs, VectorFloatHandle* out_handle);
TensorErrorCode vector_float_subtract(VectorFloatHandle lhs, VectorFloatHandle rhs, VectorFloatHandle* out_handle);
TensorErrorCode vector_float_multiply(VectorFloatHandle lhs, VectorFloatHandle rhs, VectorFloatHandle* out_handle);
TensorErrorCode vector_float_divide(VectorFloatHandle lhs, VectorFloatHandle rhs, VectorFloatHandle* out_handle);
TensorErrorCode vector_float_dot(VectorFloatHandle lhs, VectorFloatHandle rhs, float* out_value);
TensorErrorCode vector_float_norm(VectorFloatHandle handle, float* out_value);

TensorErrorCode vector_double_add(VectorDoubleHandle lhs, VectorDoubleHandle rhs, VectorDoubleHandle* out_handle);
TensorErrorCode vector_double_subtract(VectorDoubleHandle lhs, VectorDoubleHandle rhs, VectorDoubleHandle* out_handle);
TensorErrorCode vector_double_multiply(VectorDoubleHandle lhs, VectorDoubleHandle rhs, VectorDoubleHandle* out_handle);
TensorErrorCode vector_double_divide(VectorDoubleHandle lhs, VectorDoubleHandle rhs, VectorDoubleHandle* out_handle);
TensorErrorCode vector_double_dot(VectorDoubleHandle lhs, VectorDoubleHandle rhs, double* out_value);
TensorErrorCode vector_double_norm(VectorDoubleHandle handle, double* out_value);

/* ===== Matrix Operations ===== */

/* Creation and destruction */
TensorErrorCode matrix_float_create(size_t rows, size_t cols, const float* data, MatrixFloatHandle* out_handle);
TensorErrorCode matrix_float_zeros(size_t rows, size_t cols, MatrixFloatHandle* out_handle);
TensorErrorCode matrix_float_ones(size_t rows, size_t cols, MatrixFloatHandle* out_handle);
TensorErrorCode matrix_float_eye(size_t n, MatrixFloatHandle* out_handle);
TensorErrorCode matrix_float_destroy(MatrixFloatHandle handle);

TensorErrorCode matrix_double_create(size_t rows, size_t cols, const double* data, MatrixDoubleHandle* out_handle);
TensorErrorCode matrix_double_zeros(size_t rows, size_t cols, MatrixDoubleHandle* out_handle);
TensorErrorCode matrix_double_ones(size_t rows, size_t cols, MatrixDoubleHandle* out_handle);
TensorErrorCode matrix_double_eye(size_t n, MatrixDoubleHandle* out_handle);
TensorErrorCode matrix_double_destroy(MatrixDoubleHandle handle);

/* Element access */
TensorErrorCode matrix_float_get(MatrixFloatHandle handle, size_t row, size_t col, float* out_value);
TensorErrorCode matrix_float_set(MatrixFloatHandle handle, size_t row, size_t col, float value);
TensorErrorCode matrix_double_get(MatrixDoubleHandle handle, size_t row, size_t col, double* out_value);
TensorErrorCode matrix_double_set(MatrixDoubleHandle handle, size_t row, size_t col, double value);

/* Shape */
TensorErrorCode matrix_float_shape(MatrixFloatHandle handle, size_t* out_rows, size_t* out_cols);
TensorErrorCode matrix_double_shape(MatrixDoubleHandle handle, size_t* out_rows, size_t* out_cols);

/* Arithmetic operations */
TensorErrorCode matrix_float_add(MatrixFloatHandle lhs, MatrixFloatHandle rhs, MatrixFloatHandle* out_handle);
TensorErrorCode matrix_float_subtract(MatrixFloatHandle lhs, MatrixFloatHandle rhs, MatrixFloatHandle* out_handle);
TensorErrorCode matrix_float_multiply(MatrixFloatHandle lhs, MatrixFloatHandle rhs, MatrixFloatHandle* out_handle);
TensorErrorCode matrix_float_matmul(MatrixFloatHandle lhs, MatrixFloatHandle rhs, MatrixFloatHandle* out_handle);
TensorErrorCode matrix_float_transpose(MatrixFloatHandle handle, MatrixFloatHandle* out_handle);

TensorErrorCode matrix_double_add(MatrixDoubleHandle lhs, MatrixDoubleHandle rhs, MatrixDoubleHandle* out_handle);
TensorErrorCode matrix_double_subtract(MatrixDoubleHandle lhs, MatrixDoubleHandle rhs, MatrixDoubleHandle* out_handle);
TensorErrorCode matrix_double_multiply(MatrixDoubleHandle lhs, MatrixDoubleHandle rhs, MatrixDoubleHandle* out_handle);
TensorErrorCode matrix_double_matmul(MatrixDoubleHandle lhs, MatrixDoubleHandle rhs, MatrixDoubleHandle* out_handle);
TensorErrorCode matrix_double_transpose(MatrixDoubleHandle handle, MatrixDoubleHandle* out_handle);

/* Linear algebra operations */
TensorErrorCode matrix_float_inverse(MatrixFloatHandle handle, MatrixFloatHandle* out_handle);
TensorErrorCode matrix_float_determinant(MatrixFloatHandle handle, float* out_value);
TensorErrorCode matrix_float_trace(MatrixFloatHandle handle, float* out_value);
TensorErrorCode matrix_float_norm(MatrixFloatHandle handle, float* out_value);

TensorErrorCode matrix_double_inverse(MatrixDoubleHandle handle, MatrixDoubleHandle* out_handle);
TensorErrorCode matrix_double_determinant(MatrixDoubleHandle handle, double* out_value);
TensorErrorCode matrix_double_trace(MatrixDoubleHandle handle, double* out_value);
TensorErrorCode matrix_double_norm(MatrixDoubleHandle handle, double* out_value);

/* Matrix-vector operations */
TensorErrorCode matrix_float_matvec(MatrixFloatHandle mat, VectorFloatHandle vec, VectorFloatHandle* out_handle);
TensorErrorCode matrix_double_matvec(MatrixDoubleHandle mat, VectorDoubleHandle vec, VectorDoubleHandle* out_handle);

/* ===== Optimizer Operations ===== */

/* SGD Optimizer */
TensorErrorCode optimizer_sgd_create(float learning_rate, float momentum, OptimizerHandle* out_handle);
TensorErrorCode optimizer_sgd_add_parameter(OptimizerHandle handle, TensorFloatHandle tensor);
TensorErrorCode optimizer_sgd_step(OptimizerHandle handle);
TensorErrorCode optimizer_sgd_zero_grad(OptimizerHandle handle);
TensorErrorCode optimizer_sgd_destroy(OptimizerHandle handle);

/* Adam Optimizer */
TensorErrorCode optimizer_adam_create(float learning_rate, float beta1, float beta2, float epsilon, OptimizerHandle* out_handle);
TensorErrorCode optimizer_adam_add_parameter(OptimizerHandle handle, TensorFloatHandle tensor);
TensorErrorCode optimizer_adam_step(OptimizerHandle handle);
TensorErrorCode optimizer_adam_zero_grad(OptimizerHandle handle);
TensorErrorCode optimizer_adam_destroy(OptimizerHandle handle);

/* ===== I/O Operations ===== */

TensorErrorCode vector_float_save(VectorFloatHandle handle, const char* filename);
TensorErrorCode vector_float_load(const char* filename, VectorFloatHandle* out_handle);
TensorErrorCode matrix_float_save(MatrixFloatHandle handle, const char* filename);
TensorErrorCode matrix_float_load(const char* filename, MatrixFloatHandle* out_handle);

TensorErrorCode vector_double_save(VectorDoubleHandle handle, const char* filename);
TensorErrorCode vector_double_load(const char* filename, VectorDoubleHandle* out_handle);
TensorErrorCode matrix_double_save(MatrixDoubleHandle handle, const char* filename);
TensorErrorCode matrix_double_load(const char* filename, MatrixDoubleHandle* out_handle);

/* ===== Statistical Operations ===== */

TensorErrorCode vector_float_mean(VectorFloatHandle handle, float* out_value);
TensorErrorCode vector_float_variance(VectorFloatHandle handle, float* out_value);
TensorErrorCode vector_float_std(VectorFloatHandle handle, float* out_value);
TensorErrorCode vector_float_sum(VectorFloatHandle handle, float* out_value);
TensorErrorCode vector_float_min(VectorFloatHandle handle, float* out_value);
TensorErrorCode vector_float_max(VectorFloatHandle handle, float* out_value);

TensorErrorCode vector_double_mean(VectorDoubleHandle handle, double* out_value);
TensorErrorCode vector_double_variance(VectorDoubleHandle handle, double* out_value);
TensorErrorCode vector_double_std(VectorDoubleHandle handle, double* out_value);
TensorErrorCode vector_double_sum(VectorDoubleHandle handle, double* out_value);
TensorErrorCode vector_double_min(VectorDoubleHandle handle, double* out_value);
TensorErrorCode vector_double_max(VectorDoubleHandle handle, double* out_value);

TensorErrorCode matrix_float_mean(MatrixFloatHandle handle, float* out_value);
TensorErrorCode matrix_float_sum(MatrixFloatHandle handle, float* out_value);
TensorErrorCode matrix_float_min(MatrixFloatHandle handle, float* out_value);
TensorErrorCode matrix_float_max(MatrixFloatHandle handle, float* out_value);

TensorErrorCode matrix_double_mean(MatrixDoubleHandle handle, double* out_value);
TensorErrorCode matrix_double_sum(MatrixDoubleHandle handle, double* out_value);
TensorErrorCode matrix_double_min(MatrixDoubleHandle handle, double* out_value);
TensorErrorCode matrix_double_max(MatrixDoubleHandle handle, double* out_value);

/* ===== Advanced Linear Algebra Operations ===== */

/* LU Decomposition */
TensorErrorCode matrix_float_lu(MatrixFloatHandle handle, MatrixFloatHandle* out_L, MatrixFloatHandle* out_U, size_t** out_pivot, size_t* out_pivot_size);
TensorErrorCode matrix_double_lu(MatrixDoubleHandle handle, MatrixDoubleHandle* out_L, MatrixDoubleHandle* out_U, size_t** out_pivot, size_t* out_pivot_size);

/* QR Decomposition */
TensorErrorCode matrix_float_qr(MatrixFloatHandle handle, MatrixFloatHandle* out_Q, MatrixFloatHandle* out_R);
TensorErrorCode matrix_double_qr(MatrixDoubleHandle handle, MatrixDoubleHandle* out_Q, MatrixDoubleHandle* out_R);

/* Cholesky Decomposition */
TensorErrorCode matrix_float_cholesky(MatrixFloatHandle handle, MatrixFloatHandle* out_handle);
TensorErrorCode matrix_double_cholesky(MatrixDoubleHandle handle, MatrixDoubleHandle* out_handle);

/* SVD Decomposition */
TensorErrorCode matrix_float_svd(MatrixFloatHandle handle, MatrixFloatHandle* out_U, VectorFloatHandle* out_S, MatrixFloatHandle* out_Vt);
TensorErrorCode matrix_double_svd(MatrixDoubleHandle handle, MatrixDoubleHandle* out_U, VectorDoubleHandle* out_S, MatrixDoubleHandle* out_Vt);

/* Eigenvalue/Eigenvector computation */
TensorErrorCode matrix_float_eig(MatrixFloatHandle handle, VectorFloatHandle* out_eigenvalues, MatrixFloatHandle* out_eigenvectors);
TensorErrorCode matrix_double_eig(MatrixDoubleHandle handle, VectorDoubleHandle* out_eigenvalues, MatrixDoubleHandle* out_eigenvectors);

/* Linear system solvers */
TensorErrorCode matrix_float_solve(MatrixFloatHandle A, VectorFloatHandle b, VectorFloatHandle* out_x);
TensorErrorCode matrix_double_solve(MatrixDoubleHandle A, VectorDoubleHandle b, VectorDoubleHandle* out_x);

TensorErrorCode matrix_float_solve_triangular(MatrixFloatHandle A, VectorFloatHandle b, bool lower, VectorFloatHandle* out_x);
TensorErrorCode matrix_double_solve_triangular(MatrixDoubleHandle A, VectorDoubleHandle b, bool lower, VectorDoubleHandle* out_x);

/* Least squares solver */
TensorErrorCode matrix_float_lstsq(MatrixFloatHandle A, VectorFloatHandle b, VectorFloatHandle* out_x);
TensorErrorCode matrix_double_lstsq(MatrixDoubleHandle A, VectorDoubleHandle b, VectorDoubleHandle* out_x);

/* Pseudo-inverse */
TensorErrorCode matrix_float_pinv(MatrixFloatHandle handle, MatrixFloatHandle* out_handle);
TensorErrorCode matrix_double_pinv(MatrixDoubleHandle handle, MatrixDoubleHandle* out_handle);

/* Matrix rank */
TensorErrorCode matrix_float_rank(MatrixFloatHandle handle, size_t* out_rank);
TensorErrorCode matrix_double_rank(MatrixDoubleHandle handle, size_t* out_rank);

/* Kronecker product */
TensorErrorCode matrix_float_kron(MatrixFloatHandle lhs, MatrixFloatHandle rhs, MatrixFloatHandle* out_handle);
TensorErrorCode matrix_double_kron(MatrixDoubleHandle lhs, MatrixDoubleHandle rhs, MatrixDoubleHandle* out_handle);

/* 3D Cross product */
TensorErrorCode vector_float_cross(VectorFloatHandle lhs, VectorFloatHandle rhs, VectorFloatHandle* out_handle);
TensorErrorCode vector_double_cross(VectorDoubleHandle lhs, VectorDoubleHandle rhs, VectorDoubleHandle* out_handle);

/* ===== Advanced Statistical Operations ===== */

/* Correlation coefficient */
TensorErrorCode vector_float_correlation(VectorFloatHandle lhs, VectorFloatHandle rhs, float* out_value);
TensorErrorCode vector_double_correlation(VectorDoubleHandle lhs, VectorDoubleHandle rhs, double* out_value);

/* Covariance */
TensorErrorCode vector_float_covariance(VectorFloatHandle lhs, VectorFloatHandle rhs, float* out_value);
TensorErrorCode vector_double_covariance(VectorDoubleHandle lhs, VectorDoubleHandle rhs, double* out_value);

/* Spearman rank correlation */
TensorErrorCode vector_float_spearman(VectorFloatHandle lhs, VectorFloatHandle rhs, float* out_value);
TensorErrorCode vector_double_spearman(VectorDoubleHandle lhs, VectorDoubleHandle rhs, double* out_value);

/* Median */
TensorErrorCode vector_float_median(VectorFloatHandle handle, float* out_value);
TensorErrorCode vector_double_median(VectorDoubleHandle handle, double* out_value);

/* Quantile */
TensorErrorCode vector_float_quantile(VectorFloatHandle handle, float q, float* out_value);
TensorErrorCode vector_double_quantile(VectorDoubleHandle handle, double q, double* out_value);

/* Standardization (z-score normalization) */
TensorErrorCode vector_float_standardize(VectorFloatHandle handle, VectorFloatHandle* out_handle);
TensorErrorCode vector_double_standardize(VectorDoubleHandle handle, VectorDoubleHandle* out_handle);

TensorErrorCode matrix_float_standardize(MatrixFloatHandle handle, int axis, MatrixFloatHandle* out_handle);
TensorErrorCode matrix_double_standardize(MatrixDoubleHandle handle, int axis, MatrixDoubleHandle* out_handle);

/* Normalization (min-max scaling) */
TensorErrorCode vector_float_normalize(VectorFloatHandle handle, VectorFloatHandle* out_handle);
TensorErrorCode vector_double_normalize(VectorDoubleHandle handle, VectorDoubleHandle* out_handle);

TensorErrorCode matrix_float_normalize(MatrixFloatHandle handle, int axis, MatrixFloatHandle* out_handle);
TensorErrorCode matrix_double_normalize(MatrixDoubleHandle handle, int axis, MatrixDoubleHandle* out_handle);

/* ===== Mathematical Functions ===== */

/* Element-wise math functions for vectors */
TensorErrorCode vector_float_exp(VectorFloatHandle handle, VectorFloatHandle* out_handle);
TensorErrorCode vector_float_log(VectorFloatHandle handle, VectorFloatHandle* out_handle);
TensorErrorCode vector_float_sqrt(VectorFloatHandle handle, VectorFloatHandle* out_handle);
TensorErrorCode vector_float_sin(VectorFloatHandle handle, VectorFloatHandle* out_handle);
TensorErrorCode vector_float_cos(VectorFloatHandle handle, VectorFloatHandle* out_handle);
TensorErrorCode vector_float_tan(VectorFloatHandle handle, VectorFloatHandle* out_handle);
TensorErrorCode vector_float_tanh(VectorFloatHandle handle, VectorFloatHandle* out_handle);
TensorErrorCode vector_float_sigmoid(VectorFloatHandle handle, VectorFloatHandle* out_handle);
TensorErrorCode vector_float_relu(VectorFloatHandle handle, VectorFloatHandle* out_handle);

TensorErrorCode vector_double_exp(VectorDoubleHandle handle, VectorDoubleHandle* out_handle);
TensorErrorCode vector_double_log(VectorDoubleHandle handle, VectorDoubleHandle* out_handle);
TensorErrorCode vector_double_sqrt(VectorDoubleHandle handle, VectorDoubleHandle* out_handle);
TensorErrorCode vector_double_sin(VectorDoubleHandle handle, VectorDoubleHandle* out_handle);
TensorErrorCode vector_double_cos(VectorDoubleHandle handle, VectorDoubleHandle* out_handle);
TensorErrorCode vector_double_tan(VectorDoubleHandle handle, VectorDoubleHandle* out_handle);
TensorErrorCode vector_double_tanh(VectorDoubleHandle handle, VectorDoubleHandle* out_handle);
TensorErrorCode vector_double_sigmoid(VectorDoubleHandle handle, VectorDoubleHandle* out_handle);
TensorErrorCode vector_double_relu(VectorDoubleHandle handle, VectorDoubleHandle* out_handle);

/* Element-wise math functions for matrices */
TensorErrorCode matrix_float_exp(MatrixFloatHandle handle, MatrixFloatHandle* out_handle);
TensorErrorCode matrix_float_log(MatrixFloatHandle handle, MatrixFloatHandle* out_handle);
TensorErrorCode matrix_float_sqrt(MatrixFloatHandle handle, MatrixFloatHandle* out_handle);
TensorErrorCode matrix_float_tanh(MatrixFloatHandle handle, MatrixFloatHandle* out_handle);
TensorErrorCode matrix_float_sigmoid(MatrixFloatHandle handle, MatrixFloatHandle* out_handle);
TensorErrorCode matrix_float_relu(MatrixFloatHandle handle, MatrixFloatHandle* out_handle);

TensorErrorCode matrix_double_exp(MatrixDoubleHandle handle, MatrixDoubleHandle* out_handle);
TensorErrorCode matrix_double_log(MatrixDoubleHandle handle, MatrixDoubleHandle* out_handle);
TensorErrorCode matrix_double_sqrt(MatrixDoubleHandle handle, MatrixDoubleHandle* out_handle);
TensorErrorCode matrix_double_tanh(MatrixDoubleHandle handle, MatrixDoubleHandle* out_handle);
TensorErrorCode matrix_double_sigmoid(MatrixDoubleHandle handle, MatrixDoubleHandle* out_handle);
TensorErrorCode matrix_double_relu(MatrixDoubleHandle handle, MatrixDoubleHandle* out_handle);

/* ===== Slicing and Views ===== */

/* Get row from matrix */
TensorErrorCode matrix_float_get_row(MatrixFloatHandle handle, size_t row, VectorFloatHandle* out_handle);
TensorErrorCode matrix_double_get_row(MatrixDoubleHandle handle, size_t row, VectorDoubleHandle* out_handle);

/* Get column from matrix */
TensorErrorCode matrix_float_get_col(MatrixFloatHandle handle, size_t col, VectorFloatHandle* out_handle);
TensorErrorCode matrix_double_get_col(MatrixDoubleHandle handle, size_t col, VectorDoubleHandle* out_handle);

/* Get diagonal from matrix */
TensorErrorCode matrix_float_get_diag(MatrixFloatHandle handle, VectorFloatHandle* out_handle);
TensorErrorCode matrix_double_get_diag(MatrixDoubleHandle handle, VectorDoubleHandle* out_handle);

/* Get submatrix */
TensorErrorCode matrix_float_submatrix(MatrixFloatHandle handle, size_t row_start, size_t row_end, size_t col_start, size_t col_end, MatrixFloatHandle* out_handle);
TensorErrorCode matrix_double_submatrix(MatrixDoubleHandle handle, size_t row_start, size_t row_end, size_t col_start, size_t col_end, MatrixDoubleHandle* out_handle);

/* Get vector slice */
TensorErrorCode vector_float_slice(VectorFloatHandle handle, size_t start, size_t end, VectorFloatHandle* out_handle);
TensorErrorCode vector_double_slice(VectorDoubleHandle handle, size_t start, size_t end, VectorDoubleHandle* out_handle);

/* ===== Random Number Generation ===== */

/* Random uniform distribution */
TensorErrorCode vector_float_random_uniform(size_t size, float low, float high, VectorFloatHandle* out_handle);
TensorErrorCode vector_double_random_uniform(size_t size, double low, double high, VectorDoubleHandle* out_handle);

TensorErrorCode matrix_float_random_uniform(size_t rows, size_t cols, float low, float high, MatrixFloatHandle* out_handle);
TensorErrorCode matrix_double_random_uniform(size_t rows, size_t cols, double low, double high, MatrixDoubleHandle* out_handle);

/* Random normal distribution */
TensorErrorCode vector_float_random_normal(size_t size, float mean, float std, VectorFloatHandle* out_handle);
TensorErrorCode vector_double_random_normal(size_t size, double mean, double std, VectorDoubleHandle* out_handle);

TensorErrorCode matrix_float_random_normal(size_t rows, size_t cols, float mean, float std, MatrixFloatHandle* out_handle);
TensorErrorCode matrix_double_random_normal(size_t rows, size_t cols, double mean, double std, MatrixDoubleHandle* out_handle);

/* ===== Utility Functions ===== */

/* Copy operations */
TensorErrorCode vector_float_copy(VectorFloatHandle handle, VectorFloatHandle* out_handle);
TensorErrorCode vector_double_copy(VectorDoubleHandle handle, VectorDoubleHandle* out_handle);
TensorErrorCode matrix_float_copy(MatrixFloatHandle handle, MatrixFloatHandle* out_handle);
TensorErrorCode matrix_double_copy(MatrixDoubleHandle handle, MatrixDoubleHandle* out_handle);

/* Print to stdout */
TensorErrorCode vector_float_print(VectorFloatHandle handle);
TensorErrorCode vector_double_print(VectorDoubleHandle handle);
TensorErrorCode matrix_float_print(MatrixFloatHandle handle);
TensorErrorCode matrix_double_print(MatrixDoubleHandle handle);

/* Get data pointer (read-only) */
TensorErrorCode vector_float_data(VectorFloatHandle handle, const float** out_data);
TensorErrorCode vector_double_data(VectorDoubleHandle handle, const double** out_data);
TensorErrorCode matrix_float_data(MatrixFloatHandle handle, const float** out_data);
TensorErrorCode matrix_double_data(MatrixDoubleHandle handle, const double** out_data);

/* Reshape operations */
TensorErrorCode matrix_float_reshape(MatrixFloatHandle handle, size_t new_rows, size_t new_cols, MatrixFloatHandle* out_handle);
TensorErrorCode matrix_double_reshape(MatrixDoubleHandle handle, size_t new_rows, size_t new_cols, MatrixDoubleHandle* out_handle);

/* Get library version */
const char* tensor_c_version(void);

/* Get last error message (thread-local) */
const char* tensor_c_last_error(void);

/* Device management */
TensorErrorCode tensor_c_set_device(TensorDevice device);
TensorErrorCode tensor_c_get_device(TensorDevice* out_device);

/* ===== Neural Network Layers ===== */

/* Layer handles */
typedef void* LayerHandle;

/* Linear (Dense) Layer */
TensorErrorCode layer_linear_create_float(size_t in_features, size_t out_features, bool use_bias, LayerHandle* out_handle);
TensorErrorCode layer_linear_create_double(size_t in_features, size_t out_features, bool use_bias, LayerHandle* out_handle);
TensorErrorCode layer_linear_forward_float(LayerHandle handle, MatrixFloatHandle input, MatrixFloatHandle* output);
TensorErrorCode layer_linear_forward_double(LayerHandle handle, MatrixDoubleHandle input, MatrixDoubleHandle* output);
TensorErrorCode layer_linear_backward_float(LayerHandle handle, MatrixFloatHandle grad_output, MatrixFloatHandle* grad_input);
TensorErrorCode layer_linear_backward_double(LayerHandle handle, MatrixDoubleHandle grad_output, MatrixDoubleHandle* grad_input);
TensorErrorCode layer_linear_get_weights_float(LayerHandle handle, MatrixFloatHandle* weights);
TensorErrorCode layer_linear_get_bias_float(LayerHandle handle, MatrixFloatHandle* bias);
TensorErrorCode layer_linear_destroy(LayerHandle handle);

/* ReLU Layer */
TensorErrorCode layer_relu_create_float(LayerHandle* out_handle);
TensorErrorCode layer_relu_create_double(LayerHandle* out_handle);
TensorErrorCode layer_relu_forward_float(LayerHandle handle, MatrixFloatHandle input, MatrixFloatHandle* output);
TensorErrorCode layer_relu_forward_double(LayerHandle handle, MatrixDoubleHandle input, MatrixDoubleHandle* output);
TensorErrorCode layer_relu_backward_float(LayerHandle handle, MatrixFloatHandle grad_output, MatrixFloatHandle* grad_input);
TensorErrorCode layer_relu_backward_double(LayerHandle handle, MatrixDoubleHandle grad_output, MatrixDoubleHandle* grad_input);
TensorErrorCode layer_relu_destroy(LayerHandle handle);

/* Sigmoid Layer */
TensorErrorCode layer_sigmoid_create_float(LayerHandle* out_handle);
TensorErrorCode layer_sigmoid_create_double(LayerHandle* out_handle);
TensorErrorCode layer_sigmoid_forward_float(LayerHandle handle, MatrixFloatHandle input, MatrixFloatHandle* output);
TensorErrorCode layer_sigmoid_forward_double(LayerHandle handle, MatrixDoubleHandle input, MatrixDoubleHandle* output);
TensorErrorCode layer_sigmoid_backward_float(LayerHandle handle, MatrixFloatHandle grad_output, MatrixFloatHandle* grad_input);
TensorErrorCode layer_sigmoid_backward_double(LayerHandle handle, MatrixDoubleHandle grad_output, MatrixDoubleHandle* grad_input);
TensorErrorCode layer_sigmoid_destroy(LayerHandle handle);

/* Softmax Layer */
TensorErrorCode layer_softmax_create_float(LayerHandle* out_handle);
TensorErrorCode layer_softmax_create_double(LayerHandle* out_handle);
TensorErrorCode layer_softmax_forward_float(LayerHandle handle, MatrixFloatHandle input, MatrixFloatHandle* output);
TensorErrorCode layer_softmax_forward_double(LayerHandle handle, MatrixDoubleHandle input, MatrixDoubleHandle* output);
TensorErrorCode layer_softmax_backward_float(LayerHandle handle, MatrixFloatHandle grad_output, MatrixFloatHandle* grad_input);
TensorErrorCode layer_softmax_backward_double(LayerHandle handle, MatrixDoubleHandle grad_output, MatrixDoubleHandle* grad_input);
TensorErrorCode layer_softmax_destroy(LayerHandle handle);

/* Dropout Layer */
TensorErrorCode layer_dropout_create_float(float p, LayerHandle* out_handle);
TensorErrorCode layer_dropout_create_double(double p, LayerHandle* out_handle);
TensorErrorCode layer_dropout_forward_float(LayerHandle handle, MatrixFloatHandle input, MatrixFloatHandle* output);
TensorErrorCode layer_dropout_forward_double(LayerHandle handle, MatrixDoubleHandle input, MatrixDoubleHandle* output);
TensorErrorCode layer_dropout_backward_float(LayerHandle handle, MatrixFloatHandle grad_output, MatrixFloatHandle* grad_input);
TensorErrorCode layer_dropout_backward_double(LayerHandle handle, MatrixDoubleHandle grad_output, MatrixDoubleHandle* grad_input);
TensorErrorCode layer_dropout_train(LayerHandle handle, bool training);
TensorErrorCode layer_dropout_destroy(LayerHandle handle);

/* Batch Normalization Layer */
TensorErrorCode layer_batchnorm_create_float(size_t num_features, float eps, float momentum, LayerHandle* out_handle);
TensorErrorCode layer_batchnorm_create_double(size_t num_features, double eps, double momentum, LayerHandle* out_handle);
TensorErrorCode layer_batchnorm_forward_float(LayerHandle handle, MatrixFloatHandle input, MatrixFloatHandle* output);
TensorErrorCode layer_batchnorm_forward_double(LayerHandle handle, MatrixDoubleHandle input, MatrixDoubleHandle* output);
TensorErrorCode layer_batchnorm_backward_float(LayerHandle handle, MatrixFloatHandle grad_output, MatrixFloatHandle* grad_input);
TensorErrorCode layer_batchnorm_backward_double(LayerHandle handle, MatrixDoubleHandle grad_output, MatrixDoubleHandle* grad_input);
TensorErrorCode layer_batchnorm_train(LayerHandle handle, bool training);
TensorErrorCode layer_batchnorm_destroy(LayerHandle handle);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_C_H */
