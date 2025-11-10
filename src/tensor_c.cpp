#include "tensor_c.h"
#include "tensor.h"
#include "tensor_types.h"
#include "tensor_io.h"
#include "linalg.h"
#include "linalg_advanced.h"
#include "optimizers.h"
#include "nn_layers.h"
#include <new>
#include <exception>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <thread>
#include <iostream>
#include <random>

using namespace tensor4d;
using namespace tensor4d::nn;
using namespace linalg;

// Thread-local error message storage
thread_local char g_last_error[256] = {0};

// Helper macro for exception handling
#define TENSOR_TRY_BEGIN try {
#define TENSOR_TRY_END \
    return TENSOR_SUCCESS; \
    } catch (const std::bad_alloc& e) { \
        snprintf(g_last_error, sizeof(g_last_error), "Allocation error: %s", e.what()); \
        return TENSOR_ERROR_ALLOCATION; \
    } catch (const std::out_of_range& e) { \
        snprintf(g_last_error, sizeof(g_last_error), "Index error: %s", e.what()); \
        return TENSOR_ERROR_INDEX; \
    } catch (const std::runtime_error& e) { \
        snprintf(g_last_error, sizeof(g_last_error), "Runtime error: %s", e.what()); \
        return TENSOR_ERROR_COMPUTATION; \
    } catch (const std::exception& e) { \
        snprintf(g_last_error, sizeof(g_last_error), "Error: %s", e.what()); \
        return TENSOR_ERROR_COMPUTATION; \
    }

extern "C" {

// ===== Vector Operations (float) =====

TensorErrorCode vector_float_create(size_t size, const float* data, VectorFloatHandle* out_handle) {
    if (!out_handle || !data) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vector = new Vectorf({size});
    for (size_t i = 0; i < size; ++i) {
        vector->data()[i] = data[i];
    }
    *out_handle = vector;
    TENSOR_TRY_END
}

TensorErrorCode vector_float_zeros(size_t size, VectorFloatHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vector = new Vectorf({size});
    vector->fill(0.0f);
    *out_handle = vector;
    TENSOR_TRY_END
}

TensorErrorCode vector_float_ones(size_t size, VectorFloatHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vector = new Vectorf({size});
    vector->fill(1.0f);
    *out_handle = vector;
    TENSOR_TRY_END
}

TensorErrorCode vector_float_destroy(VectorFloatHandle handle) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    delete static_cast<Vectorf*>(handle);
    TENSOR_TRY_END
}

TensorErrorCode vector_float_get(VectorFloatHandle handle, size_t index, float* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    *out_value = (*vec)[{index}];
    TENSOR_TRY_END
}

TensorErrorCode vector_float_set(VectorFloatHandle handle, size_t index, float value) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    (*vec)[{index}] = value;
    TENSOR_TRY_END
}

TensorErrorCode vector_float_size(VectorFloatHandle handle, size_t* out_size) {
    if (!handle || !out_size) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    *out_size = vec->shape()[0];
    TENSOR_TRY_END
}

TensorErrorCode vector_float_add(VectorFloatHandle lhs, VectorFloatHandle rhs, VectorFloatHandle* out_handle) {
    if (!lhs || !rhs || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec_lhs = static_cast<Vectorf*>(lhs);
    auto* vec_rhs = static_cast<Vectorf*>(rhs);
    auto result = *vec_lhs + *vec_rhs;
    if (std::holds_alternative<Vectorf>(result)) {
        *out_handle = new Vectorf(std::get<Vectorf>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode vector_float_subtract(VectorFloatHandle lhs, VectorFloatHandle rhs, VectorFloatHandle* out_handle) {
    if (!lhs || !rhs || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec_lhs = static_cast<Vectorf*>(lhs);
    auto* vec_rhs = static_cast<Vectorf*>(rhs);
    auto result = *vec_lhs - *vec_rhs;
    if (std::holds_alternative<Vectorf>(result)) {
        *out_handle = new Vectorf(std::get<Vectorf>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode vector_float_multiply(VectorFloatHandle lhs, VectorFloatHandle rhs, VectorFloatHandle* out_handle) {
    if (!lhs || !rhs || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec_lhs = static_cast<Vectorf*>(lhs);
    auto* vec_rhs = static_cast<Vectorf*>(rhs);
    auto result = *vec_lhs * *vec_rhs;
    if (std::holds_alternative<Vectorf>(result)) {
        *out_handle = new Vectorf(std::get<Vectorf>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode vector_float_divide(VectorFloatHandle lhs, VectorFloatHandle rhs, VectorFloatHandle* out_handle) {
    if (!lhs || !rhs || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec_lhs = static_cast<Vectorf*>(lhs);
    auto* vec_rhs = static_cast<Vectorf*>(rhs);
    auto result = *vec_lhs / *vec_rhs;
    if (std::holds_alternative<Vectorf>(result)) {
        *out_handle = new Vectorf(std::get<Vectorf>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode vector_float_dot(VectorFloatHandle lhs, VectorFloatHandle rhs, float* out_value) {
    if (!lhs || !rhs || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec_lhs = static_cast<Vectorf*>(lhs);
    auto* vec_rhs = static_cast<Vectorf*>(rhs);
    auto result = vec_lhs->dot(*vec_rhs);
    if (std::holds_alternative<float>(result)) {
        *out_value = std::get<float>(result);
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode vector_float_norm(VectorFloatHandle handle, float* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    *out_value = linalg::norm(*vec);
    TENSOR_TRY_END
}

// ===== Vector Operations (double) =====

TensorErrorCode vector_double_create(size_t size, const double* data, VectorDoubleHandle* out_handle) {
    if (!out_handle || !data) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vector = new Vectord({size});
    for (size_t i = 0; i < size; ++i) {
        vector->data()[i] = data[i];
    }
    *out_handle = vector;
    TENSOR_TRY_END
}

TensorErrorCode vector_double_zeros(size_t size, VectorDoubleHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vector = new Vectord({size});
    vector->fill(0.0);
    *out_handle = vector;
    TENSOR_TRY_END
}

TensorErrorCode vector_double_ones(size_t size, VectorDoubleHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vector = new Vectord({size});
    vector->fill(1.0);
    *out_handle = vector;
    TENSOR_TRY_END
}

TensorErrorCode vector_double_destroy(VectorDoubleHandle handle) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    delete static_cast<Vectord*>(handle);
    TENSOR_TRY_END
}

TensorErrorCode vector_double_get(VectorDoubleHandle handle, size_t index, double* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    *out_value = (*vec)[{index}];
    TENSOR_TRY_END
}

TensorErrorCode vector_double_set(VectorDoubleHandle handle, size_t index, double value) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    (*vec)[{index}] = value;
    TENSOR_TRY_END
}

TensorErrorCode vector_double_size(VectorDoubleHandle handle, size_t* out_size) {
    if (!handle || !out_size) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    *out_size = vec->shape()[0];
    TENSOR_TRY_END
}

TensorErrorCode vector_double_add(VectorDoubleHandle lhs, VectorDoubleHandle rhs, VectorDoubleHandle* out_handle) {
    if (!lhs || !rhs || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec_lhs = static_cast<Vectord*>(lhs);
    auto* vec_rhs = static_cast<Vectord*>(rhs);
    auto result = *vec_lhs + *vec_rhs;
    if (std::holds_alternative<Vectord>(result)) {
        *out_handle = new Vectord(std::get<Vectord>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode vector_double_subtract(VectorDoubleHandle lhs, VectorDoubleHandle rhs, VectorDoubleHandle* out_handle) {
    if (!lhs || !rhs || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec_lhs = static_cast<Vectord*>(lhs);
    auto* vec_rhs = static_cast<Vectord*>(rhs);
    auto result = *vec_lhs - *vec_rhs;
    if (std::holds_alternative<Vectord>(result)) {
        *out_handle = new Vectord(std::get<Vectord>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode vector_double_multiply(VectorDoubleHandle lhs, VectorDoubleHandle rhs, VectorDoubleHandle* out_handle) {
    if (!lhs || !rhs || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec_lhs = static_cast<Vectord*>(lhs);
    auto* vec_rhs = static_cast<Vectord*>(rhs);
    auto result = *vec_lhs * *vec_rhs;
    if (std::holds_alternative<Vectord>(result)) {
        *out_handle = new Vectord(std::get<Vectord>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode vector_double_divide(VectorDoubleHandle lhs, VectorDoubleHandle rhs, VectorDoubleHandle* out_handle) {
    if (!lhs || !rhs || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec_lhs = static_cast<Vectord*>(lhs);
    auto* vec_rhs = static_cast<Vectord*>(rhs);
    auto result = *vec_lhs / *vec_rhs;
    if (std::holds_alternative<Vectord>(result)) {
        *out_handle = new Vectord(std::get<Vectord>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode vector_double_dot(VectorDoubleHandle lhs, VectorDoubleHandle rhs, double* out_value) {
    if (!lhs || !rhs || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec_lhs = static_cast<Vectord*>(lhs);
    auto* vec_rhs = static_cast<Vectord*>(rhs);
    auto result = vec_lhs->dot(*vec_rhs);
    if (std::holds_alternative<double>(result)) {
        *out_value = std::get<double>(result);
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode vector_double_norm(VectorDoubleHandle handle, double* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    *out_value = linalg::norm(*vec);
    TENSOR_TRY_END
}

// ===== Matrix Operations (float) =====

TensorErrorCode matrix_float_create(size_t rows, size_t cols, const float* data, MatrixFloatHandle* out_handle) {
    if (!out_handle || !data) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* matrix = new Matrixf({rows, cols});
    for (size_t i = 0; i < rows * cols; ++i) {
        matrix->data()[i] = data[i];
    }
    *out_handle = matrix;
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_zeros(size_t rows, size_t cols, MatrixFloatHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* matrix = new Matrixf({rows, cols});
    matrix->fill(0.0f);
    *out_handle = matrix;
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_ones(size_t rows, size_t cols, MatrixFloatHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* matrix = new Matrixf({rows, cols});
    matrix->fill(1.0f);
    *out_handle = matrix;
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_eye(size_t n, MatrixFloatHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    *out_handle = new Matrixf(linalg::eye<float>(n));
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_destroy(MatrixFloatHandle handle) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    delete static_cast<Matrixf*>(handle);
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_get(MatrixFloatHandle handle, size_t row, size_t col, float* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    *out_value = (*mat)[{row, col}];
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_set(MatrixFloatHandle handle, size_t row, size_t col, float value) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    (*mat)[{row, col}] = value;
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_shape(MatrixFloatHandle handle, size_t* out_rows, size_t* out_cols) {
    if (!handle || !out_rows || !out_cols) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    auto shape = mat->shape();
    *out_rows = shape[0];
    *out_cols = shape[1];
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_add(MatrixFloatHandle lhs, MatrixFloatHandle rhs, MatrixFloatHandle* out_handle) {
    if (!lhs || !rhs || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat_lhs = static_cast<Matrixf*>(lhs);
    auto* mat_rhs = static_cast<Matrixf*>(rhs);
    auto result = *mat_lhs + *mat_rhs;
    if (std::holds_alternative<Matrixf>(result)) {
        *out_handle = new Matrixf(std::get<Matrixf>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_subtract(MatrixFloatHandle lhs, MatrixFloatHandle rhs, MatrixFloatHandle* out_handle) {
    if (!lhs || !rhs || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat_lhs = static_cast<Matrixf*>(lhs);
    auto* mat_rhs = static_cast<Matrixf*>(rhs);
    auto result = *mat_lhs - *mat_rhs;
    if (std::holds_alternative<Matrixf>(result)) {
        *out_handle = new Matrixf(std::get<Matrixf>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_multiply(MatrixFloatHandle lhs, MatrixFloatHandle rhs, MatrixFloatHandle* out_handle) {
    if (!lhs || !rhs || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat_lhs = static_cast<Matrixf*>(lhs);
    auto* mat_rhs = static_cast<Matrixf*>(rhs);
    auto result = *mat_lhs * *mat_rhs;
    if (std::holds_alternative<Matrixf>(result)) {
        *out_handle = new Matrixf(std::get<Matrixf>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_matmul(MatrixFloatHandle lhs, MatrixFloatHandle rhs, MatrixFloatHandle* out_handle) {
    if (!lhs || !rhs || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat_lhs = static_cast<Matrixf*>(lhs);
    auto* mat_rhs = static_cast<Matrixf*>(rhs);
    auto result = mat_lhs->matmul(*mat_rhs);
    if (std::holds_alternative<Matrixf>(result)) {
        *out_handle = new Matrixf(std::get<Matrixf>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_transpose(MatrixFloatHandle handle, MatrixFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    *out_handle = new Matrixf(mat->transpose());
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_inverse(MatrixFloatHandle handle, MatrixFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    auto result = linalg::inverse(*mat);
    if (std::holds_alternative<Matrixf>(result)) {
        *out_handle = new Matrixf(std::get<Matrixf>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_determinant(MatrixFloatHandle handle, float* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    auto result = linalg::determinant(*mat);
    if (std::holds_alternative<float>(result)) {
        *out_value = std::get<float>(result);
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_trace(MatrixFloatHandle handle, float* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    *out_value = trace(*mat);
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_norm(MatrixFloatHandle handle, float* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    *out_value = frobenius_norm(*mat);
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_matvec(MatrixFloatHandle mat, VectorFloatHandle vec, VectorFloatHandle* out_handle) {
    if (!mat || !vec || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* matrix = static_cast<Matrixf*>(mat);
    auto* vector = static_cast<Vectorf*>(vec);
    *out_handle = new Vectorf(matvec(*matrix, *vector));
    TENSOR_TRY_END
}

// ===== Matrix Operations (double) =====

TensorErrorCode matrix_double_create(size_t rows, size_t cols, const double* data, MatrixDoubleHandle* out_handle) {
    if (!out_handle || !data) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* matrix = new Matrixd({rows, cols});
    // Copy data
    for (size_t i = 0; i < rows * cols; ++i) {
        matrix->data()[i] = data[i];
    }
    *out_handle = matrix;
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_zeros(size_t rows, size_t cols, MatrixDoubleHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* matrix = new Matrixd({rows, cols});
    matrix->fill(0.0);
    *out_handle = matrix;
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_ones(size_t rows, size_t cols, MatrixDoubleHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* matrix = new Matrixd({rows, cols});
    matrix->fill(1.0);
    *out_handle = matrix;
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_eye(size_t n, MatrixDoubleHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    *out_handle = new Matrixd(linalg::eye<double>(n));
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_destroy(MatrixDoubleHandle handle) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    delete static_cast<Matrixd*>(handle);
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_get(MatrixDoubleHandle handle, size_t row, size_t col, double* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    *out_value = (*mat)[{row, col}];
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_set(MatrixDoubleHandle handle, size_t row, size_t col, double value) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    (*mat)[{row, col}] = value;
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_shape(MatrixDoubleHandle handle, size_t* out_rows, size_t* out_cols) {
    if (!handle || !out_rows || !out_cols) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    auto shape = mat->shape();
    *out_rows = shape[0];
    *out_cols = shape[1];
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_add(MatrixDoubleHandle lhs, MatrixDoubleHandle rhs, MatrixDoubleHandle* out_handle) {
    if (!lhs || !rhs || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat_lhs = static_cast<Matrixd*>(lhs);
    auto* mat_rhs = static_cast<Matrixd*>(rhs);
    auto result = *mat_lhs + *mat_rhs;
    if (std::holds_alternative<Matrixd>(result)) {
        *out_handle = new Matrixd(std::get<Matrixd>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_subtract(MatrixDoubleHandle lhs, MatrixDoubleHandle rhs, MatrixDoubleHandle* out_handle) {
    if (!lhs || !rhs || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat_lhs = static_cast<Matrixd*>(lhs);
    auto* mat_rhs = static_cast<Matrixd*>(rhs);
    auto result = *mat_lhs - *mat_rhs;
    if (std::holds_alternative<Matrixd>(result)) {
        *out_handle = new Matrixd(std::get<Matrixd>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_multiply(MatrixDoubleHandle lhs, MatrixDoubleHandle rhs, MatrixDoubleHandle* out_handle) {
    if (!lhs || !rhs || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat_lhs = static_cast<Matrixd*>(lhs);
    auto* mat_rhs = static_cast<Matrixd*>(rhs);
    auto result = *mat_lhs * *mat_rhs;
    if (std::holds_alternative<Matrixd>(result)) {
        *out_handle = new Matrixd(std::get<Matrixd>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_matmul(MatrixDoubleHandle lhs, MatrixDoubleHandle rhs, MatrixDoubleHandle* out_handle) {
    if (!lhs || !rhs || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat_lhs = static_cast<Matrixd*>(lhs);
    auto* mat_rhs = static_cast<Matrixd*>(rhs);
    auto result = mat_lhs->matmul(*mat_rhs);
    if (std::holds_alternative<Matrixd>(result)) {
        *out_handle = new Matrixd(std::get<Matrixd>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_transpose(MatrixDoubleHandle handle, MatrixDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    *out_handle = new Matrixd(mat->transpose());
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_inverse(MatrixDoubleHandle handle, MatrixDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    auto result = linalg::inverse(*mat);
    if (std::holds_alternative<Matrixd>(result)) {
        *out_handle = new Matrixd(std::get<Matrixd>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_determinant(MatrixDoubleHandle handle, double* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    auto result = linalg::determinant(*mat);
    if (std::holds_alternative<double>(result)) {
        *out_value = std::get<double>(result);
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_trace(MatrixDoubleHandle handle, double* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    *out_value = trace(*mat);
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_norm(MatrixDoubleHandle handle, double* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    *out_value = frobenius_norm(*mat);
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_matvec(MatrixDoubleHandle mat, VectorDoubleHandle vec, VectorDoubleHandle* out_handle) {
    if (!mat || !vec || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* matrix = static_cast<Matrixd*>(mat);
    auto* vector = static_cast<Vectord*>(vec);
    *out_handle = new Vectord(matvec(*matrix, *vector));
    TENSOR_TRY_END
}

// ===== Optimizer Operations =====

TensorErrorCode optimizer_sgd_create(float learning_rate, float momentum, OptimizerHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    std::vector<Tensor<float, 1>*> empty_params;
    auto* optimizer = new SGD<float, 1>(empty_params, learning_rate, momentum);
    *out_handle = optimizer;
    TENSOR_TRY_END
}

TensorErrorCode optimizer_sgd_add_parameter(OptimizerHandle handle, TensorFloatHandle tensor) {
    if (!handle || !tensor) return TENSOR_ERROR_NULL_POINTER;
    
    // Note: The current optimizer design doesn't support adding parameters after creation.
    // Parameters must be provided when creating the optimizer.
    // This is a design limitation - consider redesigning if dynamic parameter addition is needed.
    snprintf(g_last_error, sizeof(g_last_error), 
             "SGD optimizer doesn't support adding parameters after creation. "
             "Create a new optimizer with all parameters.");
    return TENSOR_ERROR_INVALID_OPERATION;
}

TensorErrorCode optimizer_sgd_step(OptimizerHandle handle) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* optimizer = static_cast<SGD<float, 1>*>(handle);
    optimizer->step();
    TENSOR_TRY_END
}

TensorErrorCode optimizer_sgd_zero_grad(OptimizerHandle handle) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* optimizer = static_cast<SGD<float, 1>*>(handle);
    optimizer->zero_grad();
    TENSOR_TRY_END
}

TensorErrorCode optimizer_sgd_destroy(OptimizerHandle handle) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    delete static_cast<SGD<float, 1>*>(handle);
    TENSOR_TRY_END
}

TensorErrorCode optimizer_adam_create(float learning_rate, float beta1, float beta2, float epsilon, OptimizerHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    std::vector<Tensor<float, 1>*> empty_params;
    auto* optimizer = new Adam<float, 1>(empty_params, learning_rate, beta1, beta2, epsilon);
    *out_handle = optimizer;
    TENSOR_TRY_END
}

TensorErrorCode optimizer_adam_add_parameter(OptimizerHandle handle, TensorFloatHandle tensor) {
    if (!handle || !tensor) return TENSOR_ERROR_NULL_POINTER;
    
    // Note: The current optimizer design doesn't support adding parameters after creation.
    // Parameters must be provided when creating the optimizer.
    // This is a design limitation - consider redesigning if dynamic parameter addition is needed.
    snprintf(g_last_error, sizeof(g_last_error), 
             "Adam optimizer doesn't support adding parameters after creation. "
             "Create a new optimizer with all parameters.");
    return TENSOR_ERROR_INVALID_OPERATION;
}

TensorErrorCode optimizer_adam_step(OptimizerHandle handle) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* optimizer = static_cast<Adam<float, 1>*>(handle);
    optimizer->step();
    TENSOR_TRY_END
}

TensorErrorCode optimizer_adam_zero_grad(OptimizerHandle handle) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* optimizer = static_cast<Adam<float, 1>*>(handle);
    optimizer->zero_grad();
    TENSOR_TRY_END
}

TensorErrorCode optimizer_adam_destroy(OptimizerHandle handle) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    delete static_cast<Adam<float, 1>*>(handle);
    TENSOR_TRY_END
}

// ===== I/O Operations =====

TensorErrorCode vector_float_save(VectorFloatHandle handle, const char* filename) {
    if (!handle || !filename) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    if (!save_binary(*vec, filename)) {
        return TENSOR_ERROR_FILE_IO;
    }
    TENSOR_TRY_END
}

TensorErrorCode vector_float_load(const char* filename, VectorFloatHandle* out_handle) {
    if (!filename || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto result = load_binary<float, 1>(filename);
    if (std::holds_alternative<Tensor<float, 1>>(result)) {
        *out_handle = new Vectorf(std::get<Tensor<float, 1>>(result));
    } else {
        return TENSOR_ERROR_FILE_IO;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_save(MatrixFloatHandle handle, const char* filename) {
    if (!handle || !filename) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    if (!save_binary(*mat, filename)) {
        return TENSOR_ERROR_FILE_IO;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_load(const char* filename, MatrixFloatHandle* out_handle) {
    if (!filename || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto result = load_binary<float, 2>(filename);
    if (std::holds_alternative<Tensor<float, 2>>(result)) {
        *out_handle = new Matrixf(std::get<Tensor<float, 2>>(result));
    } else {
        return TENSOR_ERROR_FILE_IO;
    }
    TENSOR_TRY_END
}

TensorErrorCode vector_double_save(VectorDoubleHandle handle, const char* filename) {
    if (!handle || !filename) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    if (!save_binary(*vec, filename)) {
        return TENSOR_ERROR_FILE_IO;
    }
    TENSOR_TRY_END
}

TensorErrorCode vector_double_load(const char* filename, VectorDoubleHandle* out_handle) {
    if (!filename || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto result = load_binary<double, 1>(filename);
    if (std::holds_alternative<Tensor<double, 1>>(result)) {
        *out_handle = new Vectord(std::get<Tensor<double, 1>>(result));
    } else {
        return TENSOR_ERROR_FILE_IO;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_save(MatrixDoubleHandle handle, const char* filename) {
    if (!handle || !filename) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    if (!save_binary(*mat, filename)) {
        return TENSOR_ERROR_FILE_IO;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_load(const char* filename, MatrixDoubleHandle* out_handle) {
    if (!filename || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto result = load_binary<double, 2>(filename);
    if (std::holds_alternative<Tensor<double, 2>>(result)) {
        *out_handle = new Matrixd(std::get<Tensor<double, 2>>(result));
    } else {
        return TENSOR_ERROR_FILE_IO;
    }
    TENSOR_TRY_END
}

// ===== Statistical Operations (float) =====

TensorErrorCode vector_float_mean(VectorFloatHandle handle, float* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    *out_value = vec->mean();
    TENSOR_TRY_END
}

TensorErrorCode vector_float_variance(VectorFloatHandle handle, float* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    *out_value = vec->variance();
    TENSOR_TRY_END
}

TensorErrorCode vector_float_std(VectorFloatHandle handle, float* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    *out_value = vec->std();
    TENSOR_TRY_END
}

TensorErrorCode vector_float_sum(VectorFloatHandle handle, float* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    *out_value = vec->sum();
    TENSOR_TRY_END
}

TensorErrorCode vector_float_min(VectorFloatHandle handle, float* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    *out_value = vec->min();
    TENSOR_TRY_END
}

TensorErrorCode vector_float_max(VectorFloatHandle handle, float* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    *out_value = vec->max();
    TENSOR_TRY_END
}

// ===== Statistical Operations (double) =====

TensorErrorCode vector_double_mean(VectorDoubleHandle handle, double* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    *out_value = vec->mean();
    TENSOR_TRY_END
}

TensorErrorCode vector_double_variance(VectorDoubleHandle handle, double* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    *out_value = vec->variance();
    TENSOR_TRY_END
}

TensorErrorCode vector_double_std(VectorDoubleHandle handle, double* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    *out_value = vec->std();
    TENSOR_TRY_END
}

TensorErrorCode vector_double_sum(VectorDoubleHandle handle, double* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    *out_value = vec->sum();
    TENSOR_TRY_END
}

TensorErrorCode vector_double_min(VectorDoubleHandle handle, double* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    *out_value = vec->min();
    TENSOR_TRY_END
}

TensorErrorCode vector_double_max(VectorDoubleHandle handle, double* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    *out_value = vec->max();
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_mean(MatrixFloatHandle handle, float* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    *out_value = mat->mean();
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_sum(MatrixFloatHandle handle, float* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    *out_value = mat->sum();
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_min(MatrixFloatHandle handle, float* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    *out_value = mat->min();
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_max(MatrixFloatHandle handle, float* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    *out_value = mat->max();
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_mean(MatrixDoubleHandle handle, double* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    *out_value = mat->mean();
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_sum(MatrixDoubleHandle handle, double* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    *out_value = mat->sum();
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_min(MatrixDoubleHandle handle, double* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    *out_value = mat->min();
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_max(MatrixDoubleHandle handle, double* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    *out_value = mat->max();
    TENSOR_TRY_END
}

// ===== Advanced Linear Algebra Operations =====

TensorErrorCode matrix_float_lu(MatrixFloatHandle handle, MatrixFloatHandle* out_L, MatrixFloatHandle* out_U, size_t** out_pivot, size_t* out_pivot_size) {
    if (!handle || !out_L || !out_U || !out_pivot || !out_pivot_size) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    auto result = linalg::lu_decomp(*mat);
    
    if (auto* lu_data = std::get_if<std::pair<Matrixf, std::vector<int>>>(&result)) {
        const Matrixf& LU = lu_data->first;
        const std::vector<int>& pivots = lu_data->second;
        
        auto dims = LU.dims();
        size_t m = dims[0];
        size_t n = dims[1];
        size_t min_dim = std::min(m, n);
        
        // Use smart pointers for exception safety
        std::unique_ptr<Matrixf> L(new Matrixf({m, min_dim}));
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < min_dim; ++j) {
                if (i > j) {
                    (*L)[{i, j}] = LU[{i, j}];
                } else if (i == j) {
                    (*L)[{i, j}] = 1.0f;
                } else {
                    (*L)[{i, j}] = 0.0f;
                }
            }
        }
        
        std::unique_ptr<Matrixf> U(new Matrixf({min_dim, n}));
        for (size_t i = 0; i < min_dim; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i <= j) {
                    (*U)[{i, j}] = LU[{i, j}];
                } else {
                    (*U)[{i, j}] = 0.0f;
                }
            }
        }
        
        // Custom deleter for malloc-allocated memory
        struct MallocDeleter { void operator()(size_t* p) const { free(p); } };
        std::unique_ptr<size_t[], MallocDeleter> pivot_copy(
            static_cast<size_t*>(malloc(pivots.size() * sizeof(size_t)))
        );
        if (!pivot_copy) {
            return TENSOR_ERROR_ALLOCATION;
        }
        for (size_t i = 0; i < pivots.size(); ++i) {
            pivot_copy[i] = static_cast<size_t>(pivots[i]);
        }
        
        // Transfer ownership to C API
        *out_L = L.release();
        *out_U = U.release();
        *out_pivot = pivot_copy.release();
        *out_pivot_size = pivots.size();
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_lu(MatrixDoubleHandle handle, MatrixDoubleHandle* out_L, MatrixDoubleHandle* out_U, size_t** out_pivot, size_t* out_pivot_size) {
    if (!handle || !out_L || !out_U || !out_pivot || !out_pivot_size) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    auto result = linalg::lu_decomp(*mat);
    
    if (auto* lu_data = std::get_if<std::pair<Matrixd, std::vector<int>>>(&result)) {
        const Matrixd& LU = lu_data->first;
        const std::vector<int>& pivots = lu_data->second;
        
        auto dims = LU.dims();
        size_t m = dims[0];
        size_t n = dims[1];
        size_t min_dim = std::min(m, n);
        
        // Use smart pointers for exception safety
        std::unique_ptr<Matrixd> L(new Matrixd({m, min_dim}));
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < min_dim; ++j) {
                if (i > j) {
                    (*L)[{i, j}] = LU[{i, j}];
                } else if (i == j) {
                    (*L)[{i, j}] = 1.0;
                } else {
                    (*L)[{i, j}] = 0.0;
                }
            }
        }
        
        std::unique_ptr<Matrixd> U(new Matrixd({min_dim, n}));
        for (size_t i = 0; i < min_dim; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i <= j) {
                    (*U)[{i, j}] = LU[{i, j}];
                } else {
                    (*U)[{i, j}] = 0.0;
                }
            }
        }
        
        // Custom deleter for malloc-allocated memory
        struct MallocDeleter { void operator()(size_t* p) const { free(p); } };
        std::unique_ptr<size_t[], MallocDeleter> pivot_copy(
            static_cast<size_t*>(malloc(pivots.size() * sizeof(size_t)))
        );
        if (!pivot_copy) {
            return TENSOR_ERROR_ALLOCATION;
        }
        for (size_t i = 0; i < pivots.size(); ++i) {
            pivot_copy[i] = static_cast<size_t>(pivots[i]);
        }
        
        // Transfer ownership to C API
        *out_L = L.release();
        *out_U = U.release();
        *out_pivot = pivot_copy.release();
        *out_pivot_size = pivots.size();
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_qr(MatrixFloatHandle handle, MatrixFloatHandle* out_Q, MatrixFloatHandle* out_R) {
    if (!handle || !out_Q || !out_R) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    auto result = linalg::qr_decomp(*mat);
    
    if (auto* qr_data = std::get_if<std::pair<Matrixf, Matrixf>>(&result)) {
        *out_Q = new Matrixf(qr_data->first);
        *out_R = new Matrixf(qr_data->second);
        return TENSOR_SUCCESS;
    }
    
    return TENSOR_ERROR_COMPUTATION;
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_qr(MatrixDoubleHandle handle, MatrixDoubleHandle* out_Q, MatrixDoubleHandle* out_R) {
    if (!handle || !out_Q || !out_R) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    auto result = linalg::qr_decomp(*mat);
    
    if (auto* qr_data = std::get_if<std::pair<Matrixd, Matrixd>>(&result)) {
        *out_Q = new Matrixd(qr_data->first);
        *out_R = new Matrixd(qr_data->second);
        return TENSOR_SUCCESS;
    }
    
    return TENSOR_ERROR_COMPUTATION;
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_cholesky(MatrixFloatHandle handle, MatrixFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    auto result = linalg::cholesky_decomp(*mat);
    
    if (std::holds_alternative<Matrixf>(result)) {
        *out_handle = new Matrixf(std::get<Matrixf>(result));
        return TENSOR_SUCCESS;
    }
    
    return TENSOR_ERROR_COMPUTATION;
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_cholesky(MatrixDoubleHandle handle, MatrixDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    auto result = linalg::cholesky_decomp(*mat);
    
    if (std::holds_alternative<Matrixd>(result)) {
        *out_handle = new Matrixd(std::get<Matrixd>(result));
        return TENSOR_SUCCESS;
    }
    
    return TENSOR_ERROR_COMPUTATION;
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_svd(MatrixFloatHandle handle, MatrixFloatHandle* out_U, VectorFloatHandle* out_S, MatrixFloatHandle* out_Vt) {
    if (!handle || !out_U || !out_S || !out_Vt) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    auto result = linalg::svd_decomp(*mat);
    
    if (auto* svd_data = std::get_if<std::tuple<Matrixf, Vectorf, Matrixf>>(&result)) {
        *out_U = new Matrixf(std::get<0>(*svd_data));
        *out_S = new Vectorf(std::get<1>(*svd_data));
        *out_Vt = new Matrixf(std::get<2>(*svd_data));
        return TENSOR_SUCCESS;
    }
    
    return TENSOR_ERROR_COMPUTATION;
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_svd(MatrixDoubleHandle handle, MatrixDoubleHandle* out_U, VectorDoubleHandle* out_S, MatrixDoubleHandle* out_Vt) {
    if (!handle || !out_U || !out_S || !out_Vt) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    auto result = linalg::svd_decomp(*mat);
    
    if (auto* svd_data = std::get_if<std::tuple<Matrixd, Vectord, Matrixd>>(&result)) {
        *out_U = new Matrixd(std::get<0>(*svd_data));
        *out_S = new Vectord(std::get<1>(*svd_data));
        *out_Vt = new Matrixd(std::get<2>(*svd_data));
        return TENSOR_SUCCESS;
    }
    
    return TENSOR_ERROR_COMPUTATION;
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_eig(MatrixFloatHandle handle, VectorFloatHandle* out_eigenvalues, MatrixFloatHandle* out_eigenvectors) {
    if (!handle || !out_eigenvalues || !out_eigenvectors) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    auto result = linalg::eig_decomp(*mat);
    
    if (auto* eig_data = std::get_if<std::pair<Vectorf, Matrixf>>(&result)) {
        *out_eigenvalues = new Vectorf(eig_data->first);
        *out_eigenvectors = new Matrixf(eig_data->second);
        return TENSOR_SUCCESS;
    }
    
    return TENSOR_ERROR_COMPUTATION;
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_eig(MatrixDoubleHandle handle, VectorDoubleHandle* out_eigenvalues, MatrixDoubleHandle* out_eigenvectors) {
    if (!handle || !out_eigenvalues || !out_eigenvectors) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    auto result = linalg::eig_decomp(*mat);
    
    if (auto* eig_data = std::get_if<std::pair<Vectord, Matrixd>>(&result)) {
        *out_eigenvalues = new Vectord(eig_data->first);
        *out_eigenvectors = new Matrixd(eig_data->second);
        return TENSOR_SUCCESS;
    }
    
    return TENSOR_ERROR_COMPUTATION;
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_solve(MatrixFloatHandle A, VectorFloatHandle b, VectorFloatHandle* out_x) {
    if (!A || !b || !out_x) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat_A = static_cast<Matrixf*>(A);
    auto* vec_b = static_cast<Vectorf*>(b);
    auto result = linalg::solve(*mat_A, *vec_b);
    if (std::holds_alternative<Vectorf>(result)) {
        *out_x = new Vectorf(std::get<Vectorf>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_solve(MatrixDoubleHandle A, VectorDoubleHandle b, VectorDoubleHandle* out_x) {
    if (!A || !b || !out_x) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat_A = static_cast<Matrixd*>(A);
    auto* vec_b = static_cast<Vectord*>(b);
    auto result = linalg::solve(*mat_A, *vec_b);
    if (std::holds_alternative<Vectord>(result)) {
        *out_x = new Vectord(std::get<Vectord>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_solve_triangular(MatrixFloatHandle A, VectorFloatHandle b, bool lower, VectorFloatHandle* out_x) {
    if (!A || !b || !out_x) return TENSOR_ERROR_NULL_POINTER;
    
    snprintf(g_last_error, sizeof(g_last_error), "Triangular solve requires LAPACK/BLAS support (not available in current build)");
    return TENSOR_ERROR_INVALID_OPERATION;
}

TensorErrorCode matrix_double_solve_triangular(MatrixDoubleHandle A, VectorDoubleHandle b, bool lower, VectorDoubleHandle* out_x) {
    if (!A || !b || !out_x) return TENSOR_ERROR_NULL_POINTER;
    
    snprintf(g_last_error, sizeof(g_last_error), "Triangular solve requires LAPACK/BLAS support (not available in current build)");
    return TENSOR_ERROR_INVALID_OPERATION;
}

TensorErrorCode matrix_float_lstsq(MatrixFloatHandle A, VectorFloatHandle b, VectorFloatHandle* out_x) {
    if (!A || !b || !out_x) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat_A = static_cast<Matrixf*>(A);
    auto* vec_b = static_cast<Vectorf*>(b);
    auto result = linalg::lstsq(*mat_A, *vec_b);
    if (std::holds_alternative<Vectorf>(result)) {
        *out_x = new Vectorf(std::get<Vectorf>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_lstsq(MatrixDoubleHandle A, VectorDoubleHandle b, VectorDoubleHandle* out_x) {
    if (!A || !b || !out_x) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat_A = static_cast<Matrixd*>(A);
    auto* vec_b = static_cast<Vectord*>(b);
    auto result = linalg::lstsq(*mat_A, *vec_b);
    if (std::holds_alternative<Vectord>(result)) {
        *out_x = new Vectord(std::get<Vectord>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_pinv(MatrixFloatHandle handle, MatrixFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    auto result = linalg::pinv(*mat);
    if (std::holds_alternative<Matrixf>(result)) {
        *out_handle = new Matrixf(std::get<Matrixf>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_pinv(MatrixDoubleHandle handle, MatrixDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    auto result = linalg::pinv(*mat);
    if (std::holds_alternative<Matrixd>(result)) {
        *out_handle = new Matrixd(std::get<Matrixd>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_rank(MatrixFloatHandle handle, size_t* out_rank) {
    if (!handle || !out_rank) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    *out_rank = linalg::matrix_rank(*mat);
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_rank(MatrixDoubleHandle handle, size_t* out_rank) {
    if (!handle || !out_rank) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    *out_rank = linalg::matrix_rank(*mat);
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_kron(MatrixFloatHandle lhs, MatrixFloatHandle rhs, MatrixFloatHandle* out_handle) {
    if (!lhs || !rhs || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat_lhs = static_cast<Matrixf*>(lhs);
    auto* mat_rhs = static_cast<Matrixf*>(rhs);
    auto result = linalg::kron(*mat_lhs, *mat_rhs);
    if (std::holds_alternative<Matrixf>(result)) {
        *out_handle = new Matrixf(std::get<Matrixf>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_kron(MatrixDoubleHandle lhs, MatrixDoubleHandle rhs, MatrixDoubleHandle* out_handle) {
    if (!lhs || !rhs || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat_lhs = static_cast<Matrixd*>(lhs);
    auto* mat_rhs = static_cast<Matrixd*>(rhs);
    auto result = linalg::kron(*mat_lhs, *mat_rhs);
    if (std::holds_alternative<Matrixd>(result)) {
        *out_handle = new Matrixd(std::get<Matrixd>(result));
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode vector_float_cross(VectorFloatHandle lhs, VectorFloatHandle rhs, VectorFloatHandle* out_handle) {
    if (!lhs || !rhs || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec_lhs = static_cast<Vectorf*>(lhs);
    auto* vec_rhs = static_cast<Vectorf*>(rhs);
    
    // Verify 3D vectors
    if (vec_lhs->dims()[0] != 3 || vec_rhs->dims()[0] != 3) {
        snprintf(g_last_error, sizeof(g_last_error), "Cross product requires 3D vectors");
        return TENSOR_ERROR_SHAPE;
    }
    
    // Compute cross product: a × b = (a2b3 - a3b2, a3b1 - a1b3, a1b2 - a2b1)
    Vectorf* result = new Vectorf({3});
    (*result)[{0}] = (*vec_lhs)[{1}] * (*vec_rhs)[{2}] - (*vec_lhs)[{2}] * (*vec_rhs)[{1}];
    (*result)[{1}] = (*vec_lhs)[{2}] * (*vec_rhs)[{0}] - (*vec_lhs)[{0}] * (*vec_rhs)[{2}];
    (*result)[{2}] = (*vec_lhs)[{0}] * (*vec_rhs)[{1}] - (*vec_lhs)[{1}] * (*vec_rhs)[{0}];
    
    *out_handle = result;
    TENSOR_TRY_END
}

TensorErrorCode vector_double_cross(VectorDoubleHandle lhs, VectorDoubleHandle rhs, VectorDoubleHandle* out_handle) {
    if (!lhs || !rhs || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec_lhs = static_cast<Vectord*>(lhs);
    auto* vec_rhs = static_cast<Vectord*>(rhs);
    
    // Verify 3D vectors
    if (vec_lhs->dims()[0] != 3 || vec_rhs->dims()[0] != 3) {
        snprintf(g_last_error, sizeof(g_last_error), "Cross product requires 3D vectors");
        return TENSOR_ERROR_SHAPE;
    }
    
    // Compute cross product: a × b = (a2b3 - a3b2, a3b1 - a1b3, a1b2 - a2b1)
    Vectord* result = new Vectord({3});
    (*result)[{0}] = (*vec_lhs)[{1}] * (*vec_rhs)[{2}] - (*vec_lhs)[{2}] * (*vec_rhs)[{1}];
    (*result)[{1}] = (*vec_lhs)[{2}] * (*vec_rhs)[{0}] - (*vec_lhs)[{0}] * (*vec_rhs)[{2}];
    (*result)[{2}] = (*vec_lhs)[{0}] * (*vec_rhs)[{1}] - (*vec_lhs)[{1}] * (*vec_rhs)[{0}];
    
    *out_handle = result;
    TENSOR_TRY_END
}

// ===== Advanced Statistical Operations =====

TensorErrorCode vector_float_correlation(VectorFloatHandle lhs, VectorFloatHandle rhs, float* out_value) {
    if (!lhs || !rhs || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec_lhs = static_cast<Vectorf*>(lhs);
    auto* vec_rhs = static_cast<Vectorf*>(rhs);
    auto result = vec_lhs->correlation(*vec_rhs);
    if (std::holds_alternative<float>(result)) {
        *out_value = std::get<float>(result);
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode vector_double_correlation(VectorDoubleHandle lhs, VectorDoubleHandle rhs, double* out_value) {
    if (!lhs || !rhs || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec_lhs = static_cast<Vectord*>(lhs);
    auto* vec_rhs = static_cast<Vectord*>(rhs);
    auto result = vec_lhs->correlation(*vec_rhs);
    if (std::holds_alternative<double>(result)) {
        *out_value = std::get<double>(result);
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode vector_float_covariance(VectorFloatHandle lhs, VectorFloatHandle rhs, float* out_value) {
    if (!lhs || !rhs || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec_lhs = static_cast<Vectorf*>(lhs);
    auto* vec_rhs = static_cast<Vectorf*>(rhs);
    auto result = vec_lhs->covariance(*vec_rhs, 0);
    if (std::holds_alternative<float>(result)) {
        *out_value = std::get<float>(result);
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode vector_double_covariance(VectorDoubleHandle lhs, VectorDoubleHandle rhs, double* out_value) {
    if (!lhs || !rhs || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec_lhs = static_cast<Vectord*>(lhs);
    auto* vec_rhs = static_cast<Vectord*>(rhs);
    auto result = vec_lhs->covariance(*vec_rhs, 0);
    if (std::holds_alternative<double>(result)) {
        *out_value = std::get<double>(result);
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode vector_float_spearman(VectorFloatHandle lhs, VectorFloatHandle rhs, float* out_value) {
    if (!lhs || !rhs || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec_lhs = static_cast<Vectorf*>(lhs);
    auto* vec_rhs = static_cast<Vectorf*>(rhs);
    auto result = vec_lhs->spearman_correlation(*vec_rhs);
    if (std::holds_alternative<float>(result)) {
        *out_value = std::get<float>(result);
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode vector_double_spearman(VectorDoubleHandle lhs, VectorDoubleHandle rhs, double* out_value) {
    if (!lhs || !rhs || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec_lhs = static_cast<Vectord*>(lhs);
    auto* vec_rhs = static_cast<Vectord*>(rhs);
    auto result = vec_lhs->spearman_correlation(*vec_rhs);
    if (std::holds_alternative<double>(result)) {
        *out_value = std::get<double>(result);
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode vector_float_median(VectorFloatHandle handle, float* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    *out_value = vec->median();
    TENSOR_TRY_END
}

TensorErrorCode vector_double_median(VectorDoubleHandle handle, double* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    *out_value = vec->median();
    TENSOR_TRY_END
}

TensorErrorCode vector_float_quantile(VectorFloatHandle handle, float q, float* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    auto result = vec->quantile(q);
    if (std::holds_alternative<float>(result)) {
        *out_value = std::get<float>(result);
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode vector_double_quantile(VectorDoubleHandle handle, double q, double* out_value) {
    if (!handle || !out_value) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    auto result = vec->quantile(q);
    if (std::holds_alternative<double>(result)) {
        *out_value = std::get<double>(result);
    } else {
        return TENSOR_ERROR_COMPUTATION;
    }
    TENSOR_TRY_END
}

TensorErrorCode vector_float_standardize(VectorFloatHandle handle, VectorFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    *out_handle = new Vectorf(vec->standardize());
    TENSOR_TRY_END
}

TensorErrorCode vector_double_standardize(VectorDoubleHandle handle, VectorDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    *out_handle = new Vectord(vec->standardize());
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_standardize(MatrixFloatHandle handle, int axis, MatrixFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    // Note: standardize() doesn't take axis parameter in current implementation
    // It standardizes all elements. For axis-specific standardization, this needs to be implemented.
    *out_handle = new Matrixf(mat->standardize());
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_standardize(MatrixDoubleHandle handle, int axis, MatrixDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    // Note: standardize() doesn't take axis parameter in current implementation
    // It standardizes all elements. For axis-specific standardization, this needs to be implemented.
    *out_handle = new Matrixd(mat->standardize());
    TENSOR_TRY_END
}

TensorErrorCode vector_float_normalize(VectorFloatHandle handle, VectorFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    *out_handle = new Vectorf(vec->normalize());
    TENSOR_TRY_END
}

TensorErrorCode vector_double_normalize(VectorDoubleHandle handle, VectorDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    *out_handle = new Vectord(vec->normalize());
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_normalize(MatrixFloatHandle handle, int axis, MatrixFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    // Note: normalize() doesn't take axis parameter in current implementation
    // It normalizes all elements. For axis-specific normalization, this needs to be implemented.
    *out_handle = new Matrixf(mat->normalize());
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_normalize(MatrixDoubleHandle handle, int axis, MatrixDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    // Note: normalize() doesn't take axis parameter in current implementation
    // It normalizes all elements. For axis-specific normalization, this needs to be implemented.
    *out_handle = new Matrixd(mat->normalize());
    TENSOR_TRY_END
}

// ===== Mathematical Functions =====

TensorErrorCode vector_float_exp(VectorFloatHandle handle, VectorFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    *out_handle = new Vectorf(vec->exp());
    TENSOR_TRY_END
}

TensorErrorCode vector_float_log(VectorFloatHandle handle, VectorFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    *out_handle = new Vectorf(vec->log());
    TENSOR_TRY_END
}

TensorErrorCode vector_float_sqrt(VectorFloatHandle handle, VectorFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    *out_handle = new Vectorf(vec->sqrt());
    TENSOR_TRY_END
}

TensorErrorCode vector_float_sin(VectorFloatHandle handle, VectorFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    *out_handle = new Vectorf(vec->sin());
    TENSOR_TRY_END
}

TensorErrorCode vector_float_cos(VectorFloatHandle handle, VectorFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    *out_handle = new Vectorf(vec->cos());
    TENSOR_TRY_END
}

TensorErrorCode vector_float_tan(VectorFloatHandle handle, VectorFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    *out_handle = new Vectorf(vec->tan());
    TENSOR_TRY_END
}

TensorErrorCode vector_float_tanh(VectorFloatHandle handle, VectorFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    *out_handle = new Vectorf(vec->tanh());
    TENSOR_TRY_END
}

TensorErrorCode vector_float_sigmoid(VectorFloatHandle handle, VectorFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    *out_handle = new Vectorf(vec->sigmoid());
    TENSOR_TRY_END
}

TensorErrorCode vector_float_relu(VectorFloatHandle handle, VectorFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    *out_handle = new Vectorf(vec->relu());
    TENSOR_TRY_END
}

TensorErrorCode vector_double_exp(VectorDoubleHandle handle, VectorDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    *out_handle = new Vectord(vec->exp());
    TENSOR_TRY_END
}

TensorErrorCode vector_double_log(VectorDoubleHandle handle, VectorDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    *out_handle = new Vectord(vec->log());
    TENSOR_TRY_END
}

TensorErrorCode vector_double_sqrt(VectorDoubleHandle handle, VectorDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    *out_handle = new Vectord(vec->sqrt());
    TENSOR_TRY_END
}

TensorErrorCode vector_double_sin(VectorDoubleHandle handle, VectorDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    *out_handle = new Vectord(vec->sin());
    TENSOR_TRY_END
}

TensorErrorCode vector_double_cos(VectorDoubleHandle handle, VectorDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    *out_handle = new Vectord(vec->cos());
    TENSOR_TRY_END
}

TensorErrorCode vector_double_tan(VectorDoubleHandle handle, VectorDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    *out_handle = new Vectord(vec->tan());
    TENSOR_TRY_END
}

TensorErrorCode vector_double_tanh(VectorDoubleHandle handle, VectorDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    *out_handle = new Vectord(vec->tanh());
    TENSOR_TRY_END
}

TensorErrorCode vector_double_sigmoid(VectorDoubleHandle handle, VectorDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    *out_handle = new Vectord(vec->sigmoid());
    TENSOR_TRY_END
}

TensorErrorCode vector_double_relu(VectorDoubleHandle handle, VectorDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    *out_handle = new Vectord(vec->relu());
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_exp(MatrixFloatHandle handle, MatrixFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    *out_handle = new Matrixf(mat->exp());
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_log(MatrixFloatHandle handle, MatrixFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    *out_handle = new Matrixf(mat->log());
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_sqrt(MatrixFloatHandle handle, MatrixFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    *out_handle = new Matrixf(mat->sqrt());
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_tanh(MatrixFloatHandle handle, MatrixFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    *out_handle = new Matrixf(mat->tanh());
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_sigmoid(MatrixFloatHandle handle, MatrixFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    *out_handle = new Matrixf(mat->sigmoid());
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_relu(MatrixFloatHandle handle, MatrixFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    *out_handle = new Matrixf(mat->relu());
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_exp(MatrixDoubleHandle handle, MatrixDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    *out_handle = new Matrixd(mat->exp());
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_log(MatrixDoubleHandle handle, MatrixDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    *out_handle = new Matrixd(mat->log());
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_sqrt(MatrixDoubleHandle handle, MatrixDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    *out_handle = new Matrixd(mat->sqrt());
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_tanh(MatrixDoubleHandle handle, MatrixDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    *out_handle = new Matrixd(mat->tanh());
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_sigmoid(MatrixDoubleHandle handle, MatrixDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    *out_handle = new Matrixd(mat->sigmoid());
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_relu(MatrixDoubleHandle handle, MatrixDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    *out_handle = new Matrixd(mat->relu());
    TENSOR_TRY_END
}

// ===== Slicing and Views =====

TensorErrorCode matrix_float_get_row(MatrixFloatHandle handle, size_t row, VectorFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    *out_handle = new Vectorf(mat->row(row));
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_get_row(MatrixDoubleHandle handle, size_t row, VectorDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    *out_handle = new Vectord(mat->row(row));
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_get_col(MatrixFloatHandle handle, size_t col, VectorFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    *out_handle = new Vectorf(mat->col(col));
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_get_col(MatrixDoubleHandle handle, size_t col, VectorDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    *out_handle = new Vectord(mat->col(col));
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_get_diag(MatrixFloatHandle handle, VectorFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    *out_handle = new Vectorf(mat->diag());
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_get_diag(MatrixDoubleHandle handle, VectorDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    *out_handle = new Vectord(mat->diag());
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_submatrix(MatrixFloatHandle handle, size_t row_start, size_t row_end, size_t col_start, size_t col_end, MatrixFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    size_t num_rows = row_end - row_start;
    size_t num_cols = col_end - col_start;
    *out_handle = new Matrixf(mat->block(row_start, col_start, num_rows, num_cols));
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_submatrix(MatrixDoubleHandle handle, size_t row_start, size_t row_end, size_t col_start, size_t col_end, MatrixDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    size_t num_rows = row_end - row_start;
    size_t num_cols = col_end - col_start;
    *out_handle = new Matrixd(mat->block(row_start, col_start, num_rows, num_cols));
    TENSOR_TRY_END
}

TensorErrorCode vector_float_slice(VectorFloatHandle handle, size_t start, size_t end, VectorFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    Vectorf result({end - start});
    for (size_t i = 0; i < end - start; ++i) {
        result[{i}] = (*vec)[{start + i}];
    }
    *out_handle = new Vectorf(result);
    TENSOR_TRY_END
}

TensorErrorCode vector_double_slice(VectorDoubleHandle handle, size_t start, size_t end, VectorDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    Vectord result({end - start});
    for (size_t i = 0; i < end - start; ++i) {
        result[{i}] = (*vec)[{start + i}];
    }
    *out_handle = new Vectord(result);
    TENSOR_TRY_END
}

// ===== Random Number Generation =====

TensorErrorCode vector_float_random_uniform(size_t size, float low, float high, VectorFloatHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    Vectorf vec({size});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(low, high);
    for (size_t i = 0; i < size; ++i) {
        vec[{i}] = dist(gen);
    }
    *out_handle = new Vectorf(vec);
    TENSOR_TRY_END
}

TensorErrorCode vector_double_random_uniform(size_t size, double low, double high, VectorDoubleHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    Vectord vec({size});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(low, high);
    for (size_t i = 0; i < size; ++i) {
        vec[{i}] = dist(gen);
    }
    *out_handle = new Vectord(vec);
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_random_uniform(size_t rows, size_t cols, float low, float high, MatrixFloatHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    Matrixf mat({rows, cols});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(low, high);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            mat[{i, j}] = dist(gen);
        }
    }
    *out_handle = new Matrixf(mat);
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_random_uniform(size_t rows, size_t cols, double low, double high, MatrixDoubleHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    Matrixd mat({rows, cols});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(low, high);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            mat[{i, j}] = dist(gen);
        }
    }
    *out_handle = new Matrixd(mat);
    TENSOR_TRY_END
}

TensorErrorCode vector_float_random_normal(size_t size, float mean, float std, VectorFloatHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    Vectorf vec({size});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, std);
    for (size_t i = 0; i < size; ++i) {
        vec[{i}] = dist(gen);
    }
    *out_handle = new Vectorf(vec);
    TENSOR_TRY_END
}

TensorErrorCode vector_double_random_normal(size_t size, double mean, double std, VectorDoubleHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    Vectord vec({size});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(mean, std);
    for (size_t i = 0; i < size; ++i) {
        vec[{i}] = dist(gen);
    }
    *out_handle = new Vectord(vec);
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_random_normal(size_t rows, size_t cols, float mean, float std, MatrixFloatHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    Matrixf mat({rows, cols});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, std);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            mat[{i, j}] = dist(gen);
        }
    }
    *out_handle = new Matrixf(mat);
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_random_normal(size_t rows, size_t cols, double mean, double std, MatrixDoubleHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    Matrixd mat({rows, cols});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(mean, std);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            mat[{i, j}] = dist(gen);
        }
    }
    *out_handle = new Matrixd(mat);
    TENSOR_TRY_END
}

// ===== Utility Functions =====

TensorErrorCode vector_float_copy(VectorFloatHandle handle, VectorFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    *out_handle = new Vectorf(*vec);
    TENSOR_TRY_END
}

TensorErrorCode vector_double_copy(VectorDoubleHandle handle, VectorDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    *out_handle = new Vectord(*vec);
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_copy(MatrixFloatHandle handle, MatrixFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    *out_handle = new Matrixf(*mat);
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_copy(MatrixDoubleHandle handle, MatrixDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    *out_handle = new Matrixd(*mat);
    TENSOR_TRY_END
}

TensorErrorCode vector_float_print(VectorFloatHandle handle) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    auto shape = vec->shape();
    std::cout << "Vector[" << shape[0] << "]: [";
    for (size_t i = 0; i < shape[0]; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << (*vec)[{i}];
    }
    std::cout << "]" << std::endl;
    TENSOR_TRY_END
}

TensorErrorCode vector_double_print(VectorDoubleHandle handle) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    auto shape = vec->shape();
    std::cout << "Vector[" << shape[0] << "]: [";
    for (size_t i = 0; i < shape[0]; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << (*vec)[{i}];
    }
    std::cout << "]" << std::endl;
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_print(MatrixFloatHandle handle) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    auto shape = mat->shape();
    std::cout << "Matrix[" << shape[0] << "x" << shape[1] << "]:" << std::endl;
    for (size_t i = 0; i < shape[0]; ++i) {
        std::cout << "[";
        for (size_t j = 0; j < shape[1]; ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << (*mat)[{i, j}];
        }
        std::cout << "]" << std::endl;
    }
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_print(MatrixDoubleHandle handle) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    auto shape = mat->shape();
    std::cout << "Matrix[" << shape[0] << "x" << shape[1] << "]:" << std::endl;
    for (size_t i = 0; i < shape[0]; ++i) {
        std::cout << "[";
        for (size_t j = 0; j < shape[1]; ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << (*mat)[{i, j}];
        }
        std::cout << "]" << std::endl;
    }
    TENSOR_TRY_END
}

TensorErrorCode vector_float_data(VectorFloatHandle handle, const float** out_data) {
    if (!handle || !out_data) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectorf*>(handle);
    *out_data = vec->data();
    TENSOR_TRY_END
}

TensorErrorCode vector_double_data(VectorDoubleHandle handle, const double** out_data) {
    if (!handle || !out_data) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* vec = static_cast<Vectord*>(handle);
    *out_data = vec->data();
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_data(MatrixFloatHandle handle, const float** out_data) {
    if (!handle || !out_data) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    *out_data = mat->data();
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_data(MatrixDoubleHandle handle, const double** out_data) {
    if (!handle || !out_data) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    *out_data = mat->data();
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_reshape(MatrixFloatHandle handle, size_t new_rows, size_t new_cols, MatrixFloatHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    TensorIndices<2> new_shape = {new_rows, new_cols};
    auto result = mat->reshape<2>(new_shape);
    *out_handle = new Matrixf(result);
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_reshape(MatrixDoubleHandle handle, size_t new_rows, size_t new_cols, MatrixDoubleHandle* out_handle) {
    if (!handle || !out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    TensorIndices<2> new_shape = {new_rows, new_cols};
    auto result = mat->reshape<2>(new_shape);
    *out_handle = new Matrixd(result);
    TENSOR_TRY_END
}

bool tensor_c_is_gpu_available(void) {
#ifdef USE_GPU
    return is_gpu_available();
#else
    return false;
#endif
}

TensorErrorCode matrix_float_get_backend(MatrixFloatHandle handle, TensorBackend* out_backend) {
    if (!handle || !out_backend) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(handle);
    Backend backend = mat->backend();
    
    switch(backend) {
        case Backend::CPU:
            *out_backend = TENSOR_BACKEND_CPU;
            break;
        case Backend::BLAS:
            *out_backend = TENSOR_BACKEND_BLAS;
            break;
        case Backend::GPU:
            *out_backend = TENSOR_BACKEND_GPU;
            break;
    }
    return TENSOR_SUCCESS;
    TENSOR_TRY_END
}

TensorErrorCode matrix_double_get_backend(MatrixDoubleHandle handle, TensorBackend* out_backend) {
    if (!handle || !out_backend) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixd*>(handle);
    Backend backend = mat->backend();
    
    switch(backend) {
        case Backend::CPU:
            *out_backend = TENSOR_BACKEND_CPU;
            break;
        case Backend::BLAS:
            *out_backend = TENSOR_BACKEND_BLAS;
            break;
        case Backend::GPU:
            *out_backend = TENSOR_BACKEND_GPU;
            break;
    }
    return TENSOR_SUCCESS;
    TENSOR_TRY_END
}

const char* tensor_c_backend_name(TensorBackend backend) {
    switch(backend) {
        case TENSOR_BACKEND_CPU: return "CPU";
        case TENSOR_BACKEND_BLAS: return "BLAS";
        case TENSOR_BACKEND_GPU: return "GPU";
        default: return "Unknown";
    }
}

const char* tensor_c_version(void) {
    return "1.0.0";
}

const char* tensor_c_last_error(void) {
    return g_last_error;
}

// ============================================================================
// Neural Network Layers C API Implementation
// ============================================================================

// ===== Linear Layer =====

TensorErrorCode layer_linear_create_float(size_t in_features, size_t out_features, bool use_bias, LayerHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = new Linear<float>(in_features, out_features, use_bias);
    *out_handle = layer;
    TENSOR_TRY_END
}

TensorErrorCode layer_linear_create_double(size_t in_features, size_t out_features, bool use_bias, LayerHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = new Linear<double>(in_features, out_features, use_bias);
    *out_handle = layer;
    TENSOR_TRY_END
}

TensorErrorCode layer_linear_forward_float(LayerHandle handle, MatrixFloatHandle input, MatrixFloatHandle* output) {
    if (!handle || !input || !output) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<Linear<float>*>(handle);
    auto* input_mat = static_cast<Matrixf*>(input);
    auto result = layer->forward(*input_mat);
    *output = new Matrixf(result);
    TENSOR_TRY_END
}

TensorErrorCode layer_linear_forward_double(LayerHandle handle, MatrixDoubleHandle input, MatrixDoubleHandle* output) {
    if (!handle || !input || !output) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<Linear<double>*>(handle);
    auto* input_mat = static_cast<Matrixd*>(input);
    auto result = layer->forward(*input_mat);
    *output = new Matrixd(result);
    TENSOR_TRY_END
}

TensorErrorCode layer_linear_backward_float(LayerHandle handle, MatrixFloatHandle grad_output, MatrixFloatHandle* grad_input) {
    if (!handle || !grad_output || !grad_input) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<Linear<float>*>(handle);
    auto* grad_out_mat = static_cast<Matrixf*>(grad_output);
    auto result = layer->backward(*grad_out_mat);
    *grad_input = new Matrixf(result);
    TENSOR_TRY_END
}

TensorErrorCode layer_linear_backward_double(LayerHandle handle, MatrixDoubleHandle grad_output, MatrixDoubleHandle* grad_input) {
    if (!handle || !grad_output || !grad_input) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<Linear<double>*>(handle);
    auto* grad_out_mat = static_cast<Matrixd*>(grad_output);
    auto result = layer->backward(*grad_out_mat);
    *grad_input = new Matrixd(result);
    TENSOR_TRY_END
}

TensorErrorCode layer_linear_get_weights_float(LayerHandle handle, MatrixFloatHandle* weights) {
    if (!handle || !weights) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<Linear<float>*>(handle);
    *weights = &(layer->weights());
    TENSOR_TRY_END
}

TensorErrorCode layer_linear_get_bias_float(LayerHandle handle, MatrixFloatHandle* bias) {
    if (!handle || !bias) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<Linear<float>*>(handle);
    *bias = &(layer->bias());
    TENSOR_TRY_END
}

TensorErrorCode layer_linear_get_grad_weights_float(LayerHandle handle, MatrixFloatHandle* grad_weights) {
    if (!handle || !grad_weights) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<Linear<float>*>(handle);
    *grad_weights = &(layer->grad_weights());
    TENSOR_TRY_END
}

TensorErrorCode layer_linear_get_grad_bias_float(LayerHandle handle, MatrixFloatHandle* grad_bias) {
    if (!handle || !grad_bias) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<Linear<float>*>(handle);
    *grad_bias = &(layer->grad_bias());
    TENSOR_TRY_END
}

TensorErrorCode layer_linear_update_weights_float(LayerHandle handle, float learning_rate) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<Linear<float>*>(handle);
    
    // Update weights: weights = weights - learning_rate * grad_weights
    auto& weights = layer->weights();
    auto& grad_weights = layer->grad_weights();
    
    size_t rows, cols;
    auto w_shape = weights.shape();
    rows = w_shape[0];
    cols = w_shape[1];
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            weights[{i, j}] -= learning_rate * grad_weights[{i, j}];
        }
    }
    
    // Update bias: bias = bias - learning_rate * grad_bias
    auto& bias = layer->bias();
    auto& grad_bias = layer->grad_bias();
    auto b_shape = bias.shape();
    size_t bias_cols = b_shape[1];
    
    for (size_t j = 0; j < bias_cols; ++j) {
        bias[{0, j}] -= learning_rate * grad_bias[{0, j}];
    }
    
    TENSOR_TRY_END
}

TensorErrorCode layer_linear_destroy(LayerHandle handle) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    // We need to determine the type - for simplicity, we'll use a tagged approach
    // For now, just delete as Linear<float>* (user must match create/destroy types)
    delete static_cast<Linear<float>*>(handle);
    TENSOR_TRY_END
}

// ===== ReLU Layer =====

TensorErrorCode layer_relu_create_float(LayerHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = new ReLU<float>();
    *out_handle = layer;
    TENSOR_TRY_END
}

TensorErrorCode layer_relu_create_double(LayerHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = new ReLU<double>();
    *out_handle = layer;
    TENSOR_TRY_END
}

TensorErrorCode layer_relu_forward_float(LayerHandle handle, MatrixFloatHandle input, MatrixFloatHandle* output) {
    if (!handle || !input || !output) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<ReLU<float>*>(handle);
    auto* input_mat = static_cast<Matrixf*>(input);
    auto result = layer->forward(*input_mat);
    *output = new Matrixf(result);
    TENSOR_TRY_END
}

TensorErrorCode layer_relu_forward_double(LayerHandle handle, MatrixDoubleHandle input, MatrixDoubleHandle* output) {
    if (!handle || !input || !output) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<ReLU<double>*>(handle);
    auto* input_mat = static_cast<Matrixd*>(input);
    auto result = layer->forward(*input_mat);
    *output = new Matrixd(result);
    TENSOR_TRY_END
}

TensorErrorCode layer_relu_backward_float(LayerHandle handle, MatrixFloatHandle grad_output, MatrixFloatHandle* grad_input) {
    if (!handle || !grad_output || !grad_input) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<ReLU<float>*>(handle);
    auto* grad_out_mat = static_cast<Matrixf*>(grad_output);
    auto result = layer->backward(*grad_out_mat);
    *grad_input = new Matrixf(result);
    TENSOR_TRY_END
}

TensorErrorCode layer_relu_backward_double(LayerHandle handle, MatrixDoubleHandle grad_output, MatrixDoubleHandle* grad_input) {
    if (!handle || !grad_output || !grad_input) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<ReLU<double>*>(handle);
    auto* grad_out_mat = static_cast<Matrixd*>(grad_output);
    auto result = layer->backward(*grad_out_mat);
    *grad_input = new Matrixd(result);
    TENSOR_TRY_END
}

TensorErrorCode layer_relu_destroy(LayerHandle handle) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    delete static_cast<ReLU<float>*>(handle);
    TENSOR_TRY_END
}

// ===== Sigmoid Layer =====

TensorErrorCode layer_sigmoid_create_float(LayerHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = new Sigmoid<float>();
    *out_handle = layer;
    TENSOR_TRY_END
}

TensorErrorCode layer_sigmoid_create_double(LayerHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = new Sigmoid<double>();
    *out_handle = layer;
    TENSOR_TRY_END
}

TensorErrorCode layer_sigmoid_forward_float(LayerHandle handle, MatrixFloatHandle input, MatrixFloatHandle* output) {
    if (!handle || !input || !output) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<Sigmoid<float>*>(handle);
    auto* input_mat = static_cast<Matrixf*>(input);
    auto result = layer->forward(*input_mat);
    *output = new Matrixf(result);
    TENSOR_TRY_END
}

TensorErrorCode layer_sigmoid_forward_double(LayerHandle handle, MatrixDoubleHandle input, MatrixDoubleHandle* output) {
    if (!handle || !input || !output) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<Sigmoid<double>*>(handle);
    auto* input_mat = static_cast<Matrixd*>(input);
    auto result = layer->forward(*input_mat);
    *output = new Matrixd(result);
    TENSOR_TRY_END
}

TensorErrorCode layer_sigmoid_backward_float(LayerHandle handle, MatrixFloatHandle grad_output, MatrixFloatHandle* grad_input) {
    if (!handle || !grad_output || !grad_input) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<Sigmoid<float>*>(handle);
    auto* grad_out_mat = static_cast<Matrixf*>(grad_output);
    auto result = layer->backward(*grad_out_mat);
    *grad_input = new Matrixf(result);
    TENSOR_TRY_END
}

TensorErrorCode layer_sigmoid_backward_double(LayerHandle handle, MatrixDoubleHandle grad_output, MatrixDoubleHandle* grad_input) {
    if (!handle || !grad_output || !grad_input) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<Sigmoid<double>*>(handle);
    auto* grad_out_mat = static_cast<Matrixd*>(grad_output);
    auto result = layer->backward(*grad_out_mat);
    *grad_input = new Matrixd(result);
    TENSOR_TRY_END
}

TensorErrorCode layer_sigmoid_destroy(LayerHandle handle) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    delete static_cast<Sigmoid<float>*>(handle);
    TENSOR_TRY_END
}

// ===== Softmax Layer =====

TensorErrorCode layer_softmax_create_float(LayerHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = new Softmax<float>();
    *out_handle = layer;
    TENSOR_TRY_END
}

TensorErrorCode layer_softmax_create_double(LayerHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = new Softmax<double>();
    *out_handle = layer;
    TENSOR_TRY_END
}

TensorErrorCode layer_softmax_forward_float(LayerHandle handle, MatrixFloatHandle input, MatrixFloatHandle* output) {
    if (!handle || !input || !output) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<Softmax<float>*>(handle);
    auto* input_mat = static_cast<Matrixf*>(input);
    auto result = layer->forward(*input_mat);
    *output = new Matrixf(result);
    TENSOR_TRY_END
}

TensorErrorCode layer_softmax_forward_double(LayerHandle handle, MatrixDoubleHandle input, MatrixDoubleHandle* output) {
    if (!handle || !input || !output) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<Softmax<double>*>(handle);
    auto* input_mat = static_cast<Matrixd*>(input);
    auto result = layer->forward(*input_mat);
    *output = new Matrixd(result);
    TENSOR_TRY_END
}

TensorErrorCode layer_softmax_backward_float(LayerHandle handle, MatrixFloatHandle grad_output, MatrixFloatHandle* grad_input) {
    if (!handle || !grad_output || !grad_input) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<Softmax<float>*>(handle);
    auto* grad_out_mat = static_cast<Matrixf*>(grad_output);
    auto result = layer->backward(*grad_out_mat);
    *grad_input = new Matrixf(result);
    TENSOR_TRY_END
}

TensorErrorCode layer_softmax_backward_double(LayerHandle handle, MatrixDoubleHandle grad_output, MatrixDoubleHandle* grad_input) {
    if (!handle || !grad_output || !grad_input) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<Softmax<double>*>(handle);
    auto* grad_out_mat = static_cast<Matrixd*>(grad_output);
    auto result = layer->backward(*grad_out_mat);
    *grad_input = new Matrixd(result);
    TENSOR_TRY_END
}

TensorErrorCode layer_softmax_destroy(LayerHandle handle) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    delete static_cast<Softmax<float>*>(handle);
    TENSOR_TRY_END
}

// ===== Dropout Layer =====

TensorErrorCode layer_dropout_create_float(float p, LayerHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = new Dropout<float>(p);
    *out_handle = layer;
    TENSOR_TRY_END
}

TensorErrorCode layer_dropout_create_double(double p, LayerHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = new Dropout<double>(p);
    *out_handle = layer;
    TENSOR_TRY_END
}

TensorErrorCode layer_dropout_forward_float(LayerHandle handle, MatrixFloatHandle input, MatrixFloatHandle* output) {
    if (!handle || !input || !output) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<Dropout<float>*>(handle);
    auto* input_mat = static_cast<Matrixf*>(input);
    auto result = layer->forward(*input_mat);
    *output = new Matrixf(result);
    TENSOR_TRY_END
}

TensorErrorCode layer_dropout_forward_double(LayerHandle handle, MatrixDoubleHandle input, MatrixDoubleHandle* output) {
    if (!handle || !input || !output) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<Dropout<double>*>(handle);
    auto* input_mat = static_cast<Matrixd*>(input);
    auto result = layer->forward(*input_mat);
    *output = new Matrixd(result);
    TENSOR_TRY_END
}

TensorErrorCode layer_dropout_backward_float(LayerHandle handle, MatrixFloatHandle grad_output, MatrixFloatHandle* grad_input) {
    if (!handle || !grad_output || !grad_input) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<Dropout<float>*>(handle);
    auto* grad_out_mat = static_cast<Matrixf*>(grad_output);
    auto result = layer->backward(*grad_out_mat);
    *grad_input = new Matrixf(result);
    TENSOR_TRY_END
}

TensorErrorCode layer_dropout_backward_double(LayerHandle handle, MatrixDoubleHandle grad_output, MatrixDoubleHandle* grad_input) {
    if (!handle || !grad_output || !grad_input) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<Dropout<double>*>(handle);
    auto* grad_out_mat = static_cast<Matrixd*>(grad_output);
    auto result = layer->backward(*grad_out_mat);
    *grad_input = new Matrixd(result);
    TENSOR_TRY_END
}

TensorErrorCode layer_dropout_train(LayerHandle handle, bool training) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    // Try float first (user must match types)
    auto* layer = static_cast<Dropout<float>*>(handle);
    layer->train(training);
    TENSOR_TRY_END
}

TensorErrorCode layer_dropout_destroy(LayerHandle handle) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    delete static_cast<Dropout<float>*>(handle);
    TENSOR_TRY_END
}

// ===== Batch Normalization Layer =====

TensorErrorCode layer_batchnorm_create_float(size_t num_features, float eps, float momentum, LayerHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = new BatchNorm1d<float>(num_features, eps, momentum);
    *out_handle = layer;
    TENSOR_TRY_END
}

TensorErrorCode layer_batchnorm_create_double(size_t num_features, double eps, double momentum, LayerHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = new BatchNorm1d<double>(num_features, eps, momentum);
    *out_handle = layer;
    TENSOR_TRY_END
}

TensorErrorCode layer_batchnorm_forward_float(LayerHandle handle, MatrixFloatHandle input, MatrixFloatHandle* output) {
    if (!handle || !input || !output) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<BatchNorm1d<float>*>(handle);
    auto* input_mat = static_cast<Matrixf*>(input);
    auto result = layer->forward(*input_mat);
    *output = new Matrixf(result);
    TENSOR_TRY_END
}

TensorErrorCode layer_batchnorm_forward_double(LayerHandle handle, MatrixDoubleHandle input, MatrixDoubleHandle* output) {
    if (!handle || !input || !output) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<BatchNorm1d<double>*>(handle);
    auto* input_mat = static_cast<Matrixd*>(input);
    auto result = layer->forward(*input_mat);
    *output = new Matrixd(result);
    TENSOR_TRY_END
}

TensorErrorCode layer_batchnorm_backward_float(LayerHandle handle, MatrixFloatHandle grad_output, MatrixFloatHandle* grad_input) {
    if (!handle || !grad_output || !grad_input) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<BatchNorm1d<float>*>(handle);
    auto* grad_out_mat = static_cast<Matrixf*>(grad_output);
    auto result = layer->backward(*grad_out_mat);
    *grad_input = new Matrixf(result);
    TENSOR_TRY_END
}

TensorErrorCode layer_batchnorm_backward_double(LayerHandle handle, MatrixDoubleHandle grad_output, MatrixDoubleHandle* grad_input) {
    if (!handle || !grad_output || !grad_input) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<BatchNorm1d<double>*>(handle);
    auto* grad_out_mat = static_cast<Matrixd*>(grad_output);
    auto result = layer->backward(*grad_out_mat);
    *grad_input = new Matrixd(result);
    TENSOR_TRY_END
}

TensorErrorCode layer_batchnorm_train(LayerHandle handle, bool training) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* layer = static_cast<BatchNorm1d<float>*>(handle);
    layer->train(training);
    TENSOR_TRY_END
}

TensorErrorCode layer_batchnorm_destroy(LayerHandle handle) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    delete static_cast<BatchNorm1d<float>*>(handle);
    TENSOR_TRY_END
}

// ===== Loss Functions =====

TensorErrorCode matrix_float_cross_entropy_loss(MatrixFloatHandle predictions, 
                                                 MatrixFloatHandle targets, 
                                                 float* out_loss) {
    if (!predictions || !targets || !out_loss) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* pred_mat = static_cast<Matrixf*>(predictions);
    auto* target_mat = static_cast<Matrixf*>(targets);
    
    auto pred_shape = pred_mat->shape();
    auto target_shape = target_mat->shape();
    
    size_t rows = pred_shape[0];
    size_t cols = pred_shape[1];
    
    if (pred_shape[0] != target_shape[0] || pred_shape[1] != target_shape[1]) {
        snprintf(g_last_error, sizeof(g_last_error), "Shape mismatch: predictions and targets must have same shape");
        return TENSOR_ERROR_SHAPE;
    }
    
    // Compute cross-entropy loss using optimized operations
    // L = -sum(targets * log(predictions + epsilon)) / batch_size
    float epsilon = 1e-7f;
    float loss = 0.0f;
    
    // Element-wise: targets * log(predictions + epsilon)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            float pred = (*pred_mat)[{i, j}] + epsilon;
            float target = (*target_mat)[{i, j}];
            loss -= target * std::log(pred);
        }
    }
    
    *out_loss = loss / static_cast<float>(rows);
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_compute_accuracy(MatrixFloatHandle predictions,
                                               const uint8_t* labels,
                                               size_t batch_size,
                                               float* out_accuracy) {
    if (!predictions || !labels || !out_accuracy) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* pred_mat = static_cast<Matrixf*>(predictions);
    
    auto shape = pred_mat->shape();
    size_t rows = shape[0];
    size_t cols = shape[1];
    
    if (rows != batch_size) {
        snprintf(g_last_error, sizeof(g_last_error), "Shape mismatch: predictions rows (%zu) != batch_size (%zu)", rows, batch_size);
        return TENSOR_ERROR_SHAPE;
    }
    
    size_t correct = 0;
    
    // For each sample, find argmax of predictions and compare with true label
    for (size_t i = 0; i < rows; ++i) {
        size_t pred_class = 0;
        float max_val = (*pred_mat)[{i, 0}];
        
        for (size_t j = 1; j < cols; ++j) {
            float val = (*pred_mat)[{i, j}];
            if (val > max_val) {
                max_val = val;
                pred_class = j;
            }
        }
        
        if (pred_class == labels[i]) {
            correct++;
        }
    }
    
    *out_accuracy = static_cast<float>(correct) / static_cast<float>(batch_size);
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_softmax(MatrixFloatHandle input, MatrixFloatHandle* output) {
    if (!input || !output) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* input_mat = static_cast<Matrixf*>(input);
    
    auto shape = input_mat->shape();
    size_t rows = shape[0];
    size_t cols = shape[1];
    
    // Create output matrix
    auto* result = new Matrixf({rows, cols}, input_mat->uses_gpu());
    
    // Numerically stable softmax: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))
    for (size_t i = 0; i < rows; ++i) {
        // Find max in row for numerical stability
        float max_val = (*input_mat)[{i, 0}];
        for (size_t j = 1; j < cols; ++j) {
            float val = (*input_mat)[{i, j}];
            if (val > max_val) {
                max_val = val;
            }
        }
        
        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            float exp_val = std::exp((*input_mat)[{i, j}] - max_val);
            (*result)[{i, j}] = exp_val;
            sum += exp_val;
        }
        
        // Normalize
        for (size_t j = 0; j < cols; ++j) {
            (*result)[{i, j}] /= sum;
        }
    }
    
    *output = result;
    TENSOR_TRY_END
}

TensorErrorCode matrix_float_argmax_rows(MatrixFloatHandle matrix,
                                         size_t* out_indices,
                                         size_t batch_size) {
    if (!matrix || !out_indices) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    auto* mat = static_cast<Matrixf*>(matrix);
    
    auto shape = mat->shape();
    size_t rows = shape[0];
    size_t cols = shape[1];
    
    if (rows != batch_size) {
        snprintf(g_last_error, sizeof(g_last_error), "Shape mismatch: matrix rows (%zu) != batch_size (%zu)", rows, batch_size);
        return TENSOR_ERROR_SHAPE;
    }
    
    // Find argmax for each row
    for (size_t i = 0; i < rows; ++i) {
        size_t argmax = 0;
        float max_val = (*mat)[{i, 0}];
        
        for (size_t j = 1; j < cols; ++j) {
            float val = (*mat)[{i, j}];
            if (val > max_val) {
                max_val = val;
                argmax = j;
            }
        }
        
        out_indices[i] = argmax;
    }
    
    TENSOR_TRY_END
}

} // extern "C"
