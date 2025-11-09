/**
 * @file tensor_wrapper.cc
 * @brief Python bindings for the Tensor4D library using pybind11
 * 
 * This file provides Python bindings for the instantiated tensor types:
 * - Vectorf, Vectord (1D tensors)
 * - Matrixf, Matrixd (2D tensors)
 * - Tensor3f, Tensor3d (3D tensors)
 * - Tensor4f, Tensor4d (4D tensors)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>
#include <iostream>

#include "tensor.h"
#include "tensor_types.h"
#include "linalg.h"
#include "loss_functions.h"
#include "tensor_io.h"
#include "optimizers.h"

namespace py = pybind11;
using namespace tensor4d;

// Helper function to flatten nested Python list
template<typename T>
void flatten_list(py::handle obj, std::vector<T>& data, std::vector<size_t>& shape, size_t depth = 0) {
    if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
        auto seq = obj.cast<py::sequence>();
        if (depth >= shape.size()) {
            shape.push_back(seq.size());
        }
        for (size_t i = 0; i < seq.size(); ++i) {
            flatten_list<T>(seq[i], data, shape, depth + 1);
        }
    } else {
        // Leaf element
        data.push_back(obj.cast<T>());
    }
}

// Helper function to convert Python list/tuple to Tensor
template<typename T, size_t N>
Tensor<T, N> list_to_tensor(py::handle obj) {
    std::vector<T> data;
    std::vector<size_t> shape;
    
    flatten_list<T>(obj, data, shape, 0);
    
    if (shape.size() != N) {
        throw std::runtime_error("List dimension mismatch: expected " + std::to_string(N) + 
                               " dimensions, got " + std::to_string(shape.size()));
    }
    
    std::array<size_t, N> arr_shape;
    for (size_t i = 0; i < N; ++i) {
        arr_shape[i] = shape[i];
    }
    
    Tensor<T, N> tensor(arr_shape);
    if (data.size() != tensor.total_size()) {
        throw std::runtime_error("Data size mismatch");
    }
    
    for (size_t i = 0; i < data.size(); ++i) {
        tensor.data()[i] = data[i];
    }
    
    return tensor;
}

// Helper function to convert Tensor to nested Python list
template<typename T, size_t N>
py::list tensor_to_list(const Tensor<T, N>& tensor) {
    std::function<py::list(const T*, const std::array<size_t, N>&, size_t)> convert;
    
    convert = [&](const T* data, const std::array<size_t, N>& shape, size_t dim) -> py::list {
        py::list result;
        if (dim == N - 1) {
            // Last dimension: add elements directly
            for (size_t i = 0; i < shape[dim]; ++i) {
                result.append(data[i]);
            }
        } else {
            // Recursive case
            size_t stride = 1;
            for (size_t d = dim + 1; d < N; ++d) {
                stride *= shape[d];
            }
            for (size_t i = 0; i < shape[dim]; ++i) {
                result.append(convert(data + i * stride, shape, dim + 1));
            }
        }
        return result;
    };
    
    return convert(tensor.data(), tensor.shape(), 0);
}

// Binding for a single Tensor type
template<typename T, size_t N>
py::class_<Tensor<T, N>> bind_tensor(py::module& m, const std::string& name) {
    using TensorType = Tensor<T, N>;
    
    py::class_<TensorType> cls(m, name.c_str());
    
    cls.def(py::init<const std::array<size_t, N>&>(), py::arg("shape"),
             "Create a tensor with the given shape")
        .def(py::init([](py::handle data) {
            return list_to_tensor<T, N>(data);
        }), py::arg("data"), "Create tensor from Python list/tuple")
        
        // Properties
        .def_property_readonly("shape", [](const TensorType& t) {
            auto shp = t.shape();
            py::tuple result(N);
            for (size_t i = 0; i < N; ++i) {
                result[i] = shp[i];
            }
            return result;
        }, "Get tensor shape")
        .def_property_readonly("size", &TensorType::total_size, "Get total number of elements")
        .def_property_readonly("requires_grad", &TensorType::requires_grad, "Check if tensor requires gradient")
        .def("set_requires_grad", &TensorType::set_requires_grad, py::arg("value"),
             "Set requires_grad flag")
        
        // Element access
        .def("__getitem__", [](const TensorType& t, const std::array<size_t, N>& idx) {
            return t[idx];
        })
        .def("__setitem__", [](TensorType& t, const std::array<size_t, N>& idx, T value) {
            t[idx] = value;
        })
        .def("at", [](const TensorType& t, const std::vector<size_t>& idx) {
            std::array<size_t, N> arr_idx;
            for (size_t i = 0; i < N && i < idx.size(); ++i) {
                arr_idx[i] = idx[i];
            }
            return t[arr_idx];
        }, py::arg("index"), "Access element at index")
        
        // Fill operations
        .def("fill", &TensorType::fill, py::arg("value"), "Fill tensor with value")
        
        // Arithmetic operations - tensor-tensor (returns TensorResult)
        .def("__add__", [](const TensorType& a, const TensorType& b) -> TensorType {
            auto result = a + b;
            if (std::holds_alternative<TensorError>(result)) {
                throw std::runtime_error("Tensor addition failed");
            }
            return std::get<TensorType>(result);
        })
        .def("__sub__", [](const TensorType& a, const TensorType& b) -> TensorType {
            auto result = a - b;
            if (std::holds_alternative<TensorError>(result)) {
                throw std::runtime_error("Tensor subtraction failed");
            }
            return std::get<TensorType>(result);
        })
        .def("__mul__", [](const TensorType& a, const TensorType& b) -> TensorType {
            auto result = a * b;
            if (std::holds_alternative<TensorError>(result)) {
                throw std::runtime_error("Tensor multiplication failed");
            }
            return std::get<TensorType>(result);
        })
        .def("__truediv__", [](const TensorType& a, const TensorType& b) -> TensorType {
            auto result = a / b;
            if (std::holds_alternative<TensorError>(result)) {
                throw std::runtime_error("Tensor division failed");
            }
            return std::get<TensorType>(result);
        })
        // Arithmetic operations - tensor-scalar (returns Tensor directly)
        .def("__add__", [](const TensorType& a, T scalar) -> TensorType {
            return a + scalar;
        })
        .def("__sub__", [](const TensorType& a, T scalar) -> TensorType {
            return a - scalar;
        })
        .def("__mul__", [](const TensorType& a, T scalar) -> TensorType {
            return a * scalar;
        })
        .def("__truediv__", [](const TensorType& a, T scalar) -> TensorType {
            return a / scalar;
        })
        // Arithmetic operations - scalar-tensor (returns Tensor directly)
        .def("__radd__", [](const TensorType& a, T scalar) -> TensorType {
            return scalar + a;
        })
        .def("__rsub__", [](const TensorType& a, T scalar) -> TensorType {
            return scalar - a;
        })
        .def("__rmul__", [](const TensorType& a, T scalar) -> TensorType {
            return scalar * a;
        })
        .def("__rtruediv__", [](const TensorType& a, T scalar) -> TensorType {
            return scalar / a;
        })
        .def("__iadd__", [](TensorType& self, const TensorType& other) -> TensorType& { 
            self += other; 
            return self; 
        })
        .def("__isub__", [](TensorType& self, const TensorType& other) -> TensorType& { 
            self -= other; 
            return self; 
        })
        .def("__imul__", [](TensorType& self, const TensorType& other) -> TensorType& { 
            self *= other; 
            return self; 
        })
        .def("__itruediv__", [](TensorType& self, const TensorType& other) -> TensorType& { 
            self /= other; 
            return self; 
        })
        
        // Math functions
        .def("exp", &TensorType::exp, "Element-wise exponential")
        .def("log", &TensorType::log, "Element-wise natural logarithm")
        .def("sqrt", &TensorType::sqrt, "Element-wise square root")
        .def("pow", &TensorType::pow, py::arg("exponent"), "Element-wise power")
        .def("sin", &TensorType::sin, "Element-wise sine")
        .def("cos", &TensorType::cos, "Element-wise cosine")
        .def("tan", &TensorType::tan, "Element-wise tangent")
        .def("abs", &TensorType::abs, "Element-wise absolute value")
        .def("clip", &TensorType::clip, py::arg("min"), py::arg("max"), "Clip values to range")
        
        // Activation functions
        .def("sigmoid", &TensorType::sigmoid, "Sigmoid activation")
        .def("relu", &TensorType::relu, "ReLU activation")
        
        // Statistical operations
        .def("sum", &TensorType::sum, "Sum of all elements")
        .def("mean", &TensorType::mean, "Mean of all elements")
        .def("variance", &TensorType::variance, "Variance of all elements")
        .def("std", &TensorType::std, "Standard deviation")
        .def("min", &TensorType::min, "Minimum element")
        .def("max", &TensorType::max, "Maximum element")
        .def("median", &TensorType::median, "Median element")
        .def("prod", &TensorType::prod, "Product of all elements")
        
        // Reduction operations
        .def("all", &TensorType::all, "Check if all elements are non-zero")
        .def("any", &TensorType::any, "Check if any element is non-zero")
        .def("argmin", &TensorType::argmin, "Index of minimum element")
        .def("argmax", &TensorType::argmax, "Index of maximum element")
        .def("cumsum", &TensorType::cumsum, "Cumulative sum")
        .def("cumprod", &TensorType::cumprod, "Cumulative product")
        
        // Autograd
        .def("backward", 
             [](TensorType& self) {
                 auto err = self.backward(nullptr);
                 if (err.has_value()) {
                     throw std::runtime_error("backward() failed");
                 }
             },
             "Compute gradients via backpropagation")
        .def("grad", [](TensorType& self) -> TensorType* {
            return self.grad();
        }, py::return_value_policy::reference_internal, "Get gradient tensor")
        .def("zero_grad", &TensorType::zero_grad, "Zero out gradients")
        .def("detach", &TensorType::detach, "Detach from computation graph")
        
        // Conversion
        .def("tolist", [](const TensorType& t) {
            return tensor_to_list(t);
        }, "Convert to Python list")
        .def("__repr__", [](const TensorType& t) {
            return ::to_string(t);
        });
        
    return cls;
}

// Bind Matrix-specific operations
template<typename T>
void bind_matrix_ops(py::module& m, py::class_<Tensor<T, 2>>& cls) {
    using MatrixType = Tensor<T, 2>;
    
    cls.def("matmul", 
            [](const MatrixType& a, const MatrixType& b) -> MatrixType {
                return linalg::matmul(a, b);
            }, 
            py::arg("other"), "Matrix multiplication")
       .def("transpose", 
            [](const MatrixType& a) {
                return a.transpose();
            }, "Matrix transpose")
       .def("__matmul__", 
            [](const MatrixType& a, const MatrixType& b) -> MatrixType {
                return linalg::matmul(a, b);
            }, 
            py::arg("other"), "Matrix multiplication (@)")
       .def("rank", [](const MatrixType& a, T tol) {
            return linalg::rank(a, tol);
       }, py::arg("tol") = T(-1), "Matrix rank")
       .def("trace", [](const MatrixType& a) {
            return linalg::trace(a);
       }, "Matrix trace")
       .def("frobenius_norm", [](const MatrixType& a) {
            return linalg::frobenius_norm(a);
       }, "Frobenius norm")
       .def("condition_number", [](const MatrixType& a) {
            return linalg::condition_number(a);
       }, "Condition number");
}

// Bind Vector-specific operations
template<typename T>
void bind_vector_ops(py::module& m, py::class_<Tensor<T, 1>>& cls) {
    using VectorType = Tensor<T, 1>;
    
    cls.def("dot", 
            [](const VectorType& a, const VectorType& b) {
                return linalg::dot(a, b);
            }, 
            py::arg("other"), "Dot product")
       .def("norm", [](const VectorType& v) {
            return linalg::norm(v);
       }, "Euclidean norm")
       .def("normalize", [](const VectorType& v) {
            return linalg::normalize(v);
       }, "Normalize vector");
}

PYBIND11_MODULE(tensor4d, m) {
    m.doc() = "Python bindings for Tensor4D library";
    
    // TensorError enum
    py::enum_<TensorError>(m, "TensorError")
        .value("DimensionMismatch", TensorError::DimensionMismatch)
        .value("ContractionMismatch", TensorError::ContractionMismatch)
        .value("InvalidArgument", TensorError::InvalidArgument)
        .export_values();
    
    // Float tensors with specialized operations
    {
        auto cls = bind_tensor<float, 1>(m, "Vectorf");
        bind_vector_ops<float>(m, cls);
    }
    
    {
        auto cls = bind_tensor<float, 2>(m, "Matrixf");
        bind_matrix_ops<float>(m, cls);
    }
    
    bind_tensor<float, 3>(m, "Tensor3f");
    bind_tensor<float, 4>(m, "Tensor4f");
    
    // Double tensors with specialized operations
    {
        auto cls = bind_tensor<double, 1>(m, "Vectord");
        bind_vector_ops<double>(m, cls);
    }
    
    {
        auto cls = bind_tensor<double, 2>(m, "Matrixd");
        bind_matrix_ops<double>(m, cls);
    }
    
    bind_tensor<double, 3>(m, "Tensor3d");
    bind_tensor<double, 4>(m, "Tensor4d");
    
    // Loss functions
    m.def("mse_loss", 
          [](const Tensor<float, 2>& pred, const Tensor<float, 2>& target, const std::string& reduction) -> Tensor<float, 2> {
              return loss::mse_loss(pred, target, reduction);
          }, 
          py::arg("predictions"), py::arg("targets"), py::arg("reduction") = "mean",
          "Mean Squared Error loss");
    
    m.def("cross_entropy_loss", 
          [](const Tensor<float, 2>& pred, const Tensor<float, 2>& target, const std::string& reduction) -> Tensor<float, 1> {
              return loss::cross_entropy_loss(pred, target, reduction);
          },
          py::arg("predictions"), py::arg("targets"), py::arg("reduction") = "mean",
          "Cross Entropy loss");
    
    m.def("binary_cross_entropy", 
          [](const Tensor<float, 2>& pred, const Tensor<float, 2>& target, const std::string& reduction) -> Tensor<float, 1> {
              return loss::binary_cross_entropy(pred, target, reduction);
          },
          py::arg("predictions"), py::arg("targets"), py::arg("reduction") = "mean",
          "Binary Cross Entropy loss");
    
    m.def("binary_cross_entropy_loss", 
          [](const Tensor<float, 2>& pred, const Tensor<float, 2>& target, const std::string& reduction) -> Tensor<float, 1> {
              return loss::binary_cross_entropy(pred, target, reduction);
          },
          py::arg("predictions"), py::arg("targets"), py::arg("reduction") = "mean",
          "Binary Cross Entropy loss (alias)");
    
    // File I/O  
    py::enum_<TensorFileFormat>(m, "TensorIOFormat")
        .value("BINARY", TensorFileFormat::BINARY)
        .value("TEXT", TensorFileFormat::TEXT)
        .value("NPY", TensorFileFormat::NPY)
        .export_values();
    
    m.def("save_tensor", 
          [](const Tensor<float, 2>& tensor, const std::string& filename, TensorFileFormat format) {
              bool success = save(tensor, filename, format);
              if (!success) {
                  throw std::runtime_error("Failed to save tensor");
              }
          },
          py::arg("tensor"), py::arg("filename"), py::arg("format") = TensorFileFormat::BINARY,
          "Save tensor to file");
    
    m.def("load_tensor_f2", [](const std::string& filename) -> Tensor<float, 2> {
        auto result = load<float, 2>(filename);
        if (std::holds_alternative<TensorIOError>(result)) {
            throw std::runtime_error("Failed to load tensor from file");
        }
        return std::get<Tensor<float, 2>>(result);
    }, py::arg("filename"), "Load float matrix from file");
    
    m.def("save_binary", 
          [](const Tensor<float, 2>& tensor, const std::string& filename) {
              return save_binary(tensor, filename);
          },
          py::arg("tensor"), py::arg("filename"),
          "Save tensor to file in binary format");
    
    m.def("load_binary_f2", [](const std::string& filename) {
        auto result = load_binary<float, 2>(filename);
        if (std::holds_alternative<TensorIOError>(result)) {
            throw std::runtime_error("Failed to load tensor from file");
        }
        return std::get<Tensor<float, 2>>(result);
    }, py::arg("filename"), "Load float matrix from binary file");
    
    // Optimizer bindings
    
    // Base Optimizer class (abstract)
    py::class_<Optimizer<float>>(m, "Optimizer")
        .def("get_lr", &Optimizer<float>::get_lr, "Get current learning rate")
        .def("set_lr", &Optimizer<float>::set_lr, py::arg("lr"), "Set learning rate");
    
    // SGD optimizer for float matrices
    py::class_<SGD<float, 2>, Optimizer<float>>(m, "SGD")
        .def(py::init<std::vector<Tensor<float, 2>*>, float, float, float>(),
            py::arg("parameters"),
            py::arg("learning_rate") = 0.01f,
            py::arg("momentum") = 0.0f,
            py::arg("weight_decay") = 0.0f,
            "Create SGD optimizer with parameters")
        .def("step", &SGD<float, 2>::step, "Perform optimization step")
        .def("zero_grad", &SGD<float, 2>::zero_grad, "Zero out gradients");
    
    // Adam optimizer for float matrices
    py::class_<Adam<float, 2>, Optimizer<float>>(m, "Adam")
        .def(py::init<std::vector<Tensor<float, 2>*>, float, float, float, float, float>(),
            py::arg("parameters"),
            py::arg("learning_rate") = 0.001f,
            py::arg("beta1") = 0.9f,
            py::arg("beta2") = 0.999f,
            py::arg("epsilon") = 1e-8f,
            py::arg("weight_decay") = 0.0f,
            "Create Adam optimizer with parameters")
        .def("step", &Adam<float, 2>::step, "Perform optimization step")
        .def("zero_grad", &Adam<float, 2>::zero_grad, "Zero out gradients")
        .def("reset", &Adam<float, 2>::reset, "Reset optimizer state");
    
    // RMSprop optimizer for float matrices
    py::class_<RMSprop<float, 2>, Optimizer<float>>(m, "RMSprop")
        .def(py::init<std::vector<Tensor<float, 2>*>, float, float, float, float, float>(),
            py::arg("parameters"),
            py::arg("learning_rate") = 0.01f,
            py::arg("alpha") = 0.99f,
            py::arg("epsilon") = 1e-8f,
            py::arg("weight_decay") = 0.0f,
            py::arg("momentum") = 0.0f,
            "Create RMSprop optimizer with parameters")
        .def("step", &RMSprop<float, 2>::step, "Perform optimization step")
        .def("zero_grad", &RMSprop<float, 2>::zero_grad, "Zero out gradients");
    
    // Learning rate scheduler
    py::class_<ExponentialLR<float>>(m, "ExponentialLR")
        .def(py::init<Optimizer<float>*, float>(),
            py::arg("optimizer"),
            py::arg("gamma"),
            "Create exponential learning rate scheduler")
        .def("step", &ExponentialLR<float>::step, "Decay learning rate")
        .def("reset", &ExponentialLR<float>::reset, "Reset to initial learning rate");
    
    // Linear algebra functions module
    py::module linalg_mod = m.def_submodule("linalg", "Linear algebra functions");
    
    // SVD decomposition
    linalg_mod.def("svd", [](const Tensor<float, 2>& A) {
        auto result = linalg::svd(A);
        if (std::holds_alternative<linalg::LinalgError>(result)) {
            throw std::runtime_error("SVD decomposition failed");
        }
        auto& svd_result = std::get<linalg::SVDResult<float>>(result);
        return py::make_tuple(svd_result.U, svd_result.S, svd_result.Vt);
    }, py::arg("A"), "Singular Value Decomposition");
    
    linalg_mod.def("svd", [](const Tensor<double, 2>& A) {
        auto result = linalg::svd(A);
        if (std::holds_alternative<linalg::LinalgError>(result)) {
            throw std::runtime_error("SVD decomposition failed");
        }
        auto& svd_result = std::get<linalg::SVDResult<double>>(result);
        return py::make_tuple(svd_result.U, svd_result.S, svd_result.Vt);
    }, py::arg("A"), "Singular Value Decomposition (double)");
    
    // QR decomposition
    linalg_mod.def("qr", [](const Tensor<float, 2>& A) {
        auto result = linalg::qr(A);
        if (std::holds_alternative<linalg::LinalgError>(result)) {
            throw std::runtime_error("QR decomposition failed");
        }
        auto& qr_result = std::get<linalg::QRResult<float>>(result);
        return py::make_tuple(qr_result.Q, qr_result.R);
    }, py::arg("A"), "QR decomposition");
    
    linalg_mod.def("qr", [](const Tensor<double, 2>& A) {
        auto result = linalg::qr(A);
        if (std::holds_alternative<linalg::LinalgError>(result)) {
            throw std::runtime_error("QR decomposition failed");
        }
        auto& qr_result = std::get<linalg::QRResult<double>>(result);
        return py::make_tuple(qr_result.Q, qr_result.R);
    }, py::arg("A"), "QR decomposition (double)");
    
    // Cholesky decomposition
    linalg_mod.def("cholesky", [](const Tensor<float, 2>& A) {
        auto result = linalg::cholesky(A);
        if (std::holds_alternative<linalg::LinalgError>(result)) {
            throw std::runtime_error("Cholesky decomposition failed");
        }
        return std::get<Tensor<float, 2>>(result);
    }, py::arg("A"), "Cholesky decomposition");
    
    linalg_mod.def("cholesky", [](const Tensor<double, 2>& A) {
        auto result = linalg::cholesky(A);
        if (std::holds_alternative<linalg::LinalgError>(result)) {
            throw std::runtime_error("Cholesky decomposition failed");
        }
        return std::get<Tensor<double, 2>>(result);
    }, py::arg("A"), "Cholesky decomposition (double)");
    
    // LU decomposition
    linalg_mod.def("lu", [](const Tensor<float, 2>& A) {
        auto result = linalg::lu(A);
        if (std::holds_alternative<linalg::LinalgError>(result)) {
            throw std::runtime_error("LU decomposition failed");
        }
        auto& lu_result = std::get<linalg::LUResult<float>>(result);
        return py::make_tuple(lu_result.L, lu_result.U, lu_result.P);
    }, py::arg("A"), "LU decomposition with pivoting");
    
    linalg_mod.def("lu", [](const Tensor<double, 2>& A) {
        auto result = linalg::lu(A);
        if (std::holds_alternative<linalg::LinalgError>(result)) {
            throw std::runtime_error("LU decomposition failed");
        }
        auto& lu_result = std::get<linalg::LUResult<double>>(result);
        return py::make_tuple(lu_result.L, lu_result.U, lu_result.P);
    }, py::arg("A"), "LU decomposition with pivoting (double)");
    
    // Eigenvalue decomposition
    linalg_mod.def("eig", [](const Tensor<float, 2>& A) {
        auto result = linalg::eig(A);
        if (std::holds_alternative<linalg::LinalgError>(result)) {
            throw std::runtime_error("Eigenvalue decomposition failed");
        }
        auto& eig_result = std::get<linalg::EigenResult<float>>(result);
        return py::make_tuple(eig_result.eigenvalues, eig_result.eigenvectors);
    }, py::arg("A"), "Eigenvalue decomposition");
    
    linalg_mod.def("eig", [](const Tensor<double, 2>& A) {
        auto result = linalg::eig(A);
        if (std::holds_alternative<linalg::LinalgError>(result)) {
            throw std::runtime_error("Eigenvalue decomposition failed");
        }
        auto& eig_result = std::get<linalg::EigenResult<double>>(result);
        return py::make_tuple(eig_result.eigenvalues, eig_result.eigenvectors);
    }, py::arg("A"), "Eigenvalue decomposition (double)");
    
    // Linear solvers
    linalg_mod.def("solve", [](const Tensor<float, 2>& A, const Tensor<float, 2>& b) {
        auto result = linalg::solve(A, b);
        if (std::holds_alternative<linalg::LinalgError>(result)) {
            throw std::runtime_error("Linear system solving failed");
        }
        return std::get<Tensor<float, 2>>(result);
    }, py::arg("A"), py::arg("b"), "Solve linear system Ax = b");
    
    linalg_mod.def("solve", [](const Tensor<double, 2>& A, const Tensor<double, 2>& b) {
        auto result = linalg::solve(A, b);
        if (std::holds_alternative<linalg::LinalgError>(result)) {
            throw std::runtime_error("Linear system solving failed");
        }
        return std::get<Tensor<double, 2>>(result);
    }, py::arg("A"), py::arg("b"), "Solve linear system Ax = b (double)");
    
    // Least squares
    linalg_mod.def("lstsq", [](const Tensor<float, 2>& A, const Tensor<float, 2>& b) {
        auto result = linalg::lstsq(A, b);
        if (std::holds_alternative<linalg::LinalgError>(result)) {
            throw std::runtime_error("Least squares solving failed");
        }
        return std::get<Tensor<float, 2>>(result);
    }, py::arg("A"), py::arg("b"), "Least squares solution");
    
    linalg_mod.def("lstsq", [](const Tensor<double, 2>& A, const Tensor<double, 2>& b) {
        auto result = linalg::lstsq(A, b);
        if (std::holds_alternative<linalg::LinalgError>(result)) {
            throw std::runtime_error("Least squares solving failed");
        }
        return std::get<Tensor<double, 2>>(result);
    }, py::arg("A"), py::arg("b"), "Least squares solution (double)");
    
    // Kronecker product
    linalg_mod.def("kron", [](const Tensor<float, 2>& A, const Tensor<float, 2>& B) {
        return linalg::kron(A, B);
    }, py::arg("A"), py::arg("B"), "Kronecker product");
    
    linalg_mod.def("kron", [](const Tensor<double, 2>& A, const Tensor<double, 2>& B) {
        return linalg::kron(A, B);
    }, py::arg("A"), py::arg("B"), "Kronecker product (double)");
    
    // Matrix functions
    linalg_mod.def("det", [](const Tensor<float, 2>& A) {
        return linalg::det(A);
    }, py::arg("A"), "Determinant");
    
    linalg_mod.def("det", [](const Tensor<double, 2>& A) {
        return linalg::det(A);
    }, py::arg("A"), "Determinant (double)");
    
    linalg_mod.def("inv", [](const Tensor<float, 2>& A) {
        auto result = linalg::inverse(A);
        if (std::holds_alternative<linalg::LinalgError>(result)) {
            throw std::runtime_error("Matrix inversion failed");
        }
        return std::get<Tensor<float, 2>>(result);
    }, py::arg("A"), "Matrix inverse");
    
    linalg_mod.def("inv", [](const Tensor<double, 2>& A) {
        auto result = linalg::inverse(A);
        if (std::holds_alternative<linalg::LinalgError>(result)) {
            throw std::runtime_error("Matrix inversion failed");
        }
        return std::get<Tensor<double, 2>>(result);
    }, py::arg("A"), "Matrix inverse (double)");
    
    linalg_mod.def("pinv", [](const Tensor<float, 2>& A, float tol) {
        auto result = linalg::pinv(A, tol);
        if (std::holds_alternative<linalg::LinalgError>(result)) {
            throw std::runtime_error("Pseudo-inverse computation failed");
        }
        return std::get<Tensor<float, 2>>(result);
    }, py::arg("A"), py::arg("tol") = 1e-10f, "Pseudo-inverse");
    
    linalg_mod.def("pinv", [](const Tensor<double, 2>& A, double tol) {
        auto result = linalg::pinv(A, tol);
        if (std::holds_alternative<linalg::LinalgError>(result)) {
            throw std::runtime_error("Pseudo-inverse computation failed");
        }
        return std::get<Tensor<double, 2>>(result);
    }, py::arg("A"), py::arg("tol") = 1e-10, "Pseudo-inverse (double)");
    
    linalg_mod.def("matrix_rank", [](const Tensor<float, 2>& A, float tol) {
        return linalg::rank(A, tol);
    }, py::arg("A"), py::arg("tol") = 1e-10f, "Matrix rank");
    
    linalg_mod.def("matrix_rank", [](const Tensor<double, 2>& A, double tol) {
        return linalg::rank(A, tol);
    }, py::arg("A"), py::arg("tol") = 1e-10, "Matrix rank (double)");
    
    linalg_mod.def("norm", [](const Tensor<float, 1>& v) {
        return linalg::norm(v);
    }, py::arg("v"), "Vector norm");
    
    linalg_mod.def("norm", [](const Tensor<double, 1>& v) {
        return linalg::norm(v);
    }, py::arg("v"), "Vector norm (double)");
    
    linalg_mod.def("dot", [](const Tensor<float, 1>& a, const Tensor<float, 1>& b) {
        return linalg::dot(a, b);
    }, py::arg("a"), py::arg("b"), "Dot product");
    
    linalg_mod.def("dot", [](const Tensor<double, 1>& a, const Tensor<double, 1>& b) {
        return linalg::dot(a, b);
    }, py::arg("a"), py::arg("b"), "Dot product (double)");
    
    linalg_mod.def("cross", [](const Tensor<float, 1>& a, const Tensor<float, 1>& b) {
        auto result = linalg::cross(a, b);
        if (std::holds_alternative<linalg::LinalgError>(result)) {
            throw std::runtime_error("Cross product failed (vectors must be 3D)");
        }
        return std::get<Tensor<float, 1>>(result);
    }, py::arg("a"), py::arg("b"), "Cross product (3D)");
    
    linalg_mod.def("cross", [](const Tensor<double, 1>& a, const Tensor<double, 1>& b) {
        auto result = linalg::cross(a, b);
        if (std::holds_alternative<linalg::LinalgError>(result)) {
            throw std::runtime_error("Cross product failed (vectors must be 3D)");
        }
        return std::get<Tensor<double, 1>>(result);
    }, py::arg("a"), py::arg("b"), "Cross product (3D, double)");
    
    linalg_mod.def("matmul", [](const Tensor<float, 2>& A, const Tensor<float, 2>& B) {
        return linalg::matmul(A, B);
    }, py::arg("A"), py::arg("B"), "Matrix multiplication");
    
    linalg_mod.def("matmul", [](const Tensor<double, 2>& A, const Tensor<double, 2>& B) {
        return linalg::matmul(A, B);
    }, py::arg("A"), py::arg("B"), "Matrix multiplication (double)");
    
    // Constants and enums
    
    // Version info
    m.attr("__version__") = "1.4.2";
}
