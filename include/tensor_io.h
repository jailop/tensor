/**
 * @file tensor_io.h
 * @brief I/O operations for tensors (save/load/print).
 */

#ifndef TENSOR_IO_H
#define TENSOR_IO_H

#include "tensor.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstring>

/**
 * @brief Tensor file format types.
 */
enum class TensorFileFormat {
    BINARY,     ///< Binary format (raw data with header)
    TEXT,       ///< Human-readable text format
    NPY         ///< NumPy .npy format (compatible with numpy.save/numpy.load)
};

/**
 * @brief Error types for I/O operations.
 */
enum class TensorIOError {
    FILE_OPEN_FAILED,
    FILE_READ_FAILED,
    FILE_WRITE_FAILED,
    INVALID_FORMAT,
    DIMENSION_MISMATCH,
    TYPE_MISMATCH
};

template <typename T, size_t N>
using TensorIOResult = std::variant<Tensor<T, N>, TensorIOError>;

/**
 * @brief Save a tensor to a file in binary format.
 * 
 * Binary format structure:
 * - Magic number (4 bytes): "TNSR"
 * - Version (4 bytes): format version
 * - Type size (4 bytes): sizeof(T)
 * - Rank (4 bytes): N (number of dimensions)
 * - Dimensions (N * 8 bytes): size_t for each dimension
 * - Data (total_size * sizeof(T) bytes): raw tensor data
 * 
 * @param tensor The tensor to save.
 * @param filename The output file path.
 * @return true on success, false on failure.
 */
template <typename T, size_t N>
bool save_binary(const Tensor<T, N>& tensor, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Write magic number
    const char magic[4] = {'T', 'N', 'S', 'R'};
    file.write(magic, 4);
    
    // Write version
    uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    // Write type size
    uint32_t type_size = sizeof(T);
    file.write(reinterpret_cast<const char*>(&type_size), sizeof(type_size));
    
    // Write rank
    uint32_t rank = N;
    file.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
    
    // Write dimensions
    auto dims = tensor.dims();
    for (size_t i = 0; i < N; ++i) {
        uint64_t dim = dims[i];
        file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    }
    
    // Write data
    size_t total = tensor.total_size();
    const T* data = tensor.data();
    file.write(reinterpret_cast<const char*>(data), total * sizeof(T));
    
    file.close();
    return true;
}

/**
 * @brief Load a tensor from a binary file.
 * 
 * @param filename The input file path.
 * @return TensorIOResult containing the loaded tensor or an error.
 */
template <typename T, size_t N>
TensorIOResult<T, N> load_binary(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return TensorIOError::FILE_OPEN_FAILED;
    }
    
    // Read and verify magic number
    char magic[4];
    file.read(magic, 4);
    if (magic[0] != 'T' || magic[1] != 'N' || magic[2] != 'S' || magic[3] != 'R') {
        return TensorIOError::INVALID_FORMAT;
    }
    
    // Read version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    
    // Read and verify type size
    uint32_t type_size;
    file.read(reinterpret_cast<char*>(&type_size), sizeof(type_size));
    if (type_size != sizeof(T)) {
        return TensorIOError::TYPE_MISMATCH;
    }
    
    // Read and verify rank
    uint32_t rank;
    file.read(reinterpret_cast<char*>(&rank), sizeof(rank));
    if (rank != N) {
        return TensorIOError::DIMENSION_MISMATCH;
    }
    
    // Read dimensions
    std::array<size_t, N> dims;
    for (size_t i = 0; i < N; ++i) {
        uint64_t dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        dims[i] = static_cast<size_t>(dim);
    }
    
    // Create tensor
    Tensor<T, N> tensor(dims);
    
    // Read data
    size_t total = tensor.total_size();
    file.read(reinterpret_cast<char*>(tensor.data()), total * sizeof(T));
    
    if (!file) {
        return TensorIOError::FILE_READ_FAILED;
    }
    
    file.close();
    return tensor;
}

/**
 * @brief Save a tensor to a text file (CSV-like format).
 * 
 * For 1D tensors: values separated by spaces
 * For 2D tensors: rows separated by newlines, values by spaces
 * For higher dimensions: recursive structure with blank lines
 * 
 * @param tensor The tensor to save.
 * @param filename The output file path.
 * @param precision Number of decimal places (default: 6).
 * @return true on success, false on failure.
 */
template <typename T, size_t N>
bool save_text(const Tensor<T, N>& tensor, const std::string& filename, int precision) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    file << std::fixed << std::setprecision(precision);
    
    // Write dimensions as comment
    file << "# Shape: ";
    auto dims = tensor.dims();
    for (size_t i = 0; i < N; ++i) {
        if (i > 0) file << "x";
        file << dims[i];
    }
    file << "\n";
    
    // Write data
    const T* data = tensor.data();
    size_t total = tensor.total_size();
    
    if (N == 1) {
        // 1D: single line
        for (size_t i = 0; i < total; ++i) {
            if (i > 0) file << " ";
            file << data[i];
        }
        file << "\n";
    } else if (N == 2) {
        // 2D: matrix format
        for (size_t i = 0; i < dims[0]; ++i) {
            for (size_t j = 0; j < dims[1]; ++j) {
                if (j > 0) file << " ";
                file << data[i * dims[1] + j];
            }
            file << "\n";
        }
    } else {
        // Higher dimensions: flatten with structure
        for (size_t i = 0; i < total; ++i) {
            file << data[i];
            
            // Add spacing based on dimension boundaries
            bool newline = false;
            size_t stride = 1;
            for (size_t d = N - 1; d > 0; --d) {
                stride *= dims[d];
                if ((i + 1) % stride == 0) {
                    newline = true;
                    break;
                }
            }
            
            if (newline) {
                file << "\n";
                if ((i + 1) < total && (i + 1) % (dims[N - 1] * dims[N - 2]) == 0) {
                    file << "\n";  // Extra newline for higher dimension boundaries
                }
            } else {
                file << " ";
            }
        }
    }
    
    file.close();
    return true;
}

/**
 * @brief Save a tensor to NumPy .npy format (version 1.0).
 * 
 * This creates a file compatible with numpy.load() in Python.
 * 
 * @param tensor The tensor to save.
 * @param filename The output file path.
 * @return true on success, false on failure.
 */
template <typename T, size_t N>
bool save_npy(const Tensor<T, N>& tensor, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // NPY format magic number
    const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    file.write(reinterpret_cast<const char*>(magic), 6);
    
    // Version 1.0
    unsigned char version[2] = {1, 0};
    file.write(reinterpret_cast<const char*>(version), 2);
    
    // Build header dictionary
    std::string dtype_str;
    if (std::is_same<T, float>::value) {
        dtype_str = "<f4";  // little-endian float32
    } else if (std::is_same<T, double>::value) {
        dtype_str = "<f8";  // little-endian float64
    } else if (std::is_same<T, int32_t>::value) {
        dtype_str = "<i4";  // little-endian int32
    } else if (std::is_same<T, int64_t>::value) {
        dtype_str = "<i8";  // little-endian int64
    } else {
        // Default to bytes
        dtype_str = "|u" + std::to_string(sizeof(T));
    }
    
    // Build shape tuple
    std::ostringstream shape_stream;
    shape_stream << "(";
    auto dims = tensor.dims();
    for (size_t i = 0; i < N; ++i) {
        if (i > 0) shape_stream << ", ";
        shape_stream << dims[i];
    }
    if (N == 1) shape_stream << ",";  // Python tuple syntax for single element
    shape_stream << ")";
    
    // Build header string
    std::ostringstream header_stream;
    header_stream << "{'descr': '" << dtype_str << "', "
                  << "'fortran_order': False, "
                  << "'shape': " << shape_stream.str() << ", }";
    
    std::string header = header_stream.str();
    
    // Pad header to be divisible by 16 (NPY format requirement)
    // Total header size including length bytes must be divisible by 16
    size_t header_len = header.size();
    size_t padding = (16 - ((10 + header_len) % 16)) % 16;
    header.append(padding, ' ');
    header.push_back('\n');
    
    // Write header length (2 bytes, little-endian)
    uint16_t header_len_total = static_cast<uint16_t>(header.size());
    file.write(reinterpret_cast<const char*>(&header_len_total), 2);
    
    // Write header
    file.write(header.c_str(), header.size());
    
    // Write data
    size_t total = tensor.total_size();
    const T* data = tensor.data();
    file.write(reinterpret_cast<const char*>(data), total * sizeof(T));
    
    file.close();
    return true;
}

/**
 * @brief Save a tensor to a file with specified format.
 * 
 * @param tensor The tensor to save.
 * @param filename The output file path.
 * @param format The file format (default: BINARY).
 * @param precision For text format, number of decimal places (default: 6).
 * @return true on success, false on failure.
 */
template <typename T, size_t N>
bool save(const Tensor<T, N>& tensor, const std::string& filename, 
          TensorFileFormat format = TensorFileFormat::BINARY, int precision = 6) {
    switch (format) {
        case TensorFileFormat::BINARY:
            return save_binary(tensor, filename);
        case TensorFileFormat::TEXT:
            return save_text(tensor, filename, precision);
        case TensorFileFormat::NPY:
            return save_npy(tensor, filename);
        default:
            return false;
    }
}

/**
 * @brief Load a tensor from a file (auto-detects format).
 * 
 * Attempts to detect file format from magic bytes:
 * - "TNSR" -> Binary format
 * - 0x93, "NUMPY" -> NPY format
 * 
 * @param filename The input file path.
 * @return TensorIOResult containing the loaded tensor or an error.
 */
template <typename T, size_t N>
TensorIOResult<T, N> load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return TensorIOError::FILE_OPEN_FAILED;
    }
    
    // Read first few bytes to detect format
    char magic[6];
    file.read(magic, 6);
    file.close();
    
    // Check for TNSR binary format
    if (magic[0] == 'T' && magic[1] == 'N' && magic[2] == 'S' && magic[3] == 'R') {
        return load_binary<T, N>(filename);
    }
    
    // Check for NPY format
    if (magic[0] == (char)0x93 && magic[1] == 'N' && magic[2] == 'U' && 
        magic[3] == 'M' && magic[4] == 'P' && magic[5] == 'Y') {
        // NPY loading is complex, for now return error
        return TensorIOError::INVALID_FORMAT;
    }
    
    // Unknown format
    return TensorIOError::INVALID_FORMAT;
}

/**
 * @brief Pretty print a tensor with formatting options.
 * 
 * @param tensor The tensor to print.
 * @param os Output stream (default: std::cout).
 * @param precision Number of decimal places (default: 4).
 * @param max_items Maximum items to print per dimension (default: 6, 0 = all).
 * @param line_width Maximum line width before wrapping (default: 75).
 */
template <typename T, size_t N>
void print(const Tensor<T, N>& tensor, std::ostream& os, 
           int precision, size_t max_items, size_t line_width) {
    os << "Tensor<" << typeid(T).name() << ", " << N << ">(";
    
    auto dims = tensor.dims();
    os << "shape=[";
    for (size_t i = 0; i < N; ++i) {
        if (i > 0) os << ", ";
        os << dims[i];
    }
    os << "])\n";
    
    os << std::fixed << std::setprecision(precision);
    
    const T* data = tensor.data();
    size_t total = tensor.total_size();
    
    if (N == 1) {
        // 1D tensor
        os << "[";
        size_t limit = (max_items > 0 && dims[0] > max_items) ? max_items / 2 : dims[0];
        for (size_t i = 0; i < limit; ++i) {
            if (i > 0) os << ", ";
            os << data[i];
        }
        if (max_items > 0 && dims[0] > max_items) {
            os << ", ..., ";
            for (size_t i = dims[0] - max_items / 2; i < dims[0]; ++i) {
                if (i > dims[0] - max_items / 2) os << ", ";
                os << data[i];
            }
        }
        os << "]\n";
    } else if (N == 2) {
        // 2D tensor (matrix)
        os << "[";
        size_t row_limit = (max_items > 0 && dims[0] > max_items) ? max_items / 2 : dims[0];
        size_t col_limit = (max_items > 0 && dims[1] > max_items) ? max_items / 2 : dims[1];
        
        for (size_t i = 0; i < row_limit; ++i) {
            if (i > 0) os << " ";
            os << "[";
            for (size_t j = 0; j < col_limit; ++j) {
                if (j > 0) os << ", ";
                os << std::setw(precision + 3) << data[i * dims[1] + j];
            }
            if (max_items > 0 && dims[1] > max_items) {
                os << ", ..., ";
                for (size_t j = dims[1] - max_items / 2; j < dims[1]; ++j) {
                    if (j > dims[1] - max_items / 2) os << ", ";
                    os << std::setw(precision + 3) << data[i * dims[1] + j];
                }
            }
            os << "]";
            if (i < row_limit - 1 || (max_items > 0 && dims[0] > max_items)) {
                os << "\n";
            }
        }
        
        if (max_items > 0 && dims[0] > max_items) {
            os << " ...\n";
            for (size_t i = dims[0] - max_items / 2; i < dims[0]; ++i) {
                os << " [";
                for (size_t j = 0; j < col_limit; ++j) {
                    if (j > 0) os << ", ";
                    os << std::setw(precision + 3) << data[i * dims[1] + j];
                }
                if (max_items > 0 && dims[1] > max_items) {
                    os << ", ..., ";
                    for (size_t j = dims[1] - max_items / 2; j < dims[1]; ++j) {
                        if (j > dims[1] - max_items / 2) os << ", ";
                        os << std::setw(precision + 3) << data[i * dims[1] + j];
                    }
                }
                os << "]";
                if (i < dims[0] - 1) os << "\n";
            }
        }
        os << "]\n";
    } else {
        // Higher dimensional tensors: show first and last few elements
        os << "[...] (";
        if (max_items > 0 && total > max_items) {
            os << "showing first " << max_items / 2 << " and last " << max_items / 2 << " of " << total << " elements)\n[";
            for (size_t i = 0; i < max_items / 2; ++i) {
                if (i > 0) os << ", ";
                os << data[i];
            }
            os << ", ..., ";
            for (size_t i = total - max_items / 2; i < total; ++i) {
                if (i > total - max_items / 2) os << ", ";
                os << data[i];
            }
            os << "]\n";
        } else {
            os << "all " << total << " elements)\n[";
            for (size_t i = 0; i < total; ++i) {
                if (i > 0) os << ", ";
                os << data[i];
            }
            os << "]\n";
        }
    }
}

/**
 * @brief Convert tensor to string representation.
 * 
 * @param tensor The tensor to convert.
 * @param precision Number of decimal places (default: 4).
 * @param max_items Maximum items to show (default: 6).
 * @return String representation of the tensor.
 */
template <typename T, size_t N>
std::string to_string(const Tensor<T, N>& tensor, int precision = 4, size_t max_items = 6) {
    std::ostringstream oss;
    print(tensor, oss, precision, max_items, 75);
    return oss.str();
}

// Wrapper functions with default arguments to avoid friend declaration conflicts
template <typename T, size_t N>
bool save_text_default(const Tensor<T, N>& tensor, const std::string& filename, int precision = 6) {
    return save_text(tensor, filename, precision);
}

template <typename T, size_t N>
void print_default(const Tensor<T, N>& tensor, std::ostream& os = std::cout, 
                   int precision = 4, size_t max_items = 6, size_t line_width = 75) {
    print(tensor, os, precision, max_items, line_width);
}

#endif // TENSOR_IO_H
