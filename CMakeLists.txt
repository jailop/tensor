cmake_minimum_required(VERSION 3.14)
project(tensor4d LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional CUDA support
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  
  # Set CUDA architecture (86 = Ampere, adjust based on your GPU)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
  endif()
  
  add_definitions(-DUSE_GPU)
  set(USE_CUDA ON)
  message(STATUS "CUDA found: ${CMAKE_CUDA_ARCHITECTURES})")
  
  find_library(CUBLAS_LIBRARY 
               NAMES cublas
               PATHS /opt/cuda/lib64 ${CMAKE_CUDA_TOOLKIT_ROOT_DIR}/lib64
               NO_DEFAULT_PATH)
  
  if(CUBLAS_LIBRARY)
    add_definitions(-DUSE_CUBLAS)
    include_directories(/opt/cuda/targets/x86_64-linux/include)
    message(STATUS "cuBLAS found: ${CUBLAS_LIBRARY}")
  else()
    message(WARNING "cuBLAS not found")
  endif()
else()
  set(USE_CUDA OFF)
  message(STATUS "CUDA not found")
endif()

find_package(Threads REQUIRED)

find_package(TBB QUIET)

find_package(BLAS QUIET)
if(BLAS_FOUND)
  add_definitions(-DUSE_BLAS)
  message(STATUS "BLAS found")
else()
  message(STATUS "BLAS not found")
endif()

find_package(LAPACK QUIET)
if(LAPACK_FOUND)
  add_definitions(-DUSE_LAPACK)
  message(STATUS "LAPACK found")
else()
  message(STATUS "LAPACK not found")
endif()

function(link_math_libraries target_name)
  if(TBB_FOUND)
    target_link_libraries(${target_name} TBB::tbb)
  endif()
  if(BLAS_FOUND)
    target_link_libraries(${target_name} ${BLAS_LIBRARIES})
  endif()
  if(CUBLAS_LIBRARY)
    target_link_libraries(${target_name} ${CUBLAS_LIBRARY})
  endif()
endfunction()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

include_directories(include)

if(USE_CUDA)
  set(GPU_SOURCES src/tensor_gpu.cu)
  set_source_files_properties(${GPU_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()

set(TENSOR_LIB_SOURCES
  src/tensor_instantiations.cc
  src/tensor_error.cc
)

add_library(tensor4d_static STATIC
  ${TENSOR_LIB_SOURCES}
  ${GPU_SOURCES}
)

target_link_libraries(tensor4d_static
  PUBLIC Threads::Threads
)

if(TBB_FOUND)
  target_link_libraries(tensor4d_static PUBLIC TBB::tbb)
endif()

if(BLAS_FOUND)
  target_link_libraries(tensor4d_static PUBLIC ${BLAS_LIBRARIES})
endif()

if(LAPACK_FOUND)
  target_link_libraries(tensor4d_static PUBLIC ${LAPACK_LIBRARIES})
endif()

if(CUBLAS_LIBRARY)
  target_link_libraries(tensor4d_static PUBLIC ${CUBLAS_LIBRARY})
endif()

add_library(tensor4d_shared SHARED
  ${TENSOR_LIB_SOURCES}
  ${GPU_SOURCES}
)

set_target_properties(tensor4d_shared PROPERTIES OUTPUT_NAME tensor4d)

target_link_libraries(tensor4d_shared
  PUBLIC Threads::Threads
)

if(TBB_FOUND)
  target_link_libraries(tensor4d_shared PUBLIC TBB::tbb)
endif()

if(BLAS_FOUND)
  target_link_libraries(tensor4d_shared PUBLIC ${BLAS_LIBRARIES})
endif()

if(LAPACK_FOUND)
  target_link_libraries(tensor4d_shared PUBLIC ${LAPACK_LIBRARIES})
endif()

if(CUBLAS_LIBRARY)
  target_link_libraries(tensor4d_shared PUBLIC ${CUBLAS_LIBRARY})
endif()

set_target_properties(tensor4d_static PROPERTIES
  OUTPUT_NAME tensor4d
  POSITION_INDEPENDENT_CODE ON
)

set_target_properties(tensor4d_shared PROPERTIES
  VERSION 1.0.0
  SOVERSION 1
  POSITION_INDEPENDENT_CODE ON
)

message(STATUS "Building static library: libtensor4d.a")
message(STATUS "Building shared library: libtensor4d.so (version 1.0.0)")

install(TARGETS tensor4d_static tensor4d_shared
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include/tensor
  FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
  PATTERN "*.bak" EXCLUDE
  PATTERN "*.bak2" EXCLUDE
)

message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Use 'make install' to install libraries and headers")

file(GLOB TEST_SOURCES "tests/*.cc")

add_executable(
  tensor_test
  ${TEST_SOURCES}
  ${GPU_SOURCES}
)

target_link_libraries(
  tensor_test
  GTest::gtest_main
  Threads::Threads
)

if(TBB_FOUND)
  target_link_libraries(tensor_test TBB::tbb)
  message(STATUS "TBB found - parallel execution fully enabled")
else()
  message(WARNING "TBB not found - parallel execution may not work on some compilers")
endif()

# Link BLAS if found
if(BLAS_FOUND)
  target_link_libraries(tensor_test ${BLAS_LIBRARIES})
endif()

# Link cuBLAS if found
if(CUBLAS_LIBRARY)
  target_link_libraries(tensor_test ${CUBLAS_LIBRARY})
endif()

include(GoogleTest)
gtest_discover_tests(tensor_test)

# Neural Network Enhancements test
add_executable(
  tensor_nn_enhancements_test
  tests/tensor_nn_enhancements_test.cc
  ${GPU_SOURCES}
)

target_link_libraries(
  tensor_nn_enhancements_test
  GTest::gtest_main
  Threads::Threads
)

# Link TBB if found
if(TBB_FOUND)
  target_link_libraries(tensor_nn_enhancements_test TBB::tbb)
endif()

# Link BLAS if found
if(BLAS_FOUND)
  target_link_libraries(tensor_nn_enhancements_test ${BLAS_LIBRARIES})
endif()

# Link cuBLAS if found
if(CUBLAS_LIBRARY)
  target_link_libraries(tensor_nn_enhancements_test ${CUBLAS_LIBRARY})
endif()

gtest_discover_tests(tensor_nn_enhancements_test)

add_executable(
  tensor_perf
  perf/tensor_perf.cc
  ${GPU_SOURCES}
)

target_link_libraries(
  tensor_perf
  Threads::Threads
)

link_math_libraries(tensor_perf)

add_executable(
  gpu_blas_demo
  examples/gpu_blas_demo.cc
  ${GPU_SOURCES}
)

target_link_libraries(
  gpu_blas_demo
  Threads::Threads
)

link_math_libraries(gpu_blas_demo)

# Backend demo executable
add_executable(
  backend_demo
  examples/backend_demo.cc
  ${GPU_SOURCES}
)

target_link_libraries(
  backend_demo
  Threads::Threads
)

link_math_libraries(backend_demo)

# ============================================
# C Interface Library
# ============================================

# Build C interface shared library
add_library(tensor_c SHARED
  src/tensor_c.cpp
  ${GPU_SOURCES}
)

target_include_directories(tensor_c PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tensor_c
  PRIVATE tensor4d_static
  PUBLIC Threads::Threads
)

# Link TBB if found
if(TBB_FOUND)
  target_link_libraries(tensor_c PUBLIC TBB::tbb)
endif()

# Link BLAS if found
if(BLAS_FOUND)
  target_link_libraries(tensor_c PUBLIC ${BLAS_LIBRARIES})
endif()

# Link LAPACK if found
if(LAPACK_FOUND)
  target_link_libraries(tensor_c PUBLIC ${LAPACK_LIBRARIES})
endif()

set_target_properties(tensor_c PROPERTIES
  VERSION 1.0.0
  SOVERSION 1
  POSITION_INDEPENDENT_CODE ON
)

# Install C library and header
install(TARGETS tensor_c
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(FILES include/tensor_c.h
  DESTINATION include/tensor
)

message(STATUS "Building C interface library: libtensor_c.so")

# C example program
add_executable(c_example examples/c_example.c)
target_link_libraries(c_example tensor_c)

# C MNIST demo program
add_executable(mnist_demo_c examples/mnist_demo_c.c)
target_link_libraries(mnist_demo_c tensor_c)

# C test program
add_executable(tensor_c_test tests/tensor_c_test.c)
target_link_libraries(tensor_c_test tensor_c)

# Add C test to CTest
add_test(NAME TensorCTest COMMAND tensor_c_test)

# Tensor I/O demo executable
add_executable(
  tensor_io_demo
  examples/tensor_io_demo.cc
  ${GPU_SOURCES}
)

target_link_libraries(
  tensor_io_demo
  Threads::Threads
)

link_math_libraries(tensor_io_demo)

# Neural Network Layers demo executable
add_executable(
  nn_layers_demo
  examples/nn_layers_demo.cc
  ${GPU_SOURCES}
)

target_link_libraries(
  nn_layers_demo
  Threads::Threads
)

link_math_libraries(nn_layers_demo)

# MNIST demo executable
add_executable(
  mnist_demo
  examples/mnist_demo.cc
  ${GPU_SOURCES}
)

target_link_libraries(
  mnist_demo
  Threads::Threads
)

link_math_libraries(mnist_demo)

add_custom_target(run_perf
  COMMAND tensor_perf
  DEPENDS tensor_perf
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Running performance benchmarks..."
)

find_package(benchmark QUIET)

if(benchmark_FOUND)
  message(STATUS "Google Benchmark found")
  
  add_executable(
    tensor_normalize_benchmark
    benchs/tensor_normalize_benchmark.cc
    ${GPU_SOURCES}
  )
  
  target_link_libraries(
    tensor_normalize_benchmark
    benchmark::benchmark
    Threads::Threads
  )
  
  link_math_libraries(tensor_normalize_benchmark)
  
  add_custom_target(run_normalize_benchmark
    COMMAND tensor_normalize_benchmark
    DEPENDS tensor_normalize_benchmark
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running tensor normalization benchmarks..."
  )
else()
  message(STATUS "Google Benchmark not found")
endif()

find_package(Doxygen
  OPTIONAL_COMPONENTS dot)

if(DOXYGEN_FOUND)
  # Set input and output files
  set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
  set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
  set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/docs)

  # Configure Doxyfile (replace @VARIABLE@ with actual values)
  configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
  
  # Create output directory
  file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})
  
  # Add a custom target to build documentation
  add_custom_target(doc
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM
  )
  
  # Add a target to build documentation and open it
  add_custom_target(doc_open
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    COMMAND xdg-open ${DOXYGEN_OUTPUT_DIR}/html/index.html 2>/dev/null || open ${DOXYGEN_OUTPUT_DIR}/html/index.html 2>/dev/null || echo "Documentation generated at ${DOXYGEN_OUTPUT_DIR}/html/index.html"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating API documentation and opening in browser"
    VERBATIM
  )
  
  message(STATUS "Doxygen found - documentation generation enabled (use 'make doc' or 'make doc_open')")
  
  if(DOXYGEN_DOT_FOUND)
    message(STATUS "Graphviz dot found - call/caller graphs will be generated")
  else()
    message(STATUS "Graphviz dot not found - graphs will not be generated")
  endif()
else()
  message(STATUS "Doxygen not found - documentation cannot be generated")
endif()
