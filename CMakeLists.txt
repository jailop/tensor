cmake_minimum_required(VERSION 3.14)
project(tensor4d LANGUAGES CXX)

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
  message(STATUS "CUDA found - GPU acceleration enabled (Architecture: ${CMAKE_CUDA_ARCHITECTURES})")
else()
  set(USE_CUDA OFF)
  message(STATUS "CUDA not found - building CPU-only version")
endif()

# Find Threading library for parallel execution
find_package(Threads REQUIRED)

# Try to find TBB for parallel execution policy support
find_package(TBB QUIET)

# Try to find BLAS for optimized CPU operations
find_package(BLAS QUIET)
if(BLAS_FOUND)
  add_definitions(-DUSE_BLAS)
  message(STATUS "BLAS found - optimized CPU matrix operations enabled")
else()
  message(STATUS "BLAS not found - using standard CPU implementation")
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

include_directories(include)

# GPU source files (if CUDA is available)
if(USE_CUDA)
  set(GPU_SOURCES src/tensor_gpu.cu)
  set_source_files_properties(${GPU_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()

# Collect all test files from tests directory
file(GLOB TEST_SOURCES "tests/*.cc")

# Test executables
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



# Link TBB if found (required for GCC/Clang parallel execution)
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

include(GoogleTest)
gtest_discover_tests(tensor_test)

# Performance benchmark executable
add_executable(
  tensor_perf
  perf/tensor_perf.cc
  ${GPU_SOURCES}
)

target_link_libraries(
  tensor_perf
  Threads::Threads
)

# Link TBB to performance benchmark if found
if(TBB_FOUND)
  target_link_libraries(tensor_perf TBB::tbb)
endif()

# Link BLAS to performance benchmark if found
if(BLAS_FOUND)
  target_link_libraries(tensor_perf ${BLAS_LIBRARIES})
endif()

# GPU/BLAS demo executable
add_executable(
  gpu_blas_demo
  gpu_blas_demo.cc
  ${GPU_SOURCES}
)

target_link_libraries(
  gpu_blas_demo
  Threads::Threads
)

# Link TBB to demo if found
if(TBB_FOUND)
  target_link_libraries(gpu_blas_demo TBB::tbb)
endif()

# Link BLAS to demo if found
if(BLAS_FOUND)
  target_link_libraries(gpu_blas_demo ${BLAS_LIBRARIES})
endif()

# Backend demo executable
add_executable(
  backend_demo
  backend_demo.cc
  ${GPU_SOURCES}
)

target_link_libraries(
  backend_demo
  Threads::Threads
)

# Link TBB to backend demo if found
if(TBB_FOUND)
  target_link_libraries(backend_demo TBB::tbb)
endif()

# Link BLAS to backend demo if found
if(BLAS_FOUND)
  target_link_libraries(backend_demo ${BLAS_LIBRARIES})
endif()

# Tensor I/O demo executable
add_executable(
  tensor_io_demo
  tensor_io_demo.cc
  ${GPU_SOURCES}
)

target_link_libraries(
  tensor_io_demo
  Threads::Threads
)

# Link TBB to tensor_io_demo if found
if(TBB_FOUND)
  target_link_libraries(tensor_io_demo TBB::tbb)
endif()

# Link BLAS to tensor_io_demo if found
if(BLAS_FOUND)
  target_link_libraries(tensor_io_demo ${BLAS_LIBRARIES})
endif()

# Add custom target to run performance tests
add_custom_target(run_perf
  COMMAND tensor_perf
  DEPENDS tensor_perf
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Running performance benchmarks..."
)

# ============================================
# Doxygen Documentation
# ============================================
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
