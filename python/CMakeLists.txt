cmake_minimum_required(VERSION 3.15)
project(tensor4d_python)

# Find pybind11
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    # Try to find via Python
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(pybind11_DIR)
        find_package(pybind11 CONFIG PATHS ${pybind11_DIR})
    endif()
endif()

if(NOT pybind11_FOUND)
    message(WARNING "pybind11 not found. Python bindings will not be built.")
    message(STATUS "Install pybind11 with: pip install pybind11")
    return()
endif()

# Include parent directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Find required libraries
find_package(BLAS)
find_package(LAPACK)
find_package(CUDAToolkit QUIET)

# Determine available backends
set(USE_BLAS OFF)
set(USE_GPU OFF)

if(BLAS_FOUND AND LAPACK_FOUND)
    set(USE_BLAS ON)
    message(STATUS "Python bindings: BLAS support enabled")
endif()

if(CUDAToolkit_FOUND)
    set(USE_GPU ON)
    enable_language(CUDA)
    message(STATUS "Python bindings: GPU support enabled")
endif()

# Create Python module
pybind11_add_module(tensor4d 
    tensor_wrapper.cc
    ../src/tensor_instantiations.cc
)

# Link against the main tensor library if it exists
if(TARGET tensor4d_lib)
    target_link_libraries(tensor4d PRIVATE tensor4d_lib)
else()
    # Compile with necessary flags
    if(USE_BLAS)
        target_compile_definitions(tensor4d PRIVATE USE_BLAS)
        target_link_libraries(tensor4d PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    endif()
    
    if(USE_GPU)
        target_compile_definitions(tensor4d PRIVATE USE_GPU)
        target_sources(tensor4d PRIVATE ../src/tensor_gpu.cu)
        set_source_files_properties(../src/tensor_gpu.cu PROPERTIES LANGUAGE CUDA)
        target_link_libraries(tensor4d PRIVATE CUDA::cublas CUDA::cudart)
    endif()
endif()

# Set C++ standard
target_compile_features(tensor4d PRIVATE cxx_std_20)

# Installation
install(TARGETS tensor4d
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)
