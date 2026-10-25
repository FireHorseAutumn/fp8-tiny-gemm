cmake_minimum_required(VERSION 3.20)

# Select a supported CUDA host compiler before project() enables CUDA.
# With GCC 16 or newer, prefer g++-15; otherwise enable the compiler override
# and use C++20 for CUDA, since newer libstdc++ headers require it.
find_program(_cxx_compiler NAMES $ENV{CXX} c++ g++)
if(_cxx_compiler)
    execute_process(COMMAND ${_cxx_compiler} -dumpversion
                    OUTPUT_VARIABLE _gcc_version OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
endif()
if(_gcc_version VERSION_GREATER_EQUAL 16)
    find_program(_gcc15 NAMES g++-15)
    if(_gcc15)
        set(CMAKE_CUDA_HOST_COMPILER "${_gcc15}")
    else()
        set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler -std=c++20")
    endif()
endif()

project(fp8_tiny_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_STANDARD 17)
else()
    set(CMAKE_CUDA_STANDARD 20)
endif()
set(CMAKE_CUDA_ARCHITECTURES native)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

enable_testing()

add_library(ftg INTERFACE)
target_include_directories(ftg INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data)

add_executable(test_correctness tests/test_correctness.cu)
target_link_libraries(test_correctness PRIVATE ftg CUDA::cublasLt)
add_test(NAME correctness COMMAND test_correctness)

add_executable(test_fp4_accuracy tests/test_fp4_accuracy.cu)
target_link_libraries(test_fp4_accuracy PRIVATE ftg)
add_test(NAME fp4_accuracy COMMAND test_fp4_accuracy)

add_executable(bench_gemm benchmarks/bench_gemm.cu)
target_link_libraries(bench_gemm PRIVATE ftg CUDA::cublasLt)
