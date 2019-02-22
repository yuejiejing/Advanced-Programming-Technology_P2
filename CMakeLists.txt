cmake_minimum_required(VERSION 2.8)
project(P2)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGES "${CMAKE_CXX_FLAGES}" -std=c++11)

set(CUDA_NVCC_FLAGES ${CUDA_NVCC_FLAGES}; -03 -gencode arch=compute_22, code=sm_22)

include_directories(.)
include_directories(build)
include_directories(src)

set(SOURCE_FILES src/helper_cuda.h src/helper_string.h)

cuda_add_executable(heat2D3D Heat.cu src/helper_cuda.h src/helper_string.h)
