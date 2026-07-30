cmake_minimum_required(VERSION 3.20)
project(o3d_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(Open3D REQUIRED)

add_executable(o3d_test main.cpp)
target_link_libraries(o3d_test PRIVATE Open3D::Open3D)

# A static Open3D built with BUILD_CUDA_MODULE=ON does not propagate the CUDA
# runtime to a plain CXX consumer, so linking fails with undefined references
# to cudaMalloc, cudaEventCreate and friends. Link them explicitly when a CUDA
# toolkit is present. Harmless for a CPU-only Open3D install.
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(o3d_test PRIVATE
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
    )
endif()
