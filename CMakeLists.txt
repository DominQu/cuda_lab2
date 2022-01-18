cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(lab2 LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA QUIET REQUIRED)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_50 -rdc=true --expt-relaxed-constexpr -Wno-deprecated-gpu-targets")

add_executable(lab2 main.cu sorter.cu integrator.cu)
    
set_property(TARGET lab2 PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(lab2 PUBLIC ${PROJECT_SOURCE_DIR} /usr/local/cuda-11.4/include/)
