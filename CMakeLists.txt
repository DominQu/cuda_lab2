cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(lab2 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_50 --expt-relaxed-constexpr -Wno-deprecated-gpu-targets")


add_executable(lab2 main.cu sorter.cu integrator.cu)

target_include_directories(lab2 PUBLIC ${PROJECT_SOURCE_DIR})
