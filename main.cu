#include <iostream>
#include <vector>
#include <random>
#include "point.hpp"
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>
#include "sorter.cuh"
#include "integrator.cuh"


const int POINT_RANGE = 100000;
const int NUM_POINTS = 1000000;


thrust::host_vector<float> pointgenerator(int num_points) {
    
    // setup random generator
    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_real_distribution<> dist(0, 1);
    
    // pushback random points to vector
    thrust::host_vector<float> vec;
    for(int i = 0; i < num_points; i++) {
        float x = dist(eng)*POINT_RANGE + dist(eng);
        // float y = dist(eng)*POINT_RANGE + dist(eng);
        vec.push_back(x);
        // std::cout << x << "  " << y << std::endl;
    }

    // std::cout << "Generation completed" << std::endl;

    return vec;
}

int main(int argc, char *argv[]) {

    // std::vector<int> arguments;
    // for(int arg = 1; arg < argc; arg++) {
    //     arguments.push_back(argv[arg])
    // }

    std::cout.precision(15);
    thrust::host_vector<float> vecx = pointgenerator(NUM_POINTS);
    thrust::host_vector<float> vecy = pointgenerator(NUM_POINTS);
    Sorter sorter;
    Integrator integrator;
    float integralGPU;
    float integralCPU;

    auto start = std::chrono::high_resolution_clock::now();

    thrust::device_vector<float> dvecx = vecx;
    thrust::device_vector<float> dvecy = vecy;

    sorter.GPUsort(dvecx, dvecy);
    cudaDeviceSynchronize();
    integralGPU = integrator.GPUintegrator(dvecx, dvecy);
    cudaDeviceSynchronize();
    std::cout << "integralGPU = " << integralGPU << std::endl;


    auto stop = std::chrono::high_resolution_clock::now();
    auto durationmili = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    auto durationmicro = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "GPU program duration: " << durationmili.count() << "." << durationmicro.count() << " milliseconds" << std::endl;

// measure GPU time

    auto start1 = std::chrono::high_resolution_clock::now();
    
    sorter.CPUsort(vecx, vecy);

    integralCPU = integrator.CPUintegrator(vecx, vecy);
    std::cout << "integralCPU = " << integralCPU << std::endl;


    auto stop1 = std::chrono::high_resolution_clock::now();
    auto durationmili1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1);
    auto durationmicro1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
    std::cout << "CPU program duration: " << durationmili1.count() << "." << durationmicro1.count() << " milliseconds" << std::endl;

}