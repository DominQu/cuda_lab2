#include <iostream>
#include <vector>
#include <random>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>
#include "sorter.cuh"
#include "integrator.cuh"


const int POINT_RANGE = 100000;
const int NUM_POINTS = 1 << 27; //the program works for NUM_POINTS being a power of two


thrust::host_vector<float> pointgenerator(int num_points) {

    // setup random generator
    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_real_distribution<> dist(0, 1);

    // pushback random points to vector
    thrust::host_vector<float> vec;
    for (int i = 0; i < num_points; i++) {
        float x = dist(eng) * POINT_RANGE + dist(eng);
        // float y = dist(eng)*POINT_RANGE + dist(eng);
        vec.push_back(x);
        // std::cout << x << "  " << y << std::endl;
    }

    // std::cout << "Generation completed" << std::endl;

    return vec;
}

int main() {


    std::cout.precision(15);
    thrust::host_vector<float> vecx = pointgenerator(NUM_POINTS);
    thrust::host_vector<float> vecy = pointgenerator(NUM_POINTS);

    Sorter sorter;
    Integrator integrator;

    //variables for the result of integration
    float integralGPU;
    float integralCPU;


    thrust::device_vector<float> dvecx = vecx;
    thrust::device_vector<float> dvecy = vecy;

    sorter.GPUsort(dvecx, dvecy);
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    integralGPU = integrator.GPUintegrator(dvecx, dvecy);
    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();
    auto durationmili = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    auto durationmicro = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "GPU integration time: " << durationmili.count() << "." << durationmicro.count() << " milliseconds" << std::endl;
    std::cout << "integralGPU = " << integralGPU << std::endl;

    // measure GPU time


    sorter.CPUsort(vecx, vecy);

    auto start1 = std::chrono::high_resolution_clock::now();
    integralCPU = integrator.CPUintegrator(vecx, vecy);


    auto stop1 = std::chrono::high_resolution_clock::now();
    auto durationmili1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1);
    auto durationmicro1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
    std::cout << "CPU integration time: " << durationmili1.count() << "." << durationmicro1.count() << " milliseconds" << std::endl;
    std::cout << "integralCPU = " << integralCPU << std::endl;

}
