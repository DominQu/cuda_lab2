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
    thrust::host_vector<float> vecx = pointgenerator(100000);
    thrust::host_vector<float> vecy = pointgenerator(100000);
    Sorter sorter;
    Integrator integrator;
    float integralGPU;
    float integralCPU;



    sorter.CPUsort(vecx, vecy);
    auto start = std::chrono::high_resolution_clock::now();

    // thrust::sort(vecx.begin(), vecx.end());


    integralGPU = integrator.GPUintegrator(vecx, vecy);
    std::cout << "integralGPU = " << integralGPU << std::endl;
    // integralCPU = integrator.CPUintegrator(vecx, vecy);
    // std::cout << "integralCPU = " << integralCPU << std::endl;


    auto stop = std::chrono::high_resolution_clock::now();
    auto durationmili = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    auto durationmicro = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Sort duration: " << durationmili.count() << "." << durationmicro.count() << " milliseconds" << std::endl;


    // std::cout << "vecy: " << std::endl;
    // for(const auto &i:vecy){
    //     std::cout << i << std::endl;
    // }
    // std::cout << "vecx: " << std::endl;

    // for(const auto &i:vecx){
    //     std::cout << i << std::endl;
    // }




    // auto start1 = std::chrono::high_resolution_clock::now();

    // // thrust::sort_by_key(vecx.begin(), vecx.end(), vecy.begin());
    // thrust::device_vector<double> dvecx = vecx;
    // int i;
    // thrust::sort(dvecx.begin(), dvecx.end());

    // auto stop1 = std::chrono::high_resolution_clock::now();
    // auto durationmili1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1);
    // auto durationmicro1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
    // std::cout << "Sort duration: " << durationmili1.count() << "." << durationmicro1.count() << " milliseconds" << std::endl;

}