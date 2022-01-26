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
const int NUM_POINTS = 1<<10; //the program works for NUM_POINTS being a power of two
const int TESTS = 20;

thrust::host_vector<float> pointgenerator(int num_points) {
    
    // setup random generator
    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_real_distribution<> dist(0, 1);
    
    // pushback random points to vector
    thrust::host_vector<float> vec;
    for(int i = 0; i < num_points; i++) {
        float x = dist(eng)*POINT_RANGE + dist(eng);
        vec.push_back(x);
    }
    return vec;
}
void test() {
    for(int i = 1; i < TESTS; i++) {
        int numpoints = 1<<i;
        std::cout.precision(15);

        //generate random points
        thrust::host_vector<float> vecx = pointgenerator(numpoints);
        thrust::host_vector<float> vecy = pointgenerator(numpoints);
        //allocate GPU memory
        thrust::device_vector<float> dvecx = vecx;
        thrust::device_vector<float> dvecy = vecy;


        Sorter sorter;
        Integrator integrator;
        
        //variables for the result of integration
        float integralGPU;
        float integralCPU;

        // //GPU sorting
        // sorter.GPUsort(dvecx, dvecy);
        
        // cudaEvent_t start, stop;
        // float time;
        // cudaEventCreate(&start);
        // cudaEventCreate(&stop);
        // cudaEventRecord(start, 0);

        // //GPU integration
        // integralGPU = integrator.GPUintegrator(dvecx, dvecy);

        // cudaEventRecord(stop, 0);
        // cudaEventSynchronize(stop);
        // cudaEventElapsedTime(&time, start, stop);
        // printf ("GPU integration time: %f ms\n", time);
        
        // cudaDeviceSynchronize();
        // std::cout << "integralGPU = " << integralGPU << std::endl;
        
        //CPU sorting
        sorter.CPUsort(vecx, vecy);

        // auto start1 = std::chrono::high_resolution_clock::now();

        //CPU integration
        // integralCPU = integrator.CPUintegrator(vecx, vecy);

        // auto stop1 = std::chrono::high_resolution_clock::now();
        // auto durationmili1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1);
        // auto durationmicro1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
        // std::cout << "CPU integration time: " << durationmili1.count() << "." << durationmicro1.count() - 1000*durationmili1.count() << " ms" << std::endl;
        // std::cout << "integralCPU = " << integralCPU << std::endl;
    }
}

int main() {

    test();
    // std::cout.precision(15);

    // //generate random points
    // thrust::host_vector<float> vecx = pointgenerator(NUM_POINTS);
    // thrust::host_vector<float> vecy = pointgenerator(NUM_POINTS);

    // Sorter sorter;
    // Integrator integrator;
    
    // //variables for the result of integration
    // float integralGPU;
    // float integralCPU;

    // //allocate GPU memory
    // thrust::device_vector<float> dvecx = vecx;
    // thrust::device_vector<float> dvecy = vecy;

    // //GPU sorting
    // sorter.GPUsort(dvecx, dvecy);
    
    // cudaEvent_t start, stop;
    // float time;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start, 0);

    // //GPU integration
    // integralGPU = integrator.GPUintegrator(dvecx, dvecy);

    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&time, start, stop);
    // printf ("GPU integration time: %f ms\n", time);
    
    // cudaDeviceSynchronize();
    // std::cout << "integralGPU = " << integralGPU << std::endl;
    
    // //CPU sorting
    // sorter.CPUsort(vecx, vecy);

    // auto start1 = std::chrono::high_resolution_clock::now();

    // //CPU integration
    // integralCPU = integrator.CPUintegrator(vecx, vecy);

    // auto stop1 = std::chrono::high_resolution_clock::now();
    // auto durationmili1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1);
    // auto durationmicro1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
    // std::cout << "CPU integration time: " << durationmili1.count() << "." << durationmicro1.count() - 1000*durationmili1.count() << " ms" << std::endl;
    // std::cout << "integralCPU = " << integralCPU << std::endl;
    
}