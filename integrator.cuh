#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
// #include "cudaerror.cuh"

const int SHARED_SIZE = 1024;

class Integrator {
    private:
        float CPUintegral;
        float *pdGPUintegral;
        float GPUintegral; //GPU integrator works for n of points being power of two
    public:
        float CPUintegrator(const thrust::host_vector<float> &vecx,
                             const thrust::host_vector<float> &vecy);
        float GPUintegrator( thrust::device_vector<float> &vecx,
                             thrust::device_vector<float> &vecy);
        Integrator();
        ~Integrator();
};