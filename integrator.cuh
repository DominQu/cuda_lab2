#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


class Integrator {
    private:
        float CPUintegral;
        float *pdGPUintegral;
        float GPUintegral;
    public:
        float CPUintegrator(const thrust::host_vector<float> &vecx,
                             const thrust::host_vector<float> &vecy);
        float GPUintegrator(const thrust::host_vector<float> &vecx,
                             const thrust::host_vector<float> &vecy);
        Integrator();
        ~Integrator();
};