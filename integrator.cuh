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
        float GPUintegrator( thrust::device_vector<float> &vecx,
                             thrust::device_vector<float> &vecy);
        Integrator();
        ~Integrator();
};