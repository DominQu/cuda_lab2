#include "integrator.cuh"

Integrator::Integrator() : CPUintegral(0.0) {

    cudaMalloc(&pdGPUintegral, sizeof(float));
    cudaMemcpy(pdGPUintegral, &CPUintegral, sizeof(float), cudaMemcpyHostToDevice);
} 

Integrator::~Integrator() {
    cudaFree(pdGPUintegral);
}

float Integrator::CPUintegrator(const thrust::host_vector<float> &vecx,const thrust::host_vector<float> &vecy) {

    for(int i = 0; i < vecx.size() - 1; i++) {
        // std::cout << "current integral value is " << CPUintegral << " value to be added " << ((vecy[i] + vecy[i+1]) * (vecx[i+1] - vecx[i]) ) / 2 << std::endl;
        CPUintegral += ((vecy[i] + vecy[i+1]) * (vecx[i+1] - vecx[i]) ) / 2 ; // trapezoidal integration
    }

    return CPUintegral;
}
////////////////////////////////////////////////////////////////////////
// GPU implementation

__global__ void dGPUintegrator(float *dvecx, 
                               float *dvecy,
                               float *integral,
                               int *maxindex) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if(index < *maxindex - 1) {
        
        atomicAdd(integral, ((dvecy[index] + dvecy[index+1]) * (dvecx[index+1] - dvecx[index]) ) / 2);
    }
}

float Integrator::GPUintegrator(const thrust::host_vector<float> &vecx,
                                 const thrust::host_vector<float> &vecy)
{
    // allocate GPU memory
    thrust::device_vector<float> dvecx = vecx;
    thrust::device_vector<float> dvecy = vecy;

    // get device pointer
    thrust::device_ptr<float> pdvecx = dvecx.data();
    thrust::device_ptr<float> pdvecy = dvecy.data();

    int maxindex = vecx.size();
    int *pmaxindex;
    cudaMalloc(&pmaxindex, sizeof(int));
    cudaMemcpy(pmaxindex, &maxindex, sizeof(int), cudaMemcpyHostToDevice );

    // choose number of threads and blocks
    dim3 threadnum = 256;
    dim3 blocknum =  vecx.size() / threadnum.x + 1; 

    dGPUintegrator<<<blocknum, threadnum>>>(pdvecx.get(), pdvecy.get(), this->pdGPUintegral, pmaxindex);

    cudaMemcpy(&GPUintegral, this->pdGPUintegral, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(pmaxindex);

    return this->GPUintegral;
}