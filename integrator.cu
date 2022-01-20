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
__global__ void dGPUsubintegrator(float *dvecx, 
                                  float *dvecy,
                                  float *sum,
                                  int *maxindex)
{
    __shared__ float pointsx[SHARED_SIZE];
    __shared__ float pointsy[SHARED_SIZE];

    uint index = blockIdx.x * blockDim + threadIdx.x;
    if(index < *maxindex) {
        pointsx[index] = dvecx[index];
        pointsy[index] = dvecy[index];

        __syncthreads();

        pointsy[index] = ((pointsy[index] + pointsy[index+1]) * (pointsx[index+1] - pointsx[index]) ) / 2

        __syncthreads();

        if(threadIdx.x == 0) {
            for(int i = index; i < index + blockDim; i++) {
                sum[blockIdx.x] += points[i];
            }
        }
    }
}
__global__ void dGPUintegrator(float *dvecx, 
                               float *dvecy,
                               float *integral,
                               int *maxindex)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    int threadnum = SHARED_SIZE;
    int blocknum = maxindex / threadnum + 1;
    __device__ sum[blocknum];
    dGPUsubintegrator<<<blocknum, threadnum>>>(dvecx, 
                                               dvecy,
                                               sum,
                                               maxindex);
    cudaDeviceSynchronize();
    for( const auto i : sum) {
        *integral += i;
    }

}

__global__ void dsimpleGPUintegrator(float *dvecx, 
                               float *dvecy,
                               float *integral,
                               int *maxindex) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if(index < *maxindex - 1) {
        
        atomicAdd(integral, ((dvecy[index] + dvecy[index+1]) * (dvecx[index+1] - dvecx[index]) ) / 2);
    }
}

float Integrator::GPUintegrator(thrust::device_vector<float> &dvecx,
                                thrust::device_vector<float> &dvecy)
{
    // allocate GPU memory
    // thrust::device_vector<float> dvecx = vecx;
    // thrust::device_vector<float> dvecy = vecy;

    // get device pointer
    thrust::device_ptr<float> pdvecx = dvecx.data();
    thrust::device_ptr<float> pdvecy = dvecy.data();

    int maxindex = dvecx.size();
    int *pmaxindex;
    cudaMalloc(&pmaxindex, sizeof(int));
    cudaMemcpy(pmaxindex, &maxindex, sizeof(int), cudaMemcpyHostToDevice );

    // choose number of threads and blocks
    dim3 threadnum = 256;
    dim3 blocknum =  dvecx.size() / threadnum.x + 1; 

    dsimpleGPUintegrator<<<blocknum, threadnum>>>(pdvecx.get(), pdvecy.get(), this->pdGPUintegral, pmaxindex);

    cudaMemcpy(&GPUintegral, this->pdGPUintegral, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(pmaxindex);

    return this->GPUintegral;
}