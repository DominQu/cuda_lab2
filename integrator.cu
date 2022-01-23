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
    __shared__ float pointsx[SHARED_SIZE + 1];
    __shared__ float pointsy[SHARED_SIZE + 1];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    pointsx[threadIdx.x] = dvecx[index];
    pointsy[threadIdx.x] = dvecy[index];

    __syncthreads();
    if(threadIdx.x < SHARED_SIZE-1) {
        pointsy[threadIdx.x] = ((pointsy[threadIdx.x] + pointsy[threadIdx.x+1]) * (pointsx[threadIdx.x+1] - pointsx[threadIdx.x]) ) / 2;
    }
    else if(threadIdx.x == SHARED_SIZE-1 && blockIdx.x != blockDim.x-1) {
        pointsy[threadIdx.x] = ((pointsy[threadIdx.x] + dvecy[index+1]) * (dvecx[index+1] - pointsx[threadIdx.x]) ) / 2;

    }
    __syncthreads();

    if(threadIdx.x == 0) {
        if(blockIdx.x == gridDim.x - 1) {
            for(int i = 0; i < SHARED_SIZE - 1; i++) {
                // printf("i %d\n", i);
                sum[blockIdx.x*sizeof(float)] += (float)pointsy[i];
            }
            float suma = sum[blockIdx.x*sizeof(float)];
            // printf("last sum %d ", suma);
        }
        else {
            for(int i = 0; i < SHARED_SIZE; i++) {
                sum[blockIdx.x*sizeof(float)] += (float)pointsy[i];
            }

        }

    }
}
__global__ void dGPUintegrator(float *dvecx, 
                               float *dvecy,
                               float *integral,
                               int *maxindex)
{

    int threadnum = SHARED_SIZE;
    int blocknum = *maxindex / threadnum;
    // __device__ sum[blocknum];
    // printf("threadnum %d ", threadnum);
    // printf("blocknum %d ", blocknum);
    
    float *sum;
    cudaMalloc(&sum, sizeof(float)*blocknum);
    dGPUsubintegrator<<<blocknum, threadnum>>>(dvecx, 
                                               dvecy,
                                               sum,
                                               maxindex);

    cudaPeekAtLastError();
    cudaDeviceSynchronize();
    for(int i = 0 ; i < blocknum; i++) {
        *integral += (float)sum[i*sizeof(float)];
    }
    cudaFree(sum);
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

    //short arrays are sorted with simple method
    //long arrays are sorted with the usage of shared memory and dynamic parallelism
    if(maxindex <= 1024) {
        // choose number of threads and blocks
        dim3 threadnum = 256;
        dim3 blocknum =  dvecx.size() / threadnum.x + 1; 

        dsimpleGPUintegrator<<<blocknum, threadnum>>>(pdvecx.get(), pdvecy.get(), this->pdGPUintegral, pmaxindex);
    }
    else {
        dGPUintegrator<<<1,1>>>(pdvecx.get(), pdvecy.get(), this->pdGPUintegral, pmaxindex);
    }
    
    cudaMemcpy(&GPUintegral, this->pdGPUintegral, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(pmaxindex);

    return this->GPUintegral;
}