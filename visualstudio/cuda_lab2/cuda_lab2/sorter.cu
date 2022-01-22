#include "sorter.cuh"
///////////////////////////////////////////////////
// GPU section
// __global__ void dBitonicSubSort(float *pointsx, float *pointsy) {
//     // shared memory allocation
//     __shared__ s
// }
__device__ void CompareAndSwap(float& x, float& nextx, float& y, float& nexty, int direction) {
    if (direction == 0) {
        //smaller than bigger
        if (x > nextx) {
            float temp = nextx;
            nextx = x;
            x = temp;
            float tempy = nexty;
            nexty = y;
            y = tempy;
            return;
        }
        else {
            return;
        }
    }
    else {
        //bigger than smaller
        if (nextx > x) {
            float temp = nextx;
            nextx = x;
            x = temp;
            float tempy = nexty;
            nexty = y;
            y = tempy;
            return;
        }
        else {
            return;
        }
    }
}
__device__ void Swap(float& first, float& second) {

    float temp = first;
    first = second;
    second = temp;
}

//Bitonic Sort for arrays of length <=1024
__global__ void dBitonicSort(float* pointsx, float* pointsy, int* vectorlen) {

    int direction = blockIdx.x % 2;

    extern __shared__ float shared[];

    int index = threadIdx.x;
    int globalindex = blockDim.x * blockIdx.x + threadIdx.x;

    if (globalindex < *vectorlen) {
        int offset = *vectorlen > THREAD_NUM ? THREAD_NUM : *vectorlen;
        shared[index] = pointsx[globalindex];
        shared[index + offset] = pointsy[globalindex];
        // printf("sharedloaded\n");

        __syncthreads();
        //direction == 0 means sorting in ascending order
        //direction == 1 means sorting in descending order
        if (direction == 0) {

            for (int size = 2; size <= offset; size *= 2) {
                for (int step = size / 2; step > 0; step /= 2) {

                    int swapindex = index ^ step;

                    if (swapindex > index) {
                        if ((index & size) == 0) {
                            if (shared[threadIdx.x] > shared[swapindex]) {
                                Swap(shared[threadIdx.x], shared[swapindex]);
                                Swap(shared[threadIdx.x + offset], shared[swapindex + offset]);
                            }
                        }
                        else {
                            if (shared[threadIdx.x] < shared[swapindex]) {
                                Swap(shared[threadIdx.x], shared[swapindex]);
                                Swap(shared[threadIdx.x + offset], shared[swapindex + offset]);
                            }
                        }
                    }
                    __syncthreads();
                }
            }
        }
        else {
            for (int size = 2; size <= offset; size *= 2) {
                for (int step = size / 2; step > 0; step /= 2) {

                    int swapindex = index ^ step;

                    if (swapindex > index) {
                        if ((index & size) == 0) {
                            if (shared[threadIdx.x] < shared[swapindex]) {
                                Swap(shared[threadIdx.x], shared[swapindex]);
                                Swap(shared[threadIdx.x + offset], shared[swapindex + offset]);
                            }
                        }
                        else {
                            if (shared[threadIdx.x] > shared[swapindex]) {
                                Swap(shared[threadIdx.x], shared[swapindex]);
                                Swap(shared[threadIdx.x + offset], shared[swapindex + offset]);
                            }
                        }
                    }
                    __syncthreads();
                }
            }
        }

        pointsx[globalindex] = shared[index];
        pointsy[globalindex] = shared[index + offset];
    }
}

__global__ void dBitonicSortGlobalMem(float* pointsx, float* pointsy, int* vectorlen, int size, int step) {

    int offset = 2 * step;
    int separategroups = size / (2*step);
    int globalindex = blockIdx.x * blockDim.x + threadIdx.x;
    //int direction = blockIdx.x % 2;
    if (globalindex >= size / (2* separategroups)) {
        int groupnum = globalindex / step;
        globalindex += groupnum * step;
        /*if (groupnum % 2 == 1) {
            direction = 1;
        }*/
    }
    //int direction = (globalindex & (1 * step)) / (1 * step);
    int direction = (globalindex / (size)) % 2;
    //if(step == 204)
    
    /*if (size == *vectorlen) {
        direction == 0;
    }*/
    int swapindex = globalindex + step;
    __syncthreads();

    CompareAndSwap(pointsx[globalindex],
        pointsx[swapindex],
        pointsy[globalindex],
        pointsy[swapindex],
        direction);
    if (step == 2048) {
        //printf("point %d\n", (float)pointsx[globalindex]);
    }

}__global__ void dBitonicSortSharedMem(float* pointsx, float* pointsy, int* vectorlen, int size, int step) {

}

__global__ void dBitonicSortManage(float* pointsx, float* pointsy, int* vectorlen) {
    //this kernel launches other kernels responsible for sorting

    //launch n small kernels so that we get n/2 bitonic sequences
    int blocknum = *vectorlen / THREAD_NUM;
    printf("blocknum %d\n", blocknum);

    dBitonicSort << <blocknum, THREAD_NUM, 2 * THREAD_NUM * sizeof(float) >> > (pointsx, pointsy, vectorlen);

    cudaDeviceSynchronize();

    for (int size = 2; size <= *vectorlen; size *= 2) {
        for (int step = size / 2; step > 0; step /= 2) {
            for (int index = 0; index < *vectorlen; index++) {

                int swapindex = index ^ step;
                if (index == *vectorlen - 1) {
                    printf("swapindex %d\n", swapindex);
                }

                if (swapindex > index) {
                    if ((index & size) == 0) {
                        if (pointsx[index] > pointsx[swapindex]) {
                            Swap(pointsx[index], pointsx[swapindex]);
                            Swap(pointsy[index], pointsy[swapindex]);
                        }
                    }
                    else {
                        if (pointsx[index] < pointsx[swapindex]) {
                            Swap(pointsx[index], pointsx[swapindex]);
                            Swap(pointsy[index], pointsy[swapindex]);
                        }
                    }
                }
            }
        }
    }



}

void BitonicSort(thrust::device_vector<float>& dpointsx,
    thrust::device_vector<float>& dpointsy)
{
    thrust::device_ptr<float> pdpointsx = dpointsx.data();
    thrust::device_ptr<float> pdpointsy = dpointsy.data();

    int* dvectorlen;
    int vectorlen = (int)dpointsx.size();
    cudaMalloc(&dvectorlen, sizeof(int));
    cudaMemcpy(dvectorlen, &vectorlen, sizeof(int), cudaMemcpyHostToDevice);

    int threadnum = THREAD_NUM;
    // int blocknum =  dpointsx.size() / ( 2 * threadnum ) + 1;
    if (vectorlen > threadnum) {

        int blocknum = vectorlen / threadnum;

        dBitonicSort << <blocknum, THREAD_NUM, 2 * THREAD_NUM * sizeof(float) >> > (pdpointsx.get(), pdpointsy.get(), dvectorlen);

        for (int size = 2 * threadnum; size <= vectorlen; size *= 2) {
            for (int step = size / 2; step > 0; step /= 2) {
                if (step >= threadnum || true) {
                    dBitonicSortGlobalMem << <vectorlen / (2 * threadnum), threadnum >> > (pdpointsx.get(), pdpointsy.get(), dvectorlen, size, step);
                    /*std::cout << "Current step is " << step << std::endl;
                    for (auto i : dpointsx) {
                        std::cout << i << std::endl;
                    }
                    break;*/
                }
                else {
                    //dBitonicSortSharedMem << < >> > ();
                }
            }
        }
        // dBitonicSortManage<<<1,1>>>(pdpointsx.get(), pdpointsy.get(), dvectorlen);
    }
    else {
        dBitonicSort << <1, threadnum, sizeof(float) * 2 * vectorlen >> > (pdpointsx.get(), pdpointsy.get(), dvectorlen);
    }
    cudaDeviceSynchronize();
    cudaFree(dvectorlen);

}

void Sorter::GPUsort(thrust::device_vector<float>& dpointsx,
    thrust::device_vector<float>& dpointsy) {
    auto start2 = std::chrono::high_resolution_clock::now();

    // allocate GPU memory
    // thrust::device_vector<float> dpointsx = pointsx;
    // thrust::device_vector<float> dpointsy = pointsy;

    // thrust::sort_by_key(dpointsx.begin(), dpointsx.end(), dpointsy.begin());
    BitonicSort(dpointsx, dpointsy);

    // get device pointer
    // thrust::device_ptr<float> pdpointsx = dpointsx.data();
    // thrust::device_ptr<float> pdpointsy = dpointsy.data();

    cudaDeviceSynchronize();
    auto stop2 = std::chrono::high_resolution_clock::now();
    auto durationmili2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - start2);
    auto durationmicro2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);
    std::cout << "Sort duration: " << durationmili2.count() << "." << durationmicro2.count() << " miliseconds" << std::endl;
}

///////////////////////////////////////////////////
// CPU section
void Sorter::CPUsort(thrust::host_vector<float>& pointsx,
    thrust::host_vector<float>& pointsy) {

    auto start = std::chrono::high_resolution_clock::now();

    QuickSort(pointsx, pointsy, 0, pointsx.size() - 1);
    // for(auto i : pointsy) {
    //     std::cout << i << std::endl;
    // }

    auto stop = std::chrono::high_resolution_clock::now();
    auto durationmili = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    auto durationmicro = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Sort duration: " << durationmili.count() << "." << durationmicro.count() << " miliseconds" << std::endl;

}

int Sorter::DivideAndSort(thrust::host_vector<float>& pointsx,
    thrust::host_vector<float>& pointsy,
    int begin,
    int end) {

    float pivot = pointsx[end];
    // std::cout << " pivot " << pivot << std::endl;
    int smaller_ind = begin - 1;

    for (int j = begin; j <= end - 1; j++) {
        if (pointsx[j] <= pivot) {
            // std::cout << "point smaller than pivot " << pointsx[j] << std::endl;
            smaller_ind++;
            std::swap(pointsx[smaller_ind], pointsx[j]);
            std::swap(pointsy[smaller_ind], pointsy[j]);
        }
    }
    std::swap(pointsx[smaller_ind + 1], pointsx[end]);
    std::swap(pointsy[smaller_ind + 1], pointsy[end]);

    return smaller_ind + 1;
}

void Sorter::QuickSort(thrust::host_vector<float>& pointsx,
    thrust::host_vector<float>& pointsy,
    int begin,
    int end) {
    if (begin < end) {
        // std::cout << " last index " << end << std::endl;

        int good_index = Sorter::DivideAndSort(pointsx, pointsy, begin, end);
        // std::cout << " good index " << good_index << std::endl;
        Sorter::QuickSort(pointsx, pointsy, begin, good_index - 1);
        Sorter::QuickSort(pointsx, pointsy, good_index + 1, end);
    }
}
