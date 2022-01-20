#include "sorter.cuh"
///////////////////////////////////////////////////
// GPU section
// __global__ void dBitonicSubSort(float *pointsx, float *pointsy) {
//     // shared memory allocation
//     __shared__ s
// }
__device__ void CompareAndSwap(float &x, float &nextx, float &y, float &nexty, int direction) {
    if(direction == 0 ) {
        //smaller than bigger
        if( x > nextx) {
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
__device__ void Swap(float &first, float &second) {

    float temp = first;
    first = second;
    second = temp;
}

__global__ void dBitonicSort(float *pointsx, float *pointsy, int *vectorlen) {
    //main kernel responsible for sorting
    extern __shared__ float shared[];

    int index = threadIdx.x;
        // printf("curretn index %d\n", index);
    shared[index] = pointsx[index];
    // shared[index + *vectorlen] = pointsy[index];
    __syncthreads();
    // printf("vector len %d ", *vectorlen);
    // if(index < *vectorlen){
    
        for(int size = 2; size <= *vectorlen; size *= 2) {
            // int direction = (threadIdx.x& size/2 ) != 0;

            for(int step = size/2; step>0; step /= 2) {
                int swapindex = index ^ step;
                // printf("swapindex %d\n", swapindex);

                // uint element = 2*threadIdx.x - (threadIdx.x & (step -1));
                // CompareAndSwap(pointsx[element],
                //             pointsx[element + step],
                //             pointsy[element],
                //             pointsy[element + step],
                //             direction);
                if(swapindex > index) {
                    if( (index & step) == 0) {
                        if(shared[threadIdx.x] > shared[swapindex]) {
                            Swap(shared[threadIdx.x], shared[swapindex]);
                            // Swap(shared[threadIdx.x + *vectorlen], shared[swapindex + *vectorlen]);
                        }
                    }
                    else {
                        if(shared[threadIdx.x] < shared[swapindex]) {
                            Swap(shared[threadIdx.x], shared[swapindex]);
                            // Swap(shared[threadIdx.x + *vectorlen], shared[swapindex + *vectorlen]);
                        }
                    }
                }
                __syncthreads();
            }
        }
    
    pointsx[index] = shared[index];
    // pointsy[index] = shared[index + *vectorlen];
}

void BitonicSort(thrust::device_vector<float> &dpointsx,
                 thrust::device_vector<float> &dpointsy)
{
    thrust::device_ptr<float> pdpointsx = dpointsx.data();
    thrust::device_ptr<float> pdpointsy = dpointsy.data();

    int *dvectorlen;
    int vectorlen = (int)dpointsx.size();
    cudaMalloc(&dvectorlen, sizeof(int));
    cudaMemcpy(dvectorlen, &vectorlen, sizeof(int), cudaMemcpyHostToDevice);

    int threadnum = 256;
    int blocknum =  dpointsx.size() / ( 2 * threadnum ) + 1;
    std::cout << "blocknum " << blocknum << std::endl;

    for( auto i : dpointsx) {
        std::cout << i << std::endl;
    }
    std::cout << "sorting\n";
    dBitonicSort<<<1, threadnum, sizeof(float)*2*256>>>(pdpointsx.get(), pdpointsy.get(), dvectorlen);
    cudaDeviceSynchronize();
    cudaFree(dvectorlen);
    for( auto i : dpointsx) {
        std::cout << i << std::endl;
    }
}

void Sorter::GPUsort(thrust::device_vector<float> &dpointsx,
                     thrust::device_vector<float> &dpointsy) {
    auto start2 = std::chrono::high_resolution_clock::now();

    // allocate GPU memory
    // thrust::device_vector<float> dpointsx = pointsx;
    // thrust::device_vector<float> dpointsy = pointsy;

    // thrust::sort_by_key(dpointsx.begin(), dpointsx.end(), dpointsy.begin());
    BitonicSort(dpointsx, dpointsy);

    // get device pointer
    // thrust::device_ptr<float> pdpointsx = dpointsx.data();
    // thrust::device_ptr<float> pdpointsy = dpointsy.data();
    
    // dBitonicSort<<<1,1>>>(pdpointsx.get(), pdpointsy.get());
    cudaDeviceSynchronize();
    auto stop2 = std::chrono::high_resolution_clock::now();
    auto durationmili2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - start2);
    auto durationmicro2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);
    std::cout << "Sort duration: " << durationmili2.count() << "." << durationmicro2.count() << " miliseconds" << std::endl;
}

///////////////////////////////////////////////////
// CPU section
void Sorter::CPUsort(thrust::host_vector<float> &pointsx,
                     thrust::host_vector<float> &pointsy) {

    auto start = std::chrono::high_resolution_clock::now();

    QuickSort( pointsx, pointsy, 0, pointsx.size() - 1);

    auto stop = std::chrono::high_resolution_clock::now();
    auto durationmili = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    auto durationmicro = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Sort duration: " << durationmili.count() << "." << durationmicro.count() << " miliseconds" << std::endl;

}

int Sorter::DivideAndSort(thrust::host_vector<float> &pointsx,
                          thrust::host_vector<float> &pointsy,
                          int begin,
                          int end) {

    float pivot = pointsx[end];
    // std::cout << " pivot " << pivot << std::endl;
    int smaller_ind = begin - 1;

    for(int j = begin; j <= end - 1; j++) {
            if(pointsx[j] <= pivot) {
                // std::cout << "point smaller than pivot " << pointsx[j] << std::endl;
                smaller_ind++;
                std::swap(pointsx[smaller_ind], pointsx[j]);
                std::swap(pointsy[smaller_ind], pointsy[j]);
            }
        }
    std::swap(pointsx[smaller_ind+1], pointsx[end]);
    std::swap(pointsy[smaller_ind+1], pointsy[end]);

    return smaller_ind + 1;
}

void Sorter::QuickSort(thrust::host_vector<float> &pointsx,
                       thrust::host_vector<float> &pointsy,
                       int begin,
                       int end) {
    if(begin < end) {
        // std::cout << " last index " << end << std::endl;

        int good_index = Sorter::DivideAndSort( pointsx, pointsy, begin, end);
        // std::cout << " good index " << good_index << std::endl;
        Sorter::QuickSort(pointsx,pointsy,begin,good_index - 1);
        Sorter::QuickSort(pointsx,pointsy,good_index + 1,end);    
    }
}
