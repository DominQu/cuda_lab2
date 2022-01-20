#include "sorter.cuh"
///////////////////////////////////////////////////
// GPU section
__global__ void dBitonicSubSort(float *pointsx, float *pointsy) {
    // shared memory allocation
    __shared__ s
}

__global__ void dBitonicSort(float *pointsx, float *pointsy) {
    //main kernel responsible for sorting

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int threadperSM;
    cudaDeviceGetAttribute(&threadperSM, cudaDevAttrMaxThreadsPerBlock, 0);

    // int poweroftwo = (int)floor(log2((float)numSMs));
    dim3 threadnum = 512;
    dim3 blocknum = (numSMs * threadperSM) / threadnum.x + 1;
    dBitonicSubSort<<<blocknum, threadnum>>>(pointsx, pointsy);
    // printf("Power of two: %d\n",poweroftwo);

}

void Sorter::GPUsort(thrust::device_vector<float> &dpointsx,
                     thrust::device_vector<float> &dpointsy) {
    auto start2 = std::chrono::high_resolution_clock::now();

    // allocate GPU memory
    // thrust::device_vector<float> dpointsx = pointsx;
    // thrust::device_vector<float> dpointsy = pointsy;

    thrust::sort_by_key(dpointsx.begin(), dpointsx.end(), dpointsy.begin());

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

// thrust::host_vector<float> BitonicSort(thrust::host_vector<float> &pointsx,
//                  thrust::host_vector<float> &pointsy,
//                  int vecsize) {
//     if(pointsx.size() <=1) {
//         return pointsx;
//     }
//     else {
//         thrust::host_vector<float> lower = BitonicSort() 
//     }
// }