#include "sorter.cuh"

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