#pragma once

#include <vector>
#include "point.hpp"
#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


class Sorter {
    public:
        Sorter() {};
        ~Sorter() {};
        void CPUsort(thrust::host_vector<float> &pointsx,
                     thrust::host_vector<float> &pointsy);
        void GPUsort(thrust::device_vector<float> &dpointsx,
                     thrust::device_vector<float> &dpointsy);
    private:
        void QuickSort(thrust::host_vector<float> &pointsx, 
                       thrust::host_vector<float> &pointsy, 
                       int begin, 
                       int end);
        int DivideAndSort(thrust::host_vector<float> &pointsx, 
                       thrust::host_vector<float> &pointsy, 
                       int begin, 
                       int end);
};