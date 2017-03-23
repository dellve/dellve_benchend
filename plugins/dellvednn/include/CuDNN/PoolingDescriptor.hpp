#ifndef PYCUDNN_POOLING_DESCRIPTOR_HPP
#define PYCUDNN_POOLING_DESCRIPTOR_HPP

#include <cudnn.h>

#include "RAII.hpp" // RAII

namespace CuDNN {
    class PoolingDescriptor :
    public RAII< cudnnPoolingDescriptor_t,
                    cudnnCreatePoolingDescriptor,
                    cudnnDestroyPoolingDescriptor > {};
} // PyCuDNN

#endif // PYCUDNN_POOLING_DESCRIPTOR_HPP
