#ifndef PYCUDNN_POOLING_DESCRIPTOR_HPP
#define PYCUDNN_POOLING_DESCRIPTOR_HPP

#include <cudnn.h>
#include <stdio.h>

#include "RAII.hpp" // RAII

namespace CuDNN {
    class PoolingDescriptor :
    public RAII< cudnnPoolingDescriptor_t,
                    cudnnCreatePoolingDescriptor,
                    cudnnDestroyPoolingDescriptor > {
   
    public:  
        static PoolingDescriptor create (int winH, int winW,
                                         int padH, int padW,
                                         int hStride, int wStride,
                                         cudnnPoolingMode_t mode) {
            PoolingDescriptor object;

            CuDNN::checkStatus(
                cudnnSetPooling2dDescriptor(
                    object,
                    mode,
                    CUDNN_NOT_PROPAGATE_NAN,
                    winH,
                    winW,
                    padH,
                    padW,
                    hStride,
                    wStride
                )
            );
            
            return object;          
        }            
    };
} // PyCuDNN

#endif // PYCUDNN_POOLING_DESCRIPTOR_HPP
