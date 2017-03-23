#ifndef PYCUDNN_OP_TENSOR_DESCRIPTOR_HPP
#define PYCUDNN_OP_TENSOR_DESCRIPTOR_HPP

#include <cudnn.h>

#include "RAII.hpp" // RAII

namespace CuDNN {

    class OpTensorDescriptor :
        public RAII< cudnnOpTensorDescriptor_t,
                    cudnnCreateOpTensorDescriptor,
                    cudnnDestroyOpTensorDescriptor > {};
} // PyCuDNN

#endif // PYCUDNN_OP_TENSOR_DESCRIPTOR_HPP
