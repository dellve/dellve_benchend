#ifndef PYCUDNN_TENSOR_DESCRIPTOR_HPP
#define PYCUDNN_TENSOR_DESCRIPTOR_HPP

#include <cudnn.h>

#include "RAII.hpp" // RAII

namespace CuDNN {
    struct TensorDescriptor :
      RAII< cudnnTensorDescriptor_t,
            cudnnCreateTensorDescriptor,
            cudnnDestroyTensorDescriptor > {};
}

#endif // PYCUDNN_TENSOR_DESCRIPTOR_HPP
