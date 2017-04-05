#ifndef PYCUDNN_ACTIVATION_DESCRIPTOR_HPP
#define PYCUDNN_ACTIVATION_DESCRIPTOR_HPP

#include <cudnn.h>

#include "RAII.hpp" // RAII

namespace CuDNN {
    class ActivationDescriptor :
        public RAII< cudnnActivationDescriptor_t,
                    cudnnCreateActivationDescriptor,
                    cudnnDestroyActivationDescriptor > {};
}

#endif // PYCUDNN_ACTIVATION_DESCRIPTOR_HPP
