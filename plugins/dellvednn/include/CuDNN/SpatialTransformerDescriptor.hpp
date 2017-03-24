#ifndef PYCUDNN_SPATIAL_TRANSFORMER_DESCRIPTOR_HPP
#define PYCUDNN_SPATIAL_TRANSFORMER_DESCRIPTOR_HPP

#include <cudnn.h>

#include "RAII.hpp" // RAII

namespace CuDNN {
    class SpatialTransformerDescriptor :
        public RAII< cudnnSpatialTransformerDescriptor_t,
                    cudnnCreateSpatialTransformerDescriptor,
                    cudnnDestroySpatialTransformerDescriptor > {};
} // PyCuDNN

#endif // PYCUDNN_SPATIAL_TRANSFORMER_DESCRIPTOR_HPP
