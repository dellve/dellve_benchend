#ifndef DELLVE_CUDNNACTIVATION_HPP_
#define DELLVE_CUDNNACTIVATION_HPP_

#include <vector>
#include <string>

#include <cudnn.h>

#include <iostream>

#include "dellve_cudnn_benchmark.hpp"
#include "dellve_constants.hpp"
#include "CuDNN/ActivationDescriptor.hpp"
#include "CuDNN/Handle.hpp"
#include "CuDNN/Status.hpp"
#include "CuDNN/Tensor.hpp"

namespace CuDNN {
    namespace Activation {
        CuDNN::ActivationDescriptor createDescriptor(void) {
            CuDNN::ActivationDescriptor descriptor;
            CuDNN::checkStatus (
                cudnnSetActivationDescriptor ( 
                    descriptor,
                    CUDNN_ACTIVATION_TANH,
                    CUDNN_NOT_PROPAGATE_NAN,
                    0.0
                )
            );

            return descriptor;
        }

        template <typename T>
        DELLve::Benchmark forward ( int n, int c, int h, int w ) {
	        CuDNN::Handle handle;
            auto descriptor = createDescriptor();
            auto x = CuDNN::Tensor<T>::createNCHW(n, c, h, w);
            auto y = CuDNN::Tensor<T>::createNCHW(n, c, h, w);

            return [=]() {
                return cudnnActivationForward (
                    handle,
                    descriptor,
                    &(CuDNN::Constants::alpha),
                    x.getDescriptor(),
                    x,
                    &(CuDNN::Constants::beta),
                    y.getDescriptor(),
                    y 
                );
            };	

        }

        template <typename T>
        DELLve::Benchmark backward ( int n, int c, int h, int w ) {
            CuDNN::Handle handle;
            auto descriptor = createDescriptor();
            auto x = CuDNN::Tensor<T>::createNCHW(n, c, h, w);
            auto y = CuDNN::Tensor<T>::createNCHW(n, c, h, w);
            auto dx = CuDNN::Tensor<T>::createNCHW(n, c, h, w);
            auto dy = CuDNN::Tensor<T>::createNCHW(n, c, h, w);

            return [=]() {
                return cudnnActivationBackward (
                    handle,
                    descriptor,
                    &(CuDNN::Constants::alpha),
                    y.getDescriptor(),
                    y,
                    dy.getDescriptor(),
                    dy,
                    x.getDescriptor(),
                    x,
                    &(CuDNN::Constants::beta),
                    dx.getDescriptor(),
                    dx
                );
            };
        }
    };
};

#endif //DELLVE_CUDNNACTIVATION_HPP_

