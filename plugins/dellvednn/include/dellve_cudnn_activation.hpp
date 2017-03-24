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
            std::cout << "Creating activation descriptor..." << std::endl;
            CuDNN::ActivationDescriptor descriptor;

            std::cout << "Setting activation descriptor..." << std::endl;
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
            std::cout << "Creating handle..." << std::endl;
	        CuDNN::Handle handle;

            CuDNN::ActivationDescriptor descriptor = createDescriptor();

            std::cout << "Creating tensor x" << std::endl;
            auto x = CuDNN::Tensor<T>::NCHW::create(n, c, h, w);

            std::cout << "Creating tensor y" << std::endl;
            auto y = CuDNN::Tensor<T>::NCHW::create(n, c, h, w);

            std::cout << "Creating benchmark..." << std::endl;

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
            std::cout << "Creating handle..." << std::endl;
            CuDNN::Handle handle;

            CuDNN::ActivationDescriptor descriptor = createDescriptor();

            std::cout << "Creating tensor x" << std::endl;
            auto x = CuDNN::Tensor<T>::NCHW::create(n, c, h, w);

            std::cout << "Creating tensor y" << std::endl;
            auto y = CuDNN::Tensor<T>::NCHW::create(n, c, h, w);

            std::cout << "Creating tensor dx" << std::endl;
            auto dx = CuDNN::Tensor<T>::NCHW::create(n, c, h, w);

            std::cout << "Creating tensor dy" << std::endl;
            auto dy = CuDNN::Tensor<T>::NCHW::create(n, c, h, w);

            std::cout << "Creating benchmark..." << std::endl;
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

