#ifndef DELLVE_CUDNNSOFTMAX_H_
#define DELLVE_CUDNNSOFTMAX_H_

#include <vector>
#include <string>

#include <cudnn.h>

#include <iostream>

#include "dellve_cudnn_benchmark.hpp"
#include "CuDNN/Handle.hpp"
#include "CuDNN/Status.hpp"
#include "CuDNN/Tensor.hpp"
#include "CuDNN/SoftmaxAlgorithm.hpp"

namespace CuDNN {
    namespace Softmax {
        CuDNN::SoftmaxAlgorithm convAlgorithm(std::string alg) {
            std::cout << "Setting Softmax Algorithm to " << alg << std::endl;
            if(alg.compare("fast") == 0) {
                return CUDNN_SOFTMAX_FAST;
            } else if (alg.compare("accurate") == 0) {
                return CUDNN_SOFTMAX_ACCURATE;
            } else if (alg.compare("log") == 0) {
                return CUDNN_SOFTMAX_LOG;
            } else {
                std::cout << "Unrecognized Algorithm: " << alg << std::endl;
                std::cout << "Setting to Default Softmax Algorithm: FAST" << std::endl;
                return CUDNN_SOFTMAX_FAST;  
            } 
             
        }

        /**
         * Run the softmax forward method given the data pointers to the GPU memory associated with 
         * x_desc, and y_desc.
         *
         **** Inputs
         * x - Tensor class associated with the tensor descriptor x_desc (input).
         * y - Tensor class associated with the tensor descriptor y_desc (output). 
         */ 
        template <typename T>
        DELLve::Benchmark forward(int n, int c, int h, int w, std::string alg) {
            CuDNN::SoftmaxAlgorithm algorithm = convAlgorithm(alg);
            static const float alpha = 1.f;
            static const float beta = 0.f;
            std::cout << "Creating Handle" << std::endl;
            CuDNN::Handle handle;

            std::cout << "Creating tensor x" << std::endl;
            auto x = CuDNN::Tensor<T>::NCHW::create(n, c, h, w);

            std::cout << "Creating tensor y" << std::endl;
            auto y = CuDNN::Tensor<T>::NCHW::create(n, c, h, w);

            std::cout << "Creating benchmark..." << std::endl;
            return [=]() {
                return cudnnSoftmaxForward(handle,
                                    algorithm,
                                    CUDNN_SOFTMAX_MODE_CHANNEL,
                                    &alpha,
                                    x.getDescriptor(),
                                    x,
                                    &beta,
                                    y.getDescriptor(),
                                    y);
            };
        }

        /**
         * Run the softmax backward method given the data pointers to the GPU memory associated with
         * the input tensor, the input differential tensor, and the output tensor.
         *
         **** Inputs
         * y - Tensor class associated with the tensor descriptor y_desc (output).
         * dY - Tensor class associated with the input differential tensor.
         * dX - Tensor class associated with the output tensor. 
         */
        template <typename T>
        DELLve::Benchmark backward(int n, int c, int h, int w, std::string alg) {
            CuDNN::SoftmaxAlgorithm algorithm = convAlgorithm(alg);
            static const float alpha = 1.f;
            static const float beta = 0.f;
            std::cout << "Creating Handle" << std::endl;
            CuDNN::Handle handle;

            std::cout << "Creating tensor dX" << std::endl;
            auto dX = CuDNN::Tensor<T>::NCHW::create(n, c, h, w);

            std::cout << "Creating tensor y" << std::endl;
            auto y = CuDNN::Tensor<T>::NCHW::create(n, c, h, w);

            std::cout << "Creating tensor dY" << std::endl;
            auto dY = CuDNN::Tensor<T>::NCHW::create(n, c, h, w);

            std::cout << "Creating benchmark..." << std::endl;
            return [=]() {
                return cudnnSoftmaxBackward(handle,
                                            algorithm,
                                            CUDNN_SOFTMAX_MODE_CHANNEL,
                                            &alpha,
                                            y.getDescriptor(),
                                            y,
                                            dY.getDescriptor(),
                                            dY,
                                            &beta,
                                            dX.getDescriptor(),
                                            dX);
            };
        }
    };
};

#endif // DELLVE_CUDNNSOFTMAX_H_

