#ifndef DELLVE_CUDNNSOFTMAX_H_
#define DELLVE_CUDNNSOFTMAX_H_

#include <vector>
#include <string>

#include <cudnn.h>

#include <iostream>

#include "dellve_cudnn_benchmark.hpp"
#include "dellve_constants.hpp"
#include "CuDNN/Handle.hpp"
#include "CuDNN/Status.hpp"
#include "CuDNN/Tensor.hpp"
#include "CuDNN/SoftmaxAlgorithm.hpp"

namespace CuDNN {
    namespace Softmax {
        CuDNN::SoftmaxAlgorithm convAlgorithm(std::string alg) {
            // std::cout << "Setting Softmax Algorithm to " << alg << std::endl;
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
        DELLve::Benchmark forward(int w, int h, int c, int n, std::string alg) {
            CuDNN::Handle handle;
            auto algorithm = convAlgorithm(alg);
            auto x = CuDNN::Tensor<T>::createNCHW(n, c, h, w);
            auto y = CuDNN::Tensor<T>::createNCHW(n, c, h, w);

            return [=]() {
                return cudnnSoftmaxForward(handle,
                                   	  	   algorithm,
                                   	  	   CUDNN_SOFTMAX_MODE_CHANNEL,
                                   	  	   &CuDNN::Constants::alpha,
                                   	  	   x.getDescriptor(),
                                   	  	   x,
                                   	  	   &CuDNN::Constants::beta,
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
        DELLve::Benchmark backward(int w, int h, int c, int n, std::string alg) {
            CuDNN::Handle handle;
            auto algorithm = convAlgorithm(alg);
            auto dX = CuDNN::Tensor<T>::createNCHW(n, c, h, w);
            auto y = CuDNN::Tensor<T>::createNCHW(n, c, h, w);
            auto dY = CuDNN::Tensor<T>::createNCHW(n, c, h, w);

            return [=]() {
                return cudnnSoftmaxBackward(handle,
                                            algorithm,
                                            CUDNN_SOFTMAX_MODE_CHANNEL,
                                            &CuDNN::Constants::alpha,
                                            y.getDescriptor(),
                                            y,
                                            dY.getDescriptor(),
                                            dY,
                                            &CuDNN::Constants::beta,
                                            dX.getDescriptor(),
                                            dX);
            };
        }
    };
};

#endif // DELLVE_CUDNNSOFTMAX_H_

