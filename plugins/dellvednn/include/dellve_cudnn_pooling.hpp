#ifndef DELLVE_CUDNNPOOLING_H_
#define DELLVE_CUDNNPOOLING_H_

#include <vector>
#include <string>

#include <cudnn.h>

#include <iostream>

#include "dellve_cudnn_benchmark.hpp"
#include "dellve_constants.hpp"
#include "CuDNN/Handle.hpp"
#include "CuDNN/Status.hpp"
#include "CuDNN/Tensor.hpp"
#include "CuDNN/PoolingMode.hpp"
#include "CuDNN/PoolingDescriptor.hpp"

namespace CuDNN {
    namespace Pooling {
        CuDNN::PoolingMode convertMode(std::string mode) {
            std::cout << "Setting Pooling Mode to " << mode << std::endl;
            if(mode.compare("max") == 0) {
                return CUDNN_POOLING_MAX;
            } else if (mode.compare("avgpad") == 0) {
                return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
            } else if (mode.compare("avgnopad") == 0) {
                return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
            } else {
                std::cout << "Unrecognized Algorithm: " << mode << std::endl;
                std::cout << "Setting to Default Pooling Mode: MAX" << std::endl;
                return CUDNN_POOLING_MAX;  
            }
        }

        template <typename T>
        DELLve::Benchmark forward(int w, int h, int c, int n, 
                                  int winH, int winW, 
                                  int padH, int padW, 
                                  int hStride, int wStride,
                                  std::string mode) {

            CuDNN::Handle handle;
            CuDNN::PoolingMode poolingMode = convertMode(mode); 
          
            CuDNN::PoolingDescriptor descriptor = CuDNN::PoolingDescriptor::create(winH, winW,
                                                                                   padH, padW,
                                                                                   hStride, wStride,
                                                                                   poolingMode);
            auto x = CuDNN::Tensor<T>::createNCHW(n,c,h,w);
		   	std::tuple<int,int,int,int> outputDims; // NCHW tuple of output dimensions
            CuDNN::checkStatus(
                cudnnGetPooling2dForwardOutputDim(
                    descriptor,
                    x.getDescriptor(),
		   			&std::get<0>(outputDims),	// N
		   			&std::get<1>(outputDims),	// C
		   			&std::get<2>(outputDims),	// H
		   			&std::get<3>(outputDims)	// W
                )
            );
            auto y = CuDNN::Tensor<T>::createNCHW(outputDims);
    
            return [=]() {
                return cudnnPoolingForward(
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
        DELLve::Benchmark backward(int w, int h, int c, int n, 
                                   int winH, int winW, 
                                   int padH, int padW, 
                                   int hStride, int wStride,
                                   std::string mode) {
            CuDNN::Handle handle;
            CuDNN::PoolingMode poolingMode = convertMode(mode); 
          
            CuDNN::PoolingDescriptor descriptor = CuDNN::PoolingDescriptor::create(winH, winW,
                                                                                   padH, padW,
                                                                                   hStride, wStride,
                                                                                   poolingMode);
            auto x = CuDNN::Tensor<T>::createNCHW(n,c,h,w);
            auto dX = CuDNN::Tensor<T>::createNCHW(n,c,h,w);
		   	std::tuple<int,int,int,int> outputDims; // NCHW tuple of output dimensions
            CuDNN::checkStatus(
                cudnnGetPooling2dForwardOutputDim(
                    descriptor,
                    x.getDescriptor(),
		   			&std::get<0>(outputDims),	// N
		   			&std::get<1>(outputDims),	// C
		   			&std::get<2>(outputDims),	// H
		   			&std::get<3>(outputDims)	// W
                )
            );
            auto y = CuDNN::Tensor<T>::createNCHW(outputDims);
            auto dY = CuDNN::Tensor<T>::createNCHW(outputDims);

            return [=]() {
                return cudnnPoolingBackward(
                    handle,
                    descriptor,
                    &(CuDNN::Constants::alpha),
                    y.getDescriptor(),
                    y,
                    dY.getDescriptor(),
                    dY,
                    x.getDescriptor(),
                    x,
                    &(CuDNN::Constants::beta),
                    dX.getDescriptor(),
                    dX
                );
            };
        }
    }
}

#endif //DELLVE_CUDNNPOOLING_H_

