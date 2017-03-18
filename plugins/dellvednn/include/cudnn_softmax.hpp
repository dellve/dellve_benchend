#ifndef DELLVE_CUDNNSOFTMAX_H_
#define DELLVE_CUDNNSOFTMAX_H_

#include <vector>

#include <cudnn.h>

#include "cudnn_helper.hpp"
#include "tensor.hpp"

enum class CudnnSoftmaxAlgorithm { FAST, ACCURATE, LOG };

/**
 * Wraps around cuDNN calls for all of the softmax methods.
 *
 * Initializes the descriptors required the run the softmax methods when the instance is created.
 * Then, sets the algorithm based on user input. Provides user with the backwards and forwards
 * methods that uses the algorithm specified by the user to run softmax on the GPU.
 */
class CudnnSoftmax {
private:
    CudnnHandle cudnn_handle_;

    TensorDescriptor4d<float> x_desc_;
    TensorDescriptor4d<float> y_desc_;

    cudnnSoftmaxAlgorithm_t algorithm_;

    //TODO: No need for output_dims?
    std::vector<int> output_dims_;

    const float alpha_ = 1.f;
    const float beta_ = 0.f;
public: 
    /**
     * Initializes the x and y descriptors of the softmax operation based on the problem set.
     * Currently, both the dimensions of the x and y are the same. Also initializes the algorithm
     * to run (Fast, Accurate, or Log) based on the user input. This is used later in the forward
     * and backward methods.
     */
    CudnnSoftmax (int w, int h, int c, int n, int device, CudnnSoftmaxAlgorithm alg) : 
                  cudnn_handle_(device),
                  x_desc_(CUDNN_TENSOR_NCHW, n, c, h, w),
                  y_desc_(CUDNN_TENSOR_NCHW, n, c, h, w) {
        switch (alg) {
            case CudnnSoftmaxAlgorithm::FAST:
                algorithm_ = CUDNN_SOFTMAX_FAST;
                break;
            case CudnnSoftmaxAlgorithm::ACCURATE:
                algorithm_ = CUDNN_SOFTMAX_ACCURATE;
                break;
            case CudnnSoftmaxAlgorithm::LOG:
                algorithm_ = CUDNN_SOFTMAX_LOG;
                break;
            default:
                algorithm_ = CUDNN_SOFTMAX_ACCURATE;
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
    // TODO: Change SOFTMAX Algorithm and Mode dynamically
    void forward(Tensor<float> x, Tensor<float> y) {
        // Softmax Forward
        CHECK_CUDNN_ERROR(cudnnSoftmaxForward(cudnn_handle_.handle(),
                                              algorithm_,
                                              CUDNN_SOFTMAX_MODE_CHANNEL,
                                              &alpha_,
                                              x_desc_.desc(),
                                              x.begin(),
                                              &beta_,
                                              y_desc_.desc(),
                                              y.begin()));
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
    // TODO: Change SOFTMAX Algorithm and Mode dynamically
    void backward(Tensor<float> y, Tensor<float> dY, Tensor<float> dX) {
        // Softmax Backward
        CHECK_CUDNN_ERROR(cudnnSoftmaxBackward(cudnn_handle_.handle(),
                                               algorithm_,
                                               CUDNN_SOFTMAX_MODE_CHANNEL,
                                               &alpha_,
                                               y_desc_.desc(),
                                               y.begin(),
                                               y_desc_.desc(),
                                               dY.begin(),
                                               &beta_,
                                               x_desc_.desc(),
                                               dX.begin()));
    }
};

#endif // DELLVE_CUDNNSOFTMAX_H_

