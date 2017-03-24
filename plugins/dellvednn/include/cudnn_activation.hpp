#ifndef DELLVE_CUDNNACTIVATION_H_
#define DELLVE_CUDNNACTIVATION_H_

#include <vector>

#include <cudnn.h>

#include "cudnn_helper.hpp"
#include "tensor.hpp"

/**
 * Wraps around cuDNN calls for all of the activation methods.
 *
 * Initializes the activation descriptor required to run activation when the instance is created.
 * Then, provides the forward and backward methods to allow users to run the activation methods.
 */
class CudnnActivation {
private:
    CudnnHandle cudnn_handle_;
    ActivationDescriptor activation_desc_;

    TensorDescriptor4d<float> x_desc_;
    TensorDescriptor4d<float> y_desc_;

    const float alpha_ = 1.f;
    const float beta_ = 0.f;
public:
    /**
     * Initializes the x and y descriptors (currently assumed to be the same dimensions). Then,
     * initializes the activation descriptor (all handled in cudnn_helper).
     */
    // TODO: Extend activation_descriptor in cudnn_helper
    CudnnActivation(int w, int h, int c, int n, int device) : 
                    cudnn_handle_(device),
                    x_desc_(CUDNN_TENSOR_NCHW, n, c, h, w),
                    y_desc_(CUDNN_TENSOR_NCHW, n, c, h, w),
                    activation_desc_() {
    } 
    
    /**
     * Run the activation forward method given the data pointers to the GPU memory assicated with
     * the tensor descriptors for x_desc and y_desc.
     *
     **** Inputs
     * x - Tensor class associated with the tensor descriptor x_desc (input).
     * y - Tensor class associated with the tensor descriptor y_desc (output). 
     */ 
    void forward(Tensor<float> x, Tensor<float> y) {
        // Activation Forward.
        CHECK_CUDNN_ERROR(cudnnActivationForward(cudnn_handle_.handle(),
                                                 activation_desc_.desc(), 
                                                 &alpha_,
                                                 x_desc_.desc(),
                                                 x.begin(),
                                                 &beta_,
                                                 y_desc_.desc(),
                                                 y.begin()));
    }
    
    /**
     * Run the activation backward method given the data pointers to teh GPU memory associated with
     * the tensor descriptors for the input/output, and their relative differential. 
     *
     **** Inputs
     * y - Tensor class associated with the previously initialized y_desc (input).
     * dY - Tensor class associated with the previously initialized differential of y_desc.
     * x - Tensor class associated with the previously initialized x_desc (output).
     * dX - Tensor class associated with the previously initialized differential of x_desc.
     */
    void backward(Tensor<float> y, Tensor<float> dY, Tensor<float> x, Tensor<float> dX) {
        // Activation Backward.
        CHECK_CUDNN_ERROR(cudnnActivationBackward(cudnn_handle_.handle(),
                                                  activation_desc_.desc(),
                                                  &alpha_,
                                                  y_desc_.desc(),
                                                  y.begin(),
                                                  y_desc_.desc(),
                                                  dY.begin(),
                                                  x_desc_.desc(),
                                                  x.begin(),
                                                  &beta_,
                                                  x_desc_.desc(),
                                                  dX.begin()));
    }
};

#endif // DELLVE_CUDNNACTIVATION_H_

