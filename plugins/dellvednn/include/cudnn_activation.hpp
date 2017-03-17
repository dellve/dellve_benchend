#ifndef DELLVE_CUDNNACTIVATION_H_
#define DELLVE_CUDNNACTIVATION_H_

#include <vector>

#include <cudnn.h>

#include "cudnn_helper.hpp"
#include "tensor.hpp"

class CudnnActivation {
private:
    CudnnHandle cudnn_handle_;
    ActivationDescriptor activation_desc_;

    TensorDescriptor4d<float> x_desc_;
    TensorDescriptor4d<float> y_desc_;

    const float alpha_ = 1.f;
    const float beta_ = 0.f;
public:
    // TODO: Extend activation_descriptor in cudnn_helper
    CudnnActivation(int w, int h, int c, int n, int device) : 
                    cudnn_handle_(device),
                    x_desc_(CUDNN_TENSOR_NCHW, n, c, h, w),
                    y_desc_(CUDNN_TENSOR_NCHW, n, c, h, w),
                    activation_desc_() {
    } 
    
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

