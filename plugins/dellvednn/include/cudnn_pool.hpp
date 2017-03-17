#ifndef DELLVE_CUDNNPOOL_H_
#define DELLVE_CUDNNPOOL_H_

#include <vector>

#include <cudnn.h>

#include "cudnn_helper.hpp"
#include "tensor.hpp"


class CudnnPool { 
private:
    CudnnHandle cudnn_handle_;
    PoolingDescriptor pool_desc_;

    TensorDescriptor4d<float> x_desc_;
    TensorDescriptor4d<float> y_desc_;

    TensorDescriptor4d<float> dx_desc_;
    TensorDescriptor4d<float> dy_desc_;

    std::vector<int> output_dims_;

    const float alpha_ = 1.f;
    const float beta_ = 0.f;
public:
    // TODO: Extend pool_decriptor for any pooling algorithms, currently hardcoded in cudnn_helper to MAX 
    CudnnPool(int w, int h, int c, int n, int win_w, int win_h,
              int pad_w, int pad_h, int wstride, int hstride, int device) : 

              cudnn_handle_(device),
              x_desc_(CUDNN_TENSOR_NCHW, n, c, h, w),
              pool_desc_(win_h, win_w, pad_h, pad_w, hstride, wstride) {
        int out_h, out_w, out_c, out_n;

        CHECK_CUDNN_ERROR(cudnnGetPooling2dForwardOutputDim(pool_desc_.desc(),
                                                            x_desc_.desc(),
                                                            &out_n,
                                                            &out_c,
                                                            &out_h,
                                                            &out_w));
        y_desc_ = TensorDescriptor4d<float>(CUDNN_TENSOR_NCHW, out_n, out_c, out_h, out_w);
        output_dims_ = {out_w, out_h, out_c, out_n};
    }

    void forward(Tensor<float> x, Tensor<float> y) {
        // Pooling forward.
        CHECK_CUDNN_ERROR(cudnnPoolingForward(cudnn_handle_.handle(),
                                              pool_desc_.desc(),
                                              &alpha_,
                                              x_desc_.desc(),
                                              x.begin(),
                                              &beta_,
                                              y_desc_.desc(),
                                              y.begin()));
    }

    void backward(Tensor<float> y, Tensor<float> dY, Tensor<float> x, Tensor<float> dX) {
        CHECK_CUDNN_ERROR(cudnnPoolingBackward(cudnn_handle_.handle(),
                                               pool_desc_.desc(),
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
                  
    std::vector<int> get_output_dims() { 
        return output_dims_; 
    }
};

#endif // DELLVE_CUDNNPOOL_H_
