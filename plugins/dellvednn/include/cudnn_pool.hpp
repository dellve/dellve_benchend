#ifndef DELLVE_CUDNNPOOL_H_
#define DELLVE_CUDNNPOOL_H_

#include <vector>

#include <cudnn.h>

#include "cudnn_helper.hpp"
#include "tensor.hpp"

enum class CudnnPoolAlgorithm { MAX, AVGPAD, AVGNOPAD };

/**
 * Wraps around cuDNN calls for all of the pooling methods.
 *
 * Initializes the descriptors required for running pooling when the instance is created.
 * Also provides the methods that abstract cudnn pooling forward and backward.
 */
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
    /**
     * Initialize the x descriptor of pooling based on the parameters of the problem set. Then,
     * initialize the pooling descriptor based on the algorithm desired by the user after converting
     * to the cudnn data structure. Finally, intiailize the output descriptor based on the cudnn
     * forward output dimensions.
     */ 
    CudnnPool(int w, int h, int c, int n, int win_w, int win_h,
              int pad_w, int pad_h, int wstride, int hstride, int device,
              CudnnPoolAlgorithm algorithm) : 
              cudnn_handle_(device),
              x_desc_(CUDNN_TENSOR_NCHW, n, c, h, w),
              pool_desc_(win_h, win_w, pad_h, pad_w, hstride, wstride, convertAlgorithm(algorithm)) {
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
    
    /**
     * Run the pooling forward method given the data pointers to ghe GPU memory associated with the
     * tensor descriptors x_desc and y_desc.
     *
     **** Inputs
     * x - Tensor class associated with the tensor descriptor x_desc (input).
     * y - Tensor class associated with the tensor descritpor y_desc (output).
     */
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

    /**
     * Run the pooling backward method given the data pointers to the GPU memory associaterd with the
     * tensor descriptors x_desc, ydesc, and the differential of the relative input and output.
     *
     **** Inputs
     * y - Tensor class associated with the tensor descriptor y_desc (input).
     * dY - Tensor class associated with the differential of the input.
     * x - Tensor class associated with the tensor descriptor x_desc (output).
     * dX - Tensor class associated with the differential of the output.
     */
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
    
    /**
     * Returns the dimensions of the output tensor descriptor. 
     */    
    std::vector<int> get_output_dims() { 
        return output_dims_; 
    }
private:
    cudnnPoolingMode_t convertAlgorithm(CudnnPoolAlgorithm algorithm) {
        switch(algorithm) {
            case CudnnPoolAlgorithm::MAX:
                return CUDNN_POOLING_MAX;
            case CudnnPoolAlgorithm::AVGPAD:
                return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
            case CudnnPoolAlgorithm::AVGNOPAD:
                return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
            default:
                return CUDNN_POOLING_MAX;
        }
    }
};

#endif // DELLVE_CUDNNPOOL_H_
