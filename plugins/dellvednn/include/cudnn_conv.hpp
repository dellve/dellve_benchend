#ifndef DELLVE_CUDNNCONV_H_
#define DELLVE_CUDNNCONV_H_

#include <vector>

#include <cudnn.h>

#include "cudnn_helper.hpp"
#include "tensor.hpp"

class CudnnConv {
private:
    CudnnHandle cudnn_handle_;
    ConvolutionDescriptor conv_desc_;

    TensorDescriptor4d<float> x_desc_;
    TensorDescriptor4d<float> y_desc_;

    FilterDescriptor4d<float> w_desc_;

    std::vector<int> output_dims_;

    size_t fwd_workspace_size_;
    size_t bwd_data_workspace_size_;
    size_t bwd_filter_workspace_size_;

    Tensor<float> fwd_workspace_;
    Tensor<float> bwd_data_workspace_;
    Tensor<float> bwd_filter_workspace_;

    cudnnConvolutionFwdAlgo_t fwd_algo_;
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;

    const float alpha_ = 1.f;
    const float beta_  = 0.f;

public:
    CudnnConv(int w, int h, int c, int n, int k, int r, int s,
              int pad_w, int pad_h, int wstride, int hstride) :
    
              cudnn_handle_(),
              x_desc_(CUDNN_TENSOR_NCHW, n, c, h, w),
              w_desc_(CUDNN_TENSOR_NCHW, k, c, r, s),
              conv_desc_(pad_h, pad_w, hstride, wstride) {
    
        // Get output dimensions
        int out_h, out_w, out_c, out_n;
        CHECK_CUDNN_ERROR(cudnnGetConvolution2dForwardOutputDim(conv_desc_.desc(),
                                                                x_desc_.desc(),
                                                                w_desc_.desc(),
                                                                &out_n,
                                                                &out_c,
                                                                &out_h,
                                                                &out_w));
        y_desc_ = TensorDescriptor4d<float>(CUDNN_TENSOR_NCHW, out_n, out_c, out_h, out_w);
        output_dims_ = {out_w, out_h, out_c, out_n};
    }


    void initForward(void) {
        // Pick forward convolution algorithm
        CHECK_CUDNN_ERROR(cudnnGetConvolutionForwardAlgorithm(cudnn_handle_.handle(),
                                                              x_desc_.desc(),
                                                              w_desc_.desc(),
                                                              conv_desc_.desc(),
                                                              y_desc_.desc(),
                                                              CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                              0,
                                                              &fwd_algo_));
    
        // Set fwd workspace size
        CHECK_CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle_.handle(),
                                                                  x_desc_.desc(),
                                                                  w_desc_.desc(),
                                                                  conv_desc_.desc(),
                                                                  y_desc_.desc(),
                                                                  fwd_algo_,
                                                                  &fwd_workspace_size_));
    
        fwd_workspace_ = TensorCreate::zeros(std::vector<int>{static_cast<int>(fwd_workspace_size_ / sizeof(float)), 1});
    }

    void initBackwardFilter(void){
        initForward();
    
        // Pick backward convolution algorithm
        CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle_.handle(),
                                                                     x_desc_.desc(),
                                                                     y_desc_.desc(),
                                                                     conv_desc_.desc(),
                                                                     w_desc_.desc(),
                                                                     CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                                                     0,
                                                                     &bwd_filter_algo_));

        // Backward params workspace
        CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle_.handle(),
                                                                         x_desc_.desc(),
                                                                         y_desc_.desc(),
                                                                         conv_desc_.desc(),
                                                                         w_desc_.desc(),
                                                                         bwd_filter_algo_,
                                                                         &bwd_filter_workspace_size_));



        bwd_filter_workspace_ = TensorCreate::zeros(std::vector<int>{static_cast<int>(bwd_filter_workspace_size_ / sizeof(float)), 1});
    }

    void initBackwardData(void) {
        // Pick backward wrt inputs convolution algorithm
        CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle_.handle(),
                                                                   w_desc_.desc(),
                                                                   y_desc_.desc(),
                                                                   conv_desc_.desc(),
                                                                   x_desc_.desc(),
                                                                   CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                                                   0,
                                                                   &bwd_data_algo_));

        CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle_.handle(),
                                                                       w_desc_.desc(),
                                                                       y_desc_.desc(),
                                                                       conv_desc_.desc(),
                                                                       x_desc_.desc(),
                                                                       bwd_data_algo_,
                                                                       &bwd_data_workspace_size_));

        bwd_data_workspace_ = TensorCreate::zeros(std::vector<int>{static_cast<int>(bwd_data_workspace_size_ / sizeof(float)), 1}); 
    }

    void forward(Tensor<float> x, Tensor<float> filter, Tensor<float> y) {
        // Convolution forward.
        CHECK_CUDNN_ERROR(cudnnConvolutionForward(cudnn_handle_.handle(),
                                                  &alpha_,
                                                  x_desc_.desc(),
                                                  x.begin(),
                                                  w_desc_.desc(),
                                                  filter.begin(),
                                                  conv_desc_.desc(),
                                                  fwd_algo_,
                                                  fwd_workspace_.begin(),
                                                  fwd_workspace_size_,
                                                  &beta_,
                                                  y_desc_.desc(),
                                                  y.begin()));
    }

    void backwardFilter(Tensor<float> x, Tensor<float> delta, Tensor<float> dW) {
        // Convolution Backward Filter
        CHECK_CUDNN_ERROR(cudnnConvolutionBackwardFilter(cudnn_handle_.handle(),
                                                         &alpha_,
                                                         x_desc_.desc(),
                                                         x.begin(),
                                                         y_desc_.desc(),
                                                         delta.begin(),
                                                         conv_desc_.desc(),
                                                         bwd_filter_algo_,
                                                         bwd_filter_workspace_.begin(),
                                                         bwd_filter_workspace_size_,
                                                         &beta_,
                                                         w_desc_.desc(),
                                                         dW.begin()));
    }


    void backwardData(Tensor<float> filter, Tensor<float> delta, Tensor<float> dX) {
        // Convolution Backward Data
        CHECK_CUDNN_ERROR(cudnnConvolutionBackwardData(cudnn_handle_.handle(),
                                                       &alpha_,
                                                       w_desc_.desc(),
                                                       filter.begin(),
                                                       y_desc_.desc(),
                                                       delta.begin(),
                                                       conv_desc_.desc(),
                                                       bwd_data_algo_,
                                                       bwd_data_workspace_.begin(),
                                                       bwd_data_workspace_size_,
                                                       &beta_,
                                                       x_desc_.desc(),
                                                       dX.begin()));
    }



    std::vector<int> get_output_dims() { 
        return output_dims_; 
    }
};

#endif // DELLVE_CUDNNCONV_H_
