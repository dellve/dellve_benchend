#ifndef DELLVE_CUDNN_HELPER_H_
#define DELLVE_CUDNN_HELPER_H_

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <cudnn.h>

void throw_cudnn_err(cudnnStatus_t status, int line, const char* filename) {
    if (status != CUDNN_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "CUDNN failure: " << cudnnGetErrorString(status) <<
              " in " << filename << " at line: " << line << std::endl;
        throw std::runtime_error(ss.str());
    }
}

#define CHECK_CUDNN_ERROR(status) throw_cudnn_err(status, __LINE__, __FILE__)


class CudnnHandle {
    std::shared_ptr<cudnnHandle_t> handle_;

    struct CudnnHandleDeleter {
        void operator()(cudnnHandle_t * handle) {
            cudnnDestroy(*handle);
            delete handle;
        }
    };

public:
    CudnnHandle() : handle_(new cudnnHandle_t, CudnnHandleDeleter()) {
	cudaSetDevice(1);
        CHECK_CUDNN_ERROR(cudnnCreate(handle_.get()));
    }

    CudnnHandle(int device) : handle_(new cudnnHandle_t, CudnnHandleDeleter()) {
	cudaSetDevice(device);
        CHECK_CUDNN_ERROR(cudnnCreate(handle_.get()));
    }

    cudnnHandle_t handle() const { return *handle_; };
};

template<typename T>
class TensorDescriptor4d {
    std::shared_ptr<cudnnTensorDescriptor_t> desc_;

    struct TensorDescriptor4dDeleter {
        void operator()(cudnnTensorDescriptor_t * desc) {
            cudnnDestroyTensorDescriptor(*desc);
            delete desc;
        }
    };

public:

    TensorDescriptor4d() {}
    TensorDescriptor4d(const cudnnTensorFormat_t tensor_format,
                       const int n, const int c, const int h, const int w) {
        cudnnDataType_t type;
        if (std::is_same<T, float>::value)
            type = CUDNN_DATA_FLOAT;
        else
            throw std::runtime_error("Unknown type");

        cudnnTensorDescriptor_t * desc = new cudnnTensorDescriptor_t;
        CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(desc));
        CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(*desc,
                                                     tensor_format,
                                                     type,
                                                     n,
                                                     c,
                                                     h,
                                                     w));

        desc_.reset(desc, TensorDescriptor4dDeleter());
    }

    cudnnTensorDescriptor_t desc() const { return *desc_; }

};

template<typename T>
class FilterDescriptor4d {
    std::shared_ptr<cudnnFilterDescriptor_t> desc_;

    struct FilterDescriptor4dDeleter {
        void operator()(cudnnFilterDescriptor_t * desc) {
            cudnnDestroyFilterDescriptor(*desc);
            delete desc;
        }
    };

public:
    FilterDescriptor4d(const cudnnTensorFormat_t tensor_format,
                       int k, int c, int h, int w) {
        cudnnDataType_t type;
        if (std::is_same<T, float>::value)
            type = CUDNN_DATA_FLOAT;
        else
            throw std::runtime_error("Unknown type");

        cudnnFilterDescriptor_t * desc = new cudnnFilterDescriptor_t;
        CHECK_CUDNN_ERROR(cudnnCreateFilterDescriptor(desc));
        CHECK_CUDNN_ERROR(cudnnSetFilter4dDescriptor(*desc, type, tensor_format, k, c, h, w));

        desc_.reset(desc, FilterDescriptor4dDeleter());
    }

    cudnnFilterDescriptor_t desc() const { return *desc_; }

};

class ConvolutionDescriptor {
    std::shared_ptr<cudnnConvolutionDescriptor_t> desc_;

    struct ConvolutionDescriptorDeleter {
        void operator()(cudnnConvolutionDescriptor_t * desc) {
            cudnnDestroyConvolutionDescriptor(*desc);
            delete desc;
        }
    };
public:
    ConvolutionDescriptor(int pad_h, int pad_w, int hstride, int wstride) :
        desc_(new cudnnConvolutionDescriptor_t, ConvolutionDescriptorDeleter()) {

        CHECK_CUDNN_ERROR(cudnnCreateConvolutionDescriptor(desc_.get()));
        CHECK_CUDNN_ERROR(cudnnSetConvolution2dDescriptor(*desc_,
                                                          pad_h,
                                                          pad_w,
                                                          hstride,
                                                          wstride,
                                                          1,
                                                          1,
                                                          CUDNN_CONVOLUTION));
    }

    cudnnConvolutionDescriptor_t desc() const { return *desc_; };

};

class PoolingDescriptor {
    std::shared_ptr<cudnnPoolingDescriptor_t> desc_;

    struct PoolingDescriptorDeleter {
        void operator()(cudnnPoolingDescriptor_t * desc) {
            cudnnDestroyPoolingDescriptor(*desc);
            delete desc;
        }
    };
public:

    PoolingDescriptor(int win_h, int win_w, int pad_h, int pad_w, int hstride, int wstride) :
        desc_(new cudnnPoolingDescriptor_t, PoolingDescriptorDeleter()) {
        // TODO: Extend Pooling to support more algorithms (currently hardcoded to MAX    
        CHECK_CUDNN_ERROR(cudnnCreatePoolingDescriptor(desc_.get()));
        CHECK_CUDNN_ERROR(cudnnSetPooling2dDescriptor(*desc_,
                                                      CUDNN_POOLING_MAX,
                                                      CUDNN_NOT_PROPAGATE_NAN,
                                                      win_h,
                                                      win_w,
                                                      pad_h,
                                                      pad_w,
                                                      hstride,
                                                      wstride));
    }

    cudnnPoolingDescriptor_t desc() const { return *desc_; };

};

#endif // DELLVE_CUDNN_HELPER_H_
