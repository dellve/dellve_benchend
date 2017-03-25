/**
 * CuDNN Helper functions and classes to simplify and wrap accesses to the cuDNN APIs
 * available through NVIDIA. 
 */

#ifndef DELLVE_CUDNN_HELPER_H_
#define DELLVE_CUDNN_HELPER_H_

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <cudnn.h>

/**
 * Throw an error if CUDNN does not return a success status.
 */
void throw_cudnn_err(cudnnStatus_t status, int line, const char* filename) {
    if (status != CUDNN_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "CUDNN failure: " << cudnnGetErrorString(status) <<
              " in " << filename << " at line: " << line << std::endl;
        throw std::runtime_error(ss.str());
    }
}

/**
 * Simple definition to wrap CUDNN calls. If the CUDNN call is not successful, throw an
 * error. Otherwise return and continue. 
 */
#define CHECK_CUDNN_ERROR(status) throw_cudnn_err(status, __LINE__, __FILE__)

/**
 * Cudnn handle is a pointer to an opaque structure holding the cuDNN library context - NVIDIA.
 * This class is borrowed from the authors of DeepBench to wrap around this handle.
 *
 * Allows user to create a handle for a specified GPU and wraps creation/deleter around a simple
 * class.
 */
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
	    cudaSetDevice(1);  // Default GPU: 1
        CHECK_CUDNN_ERROR(cudnnCreate(handle_.get()));
    }

    CudnnHandle(int device) : handle_(new cudnnHandle_t, CudnnHandleDeleter()) {
    	cudaSetDevice(device);
        CHECK_CUDNN_ERROR(cudnnCreate(handle_.get()));
    }

    cudnnHandle_t handle() const { return *handle_; };
};

/**
 * Wrapper around cuDNN's tensor descriptor type. The tensor descriptor type is a pointer to an
 * opaque structure holding the description of a generic n-D dataset. 
 *
 * This class allows user to create a descriptor with a simple class call function. Inside, there
 * are cudnn calls to create this tensor. 
 */
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
            type = CUDNN_DATA_FLOAT;  // Currently only supports float
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

/**
 * Similar to the Tensor Descriptor type, this class wraps around cudnn's Filter Descriptor type.
 * This descriptor is a pointer to an opaque structure holding the description of a filter dataset
 * - NVIDIA.
 *
 * Wraps around creation, setting, and deletion of a filter descriptor.
 */
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
            type = CUDNN_DATA_FLOAT;  // Currently only supports float data types
        else
            throw std::runtime_error("Unknown type");

        cudnnFilterDescriptor_t * desc = new cudnnFilterDescriptor_t;
        CHECK_CUDNN_ERROR(cudnnCreateFilterDescriptor(desc));
        CHECK_CUDNN_ERROR(cudnnSetFilter4dDescriptor(*desc, type, tensor_format, k, c, h, w));

        desc_.reset(desc, FilterDescriptor4dDeleter());
    }

    cudnnFilterDescriptor_t desc() const { return *desc_; }
};

/**
 * Similar to the Tensor Descriptor type, this class wraps around cudnn's Convolution Descriptor
 * type. 
 *
 * Currently creates a convolution 2d Descriptor with upscale x and y of value 1 and convolution
 * mode of CUDNN_CONVOLUTION.
 */
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

/**
 * Similar to the Tensor Descriptor type, this class wraps around cudnn's Pooling Descriptor
 * type. 
 *
 * Allows user to specify what mode of pooling to run but hard coded to not propagate through
 * NAN values.
 */
class PoolingDescriptor {
    std::shared_ptr<cudnnPoolingDescriptor_t> desc_;

    struct PoolingDescriptorDeleter {
        void operator()(cudnnPoolingDescriptor_t * desc) {
            std::cout << "Destrying PLS NO" << std::endl;
            cudnnDestroyPoolingDescriptor(*desc);
            delete desc;
        }
    };
public:
    PoolingDescriptor(int win_h, int win_w, int pad_h, int pad_w, int hstride, int wstride,
                      cudnnPoolingMode_t mode) :
        desc_(new cudnnPoolingDescriptor_t, PoolingDescriptorDeleter()) {
        CHECK_CUDNN_ERROR(cudnnCreatePoolingDescriptor(desc_.get()));
        CHECK_CUDNN_ERROR(cudnnSetPooling2dDescriptor(*desc_,
                                                      mode,
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

/**
 * Similar to the Tensor Descriptor type,t his class wraps around cudnn's Activation Descriptor
 * type. 
 *
 * Currently uses ACITVATION_TANH method and hard coded to not propagate through NAN values.
 */
class ActivationDescriptor {
    std::shared_ptr<cudnnActivationDescriptor_t> desc_;

    struct ActivationDescriptorDeleter {
        void operator()(cudnnActivationDescriptor_t * desc) {
            cudnnDestroyActivationDescriptor(*desc);
            delete desc;
        }
    };
public:
    ActivationDescriptor(void) :
        desc_(new cudnnActivationDescriptor_t, ActivationDescriptorDeleter()) {
        // TODO: Extend Activastion to support more modes and nan option
        CHECK_CUDNN_ERROR(cudnnCreateActivationDescriptor(desc_.get()));
        CHECK_CUDNN_ERROR(cudnnSetActivationDescriptor(*desc_,
                                                       CUDNN_ACTIVATION_TANH,
                                                       CUDNN_NOT_PROPAGATE_NAN,
                                                       0.0));
    }

    cudnnActivationDescriptor_t desc() const { return *desc_; };
};

#endif // DELLVE_CUDNN_HELPER_H_
