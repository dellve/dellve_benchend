#ifndef DELLVE_TENSOR_H_
#define DELLVE_TENSOR_H_

#include <vector>
#include <numeric>
#include <memory>

#include <curand.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

/**
 * Abstracts the usage of tensor within all of the cudnn methods. 
 * Allows to user to specify the dimensions to allocate within the GPU memory and 
 * abstracts the accesses.
 */
template <typename T>
class Tensor {
    std::vector<int> dims_;
    int size_;

    struct deleteCudaPtr {
        void operator()(T *p) const {
            cudaFree(p);
        }
    };

    std::shared_ptr<T> ptr_;

public:
    /**
     * Basic Constructor.
     */
    Tensor() {}

    /**
     * Create a tensor within the GPU memory given the dimensions provided.
     */
    Tensor(std::vector<int> dims) : dims_(dims) {
        T* tmp_ptr;
        size_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());
        cudaMalloc(&tmp_ptr, sizeof(T) * size_);

        ptr_.reset(tmp_ptr, deleteCudaPtr());
    }

    /**
     * Basic access methods.
     */
    T* begin() const { return ptr_.get(); }
    T* end()   const { return ptr_.get() + size_; }
    int size() const { return size_; }
    std::vector<int> dims() const { return dims_; }
};

/**
 * Abstract creation of tensors with specified values
 */
namespace TensorCreate {
    /**
     * Create a tensor by filling in the values provided at each cell.
     */
    Tensor<float> fill(std::vector<int> dims, float val) {
         Tensor<float> tensor(dims);
         thrust::fill(thrust::device_ptr<float>(tensor.begin()),
                      thrust::device_ptr<float>(tensor.end()), val);
         return tensor;
    }
    
    /**
     * Create a tensor by filling in zeros provided at each cell.
     */
    Tensor<float> zeros(std::vector<int> dims) {
        Tensor<float> tensor(dims);
        thrust::fill(thrust::device_ptr<float>(tensor.begin()),
                     thrust::device_ptr<float>(tensor.end()), 0.f);
        return tensor;
    }
    
    /**
     * Create a tensor by filling in random values at each cell.
     */
    Tensor<float> rand(std::vector<int> dims, curandGenerator_t curand_gen) {
        Tensor<float> tensor(dims);
        curandGenerateUniform(curand_gen, tensor.begin(), tensor.size());
        return tensor;
    }
}

#endif // DELLVE_TENSOR_H_
