#ifndef DELLVE_CUDNN_POOL_DRIVER_H_
#define DELLVE_CUDNN_POOL_DRIVER_H_

#include <tuple>
#include <chrono>

#include <cuda.h>
#include <curand.h>

#include <unistd.h>

#include "cudnn_pool.hpp"
#include "cudnn_problem_set.hpp"
#include "tensor.hpp"

enum class CudnnPoolMethod { FORWARD, BACKWARD };

class CudnnPoolDriver {
private:
    int num_repeats_;
    curandGenerator_t curand_gen_;
    CudnnPoolMethod method_;
    CudnnPoolProblemSet problems_;

    int n_, w_, h_, c_; // Input parameters
    int win_w_, win_h_; // Window Parameters
    int pad_w_, pad_h_; // Padding
    int wstride_, hstride_; // Stride
    
    std::vector<int> gpus_;
public:
    CudnnPoolDriver(CudnnPoolMethod method, CudnnPoolProblemSet problems, int numRuns, std::vector<int> gpus) :
                    num_repeats_(numRuns),
                    gpus_(gpus),
                    method_(method),
                    problems_(problems) {
        cudaFree(0);
        curandCreateGenerator(&curand_gen_, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(curand_gen_, 42ULL);
    }

    int run(int problemNumber) {
        CudnnPool pool = createCudnnPool(problemNumber, gpus_[0]);

        switch(method_) {
            case CudnnPoolMethod::FORWARD:
                return forward(pool);
            case CudnnPoolMethod::BACKWARD:
                return backward(pool);
            default:
                return 0;
        }
    }

private: 
    CudnnPool createCudnnPool(int problemNumber, int deviceNumber) {
        std::tie(w_, h_, c_, n_, win_w_, win_h_, pad_w_, pad_h_, wstride_, hstride_) = problems_.get(problemNumber);
        return CudnnPool(w_, h_, c_, n_, win_w_, win_h_, pad_w_, pad_h_, wstride_, hstride_, deviceNumber);
    }

    int forward(CudnnPool &pool) {
        auto input = TensorCreate::rand(std::vector<int>{w_, h_, c_, n_}, curand_gen_);
        auto output = TensorCreate::zeros(pool.get_output_dims());

        // Warm Up
        pool.forward(input, output);
        cudaDeviceSynchronize();

        auto start = std::chrono::steady_clock::now();

        for(int i = 0; i < num_repeats_; ++i) {
            pool.forward(input, output);
        }

        cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();
        
        int fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats_); 
        return fwd_time;
    }

    int backward(CudnnPool &pool) {
        auto input = TensorCreate::rand(std::vector<int>{w_, h_, c_, n_}, curand_gen_); 
        auto dY = TensorCreate::rand(std::vector<int>{w_, h_, c_, n_}, curand_gen_); 
        auto output = TensorCreate::zeros(pool.get_output_dims());
        auto dX = TensorCreate::zeros(std::vector<int>{w_, h_, c_, n_});
        
        /* Start Forward
        pool.forward(input, output);
        cudaDeviceSynchronize();
        */
        
        // Warmup
        pool.backward(input, dY, output, dX);
        cudaDeviceSynchronize();

        auto start = std::chrono::steady_clock::now();

        for(int i = 0; i < num_repeats_; ++i) {
            pool.backward(input, dY, output, dX);
        }

        cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();
        
        int bwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats_); 
        return bwd_time;
    }  
};

#endif //DELLVE_CUDNN_POOL_DRIVER_H_
