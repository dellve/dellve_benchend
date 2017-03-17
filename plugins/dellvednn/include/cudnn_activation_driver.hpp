#ifndef DELLVE_CUDNN_ACTIVATION_DRIVER_H_
#define DELLVE_CUDNN_ACTIVATION_DRIVER_H_

#include <tuple>
#include <chrono>

#include <cuda.h>
#include <curand.h>

#include <unistd.h>

#include "cudnn_activation.hpp"
#include "cudnn_problem_set.hpp"
#include "tensor.hpp"

enum class CudnnActivationMethod { FORWARD, BACKWARD };

class CudnnActivationDriver {
private:
    int num_repeats_;
    curandGenerator_t curand_gen_;
    CudnnActivationMethod method_;
    CudnnActivationProblemSet problems_;
    
    int n_, w_, h_, c_; // Input Parameters

    std::vector<int> gpus_;
public:
    CudnnActivationDriver(CudnnActivationMethod method, CudnnActivationProblemSet problems, int numRuns,
                          std::vector<int> gpus) :
                          num_repeats_(numRuns),
                          gpus_(gpus),
                          method_(method),
                          problems_(problems) {
        cudaFree(0);
        curandCreateGenerator(&curand_gen_, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(curand_gen_, 42ULL);
    }  

    int run(int problemNumber) {
        CudnnActivation activation = createCudnnActivation(problemNumber, gpus_[0]);

        switch(method_) {
            case CudnnActivationMethod::FORWARD:
                return forward(activation);
            case CudnnActivationMethod::BACKWARD:
                return backward(activation);
            default:
                return 0;
        }
    }
private:
    CudnnActivation createCudnnActivation(int problemNumber, int deviceNumber) {
        std::tie(w_, h_, c_, n_) = problems_.get(problemNumber);
        return CudnnActivation(w_, h_, c_, n_, deviceNumber);
    };

    int forward(CudnnActivation &activation) {
        auto input = TensorCreate::rand(std::vector<int>{w_, h_, c_, n_}, curand_gen_);
        auto output = TensorCreate::zeros(std::vector<int>{w_, h_, c_, n_});

        // Warm Up
        activation.forward(input, output);
        cudaDeviceSynchronize();

        auto start = std::chrono::steady_clock::now();

        for(int i = 0; i < num_repeats_; ++i) {
            activation.forward(input, output);
        }

        cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();
        
        int fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats_); 
        return fwd_time;
    }

    int backward(CudnnActivation &activation) {
        auto input = TensorCreate::rand(std::vector<int>{w_, h_, c_, n_}, curand_gen_);
        auto dY = TensorCreate::rand(std::vector<int>{w_, h_, c_, n_}, curand_gen_);
        auto output = TensorCreate::zeros(std::vector<int>{w_,h_,c_,n_});
        auto dX = TensorCreate::zeros(std::vector<int>{w_, h_, c_, n_});

        // Warm Up
        activation.backward(input, dY, output, dX);
        cudaDeviceSynchronize();

        auto start = std::chrono::steady_clock::now();

        for(int i = 0; i < num_repeats_; ++i) {
            activation.backward(input, dY, output, dX);
        }

        cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();
        
        int bwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats_); 
        return bwd_time;
    }
};

#endif // DELLVE_CUDNN_ACTIVATION_DRIVER_H_

