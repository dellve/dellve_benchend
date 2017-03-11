#ifndef DELLVE_CUDNN_CONV_DRIVER_H_
#define DELLVE_CUDNN_CONV_DRIVER_H_

#include <tuple>
#include <chrono>

#include <cuda.h>
#include <curand.h>

#include "cudnn_conv.hpp"
#include "cudnn_conv_problem_set.hpp"
#include "tensor.hpp"

enum class CudnnConvMethod { FORWARD, BACKWARD_DATA, BACKWARD_FILTER };

class CudnnConvDriver {
private:
    int num_repeats_;
    curandGenerator_t curand_gen_;
    CudnnConvMethod method_;
    CudnnConvProblemSet problems_;

    int k_, c_, r_, s_; // lr parameters
    int n_, w_, h_; // Input parameters
    int pad_w_, pad_h_; // Padding
    int wstride_, hstride_; // Stride
    
public:
    CudnnConvDriver(CudnnConvMethod method, CudnnConvProblemSet problems) :
                    num_repeats_(50), 
                    method_(method),
                    problems_(problems) {
        cudaFree(0);
        //TODO: make num_repeats configurable
        curandCreateGenerator(&curand_gen_, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(curand_gen_, 42ULL);
    }

    int run(int problemNumber) {

        CudnnConv conv = createCudnnConv(problemNumber);

        switch(method_) {
            case CudnnConvMethod::FORWARD:
                conv.initForward();
                return forward(conv);
            case CudnnConvMethod::BACKWARD_DATA:
                conv.initBackwardData();
                return backwardData(conv);
            case CudnnConvMethod::BACKWARD_FILTER:
                conv.initBackwardFilter();
                return backwardFilter(conv);
            default:
                return 0;
        }
    }

private:
    CudnnConv createCudnnConv(int problemNumber) {

        std::tie(w_, h_, c_, n_, k_, r_, s_, pad_w_, pad_h_, wstride_, hstride_) = problems_.get(problemNumber);
        return CudnnConv(w_, h_, c_, n_, k_, r_, s_, pad_w_, pad_h_, wstride_, hstride_);
    }

    int forward(CudnnConv &conv) {
        auto filter = TensorCreate::rand(std::vector<int>{r_, s_, c_, k_}, curand_gen_);
        auto input = TensorCreate::rand(std::vector<int>{w_, h_, c_, n_}, curand_gen_);
        auto output = TensorCreate::zeros(conv.get_output_dims());
    
	// TODO
        // std::string fwd_algo_s = cnn.get_fwd_algo_string();
    
        //Warm up
        conv.forward(input, filter, output);
        cudaDeviceSynchronize();

        auto start = std::chrono::steady_clock::now();
    
        for (int i = 0; i < num_repeats_; ++i) {
            conv.forward(input, filter, output);
        }
    
        cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();
        int fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats_); 

        return fwd_time;
    }

    int backwardFilter(CudnnConv &conv){
        auto input = TensorCreate::rand(std::vector<int>{w_, h_, c_, n_}, curand_gen_);
        auto delta = TensorCreate::rand(conv.get_output_dims(), curand_gen_);
        auto dW = TensorCreate::zeros(std::vector<int>{r_, s_, c_, k_});

        conv.backwardFilter(input, delta, dW); // Warmup
        cudaDeviceSynchronize(); 

        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < num_repeats_; ++i) {
            conv.backwardFilter(input, delta, dW);
        }

        cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();
        int bwd_filter_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats_);
        
        return bwd_filter_time;
        
    }

    int backwardData(CudnnConv &conv) {
        auto filter = TensorCreate::rand(std::vector<int>{r_, s_, c_, k_}, curand_gen_);
        auto delta = TensorCreate::rand(conv.get_output_dims(), curand_gen_);
        auto dX = TensorCreate::zeros(std::vector<int>{w_, h_, c_, n_});
       
	    conv.backwardData(filter, delta, dX);
        cudaDeviceSynchronize();

        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < num_repeats_; ++i) {
            // Backward pass wrt weights
            conv.backwardData(filter, delta, dX);
        }

        cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();
        int bwd_data_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats_);
        
        return bwd_data_time;
    }
};

#endif // DELLVE_CUDNN_CONV_DRIVER_H_
