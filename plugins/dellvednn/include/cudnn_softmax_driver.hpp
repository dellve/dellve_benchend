#ifndef DELLVE_CUDNN_SOFTMAX_DRIVER_H_
#define DELLVE_CUDNN_SOFTMAX_DRIVER_H_

#include <tuple>
#include <chrono>

#include <cuda.h>
#include <curand.h>

#include <unistd.h>

#include "cudnn_softmax.hpp"
#include "cudnn_problem_set.hpp"
#include "tensor.hpp"

enum class CudnnSoftmaxForm { FORWARD_FAST, FORWARD_ACCURATE, FORWARD_LOG, BACKWARD_FAST, 
                                   BACKWARD_ACCURATE, BACKWARD_LOG };
enum class CudnnSoftmaxMethod { FORWARD, BACKWARD };

/**
 * Driver class to interface to the CudnnSoftmax class. 
 *
 * Allows user to pass in the correlated problem set to the softmax methods, number of
 * runs to average, the gpu to run on, and the softmax method.
 *
 * Currently supports 2 methods as seen above:
 * Forward Softmax
 * Backward Softmax
 *
 * + 3 Algorithms for each:
 * Fast
 * Accurate
 * Log
 *
 * All of these methods use the same problem sets but would have different impact and results.
 */
class CudnnSoftmaxDriver {
private:
    int num_repeats_;
    curandGenerator_t curand_gen_;
    CudnnSoftmaxMethod method_;
    CudnnSoftmaxProblemSet problems_;
    CudnnSoftmaxAlgorithm algorithm_;

    int n_, w_, h_, c_; // Input parameters
    
    std::vector<int> gpus_;
public:
    /**
     * Set up all the instances of the variables needed for this class and setup the curand generator
     * which will be later used to generate random data to run the softmax methods through.
     * 
     * Also, converts the form to the correlated method and algorithm to run softmax.
     */
    CudnnSoftmaxDriver(CudnnSoftmaxForm form, CudnnSoftmaxProblemSet problems, int numRuns, 
                       std::vector<int> gpus) :
                       num_repeats_(numRuns),
                       gpus_(gpus),
                       problems_(problems) {
        cudaFree(0);
        curandCreateGenerator(&curand_gen_, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(curand_gen_, 42ULL);
        convertForm(form);
    }

    /**
     * Run softmax with the method and number of runs specified in initialization using the problemSet
     * defined in the initalization with the index provided in this function as input. 
     */
    int run(int problemNumber) {
        CudnnSoftmax softmax = createCudnnSoftmax(problemNumber, gpus_[0]);

        switch(method_) {
            case CudnnSoftmaxMethod::FORWARD:
                return forward(softmax);
            case CudnnSoftmaxMethod::BACKWARD:
                return backward(softmax);
            default:
                return 0;
        }
    }

private:

    /**
     * Convert CudnnSoftmaxForm to CudnnSoftmaxMethod and CudnnSoftmaxAlgorithm. 
     */
    void convertForm(CudnnSoftmaxForm form) {
        switch (form) {
            case CudnnSoftmaxForm::FORWARD_FAST:
                method_ = CudnnSoftmaxMethod::FORWARD;
                algorithm_ = CudnnSoftmaxAlgorithm::FAST; 
                break;
            case CudnnSoftmaxForm::FORWARD_ACCURATE:
                method_ = CudnnSoftmaxMethod::FORWARD;
                algorithm_ = CudnnSoftmaxAlgorithm::ACCURATE; 
                break;
            case CudnnSoftmaxForm::FORWARD_LOG:
                method_ = CudnnSoftmaxMethod::FORWARD;
                algorithm_ = CudnnSoftmaxAlgorithm::LOG; 
                break;
            case CudnnSoftmaxForm::BACKWARD_FAST:
                method_ = CudnnSoftmaxMethod::BACKWARD;
                algorithm_ = CudnnSoftmaxAlgorithm::FAST; 
                break;
            case CudnnSoftmaxForm::BACKWARD_ACCURATE:
                method_ = CudnnSoftmaxMethod::BACKWARD;
                algorithm_ = CudnnSoftmaxAlgorithm::ACCURATE; 
                break;
            case CudnnSoftmaxForm::BACKWARD_LOG:
                method_ = CudnnSoftmaxMethod::BACKWARD;
                algorithm_ = CudnnSoftmaxAlgorithm::LOG; 
                break;
            default:
                method_ = CudnnSoftmaxMethod::FORWARD;
                algorithm_ = CudnnSoftmaxAlgorithm::ACCURATE;
        }
    }

    /**
     * Setup an instance of CudnnSoftmax by unraveling the problemset.
     */ 
    CudnnSoftmax createCudnnSoftmax(int problemNumber, int deviceNumber) {
        std::tie(w_, h_, c_, n_) = problems_.get(problemNumber);
        return CudnnSoftmax(w_, h_, c_, n_, deviceNumber, algorithm_); 
    }

    /**
     * Run Softmax Forward a given number of times and algorithm specified, and average the time that it takes to run.
     */
    int forward(CudnnSoftmax &softmax) {
        auto input = TensorCreate::rand(std::vector<int>{w_, h_, c_, n_}, curand_gen_);
        auto output = TensorCreate::zeros(std::vector<int>{w_,h_,c_,n_});

        // Warm Up
        softmax.forward(input, output);
        cudaDeviceSynchronize();

        auto start = std::chrono::steady_clock::now();

        for(int i = 0; i < num_repeats_; ++i) {
            softmax.forward(input, output);
        }

        cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();
        
        int fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats_); 
        return fwd_time;
    }

    /**
     * Run Softmax Backward a given number of times and algorithm specified, and average the time that it takes to run.
     */
    int backward(CudnnSoftmax &softmax) {
        auto input = TensorCreate::rand(std::vector<int>{w_, h_, c_, n_}, curand_gen_);
        auto dY = TensorCreate::rand(std::vector<int>{w_, h_, c_, n_}, curand_gen_);
        auto output = TensorCreate::zeros(std::vector<int>{w_,h_,c_,n_});

        // Warm Up
        softmax.backward(input, dY, output);
        cudaDeviceSynchronize();

        auto start = std::chrono::steady_clock::now();

        for(int i = 0; i < num_repeats_; ++i) {
            softmax.backward(input, dY, output);
        }

        cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();
        
        int bwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats_); 
        return bwd_time;
    }
};

#endif //DELLVE_CUDNN_SOFTMAX_DRIVER_H_
