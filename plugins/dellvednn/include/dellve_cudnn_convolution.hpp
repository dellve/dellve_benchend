#ifndef DELLVE_CUDNN_CONVOLUTION_H_
#define DELLVE_CUDNN_CONVOLUTION_H_

#include "dellve_cudnn_benchmark.hpp"

#include "CuDNN/Convolution.hpp"
#include "CuDNN/Handle.hpp"
#include "CuDNN/Status.hpp"
#include "CuDNN/Tensor.hpp"		   
#include "CuDNN/Filter.hpp"

namespace CuDNN {
    namespace Convolution {
		
		template <typename T>
		DELLve::Benchmark forward ( int w, int h, int c, int n, int k, int r, int s, 
			int padW, int padH, int strideW, int strideH ) 
		{
			CuDNN::Handle handle;

			static const T alpha = 1.0;
			static const T beta = 0.0;

			/**
			 * Create convolution input tensor
			 */
		    auto input 	= CuDNN::Tensor<T>::createNCHW(n, c, h, w);

		    /**
		     * Create convolution filter
		     */
			auto filter = CuDNN::Filter<T>::createNCHW(r, s, c, k);

			/**
			 * Create convolution descriptor
			 */
			auto convDescriptor = CuDNN::ConvolutionDescriptor::create(padW, padH, strideW, strideH);

			/**
			 * Calculate convolution output dimensions
			 */
		   	std::tuple<int,int,int,int> outputDims; // NCHW tuple of output dimensions
		   	CuDNN::checkStatus (
		   		cudnnGetConvolution2dForwardOutputDim (
		   			convDescriptor,
		   			input.getDescriptor(),
		   			filter.getDescriptor(),
		   			&std::get<0>(outputDims),	// N
		   			&std::get<1>(outputDims),	// C
		   			&std::get<2>(outputDims),	// H
		   			&std::get<3>(outputDims)	// W
		   		)
		   	);

		   	/**
		   	 * Create output tensor
		   	 */
		   	auto output = CuDNN::Tensor<T>::createNCHW(outputDims);

		   	auto algorithm = CUDNN_CONVOLUTION_FWD_ALGO_FFT;

		   	/**
		   	 * Create workspace buffer
		   	 */
		   	auto workspace = CuDNN::Convolution::
		   		createForwardWorkspace<T> ( 
		   			handle, 		
			   		input.getDescriptor(), 
			   		filter.getDescriptor(), 
			   		convDescriptor, 
			   		output.getDescriptor(), 
			   		algorithm 
			);

		   	/**
		   	 * Retun new benchmark
		   	 */
			return [=]() {
				return cudnnConvolutionForward (
					handle,
					&alpha,
					input.getDescriptor(),
					input,
					filter.getDescriptor(),
					filter,
					convDescriptor,
					algorithm,
					workspace,
					workspace.getSize(),
					&beta,
					output.getDescriptor(),
					output
				);
			};
		}
	}
}

#endif // DELLVE_CUDNN_CONVOLUTION_H_

