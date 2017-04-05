#ifndef DELLVE_CUDNN_CONVOLUTION_H_
#define DELLVE_CUDNN_CONVOLUTION_H_

#include "dellve_cudnn_benchmark.hpp"

#include "CuDNN/Convolution.hpp"
#include "CuDNN/Handle.hpp"
#include "CuDNN/Status.hpp"
#include "CuDNN/Tensor.hpp"		   
#include "CuDNN/Filter.hpp"
#include "CuDNN/ConvolutionFwdAlgo.hpp"
#include "CuDNN/ConvolutionBwdDataAlgo.hpp"

#include <iostream>

namespace CuDNN {
    namespace Convolution {
		
		template <typename T>
		DELLve::Benchmark forward ( int w, int h, int c, int n, int k, int r, int s, 
			int padW, int padH, int strideW, int strideH ) 
		{
			CuDNN::Handle handle;

			/**
			 * Create convolution input tensor
			 */
		    auto input = CuDNN::Tensor<T>::createNCHW(n, c, h, w);

		    /**
		     * Create convolution filter
		     */
			auto filter = CuDNN::Filter<T>::createNCHW(k, c, r, s);

			/**
			 * Create convolution descriptor
			 */
			auto convDescriptor = CuDNN::ConvolutionDescriptor::create(padH, padW, strideH, strideW);

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

		   	ConvolutionFwdAlgo algorithm;
		   	CuDNN::checkStatus (
		   		cudnnGetConvolutionForwardAlgorithm ( 
		   			handle,
					input.getDescriptor(),
					filter.getDescriptor(),
					convDescriptor,
					output.getDescriptor(),
					CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
					0,
					&algorithm )
		   	);

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
					&CuDNN::Constants::alpha,
					input.getDescriptor(),
					input,
					filter.getDescriptor(),
					filter,
					convDescriptor,
					algorithm,
					workspace,
					workspace.getSize(),
					&CuDNN::Constants::beta,
					output.getDescriptor(),
					output
				);
			};
		}

		template <typename T>
		DELLve::Benchmark backwardData ( int w, int h, int c, int n, int k, int r, int s, 
			int padW, int padH, int strideW, int strideH ) 
		{
			CuDNN::Handle handle;

			/**
			 * Create convolution input tensor
			 */
		    auto input = CuDNN::Tensor<T>::createNCHW(n, c, h, w);

		    /**
		     * Create convolution filter
		     */
			auto filter = CuDNN::Filter<T>::createNCHW(k, c, r, s);

			/**
			 * Create convolution descriptor
			 */
			auto convDescriptor = CuDNN::ConvolutionDescriptor::create(padH, padW, strideH, strideW);

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

		   	ConvolutionBwdDataAlgo algorithm;
		   	CuDNN::checkStatus (
		   		cudnnGetConvolutionBackwardDataAlgorithm (
		   			handle,
					filter.getDescriptor(),
					output.getDescriptor(),
					convDescriptor,
					input.getDescriptor(),
					CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
					0,
					&algorithm )
		   	);

		   	/**
		   	 * Create workspace buffer
		   	 */
		   	auto workspace = CuDNN::Convolution::
		   		createBackwardDataWorkspace<T> ( 
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
				return cudnnConvolutionBackwardData (
					handle,
					&CuDNN::Constants::alpha,
					filter.getDescriptor(),
					filter,
					output.getDescriptor(),
					output,
					convDescriptor,
					algorithm,
					workspace,
					workspace.getSize(),
					&CuDNN::Constants::beta,
					input.getDescriptor(),
					input 
				);
			};
		}

		template <typename T>
		DELLve::Benchmark backwardFilter ( int w, int h, int c, int n, int k, int r, int s, 
			int padW, int padH, int strideW, int strideH ) 
		{
			CuDNN::Handle handle;

			/**
			 * Create convolution input tensor
			 */
		    auto input = CuDNN::Tensor<T>::createNCHW(n, c, h, w);

		    /**
		     * Create convolution filter
		     */
			auto filter = CuDNN::Filter<T>::createNCHW(k, c, r, s);

			/**
			 * Create convolution descriptor
			 */
			auto convDescriptor = CuDNN::ConvolutionDescriptor::create(padH, padW, strideH, strideW);

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

		   	ConvolutionBwdFilterAlgo algorithm;
		   	CuDNN::checkStatus (
		   		cudnnGetConvolutionBackwardFilterAlgorithm(
		   			handle,
		   			input.getDescriptor(),
		   			output.getDescriptor(),
		   			convDescriptor,
		   			filter.getDescriptor(),
		   			CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
		   			0,
					&algorithm )
		   	);

		   	/**
		   	 * Create workspace buffer
		   	 */
		   	auto workspace = CuDNN::Convolution::
		   		createBackwardFilterWorkspace<T> ( 
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
				return cudnnConvolutionBackwardFilter (
					handle,
					&CuDNN::Constants::alpha,
					input.getDescriptor(),
					input,
					output.getDescriptor(),
					output,
					convDescriptor,
					algorithm,
					workspace,
					workspace.getSize(),
					&CuDNN::Constants::beta,
					filter.getDescriptor(),
					filter
				);
			};
		}
	}
}

#endif // DELLVE_CUDNN_CONVOLUTION_H_

