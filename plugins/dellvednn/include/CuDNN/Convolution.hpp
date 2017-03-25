#ifndef PYCUDNN_CONVOLUTION_HPP
#define PYCUDNN_CONVOLUTION_HPP

#include <cudnn.h>

#include "Buffer.hpp"
#include "ConvolutionDescriptor.hpp"
#include "ConvolutionFwdAlgo.hpp"
#include "ConvolutionBwdDataAlgo.hpp"
#include "ConvolutionBwdFilterAlgo.hpp"
#include "FilterDescriptor.hpp"
#include "Status.hpp"
#include "TensorDescriptor.hpp"

namespace CuDNN {

	namespace Convolution {

		template <typename T>
		Buffer<T> createForwardWorkspace (
			Handle handle, 
			TensorDescriptor inputDescriptor, 
			FilterDescriptor<T> filterDescriptor,
			ConvolutionDescriptor convolutionDescritor, 
			TensorDescriptor outputDescriptor,
			ConvolutionFwdAlgo algorithm ) 
		{
    		size_t workspaceSize;
        	CuDNN::checkStatus (
        		cudnnGetConvolutionForwardWorkspaceSize(
        			handle,
        			inputDescriptor,
        			filterDescriptor,
        			convolutionDescritor,
        			outputDescriptor,
        			algorithm,
        			&workspaceSize
        		)
        	);

        	return Buffer<T>(workspaceSize);
        }

		template <typename T>
		Buffer<T> createBackwardDataWorkspace (
			Handle handle, 
			TensorDescriptor diffInputDescriptor, 
			FilterDescriptor<T> filterDescriptor,
			ConvolutionDescriptor convolutionDescritor, 
			TensorDescriptor diffOutputDescriptor,
			ConvolutionBwdDataAlgo algorithm ) 
		{
    		size_t workspaceSize;
        	CuDNN::checkStatus (
        		cudnnGetConvolutionBackwardDataWorkspaceSize(
        			handle,
        			filterDescriptor,
        			diffOutputDescriptor,
        			convolutionDescritor,
        			diffInputDescriptor,
        			algorithm,
        			&workspaceSize
        		)
        	);

        	return Buffer<T>(workspaceSize / sizeof(T));
        }

		template <typename T>
		Buffer<T> createBackwardFilterWorkspace (
			Handle handle, 
			TensorDescriptor inputDescriptor, 
			FilterDescriptor<T> filterDescriptor,
			ConvolutionDescriptor convolutionDescritor, 
			TensorDescriptor diffOutputDescriptor,
			ConvolutionBwdFilterAlgo algorithm ) 
		{
			size_t workspaceSize;
        	CuDNN::checkStatus (
        		cudnnGetConvolutionBackwardFilterWorkspaceSize(
        			handle,
        			inputDescriptor,
        			diffOutputDescriptor,
        			convolutionDescritor,
        			filterDescriptor,
        			algorithm,
        			&workspaceSize
        		)
        	);

        	return Buffer<T>(workspaceSize / sizeof(T));
		}
	}

} // CuDNN

#endif // PYCUDNN_CONVOLUTION_HPP
