#ifndef PYCUDNN_CONVOLUTION_DESCRIPTOR_HPP
#define PYCUDNN_CONVOLUTION_DESCRIPTOR_HPP

#include <cudnn.h>

#include "RAII.hpp" // RAII

namespace CuDNN {

    class ConvolutionDescriptor :
      	public RAII<cudnnConvolutionDescriptor_t,
                    cudnnCreateConvolutionDescriptor,
                    cudnnDestroyConvolutionDescriptor> {

	public:

		static ConvolutionDescriptor create ( 
			int padH, 
			int padW, 
			int strideH, 
			int strideW, 
			int upscaleX = 1, 
			int upscaleY = 1 )
		{
			ConvolutionDescriptor object;

			CuDNN::checkStatus (
				cudnnSetConvolution2dDescriptor (
					object,
					padH,
					padW,
					strideH,
					strideW,
					upscaleX,
					upscaleY,
					CUDNN_CONVOLUTION )
			);

			return object;
		}

		static ConvolutionDescriptor createCrossCorrelation ( 
			int padH, 
			int padW, 
			int strideH, 
			int strideW, 
			int upscaleX = 1, 
			int upscaleY = 1 )
		{
			ConvolutionDescriptor object;

			CuDNN::checkStatus (
				cudnnSetConvolution2dDescriptor (
					object,
					padH,
					padW,
					strideH,
					strideW,
					upscaleX,
					upscaleY,
					CUDNN_CROSS_CORRELATION )
			);

			return object;
		}


	};
}

#endif // PYCUDNN_CONVOLUTION_DESCRIPTOR_HPP
