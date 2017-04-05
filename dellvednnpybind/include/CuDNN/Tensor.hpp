#ifndef PYCUDNN_TENSOR_HPP
#define PYCUDNN_TENSOR_HPP

#include <cudnn.h>

#include <memory>
#include <tuple>

#include "Buffer.hpp"
#include "DataType.hpp"
#include "TensorDescriptor.hpp"

namespace CuDNN {

	template <typename T>
	class Tensor {
		
		Buffer<T> mBuffer;
		CuDNN::TensorDescriptor mDescriptor;
		std::vector<int> mDims;
		
		Tensor (int n, int c, int h, int w, 
			TensorDescriptor descriptor ) :
			mBuffer(n * c * h * w),
			mDescriptor(descriptor),
			mDims({n, c, h, w}) {}

		Tensor ( const std::vector<int> dims,
			TensorDescriptor descriptor ) :
			mBuffer(accumulate(dims)),
			mDescriptor(descriptor),
			mDims(dims) {}

	private:

		static size_t accumulate ( const std::vector<int> dims ) {
			return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
		} 

		template <typename ... DecsSetTypes, typename ... ArgTypes>
		static inline CuDNN::TensorDescriptor makeDescriptor (
			Status(*descSetFunc)(DecsSetTypes...), ArgTypes ... args ) 
		{
			CuDNN::TensorDescriptor descriptor;
			CuDNN::checkStatus(descSetFunc(descriptor, args ...));
			return descriptor;
		}  

	public:

		operator T*() const {
			return mBuffer;
		}

		const TensorDescriptor& getDescriptor() const {
			return mDescriptor;
		}


	 	static Tensor<T>
	 	createNCHW ( int n, int c, int h, int w, T fillValue = 0 ) {
	 		return Tensor(n, c, h, w, makeDescriptor ( 
 				&cudnnSetTensor4dDescriptor, 
 				CUDNN_TENSOR_NCHW, 
 				CuDNN::dataType<T>::type, 
 				n, c, h, w )
	 		);
    	}

    	static Tensor<T>
	 	createNCHW ( std::tuple<int, int, int, int> NCHW, T fillValue = 0 ) {
	 		int n, c, h, w; 
	 		std::tie(n, c, h, w) = NCHW;
	 		return Tensor(n, c, h, w, makeDescriptor ( 
 				&cudnnSetTensor4dDescriptor, 
 				CUDNN_TENSOR_NCHW, 
 				CuDNN::dataType<T>::type, 
 				n, c, h, w )
	 		);
    	}
    
    	// static auto createRandNCHW ( int n, int c, int h, int w ) {
    	
    	// }
			
	 	static Tensor<T> 
	 	createNHWC ( int n, int h, int w, int c, T fillValue = 0 ) {
			return Tensor(n, h, w, c, makeDescriptor ( 
 				&cudnnSetTensor4dDescriptor, 
 				CUDNN_TENSOR_NHWC, 
 				CuDNN::dataType<T>::type, 
 				n, h, w, c )
	 		);
    	}

    	static Tensor<T> 
	 	createNHWC ( std::tuple<int, int, int, int> NHWC, T fillValue = 0 ) {
			int n, h, w, c; 
	 		std::tie(n, h, w, c) = NHWC;
			return Tensor(n, h, w, c, makeDescriptor ( 
 				&cudnnSetTensor4dDescriptor, 
 				CUDNN_TENSOR_NHWC, 
 				CuDNN::dataType<T>::type, 
 				n, h, w, c )
	 		);
    	}
    
    	// static auto createRandNHWC ( int n, int c, int h, int w ) {

    	// }

	    static Tensor<T>
	    create (
	    	const std::vector<int>& dims, 
	    	const std::vector<int>& strides, 
	    	T fillValue = 0 ) 
	    {
			return Tensor(dims, makeDescriptor ( 
 				&cudnnSetTensorNdDescriptor, 
 				CuDNN::dataType<T>::type, 
 				dims.size(),
 				dims.data(),
 				strides.data() )
	 		);
	    }

	   //  static auto createRand ( 
	   //  	const std::vector<int>& dims, 
	   //  	const std::vector<int>& strides ) 
	   //  {
	   //  	return std::make_tuple (
	 		// 	Tensor(dims),
	 		// 	TensorDescriptor ( 
	 		// 		&cudnnSetTensorNdDescriptor, 
	 		// 		CuDNN::dataType<T>::type, 
	 		// 		dims.size(),
	 		// 		dims.data(),
	 		// 		strides.data() )
	 		// );
	   //  }

	};

}

#endif // PYCUDNN_TENSOR_HPP
