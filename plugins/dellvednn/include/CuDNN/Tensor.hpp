#ifndef PYCUDNN_TENSOR_HPP
#define PYCUDNN_TENSOR_HPP

#include <cudnn.h>

#include <memory>
#include <tuple>

#include "CuDNN/DataType.hpp"
#include "CuDNN/TensorDescriptor.hpp"

namespace CuDNN {

	template <typename T>
	class Tensor {
		
		struct CudaBuffer {
			T* mData;
			size_t mSize;

			CudaBuffer (size_t size) : mSize(size) {
				cudaMalloc(&mData, size * sizeof(T));
			}

			~CudaBuffer () {
				cudaFree(mData);
			}

		};

		std::shared_ptr<CudaBuffer> mBuffer;
		CuDNN::TensorDescriptor mDescriptor;
		std::vector<int> mDims;
		
		Tensor (int n, int c, int h, int w, 
			TensorDescriptor descriptor ) :
			mBuffer(new CudaBuffer(n * c * h * w)),
			mDescriptor(descriptor),
			mDims({n, c, h, w}) {}

		Tensor ( const std::vector<int> dims,
			TensorDescriptor descriptor ) :
			mBuffer(new CudaBuffer(accumulate(dims))),
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
			return mBuffer->mData;
		}

		const TensorDescriptor& getDescriptor() const {
			return mDescriptor;
		}

		struct NCHW {

		 	static Tensor<T>
		 	create ( int n, int c, int h, int w, T fillValue = 0 ) {
		 		return Tensor(n, c, h, w, makeDescriptor ( 
	 				&cudnnSetTensor4dDescriptor, 
	 				CUDNN_TENSOR_NCHW, 
	 				CuDNN::dataType<T>::type, 
	 				n, c, h, w )
		 		);
	    	}
	    
	    	// static auto createRand ( int n, int c, int h, int w ) {
	    	
	    	// }

	    };

		struct NHWC {
			
		 	static std::tuple<CuDNN::Tensor<T>, CuDNN::TensorDescriptor> 
		 	create ( int n, int h, int w, int c, T fillValue = 0 ) {
				return Tensor(n, h, w, c, makeDescriptor ( 
	 				&cudnnSetTensor4dDescriptor, 
	 				CUDNN_TENSOR_NHWC, 
	 				CuDNN::dataType<T>::type, 
	 				n, h, w, c )
		 		);
	    	}
	    
	    	// static auto createRand ( int n, int c, int h, int w ) {

	    	// }
		};

	    static std::tuple<CuDNN::Tensor<T>, CuDNN::TensorDescriptor>
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
