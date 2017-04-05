#ifndef PYCUDNN_FILTER_DESCRIPTOR_HPP
#define PYCUDNN_FILTER_DESCRIPTOR_HPP

#include <cudnn.h>
#include <cstdint>

#include "Status.hpp"
#include "RAII.hpp"
#include "TensorFormat.hpp"

namespace CuDNN {

	template <typename T>
    class FilterDescriptor :
        public RAII<cudnnFilterDescriptor_t,
                    cudnnCreateFilterDescriptor,
                    cudnnDestroyFilterDescriptor> {

    	static inline FilterDescriptor create ( TensorFormat format, 
    		int k, int c, int h, int w ) 
    	{
	    	FilterDescriptor object;
	    	
	    	auto type = dataType<T>::type;

	    	CuDNN::checkStatus ( 
	    		cudnnSetFilter4dDescriptor(object, type, format, k, c, h, w)
	    	);

    		return object;
    	}

    public:

		static FilterDescriptor createNCHW ( int k, int c, int h, int w ) {
			return create(CUDNN_TENSOR_NCHW, k, c, h, w);
		}

		static FilterDescriptor createNHWC ( int k, int c, int h, int w ) {
			return create(CUDNN_TENSOR_NHWC, k, c, h, w);
		}
	};
};

#endif // PYCUDNN_FILTER_DESCRIPTOR_HPP
