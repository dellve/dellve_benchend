#ifndef PYCUDNN_FILTER_HPP
#define PYCUDNN_FILTER_HPP

#include <cudnn.h>
#include <cstdint>

#include "Buffer.hpp"
#include "Status.hpp"
#include "RAII.hpp"
#include "FilterDescriptor.hpp"

namespace CuDNN {

	template <typename T>
    class Filter {

		Buffer<T> 				mBuffer;
		FilterDescriptor<T> 	mDescriptor;

    	Filter ( const Buffer<T>& buffer, const FilterDescriptor<T>& descriptor ) :
    		mBuffer(buffer),
    		mDescriptor(descriptor) {}

    public:

		static Filter createNCHW ( int k, int c, int h, int w ) {
			return Filter ( Buffer<T>(k * c * h * w), 
							FilterDescriptor<T>::createNCHW(k, c, h, w) );
		}

		static Filter createNHWC ( int k, int c, int h, int w ) {
			return Filter ( Buffer<T>(k * c * h * w), 
							FilterDescriptor<T>::createNHWC(k, c, h, w) );
		}

		FilterDescriptor<T> getDescriptor() const {
			return mDescriptor;
		}

		operator T*() const {
			return mBuffer;
		}
	};
};

#endif // PYCUDNN_FILTER_HPP
