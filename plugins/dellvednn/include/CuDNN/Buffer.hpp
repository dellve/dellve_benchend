#ifndef PYCUDNN_BUFFER_HPP
#define PYCUDNN_BUFFER_HPP

#include <cuda.h>

#include <memory>

namespace CuDNN {

	template <typename T>
	class Buffer {
		
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

	public:

		Buffer (size_t size) : mBuffer(new CudaBuffer(size)) {}

		operator T*() const {
			return mBuffer->mData;
		}

		size_t getSize() const {
			return mBuffer->mSize;
		}
	};

}

#endif // PYCUDNN_BUFFER_HPP
