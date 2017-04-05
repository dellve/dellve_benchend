#ifndef PYCUDNN_HANDLE
#define PYCUDNN_HANDLE

#include <cudnn.h>

#include "RAII.hpp" // RAII

namespace CuDNN {

	class Handle :
		public RAII< cudnnHandle_t,
					cudnnCreate,
					cudnnDestroy > {};
}

#endif // PYCUDNN_HANDLE
