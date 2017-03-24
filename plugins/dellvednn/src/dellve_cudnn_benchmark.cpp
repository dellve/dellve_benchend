#include "dellve_cudnn_benchmark.hpp"

#include "CuDNN/ActivationDescriptor.hpp"
#include "CuDNN/Handle.hpp"
#include "CuDNN/Status.hpp"
#include "CuDNN/Tensor.hpp"		   
 
#include <iostream>

namespace py = pybind11;

template <typename T>
DELLve::Benchmark activationForward ( int n, int c, int h, int w ) {
	std::cout << "Creating handle..." << std::endl;

	CuDNN::Handle handle;

	std::cout << "Creating activation descriptor..." << std::endl;

	CuDNN::ActivationDescriptor descriptor;

	std::cout << "Setting activation descriptor..." << std::endl;

	CuDNN::checkStatus (
		cudnnSetActivationDescriptor ( 
			descriptor,
			CUDNN_ACTIVATION_TANH,
			CUDNN_NOT_PROPAGATE_NAN,
			0.0
		)
	);

	static const T alpha = 1.0;
	static const T beta = 0.0;

	std::cout << "Creating tensor x" << std::endl;

	auto x = CuDNN::Tensor<T>::NCHW::create(n, c, h, w);

	std::cout << "Creating tensor y" << std::endl;

	auto y = CuDNN::Tensor<T>::NCHW::create(n, c, h, w);

	std::cout << "Creating benchmark..." << std::endl;

	return [=]() {
		return cudnnActivationForward (
			handle,
			descriptor,
			&alpha,
			x.getDescriptor(),
			x,
			&beta,
			y.getDescriptor(),
			y 
		);
	};	

}

PYBIND11_PLUGIN(dellve_cudnn_benchmark) {
	py::module m("dellve_cudnn_benchmark");

	DELLve::registerBenchmark(m, "ActivationForward", &activationForward<float>);

	return m.ptr();
}