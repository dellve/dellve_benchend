#include "dellve_cudnn_benchmark.hpp"

#include "dellve_cudnn_activation.hpp"
#include "dellve_cudnn_convolution.hpp"
#include "dellve_cudnn_softmax.hpp"

#include <iostream>

namespace py = pybind11;

PYBIND11_PLUGIN(dellve_cudnn_benchmark) {
	py::module m("dellve_cudnn_benchmark");

	auto driver = py::class_<DELLve::BenchmarkDriver>(m, "Benchmark")
		.def("__init__", [=](DELLve::BenchmarkDriver& instance, int deviceId, int numRuns) {
			new (&instance) DELLve::BenchmarkDriver (deviceId, numRuns);
		});

	DELLve::registerBenchmark(driver, "activation_forward", &CuDNN::Activation::forward<float>);
	DELLve::registerBenchmark(driver, "activation_backward", &CuDNN::Activation::backward<float>);
	DELLve::registerBenchmark(driver, "softmax_forward", &CuDNN::Softmax::forward<float>); 
	DELLve::registerBenchmark(driver, "softmax_backward", &CuDNN::Softmax::backward<float>); 
	DELLve::registerBenchmark(driver, "convolution_forward", &CuDNN::Convolution::forward<float>);
	DELLve::registerBenchmark(driver, "convolution_backward_data", &CuDNN::Convolution::backwardData<float>);
	DELLve::registerBenchmark(driver, "convolution_backward_filter", &CuDNN::Convolution::backwardFilter<float>);
	
	return m.ptr();
}

