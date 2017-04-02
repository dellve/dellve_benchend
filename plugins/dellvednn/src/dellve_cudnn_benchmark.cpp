#include "dellve_cudnn_benchmark.hpp"

#include "dellve_cudnn_activation.hpp"
#include "dellve_cudnn_convolution.hpp"
#include "dellve_cudnn_softmax.hpp"
#include "dellve_cudnn_pooling.hpp"

#include <iostream>

namespace py = pybind11;

PYBIND11_PLUGIN(dellve_cudnn_benchmark) {
	py::module m("dellve_cudnn_benchmark");

	py::class_<DELLve::BenchmarkController>(m, "BenchmarkController")
        .def("start", &DELLve::BenchmarkController::start)
        .def("stop", &DELLve::BenchmarkController::stop)
        .def("get_curr_run", &DELLve::BenchmarkController::getCurrRun)
        .def("get_progress", &DELLve::BenchmarkController::getProgress)
        .def("get_curr_time_micro", &DELLve::BenchmarkController::getCurrTimeMicro)
        .def("get_avg_time_micro", &DELLve::BenchmarkController::getAvgTimeMicro);


	DELLve::registerBenchmark(m, "activation_forward", &CuDNN::Activation::forward<float>);
	DELLve::registerBenchmark(m, "activation_backward", &CuDNN::Activation::backward<float>);
	DELLve::registerBenchmark(m, "softmax_forward", &CuDNN::Softmax::forward<float>);
	DELLve::registerBenchmark(m, "softmax_backward", &CuDNN::Softmax::backward<float>);
	DELLve::registerBenchmark(m, "convolution_forward", &CuDNN::Convolution::forward<float>);
	DELLve::registerBenchmark(m, "convolution_backward_data", &CuDNN::Convolution::backwardData<float>);
	DELLve::registerBenchmark(m, "convolution_backward_filter", &CuDNN::Convolution::backwardFilter<float>);
	DELLve::registerBenchmark(m, "pooling_forward", &CuDNN::Pooling::forward<float>);
	DELLve::registerBenchmark(m, "pooling_backward", &CuDNN::Pooling::forward<float>);
	
	return m.ptr();
}

