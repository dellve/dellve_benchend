#ifndef DELLVE_CUDNN_BENCHMARK_HPP
#define DELLVE_CUDNN_BENCHMARK_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cudnn.h>

#include "CuDNN/Status.hpp"

namespace DELLve {
	/**
	 * { item_description }
	 */
	typedef std::function<CuDNN::Status(void)> Benchmark;

	template <typename ... A>
	class BenchmarkDriver {
		int deviceId_;
		int numRuns_;
		Benchmark (*benchmarkFactory_) (A...);

	public:

		BenchmarkDriver ( int deviceId, int numRuns, Benchmark (*benchmarkFactory)(A...) ) :
			deviceId_(deviceId),
			numRuns_(numRuns),
			benchmarkFactory_(benchmarkFactory) {}

		void run(A... args) {
			cudaSetDevice(deviceId_);
			auto op = benchmarkFactory_(args...);
			cudaDeviceSynchronize();
			for (int i = 0; i < numRuns_; i++)
				CuDNN::checkStatus(op());
			cudaDeviceSynchronize();
		}
	};

	template <typename ... A> 
	void registerBenchmark (pybind11::module& m, const char* className, 
		Benchmark (*benchmarkFactory)(A...) ) 
	{
		pybind11::class_<BenchmarkDriver<A...>>(m, className)
			.def("__init__", [=](BenchmarkDriver<A...>& instance, int deviceId, int numRuns) {
				new (&instance) BenchmarkDriver<A...> (deviceId, numRuns, benchmarkFactory);
	    	})
	    	.def("run", &BenchmarkDriver<A...>::run);
	}
}

#endif // DELLVE_CUDNN_BENCHMARK_HPP

