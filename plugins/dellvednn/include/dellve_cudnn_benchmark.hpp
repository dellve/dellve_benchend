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

	class BenchmarkDriver {
		
		int deviceId_;
		int numRuns_;

	public:

		BenchmarkDriver ( int deviceId, int numRuns ) :
			deviceId_(deviceId),
			numRuns_(numRuns) {}

		template <typename ... A>
	 	void run(Benchmark (*factory)(A...), A... args) const {
			cudaSetDevice(deviceId_);
			auto benchmark = factory(args...);
			cudaDeviceSynchronize();
			for (int i = 0; i < numRuns_; i++)
				CuDNN::checkStatus(benchmark());
			cudaDeviceSynchronize();
		}
	};

	template <typename T, typename ... A> 
	void registerBenchmark (T driver, const char* benchmarkName, 
		Benchmark (*benchmarkFactory)(A...) ) 
	{
		driver.def(benchmarkName, [=](const BenchmarkDriver& self, A ... args) {
			self.run(benchmarkFactory, args...);
		});
	}
}

#endif // DELLVE_CUDNN_BENCHMARK_HPP

