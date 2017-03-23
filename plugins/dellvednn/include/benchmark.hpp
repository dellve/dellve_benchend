#ifndef CUDNN_BENCHMARK_H_
#define CUDNN_BENCHMARK_H_

#include <cuda.h>
#include <cuda_device_runtime_api.h>

#include <functional>
#include <tuple>
#include <unistd.h>

#include "CuDNN/Status.hpp"

namespace Benchmark {
	namespace __Benchmark {
		template <typename F, typename Tuple, bool Done, int Total, int... N>
	    struct runImpl 
	    {
	        static void run(F f, Tuple && t) {
	            runImpl<F, Tuple, Total == 1 + sizeof...(N), Total, N..., sizeof...(N)>::run(f, std::forward<Tuple>(t));
	        }
	    };

	    template <typename F, typename Tuple, int Total, int... N>
	    struct runImpl<F, Tuple, true, Total, N...> 
	    {
	        static void run(F f, Tuple && t) {
	            CuDNN::checkStatus(f(std::get<N>(std::forward<Tuple>(t))...));
	        }
	    };

	    template <typename F, typename Tuple>
		auto run(F f, Tuple && t) 
		{
		    typedef typename std::decay<Tuple>::type ttype;
		    runImpl<F, Tuple, 0 == std::tuple_size<ttype>::value, std::tuple_size<ttype>::value>::run(f, std::forward<Tuple>(t));
		}
	}

	template <typename ConfigReturnType, typename ... ConfigArgTypes, typename ... OperationArgTypes>
	auto create (
		ConfigReturnType(*configFunc)(ConfigArgTypes...),
		CuDNN::Status(*operationFunc)(OperationArgTypes...) ) 
	{
		return [=](int gpuId, int numRuns, ConfigArgTypes ... configArgs) {
			auto operationArgs = configFunc(configArgs...);
			cudaDeviceSynchronize();
			while (numRuns--)
				__Benchmark::run(operationFunc, operationArgs);
			cudaDeviceSynchronize();
		};
	}
}

#endif // CUDNN_BENCHMARK_H_
