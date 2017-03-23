#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "benchmark.hpp"

// NOTE: benchmarks are created on demand through Benchmark::create
// 		 factory function; the first parameter to this factory is 
// 		 a configFunc(...) function, and the second parameter is
// 		 an operationFunc(...) function; this works as follows:
// 		 
// 		 1. The configFunc(...) takes in configuration parameters that 
// 		 	are needed to setup function call arguments for the
// 		 	CuDNN operation that we're trying to benchmark.
// 		 
// 		 2. The resulting Python function exposes a function call
// 		    signature derived from that of configFunc; for example,
// 		    consider the following:
// 		    
// 		    m.def("operation", Benchmark::create(&config_func, &operation_func))
// 		    
// 		    This would create a Python function dellve_cudnn_benchmark.operation, 
// 		    with the following signature: (int gpuId, int numRuns, ... args)
// 		    
// 		    The " ... args" in this Python function are derived from the signature
// 		    of the "config_func". This means that if "config_func" was defined as
// 		    'void config_func(int foo, float bar)', for example, then the corresponding Python
// 		    function would have a signature (int gpuId, int numRuns, int foo, float bar).
// 		    
// 		    The goal here is for Python to parse a CSV file, and then pass values
// 		    from the CSV file directly to dellve_cudnn_benchmark.<operation_name>, 
// 		    which then would be automatically dispatched into a particular configFunc(...)
// 		    
// 		 3. The configFunc(...) must return a tuple of arguments that should be passed 
// 		 	into the operationFunc(...); so, if operationFunc(...) takes in cuDNN handle, 
// 		 	tensor descriptor, and GPU data pointer, configFunc(...) should have a
// 		 	return statement similar to 'return std::make_tuple(handle, descriptor, data)';
// 		 	defining configFunc(...) with std::tuple<...> configFunc(...) may be painful, 
// 		 	so just use 'auto configFunc(...)' and C++ will handle things for you!
// 		 	
// 		 4. The created benchmark works as follows:
// 		 	
// 		 		- pass parameters received from Python to configFunc(...) to obtain
// 		 		  arguments that should be passed to operationFunc()
// 		 		  
// 		 		- synchronize CUDA device identified by gpuId argument
// 		 		
// 		 		- run operationFunc(...) 'numRuns' times
// 		 		
// 		 		- synchronize CUDA device again (i.e. wait for workload to finish)
//			
//				Note: we don't do timing in C++ yet, because we could do it in Python;
//					  this could be changed in the future, if needed.
//					  
//		5. An example of what we may use in practice:
//		
//			m.def("activation_forward", Benchmark::create(&cudnnActivationForwardConfig, 
//														  &cudnnActivationForward))
//					  
// DISCLAIMER: This approach has only been tested with 'dummyOp' and 'dummyConfig' functions;
// 			   unlikely, but things may come up when we start using actual cuDNN calls instead.
// 			   
// 			   
 
auto dummyOp (int i, int j, int k) {
	return CUDNN_STATUS_SUCCESS;
}

auto dummyConfig (int i, int j) {
	return std::make_tuple(i + j, 0, 1);
}

namespace py = pybind11;

PYBIND11_PLUGIN(dellve_cudnn_benchmark) {
	py::module m("dellve_cudnn_benchmark");

	m.def("dummy", Benchmark::create(&dummyConfig, &dummyOp));

	return m.ptr();
}