#ifndef CURAND_RNGTYPE_HPP
#define CURAND_RNGTYPE_HPP

#include <functional>

#include <curand.h>

namespace CuRAND {
    typedef curandRngType_t RngType;
}

namespace std {

	template<>
	struct hash<CuRAND::RngType> {
		size_t operator()(const CuRAND::RngType &type) const {
			return hash<size_t>{}((size_t) type);
		}
	};

}

#endif // CURAND_RNGTYPE_HPP
