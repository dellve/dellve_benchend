#ifndef CURAND_ORDERING_HPP
#define CURAND_ORDERING_HPP

#include <functional>

#include <curand.h>

namespace CuRAND {
    typedef curandOrdering_t Ordering;
}

namespace std {

	template<>
	struct hash<CuRAND::Ordering> {
		size_t operator()(const CuRAND::Ordering &ordering) const {
			return hash<size_t>{}((size_t) ordering);
		}
	};
	
}

#endif // CURAND_ORDERING_HPP
