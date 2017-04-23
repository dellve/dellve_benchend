#ifndef CURAND_QUASI_GENERATOR_HPP
#define CURAND_QUASI_GENERATOR_HPP

#include <unordered_set>

#include "Generator.hpp"

namespace CuRAND {
	
	class QuasiGenerator : public Generator {

		QuasiGenerator(RngType type) : Generator(type) {}

		static const std::unordered_set<Ordering> orderingSet;
		static const std::unordered_set<RngType> typeSet;

	public:

		void setDimensions (unsigned int numDimensions) {
			checkStatus(curandSetQuasiRandomGeneratorDimensions(*this, numDimensions));
		}

		static QuasiGenerator create ( RngType type,
			Ordering ordering = CURAND_ORDERING_QUASI_DEFAULT ) {	
			// Ensure correct generator type
			if (typeSet.find(type) == typeSet.end()) 
				throw Exception(CURAND_STATUS_TYPE_ERROR);
			// Ensure correct ordering type
			if (orderingSet.find(ordering) == orderingSet.end()) 
				throw Exception(CURAND_STATUS_TYPE_ERROR);
			// Create generator
			QuasiGenerator g(type);
			// Setup generator
			g.setOrdering(ordering);
			// Done!
			return g;
		}

		static QuasiGenerator createDefault ( RngType type ) {	
			return create(type, CURAND_ORDERING_QUASI_DEFAULT);
		}
	};

	const std::unordered_set<Ordering>
	QuasiGenerator::orderingSet = std::unordered_set<Ordering>({
		CURAND_ORDERING_QUASI_DEFAULT
	});

	const std::unordered_set<RngType> 
	QuasiGenerator::typeSet = std::unordered_set<RngType>({
		CURAND_RNG_QUASI_DEFAULT,
		CURAND_RNG_QUASI_SOBOL32,
		CURAND_RNG_QUASI_SCRAMBLED_SOBOL32,
		CURAND_RNG_QUASI_SOBOL64,
		CURAND_RNG_QUASI_SCRAMBLED_SOBOL64
	});
}

#endif // CURAND_QUASI_GENERATOR_HPP
