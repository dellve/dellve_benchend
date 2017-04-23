#ifndef CURAND_PSEUDO_GENERATOR_HPP
#define CURAND_PSEUDO_GENERATOR_HPP

#include <unordered_set>

#include "Generator.hpp"

namespace CuRAND {

    class PseudoGenerator : public Generator {

    	PseudoGenerator(RngType type) : Generator(type) {}

    	static const std::unordered_set<Ordering> orderingSet;
    	static const std::unordered_set<RngType> typeSet;

    public:

    	void setSeed(unsigned long long seed) {
    		checkStatus(curandSetPseudoRandomGeneratorSeed(*this, seed));
    	}

    	static PseudoGenerator create ( RngType type,
    		unsigned long long seed = 0,
    		Ordering ordering = CURAND_ORDERING_PSEUDO_DEFAULT ) {	
    		// Ensure correct generator type
    		if (typeSet.find(type) == typeSet.end()) 
    			throw Exception(CURAND_STATUS_TYPE_ERROR);
    		// Ensure correct ordering type
    		if (orderingSet.find(ordering) == orderingSet.end()) 
    			throw Exception(CURAND_STATUS_TYPE_ERROR);
    		// Create generator
    		PseudoGenerator g(type);
    		// Setup generator
    		g.setOrdering(ordering);
    		g.setSeed(seed);
    		// Done!
    		return g;
    	}

    	static PseudoGenerator createBest ( RngType type, unsigned long long seed = 0 ) {	
    		return create(type, seed, CURAND_ORDERING_PSEUDO_BEST);
    	}

    	static PseudoGenerator createDefault ( RngType type, unsigned long long seed = 0 ) {	
    		return create(type, seed, CURAND_ORDERING_PSEUDO_DEFAULT);
    	}

    	static PseudoGenerator createSeeded ( RngType type, unsigned long long seed = 0 ) {	
    		return create(type, seed, CURAND_ORDERING_PSEUDO_SEEDED);
    	}
    };

    const std::unordered_set<Ordering> 
    PseudoGenerator::orderingSet = std::unordered_set<Ordering>({
		CURAND_ORDERING_PSEUDO_BEST, 
		CURAND_ORDERING_PSEUDO_DEFAULT,
		CURAND_ORDERING_PSEUDO_SEEDED
	});

    const std::unordered_set<RngType> 
    PseudoGenerator::typeSet = std::unordered_set<RngType>({
		CURAND_RNG_PSEUDO_DEFAULT,
		CURAND_RNG_PSEUDO_XORWOW,
		CURAND_RNG_PSEUDO_MRG32K3A,
		CURAND_RNG_PSEUDO_MTGP32,
		CURAND_RNG_PSEUDO_MT19937,
		CURAND_RNG_PSEUDO_PHILOX4_32_10
	});
}

#endif // CURAND_PSEUDO_GENERATOR_HPP
