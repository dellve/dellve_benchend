#ifndef CURAND_GENERATOR_HPP
#define CURAND_GENERATOR_HPP

#include <memory>

#include <curand.h>

#include "Ordering.hpp"
#include "RngType.hpp"
#include "Status.hpp"

namespace CuRAND {
    class Generator {

        struct RawGenerator {
            curandGenerator_t gen;

            RawGenerator(RngType type) {
                checkStatus(curandCreateGenerator(&gen, type));
            }

            ~RawGenerator() {
                checkStatus(curandDestroyGenerator(gen));
            }

            operator curandGenerator_t () {
                return gen;
            }
        };

        std::shared_ptr<RawGenerator> generatorPtr;

    protected:

        Generator(RngType type) :
            generatorPtr(std::make_shared<RawGenerator>(type)) {}

    public:

        void setOffset(unsigned long long offset) {
            checkStatus(curandSetGeneratorOffset(*this, offset));
        }

        void setOrdering(curandOrdering_t order) {
            checkStatus(curandSetGeneratorOrdering(*this, order));
        }

        void SetPseudoRandomGeneratorSeed(unsigned long long seed) {
            checkStatus(curandSetPseudoRandomGeneratorSeed(*this, seed));
        } 

        void generate(unsigned int *buffer, size_t n) {
            checkStatus(curandGenerate(*this, buffer, n));
        }
        void generate(unsigned long long *buffer, size_t n){
            checkStatus(curandGenerateLongLong(*this, buffer, n));
        }

        void generateUniform(float *buffer, size_t n){
            checkStatus(curandGenerateUniform(*this, buffer, n));
        }
        void generateUniform(double *buffer, size_t n){
            checkStatus(curandGenerateUniformDouble(*this, buffer, n));
        }

        void generateNormal(float *buffer, size_t n, float mean, float stddev){
            checkStatus(curandGenerateNormal(*this, buffer, n, mean, stddev));
        }
        
        void generateNormal(double *buffer, size_t n, double mean, double stddev){
            checkStatus(curandGenerateNormalDouble(*this, buffer, n, mean, stddev));
        }

        void generateLogNormal(float *buffer, size_t n, float mean, float stddev){
            checkStatus(curandGenerateLogNormal(*this, buffer, n, mean, stddev));
        }
        void generateLogNormal(double *buffer, size_t n, double mean, double stddev){
            checkStatus(curandGenerateLogNormalDouble(*this, buffer, n, mean, stddev));
        }

        void generatePoisson(unsigned int *buffer, size_t n, double lambda){
            checkStatus(curandGeneratePoisson(*this, buffer, n, lambda));
        } 

        operator curandGenerator_t () {
            return generatorPtr->gen; 
        }
    };

    class PseudoGenerator : public Generator {

    	PseudoGenerator(RngType type) : Generator(type) {}

    	static const std::unordered_set<RngType> orderingSet {
    		CURAND_ORDERING_PSEUDO_BEST, 
    		CURAND_ORDERING_PSEUDO_DEFAULT,
    		CURAND_ORDERING_PSEUDO_SEEDED
    	};

    	static const std::unordered_set<RngType> typeSet {
    		CURAND_RNG_PSEUDO_DEFAULT,
    		CURAND_RNG_PSEUDO_XORWOW,
    		CURAND_RNG_PSEUDO_MRG32K3A,
    		CURAND_RNG_PSEUDO_MTGP32,
    		CURAND_RNG_PSEUDO_MT19937,
    		CURAND_RNG_PSEUDO_PHILOX4_32_10
    	};

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

    class QuaziGenerator : public Generator {

    	QuaziGenerator(RngType type) : Generator(type) {}

    	static const std::unordered_set<RngType> orderingSet {
    		CURAND_ORDERING_QUASI_DEFAULT
    	};

    	static const std::unordered_set<RngType> typeSet {
    		CURAND_RNG_QUASI_DEFAULT,
    		CURAND_RNG_QUASI_SOBOL32,
    		CURAND_RNG_QUASI_SCRAMBLED_SOBOL32,
    		CURAND_RNG_QUASI_SOBOL64,
    		CURAND_RNG_QUASI_SCRAMBLED_SOBOL64
    	};

    public:

    	void setDimensions (unsigned int numDimensions) {
    		checkStatus(curandSetQuasiRandomGeneratorDimensions(*this, numDimensions));
    	}

    	static QuaziGenerator create ( RngType type,
    		unsigned long long seed = 0,
    		Ordering ordering = CURAND_ORDERING_QUASI_DEFAULT ) {	
    		// Ensure correct generator type
    		if (typeSet.find(type) == typeSet.end()) 
    			throw Exception(CURAND_STATUS_TYPE_ERROR);
    		// Ensure correct ordering type
    		if (orderingSet.find(ordering) == orderingSet.end()) 
    			throw Exception(CURAND_STATUS_TYPE_ERROR);
    		// Create generator
    		QuaziGenerator g(type);
    		// Setup generator
    		g.setOrdering(ordering);
    		g.setSeed(seed);
    		// Done!
    		return g;
    	}

    	static QuaziGenerator createDefault ( RngType type, unsigned long long seed = 0 ) {	
    		return create(type, seed, CURAND_ORDERING_QUASI_DEFAULT);
    	}
    };
}

#endif // CURAND_GENERATOR_HPP
