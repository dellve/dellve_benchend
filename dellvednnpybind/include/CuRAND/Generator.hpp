#ifndef CURAND_GENERATOR_HPP
#define CURAND_GENERATOR_HPP

#include <memory>

#include <curand.h>
#include "Ordering.hpp"
#include "RngType.hpp"
#include "Status.hpp"

namespace CuRAND {
    class Generator {
    private:

        struct RawGenerator {
            curandGenerator_t gen;

            RawGenerator(RngType type = CURAND_RNG_PSEUDO_XORWOW) {
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

    public:
        Generator(RngType type = CURAND_RNG_PSEUDO_XORWOW, 
                  unsigned long long seed = 4242ULL,
                  Ordering order = CURAND_ORDERING_PSEUDO_SEEDED) :
                  generatorPtr(std::make_shared<RawGenerator>(type)) {
            SetPseudoRandomGeneratorSeed(seed);
            SetGeneratorOrdering(order);
        }

        void SetPseudoRandomGeneratorSeed(unsigned long long seed) {
            struct RawGenerator *generator = generatorPtr.get();
            checkStatus(curandSetPseudoRandomGeneratorSeed(*generator, seed));
        } 

        void SetGeneratorOrdering(Ordering order) {
            struct RawGenerator *generator = generatorPtr.get();
            checkStatus(curandSetGeneratorOrdering(*generator, order));
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
}

#endif // CURAND_GENERATOR_HPP
