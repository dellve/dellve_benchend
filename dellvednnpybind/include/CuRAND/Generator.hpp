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

        operator curandGenerator_t () {
            return generatorPtr->gen; 
        }
    };
}

#endif // CURAND_GENERATOR_HPP
