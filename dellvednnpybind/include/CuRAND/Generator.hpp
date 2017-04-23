#ifndef CURAND_GENERATOR_HPP
#define CURAND_GENERATOR_HPP

#include <curand.h>
#include "Ordering.hpp"
#include "RAII.hpp"
#include "RngType.hpp"
#include "Status.hpp"

namespace CuRAND {

    Status CreateGenerator(curandGenerator_t *generator) {
        RngType type = CURAND_RNG_PSEUDO_XORWOW;
        Ordering order = CURAND_ORDERING_PSEUDO_SEEDED;

        checkStatus(curandCreateGenerator(generator, type));
        checkStatus(curandSetPseudoRandomGeneratorSeed(*generator, 4242ULL));
        checkStatus(curandSetGeneratorOrdering(*generator, order));

        return CURAND_STATUS_SUCCESS;
    }

    class Generator : 
        public RAII<curandGenerator_t,
                    CreateGenerator,
                    curandDestroyGenerator> {};
}
#endif // CURAND_GENERATOR_HPP
