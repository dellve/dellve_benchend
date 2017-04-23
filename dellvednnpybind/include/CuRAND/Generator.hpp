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
}

#endif // CURAND_GENERATOR_HPP
