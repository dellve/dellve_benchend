/*
 * Basic test to see if our CuRAND wrapper
 * actually generates random numbers.
 */

#include <stdio.h>
#include <memory>

#include <curand.h>
#include "CuDNN/Tensor.hpp"
#include "CuRAND/Generator.hpp"
#include "CuRAND/Status.hpp"

void print_buffer(double *buf, int size) {
    for (int i = 0; i < size; i++) {
        printf("%5.3f ", buf[i]);
        if ((i + 1) % 10 == 0) {
            printf("\n");
        }
    }
}

void print_tensor(double* tensor, int size) {
    auto output = (double *) malloc(size * sizeof(double));
    cudaMemcpy(output, tensor, size * sizeof(double), cudaMemcpyDeviceToHost);
    print_buffer(output, size);
    free(output);
}

int main(int argc, char *argv[]) {
    auto w = 10;
    auto h = 10;
    auto size = w * h;
    auto input = CuDNN::Tensor<double>::createNCHW(1, 1, h, w);

    printf("\n");
    printf("---------------------------------\n");
    printf("     BEFORE RANDOM GENERATOR\n");
    printf("---------------------------------\n");
    print_tensor(input, size);

    CuRAND::Generator gen;
    auto runs = 3;
    for(int i = 0; i < runs; i++) {
        gen.generateUniform(input, size);
        printf("\n\n");
        printf("---------------------------------\n");
        printf("     AFTER RANDOM GENERATOR\n");
        printf("---------------------------------\n");
        printf("Run: %d\n", i+1);
        print_tensor(input, size);
    }

    return 0;
}
