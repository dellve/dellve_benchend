#include <string>
#include <stdio.h>

#include "cudnn_conv_driver.hpp"
#include "cudnn_conv_problem_set.hpp"

int main(int argc, char *argv[]) {
    if(argc == 1) {
        printf("Pass in csv of problem sets!\n");
        exit(0);
    }
    CudnnConvProblemSet problems(argv[1]);
    CudnnConvDriver driver(CudnnConvMethod::BACKWARD_DATA, problems);

    for (auto i = 0; i < problems.getSize(); i++) {
        int time = driver.run(i);
        printf("Run %d took %d us\n", i+1, time);
    }
    
    return 0;
}
