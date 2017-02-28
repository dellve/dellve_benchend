#include <string>
#include <stdio.h>

#include "cudnn_conv_driver.hpp"
#include "cudnn_conv_problem_set.hpp"
#include "cli_parser.hpp"

int main(int argc, char *argv[]) {
    CLIParser options(argc, argv); 
    CudnnConvProblemSet problems(options.getProblemSetFile());
    CudnnConvDriver driver(CudnnConvMethod::BACKWARD_DATA, problems);

    for (auto i = 0; i < problems.getSize(); i++) {
        int time = driver.run(i);
        printf("Run %d took %d us\n", i+1, time);
    }
    
    return 0;
}
