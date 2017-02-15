#ifndef DELLVE_CUDNN_CONV_PROBLEM_SET_H_
#define DELLVE_CUDNN_CONV_PROBLEM_SET_H_

#include <vector>
#include <tuple>
#include <string>

class CudnnConvProblemSet {
private:
    std::vector<std::tuple<int, int, int, int, int, int, int, int, int, int, int>> problems_;

public:
    CudnnConvProblemSet(std::string filename) {
        //TODO: actually read the file
        problems_ = {
            std::make_tuple(700, 161, 1, 4, 32, 5, 20, 0, 0, 2, 2),
            std::make_tuple(700, 161, 1, 8, 32, 5, 20, 0, 0, 2, 2),
            std::make_tuple(700, 161, 1, 16, 32, 5, 20, 0, 0, 2, 2),
            std::make_tuple(700, 161, 1, 32, 32, 5, 20, 0, 0, 2, 2),
            std::make_tuple(341, 79, 32, 4, 32, 5, 10, 0, 0, 2, 2),
            std::make_tuple(341, 79, 32, 8, 32, 5, 10, 0, 0, 2, 2),
            std::make_tuple(341, 79, 32, 16, 32, 5, 10, 0, 0, 2, 2),
            std::make_tuple(341, 79, 32, 32, 32, 5, 10, 0, 0, 2, 2),
            std::make_tuple(480, 48, 1, 16, 16, 3, 3, 1, 1, 1, 1),
            std::make_tuple(240, 24, 16, 16, 32, 3, 3, 1, 1, 1, 1),
            std::make_tuple(120, 12, 32, 16, 64, 3, 3, 1, 1, 1, 1),
            std::make_tuple(60, 6, 64, 16, 128, 3, 3, 1, 1, 1, 1),
            std::make_tuple(108, 108, 3, 8, 64, 3, 3, 1, 1, 2, 2),
            std::make_tuple(54, 54, 64, 8, 64, 3, 3, 1, 1, 1, 1),
            std::make_tuple(27, 27, 128, 8, 128, 3, 3, 1, 1, 1, 1),
            std::make_tuple(14, 14, 128, 8, 256, 3, 3, 1, 1, 1, 1),
            std::make_tuple(7, 7, 256, 8, 512, 3, 3, 1, 1, 1, 1),
            std::make_tuple(224, 224, 3, 8, 64, 3, 3, 1, 1, 1, 1),
            std::make_tuple(112, 112, 64, 8, 128, 3, 3, 1, 1, 1, 1),
            std::make_tuple(56, 56, 128, 8, 256, 3, 3, 1, 1, 1, 1),
            std::make_tuple(28, 28, 256, 8, 512, 3, 3, 1, 1, 1, 1),
            std::make_tuple(14, 14, 512, 8, 512, 3, 3, 1, 1, 1, 1),
            std::make_tuple(7, 7, 512, 8, 512, 3, 3, 1, 1, 1, 1),
            std::make_tuple(224, 224, 3, 16, 64, 3, 3, 1, 1, 1, 1),
            std::make_tuple(112, 112, 64, 16, 128, 3, 3, 1, 1, 1, 1),
            std::make_tuple(56, 56, 128, 16, 256, 3, 3, 1, 1, 1, 1),
            std::make_tuple(28, 28, 256, 16, 512, 3, 3, 1, 1, 1, 1),
            std::make_tuple(14, 14, 512, 16, 512, 3, 3, 1, 1, 1, 1),
            std::make_tuple(7, 7, 512, 16, 512, 3, 3, 1, 1, 1, 1),
            std::make_tuple(224, 224, 3, 16, 64, 7, 7, 3, 3, 2, 2),
            std::make_tuple(28, 28, 192, 16, 32, 5, 5, 2, 2, 1, 1),
            std::make_tuple(28, 28, 192, 16, 64, 1, 1, 0, 0, 1, 1),
            std::make_tuple(14, 14, 512, 16, 48, 5, 5, 2, 2, 1, 1),
            std::make_tuple(14, 14, 512, 16, 192, 1, 1, 0, 0, 1, 1),
            std::make_tuple(7, 7, 832, 16, 256, 1, 1, 0, 0, 1, 1),
            std::make_tuple(7, 7, 832, 16, 128, 5, 5, 2, 2, 1, 1)
        };
    }

    std::tuple<int, int, int, int, int, int, int, int, int, int, int> get(int i) {
        return problems_[i];
    }

    int getSize(void) {
        return problems_.size();
    }
};

#endif // DELLVE_CUDNN_CONV_PROBLEM_SET_H_
