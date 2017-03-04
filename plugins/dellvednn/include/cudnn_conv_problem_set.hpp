#ifndef DELLVE_CUDNN_CONV_PROBLEM_SET_H_
#define DELLVE_CUDNN_CONV_PROBLEM_SET_H_

#include <vector>
#include <tuple>
#include <string>
#include <iostream>

#include "csv.hpp"

class CudnnConvProblemSet {
private:
    std::vector<std::tuple<int, int, int, int, int, int, int, int, int, int, int>> problems_;

public:
    CudnnConvProblemSet(std::string filename) {
        problems_ = {};
        io::CSVReader<11> in(filename);
        in.read_header(io::ignore_extra_column, "w", "h", "c", "n", "k", "r", "s", "pad_w", 
                                                "pad_h", "wstride", "hstride");
        int w, h, c, n, k, r, s, pad_w, pad_h, wstride, hstride;
        while(in.read_row( w, h, c, n, k, r, s, pad_w, pad_h, wstride, hstride)) {
            problems_.push_back(std::make_tuple(w, h, c, n, k, r, s, pad_w, pad_h, wstride, hstride));
        }
    }

    std::tuple<int, int, int, int, int, int, int, int, int, int, int> get(int i) {
        return problems_[i];
    }

    int getSize(void) {
        return problems_.size();
    }
};

#endif // DELLVE_CUDNN_CONV_PROBLEM_SET_H_
