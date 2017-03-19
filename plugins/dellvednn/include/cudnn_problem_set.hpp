#ifndef DELLVE_CUDNN_PROBLEM_SET_H_
#define DELLVE_CUDNN_PROBLEM_SET_H_

#include <vector>
#include <tuple>
#include <string>
#include <iostream>

#include "csv.hpp"

/**
 * Problem Set class that holds the input values for running convolutions.
 *
 * TODO: Add description for each input
 * Holds: 
 * w
 * h
 * c
 * n
 * k
 * r
 * s
 * pad_w
 * pad_h
 * wstride
 * hstride
 */
class CudnnConvProblemSet {
private:
    std::vector<std::tuple<int, int, int, int, int, int, int, int, int, int, int>> problems_;

public:
    /**
     * Read the csv given the filename passed in and save problem sets as vectors.
     */
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

    /**
     * Return the tuple on the index passed as input.
     */
    std::tuple<int, int, int, int, int, int, int, int, int, int, int> get(int i) {
        return problems_[i];
    }

    /**
     * Return the number of tuples saved.
     */
    int getSize(void) {
        return problems_.size();
    }
};

/**
 * Problem Set class that holds the input values for running pooling.
 *
 * TODO: Add description for each input
 * Holds: 
 * w
 * h
 * c
 * n
 * win_w
 * win_h
 * pad_w
 * pad_h
 * wstride
 * hstride
 */
class CudnnPoolProblemSet {
private: 
    std::vector<std::tuple<int, int, int, int, int, int, int, int, int, int>> problems_;

public:     
    /**
     * Read the csv given the filename passed in and save problem sets as vectors.
     */
    CudnnPoolProblemSet(std::string filename) {
        problems_ = {};
        io::CSVReader<10> in(filename);
        in.read_header(io::ignore_extra_column, "w", "h", "c", "n", "win_h", "win_w", "pad_w", 
                                                "pad_h", "wstride", "hstride");
        int w, h, c, n, win_w, win_h, pad_w, pad_h, wstride, hstride;
        while(in.read_row( w, h, c, n, win_w, win_h, pad_w, pad_h, wstride, hstride)) {
            problems_.push_back(std::make_tuple(w, h, c, n, win_h, win_w, pad_w, pad_h, wstride, hstride));
        }
    } 

    /**
     * Return the tuple on the index passed as input.
     */
    std::tuple<int, int, int, int, int, int, int, int, int, int> get(int i) {
        return problems_[i];
    }

    /**
     * Return the number of tuples saved.
     */
    int getSize(void) {
        return problems_.size();
    }
};

/**
 * Problem Set class that holds the input values for running Softmax methods.
 *
 * TODO: Add description for each input
 * Holds: 
 * w
 * h
 * c
 * n
 */
class CudnnSoftmaxProblemSet {
private: 
    std::vector<std::tuple<int, int, int, int>> problems_;

public:  
    /**
     * Read the csv given the filename passed in and save problem sets as vectors.
     */
    CudnnSoftmaxProblemSet(std::string filename) {
        problems_ = {};
        io::CSVReader<4> in(filename);
        in.read_header(io::ignore_extra_column, "w", "h", "c", "n"); 
        int w, h, c, n;
        while(in.read_row( w, h, c, n)) {
            problems_.push_back(std::make_tuple(w, h, c, n));
        }
    } 

    /**
     * Return the tuple on the index passed as input.
     */
    std::tuple<int, int, int, int> get(int i) {
        return problems_[i];
    }

    /**
     * Return the number of tuples saved.
     */
    int getSize(void) {
        return problems_.size();
    }
    
};

/**
 * Problem Set class that holds the input values for running activation methods.
 *
 * TODO: Add description for each input
 * Holds: 
 * w
 * h
 * c
 * n
 */
class CudnnActivationProblemSet {
private:
    std::vector<std::tuple<int, int, int, int>> problems_;

public: 
    /**
     * Read the csv given the filename passed in and save problem sets as vectors.
     */
    CudnnActivationProblemSet(std::string filename) {
        problems_ = {};
        io::CSVReader<4> in(filename);
        in.read_header(io::ignore_extra_column, "w", "h", "c", "n"); 
        int w, h, c, n;
        //double reluceiling;
        while(in.read_row( w, h, c, n )) {
            problems_.push_back(std::make_tuple(w, h, c, n));
        }
    } 

    /**
     * Return the tuple on the index passed as input.
     */
    std::tuple<int, int, int, int> get(int i) {
        return problems_[i];
    }

    /**
     * Return the number of tuples saved.
     */
    int getSize(void) {
        return problems_.size();
    }
};

#endif // DELLVE_CUDNN_PROBLEM_SET_H_
