import dellve_cudnn_benchmark as dcb
from BenchmarkFactory import BenchmarkFactory
from helper import problem_set as ps


class ForwardActivationBenchmark(BenchmarkFactory):
    name = 'ForwardActivationBenchmark'

    def get_problem_set(self):
        return ps.csv_get_problem_set('activation/basic.csv')

    def get_controller(self):
        return dcb.activation_forward

class BackwardActivationBenchmark(BenchmarkFactory):
    name = 'BackwardActivationBenchmark'

    def get_problem_set(self):
        return ps.csv_get_problem_set('activation/basic.csv')

    def get_controller(self):
        return dcb.activation_forward

class ForwardSoftmaxBenchmark(BenchmarkFactory):
    name = 'ForwardSoftmaxBenchmark'

    def get_problem_set(self):
        result = ps.csv_get_problem_set('softmax/basic.csv')
        for s in result:
            s.append('fast')
        return result

    def get_controller(self):
        return dcb.softmax_forward

class BackwardSoftmaxBenchmark(BenchmarkFactory):
    name = 'BackwardSoftmaxBenchmark'

    def get_problem_set(self):
        result = ps.csv_get_problem_set('softmax/basic.csv')
        for s in result:
            s.append('fast')
        return result

    def get_controller(self):
        return dcb.softmax_backward

class ForwardPoolingBenchmark(BenchmarkFactory):
    name = 'ForwardPoolingBenchmark'

    def get_problem_set(self):
        result = ps.csv_get_problem_set('pooling/basic.csv')
        for s in result:
            s.append('max')
        return result

    def get_controller(self):
        return dcb.pooling_forward

class BackwardPoolingBenchmark(BenchmarkFactory):
    name = 'BackwardPoolingBenchmark'

    def get_problem_set(self):
        result = ps.csv_get_problem_set('pooling/basic.csv')
        for s in result:
            s.append('max')
        return result

    def get_controller(self):
        return dcb.pooling_backawrd

class ForwardConvolutionBenchmark(BenchmarkFactory):
    name = 'ForwardConvolutionBenchmark'

    def get_problem_set(self):
        return ps.csv_get_problem_set('convolution/forward_basic.csv')

    def get_controller(self):
        return dcb.convolution_forward

class BackwardConvolutionDataBenchmark(BenchmarkFactory):
    name = 'BackwardConvolutionDataBenchmark'

    def get_problem_set(self):
        return ps.csv_get_problem_set('convolution/backward_data_basic.csv')

    def get_controller(self):
        return dcb.convolution_backward_data

class BackwardConvolutionFilterBenchmark(BenchmarkFactory):
    name = 'BackwardConvolutionFilterBenchmark'

    def get_problem_set(self):
        return ps.csv_get_problem_set('convolution/backward_filter_basic.csv')

    def get_controller(self):
        return dcb.convolution_backward_filter
