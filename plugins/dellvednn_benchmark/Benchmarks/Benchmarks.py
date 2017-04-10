import dellve_cudnn_benchmark as dcb
from BenchmarkFactory import BenchmarkFactory
from helper import problem_set as ps


class ForwardActivationBenchmark(BenchmarkFactory):
    name = 'ForwardActivationBenchmark'
    csv_filename = 'activation/basic.csv'

    def get_controller(self):
        return dcb.activation_forward

    def get_problem_set(self):
        return ps.csv_get_problem_set(self.csv_filename)

    def get_problem_header(self):
        return ps.csv_get_header(self.csv_filename)

class BackwardActivationBenchmark(BenchmarkFactory):
    name = 'BackwardActivationBenchmark'
    csv_filename = 'activation/basic.csv'

    def get_controller(self):
        return dcb.activation_forward

    def get_problem_set(self):
        return ps.csv_get_problem_set(self.csv_filename)

    def get_problem_header(self):
        return ps.csv_get_header(self.csv_filename)

class ForwardSoftmaxBenchmark(BenchmarkFactory):
    name = 'ForwardSoftmaxBenchmark'
    csv_filename = 'softmax/basic.csv'

    def get_controller(self):
        return dcb.softmax_forward

    def get_problem_set(self):
        result = ps.csv_get_problem_set(self.csv_filename)
        for s in result:
            s.append('fast')
        return result

    def get_problem_header(self):
        header = ps.csv_get_header(self.csv_filename)
        header.append('algo')
        return header

class BackwardSoftmaxBenchmark(BenchmarkFactory):
    name = 'BackwardSoftmaxBenchmark'
    csv_filename = 'softmax/basic.csv'

    def get_controller(self):
        return dcb.softmax_backward

    def get_problem_set(self):
        result = ps.csv_get_problem_set(self.csv_filename)
        for s in result:
            s.append('fast')
        return result

    def get_problem_header(self):
        header = ps.csv_get_header(self.csv_filename)
        header.append('algo')
        return header

class ForwardPoolingBenchmark(BenchmarkFactory):
    name = 'ForwardPoolingBenchmark'
    csv_filename = 'pooling/basic.csv'

    def get_controller(self):
        return dcb.pooling_forward

    def get_problem_set(self):
        result = ps.csv_get_problem_set(self.csv_filename)
        for s in result:
            s.append('max')
        return result

    def get_problem_header(self):
        header = ps.csv_get_header(self.csv_filename)
        header.append('algo')
        return header

class BackwardPoolingBenchmark(BenchmarkFactory):
    name = 'BackwardPoolingBenchmark'
    csv_filename = 'pooling/basic.csv'

    def get_controller(self):
        return dcb.pooling_backward

    def get_problem_set(self):
        result = ps.csv_get_problem_set(self.csv_filename)
        for s in result:
            s.append('max')
        return result

    def get_problem_header(self):
        header = ps.csv_get_header(self.csv_filename)
        header.append('algo')
        return header

class ForwardConvolutionBenchmark(BenchmarkFactory):
    name = 'ForwardConvolutionBenchmark'
    csv_filename = 'convolution/forward_basic.csv'

    def get_controller(self):
        return dcb.convolution_forward

    def get_problem_set(self):
        return ps.csv_get_problem_set(self.csv_filename)

    def get_problem_header(self):
        return ps.csv_get_header(self.csv_filename)

class BackwardConvolutionDataBenchmark(BenchmarkFactory):
    name = 'BackwardConvolutionDataBenchmark'
    csv_filename = 'convolution/backward_data_basic.csv'

    def get_controller(self):
        return dcb.convolution_backward_data

    def get_problem_set(self):
        return ps.csv_get_problem_set(self.csv_filename)

    def get_problem_header(self):
        return ps.csv_get_header(self.csv_filename)

class BackwardConvolutionFilterBenchmark(BenchmarkFactory):
    name = 'BackwardConvolutionFilterBenchmark'
    csv_filename = 'convolution/backward_filter_basic.csv'

    def get_controller(self):
        return dcb.convolution_backward_filter
    def get_problem_set(self):
        return ps.csv_get_problem_set(self.csv_filename)

    def get_problem_header(self):
        return ps.csv_get_header(self.csv_filename)
