
from setuptools import setup, find_packages

setup (
    name='Benchmarks',
    version='0.0.1',
    packages=find_packages(),
    install_requires=['dellve'],
    entry_points='''
    [dellve.benchmarks]
    ForwardActivationBenchmark=Benchmarks:ForwardActivationBenchmark
    BackwardACtivationBenchmark=Benchmarks:BackwardActivationBenchmark
    ForwardSoftmaxBenchmark=Benchmarks:ForwardSoftmaxBenchmark
    BackwardSoftmaxBenchmark=Benchmarks:BackwardSoftmaxBenchmark
    ForwardPoolingBenchmark=Benchmarks:ForwardPoolingBenchmark
    BackwardPoolingBenchmark=Benchmarks:BackwardPoolingBenchmark
    BackwardConvolutionDataBenchmark=Benchmarks:BackwardConvolutionDataBenchmark
    ForwardConvolutionBenchmark=Benchmarks:ForwardConvolutionBenchmark
    BackwardConvolutionFilterBenchmark=Benchmarks:BackwardConvolutionFilterBenchmark
    '''
)
