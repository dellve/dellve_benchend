
from setuptools import setup, find_packages

setup (
    name='ForwardActivation',
    version='0.0.1',
    packages=find_packages(),
    install_requires=['dellve',
                      'pynvml'],
    entry_points='''
    [dellve.benchmarks]
    ForwardActivationStressTool=ForwardActivation:ForwardActivationStressTool
    '''
)
