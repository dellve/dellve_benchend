# Ensure that the user has setuptools!
# import ez_setup; ez_setup.use_setuptools()

from setuptools import setup, find_packages

setup(
    name='dellve',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'dellve': [
            'dellve.config.yaml'
        ]
    },
    install_requires=[
        'click',
        'daemonocle',
        'falcon',
        'gevent',
        'pick',
        'pyyaml',
        'stringcase'
    ],
    setup_requires=[
        'pybind11',
        'pytest-runner'
    ],
    tests_require=[
        'pytest'
    ],
    entry_points='''
        [console_scripts]
        dellve=dellve.dellve:cli
    '''
)
