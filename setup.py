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
        'pick',
        'pytest',
        'pyyaml',
        'zmq'
    ],
    entry_points='''
        [console_scripts]
        dellve=dellve.dellve:cli
    ''',
)

