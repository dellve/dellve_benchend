import os
import setuptools
import setuptools.command.install
import shutil
import sys
import json


setuptools.setup(
    name='dellve',
    version='1.0.0',
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
        'dellve': [
            'data/jinja2/*',
            'data/config.json'
        ]
    },
    install_requires=[
        'click',
        'daemonocle',
        'falcon',
        'gevent',
        'jinja2',
        'jsonschema',
        'pick',
        'requests',
        'stringcase',
        'tqdm',
    ],
    setup_requires=[
        'pytest-runner'
    ],
    tests_require=[
        'pytest',
        'responses'
    ],
    entry_points='''
        [console_scripts]
        dellve=dellve.cli:cli
    '''
)
