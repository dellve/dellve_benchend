from setuptools import setup, find_packages

setup(
    name='dellve',
    version='0.1',
    py_modules=[
        'dellve'
    ],
    package_data={
        'dellve': [
            'dellve.config.yaml'
        ]
    },
    include_package_data=True,
    install_requires=[
        'Click',
        'PyYaml'
    ],
    entry_points='''
        [console_scripts]
        dellve=dellve:cli
    ''',
)
