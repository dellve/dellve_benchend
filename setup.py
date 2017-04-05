import os
import setuptools
import setuptools.command.install
import shutil
import sys
import json


class CustomInstallCommand(setuptools.command.install.install):
    """Custom install command"""

    def run(self):
        # Run setuptools install command
        setuptools.command.install.install.run(self)

        # Get app directory path
        import dellve.config
        app_dir = dellve.config.get('app-dir')

        # Create app directory
        if not os.path.exists(app_dir):
            print 'Creating new applicaion directory', str(app_dir)
            os.makedirs(app_dir)
        else:
            print 'Using existing applicaion directory', str(app_dir)

        # Get paths to DELLve configuration files
        config_file = os.path.join(app_dir, 'config.json')
        default_config_file = os.path.join(app_dir, 'default.config.json')

        # Copy default JSON config files
        shutil.copy('dellve/data/config.json', default_config_file)

        # Copy main JSON config file
        if not os.path.exists(config_file):
            shutil.copy(default_config_file, config_file)

setuptools.setup(
    name='dellve',
    version='0.1.2',
    packages=setuptools.find_packages(),
    include_package_data=True,
    cmdclass={
        'install': CustomInstallCommand,
    },
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
        'pick',
        'stringcase',
        'tqdm'
    ],
    setup_requires=[
        'pytest-runner'
    ],
    tests_require=[
        'pytest'
    ],
    entry_points='''
        [console_scripts]
        dellve=dellve.cli:cli
    '''
)
