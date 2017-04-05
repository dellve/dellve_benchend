import click
import json
import os
import os.path
import pkg_resources as pr
import shutil

# Constants and default config values (for internal use only)

APP_NAME = 'dellve'
DEFAULT_APP_DIR = click.get_app_dir(APP_NAME)
DEFAULT_HTTP_PORT = 9999
DEFAULT_BENCHMARKS = map(lambda item: item.load(),
    pr.iter_entry_points(group='dellve.benchmarks', name=None))
DEFAULT_BENCHMARKS = list(DEFAULT_BENCHMARKS)  # convert generator to list
DEFAULT_PID_FILE = os.path.join(DEFAULT_APP_DIR, 'dellve.pid')

def __create_app_dir():
    # Create app directory
    app_dir = DEFAULT_APP_DIR
    if not os.path.exists(app_dir):
        os.makedirs(app_dir)

    # Get paths to DELLve configuration files
    config_file = os.path.join(app_dir, 'config.json')
    default_config_file = os.path.join(app_dir, 'default.config.json')

    # Copy default JSON config files
    shutil.copy('dellve/data/config.json', default_config_file)

    # Copy main JSON config file
    if not os.path.exists(config_file):
        shutil.copy(default_config_file, config_file)
__create_app_dir()

# Private module members (for internal use only)
#
# Note: '__config' is a private storage for configuration values that must be
#       accessed through config.get and config.set functions; please, don't
#       rely on default values defined above, as they may be overwritten by
#       user defined config file or command line options.

__config = {
    'app-dir':      DEFAULT_APP_DIR,
    'http-port':    DEFAULT_HTTP_PORT,
    'benchmarks':   DEFAULT_BENCHMARKS,
    'pid-file':     DEFAULT_PID_FILE
}


# Public module members


def load(file):
    """Loads a configuration file with JSON (.json)
    encoding.

    Args:
        file (file): Configuration file object.

    Raises:
        IOError: IOError is raised if provided file isn't of recognized format.
    """
    if file.name.endswith('.json'):
        data = json.load(file)
        if data and isinstance(data, dict):
            __config.update(data)
    else:
        raise IOError('Couldn\'t load configuration file: %s' % file.name)


def get(name):
    """Gets value of a configuration parameter.

    Args:
        name (str): Parameter's name.

    Returns:
        object: Parameter's value.
    """
    return __config[name]


def set(name, value):
    """Sets value of a configuration parameter.

    Args:
        name (str): Parameter's name.
        value (object): Parameter's value.
    """
    __config[name] = value
