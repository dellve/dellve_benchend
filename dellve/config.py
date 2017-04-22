import click
import json
import os
import os.path
import pkg_resources as pr
import shutil

# Constants and default config values (for internal use only)

APP_NAME = 'dellve'
DEFAULT_APP_DIR = click.get_app_dir(APP_NAME)
DEFAULT_HTTP_HOST = '127.0.0.1'
DEFAULT_HTTP_PORT = 9999
DEFAULT_BENCHMARKS = map(lambda item: item.load(),
    pr.iter_entry_points(group='dellve.benchmarks', name=None))
DEFAULT_BENCHMARKS = list(DEFAULT_BENCHMARKS)  # convert generator to list
DEFAULT_PID_FILE = os.path.join(DEFAULT_APP_DIR, 'dellve.pid')
DEFAULT_CONFIG_FILE = os.path.join(DEFAULT_APP_DIR, 'config.json')

# Create app directory
if not os.path.exists(DEFAULT_APP_DIR):
    os.makedirs(DEFAULT_APP_DIR)

# Create config file (if one doesn't exist)
if not os.path.exists(DEFAULT_CONFIG_FILE):
    shutil.copy(os.path.join(os.path.dirname(__file__), 'data/config.json'),
                os.path.join(DEFAULT_APP_DIR, 'config.json'))

# Copy default config file (regardless if it exists)
shutil.copy(os.path.join(os.path.dirname(__file__), 'data/config.json'),
            os.path.join(DEFAULT_APP_DIR, 'default.config.json'))

# Private module members (for internal use only)
#
# Note: '__config' is a private storage for configuration values that must be
#       accessed through config.get and config.set functions; please, don't
#       rely on default values defined above, as they may be overwritten by
#       user defined config file or command line options.

__config = {
    'app-dir':      DEFAULT_APP_DIR,
    'benchmarks':   DEFAULT_BENCHMARKS,
    'config-file':  DEFAULT_CONFIG_FILE,
    'http-host':    DEFAULT_HTTP_HOST,
    'http-port':    DEFAULT_HTTP_PORT,
    'pid-file':     DEFAULT_PID_FILE,
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
