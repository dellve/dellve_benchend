import click
import json
import logging
import logging.config
import os
import os.path
import pkg_resources as pr
import shutil

# Constants and default config values (for internal use only)

APP_NAME = 'dellve'
DEFAULT_APP_DIR = click.get_app_dir(APP_NAME)
DEFAULT_LOG_DIR = os.path.join(DEFAULT_APP_DIR, 'logs')
DEFAULT_DEBUG = False
DEFAULT_HTTP_HOST = '127.0.0.1'
DEFAULT_HTTP_PORT = 9999
DEFAULT_BENCHMARKS = map(lambda item: item.load(),
    pr.iter_entry_points(group='dellve.benchmarks', name=None))
DEFAULT_BENCHMARKS = list(DEFAULT_BENCHMARKS)
DEFAULT_BENCHMARKS.sort(key=lambda b: b.__name__)  # convert generator to list
DEFAULT_PID_FILE = os.path.join(DEFAULT_APP_DIR, 'dellve.pid')
DEFAULT_CONFIG_FILE = os.path.join(DEFAULT_APP_DIR, 'config.json')

# Create app directory
if not os.path.exists(DEFAULT_APP_DIR):
    os.makedirs(DEFAULT_APP_DIR)

# Create app-logs directory
if not os.path.exists(DEFAULT_LOG_DIR):
    os.makedirs(DEFAULT_LOG_DIR)

# Create config file (if one doesn't exist)
if not os.path.exists(DEFAULT_CONFIG_FILE):
    shutil.copy(os.path.join(os.path.dirname(__file__), 'data/config.json'),
                os.path.join(DEFAULT_APP_DIR, 'config.json'))

# Copy default config file (regardless if it exists)
shutil.copy(os.path.join(os.path.dirname(__file__), 'data/config.json'),
            os.path.join(DEFAULT_APP_DIR, 'default.config.json'))


# Configure logging
logging.config.dictConfig({
    'version': 1,
    'formatters': {
        'message': {
            'format': ' %(levelname)s: %(message)s'
        },
        'request': {
            'format':  '%(levelname)s -- %(asctime)s -- %(message)s\n'
        },
        'verbose': {
            'format': '\n%(levelname)s -- %(asctime)s\n'
                      '\n'
                      '    File:           %(filename)s\n'
                      '    Line:           %(lineno)d\n'
                      '    Process ID:     %(process)d\n'
                      '    Process Name:   %(processName)s\n'
                      '    Thread ID:      %(thread)d\n'
                      '    Thread Name:    %(threadName)s\n'
                      '\n'
                      '%(message)s\n'
        },
    },
    'handlers': {
        'click-logging-handler': {
            'class': 'dellve.util.ClickLoggingHandler',
            'formatter': 'message',
        },
        'dellve-logging-handler': {
            'backupCount': 7, # always keep last 7 logging files
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'filename': os.path.join(DEFAULT_LOG_DIR, 'log'),
            'formatter': 'verbose',
            'interval': 1, # create new logging file every day
            'level': 'DEBUG',
            'when': 'D',
        },
        'http-api-logging-handler': {
            'backupCount': 4, # always keep last 4 logging files
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'filename': os.path.join(DEFAULT_LOG_DIR, 'http-api-log'),
            'formatter': 'request',
            'interval': 15, # create new logging file every 15 minutes
            'level': 'DEBUG',
            'when': 'M',
        },
    },
    'loggers': {
        'dellve-logger': {
            'handlers': ['dellve-logging-handler'],
            'level': 'DEBUG',
            'propagate': True,
        },
        'http-api-logger': {
            'handlers': ['http-api-logging-handler'],
            'level': 'DEBUG',
            'propagate': True,
        }
    },
})

# TODO: specify logging configuratin in config.json!

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
    'debug':        DEFAULT_DEBUG,
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
