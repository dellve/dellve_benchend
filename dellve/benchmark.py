import abc
import collections
import copy
import ctypes
import datetime
import json
import jsonschema
import logging
import multiprocessing as mp
import pkg_resources
import signal
import StringIO
import sys
import time

# DELLve logger
logger = logging.getLogger('dellve-logger')

# TODO: add module import time checks for Benchmark.config correctness

class Benchmark(mp.Process):
    """Abstract base class for all DELLve benchmarks.
    """

    __metaclass__ = abc.ABCMeta

    """Defines benchmark configuration options"""
    config = {}

    """Defines benchmark configuration schema"""
    schema = {}

    """Defines benchmark description message"""
    description = ''

    def __init__(self, config={}):
        """Constructs a new Benchmark instance.
        """
        mp.Process.__init__(self)
        self.__progress = mp.Value(ctypes.c_float)
        self.__iolist = mp.Manager().list()
        self.__stopped = mp.Value(ctypes.c_bool, False)

        # Validate configuration
        if self.validate(config):
            # Okay, yse provided values
            self.config = copy.deepcopy(config)

    @property
    def progress(self):
        """Benchmark progress value.
        """
        return self.__progress.value

    @progress.setter
    def progress(self, value):
        """Benchmark progress value setter.

        Args:
            value (float): New benchmark progress value.
        """
        if value < 0 or value > 100:
            raise ValueError(value)
        self.__progress.value = value
        # Provide debuggin info
        logger.debug('%s progress set to %d' % (self.name, value))

    @property
    def output(self):
        return list(self.__iolist)

    def run(self, debug=False):
        # Create SIGTERM handler
        def handler(*args, **kwargs):
            # Provide debugging info
            logger.debug('Received SIGTERM interrupt in %s ' % self.name + \
                          'with progress %d' % self.progress)
            # Raise to interrupt routine
            raise BenchmarkInterrupt()

        # Register SIGTERM handler
        signal.signal(signal.SIGTERM, handler)

        # Re-map STDOUT and STDERR
        sys.stdout = BenchmarkIO(self.__iolist, sys.stdout if debug else None)
        sys.stderr = BenchmarkIO(self.__iolist, sys.stderr if debug else None)

        # Provide logging info
        logger.info('Started ' + self.name +
                    (' in debug mode' if debug else ''))

        # Print header
        print 'Authors:        Quinito Baula\n'     + \
              '                Travis Chau\n'       + \
              '                Abigail Johnson\n'   + \
              '                Jayesh Joshi\n'      + \
              '                Konstantyn Komarov'
        print ''
        print 'Name:           %s' % self.name
        if len(self.description):
            print 'Description:    %s' % self.description.replace('\n', '\n                ')
            print ''
        print ''

        # Dump configuration into pretty JSON string
        config_json = json.dumps(self.config, indent=4, separators=(',', ': '))
        for index, line in enumerate(config_json.split('\n')):
            if index == 0:
                print 'Configuration:  %s' % line
            else:
                print '                %s' % line
        print ''
        print 'Starting time:  %s' % datetime.datetime.now()
        print ''
        print ' -- Entering benchmark routine...'
        print ''

        # Collect timeing info
        start_time = time.time()
        start_iolist_len = len(self.__iolist)

        try: # Start routine
            self.routine()
        except BenchmarkInterrupt as e: # Report interrupt
            logger.info('Stopped %s due to interrupt' % self.name)
        except: # Report error
            logger.exception('Stopped %s due to exception' % self.name)
            # Print out exception info to report
            print '\n -- Stopping prematurely due to exception!'
        else: # Report success
            logger.info('{benchmark} stopped with progress {progress}'.format(
                benchmark=self.name,
                progress=self.progress
            ))
        finally: # Report benchmark output
            # Collect timing info
            stop_time = time.time()
            stop_iolist_len = len(self.__iolist)

            logger.info('{benchmark} output dump:\n\n{output}'.format(
                benchmark=self.name,
                output=''.join(self.output)
            ))

            # Reformat benchmark output and print to report
            for index in range(start_iolist_len, stop_iolist_len):
                self.__iolist[index] = ('    ' + self.__iolist[index])

            print ''
            print ' -- Exiting benchmark routine %s...' % \
                ('due to STOP interrupt' if self.__stopped.value else '')
            print ''
            print 'Stopping time:  %s' % datetime.datetime.now()
            print ''
            print 'Execution time: %s seconds' % str(stop_time - start_time)

    @abc.abstractmethod
    def routine(self):
        """Defines benchmark routine.

        This method is the main extension point for all user defined benchmarks.
        To create a new benchmark, one has to define a new class derived
        from dellve.benchmark.Benchmark, overwritting its routine method.

        Args:
            config: Benchmark configuration object.

        For example:

        .. code-block:: python
            :name: my_custom_benchmark-py

            # ./my_custom_benchmark.py

            # Import DELLve depencencies
            from dellve import Benchmark, BenchmarkInterrupt

            # Define custom bechmark class
            class MyCustomBenchmark(Benchmark):
                def routine(self):
                    try:
                        # Starting my custom benchmark...
                        for p in range(0, 100):
                            self.progress = p
                        # Exiting because we're done!
                    except BenchmarkInterrupt:
                        # Exiting because of interrupt...
        """

    def is_running(self):
        """Determines if benchmark is running.

        Returns:
            bool: True if benchmark is running, False otherwise
        """
        return self.is_alive()

    def start(self):
        """Starts benchmark routine in a new sub-process.

        .. note::

            This method should be called at most once per life-span of a Benchmark object.
        """

        # Provide info for debugging purposes
        logger.debug('Launching sub-process for %s...' % self.name)

        try: # Start benchamrk
            mp.Process.start(self)
        except: # Report error
            logger.exception('Couldn\'t launch sub-process for %s' % self.name)
        else: # Report success
            logger.info('Launched sub-process for %s with PID %d' % (self.name,
                                                                      self.pid))

    def stop(self):
        """Stops benchmark routine and kills the sub-process.

        .. note::

            This method may be called multiple times per life-span of a Benchmark object.
        """

        # Note: this string is used several times in messages below
        proc_with_pid = 'sub-process for %s with PID %d' % (self.name, self.pid)

        # Provide info for debugging purposes
        logger.debug('Terminating ' + proc_with_pid + ' ...')

        # Update benchmark state
        self.__stopped.value = True

        try: # Stop benchamrk
            mp.Process.terminate(self)
        except: # Report error
            logger.exception('Couldn\'t terminate ' + proc_with_pid)
        else: # Report success
            logger.info('Terminated ' + proc_with_pid)

    @classmethod
    def init_config(cls):
        pass

    @classmethod
    def validate(cls, config):
        """Validates configuration object according to classes schema.
        """
        logger.info(
            ('Validating configuration object:\n%s\n' % \
                json.dumps(config, indent=4, sort_keys=True)) + \
            ('...with configuration schema:\n%s\n' % \
                json.dumps(cls.schema, indent=4, sort_keys=True)))
        try: # Validate config object
            jsonschema.validate(config, cls.schema)
        except:
            logger.exception('Config validation for %s failed' % cls.name)
            return False
        else:
            logger.info('Config validation for %s succeeded' % cls.name)
            return True


class BenchmarkConfig(collections.OrderedDict):
    """Helper class for defining configuration of DELLve benchmarks.
    """

class BenchmarkInterrupt(Exception):
    """Exception class that is raised when benchmark is interrupted by OS
    signal.
    """


class BenchmarkIO(StringIO.StringIO):
    """Helper class for collecting output from DELLve benchmarks.
    """

    def __init__(self, olist, ofile=None):
        StringIO.StringIO.__init__(self)
        self.__olist = olist  # IPC Output list
        self.__ofile = ofile  # Output file

    def write(self, s):
        self.__olist.append(s)
        if self.__ofile is not None:
            self.__ofile.write(s)
