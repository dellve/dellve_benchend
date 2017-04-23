import abc
import collections
import copy
import ctypes
import logging
import multiprocessing as mp
import pkg_resources
import signal
import StringIO
import sys


class Benchmark(mp.Process):
    """Abstract base class for all DELLve benchmarks.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, config=None):
        """Constructs a new Benchmark instance.
        """
        mp.Process.__init__(self)
        self.__progress = mp.Value(ctypes.c_float)
        self.__queue = mp.Queue()
        self.__output = []

        # Set config (optionally)
        if config is not None:
            if not isinstance(config, dict):
                raise TypeError(config)
            else:
                self.config = config

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
        logging.debug('%s progress set to %d' % (self.name, value))

    @property
    def output(self):
        for _ in range(0, self.__queue.qsize()):
            self.__output.append(self.__queue.get())
        return self.__output

    def run(self, debug=False):
        # Create SIGTERM handler
        def handler(*args, **kwargs):
            # Provide debugging info
            logging.debug('Received SIGTERM interrupt in %s ' % self.name + \
                          'with progress %d' % self.progress)
            # Raise to interrupt routine
            raise BenchmarkInterrupt()

        # Register SIGTERM handler
        signal.signal(signal.SIGTERM, handler)

        # Re-map STDOUT and STDERR
        sys.stdout = BenchmarkIO(self.__queue, sys.stdout if debug else None)
        sys.stderr = BenchmarkIO(self.__queue, sys.stderr if debug else None)

        # Provide logging info
        logging.info('Started ' + self.name +
                    (' in debug mode' if debug else ''))

        try: # Start routine
            self.routine()
        except BenchmarkInterrupt as e: # Report interrupt
            logging.info('Stopped %s due to interrupt' % self.name)
        except: # Report error
            logging.exception('Stopped %s due to exception' % self.name)
        else: # Report success
            # Let users know benchmark stopped with certain progress
            logging.info('Stopped {benchmark} with progress {progress}'.format({
                'benchmark': self.name,
                'progress': self.progress
            }))

            # Let users see benchmark output
            logging.info('{benchmark} output dump:\n\n{output}'.format({
                'benchmark': self.name,
                'output': ''.join(self.output)
            }))


    @abc.abstractproperty
    def config(self):
        """Defines benchmark configuration options"""

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
        logging.debug('Launching sub-process for %s...' % self.name)

        try: # Start benchamrk
            mp.Process.start(self)
        except: # Report error
            logging.exception('Couldn\'t launch sub-process for %s' % self.name)
        else: # Report success
            logging.info('Launched sub-process for %s with PID %d' % (self.name,
                                                                      self.pid))

    def stop(self):
        """Stops benchmark routine and kills the sub-process.

        .. note::

            This method may be called multiple times per life-span of a Benchmark object.
        """

        # Note: this string is used several times in messages below
        proc_with_pid = 'sub-process for %s with PID %d' % (self.name, self.pid)

        # Provide info for debugging purposes
        logging.debug('Terminating ' + proc_with_pid + ' ...')

        try: # Stop benchamrk
            mp.Process.terminate(self)
        except: # Report error
            logging.exception('Couldn\'t terminate ' + proc_with_pid)
        else: # Report success
            logging.info('Terminated ' + proc_with_pid)


class BenchmarkConfig(collections.OrderedDict):
    """Helper class for defining configuration of DELLve benchmarks.
    """


class BenchmarkInterrupt(Exception):
    """Exception class that is raised when benchmark is interrupted by OS signal.
    """


class BenchmarkIO(StringIO.StringIO):
    """Helper class for collecting output from DELLve benchmarks.
    """

    def __init__(self, queue, ofile=None):
        StringIO.StringIO.__init__(self)
        self.__queue = queue  # IPC queue
        self.__ofile = ofile  # Output file

    def write(self, s):
        self.__queue.put(s)
        if self.__ofile is not None:
            self.__ofile.write(s)
