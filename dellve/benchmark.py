import abc
import multiprocessing as mp
import ctypes
import signal
import pkg_resources
import StringIO
import sys

class BenchmarkIO(StringIO.StringIO):
    """Helper class for collecting output from DELLve benchmarks.
    """
    def __init__(self, queue, *args, **kwargs):
        StringIO.StringIO.__init__(self, *args, **kwargs)
        self.__queue = queue # IPC queue

    def write(self, s):
        self.__queue.put(s)

class Benchmark(mp.Process):
    """Abstract base class for all DELLve benchmarks.
    """

    __metaclass__ = abc.ABCMeta

    memutil = 1.0


    def __init__(self):
        """Constructs a new Benchmark instance.
        """
        mp.Process.__init__(self)
        self.__progress = mp.Value(ctypes.c_float)
        self.__queue = mp.Queue()
        self.__output = []

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

    @property
    def output(self):
        for _ in range(0, self.__queue.qsize()):
            self.__output.append(self.__queue.get())
        return self.__output

    def run(self, *args, **kwargs):
        # Create SIGTERM handler
        def handler(*args, **kwargs):
            raise BenchmarkInterrupt()
        # Register SIGTERM handler
        signal.signal(signal.SIGTERM, handler)
        # Re-map STDOUT and STDERR
        sys.stdout = sys.stderr = BenchmarkIO(self.__queue)
        # Start benchmark routine
        self.routine(*args, **kwargs)

    @abc.abstractmethod
    def routine(self):
        """Defines benchmark routine.

        This method is the main extension point for all user defined benchmarks.
        To create a new benchmark, one has to define a new class derived
        from dellve.benchmark.Benchmark, overwritting its routine method.

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
        mp.Process.start(self)

    def stop(self):
        """Stops benchmark routine and kills the sub-process.

        .. note::

            This method may be called multiple times per life-span of a Benchmark object.
        """
        mp.Process.terminate(self)

class BenchmarkInterrupt(Exception):
    """Custom exception that is raised when benchmark is interrupted with SIGTERM signal.
    """
