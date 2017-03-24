
import ctypes
import multiprocessing as mp

class Executor(mp.Process):

    def __init__(self, benchmarks):
        mp.Process.__init__(self)
        # Save benchmarks info
        self.__benchmarks = benchmarks
        # Create progress bar variable
        self.__progress = mp.Value(ctypes.c_int, 0, lock=True)
        # Create variable that shows executed benchmark
        self.__benchmark_id = mp.Value(ctypes.c_int, 0, lock=True)
        # Create state control flags
        self.__start = mp.Event()
        self.__stop = mp.Event()
        self.__join = mp.Event()
        self.__ready = mp.Event()

    def run(self):
        while True:
            # Reset flags
            self.__ready.set()
            # Wait to start ...
            self.__start.wait()
            # Reset flags
            self.__ready.clear()
            # Check if we need to join now
            if self.__join.is_set(): break
            # Reset progress
            with self.__progress.get_lock():
                self.__progress.value = 0
            # Load benchmark class
            benchmark = self.__benchmarks[self.benchmark_id]
            # Create benchmark object and run it
            benchmark(self.__progress, self.__stop).run()
            # Update progress
            with self.__progress.get_lock():
                self.__progress.value = 100
            # Update flags
            self.__start.clear()
            self.__stop.clear()

    def join(self, timeout=None):
        self.__join.set()
        self.__ready.wait()
        self.__start.set()
        mp.Process.join(self)

    @property
    def progress(self):
        with self.__progress.get_lock():
            if self.__progress.value < 0:
                return None
            else:
                return self.__progress.value

    @property
    def benchmark_id(self):
        with self.__benchmark_id.get_lock():
            if self.__benchmark_id.value < 0:
                return None
            else:
                return self.__benchmark_id.value

    def start_benchmark(self, benchmark_id):
        with self.__benchmark_id.get_lock():
            if not self.__start.is_set():
                self.__benchmark_id.value = benchmark_id
                self.__stop.clear()
                self.__start.set()
                return True
            return False

    def stop_benchmark(self, benchmark_id):
        if not self.__stop.is_set():
            self.__stop.set()
            return True
        return False

