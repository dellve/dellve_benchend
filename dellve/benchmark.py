
import abc


class Benchmark(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, progress=None, stop=None):
        self.__progress = progress
        self.__stop = stop

    @property
    def stop(self):
        if self.__stop is not None:
            return self.__stop.is_set()
        else:
            return False

    @property
    def progress(self):
        if self.__progress is not None:
            with self.__progress.get_lock():
                return self.__progress.value
        else:
            return 0

    @progress.setter
    def progress(self, value):
        if self.__progress is not None:
            if not isinstance(value, int):
                raise TypeError(value)
            if value < 0 or value > 100:
                raise ValueError(value)
            with self.__progress.get_lock():
                self.__progress.value = value

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError()
