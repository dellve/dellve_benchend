
import abc


class Benchmark(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, progress, stop):
        self.__progress = progress
        self.__stop = stop

    @property
    def stop(self):
        return self.__stop.is_set()

    @property
    def progress(self):
        with self.__progress.get_lock():
            return self.__progress.value

    @progress.setter
    def progress(self, value):
        if not isinstance(value, int):
            raise TypeError(value)
        if value < 0 or value > 100:
            raise ValueError(value)
        with self.__progress.get_lock():
            self.__progress.value = value

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError()
