from abc import ABCMeta, abstractmethod
import time

from dellve import Benchmark, BenchmarkInterrupt
import dellve_cudnn_benchmark as dcb

class StressToolFactory(Benchmark):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_controller(self): pass

    def routine(self):
        try:
            self.controller = self.get_controller()
            self.controller.start(1,100)

            while (not self.complete()):
                self.update_progress()
                time.sleep(0.5)

        except BenchmarkInterrupt:
            print '\nStopping current stress tool'
            self.controller.stop()

    def complete(self):
        return self.controller.get_progress() == 1.0

    def update_progress(self):
        p = self.controller.get_progress()
        if (p > 0):
            self.progress = p * 100
