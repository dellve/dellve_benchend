
import dellve
import dellve_cudnn_benchmark as dcb
import time
from abc import ABCMeta, abstractmethod

class StressToolFactory(dellve.Benchmark):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_controller(self): pass

    def routine(self):
        controller = self.get_controller()
        controller.start(1,100)

        p = controller.get_progress()

        while (p != 1.0):
            p = controller.get_progress()
            if p > 0:
                self.progress = p * 100
            time.sleep(0.5)
